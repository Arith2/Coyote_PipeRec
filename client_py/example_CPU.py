import numpy as np
import time
import os
import sys
import multiprocessing
import argparse
import threading
from threadpoolctl import threadpool_info, ThreadpoolController
from concurrent.futures import ThreadPoolExecutor


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='RDMA Benchmark with configurable thread count')
    parser.add_argument('--threads', type=int, default=1,
                       help='Number of threads to use for processing (default: number of physical cores)')
    parser.add_argument('--server-ip', type=str, default="10.253.74.120",
                       help='Server IP address (default: 10.253.74.120)')
    parser.add_argument('--buffer-size', type=int, default=1024 * 1024 * 128,
                       help='Buffer size in bytes (default: 128MB)')
    parser.add_argument('--n-runs', type=int, default=10,
                       help='Number of runs for averaging (default: 10)')
    parser.add_argument('--n-transfers', type=int, default=64,
                       help='Number of transfers for each size (default: 64)')
    return parser.parse_args()

# Add the build directory to the Python path
build_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'lib')
sys.path.append(build_lib_dir)

# Now import the module
from rdma_module import RDMAClient
    

def format_time(ns):
    """Format nanoseconds into a human-readable string."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1000000:
        return f"{ns/1000:.2f} µs"
    elif ns < 1000000000:
        return f"{ns/1000000:.2f} ms"
    else:
        return f"{ns/1000000000:.2f} s"

def format_throughput(bytes_per_second):
    """Format bytes per second into a human-readable string."""
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.2f} B/s"
    elif bytes_per_second < 1024*1024:
        return f"{bytes_per_second/1024:.2f} KB/s"
    elif bytes_per_second < 1024*1024*1024:
        return f"{bytes_per_second/(1024*1024):.2f} MB/s"
    else:
        return f"{bytes_per_second/(1024*1024*1024):.2f} GB/s"

def process_data_single(data):
    """Process data using NumPy's built-in parallelization"""
    # start_time = time.time()
    # Reshape the data into rows of 48 columns
    n_elements = len(data)
    if n_elements % 48 != 0:
        raise ValueError(f"Data length {n_elements} is not divisible by 48 columns")
    
    n_rows = n_elements // 48
    data_2d = data.reshape(n_rows, 48)
    
    # Create output array
    result = np.zeros_like(data_2d, dtype=np.float32)
    
    # Process first 16 columns using vectorized operations
    # These operations can use multiple threads through BLAS/LAPACK
    first_16 = data_2d[:, :16]
    result[:, :16] = np.log1p(np.maximum(first_16, 0))
    
    # Process last 32 columns using vectorized operations
    # Modulo operation might not be multi-threaded by NumPy
    result[:, 16:] = data_2d[:, 16:] % 8192
    
    # end_time = time.time()
    # print(f"Processing time: {end_time - start_time} seconds")
    # Return flattened array
    return result.ravel()

def process_data_mt(data: np.ndarray, n_threads: int) -> np.ndarray:
    """多线程版本的 preprocessing，按 rows 切分到多个线程。"""
    n_elements = len(data)
    if n_elements % 48 != 0:
        raise ValueError(f"Data length {n_elements} is not divisible by 48 columns")
    
    n_rows = n_elements // 48
    data_2d = data.reshape(n_rows, 48)

    # 输出 buffer
    result = np.empty_like(data_2d, dtype=np.float32)

    # 每个线程处理 [start_row, end_row)
    def worker(start_row: int, end_row: int):
        chunk = data_2d[start_row:end_row]

        # 为该 chunk 分配一个临时结果
        res = np.empty_like(chunk, dtype=np.float32)

        # 前 16 列：log1p(max(x, 0))
        res[:, :16] = np.log1p(np.maximum(chunk[:, :16], 0))

        # 后 32 列：mod 8192
        res[:, 16:] = chunk[:, 16:] % 8192

        return start_row, res

    # 线程数至少 1
    n_threads = max(1, int(n_threads))

    # 按行平均切给各线程
    step = (n_rows + n_threads - 1) // n_threads

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = []
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            futures.append(pool.submit(worker, start, end))
        
        # 收集结果，按行写回到 result
        for fut in futures:
            start_row, res_chunk = fut.result()
            end_row = start_row + res_chunk.shape[0]
            result[start_row:end_row, :] = res_chunk

    return result.ravel()


def process_data_mt_inplace(data: np.ndarray, n_threads: int) -> np.ndarray:
    n_elements = len(data)
    if n_elements % 48 != 0:
        raise ValueError(f"Data length {n_elements} is not divisible by 48 columns")

    n_rows = n_elements // 48
    data_2d = data.reshape(n_rows, 48)

    # 输出 buffer：还是单独开的 result（而不是 per-thread 小 res）
    result = np.empty_like(data_2d, dtype=np.float32)

    def worker(start_row: int, end_row: int):
        # 输入视图
        chunk_in = data_2d[start_row:end_row]
        # 输出视图（共享同一个 result，但行不重叠）
        chunk_out = result[start_row:end_row]

        # 前 16 列：log1p(max(x, 0))，用 out 参数减少临时数组
        np.maximum(chunk_in[:, :16], 0, out=chunk_out[:, :16])
        np.log1p(chunk_out[:, :16], out=chunk_out[:, :16])

        # 后 32 列：mod 8192
        np.mod(chunk_in[:, 16:], 8192, out=chunk_out[:, 16:])

    n_threads = max(1, int(n_threads))
    step = (n_rows + n_threads - 1) // n_threads

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = []
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            futures.append(pool.submit(worker, start, end))
        for fut in futures:
            fut.result()

    return result.ravel()

def run_local_baseline(client, transfer_sizes, n_transfers=64, n_runs=10, n_threads=None):
    # 和之前一样决定用哪个 process function
    if n_threads is None or n_threads <= 1:
        def proc_fn(data):
            return process_data_single(data)
    else:
        def proc_fn(data):
            return process_data_mt_inplace(data, n_threads)

    print("\n=== Local CPU Baseline (no RDMA) ===")
    print(f"Running with {n_threads} threads for preprocessing")

    col1_width = 15
    col2_width = 20
    col3_width = 20

    print(f"{'Size (bytes)'.ljust(col1_width)} "
          f"{'Time (us)'.ljust(col2_width)} "
          f"{'Throughput (MB/s)'.ljust(col3_width)}")
    print("-" * (col1_width + col2_width + col3_width + 2))

    for size in transfer_sizes:
        if size % (48 * 4) != 0:
            print(f"Skipping size {size}, not divisible by 48*4 bytes")
            continue

        # 在 mem_cpu 里填随机数（只要做一次也可以）
        cpu_arr = client.read_cpu_memory(size)
        cpu_arr[...] = np.random.randint(
            low=0,
            high=10000,
            size=cpu_arr.shape,
            dtype=np.int32,
        )

        total_time = 0.0
        for _ in range(n_runs):
            elapsed = client.local_process_to_gpu(size, n_transfers, proc_fn)
            total_time += elapsed

        avg_time_us = total_time / n_runs * 1e-3  # ns -> us
        throughput = (size * n_transfers * 1e6) / (avg_time_us * 1024 * 1024)

        print(f"{str(size).ljust(col1_width)} "
              f"{f'{avg_time_us:.2f}'.ljust(col2_width)} "
              f"{f'{throughput:.2f}'.ljust(col3_width)}")


def run_benchmark(client, transfer_sizes, n_transfers=10, n_runs=10, n_threads=None):
    print("\n=== RDMA Benchmark ===")
    print(f"Running with {n_threads} threads for preprocessing (Python-level)")

    # 选择实际使用的处理函数：单线程 or 多线程
    if n_threads is None or n_threads <= 1:
        # 直接用原来的单线程实现
        def proc_fn(data):
            return process_data_single(data)
    else:
        # 用多线程版本；闭包把 n_threads 捕获进去
        def proc_fn(data):
            return process_data_mt_inplace(data, n_threads)

    # 打印表头
    col1_width = 15  # Size column
    col2_width = 20  # Time column
    col3_width = 20  # Throughput column
    
    print(f"{'Size (bytes)'.ljust(col1_width)} "
          f"{'Time (us)'.ljust(col2_width)} "
          f"{'Throughput (MB/s)'.ljust(col3_width)}")
    print("-" * (col1_width + col2_width + col3_width + 2))

    for size in transfer_sizes:
        # 确保能整除 48*4
        if size % (48 * 4) != 0:
            print(f"Skipping size {size}, not divisible by 48*4 bytes")
            continue

        total_time = 0
        try:
            for _ in range(n_runs):
                # 把我们包装好的 proc_fn 传给 C++ 侧
                elapsed_time = client.rdma_process_to_gpu(size, n_transfers, proc_fn)
                total_time += elapsed_time

        except Exception as e:
            print(f"Error processing size {size}: {str(e)}")
            continue

        avg_time = total_time / n_runs * 1e-3  # ns → us
        throughput = (size * n_transfers * 1e6) / (avg_time * 1024 * 1024)  # MB/s

        print(f"{str(size).ljust(col1_width)} "
              f"{f'{avg_time:.2f}'.ljust(col2_width)} "
              f"{f'{throughput:.2f}'.ljust(col3_width)}")


def check_threading_config():
    """Check and print threading configuration"""
    print("\nThreading Configuration:")
    print(f"NumPy version: {np.__version__}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    # Try to get BLAS info
    try:
        import numpy.distutils.system_info as sysinfo
        blas_info = sysinfo.get_info('blas_opt')
        print(f"BLAS info: {blas_info}")
    except:
        print("Could not get BLAS info")
    
    # Check threadpool info
    controller = ThreadpoolController()
    print("\nInitial threadpool state:")
    for info in threadpool_info():
        print(f"- {info['prefix']}: {info.get('num_threads', 'unknown')} threads")
    
    # Try to modify thread count
    print("\nTrying to set thread limit...")
    with controller.limit(limits=4):
        print("Thread limits applied:")
        for info in threadpool_info():
            print(f"- {info['prefix']}: {info.get('num_threads', 'unknown')} threads")
        
        # Test with a computation that should use multiple threads
        size = 1000
        a = np.random.rand(size, size)
        start = time.time()
        np.dot(a, a.T)
        end = time.time()
        print(f"\nMatrix multiplication time ({size}x{size}): {(end-start)*1000:.2f} ms")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check threading configuration
    # check_threading_config()
    
    print("\nStarting benchmark...")
    
    # Generate transfer sizes array
    # transfer_sizes = [192 * (2**i) for i in range(int(np.log2(50331648//192)) + 1)]
    transfer_sizes = [192 * (2**i) for i in range(int(np.log2(100663296//192)) + 1)]
    # transfer_sizes = []
    # transfer_sizes = [3145728]
    
    # # print(f"Initializing RDMA client with server IP: {args.server_ip}")
    # client = RDMAClient(args.server_ip, args.buffer_size)
    # # run_benchmark(client, transfer_sizes, args.n_transfers, args.n_runs, args.threads)
    # run_local_baseline(client, transfer_sizes, args.n_transfers, args.n_runs, args.threads)

    # 本地 baseline：不需要 server，enable_rdma=False
    client = RDMAClient("127.0.0.1", args.buffer_size, False)
    run_local_baseline(client, transfer_sizes, args.n_transfers, args.n_runs, args.threads)


if __name__ == "__main__":
    main() 