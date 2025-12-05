import numpy as np
import time
import os
import sys
import multiprocessing
import argparse
import threading
from threadpoolctl import threadpool_info, ThreadpoolController


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

def process_data(data):
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

def run_benchmark(client, transfer_sizes, n_transfers=10, n_runs=10, n_threads=None):
    print("\n=== RDMA Benchmark ===")
    print(f"Running with {n_threads} threads (NumPy internal threading)")
    
    # Define column headers and widths
    col1_width = 15  # Size column
    col2_width = 20  # Time column
    col3_width = 20  # Throughput column
    
    # Print headers with fixed widths
    print(f"{'Size (bytes)'.ljust(col1_width)} "
          f"{'Time (us)'.ljust(col2_width)} "
          f"{'Throughput (MB/s)'.ljust(col3_width)}")
    print("-" * (col1_width + col2_width + col3_width + 2))  # +2 for spaces between columns

    for size in transfer_sizes:
        # Ensure size is divisible by (48 * 4) bytes
        if size % (48 * 4) != 0:
            print(f"Skipping size {size}, not divisible by 48*4 bytes")
            continue
            
        total_time = 0
        try:
            for run in range(n_runs):
                # Measure time for RDMA transfer and processing
                elapsed_time = client.rdma_process_to_gpu(size, n_transfers, process_data)
                # elapsed_time = client.rdma_cpu_gpu(size, n_transfers)
                total_time += elapsed_time

        except Exception as e:
            print(f"Error processing size {size}: {str(e)}")
            continue

        avg_time = total_time / n_runs * 1e-3  # Convert to microseconds
        throughput = (size * n_transfers * 1e6) / (avg_time * 1024 * 1024)  # MB/s
        
        # Print results with fixed-width columns
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
    transfer_sizes = [192 * (2**i) for i in range(int(np.log2(3145728//192)) + 1)]
    # transfer_sizes = []
    # transfer_sizes = [3145728]
    
    print(f"Initializing RDMA client with server IP: {args.server_ip}")
    client = RDMAClient(args.server_ip, args.buffer_size)
    
    run_benchmark(client, transfer_sizes, args.n_transfers, args.n_runs, args.threads)

if __name__ == "__main__":
    main() 