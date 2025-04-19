import numpy as np
import time
import os
import sys

# Add the build/lib directory to the Python path
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
    # Reshape the data into rows of 48 columns
    n_elements = len(data)
    if n_elements % 48 != 0:
        raise ValueError(f"Data length {n_elements} is not divisible by 48 columns")
    
    n_rows = n_elements // 48
    data_2d = data.reshape(n_rows, 48)
    
    # Create output array with the same shape
    result = np.zeros_like(data_2d, dtype=np.float32)  # Change to float32 for log operation
    
    # Process first 16 columns:
    # 1. Convert negative values to zero
    # 2. Apply log(x+1)
    first_16_cols = data_2d[:, :16]
    first_16_cols = np.maximum(first_16_cols, 0)  # Convert negative to zero
    result[:, :16] = np.log1p(first_16_cols)  # log(x+1)
    
    # Process last 32 columns:
    # Apply modulo 8192
    result[:, 16:] = data_2d[:, 16:] % 8192
    
    # # Convert back to int32 for consistency
    # result = result.astype(np.int32)
    
    # Return flattened array
    return result.ravel()

def run_benchmark(client, transfer_sizes, n_transfers=10, n_runs=10):
    print("\n=== RDMA Benchmark ===")
    print("Size (bytes)\tTime (ns)\tThroughput (MB/s)")
    print("-" * 60)

    for size in transfer_sizes:
        # Ensure size is divisible by (48 * 4) bytes
        if size % (48 * 4) != 0:
            print(f"Skipping size {size}, not divisible by 48*4 bytes")
            continue
            
        total_time = 0
        try:
            for run in range(n_runs):
                # Measure time for RDMA transfer and processing
                elapsed_time = client.rdma_process_to_gpu_async(size, n_transfers, process_data)
                total_time += elapsed_time
                
                # # Verify the processing (for the first run only)
                # if run == 0:
                #     # Read original data from CPU memory
                #     cpu_data = client.read_cpu_memory(size)
                #     print(f"Run {run} - First row of CPU data:")
                #     print(cpu_data[:48].reshape(1, 48))
                    
                #     # Read processed data from GPU memory
                #     gpu_data = client.read_gpu_memory(size)
                #     print(f"Run {run} - First row of GPU data:")
                #     print(gpu_data[:48].reshape(1, 48))
                    
                #     # Verify that the data was processed correctly
                #     cpu_2d = cpu_data.reshape(-1, 48)
                #     expected = np.zeros_like(cpu_2d)
                #     expected[:, :16] = cpu_2d[:, :16] * 2
                #     expected[:, 16:] = cpu_2d[:, 16:] * 4
                #     expected = expected.ravel()
                    
                #     print(f"Run {run} - First row of expected data:")
                #     print(expected[:48].reshape(1, 48))
                    
                #     if not np.array_equal(gpu_data, expected):
                #         print(f"Warning: Data processing verification failed for size {size}")
                #         print("Differences in first row where they don't match:")
                #         first_row_gpu = gpu_data[:48].reshape(1, 48)
                #         first_row_expected = expected[:48].reshape(1, 48)
                #         for col in range(48):
                #             if first_row_gpu[0, col] != first_row_expected[0, col]:
                #                 print(f"Column {col}: GPU={first_row_gpu[0, col]}, Expected={first_row_expected[0, col]}")
                
                

        except Exception as e:
            print(f"Error processing size {size}: {str(e)}")
            continue

        avg_time = total_time / n_runs
        # Convert to MB/s: (size * n_transfers) bytes / (avg_time nanoseconds)
        # * 1e9 to convert ns to seconds
        # / (1024*1024) to convert bytes to MB
        throughput = (size * n_transfers * 1e9) / (avg_time * 1024 * 1024)  # MB/s
        print(f"{size}\t\t{avg_time:.2f}\t\t{throughput:.2f} MB/s")

def main():
    # Configuration
    server_ip = "10.253.74.120"  # Replace with your server IP
    buffer_size = 1024 * 1024 * 128  # 128MB buffer
    n_runs = 10  # Number of runs for averaging
    
    # Transfer sizes to test (in bytes)
    # Generate array starting from 192, multiplying by 2 until reaching close to 50331648
    transfer_sizes = [192 * (2**i) for i in range(int(np.log2(50331648//192)) + 1)]
    
    # Number of transfers for each size
    n_transfers = 64  # Adjust based on your needs
    
    print(f"Initializing RDMA client with server IP: {server_ip}")
    client = RDMAClient(server_ip, buffer_size)
    
    run_benchmark(client, transfer_sizes, n_transfers, n_runs)

if __name__ == "__main__":
    main() 