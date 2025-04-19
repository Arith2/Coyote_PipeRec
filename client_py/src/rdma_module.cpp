#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <any>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include "cThread.hpp"
#include "constants.hpp"

namespace py = pybind11;

class RDMAClient {
private:
    coyote::cThread<std::any> coyote_thread;
    int *mem_cpu;
    int *mem_gpu;
    unsigned int max_size;
    std::string server_ip;
    hipStream_t stream;  // HIP stream for async operations

    // Helper function to get CPU data as numpy array
    py::array_t<int> get_cpu_data(unsigned int size) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }
        return py::array_t<int>({size / sizeof(int)}, {sizeof(int)}, mem_cpu);
    }

    // Helper function to update CPU data from numpy array
    void update_cpu_data(py::array_t<int> data, unsigned int size) {
        if (size > max_size) {
            throw std::runtime_error("Data size exceeds buffer size");
        }
        py::buffer_info buf = data.request();
        if (buf.size * sizeof(int) != size) {
            throw std::runtime_error("Data size mismatch");
        }
        std::memcpy(mem_cpu, buf.ptr, size);
    }

public:
    RDMAClient(const std::string& ip, unsigned int buffer_size) 
        : coyote_thread(DEFAULT_VFPGA_ID, getpid(), 0), 
          max_size(buffer_size),
          server_ip(ip) {
        
        // Initialize RDMA and allocate CPU memory
        mem_cpu = (int *) coyote_thread.initRDMA(max_size, coyote::defPort, server_ip.c_str());
        if (!mem_cpu) {
            throw std::runtime_error("Could not allocate CPU memory");
        }

        // Initialize GPU and allocate GPU memory
        if (hipSetDevice(DEFAULT_GPU_ID)) {
            throw std::runtime_error("Could not select GPU device");
        }

        mem_gpu = (int *) coyote_thread.getMem({coyote::CoyoteAlloc::GPU, max_size});
        if (!mem_gpu) {
            throw std::runtime_error("Could not allocate GPU memory");
        }

        // Create HIP stream
        if (hipStreamCreate(&stream) != hipSuccess) {
            throw std::runtime_error("Could not create HIP stream");
        }
    }

    double rdma_cpu(unsigned int size, unsigned int n_transfers, bool is_write = false) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        coyote::sgEntry sg;
        sg.rdma = { .len = size };

        // Perform RDMA operation
        coyote::CoyoteOper coyote_operation = is_write ? 
            coyote::CoyoteOper::REMOTE_RDMA_WRITE : 
            coyote::CoyoteOper::REMOTE_RDMA_READ;

        // Clear completion flags and sync with server
        coyote_thread.clearCompleted();
        coyote_thread.connSync(true);  // true for client
        // std::cout << "DEBUG: Synced with server" << std::endl;

        auto begin_time = std::chrono::high_resolution_clock::now(); 
        // Invoke RDMA operation
        for (int i = 0; i < n_transfers; i++) {
            coyote_thread.invoke(coyote_operation, &sg);
        }

        // Wait for completion
        while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) != n_transfers) {}

        auto end_time = std::chrono::high_resolution_clock::now();
        double measured_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count(); 

        return measured_time;
    }

    double rdma_cpu_gpu(unsigned int size, unsigned int n_transfers, bool is_write = false) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        coyote::sgEntry sg;
        sg.rdma = { .len = size };

        // Perform RDMA operation
        coyote::CoyoteOper coyote_operation = is_write ? 
            coyote::CoyoteOper::REMOTE_RDMA_WRITE : 
            coyote::CoyoteOper::REMOTE_RDMA_READ;

        // Clear completion flags and sync with server
        coyote_thread.clearCompleted();
        coyote_thread.connSync(true);  // true for client
        // std::cout << "DEBUG: Synced with server" << std::endl;

        auto begin_time = std::chrono::high_resolution_clock::now(); 
        // Invoke RDMA operation
        for (int i = 0; i < n_transfers; i++) {
            coyote_thread.invoke(coyote_operation, &sg);
        }

        int completed = 0; 
        while (completed < n_transfers) {
            if (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) > completed) {
                completed += 1;

                auto hipMemcpy_result = hipMemcpy(mem_gpu, mem_cpu, sg.rdma.len, hipMemcpyHostToDevice);
                if (hipMemcpy_result != hipSuccess) {
                    throw std::runtime_error("hipMemcpy failed!");
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double measured_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count(); 

        return measured_time;
    }

    // New function: Read data to CPU, process it in Python, then transfer to GPU
    double rdma_process_to_gpu(unsigned int size, unsigned int n_transfers, py::function process_fn, bool is_write = false) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        coyote::sgEntry sg;
        sg.rdma = { .len = size };

        coyote::CoyoteOper coyote_operation = is_write ? 
            coyote::CoyoteOper::REMOTE_RDMA_WRITE : 
            coyote::CoyoteOper::REMOTE_RDMA_READ;

        coyote_thread.clearCompleted();
        coyote_thread.connSync(true);

        auto begin_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_transfers; i++) {
            coyote_thread.invoke(coyote_operation, &sg);
        }

        int completed = 0;
        while (completed < n_transfers) {
            if (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) > completed) {
                completed += 1;
                
                // Get CPU data as numpy array
                auto cpu_data = get_cpu_data(size);
                // Process data using Python function
                py::object result = process_fn(cpu_data);
                // Convert result to numpy array
                py::array_t<float> processed_data = result.cast<py::array_t<float>>();
                py::buffer_info buf = processed_data.request();

                // Transfer processed data directly to GPU
                auto hipMemcpy_result = hipMemcpy(mem_gpu, buf.ptr, size, hipMemcpyHostToDevice);
                if (hipMemcpy_result != hipSuccess) {
                    throw std::runtime_error("hipMemcpy failed!");
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
    }

    // Asynchronous version of rdma_process_to_gpu
    double rdma_process_to_gpu_async(unsigned int size, unsigned int n_transfers, py::function process_fn, bool is_write = false) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        coyote::sgEntry sg;
        sg.rdma = { .len = size };

        coyote::CoyoteOper coyote_operation = is_write ? 
            coyote::CoyoteOper::REMOTE_RDMA_WRITE : 
            coyote::CoyoteOper::REMOTE_RDMA_READ;

        coyote_thread.clearCompleted();
        coyote_thread.connSync(true);

        auto begin_time = std::chrono::high_resolution_clock::now();

        // Launch all RDMA operations
        for (int i = 0; i < n_transfers; i++) {
            coyote_thread.invoke(coyote_operation, &sg);
        }

        int completed = 0;
        while (completed < n_transfers) {
            if (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) > completed) {
                completed += 1;
                
                // Get CPU data as numpy array and process it
                auto cpu_data = get_cpu_data(size);
                py::object result = process_fn(cpu_data);
                py::array_t<int> processed_data = result.cast<py::array_t<int>>();
                py::buffer_info buf = processed_data.request();

                // Asynchronous transfer to GPU
                auto hipMemcpy_result = hipMemcpyAsync(mem_gpu, buf.ptr, size, 
                                                     hipMemcpyHostToDevice, stream);
                if (hipMemcpy_result != hipSuccess) {
                    throw std::runtime_error("hipMemcpyAsync failed!");
                }
            }
        }

        // Wait for all GPU operations to complete
        if (hipStreamSynchronize(stream) != hipSuccess) {
            throw std::runtime_error("hipStreamSynchronize failed!");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
    }

    // Function to get data from CPU memory for inspection/processing
    py::array_t<int> read_cpu_memory(unsigned int size) {
        return get_cpu_data(size);
    }

    // Function to get data from GPU memory
    py::array_t<int> read_gpu_memory(unsigned int size) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        // Copy data from GPU to CPU temporarily
        auto hipMemcpy_result = hipMemcpy(mem_cpu, mem_gpu, size, hipMemcpyDeviceToHost);
        if (hipMemcpy_result != hipSuccess) {
            throw std::runtime_error("hipMemcpy failed!");
        }

        return get_cpu_data(size);
    }

    ~RDMAClient() {
        try {
            // Cleanup HIP stream
            hipStreamDestroy(stream);
            // Final sync with server before cleanup
            coyote_thread.connSync(true);
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
};

PYBIND11_MODULE(rdma_module, m) {
    m.doc() = "RDMA client module for data transfer between CPU and GPU";

    py::class_<RDMAClient>(m, "RDMAClient")
        .def(py::init<const std::string&, unsigned int>())
        .def("rdma_cpu", &RDMAClient::rdma_cpu, 
             py::arg("size"), 
             py::arg("n_transfers"),
             py::arg("is_write") = false,
             "Read data via RDMA to CPU")
        .def("rdma_cpu_gpu", &RDMAClient::rdma_cpu_gpu, 
             py::arg("size"), 
             py::arg("n_transfers"),
             py::arg("is_write") = false,
             "Read data via RDMA to CPU and transfer to GPU")
        .def("rdma_process_to_gpu", &RDMAClient::rdma_process_to_gpu,
             py::arg("size"),
             py::arg("n_transfers"),
             py::arg("process_fn"),
             py::arg("is_write") = false,
             "Read data via RDMA, process it, then transfer to GPU (blocking)")
        .def("rdma_process_to_gpu_async", &RDMAClient::rdma_process_to_gpu_async,
             py::arg("size"),
             py::arg("n_transfers"),
             py::arg("process_fn"),
             py::arg("is_write") = false,
             "Read data via RDMA, process it, then transfer to GPU (async)")
        .def("read_cpu_memory", &RDMAClient::read_cpu_memory,
             py::arg("size"),
             "Read data from CPU memory")
        .def("read_gpu_memory", &RDMAClient::read_gpu_memory,
             py::arg("size"),
             "Read data from GPU memory");
} 