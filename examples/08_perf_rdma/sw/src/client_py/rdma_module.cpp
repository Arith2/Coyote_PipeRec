#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <any>
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
    }

    py::array_t<int> read_data(unsigned int size, bool is_write = false) {
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

        // Invoke RDMA operation
        coyote_thread.invoke(coyote_operation, &sg);

        // Wait for completion
        while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) != 1) {}

        // Copy data from CPU to GPU
        if (hipMemcpy(mem_gpu, mem_cpu, size, hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("Failed to copy data to GPU");
        }

        // Create a numpy array that shares memory with mem_cpu
        auto result = py::array_t<int>({size / sizeof(int)}, {sizeof(int)}, mem_cpu);
        return result;
    }

    py::array_t<int> get_gpu_data(unsigned int size) {
        if (size > max_size) {
            throw std::runtime_error("Requested size exceeds buffer size");
        }

        // Copy data from GPU to CPU for Python access
        if (hipMemcpy(mem_cpu, mem_gpu, size, hipMemcpyDeviceToHost) != hipSuccess) {
            throw std::runtime_error("Failed to copy data from GPU");
        }

        // Create a numpy array that shares memory with mem_cpu
        auto result = py::array_t<int>({size / sizeof(int)}, {sizeof(int)}, mem_cpu);
        return result;
    }

    ~RDMAClient() {
        // Cleanup is handled by coyote_thread destructor
    }
};

PYBIND11_MODULE(rdma_module, m) {
    m.doc() = "RDMA client module for data transfer between CPU and GPU"; // optional module docstring

    py::class_<RDMAClient>(m, "RDMAClient")
        .def(py::init<const std::string&, unsigned int>())
        .def("read_data", &RDMAClient::read_data, 
             py::arg("size"), 
             py::arg("is_write") = false,
             "Read data via RDMA and transfer to GPU")
        .def("get_gpu_data", &RDMAClient::get_gpu_data,
             py::arg("size"),
             "Get data from GPU memory as numpy array");
} 