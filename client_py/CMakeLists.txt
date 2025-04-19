cmake_minimum_required(VERSION 3.5)
project(rdma_module LANGUAGES CXX HIP)

enable_language(HIP)
find_package(HIP REQUIRED)

# Coyote setup
set(CYT_DIR ${CMAKE_SOURCE_DIR}/../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteSW REQUIRED)

execute_process(
    COMMAND which python
    OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE_PATH})

execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR} ${CMAKE_PREFIX_PATH})
# Pybind11 and Python setup
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

# Optional instance type (client or server)
set(INSTANCE "client_py" CACHE STRING "Build target instance: client_py")
set(TARGET_DIR "${CMAKE_SOURCE_DIR}/src")
message("*** Coyote RDMA Python Wrapper: Client Mode ***")
include_directories("${CMAKE_SOURCE_DIR}/include")


# Add source
set(SOURCES ${TARGET_DIR}/rdma_module.cpp)  # adjust filename if needed

# Create Python module using pybind11
pybind11_add_module(rdma_module MODULE ${SOURCES})

# Link Coyote and HIP
# target_link_libraries(rdma_module PUBLIC Coyote hip::device)
target_link_libraries(rdma_module PUBLIC Coyote)
target_link_directories(rdma_module PUBLIC /usr/local/lib)

# Target GPU architecture (adjust as needed)
set(AMD_GPU "gfx90a" CACHE STRING "Target GPU platform")
set_property(TARGET rdma_module PROPERTY HIP_ARCHITECTURES ${AMD_GPU})

# Optional definitions/macros
target_compile_definitions(rdma_module PUBLIC -D__HIP_PLATFORM_AMD__)
