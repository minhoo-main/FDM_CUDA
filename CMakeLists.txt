cmake_minimum_required(VERSION 3.18)
project(ELSPricerCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on your GPU)
# Common options: 75 (Turing), 80 (Ampere A100), 86 (Ampere RTX 30xx), 89 (Ada RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/Grid2D.cpp
    src/ELSProduct.cpp
    src/ADISolver.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/batched_thomas.cu
    src/cuda/CUDAADISolver.cu
)

# Create library
add_library(els_pricer_lib STATIC ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(els_pricer_lib
    CUDA::cudart
    CUDA::cublas
)

# Compiler flags
target_compile_options(els_pricer_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Example executable
add_executable(els_pricer examples/main.cpp)
target_link_libraries(els_pricer els_pricer_lib)

# CPU vs GPU comprehensive benchmark
add_executable(benchmark_cpu_vs_gpu examples/benchmark_cpu_vs_gpu.cpp)
target_link_libraries(benchmark_cpu_vs_gpu els_pricer_lib)

# Time grid (Nt) scaling benchmark
add_executable(benchmark_nt_scaling examples/benchmark_nt_scaling.cpp)
target_link_libraries(benchmark_nt_scaling els_pricer_lib)

# Price validation test
add_executable(validate_price examples/validate_price.cpp)
target_link_libraries(validate_price els_pricer_lib)

# Test executable (optional)
add_executable(test_els tests/test_pricing.cpp)
target_link_libraries(test_els els_pricer_lib)

# Print configuration
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
