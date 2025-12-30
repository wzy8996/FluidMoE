cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fluid_kernels CUDA CXX)

# ============================================================================
# Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GPU Architecture (RTX 4090 = sm89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ============================================================================
# Find Dependencies
# ============================================================================

# Python - Use the exact Python from conda environment
set(Python3_ROOT_DIR "/home/zju/miniconda3/envs/megatron")
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")

# PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
message(STATUS "PyTorch: ${TORCH_INSTALL_PREFIX}")

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")

# CUTLASS (bundled as git submodule)
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass")
if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}. Run: git submodule update --init --recursive")
endif()
set(CUTLASS_INCLUDE_DIR "${CUTLASS_DIR}/include")
message(STATUS "CUTLASS: ${CUTLASS_DIR}")

# NCCL (from PyTorch or system)
find_library(NCCL_LIBRARY nccl HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/lib /usr/lib)
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/include /usr/include)
if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "NCCL library: ${NCCL_LIBRARY}")
    message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
else()
    message(WARNING "NCCL not found, some features may be disabled")
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Release flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(NCCL_INCLUDE_DIR)
    include_directories(${NCCL_INCLUDE_DIR})
endif()

# ============================================================================
# Source Files
# ============================================================================
set(KERNEL_SOURCES
    src/grouped_gemm.cu
    src/grouped_gemm_epilogue_signal.cu
    src/alltoall_fc1_signal.cu
    src/fc2_alltoall_signal.cu
    src/moe_forward_fused.cu
)

set(BINDING_SOURCES
    python/binding.cpp
)

# ============================================================================
# Build Library
# ============================================================================

# CUDA kernel library
add_library(fluid_kernels_cuda STATIC ${KERNEL_SOURCES})
target_link_libraries(fluid_kernels_cuda
    ${TORCH_LIBRARIES}
    CUDA::cudart
)
if(NCCL_LIBRARY)
    target_link_libraries(fluid_kernels_cuda ${NCCL_LIBRARY})
    target_compile_definitions(fluid_kernels_cuda PRIVATE FLUID_USE_NCCL)
endif()

# Get PyTorch lib path from TORCH_INSTALL_PREFIX which is reliable
set(TORCH_LIB_PATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch lib path: ${TORCH_LIB_PATH}")
link_directories(${TORCH_LIB_PATH})

# Python extension module
add_library(fluid_kernels SHARED ${BINDING_SOURCES})
target_link_libraries(fluid_kernels
    fluid_kernels_cuda
    ${TORCH_LIBRARIES}
    Python3::Python
    "${TORCH_LIB_PATH}/libtorch_python.so"
)

# Set output name without lib prefix
set_target_properties(fluid_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "fluid_kernels"
)

# Platform-specific extension
if(APPLE)
    set_target_properties(fluid_kernels PROPERTIES SUFFIX ".so")
else()
    set_target_properties(fluid_kernels PROPERTIES SUFFIX ".so")
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS fluid_kernels
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../ops
)

message(STATUS "============================================")
message(STATUS "Fluid Kernels Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUTLASS: ${CUTLASS_DIR}")
message(STATUS "  NCCL: ${NCCL_LIBRARY}")
message(STATUS "============================================")
