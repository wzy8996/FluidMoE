#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>

namespace fluid {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = call;                                             \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(status) << std::endl;     \
            throw std::runtime_error(cudaGetErrorString(status));              \
        }                                                                      \
    } while (0)

// Get CUDA stream from PyTorch tensor
inline cudaStream_t get_stream(const torch::Tensor& tensor) {
    return c10::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
}

// Ensure tensor is contiguous and on CUDA
inline torch::Tensor ensure_cuda_contiguous(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    return tensor.contiguous();
}

// Get data pointer with type checking
template <typename T>
inline T* get_data_ptr(torch::Tensor& tensor) {
    return tensor.data_ptr<T>();
}

template <typename T>
inline const T* get_data_ptr(const torch::Tensor& tensor) {
    return tensor.data_ptr<T>();
}

// Align size to cache line (128 bytes)
inline size_t align_to_cacheline(size_t size) {
    constexpr size_t CACHELINE = 128;
    return (size + CACHELINE - 1) / CACHELINE * CACHELINE;
}

// Calculate number of thread blocks
inline int get_num_blocks(int total_threads, int block_size) {
    return (total_threads + block_size - 1) / block_size;
}

// Print tensor info for debugging
inline void print_tensor_info(const std::string& name, const torch::Tensor& tensor) {
    std::cout << name << ": shape=[";
    for (int i = 0; i < tensor.dim(); i++) {
        std::cout << tensor.size(i);
        if (i < tensor.dim() - 1) std::cout << ", ";
    }
    std::cout << "], dtype=" << tensor.dtype()
              << ", device=" << tensor.device() << std::endl;
}

} // namespace fluid
