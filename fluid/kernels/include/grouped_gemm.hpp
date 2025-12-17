#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

namespace fluid {

// Grouped GEMM configuration
struct GroupedGemmConfig {
    int num_experts;
    int hidden_size;      // Model hidden dimension
    int ffn_hidden_size;  // FFN intermediate dimension
    int tile_m = 128;     // Tile size for M dimension
    int tile_n = 128;     // Tile size for N dimension
    int tile_k = 32;      // Tile size for K dimension
};

// Problem description for a single GEMM in the group
struct GemmProblem {
    int M;  // Number of rows in A (and C)
    int N;  // Number of columns in B (and C)
    int K;  // Shared dimension

    const void* A;  // Input matrix A
    const void* B;  // Input matrix B (weight)
    void* C;        // Output matrix C

    int lda;  // Leading dimension of A
    int ldb;  // Leading dimension of B
    int ldc;  // Leading dimension of C
};

/**
 * Grouped GEMM for MoE expert computation
 *
 * Computes multiple GEMMs in parallel, one per expert:
 *   C[i] = A[i] @ B[i]           (trans_a=false, trans_b=false)
 *   C[i] = A[i]^T @ B[i]         (trans_a=true,  trans_b=false)
 *   C[i] = A[i] @ B[i]^T         (trans_a=false, trans_b=true)
 *   C[i] = A[i]^T @ B[i]^T       (trans_a=true,  trans_b=true)
 *
 * @param A              Input tensor, layout depends on trans_a
 * @param B              Weight tensor [num_experts, ...], layout depends on trans_b
 * @param C              Output tensor
 * @param tokens_per_expert  Number of tokens assigned to each expert [num_experts]
 * @param num_experts    Number of experts
 * @param M              Output rows (per expert, before grouping)
 * @param N              Output columns
 * @param K              Contraction dimension
 * @param trans_a        Whether to transpose A
 * @param trans_b        Whether to transpose B
 * @param stream         CUDA stream
 */
void grouped_gemm(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int M,
    int N,
    int K,
    bool trans_a,
    bool trans_b,
    cudaStream_t stream = nullptr
);

/**
 * PyTorch interface for grouped_gemm
 */
torch::Tensor grouped_gemm_forward(
    torch::Tensor A,              // [total_tokens, K] or [K, total_tokens] if trans_a
    torch::Tensor B,              // [num_experts, K, N] or [num_experts, N, K] if trans_b
    torch::Tensor tokens_per_expert,  // [num_experts]
    bool trans_a = false,
    bool trans_b = false
);

/**
 * Grouped GEMM for dW (weight gradient) computation
 *
 * Computes: dW[i] = A[tokens_i]^T @ B[tokens_i]
 *
 * @param A              Input activations [total_tokens, M]
 * @param B              Gradients [total_tokens, N]
 * @param C              Output weight gradients [num_experts, M, N]
 * @param tokens_per_expert  Number of tokens per expert [num_experts]
 * @param num_experts    Number of experts
 * @param M              Hidden dimension (rows of dW)
 * @param N              FFN dimension (cols of dW)
 * @param stream         CUDA stream
 */
void grouped_gemm_dw(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int M,
    int N,
    cudaStream_t stream = nullptr
);

/**
 * PyTorch interface for grouped_gemm_dw
 */
torch::Tensor grouped_gemm_dw_forward(
    torch::Tensor A,              // [total_tokens, M] - input activations
    torch::Tensor B,              // [total_tokens, N] - gradients
    torch::Tensor tokens_per_expert,  // [num_experts]
    int M,                        // hidden dimension
    int N                         // ffn dimension
);

/**
 * Grouped GEMM with tile-level control for fusion
 * Allows starting computation from a specific tile (for communication overlap)
 *
 * @param start_tile     First tile to compute
 * @param num_tiles      Number of tiles to compute (-1 for all)
 * @param barrier        Optional barrier array for synchronization
 */
void grouped_gemm_tiled(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int M,
    int N,
    int K,
    bool trans_a,
    bool trans_b,
    int start_tile,
    int num_tiles,
    int32_t* barrier,
    cudaStream_t stream = nullptr
);

}  // namespace fluid
