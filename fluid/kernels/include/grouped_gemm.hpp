#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAEvent.h>
#include <vector>
#include <tuple>

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
 * Zero-sync version: accepts pre-computed total_tokens to avoid .cpu() sync.
 * This enables true overlap between GEMM and communication!
 */
torch::Tensor grouped_gemm_forward_nosync(
    torch::Tensor A,              // [total_tokens, K] or [K, total_tokens] if trans_a
    torch::Tensor B,              // [num_experts, K, N] or [num_experts, N, K] if trans_b
    torch::Tensor tokens_per_expert,  // [num_experts]
    int64_t total_tokens,         // Pre-computed, avoids .cpu() sync!
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

/**
 * Chunked Grouped GEMM for pipeline overlap with AllToAll
 *
 * Divides the computation into num_chunks chunks. Each chunk computes
 * a portion of each expert's tokens, and records a CUDA event when complete.
 * This enables overlapping AllToAll communication with computation.
 *
 * Data layout:
 * - Each expert's tokens are divided into num_chunks equal parts
 * - Chunk i processes [chunk_i_start, chunk_i_end) tokens from each expert
 * - After chunk i completes, chunk_events[i] is recorded
 *
 * @param A              Input tensor [total_tokens, K]
 * @param B              Weight tensor [num_experts, K, N] or [num_experts, N, K]
 * @param C              Output tensor [total_tokens, N]
 * @param tokens_per_expert  Number of tokens per expert [num_experts]
 * @param num_experts    Number of experts
 * @param N              Output columns
 * @param K              Contraction dimension
 * @param trans_a        Whether to transpose A
 * @param trans_b        Whether to transpose B
 * @param num_chunks     Number of chunks to divide the computation into
 * @param chunk_events   Array of CUDA events to record after each chunk
 * @param stream         CUDA stream
 */
void grouped_gemm_chunked(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int N,
    int K,
    bool trans_a,
    bool trans_b,
    int num_chunks,
    cudaEvent_t* chunk_events,
    cudaStream_t stream = nullptr
);

/**
 * Single chunk grouped GEMM computation
 *
 * Computes one chunk of the grouped GEMM. Each chunk processes the
 * corresponding portion of each expert's tokens.
 *
 * @param A              Input tensor [total_tokens, K]
 * @param B              Weight tensor [num_experts, K, N] or [num_experts, N, K]
 * @param tokens_per_expert  Number of tokens per expert [num_experts]
 * @param C              Pre-allocated output tensor [total_tokens, N]
 * @param trans_a        Whether to transpose A
 * @param trans_b        Whether to transpose B
 * @param num_chunks     Total number of chunks
 * @param chunk_idx      Index of this chunk (0 to num_chunks-1)
 * @return               The output tensor C (same as input)
 */
torch::Tensor grouped_gemm_single_chunk(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    torch::Tensor C,
    bool trans_a,
    bool trans_b,
    int num_chunks,
    int chunk_idx
);

/**
 * Get chunk info for a specific chunk
 *
 * @param tokens_per_expert  Number of tokens per expert [num_experts]
 * @param num_chunks         Total number of chunks
 * @param chunk_idx          Index of this chunk
 * @return                   Tensor [num_experts, 2] with (global_start, chunk_size) per expert
 */
torch::Tensor get_chunk_info(
    torch::Tensor tokens_per_expert,
    int num_chunks,
    int chunk_idx
);

/**
 * Compute chunk boundaries for all chunks
 *
 * @param tokens_per_expert  Number of tokens per expert [num_experts]
 * @param num_chunks         Total number of chunks
 * @return                   Tensor [num_chunks, num_experts, 3] with
 *                           (global_start, chunk_start_in_expert, chunk_size)
 */
torch::Tensor compute_chunk_boundaries(
    torch::Tensor tokens_per_expert,
    int num_chunks
);

/**
 * Grouped GEMM with Gather - for non-contiguous input access
 *
 * Computes grouped GEMM where input A is accessed via an index array,
 * eliminating the need for explicit gather (index_select) before GEMM.
 *
 * This fuses the gather operation into the GEMM kernel for better performance.
 *
 * Use case: dX computation for chunked AllToAll pipeline where each chunk's
 * input tokens are not contiguous in memory.
 *
 * @param A              Full input tensor [total_tokens, K]
 * @param B              Weight tensor [num_experts, N, K] (for trans_b=true)
 * @param indices        Index array [chunk_total] specifying which rows of A to use
 * @param tokens_per_expert  Number of tokens per expert in this chunk [num_experts]
 * @param trans_b        Whether B is transposed (currently must be true)
 * @return               Output tensor [chunk_total, N]
 */
torch::Tensor grouped_gemm_with_gather(
    torch::Tensor A,              // [total_tokens, K] - full input
    torch::Tensor B,              // [num_experts, N, K] - weights
    torch::Tensor indices,        // [chunk_total] - indices into A
    torch::Tensor tokens_per_expert,  // [num_experts] - tokens per expert in chunk
    bool trans_b = true
);

/**
 * Fused dX kernel with chunked output for pipeline overlap
 *
 * Computes the complete dX backward pass with chunked output, enabling
 * true overlap between computation and AllToAll communication.
 *
 * The kernel computes all chunks in C++ without returning to Python.
 * After each chunk is computed, chunk_ready_flags[i] is set to 1.
 * The Python code can poll these flags to know when to start AllToAll.
 *
 * @param grad_fc2       Gradient from fc2 output [total_tokens, ffn_size]
 * @param probs          Expert probabilities [total_tokens] or [total_tokens, 1]
 * @param act_deriv      Activation derivative [total_tokens, intermediate_size]
 * @param w1             Weight matrix w1 [num_experts, hidden_size, ffn_size]
 * @param w2             Weight matrix w2 [num_experts, ffn_size, hidden_size]
 * @param chunk_indices  Index array [total_chunk_tokens] for gather
 * @param chunk_offsets  Offset array [num_chunks+1] for chunk boundaries
 * @param num_chunks     Number of chunks
 * @param act_val        Optional activation values for GLU
 * @param x_2            Optional x_2 for GLU
 * @return               [output, chunk_ready_flags]
 */
std::vector<torch::Tensor> grouped_gemm_dx_chunked(
    torch::Tensor grad_fc2,         // [total_tokens, ffn_size]
    torch::Tensor probs,            // [total_tokens, 1] or [total_tokens]
    torch::Tensor act_deriv,        // [total_tokens, intermediate_size]
    torch::Tensor w1,               // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,               // [num_experts, ffn_size, hidden_size]
    torch::Tensor chunk_indices,    // [total_chunk_tokens]
    torch::Tensor chunk_offsets,    // [num_chunks+1]
    int num_chunks,
    c10::optional<torch::Tensor> act_val = c10::nullopt,
    c10::optional<torch::Tensor> x_2 = c10::nullopt
);

}  // namespace fluid
