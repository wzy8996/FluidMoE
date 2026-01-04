#include "grouped_gemm.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/threadblock/default_mma.h>

#include <vector>
#include <iostream>

namespace fluid {

// ============================================================================
// CUTLASS GEMM Type Definitions for different transpose combinations
// ============================================================================

// Element types - Native BF16 Support
using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;

// Epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
>;

// GEMM configuration for sm80 (compatible with sm89)
// A: RowMajor, B: RowMajor (trans_a=false, trans_b=false)
using GemmNN = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,      // A: [M, K] row-major
    ElementInput, cutlass::layout::RowMajor,      // B: [K, N] row-major
    ElementOutput, cutlass::layout::RowMajor,     // C: [M, N] row-major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,       // ThreadBlock shape
    cutlass::gemm::GemmShape<64, 64, 32>,         // Warp shape
    cutlass::gemm::GemmShape<16, 8, 16>,          // MMA shape
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3  // Stages
>;

// A: RowMajor, B: ColumnMajor (trans_a=false, trans_b=true)
// B^T means: original B is [N, K], stored as ColumnMajor = RowMajor of [K, N]
using GemmNT = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,      // A: [M, K] row-major
    ElementInput, cutlass::layout::ColumnMajor,   // B: [K, N] col-major = [N, K]^T
    ElementOutput, cutlass::layout::RowMajor,     // C: [M, N] row-major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// A: ColumnMajor, B: RowMajor (trans_a=true, trans_b=false)
// A^T means: original A is [K, M], stored as ColumnMajor = RowMajor of [M, K]
using GemmTN = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,   // A: [M, K] col-major = [K, M]^T
    ElementInput, cutlass::layout::RowMajor,      // B: [K, N] row-major
    ElementOutput, cutlass::layout::RowMajor,     // C: [M, N] row-major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// A: ColumnMajor, B: ColumnMajor (trans_a=true, trans_b=true)
using GemmTT = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,   // A: [M, K] col-major = [K, M]^T
    ElementInput, cutlass::layout::ColumnMajor,   // B: [K, N] col-major = [N, K]^T
    ElementOutput, cutlass::layout::RowMajor,     // C: [M, N] row-major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// ============================================================================
// Helper: Launch a single GEMM
// ============================================================================

template<typename GemmType>
cudaError_t launch_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    cudaStream_t stream
) {
    using ElementA = typename GemmType::ElementA;
    using ElementB = typename GemmType::ElementB;
    using ElementC = typename GemmType::ElementC;

    typename GemmType::Arguments args(
        {M, N, K},                                              // Problem size
        {static_cast<const ElementA*>(A), lda},                 // A
        {static_cast<const ElementB*>(B), ldb},                 // B
        {static_cast<ElementC*>(C), ldc},                       // C (input for beta)
        {static_cast<ElementC*>(C), ldc},                       // D (output)
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}    // alpha, beta
    );

    GemmType gemm_op;
    cutlass::Status status = gemm_op(args, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

// ============================================================================
// Grouped GEMM Implementation
// ============================================================================

void grouped_gemm(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int M,  // Not used directly, computed from tokens_per_expert
    int N,
    int K,
    bool trans_a,
    bool trans_b,
    cudaStream_t stream
) {
    // Copy tokens_per_expert to host for offset calculation
    std::vector<int> h_tokens(num_experts);
    cudaMemcpy(h_tokens.data(), tokens_per_expert,
               num_experts * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate offsets for each expert
    std::vector<int> offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        offsets[i + 1] = offsets[i] + h_tokens[i];
    }

    const auto* A_ptr = static_cast<const cutlass::bfloat16_t*>(A);
    const auto* B_ptr = static_cast<const cutlass::bfloat16_t*>(B);
    auto* C_ptr = static_cast<cutlass::bfloat16_t*>(C);

    // Launch GEMM for each expert
    for (int expert = 0; expert < num_experts; expert++) {
        int m = h_tokens[expert];  // Number of tokens for this expert
        if (m == 0) continue;

        int token_offset = offsets[expert];

        // Pointer to this expert's input slice
        // A: [total_tokens, K] -> A[expert]: [m, K] starting at row token_offset
        const void* A_expert;
        int lda;
        if (!trans_a) {
            // A is [total_tokens, K], row-major
            A_expert = A_ptr + token_offset * K;
            lda = K;
        } else {
            // A is [K, total_tokens], col-major (for dW computation)
            A_expert = A_ptr + token_offset;  // Column offset
            lda = offsets[num_experts];  // Total tokens (leading dimension)
        }

        // Pointer to this expert's weight
        // B: [num_experts, K, N] or [num_experts, N, K] depending on trans_b
        const void* B_expert;
        int ldb;
        if (!trans_b) {
            // B is [num_experts, K, N], each expert's weight is [K, N]
            B_expert = B_ptr + expert * K * N;
            ldb = N;
        } else {
            // B is [num_experts, N, K], each expert's weight is [N, K]
            // When trans_b=true, we compute A @ B^T
            B_expert = B_ptr + expert * N * K;
            ldb = K;
        }

        // Output pointer
        // C: [total_tokens, N] or [num_experts, M, N] for dW
        void* C_expert;
        int ldc;
        if (!trans_a) {
            // Normal case: C is [total_tokens, N]
            C_expert = C_ptr + token_offset * N;
            ldc = N;
        } else {
            // dW case: C is [num_experts, K, N] where K is the first dim of original A
            // Actually for dW: C[expert] is [K, N]
            C_expert = C_ptr + expert * K * N;
            ldc = N;
        }

        // Launch appropriate GEMM based on transpose flags
        cudaError_t err;
        if (!trans_a && !trans_b) {
            err = launch_gemm<GemmNN>(A_expert, B_expert, C_expert, m, N, K, lda, ldb, ldc, stream);
        } else if (!trans_a && trans_b) {
            err = launch_gemm<GemmNT>(A_expert, B_expert, C_expert, m, N, K, lda, ldb, ldc, stream);
        } else if (trans_a && !trans_b) {
            // For trans_a: M and K are swapped in the GEMM call
            // Original: [K, tokens] @ [tokens, N] = [K, N]
            // CUTLASS sees: A^T [tokens, K] stored col-major, B [tokens, N]
            // Result: [K, N]
            err = launch_gemm<GemmTN>(A_expert, B_expert, C_expert, K, N, m, lda, ldb, ldc, stream);
        } else {
            err = launch_gemm<GemmTT>(A_expert, B_expert, C_expert, K, N, m, lda, ldb, ldc, stream);
        }

        if (err != cudaSuccess) {
            std::cerr << "GEMM failed for expert " << expert << std::endl;
        }
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor grouped_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    bool trans_a,
    bool trans_b
) {
    // Validate inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(tokens_per_expert.is_cuda(), "tokens_per_expert must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");

    int num_experts = tokens_per_expert.size(0);

    // Determine dimensions based on transpose flags
    int K, N;
    if (!trans_b) {
        // B is [num_experts, K, N]
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, K, N]");
        K = B.size(1);
        N = B.size(2);
    } else {
        // B is [num_experts, N, K]
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, N, K]");
        N = B.size(1);
        K = B.size(2);
    }

    // Calculate total tokens
    auto tokens_cpu = tokens_per_expert.to(torch::kCPU, torch::kInt32);
    int total_tokens = tokens_cpu.sum().item<int>();

    // Allocate output
    torch::Tensor C;
    if (!trans_a) {
        // Normal case: output is [total_tokens, N]
        C = torch::empty({total_tokens, N}, A.options());
    } else {
        // dW case: output is [num_experts, K, N]
        // Here K is the first dim of the original (non-transposed) A
        int orig_K = A.size(0);  // A is [K, total_tokens] when trans_a=true
        C = torch::empty({num_experts, orig_K, N}, A.options());
    }

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch grouped GEMM
    grouped_gemm(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        tokens_per_expert.data_ptr<int>(),
        num_experts,
        0,  // M is computed from tokens_per_expert
        N,
        K,
        trans_a,
        trans_b,
        stream
    );

    return C;
}

// ============================================================================
// Zero-sync version: accepts pre-computed total_tokens to avoid .cpu() call
// This enables true overlap between GEMM and communication!
// ============================================================================

torch::Tensor grouped_gemm_forward_nosync(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    int64_t total_tokens,  // Pre-computed, avoids .cpu() sync!
    bool trans_a,
    bool trans_b
) {
    // Validate inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(tokens_per_expert.is_cuda(), "tokens_per_expert must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");

    int num_experts = tokens_per_expert.size(0);

    // Determine dimensions based on transpose flags
    int K, N;
    if (!trans_b) {
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, K, N]");
        K = B.size(1);
        N = B.size(2);
    } else {
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, N, K]");
        N = B.size(1);
        K = B.size(2);
    }

    // Use pre-computed total_tokens - NO .cpu() call!
    // Allocate output
    torch::Tensor C;
    if (!trans_a) {
        C = torch::empty({total_tokens, N}, A.options());
    } else {
        int orig_K = A.size(0);
        C = torch::empty({num_experts, orig_K, N}, A.options());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    grouped_gemm(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        tokens_per_expert.data_ptr<int>(),
        num_experts,
        0,
        N,
        K,
        trans_a,
        trans_b,
        stream
    );

    return C;
}

// ============================================================================
// Grouped GEMM for dW computation
// dW[i] = A[tokens_i]^T @ B[tokens_i]
// A: [total_tokens, M] - input activations
// B: [total_tokens, N] - gradients
// C: [num_experts, M, N] - weight gradients
// ============================================================================

void grouped_gemm_dw(
    const void* A,
    const void* B,
    void* C,
    const int* tokens_per_expert,
    int num_experts,
    int M,  // hidden_size (output dim of A^T)
    int N,  // ffn_size (output dim)
    cudaStream_t stream
) {
    // Copy tokens_per_expert to host for offset calculation
    std::vector<int> h_tokens(num_experts);
    cudaMemcpy(h_tokens.data(), tokens_per_expert,
               num_experts * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate offsets for each expert
    std::vector<int> offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        offsets[i + 1] = offsets[i] + h_tokens[i];
    }
    int total_tokens = offsets[num_experts];

    const auto* A_ptr = static_cast<const cutlass::bfloat16_t*>(A);
    const auto* B_ptr = static_cast<const cutlass::bfloat16_t*>(B);
    auto* C_ptr = static_cast<cutlass::bfloat16_t*>(C);

    // Launch GEMM for each expert: C[i] = A[tokens_i]^T @ B[tokens_i]
    for (int expert = 0; expert < num_experts; expert++) {
        int num_tokens = h_tokens[expert];
        if (num_tokens == 0) continue;

        int token_offset = offsets[expert];

        // A slice: [num_tokens, M] at row offset token_offset
        // We need A^T, so use ColumnMajor layout
        const void* A_expert = A_ptr + token_offset * M;
        int lda = M;  // Leading dimension for row-major [tokens, M]

        // B slice: [num_tokens, N] at row offset token_offset
        const void* B_expert = B_ptr + token_offset * N;
        int ldb = N;  // Leading dimension for row-major [tokens, N]

        // Output: C[expert] = [M, N]
        void* C_expert = C_ptr + expert * M * N;
        int ldc = N;

        // Compute: A^T @ B = [M, tokens] @ [tokens, N] = [M, N]
        // Using GemmTN: A is col-major (= row-major A transposed), B is row-major
        cudaError_t err = launch_gemm<GemmTN>(
            A_expert, B_expert, C_expert,
            M,           // Output rows
            N,           // Output cols
            num_tokens,  // Contraction dimension (K)
            lda, ldb, ldc,
            stream
        );

        if (err != cudaSuccess) {
            std::cerr << "GEMM dW failed for expert " << expert << std::endl;
        }
    }
}

// PyTorch interface for dW computation
torch::Tensor grouped_gemm_dw_forward(
    torch::Tensor A,              // [total_tokens, M] - input
    torch::Tensor B,              // [total_tokens, N] - grad
    torch::Tensor tokens_per_expert,  // [num_experts]
    int M,                        // hidden dimension
    int N                         // ffn dimension
) {
    // Validate inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(tokens_per_expert.is_cuda(), "tokens_per_expert must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");

    int num_experts = tokens_per_expert.size(0);

    // Allocate output: [num_experts, M, N]
    torch::Tensor C = torch::empty({num_experts, M, N}, A.options());

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch grouped GEMM for dW
    grouped_gemm_dw(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        tokens_per_expert.data_ptr<int>(),
        num_experts,
        M,
        N,
        stream
    );

    return C;
}

// ============================================================================
// Tiled Grouped GEMM (for fusion with communication)
// ============================================================================

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
    cudaStream_t stream
) {
    // TODO: Implement tile-level control for fusion
    // For now, just call the regular grouped_gemm
    // This will be extended to support:
    // 1. Starting from a specific tile
    // 2. Computing only a subset of tiles
    // 3. Synchronizing with barrier after each tile

    grouped_gemm(A, B, C, tokens_per_expert, num_experts, M, N, K, trans_a, trans_b, stream);
}

// ============================================================================
// Chunked Grouped GEMM - Compute in chunks with events for pipeline overlap
// ============================================================================
//
// This function divides the tokens into num_chunks chunks and computes each
// chunk separately. After each chunk is computed, a CUDA event is recorded.
// This allows overlapping AllToAll communication with computation.
//
// Key insight: Each chunk contains tokens from ALL experts. We need to
// carefully handle the per-expert token distribution within each chunk.
//
// Data layout:
// - Input A: [total_tokens, K] - tokens ordered by expert
// - Input B: [num_experts, K, N] or [num_experts, N, K]
// - Output C: [total_tokens, N] - tokens ordered by expert
// - tokens_per_expert: [num_experts] - number of tokens per expert
//
// Chunking strategy:
// - Divide each expert's tokens into num_chunks equal parts
// - Process all experts' chunk_i together, then record event_i
// - This ensures each chunk's output is contiguous per expert

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
    cudaEvent_t* chunk_events,  // Array of num_chunks events to record after each chunk
    cudaStream_t stream
) {
    // Copy tokens_per_expert to host for offset calculation
    std::vector<int> h_tokens(num_experts);
    cudaMemcpy(h_tokens.data(), tokens_per_expert,
               num_experts * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate offsets for each expert
    std::vector<int> offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        offsets[i + 1] = offsets[i] + h_tokens[i];
    }
    int total_tokens = offsets[num_experts];

    const auto* A_ptr = static_cast<const cutlass::bfloat16_t*>(A);
    const auto* B_ptr = static_cast<const cutlass::bfloat16_t*>(B);
    auto* C_ptr = static_cast<cutlass::bfloat16_t*>(C);

    // Process each chunk
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // For each chunk, process all experts' portion of that chunk
        for (int expert = 0; expert < num_experts; expert++) {
            int expert_tokens = h_tokens[expert];
            if (expert_tokens == 0) continue;

            // Calculate this expert's chunk boundaries
            int chunk_size = (expert_tokens + num_chunks - 1) / num_chunks;  // Round up
            int chunk_start = chunk * chunk_size;
            int chunk_end = std::min(chunk_start + chunk_size, expert_tokens);
            int m = chunk_end - chunk_start;  // Tokens in this chunk for this expert

            if (m <= 0) continue;

            // Global token offset for this expert + chunk offset within expert
            int global_offset = offsets[expert] + chunk_start;

            // Pointer to this chunk's input slice
            const void* A_chunk;
            int lda;
            if (!trans_a) {
                // A is [total_tokens, K], row-major
                A_chunk = A_ptr + global_offset * K;
                lda = K;
            } else {
                // A is [K, total_tokens], col-major (for dW computation)
                A_chunk = A_ptr + global_offset;
                lda = total_tokens;
            }

            // Pointer to this expert's weight (same for all chunks)
            const void* B_expert;
            int ldb;
            if (!trans_b) {
                B_expert = B_ptr + expert * K * N;
                ldb = N;
            } else {
                B_expert = B_ptr + expert * N * K;
                ldb = K;
            }

            // Output pointer for this chunk
            void* C_chunk;
            int ldc;
            if (!trans_a) {
                // Normal case: C is [total_tokens, N]
                C_chunk = C_ptr + global_offset * N;
                ldc = N;
            } else {
                // dW case: Not supported for chunked version
                // dW doesn't benefit from chunking since it's for weight gradients
                C_chunk = C_ptr + expert * K * N;
                ldc = N;
            }

            // Launch appropriate GEMM based on transpose flags
            cudaError_t err;
            if (!trans_a && !trans_b) {
                err = launch_gemm<GemmNN>(A_chunk, B_expert, C_chunk, m, N, K, lda, ldb, ldc, stream);
            } else if (!trans_a && trans_b) {
                err = launch_gemm<GemmNT>(A_chunk, B_expert, C_chunk, m, N, K, lda, ldb, ldc, stream);
            } else if (trans_a && !trans_b) {
                err = launch_gemm<GemmTN>(A_chunk, B_expert, C_chunk, K, N, m, lda, ldb, ldc, stream);
            } else {
                err = launch_gemm<GemmTT>(A_chunk, B_expert, C_chunk, K, N, m, lda, ldb, ldc, stream);
            }

            if (err != cudaSuccess) {
                std::cerr << "Chunked GEMM failed for expert " << expert
                          << ", chunk " << chunk << std::endl;
            }
        }

        // Record event after this chunk is complete
        if (chunk_events != nullptr) {
            cudaEventRecord(chunk_events[chunk], stream);
        }
    }
}

// PyTorch interface for chunked grouped GEMM - compute only one chunk
// This is simpler: compute one chunk at a time, record event, then Python can dispatch AllToAll
torch::Tensor grouped_gemm_single_chunk(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    torch::Tensor C,  // Pre-allocated output tensor
    bool trans_a,
    bool trans_b,
    int num_chunks,
    int chunk_idx
) {
    // Validate inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    // Note: tokens_per_expert can now be on CPU to avoid sync overhead
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be positive");
    TORCH_CHECK(chunk_idx >= 0 && chunk_idx < num_chunks, "chunk_idx out of range");

    int num_experts = tokens_per_expert.size(0);

    // Determine dimensions based on transpose flags
    int K, N;
    if (!trans_b) {
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, K, N]");
        K = B.size(1);
        N = B.size(2);
    } else {
        TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, N, K]");
        N = B.size(1);
        K = B.size(2);
    }

    // Get tokens_per_expert - handle both CPU and GPU tensors
    std::vector<int> h_tokens(num_experts);
    if (tokens_per_expert.is_cuda()) {
        // GPU tensor - need sync (expensive!)
        cudaMemcpy(h_tokens.data(), tokens_per_expert.data_ptr<int>(),
                   num_experts * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        // CPU tensor - no sync needed (fast!)
        auto tpe_accessor = tokens_per_expert.accessor<int, 1>();
        for (int i = 0; i < num_experts; i++) {
            h_tokens[i] = tpe_accessor[i];
        }
    }

    // Calculate offsets
    std::vector<int> offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        offsets[i + 1] = offsets[i] + h_tokens[i];
    }
    int total_tokens = offsets[num_experts];

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const auto* A_ptr = static_cast<const cutlass::bfloat16_t*>(A.data_ptr());
    const auto* B_ptr = static_cast<const cutlass::bfloat16_t*>(B.data_ptr());
    auto* C_ptr = static_cast<cutlass::bfloat16_t*>(C.data_ptr());

    // Process only the specified chunk
    for (int expert = 0; expert < num_experts; expert++) {
        int expert_tokens = h_tokens[expert];
        if (expert_tokens == 0) continue;

        // Calculate this expert's chunk boundaries
        int chunk_size = (expert_tokens + num_chunks - 1) / num_chunks;
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, expert_tokens);
        int m = chunk_end - chunk_start;

        if (m <= 0) continue;

        // Global token offset
        int global_offset = offsets[expert] + chunk_start;

        // Input pointer
        const void* A_chunk;
        int lda;
        if (!trans_a) {
            A_chunk = A_ptr + global_offset * K;
            lda = K;
        } else {
            A_chunk = A_ptr + global_offset;
            lda = total_tokens;
        }

        // Weight pointer
        const void* B_expert;
        int ldb;
        if (!trans_b) {
            B_expert = B_ptr + expert * K * N;
            ldb = N;
        } else {
            B_expert = B_ptr + expert * N * K;
            ldb = K;
        }

        // Output pointer
        void* C_chunk;
        int ldc;
        if (!trans_a) {
            C_chunk = C_ptr + global_offset * N;
            ldc = N;
        } else {
            C_chunk = C_ptr + expert * K * N;
            ldc = N;
        }

        // Launch GEMM
        cudaError_t err;
        if (!trans_a && !trans_b) {
            err = launch_gemm<GemmNN>(A_chunk, B_expert, C_chunk, m, N, K, lda, ldb, ldc, stream);
        } else if (!trans_a && trans_b) {
            err = launch_gemm<GemmNT>(A_chunk, B_expert, C_chunk, m, N, K, lda, ldb, ldc, stream);
        } else if (trans_a && !trans_b) {
            err = launch_gemm<GemmTN>(A_chunk, B_expert, C_chunk, K, N, m, lda, ldb, ldc, stream);
        } else {
            err = launch_gemm<GemmTT>(A_chunk, B_expert, C_chunk, K, N, m, lda, ldb, ldc, stream);
        }

        if (err != cudaSuccess) {
            std::cerr << "Single chunk GEMM failed for expert " << expert
                      << ", chunk " << chunk_idx << std::endl;
        }
    }

    return C;
}

// Get chunk info: returns tensor of shape [num_experts, 2] with (start, size) for each expert
torch::Tensor get_chunk_info(
    torch::Tensor tokens_per_expert,
    int num_chunks,
    int chunk_idx
) {
    int num_experts = tokens_per_expert.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor chunk_info = torch::zeros({num_experts, 2}, options);

    // Copy tokens_per_expert to host
    auto tpe_cpu = tokens_per_expert.to(torch::kCPU, torch::kInt32);
    int* tpe_data = tpe_cpu.data_ptr<int>();
    int* info_data = chunk_info.data_ptr<int>();

    int total_offset = 0;
    for (int e = 0; e < num_experts; e++) {
        int expert_tokens = tpe_data[e];
        int chunk_size = (expert_tokens + num_chunks - 1) / num_chunks;
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, expert_tokens);
        int m = std::max(0, chunk_end - chunk_start);

        // Global start offset
        info_data[e * 2] = total_offset + chunk_start;
        // Chunk size
        info_data[e * 2 + 1] = m;

        total_offset += expert_tokens;
    }

    return chunk_info;
}

// Compute chunk boundaries for given tokens_per_expert
// Returns: tensor of shape [num_chunks, num_experts, 3] with (global_start, chunk_start_in_expert, chunk_size) for each chunk/expert
torch::Tensor compute_chunk_boundaries(
    torch::Tensor tokens_per_expert,
    int num_chunks
) {
    int num_experts = tokens_per_expert.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor boundaries = torch::zeros({num_chunks, num_experts, 3}, options);

    // Copy tokens_per_expert to host
    auto tpe_cpu = tokens_per_expert.to(torch::kCPU, torch::kInt32);
    int* tpe_data = tpe_cpu.data_ptr<int>();
    int* bound_data = boundaries.data_ptr<int>();

    // Calculate expert offsets
    std::vector<int> expert_offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; e++) {
        expert_offsets[e + 1] = expert_offsets[e] + tpe_data[e];
    }

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        for (int e = 0; e < num_experts; e++) {
            int expert_tokens = tpe_data[e];
            if (expert_tokens == 0) {
                // boundary at [chunk_idx, e, :] = (expert_offsets[e], 0, 0)
                int idx = (chunk_idx * num_experts + e) * 3;
                bound_data[idx] = expert_offsets[e];
                bound_data[idx + 1] = 0;
                bound_data[idx + 2] = 0;
                continue;
            }

            // Calculate this expert's chunk boundaries
            int chunk_size = (expert_tokens + num_chunks - 1) / num_chunks;  // Round up
            int chunk_start = chunk_idx * chunk_size;
            int chunk_end = std::min(chunk_start + chunk_size, expert_tokens);
            int m = std::max(0, chunk_end - chunk_start);

            int idx = (chunk_idx * num_experts + e) * 3;
            bound_data[idx] = expert_offsets[e] + chunk_start;  // global_start
            bound_data[idx + 1] = chunk_start;                   // chunk_start_in_expert
            bound_data[idx + 2] = m;                             // chunk_size
        }
    }

    return boundaries;
}

// ============================================================================
// Gather GEMM Type Definitions using GemmUniversal
// ============================================================================
// These types support gathering rows from A matrix using an index array

// GatherNT: Gather rows from A, then compute A[indices] @ B^T
// A is [total_tokens, K] row-major, we gather rows to get [M, K]
// B is [num_experts, N, K] (trans_b=true means B^T)
using GatherGemmNT = cutlass::gemm::device::GemmUniversal<
    ElementInput,
    cutlass::layout::RowMajor,      // A: row-major (gathered by row)
    ElementInput,
    cutlass::layout::ColumnMajor,   // B: col-major (for trans_b=true)
    ElementOutput,
    cutlass::layout::RowMajor,      // C: row-major
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,     // Stages
    8,     // AlignmentA
    8,     // AlignmentB
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    true,  // GatherA - gather rows from A using index array
    false, // GatherB
    false  // ScatterD
>;

// ============================================================================
// Grouped GEMM with Gather - for non-contiguous input access
// ============================================================================
//
// This function computes grouped GEMM where input A is accessed via an index
// array, eliminating the need for explicit gather (index_select) before GEMM.
//
// Use case: dX computation for chunked AllToAll pipeline where each chunk's
// input tokens are not contiguous in memory.
//
// @param A              Full input tensor [total_tokens, K]
// @param B              Weight tensor [num_experts, N, K] (for trans_b=true)
// @param C              Output tensor [chunk_total, N]
// @param indices        Index array [chunk_total] specifying which rows of A to use
// @param tokens_per_expert  Number of tokens per expert in this chunk [num_experts]
// @param num_experts    Number of experts
// @param N              Output columns (hidden_size for dX)
// @param K              Contraction dimension (ffn_size for dX)
// @param total_tokens   Total tokens in A (for lda calculation)
// @param stream         CUDA stream

torch::Tensor grouped_gemm_with_gather(
    torch::Tensor A,              // [total_tokens, K] - full input
    torch::Tensor B,              // [num_experts, N, K] - weights (trans_b=true layout)
    torch::Tensor indices,        // [chunk_total] - indices into A
    torch::Tensor tokens_per_expert,  // [num_experts] - tokens per expert in chunk
    bool trans_b
) {
    // Validate inputs
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
                "indices must be int32 or int64");
    TORCH_CHECK(trans_b, "Currently only trans_b=true is supported for gather GEMM");

    int num_experts = tokens_per_expert.size(0);
    int chunk_total = indices.size(0);
    int total_tokens = A.size(0);
    int K = A.size(1);  // Input dimension

    // B is [num_experts, N, K] for trans_b=true
    TORCH_CHECK(B.dim() == 3, "B must be 3D [num_experts, N, K]");
    int N = B.size(1);  // Output dimension
    TORCH_CHECK(B.size(2) == K, "B's K dimension must match A's K dimension");

    // Allocate output: [chunk_total, N]
    torch::Tensor C = torch::empty({chunk_total, N}, A.options());

    if (chunk_total == 0) {
        return C;
    }

    // Convert indices to int32 if needed (CUTLASS expects int)
    torch::Tensor indices_int32;
    if (indices.dtype() == torch::kInt64) {
        indices_int32 = indices.to(torch::kInt32);
    } else {
        indices_int32 = indices;
    }

    // Get tokens_per_expert on host
    std::vector<int> h_tokens(num_experts);
    if (tokens_per_expert.is_cuda()) {
        cudaMemcpy(h_tokens.data(), tokens_per_expert.data_ptr<int>(),
                   num_experts * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        auto tpe_accessor = tokens_per_expert.accessor<int, 1>();
        for (int i = 0; i < num_experts; i++) {
            h_tokens[i] = tpe_accessor[i];
        }
    }

    // Calculate offsets
    std::vector<int> offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        offsets[i + 1] = offsets[i] + h_tokens[i];
    }

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const auto* A_ptr = static_cast<const cutlass::bfloat16_t*>(A.data_ptr());
    const auto* B_ptr = static_cast<const cutlass::bfloat16_t*>(B.data_ptr());
    auto* C_ptr = static_cast<cutlass::bfloat16_t*>(C.data_ptr());
    const int* indices_ptr = indices_int32.data_ptr<int>();

    // Launch GEMM for each expert with gather
    for (int expert = 0; expert < num_experts; expert++) {
        int m = h_tokens[expert];
        if (m == 0) continue;

        int chunk_offset = offsets[expert];  // Offset in chunk output

        // For GatherA: we pass the full A matrix and the indices
        // The indices point to rows in A
        // A_ptr stays the same, indices point to the correct rows

        // B for this expert: [N, K] starting at expert * N * K
        const void* B_expert = B_ptr + expert * N * K;
        int ldb = K;  // For ColumnMajor [N, K], leading dim is K

        // Output for this expert's chunk
        void* C_chunk = C_ptr + chunk_offset * N;
        int ldc = N;

        // Indices for this expert's tokens in the chunk
        const int* expert_indices = indices_ptr + chunk_offset;

        // Use GemmUniversal with GatherA
        typename GatherGemmNT::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {m, N, K},                    // Problem size
            1,                            // Batch count (split_k_slices)
            {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},  // alpha, beta
            A_ptr,                        // A (full matrix, will be gathered)
            B_expert,                     // B (expert weight)
            C_chunk,                      // C (unused, beta=0)
            C_chunk,                      // D (output)
            total_tokens * K,             // batch_stride_A (unused for non-batched)
            0,                            // batch_stride_B
            0,                            // batch_stride_C
            0,                            // batch_stride_D
            K,                            // lda (A is [total_tokens, K] row-major)
            ldb,                          // ldb
            ldc,                          // ldc
            ldc,                          // ldd
            expert_indices,               // gather_A_indices
            nullptr,                      // gather_B_indices
            nullptr                       // scatter_D_indices
        };

        GatherGemmNT gemm_op;
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GatherGemm cannot implement for expert " << expert
                      << ", m=" << m << ", N=" << N << ", K=" << K << std::endl;
            continue;
        }

        status = gemm_op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GatherGemm initialize failed for expert " << expert << std::endl;
            continue;
        }

        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GatherGemm execution failed for expert " << expert << std::endl;
        }
    }

    return C;
}

// ============================================================================
// Fused dX computation for a single chunk - PyTorch interface
// ============================================================================
//
// This function computes dX for a single chunk, fusing:
//   1. grad_intermediate = grad_fc2[chunk_indices] @ W2.T (gather GEMM)
//   2. grad_fc1 = grad_intermediate * act_deriv[chunk_indices] * probs[chunk_indices]
//   3. dx = grad_fc1 @ W1.T
//
// The Python loop still exists, but each iteration only has one kernel call
// instead of multiple separate operations.

std::vector<torch::Tensor> grouped_gemm_dx_fused(
    torch::Tensor grad_fc2,         // [total_tokens, hidden_size]
    torch::Tensor probs,            // [total_tokens] or [total_tokens, 1]
    torch::Tensor act_deriv,        // [total_tokens, intermediate_size]
    torch::Tensor w1,               // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,               // [num_experts, intermediate_size, hidden_size]
    torch::Tensor chunk_indices,    // [chunk_size] - indices into grad_fc2
    c10::optional<torch::Tensor> act_val = c10::nullopt,
    c10::optional<torch::Tensor> x_2 = c10::nullopt
);

// ============================================================================
// Fused dX computation with all chunks (C++ loop)
// ============================================================================
//
// This function computes dX for MoE backward pass with chunked output.
// It launches GEMM kernels for each chunk and records CUDA events after each
// chunk completes, enabling overlap with AllToAll communication.
//
// The key optimization is that all chunk computations are launched from C++
// without returning to Python, eliminating Python loop overhead.
//
// Computation per chunk:
//   1. grad_intermediate = grad_fc2[chunk_indices] @ W2.T
//   2. grad_fc1 = grad_intermediate * act_deriv[chunk_indices] * probs[chunk_indices]
//   3. dx[chunk] = grad_fc1 @ W1.T
//
// For num_local_experts == 1, all tokens belong to the same expert, so we can
// use efficient batch GEMM operations.

// Helper kernel for elementwise operations: grad_fc1 = grad_inter * act_deriv * probs
__global__ void compute_grad_fc1_kernel(
    const half* __restrict__ grad_inter,      // [chunk_size, intermediate_size]
    const half* __restrict__ act_deriv,       // [total_tokens, intermediate_size]
    const half* __restrict__ probs,           // [total_tokens, 1]
    half* __restrict__ grad_fc1,              // [chunk_size, intermediate_size]
    const int* __restrict__ indices,          // [chunk_size] - indices into act_deriv/probs
    int chunk_size,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = chunk_size * intermediate_size;

    if (idx < total_elements) {
        int row = idx / intermediate_size;
        int col = idx % intermediate_size;
        int global_idx = indices[row];

        float g = __half2float(grad_inter[idx]);
        float a = __half2float(act_deriv[global_idx * intermediate_size + col]);
        float p = __half2float(probs[global_idx]);

        grad_fc1[idx] = __float2half(g * a * p);
    }
}

// Helper kernel for gated linear unit: grad_fc1 = [grad_x1, grad_x2]
// grad_x1 = grad_inter * act_deriv * x_2 * probs
// grad_x2 = grad_inter * act_val * probs
__global__ void compute_grad_fc1_glu_kernel(
    const half* __restrict__ grad_inter,      // [chunk_size, intermediate_size]
    const half* __restrict__ act_deriv,       // [total_tokens, intermediate_size]
    const half* __restrict__ act_val,         // [total_tokens, intermediate_size]
    const half* __restrict__ x_2,             // [total_tokens, intermediate_size]
    const half* __restrict__ probs,           // [total_tokens, 1]
    half* __restrict__ grad_fc1,              // [chunk_size, 2*intermediate_size]
    const int* __restrict__ indices,          // [chunk_size]
    int chunk_size,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = chunk_size * intermediate_size;

    if (idx < total_elements) {
        int row = idx / intermediate_size;
        int col = idx % intermediate_size;
        int global_idx = indices[row];

        float g = __half2float(grad_inter[idx]);
        float a = __half2float(act_deriv[global_idx * intermediate_size + col]);
        float av = __half2float(act_val[global_idx * intermediate_size + col]);
        float x2 = __half2float(x_2[global_idx * intermediate_size + col]);
        float p = __half2float(probs[global_idx]);

        // grad_x1 at position [row, col]
        grad_fc1[row * 2 * intermediate_size + col] = __float2half(g * a * x2 * p);
        // grad_x2 at position [row, intermediate_size + col]
        grad_fc1[row * 2 * intermediate_size + intermediate_size + col] = __float2half(g * av * p);
    }
}

// Fused dX computation with chunked output
// Returns: output tensor [total_chunk_tokens, hidden_size]
// Also populates chunk_ready_flags atomically when each chunk is done
void grouped_gemm_dx_chunked_impl(
    const half* grad_fc2,           // [total_tokens, ffn_size]
    const half* probs,              // [total_tokens, 1]
    const half* act_deriv,          // [total_tokens, intermediate_size]
    const half* act_val,            // [total_tokens, intermediate_size] or nullptr
    const half* x_2,                // [total_tokens, intermediate_size] or nullptr
    const half* w1,                 // [num_experts, hidden_size, ffn_size] (for trans_b)
    const half* w2,                 // [num_experts, ffn_size, hidden_size] (for trans_b)
    half* output,                   // [total_chunk_tokens, hidden_size]
    const int* chunk_indices,       // [total_chunk_tokens] - indices into grad_fc2
    const int* chunk_offsets,       // [num_chunks+1] - offset of each chunk in output
    const int* tokens_per_expert,   // [num_experts]
    int* chunk_ready_flags,         // [num_chunks] - set to 1 when chunk is ready
    int num_chunks,
    int num_experts,
    int total_tokens,
    int hidden_size,
    int ffn_size,
    int intermediate_size,
    bool gated_linear_unit,
    cudaStream_t compute_stream
) {
    // For num_experts == 1, we have a single expert
    // For each chunk:
    //   1. Use GatherGEMM: grad_inter = grad_fc2[indices] @ W2.T
    //   2. Elementwise: grad_fc1 = grad_inter * act_deriv * probs
    //   3. GEMM: dx = grad_fc1 @ W1.T
    //   4. Signal completion via atomic flag

    // Allocate temporary buffer for grad_intermediate
    // We need space for the largest chunk
    int max_chunk_size = 0;
    std::vector<int> h_chunk_offsets(num_chunks + 1);
    cudaMemcpy(h_chunk_offsets.data(), chunk_offsets, (num_chunks + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = h_chunk_offsets[c + 1] - h_chunk_offsets[c];
        max_chunk_size = std::max(max_chunk_size, chunk_size);
    }

    // Temporary buffers
    half* grad_inter_buf = nullptr;
    half* grad_fc1_buf = nullptr;
    int fc1_dim = gated_linear_unit ? 2 * intermediate_size : intermediate_size;
    cudaMalloc(&grad_inter_buf, max_chunk_size * intermediate_size * sizeof(half));
    cudaMalloc(&grad_fc1_buf, max_chunk_size * fc1_dim * sizeof(half));

    // Process each chunk
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int chunk_start = h_chunk_offsets[chunk_idx];
        int chunk_end = h_chunk_offsets[chunk_idx + 1];
        int chunk_size = chunk_end - chunk_start;

        if (chunk_size == 0) {
            // Signal empty chunk as ready
            if (chunk_ready_flags) {
                int one = 1;
                cudaMemcpyAsync(&chunk_ready_flags[chunk_idx], &one, sizeof(int),
                               cudaMemcpyHostToDevice, compute_stream);
            }
            continue;
        }

        const int* chunk_idx_ptr = chunk_indices + chunk_start;
        half* output_chunk = output + chunk_start * hidden_size;

        // Step 1: grad_inter = grad_fc2[indices] @ W2.T
        // grad_fc2: [total_tokens, hidden_size]
        // w2: [num_experts, intermediate_size, hidden_size] with trans_b
        //     acts as [intermediate_size, hidden_size].T = [hidden_size, intermediate_size]
        // grad_inter: [chunk_size, intermediate_size]
        // Using GatherGEMM: C[M,N] = A[M,K] @ B[N,K].T where A is gathered
        {
            typename GatherGemmNT::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {chunk_size, intermediate_size, hidden_size},  // M=chunk_size, N=intermediate_size, K=hidden_size
                1,
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2),
                reinterpret_cast<const cutlass::bfloat16_t*>(w2),  // [num_experts, intermediate_size, hidden_size]
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf),
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf),
                total_tokens * hidden_size,  // batch_stride_A (unused)
                0, 0, 0,
                hidden_size,        // lda: grad_fc2 is [total_tokens, hidden_size] row-major
                hidden_size,        // ldb: w2[0] is [intermediate_size, hidden_size] row-major, for NT use K
                intermediate_size,  // ldc
                intermediate_size,  // ldd
                chunk_idx_ptr,
                nullptr, nullptr
            };

            GatherGemmNT gemm_op;
            gemm_op.initialize(args, nullptr, compute_stream);
            gemm_op(compute_stream);
        }

        // Step 2: grad_fc1 = grad_inter * act_deriv * probs (elementwise)
        // grad_inter: [chunk_size, intermediate_size]
        // act_deriv: [total_tokens, intermediate_size]
        // probs: [total_tokens, 1]
        // grad_fc1: [chunk_size, fc1_dim] where fc1_dim = ffn_size
        //           For non-GLU: intermediate_size == ffn_size, so fc1_dim = intermediate_size
        //           For GLU: intermediate_size == ffn_size/2, fc1_dim = 2*intermediate_size = ffn_size
        {
            int block_size = 256;
            int num_blocks = (chunk_size * intermediate_size + block_size - 1) / block_size;

            if (gated_linear_unit) {
                // For GLU: grad_fc1 = [grad_x1, grad_x2] where
                //   grad_x1 = grad_inter * act_deriv * x_2 * probs
                //   grad_x2 = grad_inter * act_val * probs
                // Output dim = 2 * intermediate_size = ffn_size
                compute_grad_fc1_glu_kernel<<<num_blocks, block_size, 0, compute_stream>>>(
                    grad_inter_buf, act_deriv, act_val, x_2, probs,
                    grad_fc1_buf, chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            } else {
                // For non-GLU: grad_fc1 = grad_inter * act_deriv * probs
                // Output dim = intermediate_size = ffn_size
                compute_grad_fc1_kernel<<<num_blocks, block_size, 0, compute_stream>>>(
                    grad_inter_buf, act_deriv, probs,
                    grad_fc1_buf, chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            }
        }

        // Step 3: dx = grad_fc1 @ W1.T
        // grad_fc1: [chunk_size, fc1_dim] where fc1_dim = ffn_size
        // w1: [num_experts, hidden_size, ffn_size] with trans_b
        // output: [chunk_size, hidden_size]
        {
            // Use regular GEMM since grad_fc1 is already contiguous
            // C[M,N] = A[M,K] @ B[N,K].T
            typename GemmNT::Arguments args(
                {chunk_size, hidden_size, fc1_dim},  // M=chunk_size, N=hidden_size, K=fc1_dim
                {reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1_buf), fc1_dim},  // A row-major
                {reinterpret_cast<const cutlass::bfloat16_t*>(w1), fc1_dim},            // B is [hidden_size, fc1_dim] row-major, for NT use K
                {reinterpret_cast<cutlass::bfloat16_t*>(output_chunk), hidden_size},
                {reinterpret_cast<cutlass::bfloat16_t*>(output_chunk), hidden_size},
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
            );

            GemmNT gemm_op;
            gemm_op.initialize(args, nullptr, compute_stream);
            gemm_op(compute_stream);
        }

        // Step 4: Signal chunk completion
        if (chunk_ready_flags) {
            // Use a small kernel to set the flag atomically
            // This ensures the flag is set only after all previous work on this stream completes
            int one = 1;
            cudaMemcpyAsync(&chunk_ready_flags[chunk_idx], &one, sizeof(int),
                           cudaMemcpyHostToDevice, compute_stream);
        }
    }

    // Cleanup
    cudaFree(grad_inter_buf);
    cudaFree(grad_fc1_buf);
}

// PyTorch interface for fused dX with chunked output
// Computes dX for MoE backward pass:
//   1. grad_inter = grad_fc2[indices] @ W2.T  -> [chunk_size, intermediate_size]
//   2. grad_fc1 = grad_inter * act_deriv * probs (elementwise)
//   3. dx = grad_fc1 @ W1.T  -> [chunk_size, hidden_size]
//
// Weight layouts:
//   w1: [num_experts, hidden_size, ffn_size]
//   w2: [num_experts, intermediate_size, hidden_size]
//       where intermediate_size = ffn_size (non-GLU) or ffn_size/2 (GLU)
std::vector<torch::Tensor> grouped_gemm_dx_chunked(
    torch::Tensor grad_fc2,         // [total_tokens, hidden_size] - gradient from fc2 output
    torch::Tensor probs,            // [total_tokens, 1] or [total_tokens]
    torch::Tensor act_deriv,        // [total_tokens, intermediate_size] - activation derivative
    torch::Tensor w1,               // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,               // [num_experts, intermediate_size, hidden_size]
    torch::Tensor chunk_indices,    // [total_chunk_tokens]
    torch::Tensor chunk_offsets,    // [num_chunks+1]
    int num_chunks,
    c10::optional<torch::Tensor> act_val,   // for GLU: [total_tokens, intermediate_size]
    c10::optional<torch::Tensor> x_2        // for GLU: [total_tokens, intermediate_size]
) {
    TORCH_CHECK(grad_fc2.is_cuda(), "grad_fc2 must be CUDA tensor");
    TORCH_CHECK(grad_fc2.dtype() == torch::kBFloat16, "grad_fc2 must be bfloat16");

    int total_tokens = grad_fc2.size(0);
    int num_experts = w1.size(0);
    int hidden_size = w1.size(1);  // from w1 [num_experts, hidden_size, ffn_size]
    int ffn_size = w1.size(2);     // from w1 [num_experts, hidden_size, ffn_size]
    int intermediate_size = w2.size(1);  // from w2 [num_experts, intermediate_size, hidden_size]
    int total_chunk_tokens = chunk_indices.size(0);
    bool gated_linear_unit = act_val.has_value() && x_2.has_value();

    // Validate dimensions
    TORCH_CHECK(grad_fc2.size(1) == hidden_size,
                "grad_fc2 hidden_size mismatch: ", grad_fc2.size(1), " vs ", hidden_size);
    TORCH_CHECK(w2.size(2) == hidden_size,
                "w2 hidden_size mismatch: ", w2.size(2), " vs ", hidden_size);
    TORCH_CHECK(act_deriv.size(1) == intermediate_size,
                "act_deriv intermediate_size mismatch: ", act_deriv.size(1), " vs ", intermediate_size);

    // Allocate output
    torch::Tensor output = torch::empty({total_chunk_tokens, hidden_size}, grad_fc2.options());

    // Allocate chunk ready flags
    torch::Tensor chunk_ready_flags = torch::zeros({num_chunks},
        torch::TensorOptions().dtype(torch::kInt32).device(grad_fc2.device()));

    // Ensure probs is 2D
    torch::Tensor probs_2d = probs.dim() == 1 ? probs.unsqueeze(1) : probs;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    grouped_gemm_dx_chunked_impl(
        reinterpret_cast<const half*>(grad_fc2.data_ptr()),
        reinterpret_cast<const half*>(probs_2d.data_ptr()),
        reinterpret_cast<const half*>(act_deriv.data_ptr()),
        gated_linear_unit ? reinterpret_cast<const half*>(act_val->data_ptr()) : nullptr,
        gated_linear_unit ? reinterpret_cast<const half*>(x_2->data_ptr()) : nullptr,
        reinterpret_cast<const half*>(w1.data_ptr()),
        reinterpret_cast<const half*>(w2.data_ptr()),
        reinterpret_cast<half*>(output.data_ptr()),
        chunk_indices.data_ptr<int>(),
        chunk_offsets.data_ptr<int>(),
        nullptr,  // tokens_per_expert not needed for num_experts==1
        chunk_ready_flags.data_ptr<int>(),
        num_chunks,
        num_experts,
        total_tokens,
        hidden_size,
        ffn_size,
        intermediate_size,
        gated_linear_unit,
        stream
    );

    return {output, chunk_ready_flags};
}

// ============================================================================
// grouped_gemm_dx_all_chunks - Compute dX for all chunks in one C++ call
// ============================================================================
// This eliminates Python loop overhead by computing all chunks in C++.
// Returns: [full_dx, full_grad_fc1]
//   - full_dx: [total_chunk_tokens, hidden_size] (concatenated all chunk dx)
//   - full_grad_fc1: [total_chunk_tokens, fc1_dim] (concatenated all chunk grad_fc1)
//
// The caller can then split these outputs by chunk boundaries for AllToAll.
std::vector<torch::Tensor> grouped_gemm_dx_all_chunks(
    torch::Tensor grad_fc2,         // [total_tokens, hidden_size]
    torch::Tensor probs,            // [total_tokens] or [total_tokens, 1]
    torch::Tensor act_deriv,        // [total_tokens, intermediate_size]
    torch::Tensor w1,               // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,               // [num_experts, intermediate_size, hidden_size]
    torch::Tensor all_chunk_indices,    // [total_chunk_tokens] - all indices concatenated
    torch::Tensor chunk_offsets,    // [num_chunks+1] - offsets for each chunk
    int num_chunks,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
) {
    TORCH_CHECK(grad_fc2.is_cuda(), "grad_fc2 must be CUDA tensor");
    TORCH_CHECK(grad_fc2.dtype() == torch::kBFloat16, "grad_fc2 must be bfloat16");

    int total_tokens = grad_fc2.size(0);
    int hidden_size = w1.size(1);
    int ffn_size = w1.size(2);
    int intermediate_size = w2.size(1);
    int total_chunk_tokens = all_chunk_indices.size(0);
    bool gated_linear_unit = act_val.has_value() && x_2.has_value();
    int fc1_dim = gated_linear_unit ? 2 * intermediate_size : intermediate_size;

    if (total_chunk_tokens == 0) {
        return {torch::empty({0, hidden_size}, grad_fc2.options()),
                torch::empty({0, fc1_dim}, grad_fc2.options())};
    }

    // Convert indices to int32 if needed
    torch::Tensor indices_int32 = all_chunk_indices.dtype() == torch::kInt64 ?
        all_chunk_indices.to(torch::kInt32) : all_chunk_indices;

    // Allocate output tensors for all chunks
    torch::Tensor full_dx = torch::empty({total_chunk_tokens, hidden_size}, grad_fc2.options());
    torch::Tensor full_grad_fc1 = torch::empty({total_chunk_tokens, fc1_dim}, grad_fc2.options());

    // Ensure probs is 2D
    torch::Tensor probs_2d = probs.dim() == 1 ? probs.unsqueeze(1) : probs;

    // Get chunk offsets on host
    std::vector<int> h_chunk_offsets(num_chunks + 1);
    if (chunk_offsets.is_cuda()) {
        cudaMemcpy(h_chunk_offsets.data(), chunk_offsets.data_ptr<int>(),
                   (num_chunks + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        auto acc = chunk_offsets.accessor<int, 1>();
        for (int i = 0; i <= num_chunks; i++) h_chunk_offsets[i] = acc[i];
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int* indices_ptr = indices_int32.data_ptr<int>();

    // Find max chunk size for temporary buffer
    int max_chunk_size = 0;
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = h_chunk_offsets[c + 1] - h_chunk_offsets[c];
        max_chunk_size = std::max(max_chunk_size, chunk_size);
    }

    // Allocate single temp buffer for grad_intermediate (reused across chunks)
    torch::Tensor grad_inter_buf = torch::empty({max_chunk_size, intermediate_size}, grad_fc2.options());

    // Process all chunks
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int chunk_start = h_chunk_offsets[chunk_idx];
        int chunk_end = h_chunk_offsets[chunk_idx + 1];
        int chunk_size = chunk_end - chunk_start;

        if (chunk_size == 0) continue;

        const int* chunk_idx_ptr = indices_ptr + chunk_start;
        half* dx_chunk = reinterpret_cast<half*>(full_dx.data_ptr()) + chunk_start * hidden_size;
        half* grad_fc1_chunk = reinterpret_cast<half*>(full_grad_fc1.data_ptr()) + chunk_start * fc1_dim;

        // Step 1: grad_inter = grad_fc2[indices] @ W2.T
        {
            typename GatherGemmNT::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {chunk_size, intermediate_size, hidden_size},
                1,
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2.data_ptr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(w2.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf.data_ptr()),
                total_tokens * hidden_size,
                0, 0, 0,
                hidden_size, hidden_size, intermediate_size, intermediate_size,
                chunk_idx_ptr, nullptr, nullptr
            };

            GatherGemmNT gemm_op;
            gemm_op.initialize(args, nullptr, stream);
            gemm_op(stream);
        }

        // Step 2: grad_fc1 = grad_inter * act_deriv[indices] * probs[indices]
        {
            int block_size = 256;
            int num_blocks = (chunk_size * intermediate_size + block_size - 1) / block_size;

            if (gated_linear_unit) {
                compute_grad_fc1_glu_kernel<<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const half*>(grad_inter_buf.data_ptr()),
                    reinterpret_cast<const half*>(act_deriv.data_ptr()),
                    reinterpret_cast<const half*>(act_val->data_ptr()),
                    reinterpret_cast<const half*>(x_2->data_ptr()),
                    reinterpret_cast<const half*>(probs_2d.data_ptr()),
                    grad_fc1_chunk,
                    chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            } else {
                compute_grad_fc1_kernel<<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const half*>(grad_inter_buf.data_ptr()),
                    reinterpret_cast<const half*>(act_deriv.data_ptr()),
                    reinterpret_cast<const half*>(probs_2d.data_ptr()),
                    grad_fc1_chunk,
                    chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            }
        }

        // Step 3: dx = grad_fc1 @ W1.T
        {
            typename GemmNT::Arguments args(
                {chunk_size, hidden_size, fc1_dim},
                {reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1_chunk), fc1_dim},
                {reinterpret_cast<const cutlass::bfloat16_t*>(w1.data_ptr()), fc1_dim},
                {reinterpret_cast<cutlass::bfloat16_t*>(dx_chunk), hidden_size},
                {reinterpret_cast<cutlass::bfloat16_t*>(dx_chunk), hidden_size},
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
            );

            GemmNT gemm_op;
            gemm_op.initialize(args, nullptr, stream);
            gemm_op(stream);
        }
    }

    return {full_dx, full_grad_fc1};
}

// ============================================================================
// grouped_gemm_dx_pipelined - True pipelined dX computation with CUDA events
// ============================================================================
// This function computes dX for all chunks and returns CUDA events for each chunk.
// The caller (Python) can wait on each event and launch AllToAll immediately,
// achieving true overlap between dX[i+1] computation and AllToAll[i].
//
// Returns: [full_dx, full_grad_fc1, event_tensors]
//   - full_dx: [total_chunk_tokens, hidden_size]
//   - full_grad_fc1: [total_chunk_tokens, fc1_dim]
//   - event_tensors: List of int64 tensors, each containing a cudaEvent_t pointer
//
// Timeline:
//   C++ compute stream: |-- dX_0 --|-- dX_1 --|-- dX_2 --|-- ...
//                              ^event0    ^event1    ^event2
//   Python comm stream:        wait(e0) |--- A2A_0 ---|
//                                       wait(e1) |--- A2A_1 ---|
//
std::vector<torch::Tensor> grouped_gemm_dx_pipelined(
    torch::Tensor grad_fc2,             // [total_tokens, hidden_size]
    torch::Tensor probs,                // [total_tokens] or [total_tokens, 1]
    torch::Tensor act_deriv,            // [total_tokens, intermediate_size]
    torch::Tensor w1,                   // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,                   // [num_experts, intermediate_size, hidden_size]
    torch::Tensor all_chunk_indices,    // [total_chunk_tokens] - all indices concatenated
    torch::Tensor chunk_offsets,        // [num_chunks+1] - offsets for each chunk
    int num_chunks,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
) {
    TORCH_CHECK(grad_fc2.is_cuda(), "grad_fc2 must be CUDA tensor");
    TORCH_CHECK(grad_fc2.dtype() == torch::kBFloat16, "grad_fc2 must be bfloat16");

    int total_tokens = grad_fc2.size(0);
    int hidden_size = w1.size(1);
    int ffn_size = w1.size(2);
    int intermediate_size = w2.size(1);
    int total_chunk_tokens = all_chunk_indices.size(0);
    bool gated_linear_unit = act_val.has_value() && x_2.has_value();
    int fc1_dim = gated_linear_unit ? 2 * intermediate_size : intermediate_size;

    // Handle empty case
    if (total_chunk_tokens == 0 || num_chunks == 0) {
        return {torch::empty({0, hidden_size}, grad_fc2.options()),
                torch::empty({0, fc1_dim}, grad_fc2.options()),
                torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))};
    }

    // Convert indices to int32 if needed
    torch::Tensor indices_int32 = all_chunk_indices.dtype() == torch::kInt64 ?
        all_chunk_indices.to(torch::kInt32) : all_chunk_indices;

    // Allocate output tensors
    torch::Tensor full_dx = torch::empty({total_chunk_tokens, hidden_size}, grad_fc2.options());
    torch::Tensor full_grad_fc1 = torch::empty({total_chunk_tokens, fc1_dim}, grad_fc2.options());

    // Ensure probs is 2D
    torch::Tensor probs_2d = probs.dim() == 1 ? probs.unsqueeze(1) : probs;

    // Get chunk offsets on host
    std::vector<int> h_chunk_offsets(num_chunks + 1);
    if (chunk_offsets.is_cuda()) {
        cudaMemcpy(h_chunk_offsets.data(), chunk_offsets.data_ptr<int>(),
                   (num_chunks + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        auto acc = chunk_offsets.accessor<int, 1>();
        for (int i = 0; i <= num_chunks; i++) h_chunk_offsets[i] = acc[i];
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int* indices_ptr = indices_int32.data_ptr<int>();

    // Find max chunk size for temporary buffer
    int max_chunk_size = 0;
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = h_chunk_offsets[c + 1] - h_chunk_offsets[c];
        max_chunk_size = std::max(max_chunk_size, chunk_size);
    }

    // Allocate temp buffer for grad_intermediate (reused across chunks)
    torch::Tensor grad_inter_buf = torch::empty({max_chunk_size, intermediate_size}, grad_fc2.options());

    // Create CUDA events for each chunk (store as int64 pointers)
    // Events will be recorded after each chunk's dX computation completes
    std::vector<cudaEvent_t> chunk_events(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaEventCreate(&chunk_events[i]);
    }

    // Process all chunks and record events
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int chunk_start = h_chunk_offsets[chunk_idx];
        int chunk_end = h_chunk_offsets[chunk_idx + 1];
        int chunk_size = chunk_end - chunk_start;

        if (chunk_size == 0) {
            // Record event immediately for empty chunk
            cudaEventRecord(chunk_events[chunk_idx], stream);
            continue;
        }

        const int* chunk_idx_ptr = indices_ptr + chunk_start;
        half* dx_chunk = reinterpret_cast<half*>(full_dx.data_ptr()) + chunk_start * hidden_size;
        half* grad_fc1_chunk = reinterpret_cast<half*>(full_grad_fc1.data_ptr()) + chunk_start * fc1_dim;

        // Step 1: grad_inter = grad_fc2[indices] @ W2.T
        {
            typename GatherGemmNT::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {chunk_size, intermediate_size, hidden_size},
                1,
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2.data_ptr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(w2.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_buf.data_ptr()),
                total_tokens * hidden_size,
                0, 0, 0,
                hidden_size, hidden_size, intermediate_size, intermediate_size,
                chunk_idx_ptr, nullptr, nullptr
            };

            GatherGemmNT gemm_op;
            gemm_op.initialize(args, nullptr, stream);
            gemm_op(stream);
        }

        // Step 2: grad_fc1 = grad_inter * act_deriv[indices] * probs[indices]
        {
            int block_size = 256;
            int num_blocks = (chunk_size * intermediate_size + block_size - 1) / block_size;

            if (gated_linear_unit) {
                compute_grad_fc1_glu_kernel<<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const half*>(grad_inter_buf.data_ptr()),
                    reinterpret_cast<const half*>(act_deriv.data_ptr()),
                    reinterpret_cast<const half*>(act_val->data_ptr()),
                    reinterpret_cast<const half*>(x_2->data_ptr()),
                    reinterpret_cast<const half*>(probs_2d.data_ptr()),
                    grad_fc1_chunk,
                    chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            } else {
                compute_grad_fc1_kernel<<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const half*>(grad_inter_buf.data_ptr()),
                    reinterpret_cast<const half*>(act_deriv.data_ptr()),
                    reinterpret_cast<const half*>(probs_2d.data_ptr()),
                    grad_fc1_chunk,
                    chunk_idx_ptr,
                    chunk_size, intermediate_size
                );
            }
        }

        // Step 3: dx = grad_fc1 @ W1.T
        {
            typename GemmNT::Arguments args(
                {chunk_size, hidden_size, fc1_dim},
                {reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1_chunk), fc1_dim},
                {reinterpret_cast<const cutlass::bfloat16_t*>(w1.data_ptr()), fc1_dim},
                {reinterpret_cast<cutlass::bfloat16_t*>(dx_chunk), hidden_size},
                {reinterpret_cast<cutlass::bfloat16_t*>(dx_chunk), hidden_size},
                {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
            );

            GemmNT gemm_op;
            gemm_op.initialize(args, nullptr, stream);
            gemm_op(stream);
        }

        // Record event after this chunk's computation completes
        cudaEventRecord(chunk_events[chunk_idx], stream);
    }

    // Convert events to tensor (store as int64 pointers)
    torch::Tensor event_ptrs = torch::empty({num_chunks}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    int64_t* event_data = event_ptrs.data_ptr<int64_t>();
    for (int i = 0; i < num_chunks; i++) {
        event_data[i] = reinterpret_cast<int64_t>(chunk_events[i]);
    }

    return {full_dx, full_grad_fc1, event_ptrs};
}

// Helper function to wait on a CUDA event from Python
void wait_cuda_event(int64_t event_ptr, int64_t stream_ptr) {
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaStreamWaitEvent(stream, event, 0);
}

// Helper function to destroy CUDA events
void destroy_cuda_events(torch::Tensor event_ptrs) {
    int64_t* data = event_ptrs.data_ptr<int64_t>();
    int num_events = event_ptrs.size(0);
    for (int i = 0; i < num_events; i++) {
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(data[i]);
        cudaEventDestroy(event);
    }
}

// ============================================================================
// grouped_gemm_dx_fused - Single chunk fused dX computation
// ============================================================================
// For use in Python loop - fuses gather GEMM + elementwise + GEMM into one call
// Returns: [dx, grad_fc1] - both outputs for downstream dW computation
std::vector<torch::Tensor> grouped_gemm_dx_fused(
    torch::Tensor grad_fc2,         // [total_tokens, hidden_size]
    torch::Tensor probs,            // [total_tokens] or [total_tokens, 1]
    torch::Tensor act_deriv,        // [total_tokens, intermediate_size]
    torch::Tensor w1,               // [num_experts, hidden_size, ffn_size]
    torch::Tensor w2,               // [num_experts, intermediate_size, hidden_size]
    torch::Tensor chunk_indices,    // [chunk_size] - indices into grad_fc2
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
) {
    TORCH_CHECK(grad_fc2.is_cuda(), "grad_fc2 must be CUDA tensor");
    TORCH_CHECK(grad_fc2.dtype() == torch::kBFloat16, "grad_fc2 must be bfloat16");

    int total_tokens = grad_fc2.size(0);
    int num_experts = w1.size(0);
    int hidden_size = w1.size(1);
    int ffn_size = w1.size(2);
    int intermediate_size = w2.size(1);
    int chunk_size = chunk_indices.size(0);
    bool gated_linear_unit = act_val.has_value() && x_2.has_value();
    int fc1_dim = gated_linear_unit ? 2 * intermediate_size : intermediate_size;

    if (chunk_size == 0) {
        return {torch::empty({0, hidden_size}, grad_fc2.options()),
                torch::empty({0, fc1_dim}, grad_fc2.options())};
    }

    // Convert indices to int32 if needed
    torch::Tensor indices_int32;
    if (chunk_indices.dtype() == torch::kInt64) {
        indices_int32 = chunk_indices.to(torch::kInt32);
    } else {
        indices_int32 = chunk_indices;
    }

    // Allocate output and temporary buffers
    torch::Tensor output = torch::empty({chunk_size, hidden_size}, grad_fc2.options());
    torch::Tensor grad_inter = torch::empty({chunk_size, intermediate_size}, grad_fc2.options());
    torch::Tensor grad_fc1 = torch::empty({chunk_size, fc1_dim}, grad_fc2.options());

    // Ensure probs is 2D
    torch::Tensor probs_2d = probs.dim() == 1 ? probs.unsqueeze(1) : probs;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int* indices_ptr = indices_int32.data_ptr<int>();

    // Step 1: grad_inter = grad_fc2[indices] @ W2.T using GatherGEMM
    {
        typename GatherGemmNT::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {chunk_size, intermediate_size, hidden_size},
            1,
            {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
            reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2.data_ptr()),
            reinterpret_cast<const cutlass::bfloat16_t*>(w2.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(grad_inter.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(grad_inter.data_ptr()),
            total_tokens * hidden_size,
            0, 0, 0,
            hidden_size,
            hidden_size,
            intermediate_size,
            intermediate_size,
            indices_ptr,
            nullptr, nullptr
        };

        GatherGemmNT gemm_op;
        gemm_op.initialize(args, nullptr, stream);
        gemm_op(stream);
    }

    // Step 2: grad_fc1 = grad_inter * act_deriv[indices] * probs[indices]
    {
        int block_size = 256;
        int num_blocks = (chunk_size * intermediate_size + block_size - 1) / block_size;

        if (gated_linear_unit) {
            compute_grad_fc1_glu_kernel<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const half*>(grad_inter.data_ptr()),
                reinterpret_cast<const half*>(act_deriv.data_ptr()),
                reinterpret_cast<const half*>(act_val->data_ptr()),
                reinterpret_cast<const half*>(x_2->data_ptr()),
                reinterpret_cast<const half*>(probs_2d.data_ptr()),
                reinterpret_cast<half*>(grad_fc1.data_ptr()),
                indices_ptr,
                chunk_size, intermediate_size
            );
        } else {
            compute_grad_fc1_kernel<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const half*>(grad_inter.data_ptr()),
                reinterpret_cast<const half*>(act_deriv.data_ptr()),
                reinterpret_cast<const half*>(probs_2d.data_ptr()),
                reinterpret_cast<half*>(grad_fc1.data_ptr()),
                indices_ptr,
                chunk_size, intermediate_size
            );
        }
    }

    // Step 3: output = grad_fc1 @ W1.T
    {
        typename GemmNT::Arguments args(
            {chunk_size, hidden_size, fc1_dim},
            {reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1.data_ptr()), fc1_dim},
            {reinterpret_cast<const cutlass::bfloat16_t*>(w1.data_ptr()), fc1_dim},
            {reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()), hidden_size},
            {reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()), hidden_size},
            {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
        );

        GemmNT gemm_op;
        gemm_op.initialize(args, nullptr, stream);
        gemm_op(stream);
    }

    return {output, grad_fc1};
}

}  // namespace fluid
