#include "grouped_gemm.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/threadblock/default_mma.h>

#include <vector>
#include <iostream>

namespace fluid {

// ============================================================================
// CUTLASS GEMM Type Definitions for different transpose combinations
// ============================================================================

// Element types
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
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

    const auto* A_ptr = static_cast<const cutlass::half_t*>(A);
    const auto* B_ptr = static_cast<const cutlass::half_t*>(B);
    auto* C_ptr = static_cast<cutlass::half_t*>(C);

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
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

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

    const auto* A_ptr = static_cast<const cutlass::half_t*>(A);
    const auto* B_ptr = static_cast<const cutlass::half_t*>(B);
    auto* C_ptr = static_cast<cutlass::half_t*>(C);

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
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

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

}  // namespace fluid
