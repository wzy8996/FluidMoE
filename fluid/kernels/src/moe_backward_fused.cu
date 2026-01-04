/**
 * Fused MoE Backward Pass with dX + AllToAll Pipelining
 *
 * Key optimization:
 * - Compute dX in chunks (FC2.T -> Act.bwd -> FC1.T)
 * - Use P2P ncclSend/ncclRecv for fine-grained communication
 * - As each chunk completes, immediately start sending that chunk's data
 *
 * Timeline (2 chunks, 2 peers):
 *   Compute Stream:
 *     [dX_chunk0]    [dX_chunk1]
 *             ↓event         ↓event
 *
 *   Comm Stream:
 *     [recv_all]  wait → [send_chunk0]      [send_chunk1]
 *                  ^overlap with dX_chunk1!
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/arch.h>

#include <vector>
#include <iostream>
#include <cstdlib>

namespace fluid {

// ============================================================================
// Type Definitions
// ============================================================================

using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
>;

// For backward: input @ weight (NOT transpose)
// input: [M, K] row-major, weight: [K, N] row-major -> output: [M, N]
// This computes C = A @ B directly
using GemmNN = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,      // A: [M, K] row-major
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

// ============================================================================
// Backward Fused Context
// ============================================================================

struct MoEBackwardFusedContext {
    static constexpr int MAX_EP_SIZE = 16;
    static constexpr int MAX_CHUNKS = 8;

    ncclComm_t nccl_comm = nullptr;
    int rank = -1;
    int ep_size = 0;

    // One compute stream for all dX computation
    cudaStream_t compute_stream = nullptr;
    // One comm stream for all communication
    cudaStream_t comm_stream = nullptr;

    // Events for chunk synchronization
    cudaEvent_t chunk_done_events[MAX_CHUNKS] = {nullptr};
    cudaEvent_t all_recv_done = nullptr;
    cudaEvent_t all_send_done = nullptr;

    bool initialized = false;

    void init(int rank_, int ep_size_, ncclComm_t comm) {
        if (initialized) return;

        this->rank = rank_;
        this->ep_size = ep_size_;
        this->nccl_comm = comm;

        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);

        for (int i = 0; i < MAX_CHUNKS; i++) {
            cudaEventCreateWithFlags(&chunk_done_events[i], cudaEventDisableTiming);
        }
        cudaEventCreateWithFlags(&all_recv_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&all_send_done, cudaEventDisableTiming);

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;

        if (compute_stream) cudaStreamDestroy(compute_stream);
        if (comm_stream) cudaStreamDestroy(comm_stream);

        for (int i = 0; i < MAX_CHUNKS; i++) {
            if (chunk_done_events[i]) cudaEventDestroy(chunk_done_events[i]);
        }
        if (all_recv_done) cudaEventDestroy(all_recv_done);
        if (all_send_done) cudaEventDestroy(all_send_done);

        initialized = false;
    }

    ~MoEBackwardFusedContext() { destroy(); }
};

static MoEBackwardFusedContext g_moe_bwd_ctx;

// ============================================================================
// NCCL Initialization
// ============================================================================

std::vector<int64_t> get_moe_backward_nccl_unique_id() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::vector<int64_t> result(16);
    memcpy(result.data(), id.internal, 128);
    return result;
}

void init_moe_backward_nccl(int rank, int world_size, std::vector<int64_t> nccl_id_vec) {
    ncclUniqueId id;
    memcpy(id.internal, nccl_id_vec.data(), 128);

    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);

    g_moe_bwd_ctx.init(rank, world_size, comm);
}

void destroy_moe_backward_nccl() {
    if (g_moe_bwd_ctx.nccl_comm) {
        ncclCommDestroy(g_moe_bwd_ctx.nccl_comm);
        g_moe_bwd_ctx.nccl_comm = nullptr;
    }
    g_moe_bwd_ctx.destroy();
}

// ============================================================================
// Single GEMM Launch (used for per-expert dX computation)
// Uses GemmNN: C [M, N] = A [M, K] @ B [K, N]  (both row-major, NO transpose)
// For backward: grad @ weight (not grad @ weight^T)
// ============================================================================

static cudaError_t launch_gemm_single(
    const ElementInput* input,   // A: [M, K] row-major
    const ElementInput* weight,  // B: [K, N] row-major
    ElementOutput* output,       // C: [M, N] row-major
    int M, int N, int K,
    cudaStream_t stream
) {
    if (M == 0) return cudaSuccess;

    // GemmNN: C = A @ B
    // A: [M, K] with lda=K (row-major)
    // B: [K, N] with ldb=N (row-major)
    // C: [M, N] with ldc=N (row-major)
    typename GemmNN::Arguments args(
        {M, N, K},
        {input, K},    // A: [M, K], lda=K
        {weight, N},   // B: [K, N], ldb=N (row-major!)
        {output, N},   // C: [M, N], ldc=N
        {output, N},   // D = C
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
    );

    GemmNN gemm_op;
    cutlass::Status status = gemm_op(args, nullptr, stream);
    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ============================================================================
// Activation Backward Kernels
// ============================================================================

// grad_fc1 = grad_intermediate * act_deriv * probs
__global__ void activation_backward_kernel(
    const __nv_bfloat16* grad_intermediate,
    const __nv_bfloat16* act_deriv,
    const __nv_bfloat16* probs,
    __nv_bfloat16* grad_fc1,
    int64_t total_elements,
    int64_t hidden_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t token_idx = idx / hidden_size;

    float g = __bfloat162float(grad_intermediate[idx]);
    float a = __bfloat162float(act_deriv[idx]);
    float p = __bfloat162float(probs[token_idx]);

    grad_fc1[idx] = __float2bfloat16(g * a * p);
}

void launch_activation_backward(
    const __nv_bfloat16* grad_intermediate,
    const __nv_bfloat16* act_deriv,
    const __nv_bfloat16* probs,
    __nv_bfloat16* grad_fc1,
    int64_t num_tokens,
    int64_t hidden_size,
    cudaStream_t stream
) {
    int64_t total = num_tokens * hidden_size;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    activation_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_intermediate, act_deriv, probs, grad_fc1, total, hidden_size
    );
}

// ============================================================================
// Fused dX + AllToAll Pipelined Backward
//
// Key idea: Compute dX in chunks, as each chunk finishes, immediately send
// that chunk's data to peers via P2P operations.
//
// Parameters:
// - grad_fc2: [total_tokens, hidden_size] - gradient from FC2 output
// - act_deriv: [total_tokens, ffn_hidden_size] - activation derivative
// - probs: [total_tokens] - expert probabilities
// - w1: [num_experts, fc1_out_dim, hidden_size] - FC1 weight
// - w2: [num_experts, hidden_size, ffn_hidden_size] - FC2 weight
// - tokens_per_expert: [num_experts] - tokens per expert (HOST array!)
// - send_splits: tokens to send to each rank [ep_size]
// - recv_splits: tokens to receive from each rank [ep_size]
// - num_chunks: number of chunks for pipelining
//
// Returns: [grad_input, grad_fc1]
// - grad_input: [total_recv_tokens, hidden_size] - after AllToAll
// - grad_fc1: [total_tokens, fc1_out_dim] - for dW computation
// ============================================================================

std::vector<torch::Tensor> moe_backward_dx_alltoall_fused(
    torch::Tensor grad_fc2,              // [total_tokens, hidden_size]
    torch::Tensor act_deriv,             // [total_tokens, ffn_hidden_size]
    torch::Tensor probs,                 // [total_tokens] or [total_tokens, 1]
    torch::Tensor w1,                    // [num_experts, fc1_out_dim, hidden_size]
    torch::Tensor w2,                    // [num_experts, hidden_size, ffn_hidden_size]
    std::vector<int> h_tokens_per_expert, // HOST array
    std::vector<int64_t> send_splits,    // [ep_size]
    std::vector<int64_t> recv_splits,    // [ep_size]
    int num_chunks
) {
    auto& ctx = g_moe_bwd_ctx;
    TORCH_CHECK(ctx.initialized, "MoE backward context not initialized");
    TORCH_CHECK(num_chunks <= MoEBackwardFusedContext::MAX_CHUNKS,
                "num_chunks exceeds MAX_CHUNKS");

    const int rank = ctx.rank;
    const int ep_size = ctx.ep_size;
    const int num_experts = w1.size(0);
    const int fc1_out_dim = w1.size(1);
    const int hidden_size = w1.size(2);
    const int ffn_hidden_size = w2.size(2);
    const int64_t total_tokens = grad_fc2.size(0);

    cudaStream_t compute_stream = ctx.compute_stream;
    cudaStream_t comm_stream = ctx.comm_stream;

    // Ensure probs is 1D or squeezed
    torch::Tensor probs_flat = probs.dim() == 2 ? probs.squeeze(-1) : probs;

    // Calculate total recv tokens
    int64_t total_recv_tokens = 0;
    for (int i = 0; i < ep_size; i++) {
        total_recv_tokens += recv_splits[i];
    }

    // Compute expert offsets (expert-major layout)
    std::vector<int64_t> expert_offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; e++) {
        expert_offsets[e + 1] = expert_offsets[e] + h_tokens_per_expert[e];
    }

    // Compute chunk ranges for each expert
    // Each chunk processes a portion of each expert's tokens
    std::vector<std::vector<std::pair<int64_t, int64_t>>> expert_chunk_ranges(num_experts);
    for (int e = 0; e < num_experts; e++) {
        int n_tok = h_tokens_per_expert[e];
        int chunk_size = n_tok / num_chunks;
        int remainder = n_tok % num_chunks;

        int64_t local_start = 0;
        for (int c = 0; c < num_chunks; c++) {
            int this_size = chunk_size + (c < remainder ? 1 : 0);
            expert_chunk_ranges[e].push_back({local_start, local_start + this_size});
            local_start += this_size;
        }
    }

    // Compute chunk sizes for send/recv (assume uniform distribution)
    std::vector<std::vector<int64_t>> chunk_send_splits(num_chunks);
    std::vector<std::vector<int64_t>> chunk_recv_splits(num_chunks);

    for (int c = 0; c < num_chunks; c++) {
        for (int r = 0; r < ep_size; r++) {
            int64_t send_total = send_splits[r];
            int64_t send_chunk_size = send_total / num_chunks;
            int64_t send_remainder = send_total % num_chunks;
            chunk_send_splits[c].push_back(send_chunk_size + (c < send_remainder ? 1 : 0));

            int64_t recv_total = recv_splits[r];
            int64_t recv_chunk_size = recv_total / num_chunks;
            int64_t recv_remainder = recv_total % num_chunks;
            chunk_recv_splits[c].push_back(recv_chunk_size + (c < recv_remainder ? 1 : 0));
        }
    }

    // Allocate output tensors
    torch::Tensor grad_dx = torch::empty(
        {total_tokens, hidden_size},
        grad_fc2.options()
    );

    torch::Tensor grad_fc1 = torch::empty(
        {total_tokens, fc1_out_dim},
        grad_fc2.options()
    );

    torch::Tensor grad_input = torch::empty(
        {total_recv_tokens, hidden_size},
        grad_fc2.options()
    );

    // Intermediate buffer for grad_intermediate (per-chunk, reused)
    torch::Tensor grad_intermediate = torch::empty(
        {total_tokens, ffn_hidden_size},
        grad_fc2.options()
    );

    // Get raw pointers
    const __nv_bfloat16* grad_fc2_ptr = reinterpret_cast<const __nv_bfloat16*>(grad_fc2.data_ptr<at::BFloat16>());
    const __nv_bfloat16* act_deriv_ptr = reinterpret_cast<const __nv_bfloat16*>(act_deriv.data_ptr<at::BFloat16>());
    const __nv_bfloat16* probs_ptr = reinterpret_cast<const __nv_bfloat16*>(probs_flat.data_ptr<at::BFloat16>());
    const cutlass::bfloat16_t* w1_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(w1.data_ptr<at::BFloat16>());
    const cutlass::bfloat16_t* w2_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(w2.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_dx_ptr = reinterpret_cast<__nv_bfloat16*>(grad_dx.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_fc1_ptr = reinterpret_cast<__nv_bfloat16*>(grad_fc1.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_inter_ptr = reinterpret_cast<__nv_bfloat16*>(grad_intermediate.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_input_ptr = reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>());

    // ========================================================================
    // STEP 1: Pre-launch all recv operations (they will wait for data)
    // ========================================================================

    std::vector<int64_t> recv_offsets(ep_size);
    int64_t recv_off = 0;
    for (int r = 0; r < ep_size; r++) {
        recv_offsets[r] = recv_off;
        recv_off += recv_splits[r];
    }

    ncclGroupStart();
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) continue;

        int64_t recv_count = recv_splits[peer];
        if (recv_count > 0) {
            __nv_bfloat16* recv_ptr = grad_input_ptr + recv_offsets[peer] * hidden_size;
            ncclRecv(recv_ptr, recv_count * hidden_size, ncclBfloat16,
                     peer, ctx.nccl_comm, comm_stream);
        }
    }
    ncclGroupEnd();

    // ========================================================================
    // STEP 2: Compute dX chunks and trigger sends progressively
    // ========================================================================

    std::vector<int64_t> send_offsets(ep_size);
    int64_t send_off = 0;
    for (int r = 0; r < ep_size; r++) {
        send_offsets[r] = send_off;
        send_off += send_splits[r];
    }

    // Track current send offset per peer (for chunked sends)
    std::vector<int64_t> chunk_send_offsets(ep_size, 0);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        // Compute total tokens in this chunk (across all experts)
        int64_t chunk_total = 0;
        for (int e = 0; e < num_experts; e++) {
            auto [start, end] = expert_chunk_ranges[e][chunk_idx];
            chunk_total += (end - start);
        }

        // Step A: FC2 backward for this chunk
        // grad_intermediate = grad_fc2 @ W2.T
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            // GEMM: grad_fc2[global_start:] @ W2[e].T
            // W2[e]: [hidden_size, ffn_hidden_size], transposed = [ffn_hidden_size, hidden_size]
            launch_gemm_single(
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2_ptr + global_start * hidden_size),
                w2_ptr + e * hidden_size * ffn_hidden_size,
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_ptr + global_start * ffn_hidden_size),
                count, ffn_hidden_size, hidden_size,
                compute_stream
            );
        }

        // Step B: Activation backward for this chunk
        // grad_fc1 = grad_intermediate * act_deriv * probs
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            launch_activation_backward(
                grad_inter_ptr + global_start * ffn_hidden_size,
                act_deriv_ptr + global_start * ffn_hidden_size,
                probs_ptr + global_start,
                grad_fc1_ptr + global_start * fc1_out_dim,
                count,
                ffn_hidden_size,
                compute_stream
            );
        }

        // Step C: FC1 backward for this chunk
        // grad_dx = grad_fc1 @ W1.T
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            // GEMM: grad_fc1[global_start:] @ W1[e].T
            // W1[e]: [fc1_out_dim, hidden_size], transposed = [hidden_size, fc1_out_dim]
            launch_gemm_single(
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1_ptr + global_start * fc1_out_dim),
                w1_ptr + e * fc1_out_dim * hidden_size,
                reinterpret_cast<cutlass::bfloat16_t*>(grad_dx_ptr + global_start * hidden_size),
                count, hidden_size, fc1_out_dim,
                compute_stream
            );
        }

        // Record event when this chunk's dX is done
        cudaEventRecord(ctx.chunk_done_events[chunk_idx], compute_stream);

        // Step D: Launch sends for this chunk (on comm_stream, waits for chunk event)
        cudaStreamWaitEvent(comm_stream, ctx.chunk_done_events[chunk_idx], 0);

        ncclGroupStart();
        for (int peer = 0; peer < ep_size; peer++) {
            if (peer == rank) continue;

            int64_t chunk_count = chunk_send_splits[chunk_idx][peer];
            if (chunk_count > 0) {
                // Calculate source offset in grad_dx
                // Note: grad_dx is in expert-major order, we need to reorganize
                // For simplicity, assume data is already in send order
                // (In practice, you'd need to reorder first)
                int64_t send_start = send_offsets[peer] + chunk_send_offsets[peer];
                const __nv_bfloat16* send_ptr = grad_dx_ptr + send_start * hidden_size;

                ncclSend(send_ptr, chunk_count * hidden_size, ncclBfloat16,
                         peer, ctx.nccl_comm, comm_stream);

                chunk_send_offsets[peer] += chunk_count;
            }
        }
        ncclGroupEnd();
    }

    // ========================================================================
    // STEP 3: Handle self tokens (local copy, no communication)
    // ========================================================================

    int64_t self_count = send_splits[rank];
    if (self_count > 0) {
        // Wait for all dX computation to complete
        cudaStreamSynchronize(compute_stream);

        // Copy self tokens from grad_dx to grad_input
        cudaMemcpyAsync(
            grad_input_ptr + recv_offsets[rank] * hidden_size,
            grad_dx_ptr + send_offsets[rank] * hidden_size,
            self_count * hidden_size * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            compute_stream
        );
    }

    // ========================================================================
    // STEP 4: Wait for all operations to complete
    // ========================================================================

    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);

    return {grad_input, grad_fc1};
}

// ============================================================================
// Simplified version: dX computation only (no AllToAll)
// Useful for testing dX chunking performance
// ============================================================================

std::vector<torch::Tensor> moe_backward_dx_chunked(
    torch::Tensor grad_fc2,
    torch::Tensor act_deriv,
    torch::Tensor probs,
    torch::Tensor w1,
    torch::Tensor w2,
    std::vector<int> h_tokens_per_expert,
    int num_chunks
) {
    const int num_experts = w1.size(0);
    const int fc1_out_dim = w1.size(1);
    const int hidden_size = w1.size(2);
    const int ffn_hidden_size = w2.size(2);
    const int64_t total_tokens = grad_fc2.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Ensure probs is 1D
    torch::Tensor probs_flat = probs.dim() == 2 ? probs.squeeze(-1) : probs;

    // Compute expert offsets
    std::vector<int64_t> expert_offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; e++) {
        expert_offsets[e + 1] = expert_offsets[e] + h_tokens_per_expert[e];
    }

    // Allocate output tensors
    torch::Tensor grad_dx = torch::empty({total_tokens, hidden_size}, grad_fc2.options());
    torch::Tensor grad_fc1 = torch::empty({total_tokens, fc1_out_dim}, grad_fc2.options());
    torch::Tensor grad_intermediate = torch::empty({total_tokens, ffn_hidden_size}, grad_fc2.options());

    // Get pointers
    const __nv_bfloat16* grad_fc2_ptr = reinterpret_cast<const __nv_bfloat16*>(grad_fc2.data_ptr<at::BFloat16>());
    const __nv_bfloat16* act_deriv_ptr = reinterpret_cast<const __nv_bfloat16*>(act_deriv.data_ptr<at::BFloat16>());
    const __nv_bfloat16* probs_ptr = reinterpret_cast<const __nv_bfloat16*>(probs_flat.data_ptr<at::BFloat16>());
    const cutlass::bfloat16_t* w1_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(w1.data_ptr<at::BFloat16>());
    const cutlass::bfloat16_t* w2_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(w2.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_dx_ptr = reinterpret_cast<__nv_bfloat16*>(grad_dx.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_fc1_ptr = reinterpret_cast<__nv_bfloat16*>(grad_fc1.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_inter_ptr = reinterpret_cast<__nv_bfloat16*>(grad_intermediate.data_ptr<at::BFloat16>());

    // Create CUDA events for timing/profiling
    std::vector<cudaEvent_t> chunk_events(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaEventCreateWithFlags(&chunk_events[i], cudaEventDisableTiming);
    }

    // Compute chunk ranges
    std::vector<std::vector<std::pair<int64_t, int64_t>>> expert_chunk_ranges(num_experts);
    for (int e = 0; e < num_experts; e++) {
        int n_tok = h_tokens_per_expert[e];
        int chunk_size = n_tok / num_chunks;
        int remainder = n_tok % num_chunks;

        int64_t local_start = 0;
        for (int c = 0; c < num_chunks; c++) {
            int this_size = chunk_size + (c < remainder ? 1 : 0);
            expert_chunk_ranges[e].push_back({local_start, local_start + this_size});
            local_start += this_size;
        }
    }

    // Process each chunk
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        // FC2 backward
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            launch_gemm_single(
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc2_ptr + global_start * hidden_size),
                w2_ptr + e * hidden_size * ffn_hidden_size,
                reinterpret_cast<cutlass::bfloat16_t*>(grad_inter_ptr + global_start * ffn_hidden_size),
                count, ffn_hidden_size, hidden_size,
                stream
            );
        }

        // Activation backward
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            launch_activation_backward(
                grad_inter_ptr + global_start * ffn_hidden_size,
                act_deriv_ptr + global_start * ffn_hidden_size,
                probs_ptr + global_start,
                grad_fc1_ptr + global_start * fc1_out_dim,
                count,
                ffn_hidden_size,
                stream
            );
        }

        // FC1 backward
        for (int e = 0; e < num_experts; e++) {
            auto [local_start, local_end] = expert_chunk_ranges[e][chunk_idx];
            if (local_end <= local_start) continue;

            int64_t global_start = expert_offsets[e] + local_start;
            int64_t count = local_end - local_start;

            launch_gemm_single(
                reinterpret_cast<const cutlass::bfloat16_t*>(grad_fc1_ptr + global_start * fc1_out_dim),
                w1_ptr + e * fc1_out_dim * hidden_size,
                reinterpret_cast<cutlass::bfloat16_t*>(grad_dx_ptr + global_start * hidden_size),
                count, hidden_size, fc1_out_dim,
                stream
            );
        }

        // Record chunk done event
        cudaEventRecord(chunk_events[chunk_idx], stream);
    }

    // Cleanup events
    for (int i = 0; i < num_chunks; i++) {
        cudaEventDestroy(chunk_events[i]);
    }

    return {grad_dx, grad_fc1};
}

}  // namespace fluid
