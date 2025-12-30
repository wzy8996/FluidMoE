/**
 * AllToAll + FC1 Overlap with Local-First Strategy
 *
 * Key Idea:
 * - After router, permuted_tokens are arranged by destination rank: [to_R0][to_R1]...[to_RN]
 * - Self-copy tokens (to_R[my_rank]) can skip AllToAll network transfer
 * - Launch FC1 for self-copy tokens while AllToAll is sending/receiving remote tokens
 *
 * Two Overlap Strategies:
 *
 * 1. Local-First (alltoall_fc1_localfirst):
 *    Timeline:
 *      Self FC1:    [════════════════]
 *      AllToAll:    [════════════════]  ← Fully overlapped
 *      Remote FC1:                      [═══════════]
 *
 * 2. Per-Peer Pipelined (alltoall_fc1_pipelined):
 *    Timeline (finer-grained overlap):
 *      Self FC1:    [═════════]
 *      Peer0 Recv:  [═══════]
 *      Peer0 FC1:          [═══════]      ← Start immediately after recv
 *      Peer1 Recv:    [═════════]
 *      Peer1 FC1:              [═════════] ← Start immediately after recv
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

// NCCL for AllToAll
#include <nccl.h>

// Use existing grouped_gemm implementation
#include "grouped_gemm.hpp"

#include <vector>
#include <algorithm>

namespace fluid {

// ============================================================================
// Global NCCL Communicator for AllToAll+FC1
// ============================================================================

struct AllToAllFC1Context {
    ncclComm_t nccl_comm = nullptr;
    int nccl_rank = -1;
    int nccl_world_size = 0;
    bool nccl_initialized = false;
    bool nccl_owns_comm = false;  // Whether we created the comm

    // Persistent streams and events for low-overhead overlap (localfirst mode)
    cudaStream_t comm_stream = nullptr;      // Communication stream
    cudaStream_t compute_stream = nullptr;   // Will use PyTorch's current stream
    cudaEvent_t comm_done_event = nullptr;   // Signal when comm completes
    cudaEvent_t self_fc1_done_event = nullptr; // Signal when self FC1 completes
    bool streams_initialized = false;

    // Persistent streams and events for pipelined mode (per-peer overlap)
    static constexpr int MAX_EP_SIZE = 16;  // Support up to 16-way EP
    cudaStream_t pipelined_comm_stream = nullptr;
    cudaStream_t pipelined_peer_streams[MAX_EP_SIZE];
    cudaEvent_t pipelined_peer_recv_done[MAX_EP_SIZE];
    cudaEvent_t pipelined_peer_fc1_done[MAX_EP_SIZE];
    cudaEvent_t pipelined_comm_done = nullptr;
    cudaEvent_t pipelined_self_fc1_done = nullptr;
    int pipelined_ep_size = 0;
    bool pipelined_streams_initialized = false;

    void init_streams() {
        if (streams_initialized) return;
        // CRITICAL: Use cudaStreamNonBlocking to prevent implicit sync with default stream!
        // Without this flag, comm_stream will synchronize with default stream,
        // breaking the overlap between communication and computation.
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
        cudaEventCreate(&comm_done_event);
        cudaEventCreate(&self_fc1_done_event);
        streams_initialized = true;
    }

    void destroy_streams() {
        if (!streams_initialized) return;
        cudaStreamDestroy(comm_stream);
        cudaEventDestroy(comm_done_event);
        cudaEventDestroy(self_fc1_done_event);
        streams_initialized = false;
        comm_stream = nullptr;
        comm_done_event = nullptr;
        self_fc1_done_event = nullptr;
    }

    void init_pipelined_streams(int ep_size) {
        if (pipelined_streams_initialized && pipelined_ep_size == ep_size) return;

        destroy_pipelined_streams();

        if (ep_size > MAX_EP_SIZE) {
            throw std::runtime_error("EP size exceeds maximum supported size");
        }

        cudaStreamCreateWithFlags(&pipelined_comm_stream, cudaStreamNonBlocking);
        cudaEventCreate(&pipelined_comm_done);
        cudaEventCreate(&pipelined_self_fc1_done);

        for (int i = 0; i < ep_size; i++) {
            cudaStreamCreateWithFlags(&pipelined_peer_streams[i], cudaStreamNonBlocking);
            cudaEventCreate(&pipelined_peer_recv_done[i]);
            cudaEventCreate(&pipelined_peer_fc1_done[i]);
        }

        pipelined_ep_size = ep_size;
        pipelined_streams_initialized = true;
    }

    void destroy_pipelined_streams() {
        if (!pipelined_streams_initialized) return;

        if (pipelined_comm_stream) {
            cudaStreamDestroy(pipelined_comm_stream);
            pipelined_comm_stream = nullptr;
        }
        if (pipelined_comm_done) {
            cudaEventDestroy(pipelined_comm_done);
            pipelined_comm_done = nullptr;
        }
        if (pipelined_self_fc1_done) {
            cudaEventDestroy(pipelined_self_fc1_done);
            pipelined_self_fc1_done = nullptr;
        }

        for (int i = 0; i < pipelined_ep_size; i++) {
            if (pipelined_peer_streams[i]) {
                cudaStreamDestroy(pipelined_peer_streams[i]);
                pipelined_peer_streams[i] = nullptr;
            }
            if (pipelined_peer_recv_done[i]) {
                cudaEventDestroy(pipelined_peer_recv_done[i]);
                pipelined_peer_recv_done[i] = nullptr;
            }
            if (pipelined_peer_fc1_done[i]) {
                cudaEventDestroy(pipelined_peer_fc1_done[i]);
                pipelined_peer_fc1_done[i] = nullptr;
            }
        }

        pipelined_ep_size = 0;
        pipelined_streams_initialized = false;
    }
};

static AllToAllFC1Context* g_alltoall_fc1_ctx = nullptr;

// Initialize NCCL communicator for AllToAll+FC1
void init_alltoall_fc1_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id_vec) {
    if (g_alltoall_fc1_ctx == nullptr) {
        g_alltoall_fc1_ctx = new AllToAllFC1Context();
    }

    // Skip if already initialized with same rank/world_size
    if (g_alltoall_fc1_ctx->nccl_initialized &&
        g_alltoall_fc1_ctx->nccl_rank == rank &&
        g_alltoall_fc1_ctx->nccl_world_size == world_size) {
        return;  // Already initialized
    }

    // If previously initialized with different config, destroy first
    if (g_alltoall_fc1_ctx->nccl_initialized &&
        g_alltoall_fc1_ctx->nccl_owns_comm &&
        g_alltoall_fc1_ctx->nccl_comm != nullptr) {
        ncclCommDestroy(g_alltoall_fc1_ctx->nccl_comm);
        g_alltoall_fc1_ctx->nccl_initialized = false;
    }

    // Convert vector back to ncclUniqueId
    ncclUniqueId nccl_id;
    static_assert(sizeof(ncclUniqueId) == 128, "ncclUniqueId size must be 128 bytes");
    if (nccl_id_vec.size() < 16) {
        throw std::runtime_error("nccl_id must have at least 16 int64 elements (128 bytes)");
    }
    memcpy(&nccl_id, nccl_id_vec.data(), sizeof(ncclUniqueId));

    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, nccl_id, rank);

    g_alltoall_fc1_ctx->nccl_comm = comm;
    g_alltoall_fc1_ctx->nccl_rank = rank;
    g_alltoall_fc1_ctx->nccl_world_size = world_size;
    g_alltoall_fc1_ctx->nccl_initialized = true;
    g_alltoall_fc1_ctx->nccl_owns_comm = true;

    // Initialize persistent streams and events
    g_alltoall_fc1_ctx->init_streams();
}

// Get the global NCCL communicator
ncclComm_t get_alltoall_fc1_nccl_comm() {
    if (g_alltoall_fc1_ctx == nullptr || !g_alltoall_fc1_ctx->nccl_initialized) {
        throw std::runtime_error("AllToAll+FC1 NCCL communicator not initialized. Call init_alltoall_fc1_nccl first.");
    }
    return g_alltoall_fc1_ctx->nccl_comm;
}

// Cleanup
void destroy_alltoall_fc1_nccl() {
    if (g_alltoall_fc1_ctx != nullptr) {
        g_alltoall_fc1_ctx->destroy_streams();
        g_alltoall_fc1_ctx->destroy_pipelined_streams();

        if (g_alltoall_fc1_ctx->nccl_initialized &&
            g_alltoall_fc1_ctx->nccl_owns_comm &&
            g_alltoall_fc1_ctx->nccl_comm != nullptr) {
            ncclCommDestroy(g_alltoall_fc1_ctx->nccl_comm);
        }
        delete g_alltoall_fc1_ctx;
        g_alltoall_fc1_ctx = nullptr;
    }
}

// ============================================================================
// Local-First Overlap: Self FC1 || AllToAll -> Remote FC1
// ============================================================================

/**
 * Compute indices for Local-First overlap
 *
 * Call this ONCE when routing changes, then reuse indices in alltoall_fc1_localfirst
 *
 * Returns: [sort_indices, remote_input_indices, self_output_indices,
 *           remote_output_indices, remote_tokens_per_expert]
 */
std::vector<torch::Tensor> compute_localfirst_indices(
    torch::Tensor tokens_per_expert,
    torch::Tensor self_tokens_per_expert,
    torch::Tensor num_global_tokens_per_local_expert,
    int my_rank,
    torch::Device device
) {
    int ep_size = num_global_tokens_per_local_expert.size(0);
    int num_local_experts = num_global_tokens_per_local_expert.size(1);

    auto num_global_cpu = num_global_tokens_per_local_expert.cpu();
    auto tokens_per_expert_cpu = tokens_per_expert.cpu();
    auto self_tokens_per_expert_cpu = self_tokens_per_expert.cpu();

    auto num_global_acc = num_global_cpu.accessor<int64_t, 2>();
    auto tokens_per_expert_acc = tokens_per_expert_cpu.accessor<int32_t, 1>();
    auto self_tokens_per_expert_acc = self_tokens_per_expert_cpu.accessor<int32_t, 1>();

    int64_t total_output_tokens = 0;
    for (int s = 0; s < ep_size; s++) {
        for (int e = 0; e < num_local_experts; e++) {
            total_output_tokens += num_global_acc[s][e];
        }
    }

    // Compute Sort indices
    std::vector<int64_t> sort_indices;
    sort_indices.reserve(total_output_tokens);
    for (int expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        for (int src_rank = 0; src_rank < ep_size; src_rank++) {
            int64_t read_offset = 0;
            for (int r = 0; r < src_rank; r++) {
                for (int e = 0; e < num_local_experts; e++) {
                    read_offset += num_global_acc[r][e];
                }
            }
            for (int e = 0; e < expert_idx; e++) {
                read_offset += num_global_acc[src_rank][e];
            }
            int64_t chunk_size = num_global_acc[src_rank][expert_idx];
            for (int64_t i = 0; i < chunk_size; i++) {
                sort_indices.push_back(read_offset + i);
            }
        }
    }

    // Compute remote_input_indices and output indices
    std::vector<int64_t> remote_input_indices;
    std::vector<int64_t> self_output_indices;
    std::vector<int64_t> remote_output_indices;

    int64_t sorted_pos = 0;
    for (int expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        for (int src_rank = 0; src_rank < ep_size; src_rank++) {
            int64_t chunk_size = num_global_acc[src_rank][expert_idx];
            if (chunk_size > 0) {
                for (int64_t i = 0; i < chunk_size; i++) {
                    if (src_rank == my_rank) {
                        self_output_indices.push_back(sorted_pos + i);
                    } else {
                        remote_input_indices.push_back(sorted_pos + i);
                        remote_output_indices.push_back(sorted_pos + i);
                    }
                }
                sorted_pos += chunk_size;
            }
        }
    }

    // Compute remote_tokens_per_expert
    std::vector<int32_t> remote_tokens_vec(num_local_experts);
    for (int i = 0; i < num_local_experts; i++) {
        remote_tokens_vec[i] = tokens_per_expert_acc[i] - self_tokens_per_expert_acc[i];
    }

    // Transfer to GPU
    torch::Tensor sort_indices_tensor = torch::from_blob(
        sort_indices.data(), {static_cast<int64_t>(sort_indices.size())}, torch::kInt64
    ).clone().to(device);

    torch::Tensor remote_input_indices_tensor = torch::from_blob(
        remote_input_indices.data(), {static_cast<int64_t>(remote_input_indices.size())}, torch::kInt64
    ).clone().to(device);

    torch::Tensor self_output_indices_tensor = torch::from_blob(
        self_output_indices.data(), {static_cast<int64_t>(self_output_indices.size())}, torch::kInt64
    ).clone().to(device);

    torch::Tensor remote_output_indices_tensor = torch::from_blob(
        remote_output_indices.data(), {static_cast<int64_t>(remote_output_indices.size())}, torch::kInt64
    ).clone().to(device);

    torch::Tensor remote_tokens_tensor = torch::from_blob(
        remote_tokens_vec.data(), {num_local_experts}, torch::kInt32
    ).clone().to(device);

    return {sort_indices_tensor, remote_input_indices_tensor, self_output_indices_tensor,
            remote_output_indices_tensor, remote_tokens_tensor};
}

/**
 * AllToAll + FC1 with Local-First Overlap (Zero-Sync Version)
 *
 * This is the BEST performing version that achieves TRUE overlap between
 * AllToAll communication and Self FC1 computation.
 *
 * Key Optimizations:
 * 1. Uses cudaStreamNonBlocking for comm_stream to prevent implicit sync
 * 2. All CPU values (splits, offsets) are pre-computed and passed in
 * 3. Uses grouped_gemm_forward_nosync to avoid .cpu() sync inside GEMM
 * 4. Uses index_copy_ for direct scatter writes (faster than cat+index_select)
 *
 * Timeline:
 *   AllToAll comm:  [════════════════]  <- On comm_stream
 *   Self FC1:       [════════════════]  <- On compute_stream (TRUE OVERLAP!)
 *   Sort:                              [=]
 *   Remote FC1:                            [===========]
 *   Scatter:                                           [=]
 */
std::vector<torch::Tensor> alltoall_fc1_localfirst(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    // Pre-computed CPU values passed as vectors (no .cpu() calls needed!)
    std::vector<int64_t> input_splits_vec,   // [ep_size]
    std::vector<int64_t> output_splits_vec,  // [ep_size]
    int64_t self_input_offset,
    int64_t self_input_count,
    // GPU tensors
    torch::Tensor self_tokens_per_expert,
    torch::Tensor sort_indices,
    torch::Tensor remote_input_indices,
    torch::Tensor self_output_indices,
    torch::Tensor remote_output_indices,
    torch::Tensor remote_tokens_per_expert,
    int my_rank
) {
    int hidden_size = permuted_tokens.size(1);
    int ffn_hidden_size = fc1_weight.size(2);
    int ep_size = input_splits_vec.size();
    int64_t total_output_tokens = sort_indices.size(0);

    ncclComm_t nccl_comm = get_alltoall_fc1_nccl_comm();
    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t comm_stream = g_alltoall_fc1_ctx->comm_stream;
    cudaEvent_t alltoall_done = g_alltoall_fc1_ctx->comm_done_event;

    auto options = torch::TensorOptions()
        .dtype(permuted_tokens.dtype())
        .device(permuted_tokens.device());

    // Extract self tokens (no .cpu() needed - offset/count passed in)
    torch::Tensor self_copy_tokens = permuted_tokens.slice(0, self_input_offset, self_input_offset + self_input_count);

    // Allocate buffers
    torch::Tensor output_tokens = torch::empty({total_output_tokens, hidden_size}, options);
    torch::Tensor final_output = torch::empty({total_output_tokens, ffn_hidden_size}, options);

    // Setup NCCL (using pre-computed vectors - no .cpu() calls!)
    auto input_ptr = permuted_tokens.data_ptr<at::Half>();
    auto output_ptr = output_tokens.data_ptr<at::Half>();

    std::vector<size_t> send_counts(ep_size), recv_counts(ep_size);
    std::vector<size_t> send_offsets(ep_size), recv_offsets(ep_size);
    size_t send_offset = 0, recv_offset = 0;
    for (int i = 0; i < ep_size; i++) {
        send_counts[i] = input_splits_vec[i] * hidden_size;
        recv_counts[i] = output_splits_vec[i] * hidden_size;
        send_offsets[i] = send_offset;
        recv_offsets[i] = recv_offset;
        send_offset += send_counts[i];
        recv_offset += recv_counts[i];
    }

    // Step 1: Launch AllToAll on comm_stream FIRST
    ncclGroupStart();
    for (int peer = 0; peer < ep_size; peer++) {
        if (send_counts[peer] > 0) {
            ncclSend(reinterpret_cast<half*>(input_ptr) + send_offsets[peer],
                     send_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
        if (recv_counts[peer] > 0) {
            ncclRecv(reinterpret_cast<half*>(output_ptr) + recv_offsets[peer],
                     recv_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
    }
    ncclGroupEnd();
    cudaEventRecord(alltoall_done, comm_stream);

    // Step 2: Self FC1 on compute_stream (PARALLEL with AllToAll!)
    torch::Tensor self_fc1_output;
    if (self_input_count > 0) {
        self_fc1_output = grouped_gemm_forward_nosync(
            self_copy_tokens, fc1_weight, self_tokens_per_expert,
            self_input_count, false, false);
        self_fc1_output = torch::gelu(self_fc1_output);
    }

    // Step 3: Wait for AllToAll, then Sort
    cudaStreamWaitEvent(compute_stream, alltoall_done);
    torch::Tensor sorted_tokens = torch::index_select(output_tokens, 0, sort_indices);

    // Step 4: Extract remote tokens and compute Remote FC1
    int64_t remote_token_count = remote_input_indices.size(0);

    if (remote_token_count > 0) {
        torch::Tensor remote_tokens = torch::index_select(sorted_tokens, 0, remote_input_indices);

        torch::Tensor remote_fc1_output = grouped_gemm_forward_nosync(
            remote_tokens, fc1_weight, remote_tokens_per_expert,
            remote_token_count, false, false);
        remote_fc1_output = torch::gelu(remote_fc1_output);

        // Direct scatter writes
        if (self_fc1_output.defined() && self_output_indices.size(0) > 0) {
            final_output.index_copy_(0, self_output_indices, self_fc1_output);
        }
        if (remote_output_indices.size(0) > 0) {
            final_output.index_copy_(0, remote_output_indices, remote_fc1_output);
        }
    } else {
        if (self_fc1_output.defined() && self_output_indices.size(0) > 0) {
            final_output.index_copy_(0, self_output_indices, self_fc1_output);
        }
    }

    return {final_output};
}

// ============================================================================
// Per-Peer Pipelined: Finer-grained overlap
// ============================================================================

/**
 * Per-Peer Pipelined AllToAll + FC1 (Zero-Sync Version)
 *
 * Key Improvement over alltoall_fc1_localfirst:
 * - Instead of waiting for all AllToAll to complete, compute FC1 as each peer's data arrives
 * - Achieves higher overlap when AllToAll time > Self FC1 time
 *
 * Two execution modes:
 *   serialize_peer_fc1=true:  Serialize peer FC1 to avoid SM competition (recommended)
 *   serialize_peer_fc1=false: Parallel peer FC1 on independent streams (may cause SM competition)
 *
 * Timeline (serialize_peer_fc1=true):
 *   Self FC1:      [═════════]
 *   AllToAll:      [═══════════════════════]
 *   Peer0 FC1:               [═════════]      <- Wait for peer0 data + previous FC1
 *   Peer1 FC1:                         [═════════]
 */
std::vector<torch::Tensor> alltoall_fc1_pipelined(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    // Pre-computed CPU values (no .cpu() calls needed!)
    std::vector<int64_t> input_splits_vec,
    std::vector<int64_t> output_splits_vec,
    int64_t self_input_offset,
    int64_t self_input_count,
    // GPU tensors
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,  // Pre-computed on GPU
    std::vector<int64_t> peer_token_counts,                  // Pre-computed token counts
    // Merge indices (pre-computed)
    std::vector<int64_t> num_global_flat,  // [ep_size * num_local_experts] flattened
    int num_local_experts,
    int my_rank,
    bool serialize_peer_fc1 = true
) {
    int hidden_size = permuted_tokens.size(1);
    int ffn_hidden_size = fc1_weight.size(2);
    int ep_size = input_splits_vec.size();

    ncclComm_t nccl_comm = get_alltoall_fc1_nccl_comm();
    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();

    // Create independent streams for each peer
    std::vector<cudaStream_t> peer_streams(ep_size);
    std::vector<cudaEvent_t> peer_recv_done(ep_size);
    std::vector<cudaEvent_t> peer_fc1_done(ep_size);

    for (int i = 0; i < ep_size; i++) {
        cudaStreamCreateWithFlags(&peer_streams[i], cudaStreamNonBlocking);
        cudaEventCreate(&peer_recv_done[i]);
        cudaEventCreate(&peer_fc1_done[i]);
    }

    // Calculate total output tokens (no .cpu() needed!)
    int64_t total_output_tokens = 0;
    for (int i = 0; i < ep_size; i++) {
        total_output_tokens += output_splits_vec[i];
    }

    auto options = torch::TensorOptions()
        .dtype(permuted_tokens.dtype())
        .device(permuted_tokens.device());

    torch::Tensor output_tokens = torch::empty({total_output_tokens, hidden_size}, options);

    // Build send/recv offsets (using pre-computed vectors - no .cpu()!)
    std::vector<size_t> send_counts(ep_size), recv_counts(ep_size);
    std::vector<size_t> send_offsets(ep_size), recv_offsets(ep_size);

    size_t send_offset = 0, recv_offset = 0;
    for (int i = 0; i < ep_size; i++) {
        send_counts[i] = input_splits_vec[i] * hidden_size;
        recv_counts[i] = output_splits_vec[i] * hidden_size;
        send_offsets[i] = send_offset;
        recv_offsets[i] = recv_offset;
        send_offset += send_counts[i];
        recv_offset += recv_counts[i];
    }

    auto input_ptr = permuted_tokens.data_ptr<at::Half>();
    auto output_ptr = output_tokens.data_ptr<at::Half>();

    // Create single communication stream
    cudaStream_t comm_stream;
    cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
    cudaEvent_t comm_done;
    cudaEventCreate(&comm_done);

    // ========================================
    // Step 1: Launch AllToAll on comm_stream FIRST
    // ========================================
    ncclGroupStart();
    for (int peer = 0; peer < ep_size; peer++) {
        if (send_counts[peer] > 0) {
            ncclSend(reinterpret_cast<half*>(input_ptr) + send_offsets[peer],
                     send_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
        if (recv_counts[peer] > 0) {
            ncclRecv(reinterpret_cast<half*>(output_ptr) + recv_offsets[peer],
                     recv_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
    }
    ncclGroupEnd();
    cudaEventRecord(comm_done, comm_stream);

    for (int peer = 0; peer < ep_size; peer++) {
        cudaEventRecord(peer_recv_done[peer], comm_stream);
    }

    // ========================================
    // Step 2: Launch Self FC1 on compute_stream (PARALLEL with AllToAll!)
    // ========================================
    torch::Tensor self_copy_tokens = permuted_tokens.slice(
        0, self_input_offset, self_input_offset + self_input_count
    );

    torch::Tensor self_fc1_output;
    cudaEvent_t self_fc1_done_event;
    cudaEventCreate(&self_fc1_done_event);

    if (self_input_count > 0) {
        // Use nosync version - no .cpu() sync!
        self_fc1_output = grouped_gemm_forward_nosync(
            self_copy_tokens, fc1_weight, self_tokens_per_expert,
            self_input_count, false, false
        );
        self_fc1_output = torch::gelu(self_fc1_output);
        cudaEventRecord(self_fc1_done_event, compute_stream);
    }

    std::vector<torch::Tensor> peer_fc1_outputs(ep_size);

    // ========================================
    // Step 3: Compute Peer FC1
    // ========================================
    if (serialize_peer_fc1) {
        // Serialize peer FC1 to avoid SM competition
        cudaEvent_t prev_fc1_done = self_fc1_done_event;

        for (int peer = 0; peer < ep_size; peer++) {
            if (peer == my_rank || recv_counts[peer] == 0) continue;

            cudaStreamWaitEvent(compute_stream, prev_fc1_done);
            cudaStreamWaitEvent(compute_stream, peer_recv_done[peer]);

            int64_t peer_token_offset = recv_offsets[peer] / hidden_size;
            int64_t peer_token_count = peer_token_counts[peer];

            torch::Tensor peer_tokens = output_tokens.slice(
                0, peer_token_offset, peer_token_offset + peer_token_count
            );

            // Use nosync version - no .cpu() sync!
            peer_fc1_outputs[peer] = grouped_gemm_forward_nosync(
                peer_tokens, fc1_weight, peer_tokens_per_expert_vec[peer],
                peer_token_count, false, false
            );
            peer_fc1_outputs[peer] = torch::gelu(peer_fc1_outputs[peer]);

            cudaEventRecord(peer_fc1_done[peer], compute_stream);
            prev_fc1_done = peer_fc1_done[peer];
        }
    } else {
        // Parallel peer FC1 (may cause SM competition)
        for (int peer = 0; peer < ep_size; peer++) {
            if (peer == my_rank || recv_counts[peer] == 0) continue;

            cudaStreamWaitEvent(peer_streams[peer], peer_recv_done[peer]);

            int64_t peer_token_offset = recv_offsets[peer] / hidden_size;
            int64_t peer_token_count = peer_token_counts[peer];

            torch::Tensor peer_tokens = output_tokens.slice(
                0, peer_token_offset, peer_token_offset + peer_token_count
            );

            cudaStream_t saved_stream = c10::cuda::getCurrentCUDAStream();
            at::cuda::CUDAStream peer_cuda_stream =
                at::cuda::getStreamFromExternal(peer_streams[peer], peer_tokens.device().index());
            c10::cuda::setCurrentCUDAStream(peer_cuda_stream);

            // Use nosync version - no .cpu() sync!
            peer_fc1_outputs[peer] = grouped_gemm_forward_nosync(
                peer_tokens, fc1_weight, peer_tokens_per_expert_vec[peer],
                peer_token_count, false, false
            );
            peer_fc1_outputs[peer] = torch::gelu(peer_fc1_outputs[peer]);

            cudaEventRecord(peer_fc1_done[peer], peer_streams[peer]);

            at::cuda::CUDAStream saved_cuda_stream =
                at::cuda::getStreamFromExternal(saved_stream, peer_tokens.device().index());
            c10::cuda::setCurrentCUDAStream(saved_cuda_stream);
        }
    }

    // ========================================
    // Step 4: Wait for all peer FC1 to complete
    // ========================================
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer != my_rank && recv_counts[peer] > 0) {
            cudaStreamWaitEvent(compute_stream, peer_fc1_done[peer]);
        }
    }
    if (self_input_count > 0) {
        cudaStreamWaitEvent(compute_stream, self_fc1_done_event);
    }

    // ========================================
    // Step 5: Merge all FC1 outputs in expert-major order
    // ========================================
    torch::Tensor combined_fc1_output = torch::empty(
        {total_output_tokens, ffn_hidden_size},
        options.dtype(fc1_weight.dtype())
    );

    // Use pre-computed num_global_flat (no .cpu() needed!)
    int64_t write_offset = 0;
    for (int expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        for (int src_rank = 0; src_rank < ep_size; src_rank++) {
            int64_t chunk_size = num_global_flat[src_rank * num_local_experts + expert_idx];
            if (chunk_size == 0) continue;

            int64_t peer_read_offset = 0;
            for (int e = 0; e < expert_idx; e++) {
                peer_read_offset += num_global_flat[src_rank * num_local_experts + e];
            }

            if (src_rank == my_rank) {
                combined_fc1_output.slice(0, write_offset, write_offset + chunk_size).copy_(
                    self_fc1_output.slice(0, peer_read_offset, peer_read_offset + chunk_size)
                );
            } else {
                combined_fc1_output.slice(0, write_offset, write_offset + chunk_size).copy_(
                    peer_fc1_outputs[src_rank].slice(0, peer_read_offset, peer_read_offset + chunk_size)
                );
            }
            write_offset += chunk_size;
        }
    }

    // Cleanup
    for (int i = 0; i < ep_size; i++) {
        cudaStreamDestroy(peer_streams[i]);
        cudaEventDestroy(peer_recv_done[i]);
        cudaEventDestroy(peer_fc1_done[i]);
    }
    cudaEventDestroy(self_fc1_done_event);
    cudaStreamDestroy(comm_stream);
    cudaEventDestroy(comm_done);

    return {combined_fc1_output};
}

/**
 * AllToAll + FC1 Pipelined WITHOUT reordering.
 *
 * Same as alltoall_fc1_pipelined, but skips Step 5 (reordering).
 * Output format: [self_fc1_output, peer0_fc1_output, peer1_fc1_output, ...]
 *
 * This is designed to work with fc2_alltoall_pipelined which expects this format.
 * The reordering is deferred to the end of FC2+AllToAll, saving one memcpy.
 *
 * Returns:
 *   - combined_output: Tensor [total_tokens, ffn_hidden_size] in [self, peer0, peer1, ...] order
 *   - segment_sizes: Tensor [ep_size] containing token count for each segment
 */
std::vector<torch::Tensor> alltoall_fc1_pipelined_no_reorder(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    // Pre-computed CPU values (no .cpu() calls needed!)
    std::vector<int64_t> input_splits_vec,
    std::vector<int64_t> output_splits_vec,
    int64_t self_input_offset,
    int64_t self_input_count,
    // GPU tensors
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,
    std::vector<int64_t> peer_token_counts,
    int my_rank,
    bool serialize_peer_fc1 = true
) {
    int hidden_size = permuted_tokens.size(1);
    int ffn_hidden_size = fc1_weight.size(2);
    int ep_size = input_splits_vec.size();

    ncclComm_t nccl_comm = get_alltoall_fc1_nccl_comm();
    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();

    // Initialize persistent streams/events for pipelined mode (only creates on first call or EP size change)
    g_alltoall_fc1_ctx->init_pipelined_streams(ep_size);

    // Use persistent streams and events (no allocation per call!)
    cudaStream_t* peer_streams = g_alltoall_fc1_ctx->pipelined_peer_streams;
    cudaEvent_t* peer_recv_done = g_alltoall_fc1_ctx->pipelined_peer_recv_done;
    cudaEvent_t* peer_fc1_done = g_alltoall_fc1_ctx->pipelined_peer_fc1_done;
    cudaStream_t comm_stream = g_alltoall_fc1_ctx->pipelined_comm_stream;
    cudaEvent_t comm_done = g_alltoall_fc1_ctx->pipelined_comm_done;
    cudaEvent_t self_fc1_done_event = g_alltoall_fc1_ctx->pipelined_self_fc1_done;

    // Calculate total output tokens
    int64_t total_output_tokens = 0;
    for (int i = 0; i < ep_size; i++) {
        total_output_tokens += output_splits_vec[i];
    }

    auto options = torch::TensorOptions()
        .dtype(permuted_tokens.dtype())
        .device(permuted_tokens.device());

    torch::Tensor output_tokens = torch::empty({total_output_tokens, hidden_size}, options);

    // Build send/recv offsets
    std::vector<size_t> send_counts(ep_size), recv_counts(ep_size);
    std::vector<size_t> send_offsets(ep_size), recv_offsets(ep_size);

    size_t send_offset = 0, recv_offset = 0;
    for (int i = 0; i < ep_size; i++) {
        send_counts[i] = input_splits_vec[i] * hidden_size;
        recv_counts[i] = output_splits_vec[i] * hidden_size;
        send_offsets[i] = send_offset;
        recv_offsets[i] = recv_offset;
        send_offset += send_counts[i];
        recv_offset += recv_counts[i];
    }

    auto input_ptr = permuted_tokens.data_ptr<at::Half>();
    auto output_ptr = output_tokens.data_ptr<at::Half>();

    // NOTE: comm_stream and comm_done are now persistent (no create/destroy per call)

    // ========================================
    // TRUE OVERLAP DESIGN:
    // 1. Start Self FC1 FIRST (uses local data, no wait needed)
    // 2. Launch AllToAll (parallel with Self FC1)
    // 3. After AllToAll completes, compute Peer FC1
    //
    // Timeline (ideal):
    //   Self FC1:     [===========]
    //   AllToAll:     [==================]  <- parallel with Self FC1
    //   Peer FC1:                         [==========]
    // ========================================

    // Step 1: Launch Self FC1 FIRST (no dependency on AllToAll!)
    torch::Tensor self_copy_tokens = permuted_tokens.slice(
        0, self_input_offset, self_input_offset + self_input_count
    );

    torch::Tensor self_fc1_output;

    if (self_input_count > 0) {
        self_fc1_output = grouped_gemm_forward_nosync(
            self_copy_tokens, fc1_weight, self_tokens_per_expert,
            self_input_count, false, false
        );
        self_fc1_output = torch::gelu(self_fc1_output);
        cudaEventRecord(self_fc1_done_event, compute_stream);
    }

    // Step 2: Launch AllToAll (can run in parallel with Self FC1)
    ncclGroupStart();
    for (int peer = 0; peer < ep_size; peer++) {
        if (send_counts[peer] > 0) {
            ncclSend(reinterpret_cast<half*>(input_ptr) + send_offsets[peer],
                     send_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
        if (recv_counts[peer] > 0) {
            ncclRecv(reinterpret_cast<half*>(output_ptr) + recv_offsets[peer],
                     recv_counts[peer], ncclFloat16, peer, nccl_comm, comm_stream);
        }
    }
    ncclGroupEnd();
    cudaEventRecord(comm_done, comm_stream);

    // Step 3: Wait for AllToAll to complete, then compute Peer FC1
    // Since NCCL ncclGroup makes all sends/recvs atomic, we can only wait for all
    std::vector<torch::Tensor> peer_fc1_outputs(ep_size);

    // Wait for AllToAll to complete
    cudaStreamWaitEvent(compute_stream, comm_done, 0);

    // Now compute all peer FC1 (after AllToAll is done)
    if (serialize_peer_fc1) {
        for (int peer = 0; peer < ep_size; peer++) {
            if (peer == my_rank || recv_counts[peer] == 0) continue;

            // Convert peer rank to index in peer_token_counts/peer_tokens_per_expert_vec
            int peer_idx = (peer < my_rank) ? peer : peer - 1;

            int64_t peer_token_offset = recv_offsets[peer] / hidden_size;
            int64_t peer_token_count = peer_token_counts[peer_idx];

            torch::Tensor peer_tokens = output_tokens.slice(
                0, peer_token_offset, peer_token_offset + peer_token_count
            );

            peer_fc1_outputs[peer] = grouped_gemm_forward_nosync(
                peer_tokens, fc1_weight, peer_tokens_per_expert_vec[peer_idx],
                peer_token_count, false, false
            );
            peer_fc1_outputs[peer] = torch::gelu(peer_fc1_outputs[peer]);

            cudaEventRecord(peer_fc1_done[peer], compute_stream);
        }
    } else {
        // Non-serialized mode: each peer FC1 on separate stream
        // But still need to wait for AllToAll to complete first
        for (int peer = 0; peer < ep_size; peer++) {
            if (peer == my_rank || recv_counts[peer] == 0) continue;

            // Wait for AllToAll to complete on this peer stream
            cudaStreamWaitEvent(peer_streams[peer], comm_done, 0);

            // Convert peer rank to index in peer_token_counts/peer_tokens_per_expert_vec
            int peer_idx = (peer < my_rank) ? peer : peer - 1;

            int64_t peer_token_offset = recv_offsets[peer] / hidden_size;
            int64_t peer_token_count = peer_token_counts[peer_idx];

            torch::Tensor peer_tokens = output_tokens.slice(
                0, peer_token_offset, peer_token_offset + peer_token_count
            );

            cudaStream_t saved_stream = c10::cuda::getCurrentCUDAStream();
            at::cuda::CUDAStream peer_cuda_stream =
                at::cuda::getStreamFromExternal(peer_streams[peer], peer_tokens.device().index());
            c10::cuda::setCurrentCUDAStream(peer_cuda_stream);

            peer_fc1_outputs[peer] = grouped_gemm_forward_nosync(
                peer_tokens, fc1_weight, peer_tokens_per_expert_vec[peer_idx],
                peer_token_count, false, false
            );
            peer_fc1_outputs[peer] = torch::gelu(peer_fc1_outputs[peer]);

            cudaEventRecord(peer_fc1_done[peer], peer_streams[peer]);

            at::cuda::CUDAStream saved_cuda_stream =
                at::cuda::getStreamFromExternal(saved_stream, peer_tokens.device().index());
            c10::cuda::setCurrentCUDAStream(saved_cuda_stream);
        }
    }

    // ========================================
    // Step 4: Wait for all operations to complete
    // ========================================
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer != my_rank && recv_counts[peer] > 0) {
            cudaStreamWaitEvent(compute_stream, peer_fc1_done[peer], 0);
        }
    }
    if (self_input_count > 0) {
        cudaStreamWaitEvent(compute_stream, self_fc1_done_event, 0);
    }

    // ========================================
    // Step 5: Concatenate outputs in [self, peer0, peer1, ...] order (NO expert reorder!)
    // ========================================
    torch::Tensor combined_output = torch::empty(
        {total_output_tokens, ffn_hidden_size},
        options.dtype(fc1_weight.dtype())
    );

    // Create segment sizes tensor for FC2+AllToAll
    torch::Tensor segment_sizes = torch::empty({ep_size}, torch::kInt64);
    auto segment_ptr = segment_sizes.data_ptr<int64_t>();

    int64_t write_offset = 0;

    // Self first
    if (self_input_count > 0 && self_fc1_output.defined()) {
        int64_t self_output_count = self_fc1_output.size(0);
        combined_output.slice(0, write_offset, write_offset + self_output_count).copy_(self_fc1_output);
        segment_ptr[my_rank] = self_output_count;
        write_offset += self_output_count;
    } else {
        segment_ptr[my_rank] = 0;
    }

    // Then peers in order
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == my_rank) continue;

        if (peer_fc1_outputs[peer].defined() && peer_fc1_outputs[peer].numel() > 0) {
            int64_t peer_output_count = peer_fc1_outputs[peer].size(0);
            combined_output.slice(0, write_offset, write_offset + peer_output_count).copy_(peer_fc1_outputs[peer]);
            segment_ptr[peer] = peer_output_count;
            write_offset += peer_output_count;
        } else {
            segment_ptr[peer] = 0;
        }
    }

    // Wait for all compute operations to complete on compute_stream
    // Note: We don't need cudaDeviceSynchronize() here - just wait for our operations
    cudaStreamSynchronize(compute_stream);

    // NOTE: No cleanup needed - streams and events are persistent in g_alltoall_fc1_ctx

    return {combined_output, segment_sizes};
}

} // namespace fluid
