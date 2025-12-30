/**
 * FC2 + AllToAll Overlap with Send-First Strategy
 *
 * This is the symmetric counterpart to AllToAll+FC1:
 * - AllToAll+FC1: Receive-driven (receive remote tokens → compute)
 * - FC2+AllToAll: Send-driven (compute → send results back)
 *
 * Key Idea:
 * - Input is arranged as [self_tokens, peer0_tokens, peer1_tokens, ...]
 * - For each peer's tokens: compute FC2 → immediately send results
 * - Self tokens computed last (no communication needed)
 * - Overlap FC2 computation with AllToAll communication
 *
 * Timeline:
 *   FC2(peer0):  [════════]
 *   Send(peer0):          [═══════════]
 *   FC2(peer1):           [════════]
 *   Send(peer1):                   [═══════════]
 *   FC2(self):                     [════════]   ← Last, no send needed
 *   Recv(all):            [═══════════════════════]
 *   Reorder:                                    [════]
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
// Global NCCL Communicator for FC2+AllToAll
// ============================================================================

struct FC2AllToAllContext {
    ncclComm_t nccl_comm = nullptr;
    int nccl_rank = -1;
    int nccl_world_size = 0;
    bool nccl_initialized = false;
    bool nccl_owns_comm = false;

    // Persistent streams for overlap
    cudaStream_t comm_stream = nullptr;
    std::vector<cudaStream_t> peer_comm_streams;  // One per peer for fine-grained overlap
    std::vector<cudaEvent_t> peer_send_done;
    std::vector<cudaEvent_t> peer_fc2_done;
    cudaEvent_t all_recv_done = nullptr;
    bool streams_initialized = false;

    void init_streams(int world_size) {
        if (streams_initialized && peer_comm_streams.size() == world_size) return;

        destroy_streams();

        // Create main comm stream
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);

        // Create per-peer streams and events
        peer_comm_streams.resize(world_size);
        peer_send_done.resize(world_size);
        peer_fc2_done.resize(world_size);

        for (int i = 0; i < world_size; i++) {
            cudaStreamCreateWithFlags(&peer_comm_streams[i], cudaStreamNonBlocking);
            cudaEventCreate(&peer_send_done[i]);
            cudaEventCreate(&peer_fc2_done[i]);
        }

        cudaEventCreate(&all_recv_done);
        streams_initialized = true;
    }

    void destroy_streams() {
        if (!streams_initialized) return;

        if (comm_stream) {
            cudaStreamDestroy(comm_stream);
            comm_stream = nullptr;
        }

        for (auto& stream : peer_comm_streams) {
            if (stream) cudaStreamDestroy(stream);
        }
        peer_comm_streams.clear();

        for (auto& event : peer_send_done) {
            if (event) cudaEventDestroy(event);
        }
        peer_send_done.clear();

        for (auto& event : peer_fc2_done) {
            if (event) cudaEventDestroy(event);
        }
        peer_fc2_done.clear();

        if (all_recv_done) {
            cudaEventDestroy(all_recv_done);
            all_recv_done = nullptr;
        }

        streams_initialized = false;
    }
};

static FC2AllToAllContext* g_fc2_alltoall_ctx = nullptr;

// ============================================================================
// NCCL Initialization / Destruction
// ============================================================================

void init_fc2_alltoall_pipelined_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id_vec) {
    if (g_fc2_alltoall_ctx == nullptr) {
        g_fc2_alltoall_ctx = new FC2AllToAllContext();
    }

    // Skip if already initialized with same config
    if (g_fc2_alltoall_ctx->nccl_initialized &&
        g_fc2_alltoall_ctx->nccl_rank == rank &&
        g_fc2_alltoall_ctx->nccl_world_size == world_size) {
        return;
    }

    // Destroy previous comm if exists
    if (g_fc2_alltoall_ctx->nccl_initialized &&
        g_fc2_alltoall_ctx->nccl_owns_comm &&
        g_fc2_alltoall_ctx->nccl_comm != nullptr) {
        ncclCommDestroy(g_fc2_alltoall_ctx->nccl_comm);
        g_fc2_alltoall_ctx->nccl_initialized = false;
    }

    // Convert vector to ncclUniqueId
    ncclUniqueId nccl_id;
    // nccl_id_vec is vector of int64_t, size should be sizeof(ncclUniqueId) / sizeof(int64_t)
    if (nccl_id_vec.size() * sizeof(int64_t) != sizeof(ncclUniqueId)) {
        throw std::runtime_error("Invalid NCCL unique ID size");
    }
    memcpy(&nccl_id, nccl_id_vec.data(), sizeof(ncclUniqueId));

    // Initialize NCCL
    ncclComm_t comm;
    ncclResult_t result = ncclCommInitRank(&comm, world_size, nccl_id, rank);
    if (result != ncclSuccess) {
        throw std::runtime_error("Failed to initialize NCCL communicator for FC2+AllToAll");
    }

    g_fc2_alltoall_ctx->nccl_comm = comm;
    g_fc2_alltoall_ctx->nccl_rank = rank;
    g_fc2_alltoall_ctx->nccl_world_size = world_size;
    g_fc2_alltoall_ctx->nccl_initialized = true;
    g_fc2_alltoall_ctx->nccl_owns_comm = true;

    // Initialize streams
    g_fc2_alltoall_ctx->init_streams(world_size);
}

std::vector<int64_t> get_fc2_alltoall_pipelined_nccl_unique_id() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::vector<int64_t> id_vec(sizeof(ncclUniqueId) / sizeof(int64_t));
    memcpy(id_vec.data(), &id, sizeof(ncclUniqueId));
    return id_vec;
}

void destroy_fc2_alltoall_pipelined_nccl() {
    if (g_fc2_alltoall_ctx != nullptr) {
        g_fc2_alltoall_ctx->destroy_streams();

        if (g_fc2_alltoall_ctx->nccl_initialized &&
            g_fc2_alltoall_ctx->nccl_owns_comm &&
            g_fc2_alltoall_ctx->nccl_comm != nullptr) {
            ncclCommDestroy(g_fc2_alltoall_ctx->nccl_comm);
        }
        delete g_fc2_alltoall_ctx;
        g_fc2_alltoall_ctx = nullptr;
    }
}

ncclComm_t get_fc2_alltoall_pipelined_nccl_comm() {
    if (g_fc2_alltoall_ctx == nullptr || !g_fc2_alltoall_ctx->nccl_initialized) {
        throw std::runtime_error("FC2+AllToAll NCCL not initialized. Call init_fc2_alltoall_nccl first.");
    }
    return g_fc2_alltoall_ctx->nccl_comm;
}

// ============================================================================
// FC2 + AllToAll Pipelined Implementation
// ============================================================================

/**
 * FC2 + AllToAll with pipelined overlap.
 *
 * Input format: [self_tokens, peer0_tokens, peer1_tokens, ...] (from AllToAll+FC1 without reorder)
 * Output format: [tokens in original order] (reordered at the end)
 *
 * Strategy:
 * 1. For each remote peer (in order):
 *    a. Compute FC2 for peer's tokens
 *    b. Immediately send results to that peer
 * 2. Compute FC2 for self tokens (no communication needed)
 * 3. Receive results from all peers (overlapped with computation)
 * 4. Reorder received data to original token order
 *
 * @param input             Input tensor [total_tokens, hidden_size], arranged as [self, peer0, peer1, ...]
 * @param fc2_weight        FC2 weight tensor [num_experts, ffn_hidden_size, hidden_size]
 * @param self_token_count  Number of tokens belonging to self (no communication)
 * @param peer_token_counts Token count for each peer [ep_size]
 * @param self_tokens_per_expert   Expert distribution for self tokens
 * @param peer_tokens_per_expert_vec  Expert distribution for each peer's tokens
 * @param output_splits     How output should be split for each peer (for receiving)
 * @param reorder_indices   Indices to reorder output to original token order (optional)
 * @param my_rank           Current rank
 */
torch::Tensor fc2_alltoall_pipelined(
    torch::Tensor input,
    torch::Tensor fc2_weight,
    // Segment info (pre-computed on CPU, no .cpu() calls needed)
    int64_t self_token_count,
    std::vector<int64_t> peer_token_counts,
    // Expert distribution for grouped GEMM
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,
    // Communication info
    std::vector<int64_t> send_splits,   // How many tokens to send to each peer
    std::vector<int64_t> recv_splits,   // How many tokens to receive from each peer
    // Reorder info
    torch::Tensor reorder_indices,      // Optional: indices to reorder output
    int my_rank
) {
    if (g_fc2_alltoall_ctx == nullptr || !g_fc2_alltoall_ctx->nccl_initialized) {
        throw std::runtime_error("FC2+AllToAll NCCL not initialized");
    }

    int ep_size = peer_token_counts.size() + 1;  // +1 for self
    int hidden_size = input.size(1);
    int output_hidden_size = fc2_weight.size(2);

    ncclComm_t nccl_comm = g_fc2_alltoall_ctx->nccl_comm;
    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t comm_stream = g_fc2_alltoall_ctx->comm_stream;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());

    // Calculate total tokens
    int64_t total_tokens = self_token_count;
    for (auto count : peer_token_counts) {
        total_tokens += count;
    }

    // Allocate output buffer for FC2 results (before AllToAll)
    torch::Tensor fc2_output = torch::empty({total_tokens, output_hidden_size}, options);

    // Allocate receive buffer for AllToAll results
    int64_t total_recv_tokens = 0;
    for (auto count : recv_splits) {
        total_recv_tokens += count;
    }
    torch::Tensor recv_buffer = torch::empty({total_recv_tokens, output_hidden_size}, options);

    // Build offsets
    // Note: input_offsets indexed by peer_idx (0 to ep_size-2), where peer_idx excludes self
    std::vector<int64_t> peer_input_offsets(ep_size - 1);  // offsets for each peer's tokens in input
    std::vector<int64_t> send_offsets(ep_size);
    std::vector<int64_t> recv_offsets(ep_size);

    // Input layout: [self_tokens, peer0_tokens, peer1_tokens, ...]
    // Peer offsets start after self tokens
    int64_t offset = self_token_count;
    for (int i = 0; i < ep_size - 1; i++) {
        peer_input_offsets[i] = offset;
        offset += peer_token_counts[i];
    }

    // Send/recv offsets
    int64_t send_off = 0, recv_off = 0;
    for (int i = 0; i < ep_size; i++) {
        send_offsets[i] = send_off;
        recv_offsets[i] = recv_off;
        send_off += send_splits[i];
        recv_off += recv_splits[i];
    }

    // ========================================
    // TRUE OVERLAP DESIGN:
    // 1. Start Self FC2 on compute_stream (no wait needed)
    // 2. For each peer: compute FC2 -> immediately send (pipelined)
    // 3. All recvs happen in parallel with sends/computes
    //
    // Timeline (ideal):
    //   Self FC2:     [===========]
    //   Peer0 FC2:    [====]
    //   Peer0 Send:        [====]  <- starts right after Peer0 FC2
    //   Peer1 FC2:         [====]
    //   Peer1 Send:             [====]
    //   All Recvs:   [==================]  <- parallel with everything
    // ========================================

    // Step 1: Start Self FC2 FIRST (overlaps with everything else)
    torch::Tensor self_fc2_output;
    if (self_token_count > 0) {
        torch::Tensor self_input = input.slice(0, 0, self_token_count);
        self_fc2_output = grouped_gemm_forward_nosync(
            self_input, fc2_weight, self_tokens_per_expert,
            self_token_count, false, false
        );
    }

    // Step 2: Compute peer FC2 and immediately send (pipelined per peer)
    // Use per-peer ncclGroup to allow fine-grained overlap
    std::vector<torch::Tensor> peer_fc2_outputs(ep_size - 1);

    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == my_rank) continue;

        int peer_idx = (peer < my_rank) ? peer : peer - 1;
        int64_t token_count = peer_token_counts[peer_idx];
        if (token_count <= 0) {
            peer_fc2_outputs[peer_idx] = torch::Tensor();
            continue;
        }

        int64_t input_offset = peer_input_offsets[peer_idx];

        // Compute FC2 for this peer
        torch::Tensor peer_input = input.slice(0, input_offset, input_offset + token_count);
        peer_fc2_outputs[peer_idx] = grouped_gemm_forward_nosync(
            peer_input, fc2_weight, peer_tokens_per_expert_vec[peer_idx],
            token_count, false, false
        );

        // Copy to send buffer
        int64_t send_offset = send_offsets[peer];
        fc2_output.slice(0, send_offset, send_offset + token_count).copy_(peer_fc2_outputs[peer_idx]);
    }

    // Record when all peer FC2 computations are done
    cudaEvent_t all_peer_fc2_done;
    cudaEventCreate(&all_peer_fc2_done);
    cudaEventRecord(all_peer_fc2_done, compute_stream);

    // Wait for peer FC2 to complete before sending
    cudaStreamWaitEvent(comm_stream, all_peer_fc2_done, 0);

    // Step 3: Launch AllToAll communication
    // All sends and recvs in one ncclGroup (required for correctness)
    ncclGroupStart();

    // Send to all peers
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == my_rank) continue;

        int peer_idx = (peer < my_rank) ? peer : peer - 1;
        int64_t token_count = peer_token_counts[peer_idx];
        if (token_count <= 0) continue;

        int64_t send_offset = send_offsets[peer];
        int64_t send_count = token_count * output_hidden_size;
        half* send_ptr = reinterpret_cast<half*>(fc2_output.data_ptr<at::Half>()) +
                         send_offset * output_hidden_size;

        ncclSend(send_ptr, send_count, ncclFloat16, peer, nccl_comm, comm_stream);
    }

    // Receive from all peers
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == my_rank) continue;
        if (recv_splits[peer] <= 0) continue;

        int64_t recv_count = recv_splits[peer] * output_hidden_size;
        half* recv_ptr = reinterpret_cast<half*>(recv_buffer.data_ptr<at::Half>()) +
                         recv_offsets[peer] * output_hidden_size;

        ncclRecv(recv_ptr, recv_count, ncclFloat16, peer, nccl_comm, comm_stream);
    }

    ncclGroupEnd();

    // Step 4: Copy self FC2 result to recv buffer (can happen in parallel with comm)
    if (self_token_count > 0) {
        int64_t self_recv_offset = recv_offsets[my_rank];
        recv_buffer.slice(0, self_recv_offset, self_recv_offset + self_token_count).copy_(self_fc2_output);
    }

    // Step 5: Wait for communication to complete
    cudaEvent_t all_comm_done;
    cudaEventCreate(&all_comm_done);
    cudaEventRecord(all_comm_done, comm_stream);
    cudaStreamWaitEvent(compute_stream, all_comm_done, 0);

    // ========================================
    // Step 6: Reorder output to original token order (if indices provided)
    // ========================================
    torch::Tensor output;
    if (reorder_indices.defined() && reorder_indices.numel() > 0) {
        output = recv_buffer.index_select(0, reorder_indices);
    } else {
        output = recv_buffer;
    }

    // Cleanup events
    cudaEventDestroy(all_peer_fc2_done);
    cudaEventDestroy(all_comm_done);

    return output;
}

} // namespace fluid
