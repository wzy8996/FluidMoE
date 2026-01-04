/**
 * Stream Synchronization Primitives for True Async Communication
 *
 * This implements FlowMoE-style stream synchronization primitives:
 * - compute_stream_release: Record event on compute stream
 * - compute_stream_acquire: Wait for comm stream event on compute stream
 * - comm_stream_release: Record event on comm stream
 * - comm_stream_acquire: Wait for compute stream event on comm stream
 * - async_alltoall: Execute AllToAll on comm stream
 *
 * Key insight from FlowMoE:
 * - forward: release corresponds to backward: acquire
 * - This creates correct dependencies in autograd graph
 *
 * Example timeline:
 *   Compute:  [dX_0]  release  [dX_1]  release
 *                       ↓          ↓
 *   Comm:           acquire [A2A_0] acquire [A2A_1]
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

#include <vector>
#include <map>

namespace fluid {

// ============================================================================
// Global State for Stream Synchronization
// ============================================================================

struct StreamSyncContext {
    // CUDA streams
    cudaStream_t compute_stream = nullptr;  // Default compute stream
    cudaStream_t comm_stream = nullptr;     // Communication stream

    // NCCL communicator
    ncclComm_t nccl_comm = nullptr;
    int rank = -1;
    int world_size = -1;

    // Event pool for efficient reuse
    // Key: sync_idx, Value: (compute_event, comm_event)
    std::map<int, std::pair<cudaEvent_t, cudaEvent_t>> event_pool;

    // Initialized flag
    bool initialized = false;

    // Destructor
    ~StreamSyncContext() {
        cleanup();
    }

    void cleanup() {
        // Destroy events
        for (auto& pair : event_pool) {
            if (pair.second.first) cudaEventDestroy(pair.second.first);
            if (pair.second.second) cudaEventDestroy(pair.second.second);
        }
        event_pool.clear();

        // Destroy comm stream (compute stream is borrowed from PyTorch)
        if (comm_stream) {
            cudaStreamDestroy(comm_stream);
            comm_stream = nullptr;
        }

        // Destroy NCCL communicator
        if (nccl_comm) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }

        initialized = false;
    }
};

// Global context (per-process singleton)
static StreamSyncContext g_sync_ctx;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Get NCCL unique ID for stream sync initialization
 */
std::vector<int64_t> get_stream_sync_nccl_unique_id() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::vector<int64_t> id_vec(sizeof(ncclUniqueId) / sizeof(int64_t) + 1);
    memcpy(id_vec.data(), &id, sizeof(ncclUniqueId));
    return id_vec;
}

/**
 * Initialize stream sync context
 */
void init_stream_sync(int rank, int world_size, const std::vector<int64_t>& nccl_id_vec) {
    if (g_sync_ctx.initialized) {
        // Already initialized, cleanup first
        g_sync_ctx.cleanup();
    }

    // Store rank info
    g_sync_ctx.rank = rank;
    g_sync_ctx.world_size = world_size;

    // Get current PyTorch stream as compute stream
    g_sync_ctx.compute_stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create dedicated communication stream with high priority
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&g_sync_ctx.comm_stream, cudaStreamNonBlocking, greatest_priority);

    // Initialize NCCL communicator
    ncclUniqueId id;
    memcpy(&id, nccl_id_vec.data(), sizeof(ncclUniqueId));
    ncclCommInitRank(&g_sync_ctx.nccl_comm, world_size, id, rank);

    g_sync_ctx.initialized = true;
}

/**
 * Cleanup stream sync context
 */
void destroy_stream_sync() {
    g_sync_ctx.cleanup();
}

/**
 * Check if stream sync is initialized
 */
bool is_stream_sync_initialized() {
    return g_sync_ctx.initialized;
}

/**
 * Get or create event pair for a sync index
 */
std::pair<cudaEvent_t, cudaEvent_t>& get_or_create_events(int sync_idx) {
    auto it = g_sync_ctx.event_pool.find(sync_idx);
    if (it == g_sync_ctx.event_pool.end()) {
        cudaEvent_t compute_event, comm_event;
        cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming);
        g_sync_ctx.event_pool[sync_idx] = {compute_event, comm_event};
    }
    return g_sync_ctx.event_pool[sync_idx];
}

// ============================================================================
// Stream Synchronization Primitives
// ============================================================================

/**
 * Record event on compute stream (for comm stream to wait)
 *
 * Usage in autograd:
 *   forward: release -> backward: acquire (wait for comm)
 */
torch::Tensor compute_stream_release(torch::Tensor input, int sync_idx) {
    TORCH_CHECK(g_sync_ctx.initialized, "Stream sync not initialized. Call init_stream_sync first.");

    auto& events = get_or_create_events(sync_idx);
    cudaEvent_t& compute_event = events.first;

    // Record event on compute stream
    cudaEventRecord(compute_event, g_sync_ctx.compute_stream);

    return input;  // Pass-through
}

/**
 * Wait for comm stream event on compute stream
 *
 * Usage in autograd:
 *   forward: acquire (wait for comm) -> backward: release
 */
torch::Tensor compute_stream_acquire(torch::Tensor input, int sync_idx) {
    TORCH_CHECK(g_sync_ctx.initialized, "Stream sync not initialized. Call init_stream_sync first.");

    auto& events = get_or_create_events(sync_idx);
    cudaEvent_t& comm_event = events.second;

    // Wait for comm event on compute stream
    cudaStreamWaitEvent(g_sync_ctx.compute_stream, comm_event, 0);

    return input;  // Pass-through
}

/**
 * Record event on comm stream (for compute stream to wait)
 */
torch::Tensor comm_stream_release(torch::Tensor input, int sync_idx) {
    TORCH_CHECK(g_sync_ctx.initialized, "Stream sync not initialized. Call init_stream_sync first.");

    auto& events = get_or_create_events(sync_idx);
    cudaEvent_t& comm_event = events.second;

    // Record event on comm stream
    cudaEventRecord(comm_event, g_sync_ctx.comm_stream);

    return input;  // Pass-through
}

/**
 * Wait for compute stream event on comm stream
 */
torch::Tensor comm_stream_acquire(torch::Tensor input, int sync_idx) {
    TORCH_CHECK(g_sync_ctx.initialized, "Stream sync not initialized. Call init_stream_sync first.");

    auto& events = get_or_create_events(sync_idx);
    cudaEvent_t& compute_event = events.first;

    // Wait for compute event on comm stream
    cudaStreamWaitEvent(g_sync_ctx.comm_stream, compute_event, 0);

    return input;  // Pass-through
}

// ============================================================================
// Async AllToAll Communication
// ============================================================================

/**
 * Execute AllToAll on comm stream asynchronously
 *
 * This is the key function for true async communication:
 * - Waits for compute stream to produce data (via comm_stream_acquire)
 * - Executes NCCL AllToAll on comm stream
 * - Records completion event (via comm_stream_release)
 *
 * Args:
 *   input: Input tensor [total_tokens, hidden_size]
 *   output: Pre-allocated output tensor [total_tokens, hidden_size]
 *   send_splits: Number of elements to send to each rank
 *   recv_splits: Number of elements to receive from each rank
 *   sync_idx: Sync index for event management
 */
torch::Tensor async_alltoall(
    torch::Tensor input,
    std::vector<int64_t> send_splits,
    std::vector<int64_t> recv_splits,
    int sync_idx
) {
    TORCH_CHECK(g_sync_ctx.initialized, "Stream sync not initialized. Call init_stream_sync first.");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int world_size = g_sync_ctx.world_size;
    TORCH_CHECK(send_splits.size() == world_size, "send_splits size mismatch");
    TORCH_CHECK(recv_splits.size() == world_size, "recv_splits size mismatch");

    // Calculate total receive size
    int64_t total_recv = 0;
    for (auto s : recv_splits) total_recv += s;

    // Allocate output
    auto options = input.options();
    int hidden_size = input.size(1);
    torch::Tensor output = torch::empty({total_recv, hidden_size}, options);

    // Get events
    auto& events = get_or_create_events(sync_idx);
    cudaEvent_t& compute_event = events.first;
    cudaEvent_t& comm_event = events.second;

    // Wait for compute stream to finish producing data
    cudaStreamWaitEvent(g_sync_ctx.comm_stream, compute_event, 0);

    // Determine NCCL datatype
    ncclDataType_t dtype;
    if (input.scalar_type() == torch::kFloat16) {
        dtype = ncclFloat16;
    } else if (input.scalar_type() == torch::kBFloat16) {
        dtype = ncclBfloat16;
    } else if (input.scalar_type() == torch::kFloat32) {
        dtype = ncclFloat32;
    } else {
        TORCH_CHECK(false, "Unsupported dtype for async_alltoall");
    }

    // Execute AllToAll using ncclSend/ncclRecv pairs
    ncclGroupStart();

    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    for (int r = 0; r < world_size; r++) {
        int64_t send_count = send_splits[r] * hidden_size;
        int64_t recv_count = recv_splits[r] * hidden_size;

        if (send_count > 0) {
            ncclSend(
                static_cast<const char*>(input.data_ptr()) + send_offset * input.element_size(),
                send_count,
                dtype,
                r,
                g_sync_ctx.nccl_comm,
                g_sync_ctx.comm_stream
            );
        }

        if (recv_count > 0) {
            ncclRecv(
                static_cast<char*>(output.data_ptr()) + recv_offset * output.element_size(),
                recv_count,
                dtype,
                r,
                g_sync_ctx.nccl_comm,
                g_sync_ctx.comm_stream
            );
        }

        send_offset += send_splits[r];
        recv_offset += recv_splits[r];
    }

    ncclGroupEnd();

    // Record completion event on comm stream
    cudaEventRecord(comm_event, g_sync_ctx.comm_stream);

    return output;
}

/**
 * Synchronize comm stream back to compute stream
 * Call this at the end of a backward pass to ensure all communication is complete
 */
void sync_comm_to_compute_sync() {
    if (!g_sync_ctx.initialized) return;
    cudaStreamSynchronize(g_sync_ctx.comm_stream);
}

/**
 * Non-blocking sync: make compute stream wait for comm stream
 */
void wait_comm_on_compute() {
    if (!g_sync_ctx.initialized) return;

    // Record event on comm stream
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, g_sync_ctx.comm_stream);

    // Wait on compute stream
    cudaStreamWaitEvent(g_sync_ctx.compute_stream, event, 0);

    cudaEventDestroy(event);
}

/**
 * Get comm stream handle (for external use)
 */
int64_t get_comm_stream_handle() {
    if (!g_sync_ctx.initialized) return 0;
    return reinterpret_cast<int64_t>(g_sync_ctx.comm_stream);
}

/**
 * Get compute stream handle (for external use)
 */
int64_t get_compute_stream_handle() {
    if (!g_sync_ctx.initialized) return 0;
    return reinterpret_cast<int64_t>(g_sync_ctx.compute_stream);
}

}  // namespace fluid
