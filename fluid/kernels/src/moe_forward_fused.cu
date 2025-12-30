/**
 * Fully Fused MoE Forward Pass
 *
 * This file implements a fully fused MoE forward pass that combines:
 * AllToAll + FC1 + GELU + FC2 + AllToAll
 *
 * Key optimizations:
 * 1. GEMM outputs directly to destination buffer (no intermediate tensor + copy!)
 * 2. True overlap: Self FC1/FC2 runs in parallel with AllToAll communication
 * 3. Symmetric design: FC1 output format [self, peer0, peer1, ...] goes directly to FC2
 *    - NO reorder between FC1 and FC2
 *    - Only final reorder after FC2+AllToAll
 *
 * Data flow:
 * Input: [total_tokens, hidden_size] - permuted tokens ordered by destination rank
 *
 * Stage 1: AllToAll (dispatch) + FC1 + GELU
 *   - Self tokens: no communication, direct FC1
 *   - Peer tokens: receive via AllToAll, then FC1
 *   - Output: [local_tokens, ffn_hidden_size] in format [self, peer0, peer1, ...]
 *   - ** GEMM writes directly to combined_output buffer, no copy! **
 *
 * Stage 2: FC2 + AllToAll (combine)
 *   - Input format: [self, peer0, peer1, ...] (same as FC1 output)
 *   - Compute FC2 for each source
 *   - Send results back to original ranks
 *   - Output: [total_tokens, hidden_size] reordered to original positions
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/arch.h>

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace fluid {

// ============================================================================
// CUTLASS GEMM Type Definitions
// ============================================================================

using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
>;

// Tile sizes for GroupedGEMM
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 32;

// A: RowMajor, B: ColumnMajor (input @ weight.T)
// For FC layers: input [M, K] @ weight [N, K]^T = output [M, N]
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

// ============================================================================
// True CUTLASS GroupedGEMM - Single Kernel for All Experts!
// ============================================================================

// Define GroupedGEMM kernel type
// A: RowMajor [M, K], B: ColumnMajor [K, N] (= RowMajor [N, K] transposed), C: RowMajor [M, N]
using GroupedGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,   // A
    ElementInput, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8, // B (weight.T)
    ElementOutput, cutlass::layout::RowMajor,                                        // C
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TILE_M, TILE_N, TILE_K>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,  // stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GroupedGemmKernel>;

// ============================================================================
// Persistent Context for GroupedGEMM (avoids repeated allocations)
// ============================================================================

struct GroupedGemmContext {
    static constexpr int MAX_EXPERTS = 32;

    // Device memory for problem descriptors
    cutlass::gemm::GemmCoord* d_problem_sizes = nullptr;
    ElementInput** d_ptr_A = nullptr;
    ElementInput** d_ptr_B = nullptr;
    ElementOutput** d_ptr_C = nullptr;
    ElementOutput** d_ptr_D = nullptr;
    int64_t* d_lda = nullptr;
    int64_t* d_ldb = nullptr;
    int64_t* d_ldc = nullptr;
    int64_t* d_ldd = nullptr;

    // Host pinned memory for fast async transfer
    cutlass::gemm::GemmCoord* h_problem_sizes = nullptr;
    ElementInput** h_ptr_A = nullptr;
    ElementInput** h_ptr_B = nullptr;
    ElementOutput** h_ptr_C = nullptr;
    int64_t* h_lda = nullptr;
    int64_t* h_ldb = nullptr;
    int64_t* h_ldc = nullptr;

    // Workspace for GroupedGEMM
    void* workspace = nullptr;
    size_t workspace_size = 0;

    int sm_count = 0;
    bool initialized = false;

    void init() {
        if (initialized) return;

        // Get SM count
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        // Allocate device memory
        cudaMalloc(&d_problem_sizes, MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord));
        cudaMalloc(&d_ptr_A, MAX_EXPERTS * sizeof(ElementInput*));
        cudaMalloc(&d_ptr_B, MAX_EXPERTS * sizeof(ElementInput*));
        cudaMalloc(&d_ptr_C, MAX_EXPERTS * sizeof(ElementOutput*));
        cudaMalloc(&d_ptr_D, MAX_EXPERTS * sizeof(ElementOutput*));
        cudaMalloc(&d_lda, MAX_EXPERTS * sizeof(int64_t));
        cudaMalloc(&d_ldb, MAX_EXPERTS * sizeof(int64_t));
        cudaMalloc(&d_ldc, MAX_EXPERTS * sizeof(int64_t));
        cudaMalloc(&d_ldd, MAX_EXPERTS * sizeof(int64_t));

        // Allocate pinned host memory
        cudaMallocHost(&h_problem_sizes, MAX_EXPERTS * sizeof(cutlass::gemm::GemmCoord));
        cudaMallocHost(&h_ptr_A, MAX_EXPERTS * sizeof(ElementInput*));
        cudaMallocHost(&h_ptr_B, MAX_EXPERTS * sizeof(ElementInput*));
        cudaMallocHost(&h_ptr_C, MAX_EXPERTS * sizeof(ElementOutput*));
        cudaMallocHost(&h_lda, MAX_EXPERTS * sizeof(int64_t));
        cudaMallocHost(&h_ldb, MAX_EXPERTS * sizeof(int64_t));
        cudaMallocHost(&h_ldc, MAX_EXPERTS * sizeof(int64_t));

        // Initial workspace allocation
        workspace_size = 1024 * 1024;  // 1MB initial
        cudaMalloc(&workspace, workspace_size);

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;

        cudaFree(d_problem_sizes);
        cudaFree(d_ptr_A);
        cudaFree(d_ptr_B);
        cudaFree(d_ptr_C);
        cudaFree(d_ptr_D);
        cudaFree(d_lda);
        cudaFree(d_ldb);
        cudaFree(d_ldc);
        cudaFree(d_ldd);

        cudaFreeHost(h_problem_sizes);
        cudaFreeHost(h_ptr_A);
        cudaFreeHost(h_ptr_B);
        cudaFreeHost(h_ptr_C);
        cudaFreeHost(h_lda);
        cudaFreeHost(h_ldb);
        cudaFreeHost(h_ldc);

        if (workspace) cudaFree(workspace);

        initialized = false;
    }

    ~GroupedGemmContext() { destroy(); }
};

static GroupedGemmContext g_grouped_gemm_ctx;

// ============================================================================
// Activation Function Kernels
// ============================================================================
// Supported: 0=GELU, 1=SiLU(Swish), 2=ReLU, 3=None(identity)

enum class ActivationType : int {
    GELU = 0,
    SILU = 1,  // Also known as Swish
    RELU = 2,
    NONE = 3   // Identity (no activation)
};

__global__ void activation_kernel(half* data, int64_t n, int act_type) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        float result;

        switch (act_type) {
            case 0:  // GELU approximation
                {
                    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                    result = x * cdf;
                }
                break;
            case 1:  // SiLU (Swish): x * sigmoid(x)
                result = x / (1.0f + expf(-x));
                break;
            case 2:  // ReLU
                result = fmaxf(0.0f, x);
                break;
            default:  // None (identity)
                result = x;
                break;
        }

        data[idx] = __float2half(result);
    }
}

void launch_activation(half* data, int64_t n, int act_type, cudaStream_t stream) {
    if (act_type == 3) return;  // Skip for identity
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    activation_kernel<<<num_blocks, block_size, 0, stream>>>(data, n, act_type);
}

// Legacy GELU interface for compatibility
void launch_gelu(half* data, int64_t n, cudaStream_t stream) {
    launch_activation(data, n, 0, stream);
}

// ============================================================================
// Single GEMM Launch - Direct to Specified Output Location (FALLBACK)
// ============================================================================

static cudaError_t launch_gemm_single(
    const cutlass::half_t* input,   // [M, K] row-major
    const cutlass::half_t* weight,  // [N, K] row-major (will be transposed)
    cutlass::half_t* output,        // [M, N] row-major (pre-allocated!)
    int M, int N, int K,
    cudaStream_t stream
) {
    if (M == 0) return cudaSuccess;

    typename GemmNT::Arguments args(
        {M, N, K},
        {input, K},    // A: [M, K], lda=K
        {weight, K},   // B: [N, K], ldb=K (col-major view)
        {output, N},   // C: [M, N], ldc=N
        {output, N},   // D: same as C
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)}
    );

    GemmNT gemm_op;
    cutlass::Status status = gemm_op(args, nullptr, stream);

    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ============================================================================
// TRUE GroupedGEMM - Single Kernel for All Experts!
//
// This is the KEY optimization! Instead of launching N separate GEMM kernels
// (one per expert), we use CUTLASS GroupedGEMM to launch a single persistent
// kernel that processes all experts. This eliminates kernel launch overhead!
//
// The GroupedGEMM kernel uses "persistent" threads that iterate over multiple
// problems, achieving much better GPU utilization than separate kernel launches.
// ============================================================================

static void grouped_gemm_true(
    const cutlass::half_t* input,           // [total_input_tokens, K]
    const cutlass::half_t* weight,          // [num_experts, N, K]
    cutlass::half_t* output,                // Pre-allocated output buffer
    const int* h_tokens_per_expert,         // Host array!
    int num_experts,
    int N,                                  // Output dimension (per expert)
    int K,                                  // Input dimension
    cudaStream_t stream
) {
    auto& ctx = g_grouped_gemm_ctx;
    if (!ctx.initialized) {
        ctx.init();
    }

    // Count valid problems (experts with tokens > 0)
    int num_valid_problems = 0;
    int total_tokens = 0;

    // First pass: count valid problems and total tokens
    for (int e = 0; e < num_experts; e++) {
        if (h_tokens_per_expert[e] > 0) {
            num_valid_problems++;
            total_tokens += h_tokens_per_expert[e];
        }
    }

    if (num_valid_problems == 0) return;

    // Populate pinned host memory with problem descriptors
    int input_offset = 0;
    int output_offset = 0;
    int valid_idx = 0;

    for (int e = 0; e < num_experts; e++) {
        int m = h_tokens_per_expert[e];
        if (m == 0) continue;

        // Problem size: M x N x K
        ctx.h_problem_sizes[valid_idx] = cutlass::gemm::GemmCoord(m, N, K);

        // Input for this expert: [m, K] starting at input_offset
        ctx.h_ptr_A[valid_idx] = const_cast<ElementInput*>(input + input_offset * K);
        // Weight for this expert: [N, K] (stored contiguously per expert)
        ctx.h_ptr_B[valid_idx] = const_cast<ElementInput*>(weight + e * N * K);
        // Output for this expert: [m, N] starting at output_offset
        ctx.h_ptr_C[valid_idx] = output + output_offset * N;

        // Leading dimensions
        ctx.h_lda[valid_idx] = K;   // A is [m, K], row-major, lda = K
        ctx.h_ldb[valid_idx] = K;   // B is [N, K], col-major view, ldb = K
        ctx.h_ldc[valid_idx] = N;   // C is [m, N], row-major, ldc = N

        input_offset += m;
        output_offset += m;
        valid_idx++;
    }

    // Async copy to device
    cudaMemcpyAsync(ctx.d_problem_sizes, ctx.h_problem_sizes,
                    num_valid_problems * sizeof(cutlass::gemm::GemmCoord),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ptr_A, ctx.h_ptr_A,
                    num_valid_problems * sizeof(ElementInput*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ptr_B, ctx.h_ptr_B,
                    num_valid_problems * sizeof(ElementInput*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ptr_C, ctx.h_ptr_C,
                    num_valid_problems * sizeof(ElementOutput*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ptr_D, ctx.h_ptr_C,  // D = C for in-place
                    num_valid_problems * sizeof(ElementOutput*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_lda, ctx.h_lda,
                    num_valid_problems * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ldb, ctx.h_ldb,
                    num_valid_problems * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ldc, ctx.h_ldc,
                    num_valid_problems * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_ldd, ctx.h_ldc,  // ldd = ldc
                    num_valid_problems * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    // Calculate total tiles for workspace sizing
    int total_tiles = 0;
    for (int i = 0; i < num_valid_problems; i++) {
        int m = ctx.h_problem_sizes[i].m();
        int n = ctx.h_problem_sizes[i].n();
        int tiles_m = (m + TILE_M - 1) / TILE_M;
        int tiles_n = (n + TILE_N - 1) / TILE_N;
        total_tiles += tiles_m * tiles_n;
    }

    // Create GroupedGEMM arguments
    typename GemmGrouped::Arguments args(
        ctx.d_problem_sizes,
        num_valid_problems,
        ctx.sm_count,  // threadblock_count
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},  // alpha, beta
        ctx.d_ptr_A,
        ctx.d_ptr_B,
        ctx.d_ptr_C,
        ctx.d_ptr_D,
        ctx.d_lda,
        ctx.d_ldb,
        ctx.d_ldc,
        ctx.d_ldd,
        ctx.h_problem_sizes  // host_problem_sizes for tile count
    );

    // Get workspace size
    size_t needed_workspace = GemmGrouped::get_workspace_size(args);
    if (needed_workspace > ctx.workspace_size) {
        if (ctx.workspace) cudaFree(ctx.workspace);
        ctx.workspace_size = needed_workspace * 2;  // Over-allocate for future
        cudaMalloc(&ctx.workspace, ctx.workspace_size);
    }

    // Initialize and run
    GemmGrouped gemm_grouped;
    cutlass::Status status = gemm_grouped.initialize(args, ctx.workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GroupedGEMM initialize failed: " << int(status) << std::endl;
        return;
    }

    status = gemm_grouped.run(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GroupedGEMM run failed: " << int(status) << std::endl;
    }
}

// ============================================================================
// Loop-based Grouped GEMM (fallback, for comparison/debugging)
// ============================================================================

static void grouped_gemm_loop(
    const cutlass::half_t* input,           // [total_input_tokens, K]
    const cutlass::half_t* weight,          // [num_experts, N, K]
    cutlass::half_t* output,                // Pre-allocated output buffer
    const int* h_tokens_per_expert,         // Host array!
    int num_experts,
    int N,                                  // Output dimension (per expert)
    int K,                                  // Input dimension
    cudaStream_t stream
) {
    int input_offset = 0;
    int output_offset = 0;

    for (int e = 0; e < num_experts; e++) {
        int m = h_tokens_per_expert[e];
        if (m == 0) continue;

        launch_gemm_single(
            input + input_offset * K,
            weight + e * N * K,
            output + output_offset * N,
            m, N, K,
            stream
        );

        input_offset += m;
        output_offset += m;
    }
}

// ============================================================================
// Wrapper that selects best GEMM strategy based on problem size
// For small problems, loop-based is faster (less setup overhead)
// For large problems, true GroupedGEMM is faster (better GPU utilization)
// ============================================================================

static void grouped_gemm_to_buffer(
    const cutlass::half_t* input,           // [total_input_tokens, K]
    const cutlass::half_t* weight,          // [num_experts, N, K]
    cutlass::half_t* output,                // Pre-allocated output buffer
    const int* h_tokens_per_expert,         // Host array!
    int num_experts,
    int N,                                  // Output dimension (per expert)
    int K,                                  // Input dimension
    cudaStream_t stream
) {
    // Calculate total tokens
    int total_tokens = 0;
    for (int e = 0; e < num_experts; e++) {
        total_tokens += h_tokens_per_expert[e];
    }

    // Heuristic: use loop-based GEMM for better overlap
    // GroupedGEMM has ~4ms setup overhead that blocks the CPU and reduces overlap
    // Loop-based GEMM launches kernels faster, enabling better compute-comm overlap
    // For fused MoE with overlap, loop mode is generally better
    bool use_loop = true;  // Default to loop for better overlap

    // Check environment variable for forcing one mode
    static int force_mode = -1;
    if (force_mode == -1) {
        const char* env = getenv("FLUID_GEMM_MODE");
        if (env) {
            if (strcmp(env, "loop") == 0) force_mode = 0;
            else if (strcmp(env, "grouped") == 0) force_mode = 1;
            else force_mode = 2;  // auto
        } else {
            force_mode = 2;  // auto
        }
    }

    if (force_mode == 0) use_loop = true;
    else if (force_mode == 1) use_loop = false;

    if (use_loop) {
        grouped_gemm_loop(input, weight, output, h_tokens_per_expert, num_experts, N, K, stream);
    } else {
        grouped_gemm_true(input, weight, output, h_tokens_per_expert, num_experts, N, K, stream);
    }
}

// ============================================================================
// MoE Forward Fused Context
// Pre-allocates all streams and events to minimize per-call overhead
//
// Design (per user request):
// - ONE compute stream for all computation
// - Multiple comm streams (one per peer) for parallel communication
// - All streams/events created once at init, destroyed at cleanup
// ============================================================================

struct MoEForwardFusedContext {
    static constexpr int MAX_EP_SIZE = 16;

    // NCCL
    ncclComm_t nccl_comm = nullptr;
    int rank = -1;
    int ep_size = 0;

    // ONE compute stream for ALL computation
    cudaStream_t compute_stream = nullptr;

    // Multiple comm streams - one per peer for parallel communication
    cudaStream_t comm_streams[MAX_EP_SIZE] = {nullptr};

    // Events for synchronization (disable timing for lower overhead)
    cudaEvent_t self_fc1_done = nullptr;
    cudaEvent_t peer_recv_done[MAX_EP_SIZE] = {nullptr};  // Per-peer recv complete
    cudaEvent_t all_comm_done = nullptr;  // All communication complete

    bool initialized = false;

    void init(int rank_, int ep_size_, ncclComm_t comm) {
        if (initialized) return;

        this->rank = rank_;
        this->ep_size = ep_size_;
        this->nccl_comm = comm;

        // CRITICAL: Use cudaStreamNonBlocking to prevent implicit sync with default stream!
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

        // Create one comm stream per peer
        for (int i = 0; i < ep_size_; i++) {
            if (i != rank_) {
                cudaStreamCreateWithFlags(&comm_streams[i], cudaStreamNonBlocking);
            }
        }

        // Create events with disable timing for lower overhead
        cudaEventCreateWithFlags(&self_fc1_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&all_comm_done, cudaEventDisableTiming);

        for (int i = 0; i < ep_size_; i++) {
            if (i != rank_) {
                cudaEventCreateWithFlags(&peer_recv_done[i], cudaEventDisableTiming);
            }
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized) return;

        if (compute_stream) cudaStreamDestroy(compute_stream);

        for (int i = 0; i < ep_size; i++) {
            if (comm_streams[i]) cudaStreamDestroy(comm_streams[i]);
        }

        if (self_fc1_done) cudaEventDestroy(self_fc1_done);
        if (all_comm_done) cudaEventDestroy(all_comm_done);

        for (int i = 0; i < ep_size; i++) {
            if (peer_recv_done[i]) cudaEventDestroy(peer_recv_done[i]);
        }

        compute_stream = nullptr;
        for (int i = 0; i < MAX_EP_SIZE; i++) {
            comm_streams[i] = nullptr;
            peer_recv_done[i] = nullptr;
        }
        self_fc1_done = nullptr;
        all_comm_done = nullptr;

        initialized = false;
    }

    ~MoEForwardFusedContext() {
        destroy();
    }
};

// Global context instance
static MoEForwardFusedContext g_moe_fused_ctx;

// ============================================================================
// NCCL Initialization Functions
// ============================================================================

std::vector<int64_t> get_moe_fused_nccl_unique_id() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::vector<int64_t> result(16);
    memcpy(result.data(), id.internal, 128);
    return result;
}

void init_moe_fused_nccl(int rank, int world_size, std::vector<int64_t> nccl_id_vec) {
    ncclUniqueId id;
    memcpy(id.internal, nccl_id_vec.data(), 128);

    ncclComm_t comm;
    ncclCommInitRank(&comm, world_size, id, rank);

    g_moe_fused_ctx.init(rank, world_size, comm);
}

void destroy_moe_fused_nccl() {
    if (g_moe_fused_ctx.nccl_comm) {
        ncclCommDestroy(g_moe_fused_ctx.nccl_comm);
        g_moe_fused_ctx.nccl_comm = nullptr;
    }
    g_moe_fused_ctx.destroy();
}

// ============================================================================
// Fused AllToAll + FC1 + Activation with True Pipelined Overlap
//
// Key optimizations:
// 1. ALL tokens_per_expert arrays are passed as HOST vectors (pre-computed)
// 2. NO cudaMemcpy D2H, NO cudaStreamSynchronize during critical path!
// 3. Per-peer communication: each peer has independent send/recv
// 4. TRUE PIPELINE: "先到先算" - process each peer as soon as its data arrives
//
// Timeline (with 3 peers):
//   Peer0 Recv:  [=====]
//   Peer1 Recv:  [=======]
//   Peer2 Recv:  [=========]
//   Self FC1:    [========]           <- Overlaps with all recv!
//   Peer0 FC1:         [====]         <- Starts when peer0 data arrives
//   Peer1 FC1:              [====]    <- Starts when peer1 data arrives
//   Peer2 FC1:                   [====]
//
// Parameters:
//   h_self_tokens_per_expert: Host vector, pre-computed before calling
//   h_peer_tokens_per_expert_all: Host vectors, pre-computed before calling
// ============================================================================

// Returns: (fc1_output, segment_sizes, dispatched_input, fc1_pre_activation)
// - fc1_output: [total_recv_tokens, ffn_hidden_size] - after activation
// - segment_sizes: [ep_size] - tokens per source rank
// - dispatched_input: [total_recv_tokens, hidden_size] - FC1 input (for dW1)
// - fc1_pre_activation: [total_recv_tokens, ffn_hidden_size] - before activation (for activation backward)
// Note: probs multiplication happens in Python at unpermute stage (standard Megatron behavior)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_alltoall_fc1_fused(
    torch::Tensor permuted_tokens,            // [total_tokens, hidden_size]
    torch::Tensor fc1_weight,                 // [num_experts, hidden_size, ffn_hidden_size]
    int64_t self_input_offset,                // Offset of self tokens in input
    int64_t self_input_count,                 // Number of self tokens (from self to self)
    std::vector<int64_t> send_splits,         // Tokens to send to each rank [ep_size]
    std::vector<int64_t> recv_splits,         // Tokens to receive from each rank [ep_size]
    std::vector<int64_t> peer_token_counts,   // Total tokens from each peer [ep_size-1]
    // PRE-COMPUTED HOST ARRAYS (the key to zero-sync!)
    std::vector<int> h_self_tokens_per_expert,               // [num_experts] on host
    std::vector<std::vector<int>> h_peer_tokens_per_expert_all,  // [ep_size-1][num_experts] on host
    int activation_type = 0                   // 0=GELU, 1=SiLU, 2=ReLU, 3=None
) {
    auto& ctx = g_moe_fused_ctx;
    TORCH_CHECK(ctx.initialized, "MoE fused context not initialized");

    const int rank = ctx.rank;
    const int ep_size = ctx.ep_size;
    const int num_experts = fc1_weight.size(0);
    const int hidden_size = fc1_weight.size(1);
    const int ffn_hidden_size = fc1_weight.size(2);

    cudaStream_t compute_stream = ctx.compute_stream;

    // Calculate total output tokens and offsets (CPU work, no GPU sync)
    int64_t total_recv_tokens = 0;
    for (int i = 0; i < ep_size; i++) {
        total_recv_tokens += recv_splits[i];
    }

    std::vector<int64_t> send_offsets(ep_size);
    int64_t send_off = 0;
    for (int i = 0; i < ep_size; i++) {
        send_offsets[i] = send_off;
        send_off += send_splits[i];
    }

    // Allocate output buffers BEFORE launching any GPU work
    torch::Tensor fc1_output = torch::empty(
        {total_recv_tokens, ffn_hidden_size},
        permuted_tokens.options()
    );

    // Allocate fc1_pre_activation for backward (same size as fc1_output)
    torch::Tensor fc1_pre_activation = torch::empty(
        {total_recv_tokens, ffn_hidden_size},
        permuted_tokens.options()
    );

    // Allocate dispatched_input for backward (FC1 input after AllToAll)
    torch::Tensor dispatched_input = torch::empty(
        {total_recv_tokens, hidden_size},
        permuted_tokens.options()
    );

    // Allocate separate recv buffer for each peer (enables per-peer tracking)
    std::vector<int64_t> peer_recv_offsets(ep_size);  // Where each peer's data starts

    int64_t write_offset = self_input_count;  // Self tokens at beginning
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) {
            peer_recv_offsets[peer] = 0;  // Self at beginning
            continue;
        }
        int peer_idx = (peer < rank) ? peer : peer - 1;
        int64_t peer_count = peer_token_counts[peer_idx];

        peer_recv_offsets[peer] = write_offset;
        write_offset += peer_count;
    }

    torch::Tensor segment_sizes = torch::empty({ep_size}, torch::kInt64);
    auto segment_ptr = segment_sizes.data_ptr<int64_t>();

    // ========================================================================
    // STEP 1: Launch per-peer communication (each peer independent!)
    // Receive directly into dispatched_input buffer
    // ========================================================================

    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) continue;

        cudaStream_t peer_comm_stream = ctx.comm_streams[peer];
        int peer_idx = (peer < rank) ? peer : peer - 1;
        int64_t peer_count = peer_token_counts[peer_idx];

        ncclGroupStart();

        // Send tokens to this peer
        if (send_splits[peer] > 0) {
            const half* send_ptr = reinterpret_cast<const half*>(permuted_tokens.data_ptr<at::Half>())
                                   + send_offsets[peer] * hidden_size;
            ncclSend(send_ptr, send_splits[peer] * hidden_size, ncclFloat16,
                     peer, ctx.nccl_comm, peer_comm_stream);
        }

        // Receive tokens directly into dispatched_input
        if (recv_splits[peer] > 0) {
            half* recv_ptr = reinterpret_cast<half*>(dispatched_input.data_ptr<at::Half>())
                             + peer_recv_offsets[peer] * hidden_size;
            ncclRecv(recv_ptr, recv_splits[peer] * hidden_size, ncclFloat16,
                     peer, ctx.nccl_comm, peer_comm_stream);
        }

        ncclGroupEnd();

        cudaEventRecord(ctx.peer_recv_done[peer], peer_comm_stream);
    }

    // ========================================================================
    // STEP 2: Launch Self FC1 on compute_stream (runs in PARALLEL with all recv!)
    // ========================================================================

    segment_ptr[rank] = self_input_count;

    if (self_input_count > 0) {
        const cutlass::half_t* self_input =
            reinterpret_cast<const cutlass::half_t*>(permuted_tokens.data_ptr<at::Half>())
            + self_input_offset * hidden_size;

        // Copy self tokens to dispatched_input for backward
        cudaMemcpyAsync(
            dispatched_input.data_ptr<at::Half>(),
            permuted_tokens.data_ptr<at::Half>() + self_input_offset * hidden_size,
            self_input_count * hidden_size * sizeof(at::Half),
            cudaMemcpyDeviceToDevice,
            compute_stream
        );

        cutlass::half_t* self_fc1_out =
            reinterpret_cast<cutlass::half_t*>(fc1_output.data_ptr<at::Half>());

        // FC1: dispatched_input -> fc1_output
        grouped_gemm_to_buffer(
            self_input,
            reinterpret_cast<const cutlass::half_t*>(fc1_weight.data_ptr<at::Half>()),
            self_fc1_out,
            h_self_tokens_per_expert.data(),
            num_experts,
            ffn_hidden_size,
            hidden_size,
            compute_stream
        );

        // Copy fc1_output to fc1_pre_activation (before activation)
        cudaMemcpyAsync(
            fc1_pre_activation.data_ptr<at::Half>(),
            fc1_output.data_ptr<at::Half>(),
            self_input_count * ffn_hidden_size * sizeof(at::Half),
            cudaMemcpyDeviceToDevice,
            compute_stream
        );

        // Apply activation in-place on fc1_output
        launch_activation(
            reinterpret_cast<half*>(self_fc1_out),
            self_input_count * ffn_hidden_size,
            activation_type,
            compute_stream
        );
    }

    cudaEventRecord(ctx.self_fc1_done, compute_stream);

    // ========================================================================
    // STEP 3: Process each peer as soon as its data arrives (TRUE PIPELINE!)
    // Data is already in dispatched_input from AllToAll
    // ========================================================================

    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) continue;

        int peer_idx = (peer < rank) ? peer : peer - 1;
        int64_t peer_count = peer_token_counts[peer_idx];

        segment_ptr[peer] = peer_count;

        if (peer_count > 0) {
            cudaStreamWaitEvent(compute_stream, ctx.peer_recv_done[peer], 0);

            const std::vector<int>& h_peer_tokens = h_peer_tokens_per_expert_all[peer_idx];

            // Input from dispatched_input (received via AllToAll)
            const cutlass::half_t* peer_input =
                reinterpret_cast<const cutlass::half_t*>(dispatched_input.data_ptr<at::Half>())
                + peer_recv_offsets[peer] * hidden_size;

            cutlass::half_t* peer_fc1_out =
                reinterpret_cast<cutlass::half_t*>(fc1_output.data_ptr<at::Half>())
                + peer_recv_offsets[peer] * ffn_hidden_size;

            cutlass::half_t* peer_fc1_pre =
                reinterpret_cast<cutlass::half_t*>(fc1_pre_activation.data_ptr<at::Half>())
                + peer_recv_offsets[peer] * ffn_hidden_size;

            // FC1: dispatched_input -> fc1_output
            grouped_gemm_to_buffer(
                peer_input,
                reinterpret_cast<const cutlass::half_t*>(fc1_weight.data_ptr<at::Half>()),
                peer_fc1_out,
                h_peer_tokens.data(),
                num_experts,
                ffn_hidden_size,
                hidden_size,
                compute_stream
            );

            // Copy to fc1_pre_activation before activation
            cudaMemcpyAsync(
                peer_fc1_pre,
                peer_fc1_out,
                peer_count * ffn_hidden_size * sizeof(at::Half),
                cudaMemcpyDeviceToDevice,
                compute_stream
            );

            // Apply activation in-place
            launch_activation(
                reinterpret_cast<half*>(peer_fc1_out),
                peer_count * ffn_hidden_size,
                activation_type,
                compute_stream
            );
        }
    }

    cudaStreamSynchronize(compute_stream);

    return {fc1_output, segment_sizes, dispatched_input, fc1_pre_activation};
}

// ============================================================================
// Fused FC2 + AllToAll with TRUE Overlap
//
// Input format: [self_tokens, peer0_tokens, peer1_tokens, ...] from FC1
// Uses segment_sizes to know how to split input for AllToAll
//
// TRUE OVERLAP DESIGN (reversed from FC1 stage!):
// 1. First compute FC2 for PEER tokens (remote tokens that need to be sent)
// 2. Start sending peer tokens immediately after each peer's FC2 completes
// 3. While sending, compute FC2 for SELF tokens (overlaps with AllToAll!)
// 4. Copy self segment locally (no communication needed)
// 5. Wait for all communication to complete
//
// Timeline (with 2 peers):
//   Peer0 FC2:     [====]
//   Peer0 Send:         [=======]
//   Peer1 FC2:          [====]
//   Peer1 Send:              [=======]
//   Self FC2:                [====]      <- Overlaps with Peer sends!
//   Self copy:                    [=]
//   Recv:          [===================] <- Runs in parallel throughout
//
// Key insight: FC2(self) overlaps with AllToAll(peer), NOT the other way around!
// ============================================================================

torch::Tensor moe_fc2_alltoall_fused(
    torch::Tensor fc1_output,                 // [total_local_tokens, ffn_hidden_size]
    torch::Tensor fc2_weight,                 // [num_experts, ffn_hidden_size, hidden_size]
    torch::Tensor segment_sizes,              // [ep_size] - tokens per source rank
    std::vector<int64_t> original_send_splits, // Original send splits (for reverse AllToAll)
    // PRE-COMPUTED HOST ARRAYS for per-segment FC2
    std::vector<int> h_self_tokens_per_expert,                   // [num_experts] on host for self
    std::vector<std::vector<int>> h_peer_tokens_per_expert_all   // [ep_size-1][num_experts] on host for peers
) {
    auto& ctx = g_moe_fused_ctx;
    TORCH_CHECK(ctx.initialized, "MoE fused context not initialized");

    const int rank = ctx.rank;
    const int ep_size = ctx.ep_size;
    const int num_experts = fc2_weight.size(0);
    const int ffn_hidden_size = fc2_weight.size(1);
    const int hidden_size = fc2_weight.size(2);
    const int64_t total_local_tokens = fc1_output.size(0);

    cudaStream_t compute_stream = ctx.compute_stream;

    // Get segment sizes on host
    auto seg_ptr = segment_sizes.data_ptr<int64_t>();

    // Calculate total output tokens and offsets for receiving
    int64_t total_output_tokens = 0;
    std::vector<int64_t> recv_offsets(ep_size);
    for (int i = 0; i < ep_size; i++) {
        recv_offsets[i] = total_output_tokens;
        total_output_tokens += original_send_splits[i];
    }

    // Calculate send offsets (where each segment starts in fc1_output)
    // Layout: [self_tokens, peer0_tokens, peer1_tokens, ...]
    std::vector<int64_t> send_offsets(ep_size);
    int64_t send_off = 0;
    for (int i = 0; i < ep_size; i++) {
        send_offsets[i] = send_off;
        send_off += seg_ptr[i];
    }

    // Allocate output buffers
    torch::Tensor fc2_output = torch::empty(
        {total_local_tokens, hidden_size},
        fc1_output.options()
    );

    torch::Tensor output = torch::empty(
        {total_output_tokens, hidden_size},
        fc1_output.options()
    );

    // ========================================================================
    // STEP 1: Start receiving from all peers (runs throughout the operation)
    // We need to start recv early so peers can send as soon as their FC2 is done
    // ========================================================================

    // We'll use one comm stream for all recv operations
    cudaStream_t recv_stream = nullptr;
    for (int i = 0; i < ep_size; i++) {
        if (i != rank && ctx.comm_streams[i]) {
            recv_stream = ctx.comm_streams[i];
            break;
        }
    }

    // Start all recv operations in a group
    ncclGroupStart();
    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) continue;

        int64_t recv_count = original_send_splits[peer];
        if (recv_count > 0) {
            half* recv_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>())
                             + recv_offsets[peer] * hidden_size;
            ncclRecv(recv_ptr, recv_count * hidden_size, ncclFloat16,
                     peer, ctx.nccl_comm, recv_stream);
        }
    }

    // ========================================================================
    // STEP 2: Compute FC2 for PEER tokens and send immediately
    // Each peer's FC2 is followed by send - this enables overlap with subsequent peers
    // ========================================================================

    for (int peer = 0; peer < ep_size; peer++) {
        if (peer == rank) continue;

        int peer_idx = (peer < rank) ? peer : peer - 1;
        int64_t peer_count = seg_ptr[peer];

        if (peer_count > 0) {
            // Compute FC2 for this peer's tokens
            const cutlass::half_t* peer_fc1_ptr =
                reinterpret_cast<const cutlass::half_t*>(fc1_output.data_ptr<at::Half>())
                + send_offsets[peer] * ffn_hidden_size;

            cutlass::half_t* peer_fc2_ptr =
                reinterpret_cast<cutlass::half_t*>(fc2_output.data_ptr<at::Half>())
                + send_offsets[peer] * hidden_size;

            // Use per-peer tokens_per_expert (pre-computed host array)
            grouped_gemm_to_buffer(
                peer_fc1_ptr,
                reinterpret_cast<const cutlass::half_t*>(fc2_weight.data_ptr<at::Half>()),
                peer_fc2_ptr,
                h_peer_tokens_per_expert_all[peer_idx].data(),
                num_experts,
                hidden_size,
                ffn_hidden_size,
                compute_stream
            );

            // Record when this peer's FC2 is done
            cudaEventRecord(ctx.peer_recv_done[peer], compute_stream);
        }

        // Start send for this peer (in the ncclGroup started above)
        int64_t send_count = seg_ptr[peer];
        if (send_count > 0) {
            // Wait for this peer's FC2 to complete before sending
            cudaStreamWaitEvent(recv_stream, ctx.peer_recv_done[peer], 0);

            const half* send_ptr = reinterpret_cast<const half*>(fc2_output.data_ptr<at::Half>())
                                   + send_offsets[peer] * hidden_size;
            ncclSend(send_ptr, send_count * hidden_size, ncclFloat16,
                     peer, ctx.nccl_comm, recv_stream);
        }
    }
    ncclGroupEnd();

    // ========================================================================
    // STEP 3: Compute FC2 for SELF tokens (overlaps with peer sends!)
    // This is the key optimization: Self FC2 runs while AllToAll is in progress
    // ========================================================================

    int64_t self_count = seg_ptr[rank];
    if (self_count > 0) {
        const cutlass::half_t* self_fc1_ptr =
            reinterpret_cast<const cutlass::half_t*>(fc1_output.data_ptr<at::Half>())
            + send_offsets[rank] * ffn_hidden_size;

        cutlass::half_t* self_fc2_ptr =
            reinterpret_cast<cutlass::half_t*>(fc2_output.data_ptr<at::Half>())
            + send_offsets[rank] * hidden_size;

        // Use self tokens_per_expert (pre-computed host array)
        grouped_gemm_to_buffer(
            self_fc1_ptr,
            reinterpret_cast<const cutlass::half_t*>(fc2_weight.data_ptr<at::Half>()),
            self_fc2_ptr,
            h_self_tokens_per_expert.data(),
            num_experts,
            hidden_size,
            ffn_hidden_size,
            compute_stream
        );

        // ========================================================================
        // STEP 4: Copy self segment locally (no AllToAll needed for self)
        // ========================================================================

        const half* self_out_src = reinterpret_cast<const half*>(fc2_output.data_ptr<at::Half>())
                                   + send_offsets[rank] * hidden_size;
        half* self_out_dst = reinterpret_cast<half*>(output.data_ptr<at::Half>())
                             + recv_offsets[rank] * hidden_size;

        cudaMemcpyAsync(self_out_dst, self_out_src,
                        self_count * hidden_size * sizeof(half),
                        cudaMemcpyDeviceToDevice, compute_stream);
    }

    // Record self FC2 completion
    cudaEventRecord(ctx.self_fc1_done, compute_stream);

    // ========================================================================
    // STEP 5: Wait for all operations to complete
    // ========================================================================

    // Wait for compute stream (self FC2 + self copy)
    cudaStreamSynchronize(compute_stream);

    // Wait for all communication (recv_stream handles both send and recv)
    cudaStreamSynchronize(recv_stream);

    return output;
}

// ============================================================================
// Python-callable grouped_gemm that writes directly to a pre-allocated buffer
//
// NOW USES TRUE GroupedGEMM! Single kernel for all experts!
//
// This avoids:
// 1. Tensor allocation + copy overhead
// 2. Multiple kernel launch overhead (the key fix!)
//
// Parameters:
//   input: [total_tokens, K] input tensor
//   weight: [num_experts, K, N] weight tensor (note: K, N not N, K!)
//   output: [total_tokens, N] PRE-ALLOCATED output tensor
//   tokens_per_expert: [num_experts] tokens per expert (int32 GPU tensor)
//   output_offset: Starting offset in output tensor to write to
// ============================================================================

void grouped_gemm_to_output(
    torch::Tensor input,              // [total_tokens, K]
    torch::Tensor weight,             // [num_experts, K, N]
    torch::Tensor output,             // [output_size, N] pre-allocated!
    torch::Tensor tokens_per_expert,  // [num_experts] int32
    int64_t output_offset             // Where to start writing in output
) {
    const int num_experts = weight.size(0);
    const int K = weight.size(1);
    const int N = weight.size(2);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Copy tokens_per_expert to host (required for GEMM dispatch)
    std::vector<int> h_tokens(num_experts);
    cudaMemcpyAsync(h_tokens.data(), tokens_per_expert.data_ptr<int>(),
                    num_experts * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Get pointers (apply output_offset)
    const cutlass::half_t* input_ptr = reinterpret_cast<const cutlass::half_t*>(
        input.data_ptr<at::Half>());
    const cutlass::half_t* weight_ptr = reinterpret_cast<const cutlass::half_t*>(
        weight.data_ptr<at::Half>());
    cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(
        output.data_ptr<at::Half>()) + output_offset * N;

    // Use TRUE GroupedGEMM - single kernel for all experts!
    grouped_gemm_true(input_ptr, weight_ptr, output_ptr, h_tokens.data(),
                      num_experts, N, K, stream);
}

// Same function but with GELU activation
// Note: GELU is applied AFTER the GroupedGEMM completes
void grouped_gemm_gelu_to_output(
    torch::Tensor input,              // [total_tokens, K]
    torch::Tensor weight,             // [num_experts, K, N]
    torch::Tensor output,             // [output_size, N] pre-allocated!
    torch::Tensor tokens_per_expert,  // [num_experts] int32
    int64_t output_offset             // Where to start writing in output
) {
    const int num_experts = weight.size(0);
    const int K = weight.size(1);
    const int N = weight.size(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    std::vector<int> h_tokens(num_experts);
    cudaMemcpyAsync(h_tokens.data(), tokens_per_expert.data_ptr<int>(),
                    num_experts * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Calculate total tokens for GELU
    int total_tokens = 0;
    for (int i = 0; i < num_experts; i++) {
        total_tokens += h_tokens[i];
    }

    const cutlass::half_t* input_ptr = reinterpret_cast<const cutlass::half_t*>(
        input.data_ptr<at::Half>());
    const cutlass::half_t* weight_ptr = reinterpret_cast<const cutlass::half_t*>(
        weight.data_ptr<at::Half>());
    cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(
        output.data_ptr<at::Half>()) + output_offset * N;

    // Use TRUE GroupedGEMM - single kernel for all experts!
    grouped_gemm_true(input_ptr, weight_ptr, output_ptr, h_tokens.data(),
                      num_experts, N, K, stream);

    // Apply GELU to entire output at once (more efficient than per-expert)
    launch_gelu(reinterpret_cast<half*>(output_ptr), total_tokens * N, stream);
}

// ============================================================================
// Python-callable TRUE grouped_gemm (standard interface, returns new tensor)
//
// This is a drop-in replacement for the existing grouped_gemm that uses
// the true CUTLASS GroupedGEMM kernel instead of loop-over-experts.
// ============================================================================

torch::Tensor grouped_gemm_true_forward(
    torch::Tensor input,              // [total_tokens, K]
    torch::Tensor weight,             // [num_experts, K, N] (or [num_experts, N, K] if trans_b)
    torch::Tensor tokens_per_expert,  // [num_experts] int32
    int64_t total_tokens,             // Pre-computed total tokens (avoids sync)
    bool trans_a,                     // Currently must be false
    bool trans_b                      // If true, weight is [num_experts, N, K]
) {
    TORCH_CHECK(!trans_a, "trans_a=true not supported yet");

    const int num_experts = weight.size(0);
    int K, N;

    if (!trans_b) {
        // weight: [num_experts, K, N]
        K = weight.size(1);
        N = weight.size(2);
    } else {
        // weight: [num_experts, N, K]
        N = weight.size(1);
        K = weight.size(2);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Copy tokens_per_expert to host
    std::vector<int> h_tokens(num_experts);
    cudaMemcpyAsync(h_tokens.data(), tokens_per_expert.data_ptr<int>(),
                    num_experts * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Allocate output
    torch::Tensor output = torch::empty({total_tokens, N}, input.options());

    const cutlass::half_t* input_ptr = reinterpret_cast<const cutlass::half_t*>(
        input.data_ptr<at::Half>());
    const cutlass::half_t* weight_ptr = reinterpret_cast<const cutlass::half_t*>(
        weight.data_ptr<at::Half>());
    cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(
        output.data_ptr<at::Half>());

    // Use TRUE GroupedGEMM
    grouped_gemm_true(input_ptr, weight_ptr, output_ptr, h_tokens.data(),
                      num_experts, N, K, stream);

    return output;
}

// ============================================================================
// Reorder Kernel: rank-major -> expert-major
// ============================================================================

// CUDA kernel for reordering data from rank-major to expert-major layout
// This kernel copies data while reordering according to reorder_indices
template<typename T>
__global__ void reorder_to_expert_major_kernel(
    const T* __restrict__ input,      // [total_tokens, dim]
    T* __restrict__ output,           // [total_tokens, dim]
    const int64_t* __restrict__ reorder_indices,  // [total_tokens]: expert_pos -> rank_pos
    int64_t total_tokens,
    int64_t dim
) {
    int64_t expert_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_pos >= total_tokens) return;

    int64_t rank_pos = reorder_indices[expert_pos];

    // Copy entire row
    for (int64_t d = 0; d < dim; d++) {
        output[expert_pos * dim + d] = input[rank_pos * dim + d];
    }
}

// Optimized version with vectorized memory access
template<typename T>
__global__ void reorder_to_expert_major_vectorized_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t* __restrict__ reorder_indices,
    int64_t total_tokens,
    int64_t dim
) {
    int64_t expert_pos = blockIdx.x;
    if (expert_pos >= total_tokens) return;

    int64_t rank_pos = reorder_indices[expert_pos];

    // Use vectorized load/store for better memory bandwidth
    const float4* in_vec = reinterpret_cast<const float4*>(input + rank_pos * dim);
    float4* out_vec = reinterpret_cast<float4*>(output + expert_pos * dim);

    int64_t vec_dim = dim / 8;  // float4 = 8 half elements
    for (int64_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        out_vec[i] = in_vec[i];
    }

    // Handle remainder
    int64_t remainder_start = vec_dim * 8;
    for (int64_t i = remainder_start + threadIdx.x; i < dim; i += blockDim.x) {
        output[expert_pos * dim + i] = input[rank_pos * dim + i];
    }
}

// Launch reorder kernel
void launch_reorder_to_expert_major(
    const void* input,
    void* output,
    const int64_t* reorder_indices,
    int64_t total_tokens,
    int64_t dim,
    cudaStream_t stream
) {
    if (total_tokens == 0) return;

    // Use vectorized kernel for better performance
    int block_size = 256;
    reorder_to_expert_major_vectorized_kernel<half><<<total_tokens, block_size, 0, stream>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<half*>(output),
        reorder_indices,
        total_tokens,
        dim
    );
}

// ============================================================================
// Compute reorder indices (rank-major -> expert-major)
// ============================================================================
// Input layout (rank-major):  [self_e0, self_e1, peer0_e0, peer0_e1, ...]
// Output layout (expert-major): [e0_all, e1_all] where each contains [self, peer0, peer1, ...]
// reorder_indices[expert_pos] = rank_pos

__global__ void compute_reorder_indices_kernel(
    int64_t* reorder_indices,         // [total_tokens]
    int64_t* inverse_indices,         // [total_tokens]
    const int* segment_tokens,        // [num_segments * num_experts]
    int num_segments,
    int num_experts,
    int64_t total_tokens
) {
    // Single-threaded kernel to compute indices (small problem size)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Compute expert offsets in expert-major layout
    int64_t expert_offsets[32];  // Max 32 experts
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        int64_t expert_total = 0;
        for (int s = 0; s < num_segments; s++) {
            expert_total += segment_tokens[s * num_experts + e];
        }
        if (e + 1 < num_experts) {
            expert_offsets[e + 1] = expert_offsets[e] + expert_total;
        }
    }

    // Track write position for each expert
    int64_t expert_write_pos[32];
    for (int e = 0; e < num_experts; e++) {
        expert_write_pos[e] = expert_offsets[e];
    }

    // Compute segment starts in rank-major layout
    int64_t segment_starts[64];  // Max 64 segments
    segment_starts[0] = 0;
    for (int s = 0; s < num_segments; s++) {
        int64_t seg_total = 0;
        for (int e = 0; e < num_experts; e++) {
            seg_total += segment_tokens[s * num_experts + e];
        }
        if (s + 1 < num_segments) {
            segment_starts[s + 1] = segment_starts[s] + seg_total;
        }
    }

    // Fill reorder_indices
    for (int s = 0; s < num_segments; s++) {
        int64_t seg_offset = segment_starts[s];
        for (int e = 0; e < num_experts; e++) {
            int count = segment_tokens[s * num_experts + e];
            for (int t = 0; t < count; t++) {
                int64_t rank_pos = seg_offset + t;
                int64_t expert_pos = expert_write_pos[e]++;
                reorder_indices[expert_pos] = rank_pos;
                inverse_indices[rank_pos] = expert_pos;
            }
            seg_offset += count;
        }
    }
}

// Compute reorder indices on GPU
std::pair<torch::Tensor, torch::Tensor> compute_reorder_indices(
    const std::vector<int>& h_self_tokens_per_expert,
    const std::vector<std::vector<int>>& h_peer_tokens_per_expert_all,
    int64_t total_tokens,
    torch::Device device,
    cudaStream_t stream
) {
    int num_experts = h_self_tokens_per_expert.size();
    int num_segments = 1 + h_peer_tokens_per_expert_all.size();

    // Flatten segment tokens to GPU
    std::vector<int> segment_tokens_flat;
    for (int t : h_self_tokens_per_expert) {
        segment_tokens_flat.push_back(t);
    }
    for (const auto& peer_tokens : h_peer_tokens_per_expert_all) {
        for (int t : peer_tokens) {
            segment_tokens_flat.push_back(t);
        }
    }

    // Allocate GPU memory
    int* d_segment_tokens;
    cudaMalloc(&d_segment_tokens, segment_tokens_flat.size() * sizeof(int));
    cudaMemcpyAsync(d_segment_tokens, segment_tokens_flat.data(),
                    segment_tokens_flat.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    torch::Tensor reorder_indices = torch::empty({total_tokens}, torch::dtype(torch::kInt64).device(device));
    torch::Tensor inverse_indices = torch::empty({total_tokens}, torch::dtype(torch::kInt64).device(device));

    // Launch kernel
    compute_reorder_indices_kernel<<<1, 1, 0, stream>>>(
        reorder_indices.data_ptr<int64_t>(),
        inverse_indices.data_ptr<int64_t>(),
        d_segment_tokens,
        num_segments,
        num_experts,
        total_tokens
    );

    cudaFree(d_segment_tokens);

    return {reorder_indices, inverse_indices};
}

// ============================================================================
// Fused AllToAll + FC1 + Activation with Expert-Major Output
// ============================================================================
// Same as moe_alltoall_fc1_fused, but outputs in expert-major layout
// and returns reorder indices for backward

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_alltoall_fc1_fused_expert_major(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    int64_t self_input_offset,
    int64_t self_input_count,
    std::vector<int64_t> send_splits,
    std::vector<int64_t> recv_splits,
    std::vector<int64_t> peer_token_counts,
    std::vector<int> h_self_tokens_per_expert,
    std::vector<std::vector<int>> h_peer_tokens_per_expert_all,
    int activation_type = 0
) {
    auto& ctx = g_moe_fused_ctx;
    TORCH_CHECK(ctx.initialized, "MoE fused context not initialized");

    const int rank = ctx.rank;
    const int ep_size = ctx.ep_size;
    const int num_experts = fc1_weight.size(0);
    const int hidden_size = fc1_weight.size(1);
    const int ffn_hidden_size = fc1_weight.size(2);

    cudaStream_t compute_stream = ctx.compute_stream;

    // Calculate total output tokens
    int64_t total_recv_tokens = 0;
    for (int i = 0; i < ep_size; i++) {
        total_recv_tokens += recv_splits[i];
    }

    // First, call the original function to get rank-major outputs
    auto [fc1_output, segment_sizes, dispatched_input_rank, fc1_pre_act_rank] =
        moe_alltoall_fc1_fused(
            permuted_tokens, fc1_weight,
            self_input_offset, self_input_count,
            send_splits, recv_splits, peer_token_counts,
            h_self_tokens_per_expert, h_peer_tokens_per_expert_all,
            activation_type
        );

    // Compute reorder indices on GPU
    auto [reorder_indices, inverse_indices] = compute_reorder_indices(
        h_self_tokens_per_expert, h_peer_tokens_per_expert_all,
        total_recv_tokens, permuted_tokens.device(), compute_stream
    );

    // fc1_output stays in RANK-MAJOR layout (needed by FC2!)
    // Only reorder dispatched_input and fc1_pre_act to EXPERT-MAJOR (for backward)
    torch::Tensor dispatched_input = torch::empty_like(dispatched_input_rank);
    torch::Tensor fc1_pre_act = torch::empty_like(fc1_pre_act_rank);

    // Reorder dispatched_input and fc1_pre_act to expert-major
    // This reordering can overlap with FC2 computation!
    launch_reorder_to_expert_major(
        dispatched_input_rank.data_ptr(),
        dispatched_input.data_ptr(),
        reorder_indices.data_ptr<int64_t>(),
        total_recv_tokens,
        hidden_size,
        compute_stream
    );

    launch_reorder_to_expert_major(
        fc1_pre_act_rank.data_ptr(),
        fc1_pre_act.data_ptr(),
        reorder_indices.data_ptr<int64_t>(),
        total_recv_tokens,
        ffn_hidden_size,
        compute_stream
    );

    // Don't sync - let reordering overlap with FC2!
    // The caller will sync when needed (e.g., when accessing the reordered tensors)

    return {fc1_output, segment_sizes, dispatched_input, fc1_pre_act, reorder_indices, inverse_indices};
}

}  // namespace fluid
