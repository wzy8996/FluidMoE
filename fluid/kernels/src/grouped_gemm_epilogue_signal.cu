/**
 * CUTLASS GroupedGEMM with True Epilogue-Level Signaling
 *
 * This implements FlashOverlap-style fine-grained signaling where:
 * - Signals are emitted in the Epilogue phase using atomicAdd
 * - Each tile's completion triggers an atomic increment
 * - Communication stream can poll/wait for tile/wave completion
 *
 * Architecture (following FlashOverlap pattern):
 * 1. EpilogueVisitorSignaling - Custom visitor with atomicAdd in end_epilogue()
 * 2. EpilogueWithVisitor - CUTLASS epilogue that calls visitor callbacks
 * 3. GemmGroupedWithEpilogueVisitor - Modified GroupedGEMM kernel using EpilogueWithVisitor
 *
 * The key insight from FlashOverlap:
 * - Use CUTLASS's EpilogueWithVisitorFromExistingEpilogue to create EpilogueWithVisitor
 * - In end_epilogue(), only thread 0 executes atomicAdd to signal tile completion
 * - Use __threadfence_system() to ensure visibility across streams
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

// NCCL includes for FlashOverlap-style C++ loop
#include <nccl.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/arch/memory.h>

#include <vector>
#include <algorithm>
#include <iostream>

namespace fluid {

// ============================================================================
// Type Definitions
// ============================================================================

using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

// ============================================================================
// Wait Kernel (FlashOverlap style)
// ============================================================================

/**
 * Wait kernel that polls an atomic counter until it reaches target value.
 * Does NOT reset the counter - allows multiple waits for the same wave.
 * Used by communication stream to wait for compute tiles to complete.
 */
__global__ void kernel_wait_signal(const int target, int* addr) {
    // Poll until counter reaches target, but don't reset it
    // This allows subsequent calls (like wait_for_all_waves) to still see the value
    while (atomicAdd(addr, 0) < target) {
        __nanosleep(100);
    }
}

/**
 * Batched wait kernel that processes all waves sequentially.
 * Eliminates Python loop overhead by doing all waits in a single kernel launch.
 */
__global__ void kernel_wait_all_waves(
    int num_waves,
    const int* __restrict__ wave_targets,
    int* __restrict__ wave_counters
) {
    for (int wave_idx = 0; wave_idx < num_waves; wave_idx++) {
        int target = wave_targets[wave_idx];
        int* addr = wave_counters + wave_idx;

        while (atomicCAS(addr, target, 0) != target) {
            __nanosleep(100);
        }
    }
}

// ============================================================================
// EpilogueVisitorSignaling - FlashOverlap style signaling in Epilogue
// ============================================================================

/**
 * Custom EpilogueVisitor that emits signals when tiles complete.
 *
 * Based on FlashOverlap's EpilogueVisitorSignaling:
 * - Thread 0 of each threadblock signals in end_epilogue()
 * - Uses atomicAdd to increment wave counter
 * - __threadfence_system() ensures cross-stream visibility
 */
template <
    typename ThreadblockShape_,
    int ThreadCount,
    typename OutputTileIterator_,
    typename AccumulatorTile_,
    typename ElementAccumulator_,
    typename ElementwiseFunctor_
>
class EpilogueVisitorSignaling {
public:
    using AccumulatorTile = AccumulatorTile_;
    using ThreadblockShape = ThreadblockShape_;
    static int const kThreadCount = ThreadCount;

    using OutputTileIterator = OutputTileIterator_;
    using ElementwiseFunctor = ElementwiseFunctor_;

    static int const kIterations = OutputTileIterator::kIterations;
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    using ElementOutput = typename OutputTileIterator::Element;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ElementAccumulator = ElementAccumulator_;

    using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    using OutputVector = cutlass::Array<ElementOutput, kElementsPerAccess>;
    using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

    static int const ThreadblockM = ThreadblockShape::kM;
    static int const ThreadblockN = ThreadblockShape::kN;

    /// Argument structure
    struct Arguments {
        typename ElementwiseFunctor::Params elementwise;
        TensorRefD ref_C;
        TensorRefD ref_D;

        // Signal parameters
        int* wave_counters;        // Array of counters, one per wave
        int* wave_tile_counts;     // Expected tiles per wave (for reference)
        int tiles_per_wave;        // Tiles per wave
        int* problem_tile_offsets; // Cumulative tile offset per problem
        int total_problems;        // Total number of problems (experts)

        // Monitoring mode parameters (FlashOverlap hint collection)
        bool monitor_mode;         // If true, collect tile completion order
        int* hint_buffer;          // Buffer to store completion order
        int* reorder_array;        // Tile reorder array (RA) for output locality
        int* completion_counter;   // Global counter for tile completion order

        Arguments() : monitor_mode(false), hint_buffer(nullptr), reorder_array(nullptr), completion_counter(nullptr) {}

        Arguments(
            typename ElementwiseFunctor::Params elementwise_,
            TensorRefD ref_C_,
            TensorRefD ref_D_,
            int* wave_counters_,
            int* wave_tile_counts_,
            int tiles_per_wave_,
            int* problem_tile_offsets_,
            int total_problems_,
            bool monitor_mode_ = false,
            int* hint_buffer_ = nullptr,
            int* reorder_array_ = nullptr,
            int* completion_counter_ = nullptr
        ):
            elementwise(elementwise_),
            ref_C(ref_C_),
            ref_D(ref_D_),
            wave_counters(wave_counters_),
            wave_tile_counts(wave_tile_counts_),
            tiles_per_wave(tiles_per_wave_),
            problem_tile_offsets(problem_tile_offsets_),
            total_problems(total_problems_),
            monitor_mode(monitor_mode_),
            hint_buffer(hint_buffer_),
            reorder_array(reorder_array_),
            completion_counter(completion_counter_)
        {}
    };

    struct Params {
        typename ElementwiseFunctor::Params elementwise;

        // For per-problem output, we store raw strides and pointers
        // The actual OutputTileIterator::Params is created in the visitor constructor
        typename OutputTileIterator::Element* ptr_C;
        typename OutputTileIterator::Element* ptr_D;
        int64_t ldc;  // stride for C
        int64_t ldd;  // stride for D

        // Signal parameters
        int* wave_counters;
        int* wave_tile_counts;
        int tiles_per_wave;
        int* problem_tile_offsets;
        int total_problems;

        // Monitoring mode parameters
        bool monitor_mode;
        int* hint_buffer;
        int* reorder_array;
        int* completion_counter;

        // Per-problem info (set by kernel)
        int problem_idx;
        int num_tiles_n;

        CUTLASS_HOST_DEVICE
        Params() : ptr_C(nullptr), ptr_D(nullptr), ldc(0), ldd(0),
                   wave_counters(nullptr), problem_idx(0), num_tiles_n(0),
                   monitor_mode(false), hint_buffer(nullptr), reorder_array(nullptr),
                   completion_counter(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            elementwise(args.elementwise),
            ptr_C(args.ref_C.data()),
            ptr_D(args.ref_D.data()),
            ldc(args.ref_C.layout().stride(0)),
            ldd(args.ref_D.layout().stride(0)),
            wave_counters(args.wave_counters),
            wave_tile_counts(args.wave_tile_counts),
            tiles_per_wave(args.tiles_per_wave),
            problem_tile_offsets(args.problem_tile_offsets),
            total_problems(args.total_problems),
            monitor_mode(args.monitor_mode),
            hint_buffer(args.hint_buffer),
            reorder_array(args.reorder_array),
            completion_counter(args.completion_counter),
            problem_idx(0),
            num_tiles_n(0)
        {}
    };

    struct SharedStorage {};

private:
    Params params_;
    SharedStorage& shared_storage_;
    cutlass::MatrixCoord extent_;
    ElementwiseFunctor elementwise_;

    OutputTileIterator iterator_C_;
    OutputTileIterator iterator_D_;
    typename OutputTileIterator::Fragment fragment_C_;
    typename OutputTileIterator::Fragment fragment_D_;

    ElementAccumulator alpha_;
    ElementAccumulator beta_;

    cutlass::MatrixCoord threadblock_offset_;

public:
    CUTLASS_DEVICE
    EpilogueVisitorSignaling(
        Params const& params,
        SharedStorage& shared_storage,
        cutlass::MatrixCoord const& problem_size,
        int thread_idx,
        int warp_idx,
        int lane_idx,
        cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0)
    ):
        params_(params),
        shared_storage_(shared_storage),
        extent_(problem_size),
        elementwise_(params.elementwise),
        // Create OutputTileIterator::Params from raw strides
        iterator_C_(typename OutputTileIterator::Params(LayoutOutput(params.ldc)),
                    params.ptr_C, problem_size, thread_idx, threadblock_offset),
        iterator_D_(typename OutputTileIterator::Params(LayoutOutput(params.ldd)),
                    params.ptr_D, problem_size, thread_idx, threadblock_offset),
        threadblock_offset_(threadblock_offset)
    {
        alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
        beta_ = (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

        if (beta_ == ElementAccumulator()) {
            iterator_C_.clear_mask();
        }
    }

    CUTLASS_DEVICE
    void set_k_partition(int split_k_index, int split_k_slices) {}

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {}

    CUTLASS_DEVICE
    void begin_epilogue() {}

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        fragment_D_.clear();
        iterator_C_.load(fragment_C_);
        ++iterator_C_;
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {}

    CUTLASS_DEVICE
    void visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorFragment const& accum
    ) {
        using Converter = cutlass::NumericArrayConverter<
            ElementAccumulator, ElementOutput, kElementsPerAccess>;

        Converter converter;
        AccumulatorFragment source = converter(
            reinterpret_cast<OutputVector const*>(&fragment_C_)[frag_idx]);

        // Apply linear combination: D = alpha * accum + beta * C
        AccumulatorFragment result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
            result[i] = alpha_ * accum[i] + beta_ * source[i];
        }

        // Convert back to output type
        cutlass::NumericArrayConverter<ElementOutput, ElementAccumulator, kElementsPerAccess>
            output_converter;
        reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx] = output_converter(result);
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {}

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        iterator_D_.store(fragment_D_);
        ++iterator_D_;
    }

    /// Called after all steps have been completed - THIS IS WHERE WE SIGNAL!
    CUTLASS_DEVICE
    void end_epilogue() {
        // Only thread 0 of this threadblock signals
        if (threadIdx.x > 0) return;

        if (params_.problem_tile_offsets == nullptr) return;

        // Calculate this tile's index within the problem
        int tile_m = threadblock_offset_.row() / ThreadblockM;
        int tile_n = threadblock_offset_.column() / ThreadblockN;
        int local_tile_idx = tile_m * params_.num_tiles_n + tile_n;

        // Global tile index considering problem offset
        int global_tile_idx = params_.problem_tile_offsets[params_.problem_idx] + local_tile_idx;

        // ========================================
        // MONITORING MODE: Record tile completion order (FlashOverlap hint collection)
        // ========================================
        if (params_.monitor_mode && params_.hint_buffer != nullptr && params_.completion_counter != nullptr) {
            // Use atomicAdd on the external counter to get unique completion order
            // FlashOverlap style: atomicAdd has acquire-release semantics, no explicit fence needed
            int order = atomicAdd(params_.completion_counter, 1);
            params_.hint_buffer[global_tile_idx] = order;
            return;  // In monitoring mode, don't signal waves
        }

        // ========================================
        // TILE REORDERING: Map original tile to reordered position
        // ========================================
        int mapped_tile_idx = global_tile_idx;
        if (params_.reorder_array != nullptr) {
            // reorder_array[new_pos] = old_tile_idx
            // We need inverse: given old_tile_idx, find new_pos
            // For efficiency, we compute wave from original tile position
            // The reorder array is used during communication to know which tiles are ready
            mapped_tile_idx = global_tile_idx;  // Keep original for wave calculation
        }

        // ========================================
        // WAVE SIGNALING: Increment wave counter
        // ========================================
        if (params_.wave_counters == nullptr) return;

        // Determine which wave this tile belongs to
        int wave_idx = mapped_tile_idx / params_.tiles_per_wave;

        // Atomically increment the wave counter
        // FlashOverlap style: atomicAdd has acquire-release semantics, no explicit fence needed
        atomicAdd(&params_.wave_counters[wave_idx], 1);
    }
};

// ============================================================================
// Type definitions for the GroupedGEMM kernel with EpilogueVisitor
// ============================================================================

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
>;

// First, get the default kernel to extract Mma and Epilogue types
using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementInput, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementOutput, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TILE_M, TILE_N, TILE_K>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
>::GemmKernel;

// Create our custom EpilogueVisitor
using CustomEpilogueVisitor = EpilogueVisitorSignaling<
    typename DefaultGemmKernel::ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    typename DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
    ElementAccumulator,
    EpilogueOp
>;

// Create EpilogueWithVisitor from existing epilogue
using EpilogueWithVisitor = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    CustomEpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
>::Epilogue;

// Standard GemmGrouped for comparison (without custom epilogue)
using GemmGrouped = cutlass::gemm::device::GemmGrouped<DefaultGemmKernel>;

// ============================================================================
// Custom GroupedGEMM Kernel with EpilogueVisitor
// ============================================================================

/**
 * Modified GemmGrouped kernel that uses our custom EpilogueVisitor.
 * This is similar to GemmWithEpilogueVisitor but for grouped problems.
 */
template <
    typename Mma_,
    typename Epilogue_,
    typename EpilogueVisitor_,
    typename ThreadblockSwizzle_,
    cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_,
    bool Transposed = false
>
struct GemmGroupedWithVisitor {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueVisitor = EpilogueVisitor_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static cutlass::gemm::kernel::GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
    static bool const kTransposed = Transposed;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename EpilogueVisitor::ElementOutput;
    using LayoutC = cutlass::layout::RowMajor;

    using ThreadblockShape = typename Mma::Shape;
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    using ProblemVisitor = cutlass::gemm::kernel::GemmGroupedProblemVisitor<
        ThreadblockShape,
        kGroupScheduleMode,
        kThreadCount,
        kThreadCount,
        kTransposed
    >;

    struct Arguments {
        cutlass::gemm::GemmCoord *problem_sizes;
        int problem_count;
        int threadblock_count;

        typename EpilogueVisitor::Params epilogue_visitor;

        ElementA ** ptr_A;
        ElementB ** ptr_B;
        ElementC ** ptr_C;
        ElementC ** ptr_D;

        int64_t *lda;
        int64_t *ldb;
        int64_t *ldc;
        int64_t *ldd;

        cutlass::gemm::GemmCoord *host_problem_sizes;

        Arguments() = default;

        Arguments(
            cutlass::gemm::GemmCoord *problem_sizes_,
            int problem_count_,
            int threadblock_count_,
            typename EpilogueVisitor::Params epilogue_visitor_,
            ElementA ** ptr_A_,
            ElementB ** ptr_B_,
            ElementC ** ptr_C_,
            ElementC ** ptr_D_,
            int64_t *lda_,
            int64_t *ldb_,
            int64_t *ldc_,
            int64_t *ldd_,
            cutlass::gemm::GemmCoord *host_problem_sizes_ = nullptr
        ):
            problem_sizes(problem_sizes_),
            problem_count(problem_count_),
            threadblock_count(threadblock_count_),
            epilogue_visitor(epilogue_visitor_),
            ptr_A(ptr_A_),
            ptr_B(ptr_B_),
            ptr_C(ptr_C_),
            ptr_D(ptr_D_),
            lda(lda_),
            ldb(ldb_),
            ldc(ldc_),
            ldd(ldd_),
            host_problem_sizes(host_problem_sizes_)
        {}
    };

    struct Params {
        typename ProblemVisitor::Params problem_visitor;
        int threadblock_count;

        typename EpilogueVisitor::Params epilogue_visitor;

        ElementA ** ptr_A;
        ElementB ** ptr_B;
        ElementC ** ptr_C;
        ElementC ** ptr_D;

        int64_t *lda;
        int64_t *ldb;
        int64_t *ldc;
        int64_t *ldd;

        Params() = default;

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args, void *workspace = nullptr, int tile_count = 0):
            problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
            threadblock_count(args.threadblock_count),
            epilogue_visitor(args.epilogue_visitor),
            ptr_A(args.ptr_A),
            ptr_B(args.ptr_B),
            ptr_C(args.ptr_C),
            ptr_D(args.ptr_D),
            lda(args.lda),
            ldb(args.ldb),
            ldc(args.ldc),
            ldd(args.ldd)
        {}
    };

    struct SharedStorage {
        union {
            typename Mma::SharedStorage main_loop;
            struct {
                typename Epilogue::SharedStorage epilogue;
                typename EpilogueVisitor::SharedStorage visitor;
            } epilogue;
        } kernel;

        typename ProblemVisitor::SharedStorage problem_visitor;
    };

    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        ProblemVisitor problem_visitor(
            params.problem_visitor,
            shared_storage.problem_visitor,
            blockIdx.x);

        while (problem_visitor.next_tile()) {
            cutlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
            int32_t problem_idx = problem_visitor.problem_index();
            int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

            cutlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

            cutlass::gemm::GemmCoord threadblock_offset(
                int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
                int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
                0);

            ElementA *ptr_A = params.ptr_A[problem_idx];
            int64_t ldm_A = params.lda[problem_idx];

            ElementB *ptr_B = params.ptr_B[problem_idx];
            int64_t ldm_B = params.ldb[problem_idx];

            cutlass::MatrixCoord tb_offset_A{threadblock_offset.m(), 0};
            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            int thread_idx = threadIdx.x;

            typename Mma::IteratorA iterator_A(
                LayoutA(ldm_A),
                ptr_A,
                {problem_size.m(), problem_size.k()},
                thread_idx,
                tb_offset_A);

            typename Mma::IteratorB iterator_B(
                LayoutB(ldm_B),
                ptr_B,
                {problem_size.k(), problem_size.n()},
                thread_idx,
                tb_offset_B);

            typename Mma::FragmentC accumulators;
            accumulators.clear();

            int warp_idx = cutlass::canonical_warp_idx_sync();
            int lane_idx = threadIdx.x % 32;

            Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            __syncthreads();

            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            //
            // Epilogue with Visitor
            //

            ElementC *ptr_C = params.ptr_C[problem_idx];
            ElementC *ptr_D = params.ptr_D[problem_idx];
            int64_t ldc = params.ldc[problem_idx];
            int64_t ldd = params.ldd[problem_idx];

            // Create visitor params with problem-specific info
            typename EpilogueVisitor::Params visitor_params = params.epilogue_visitor;

            // Update with problem-specific values (using raw strides, not Params objects)
            visitor_params.ptr_C = ptr_C;
            visitor_params.ptr_D = ptr_D;
            visitor_params.ldc = ldc;
            visitor_params.ldd = ldd;
            visitor_params.problem_idx = problem_idx;

            // Calculate tiles in N dimension for this problem
            int tiles_n = (problem_size.n() + Mma::Shape::kN - 1) / Mma::Shape::kN;
            visitor_params.num_tiles_n = tiles_n;

            EpilogueVisitor epilogue_visitor(
                visitor_params,
                shared_storage.kernel.epilogue.visitor,
                problem_size.mn(),
                thread_idx,
                warp_idx,
                lane_idx,
                threadblock_offset.mn());

            Epilogue epilogue(
                shared_storage.kernel.epilogue.epilogue,
                thread_idx,
                warp_idx,
                lane_idx);

            epilogue(epilogue_visitor, accumulators);

            problem_visitor.advance(gridDim.x);
        }
    }
};

// Define our custom kernel
using GemmGroupedWithSignal = GemmGroupedWithVisitor<
    typename DefaultGemmKernel::Mma,
    EpilogueWithVisitor,
    CustomEpilogueVisitor,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    false
>;

// ============================================================================
// Context Structure
// ============================================================================

// Pre-allocation constants for reduced malloc overhead
static constexpr int MAX_EXPERTS = 64;
static constexpr int MAX_WAVES = 256;

struct EpilogueSignalContext {
    // ========================================
    // UNIFIED DEVICE MEMORY BLOCK
    // All device allocations in one block to reduce cudaMalloc calls
    // ========================================
    void* d_unified_block = nullptr;
    size_t unified_block_size = 0;

    // Pointers into unified block (no separate allocation needed)
    int* d_wave_counters = nullptr;
    int* d_wave_tile_counts = nullptr;
    int* d_problem_tile_offsets = nullptr;
    cutlass::gemm::GemmCoord* d_problem_sizes = nullptr;
    ElementInput** d_ptr_A = nullptr;
    ElementInput** d_ptr_B = nullptr;
    ElementOutput** d_ptr_C = nullptr;
    ElementOutput** d_ptr_D = nullptr;
    int64_t* d_lda = nullptr;
    int64_t* d_ldb = nullptr;
    int64_t* d_ldc = nullptr;
    int64_t* d_ldd = nullptr;

    // ========================================
    // PINNED HOST MEMORY (for fast async transfer)
    // ========================================
    void* h_pinned_block = nullptr;
    size_t pinned_block_size = 0;

    // Pointers into pinned block
    ElementInput** h_ptr_A = nullptr;
    ElementInput** h_ptr_B = nullptr;
    ElementOutput** h_ptr_C = nullptr;
    int64_t* h_lda = nullptr;
    int64_t* h_ldb = nullptr;
    int64_t* h_ldc = nullptr;
    cutlass::gemm::GemmCoord* h_problem_sizes = nullptr;
    int* h_wave_tile_counts = nullptr;
    int* h_problem_tile_offsets = nullptr;

    // Workspace (separate allocation, size varies)
    void* workspace = nullptr;
    size_t workspace_size = 0;

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;

    // NCCL for FlashOverlap-style C++ loop
    ncclComm_t nccl_comm = nullptr;
    int nccl_rank = -1;
    int nccl_world_size = 0;
    bool nccl_initialized = false;
    bool nccl_owns_comm = false;

    // Wave boundaries for NCCL (token ranges)
    std::vector<int> wave_token_starts;
    std::vector<int> wave_token_counts;

    int num_experts;
    int num_waves;
    int tiles_per_wave;
    int total_tiles;
    int sm_count;
    int K, N;
    int total_tokens;
    int allocated_experts;  // Track allocated capacity

    std::vector<int> expert_offsets;
    std::vector<int> tokens_per_expert_vec;
    std::vector<int> expert_tiles;

    bool initialized;
    bool memory_allocated;  // Track if unified blocks are allocated

    EpilogueSignalContext() : initialized(false), memory_allocated(false),
        d_unified_block(nullptr), unified_block_size(0),
        h_pinned_block(nullptr), pinned_block_size(0),
        d_wave_counters(nullptr), d_wave_tile_counts(nullptr),
        d_problem_tile_offsets(nullptr), d_problem_sizes(nullptr),
        d_ptr_A(nullptr), d_ptr_B(nullptr), d_ptr_C(nullptr), d_ptr_D(nullptr),
        d_lda(nullptr), d_ldb(nullptr), d_ldc(nullptr), d_ldd(nullptr),
        h_ptr_A(nullptr), h_ptr_B(nullptr), h_ptr_C(nullptr),
        h_lda(nullptr), h_ldb(nullptr), h_ldc(nullptr),
        h_problem_sizes(nullptr), h_wave_tile_counts(nullptr), h_problem_tile_offsets(nullptr),
        workspace(nullptr), workspace_size(0),
        compute_stream(nullptr), comm_stream(nullptr),
        nccl_comm(nullptr), nccl_rank(0), nccl_world_size(1), nccl_initialized(false),
        allocated_experts(0) {}

    // Helper to align pointer to 16-byte boundary
    static inline char* align16(char* ptr) {
        return reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(ptr) + 15) & ~15);
    }

    // Allocate unified memory blocks (called once, reused across calls)
    void allocate_memory(int max_experts = MAX_EXPERTS, int max_waves = MAX_WAVES) {
        if (memory_allocated && allocated_experts >= max_experts) return;

        // Free old allocations if resizing
        if (d_unified_block) cudaFree(d_unified_block);
        if (h_pinned_block) cudaFreeHost(h_pinned_block);

        // Calculate unified device block size with 16-byte alignment for each section
        // Add extra padding for alignment
        size_t device_size = 0;
        device_size += max_waves * sizeof(int) + 16;           // d_wave_counters
        device_size += max_waves * sizeof(int) + 16;           // d_wave_tile_counts
        device_size += (max_experts + 1) * sizeof(int) + 16;   // d_problem_tile_offsets
        device_size += max_experts * sizeof(cutlass::gemm::GemmCoord) + 16;  // d_problem_sizes
        device_size += max_experts * sizeof(ElementInput*) + 16;   // d_ptr_A
        device_size += max_experts * sizeof(ElementInput*) + 16;   // d_ptr_B
        device_size += max_experts * sizeof(ElementOutput*) + 16;  // d_ptr_C
        device_size += max_experts * sizeof(ElementOutput*) + 16;  // d_ptr_D
        device_size += max_experts * sizeof(int64_t) + 16;     // d_lda
        device_size += max_experts * sizeof(int64_t) + 16;     // d_ldb
        device_size += max_experts * sizeof(int64_t) + 16;     // d_ldc
        device_size += max_experts * sizeof(int64_t) + 16;     // d_ldd
        device_size = (device_size + 255) & ~255;  // Align total to 256 bytes

        cudaMalloc(&d_unified_block, device_size);
        unified_block_size = device_size;

        // Partition device memory with 16-byte alignment
        char* ptr = static_cast<char*>(d_unified_block);
        d_wave_counters = reinterpret_cast<int*>(ptr);
        ptr += max_waves * sizeof(int); ptr = align16(ptr);
        d_wave_tile_counts = reinterpret_cast<int*>(ptr);
        ptr += max_waves * sizeof(int); ptr = align16(ptr);
        d_problem_tile_offsets = reinterpret_cast<int*>(ptr);
        ptr += (max_experts + 1) * sizeof(int); ptr = align16(ptr);
        d_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(ptr);
        ptr += max_experts * sizeof(cutlass::gemm::GemmCoord); ptr = align16(ptr);
        d_ptr_A = reinterpret_cast<ElementInput**>(ptr);
        ptr += max_experts * sizeof(ElementInput*); ptr = align16(ptr);
        d_ptr_B = reinterpret_cast<ElementInput**>(ptr);
        ptr += max_experts * sizeof(ElementInput*); ptr = align16(ptr);
        d_ptr_C = reinterpret_cast<ElementOutput**>(ptr);
        ptr += max_experts * sizeof(ElementOutput*); ptr = align16(ptr);
        d_ptr_D = reinterpret_cast<ElementOutput**>(ptr);
        ptr += max_experts * sizeof(ElementOutput*); ptr = align16(ptr);
        d_lda = reinterpret_cast<int64_t*>(ptr);
        ptr += max_experts * sizeof(int64_t); ptr = align16(ptr);
        d_ldb = reinterpret_cast<int64_t*>(ptr);
        ptr += max_experts * sizeof(int64_t); ptr = align16(ptr);
        d_ldc = reinterpret_cast<int64_t*>(ptr);
        ptr += max_experts * sizeof(int64_t); ptr = align16(ptr);
        d_ldd = reinterpret_cast<int64_t*>(ptr);

        // Calculate pinned host block size with 16-byte alignment
        size_t host_size = 0;
        host_size += max_experts * sizeof(ElementInput*) + 16;   // h_ptr_A
        host_size += max_experts * sizeof(ElementInput*) + 16;   // h_ptr_B
        host_size += max_experts * sizeof(ElementOutput*) + 16;  // h_ptr_C
        host_size += max_experts * sizeof(int64_t) + 16;         // h_lda
        host_size += max_experts * sizeof(int64_t) + 16;         // h_ldb
        host_size += max_experts * sizeof(int64_t) + 16;         // h_ldc
        host_size += max_experts * sizeof(cutlass::gemm::GemmCoord) + 16;  // h_problem_sizes
        host_size += max_waves * sizeof(int) + 16;               // h_wave_tile_counts
        host_size += (max_experts + 1) * sizeof(int) + 16;       // h_problem_tile_offsets
        host_size = (host_size + 255) & ~255;

        cudaHostAlloc(&h_pinned_block, host_size, cudaHostAllocDefault);
        pinned_block_size = host_size;

        // Partition pinned host memory with 16-byte alignment
        char* hptr = static_cast<char*>(h_pinned_block);
        h_ptr_A = reinterpret_cast<ElementInput**>(hptr);
        hptr += max_experts * sizeof(ElementInput*); hptr = align16(hptr);
        h_ptr_B = reinterpret_cast<ElementInput**>(hptr);
        hptr += max_experts * sizeof(ElementInput*); hptr = align16(hptr);
        h_ptr_C = reinterpret_cast<ElementOutput**>(hptr);
        hptr += max_experts * sizeof(ElementOutput*); hptr = align16(hptr);
        h_lda = reinterpret_cast<int64_t*>(hptr);
        hptr += max_experts * sizeof(int64_t); hptr = align16(hptr);
        h_ldb = reinterpret_cast<int64_t*>(hptr);
        hptr += max_experts * sizeof(int64_t); hptr = align16(hptr);
        h_ldc = reinterpret_cast<int64_t*>(hptr);
        hptr += max_experts * sizeof(int64_t); hptr = align16(hptr);
        h_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(hptr);
        hptr += max_experts * sizeof(cutlass::gemm::GemmCoord); hptr = align16(hptr);
        h_wave_tile_counts = reinterpret_cast<int*>(hptr);
        hptr += max_waves * sizeof(int); hptr = align16(hptr);
        h_problem_tile_offsets = reinterpret_cast<int*>(hptr);

        allocated_experts = max_experts;
        memory_allocated = true;
    }
};

static EpilogueSignalContext* g_epi_ctx = nullptr;

// ============================================================================
// Helper Functions
// ============================================================================

inline int get_sm_count_epi() {
    static int sm_count = -1;
    if (sm_count < 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    return sm_count;
}

int calculate_expert_tiles_epi(int tokens, int N) {
    if (tokens <= 0) return 0;
    int tiles_m = (tokens + TILE_M - 1) / TILE_M;
    int tiles_n = (N + TILE_N - 1) / TILE_N;
    return tiles_m * tiles_n;
}

// ============================================================================
// Context Management
// ============================================================================

// Structure to preserve NCCL state across context reinitializations
struct NCCLState {
    ncclComm_t comm = nullptr;
    int rank = -1;
    int world_size = 0;
    bool initialized = false;
    bool owns_comm = false;
};

static NCCLState g_nccl_state;

void cleanup_epilogue_signal_context_internal(bool preserve_nccl) {
    if (g_epi_ctx == nullptr) return;

    // Preserve NCCL state if requested
    if (preserve_nccl && g_epi_ctx->nccl_initialized) {
        g_nccl_state.comm = g_epi_ctx->nccl_comm;
        g_nccl_state.rank = g_epi_ctx->nccl_rank;
        g_nccl_state.world_size = g_epi_ctx->nccl_world_size;
        g_nccl_state.initialized = true;
        g_nccl_state.owns_comm = g_epi_ctx->nccl_owns_comm;
    } else {
        // Cleanup NCCL if we own it and not preserving
        if (g_epi_ctx->nccl_initialized && g_epi_ctx->nccl_owns_comm && g_epi_ctx->nccl_comm != nullptr) {
            ncclCommDestroy(g_epi_ctx->nccl_comm);
        }
    }

    // Free unified device block (single cudaFree instead of many)
    if (g_epi_ctx->d_unified_block) cudaFree(g_epi_ctx->d_unified_block);

    // Free pinned host block (single cudaFreeHost instead of many)
    if (g_epi_ctx->h_pinned_block) cudaFreeHost(g_epi_ctx->h_pinned_block);

    // Free workspace separately (variable size)
    if (g_epi_ctx->workspace) cudaFree(g_epi_ctx->workspace);

    if (g_epi_ctx->compute_stream) cudaStreamDestroy(g_epi_ctx->compute_stream);
    if (g_epi_ctx->comm_stream) cudaStreamDestroy(g_epi_ctx->comm_stream);

    delete g_epi_ctx;
    g_epi_ctx = nullptr;
}

void cleanup_epilogue_signal_context() {
    // For explicit cleanup, destroy NCCL too
    cleanup_epilogue_signal_context_internal(false);
    // Also clear preserved state
    if (g_nccl_state.initialized && g_nccl_state.owns_comm && g_nccl_state.comm != nullptr) {
        ncclCommDestroy(g_nccl_state.comm);
    }
    g_nccl_state = NCCLState();
}

bool needs_reinit_epilogue(
    const int* tokens_per_expert,
    int num_experts,
    int K, int N,
    int total_tokens,
    int tiles_per_wave
) {
    if (g_epi_ctx == nullptr || !g_epi_ctx->initialized) return true;
    if (g_epi_ctx->num_experts != num_experts) return true;
    if (g_epi_ctx->K != K || g_epi_ctx->N != N) return true;
    if (g_epi_ctx->total_tokens != total_tokens) return true;
    // CRITICAL: Check if tiles_per_wave changed - affects wave counter targets
    if (g_epi_ctx->tiles_per_wave != tiles_per_wave) return true;

    for (int e = 0; e < num_experts; e++) {
        if (g_epi_ctx->tokens_per_expert_vec[e] != tokens_per_expert[e]) return true;
    }
    return false;
}

void init_epilogue_signal_context(
    const int* tokens_per_expert,
    int num_experts,
    int K, int N,
    int total_tokens,
    ElementInput* input_ptr,
    ElementInput* weight_ptr,
    ElementOutput* output_ptr,
    int tiles_per_wave
) {
    // ========================================
    // FAST PATH: Reuse existing context (most common case)
    // Only update pointers and reset wave counters
    // ========================================
    if (!needs_reinit_epilogue(tokens_per_expert, num_experts, K, N, total_tokens, tiles_per_wave)) {
        // Update pointers in pinned memory (no allocation needed)
        for (int e = 0; e < num_experts; e++) {
            int offset = g_epi_ctx->expert_offsets[e];
            g_epi_ctx->h_ptr_A[e] = input_ptr + offset * K;
            g_epi_ctx->h_ptr_B[e] = weight_ptr + e * K * N;
            g_epi_ctx->h_ptr_C[e] = output_ptr + offset * N;
        }

        // Async memcpy from pinned memory is truly async
        cudaMemcpyAsync(g_epi_ctx->d_ptr_A, g_epi_ctx->h_ptr_A,
                        num_experts * sizeof(ElementInput*), cudaMemcpyHostToDevice,
                        g_epi_ctx->compute_stream);
        cudaMemcpyAsync(g_epi_ctx->d_ptr_B, g_epi_ctx->h_ptr_B,
                        num_experts * sizeof(ElementInput*), cudaMemcpyHostToDevice,
                        g_epi_ctx->compute_stream);
        cudaMemcpyAsync(g_epi_ctx->d_ptr_C, g_epi_ctx->h_ptr_C,
                        num_experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice,
                        g_epi_ctx->compute_stream);
        cudaMemcpyAsync(g_epi_ctx->d_ptr_D, g_epi_ctx->h_ptr_C,
                        num_experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice,
                        g_epi_ctx->compute_stream);

        // Reset wave counters to 0
        cudaMemsetAsync(g_epi_ctx->d_wave_counters, 0,
                        g_epi_ctx->num_waves * sizeof(int),
                        g_epi_ctx->compute_stream);
        return;
    }

    // ========================================
    // FULL INIT PATH: Create new context with unified memory blocks
    // ========================================
    if (g_epi_ctx != nullptr) {
        cleanup_epilogue_signal_context_internal(true);
    }

    g_epi_ctx = new EpilogueSignalContext();

    // Restore NCCL state if preserved
    if (g_nccl_state.initialized) {
        g_epi_ctx->nccl_comm = g_nccl_state.comm;
        g_epi_ctx->nccl_rank = g_nccl_state.rank;
        g_epi_ctx->nccl_world_size = g_nccl_state.world_size;
        g_epi_ctx->nccl_initialized = true;
        g_epi_ctx->nccl_owns_comm = g_nccl_state.owns_comm;
        g_nccl_state = NCCLState();
    }

    // Store parameters
    g_epi_ctx->num_experts = num_experts;
    g_epi_ctx->K = K;
    g_epi_ctx->N = N;
    g_epi_ctx->total_tokens = total_tokens;
    g_epi_ctx->sm_count = get_sm_count_epi();
    g_epi_ctx->tiles_per_wave = (tiles_per_wave > 0) ? tiles_per_wave : g_epi_ctx->sm_count;

    // Store tokens per expert
    g_epi_ctx->tokens_per_expert_vec.resize(num_experts);
    for (int e = 0; e < num_experts; e++) {
        g_epi_ctx->tokens_per_expert_vec[e] = tokens_per_expert[e];
    }

    // Calculate expert offsets
    g_epi_ctx->expert_offsets.resize(num_experts + 1);
    g_epi_ctx->expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        g_epi_ctx->expert_offsets[e + 1] = g_epi_ctx->expert_offsets[e] + tokens_per_expert[e];
    }

    // Calculate tiles
    g_epi_ctx->expert_tiles.resize(num_experts);
    g_epi_ctx->total_tiles = 0;
    for (int e = 0; e < num_experts; e++) {
        g_epi_ctx->expert_tiles[e] = calculate_expert_tiles_epi(tokens_per_expert[e], N);
        g_epi_ctx->total_tiles += g_epi_ctx->expert_tiles[e];
    }

    g_epi_ctx->num_waves = (g_epi_ctx->total_tiles + g_epi_ctx->tiles_per_wave - 1)
                           / g_epi_ctx->tiles_per_wave;

    // ========================================
    // ALLOCATE UNIFIED MEMORY BLOCKS (single cudaMalloc for all buffers)
    // ========================================
    int max_experts = std::max(num_experts, MAX_EXPERTS);
    int max_waves = std::max(g_epi_ctx->num_waves, MAX_WAVES);
    g_epi_ctx->allocate_memory(max_experts, max_waves);

    // ========================================
    // POPULATE PINNED HOST MEMORY (for fast async transfer)
    // ========================================

    // Wave tile counts
    for (int w = 0; w < g_epi_ctx->num_waves; w++) {
        int start_tile = w * g_epi_ctx->tiles_per_wave;
        int end_tile = std::min(start_tile + g_epi_ctx->tiles_per_wave, g_epi_ctx->total_tiles);
        g_epi_ctx->h_wave_tile_counts[w] = end_tile - start_tile;
    }

    // Problem tile offsets
    g_epi_ctx->h_problem_tile_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        g_epi_ctx->h_problem_tile_offsets[e + 1] = g_epi_ctx->h_problem_tile_offsets[e] +
                                                   g_epi_ctx->expert_tiles[e];
    }

    // Problem sizes and pointers
    for (int e = 0; e < num_experts; e++) {
        int offset = g_epi_ctx->expert_offsets[e];
        g_epi_ctx->h_problem_sizes[e] = cutlass::gemm::GemmCoord(tokens_per_expert[e], N, K);
        g_epi_ctx->h_ptr_A[e] = input_ptr + offset * K;
        g_epi_ctx->h_ptr_B[e] = weight_ptr + e * K * N;
        g_epi_ctx->h_ptr_C[e] = output_ptr + offset * N;
        g_epi_ctx->h_lda[e] = K;
        g_epi_ctx->h_ldb[e] = N;
        g_epi_ctx->h_ldc[e] = N;
    }

    // ========================================
    // CREATE STREAMS (before async ops)
    // ========================================
    cudaStreamCreateWithPriority(&g_epi_ctx->compute_stream, cudaStreamNonBlocking, 0);
    cudaStreamCreateWithPriority(&g_epi_ctx->comm_stream, cudaStreamNonBlocking, -5);

    // ========================================
    // ASYNC DEVICE TRANSFERS (all using pinned memory for true async)
    // ========================================
    cudaStream_t stream = g_epi_ctx->compute_stream;

    // Wave data
    cudaMemsetAsync(g_epi_ctx->d_wave_counters, 0,
                    g_epi_ctx->num_waves * sizeof(int), stream);
    cudaMemcpyAsync(g_epi_ctx->d_wave_tile_counts, g_epi_ctx->h_wave_tile_counts,
                    g_epi_ctx->num_waves * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_problem_tile_offsets, g_epi_ctx->h_problem_tile_offsets,
                    (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Problem data
    cudaMemcpyAsync(g_epi_ctx->d_problem_sizes, g_epi_ctx->h_problem_sizes,
                    num_experts * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream);

    // Pointers
    cudaMemcpyAsync(g_epi_ctx->d_ptr_A, g_epi_ctx->h_ptr_A,
                    num_experts * sizeof(ElementInput*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ptr_B, g_epi_ctx->h_ptr_B,
                    num_experts * sizeof(ElementInput*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ptr_C, g_epi_ctx->h_ptr_C,
                    num_experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ptr_D, g_epi_ctx->h_ptr_C,
                    num_experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice, stream);

    // Strides
    cudaMemcpyAsync(g_epi_ctx->d_lda, g_epi_ctx->h_lda,
                    num_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ldb, g_epi_ctx->h_ldb,
                    num_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ldc, g_epi_ctx->h_ldc,
                    num_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_epi_ctx->d_ldd, g_epi_ctx->h_ldc,
                    num_experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    g_epi_ctx->initialized = true;
}

// ============================================================================
// Main API
// ============================================================================

std::vector<torch::Tensor> grouped_gemm_epilogue_signal(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    int tiles_per_wave,
    int max_sm_usage  // 0 = use all SMs, >0 = limit to this many SMs for GEMM
) {
    int total_tokens = input.size(0);
    int K = input.size(1);
    int N = weight.size(2);
    int num_experts = tokens_per_expert.size(0);

    auto tokens_cpu = tokens_per_expert.to(torch::kCPU).to(torch::kInt32);
    auto tokens_ptr = tokens_cpu.data_ptr<int>();

    auto input_ptr = reinterpret_cast<ElementInput*>(input.data_ptr<at::Half>());
    auto weight_ptr = reinterpret_cast<ElementInput*>(weight.data_ptr<at::Half>());
    auto output_ptr = reinterpret_cast<ElementOutput*>(output.data_ptr<at::Half>());

    init_epilogue_signal_context(tokens_ptr, num_experts, K, N, total_tokens,
                                  input_ptr, weight_ptr, output_ptr, tiles_per_wave);

    cudaStream_t stream = g_epi_ctx->compute_stream;
    int sm_count = g_epi_ctx->sm_count;

    // Limit SM usage if requested (leave SMs for NCCL)
    int effective_sm_count = (max_sm_usage > 0 && max_sm_usage < sm_count)
                             ? max_sm_usage : sm_count;

    // Create EpilogueVisitor params
    typename CustomEpilogueVisitor::Params visitor_params;
    visitor_params.elementwise = {ElementAccumulator(1.0f), ElementAccumulator(0.0f)};
    visitor_params.wave_counters = g_epi_ctx->d_wave_counters;
    visitor_params.wave_tile_counts = g_epi_ctx->d_wave_tile_counts;
    visitor_params.tiles_per_wave = g_epi_ctx->tiles_per_wave;
    visitor_params.problem_tile_offsets = g_epi_ctx->d_problem_tile_offsets;
    visitor_params.total_problems = num_experts;

    // Create kernel arguments
    // Use effective_sm_count to limit SM usage for GEMM (leave SMs for NCCL)
    typename GemmGroupedWithSignal::Arguments args(
        g_epi_ctx->d_problem_sizes,
        num_experts,
        effective_sm_count,
        visitor_params,
        g_epi_ctx->d_ptr_A,
        g_epi_ctx->d_ptr_B,
        g_epi_ctx->d_ptr_C,
        g_epi_ctx->d_ptr_D,
        g_epi_ctx->d_lda,
        g_epi_ctx->d_ldb,
        g_epi_ctx->d_ldc,
        g_epi_ctx->d_ldd,
        g_epi_ctx->h_problem_sizes
    );

    // Calculate workspace size
    size_t workspace_bytes = 0;
    // For GroupScheduleMode::kDeviceOnly, workspace is needed for tile count prefix sum
    int total_tiles = g_epi_ctx->total_tiles;
    workspace_bytes = sizeof(int) * num_experts;  // For prefix sum

    if (workspace_bytes > g_epi_ctx->workspace_size) {
        if (g_epi_ctx->workspace) {
            cudaFree(g_epi_ctx->workspace);
        }
        cudaMalloc(&g_epi_ctx->workspace, workspace_bytes);
        g_epi_ctx->workspace_size = workspace_bytes;
    }

    // Create params
    typename GemmGroupedWithSignal::Params params(args, g_epi_ctx->workspace, total_tiles);

    // Calculate grid and block dimensions
    // Use effective_sm_count to limit parallelism, leaving SMs for NCCL
    dim3 grid(effective_sm_count, 1, 1);
    dim3 block(GemmGroupedWithSignal::kThreadCount, 1, 1);

    int smem_size = sizeof(typename GemmGroupedWithSignal::SharedStorage);

    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(
            cutlass::Kernel<GemmGroupedWithSignal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }

    // Launch kernel
    cutlass::Kernel<GemmGroupedWithSignal><<<grid, block, smem_size, stream>>>(params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Return context information
    int num_waves = g_epi_ctx->num_waves;

    auto wave_info = torch::zeros({num_waves, 3}, torch::kInt32);
    auto wave_info_ptr = wave_info.data_ptr<int>();
    int tile_start = 0;
    for (int w = 0; w < num_waves; w++) {
        wave_info_ptr[w * 3 + 0] = tile_start;
        wave_info_ptr[w * 3 + 1] = g_epi_ctx->h_wave_tile_counts[w];
        wave_info_ptr[w * 3 + 2] = g_epi_ctx->tiles_per_wave;
        tile_start += g_epi_ctx->h_wave_tile_counts[w];
    }

    auto counter_ptr = torch::tensor({reinterpret_cast<int64_t>(g_epi_ctx->d_wave_counters)},
                                     torch::kInt64);

    auto tile_counts = torch::zeros({num_waves}, torch::kInt32);
    auto tile_counts_ptr = tile_counts.data_ptr<int>();
    for (int w = 0; w < num_waves; w++) {
        tile_counts_ptr[w] = g_epi_ctx->h_wave_tile_counts[w];
    }

    auto info = torch::zeros({6}, torch::kInt32);
    auto info_ptr = info.data_ptr<int>();
    info_ptr[0] = g_epi_ctx->total_tiles;
    info_ptr[1] = num_waves;
    info_ptr[2] = g_epi_ctx->tiles_per_wave;
    info_ptr[3] = sm_count;
    info_ptr[4] = num_experts;
    info_ptr[5] = reinterpret_cast<int64_t>(stream) & 0xFFFFFFFF;

    return {wave_info, counter_ptr, tile_counts, info};
}

void wait_for_wave_signal(int wave_idx, cudaStream_t comm_stream) {
    if (g_epi_ctx == nullptr || wave_idx >= g_epi_ctx->num_waves) return;

    int target = g_epi_ctx->h_wave_tile_counts[wave_idx];
    int* counter_addr = g_epi_ctx->d_wave_counters + wave_idx;

    kernel_wait_signal<<<1, 1, 0, comm_stream>>>(target, counter_addr);
}

void wait_for_wave_signal_pytorch(int wave_idx) {
    if (g_epi_ctx == nullptr || wave_idx >= g_epi_ctx->num_waves) return;
    wait_for_wave_signal(wave_idx, g_epi_ctx->comm_stream);
}

/**
 * Wait for all waves to complete in a single kernel launch.
 * This eliminates Python loop overhead by doing all waits in C++/CUDA.
 */
void wait_for_all_waves(cudaStream_t comm_stream) {
    if (g_epi_ctx == nullptr) return;

    kernel_wait_all_waves<<<1, 1, 0, comm_stream>>>(
        g_epi_ctx->num_waves,
        g_epi_ctx->d_wave_tile_counts,
        g_epi_ctx->d_wave_counters
    );
}

void wait_for_all_waves_pytorch() {
    if (g_epi_ctx == nullptr) return;
    wait_for_all_waves(g_epi_ctx->comm_stream);
}

cudaStream_t get_comm_stream() {
    if (g_epi_ctx == nullptr) return nullptr;
    return g_epi_ctx->comm_stream;
}

cudaStream_t get_compute_stream() {
    if (g_epi_ctx == nullptr) return nullptr;
    return g_epi_ctx->compute_stream;
}

void sync_comm_to_compute() {
    if (g_epi_ctx == nullptr) return;

    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, g_epi_ctx->comm_stream);
    cudaStreamWaitEvent(g_epi_ctx->compute_stream, event, 0);
    cudaEventDestroy(event);
}

torch::Tensor get_wave_boundaries_epi() {
    if (g_epi_ctx == nullptr) {
        return torch::zeros({0, 3}, torch::kInt32);
    }

    int num_waves = g_epi_ctx->num_waves;
    auto boundaries = torch::zeros({num_waves, 3}, torch::kInt32);
    auto ptr = boundaries.data_ptr<int>();

    int total_tiles = g_epi_ctx->total_tiles;
    int total_tokens = g_epi_ctx->total_tokens;

    for (int w = 0; w < num_waves; w++) {
        int wave_start_tile = w * g_epi_ctx->tiles_per_wave;
        int wave_end_tile = std::min(wave_start_tile + g_epi_ctx->h_wave_tile_counts[w],
                                      total_tiles);

        int token_start = (wave_start_tile * total_tokens) / total_tiles;
        int token_end = (wave_end_tile * total_tokens) / total_tiles;

        ptr[w * 3 + 0] = token_start;
        ptr[w * 3 + 1] = token_end - token_start;
        ptr[w * 3 + 2] = g_epi_ctx->h_wave_tile_counts[w];
    }

    return boundaries;
}

void destroy_epilogue_signal_context() {
    cleanup_epilogue_signal_context();
}

torch::Tensor get_epilogue_signal_info() {
    if (g_epi_ctx == nullptr) {
        return torch::zeros({0}, torch::kInt32);
    }

    auto info = torch::zeros({6}, torch::kInt32);
    auto ptr = info.data_ptr<int>();
    ptr[0] = g_epi_ctx->total_tiles;
    ptr[1] = g_epi_ctx->num_waves;
    ptr[2] = g_epi_ctx->tiles_per_wave;
    ptr[3] = g_epi_ctx->sm_count;
    ptr[4] = g_epi_ctx->num_experts;
    ptr[5] = g_epi_ctx->total_tokens;

    return info;
}

// ============================================================================
// NCCL Initialization for FlashOverlap-style C++ loop
// ============================================================================

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                              \
  if (r!= ncclSuccess) {                             \
    printf("NCCL failure %s:%d '%s'\n",              \
        __FILE__,__LINE__,ncclGetErrorString(r));    \
  }                                                  \
} while(0)

/**
 * Get NCCL unique ID for distributed initialization.
 * Rank 0 should call this and broadcast the ID to all ranks.
 */
std::vector<int64_t> get_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    // Convert to vector of int64 for Python
    std::vector<int64_t> id_vec(NCCL_UNIQUE_ID_BYTES / sizeof(int64_t) + 1);
    memcpy(id_vec.data(), id.internal, NCCL_UNIQUE_ID_BYTES);

    return id_vec;
}

/**
 * Initialize NCCL communicator independently (FlashOverlap style).
 * This creates our own NCCL comm instead of using PyTorch's.
 */
void init_nccl_comm_with_id(const std::vector<int64_t>& nccl_id, int rank, int world_size) {
    if (g_epi_ctx == nullptr) {
        std::cerr << "Error: Call grouped_gemm_epilogue_signal first" << std::endl;
        return;
    }

    // Skip if already initialized with same rank/world_size
    if (g_epi_ctx->nccl_initialized &&
        g_epi_ctx->nccl_rank == rank &&
        g_epi_ctx->nccl_world_size == world_size) {
        return;  // Already initialized, skip
    }

    // If previously initialized with different config, destroy first (only if we own it)
    if (g_epi_ctx->nccl_initialized && g_epi_ctx->nccl_owns_comm && g_epi_ctx->nccl_comm != nullptr) {
        ncclCommDestroy(g_epi_ctx->nccl_comm);
        g_epi_ctx->nccl_initialized = false;
    }

    // Convert vector back to ncclUniqueId
    ncclUniqueId id;
    memcpy(id.internal, nccl_id.data(), NCCL_UNIQUE_ID_BYTES);

    // Initialize NCCL communicator
    NCCL_CHECK(ncclCommInitRank(&g_epi_ctx->nccl_comm, world_size, id, rank));

    g_epi_ctx->nccl_rank = rank;
    g_epi_ctx->nccl_world_size = world_size;
    g_epi_ctx->nccl_initialized = true;
    g_epi_ctx->nccl_owns_comm = true;  // We created it, we own it
}

/**
 * Initialize using existing ncclComm_t pointer (from PyTorch).
 */
void init_nccl_comm(int64_t nccl_comm_ptr, int rank, int world_size) {
    if (g_epi_ctx == nullptr) {
        std::cerr << "Error: Call grouped_gemm_epilogue_signal first" << std::endl;
        return;
    }

    g_epi_ctx->nccl_comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    g_epi_ctx->nccl_rank = rank;
    g_epi_ctx->nccl_world_size = world_size;
    g_epi_ctx->nccl_initialized = true;
    g_epi_ctx->nccl_owns_comm = false;  // Borrowed from PyTorch, don't destroy
}

/**
 * FlashOverlap-style overlap: Queue all wait kernels + NCCL in C++ loop.
 *
 * This is the key optimization from FlashOverlap:
 * - Launch GEMM on compute_stream
 * - In a tight C++ loop, queue (wait_kernel + NCCL) pairs on comm_stream
 * - No Python in the critical path
 * - NCCL is pre-queued and starts immediately when signal arrives
 *
 * Timeline:
 *   Compute: [GEMM tile0][tile1][tile2]...[tileN]
 *                   |signal     |signal
 *                   v           v
 *   Comm:    [wait]→[NCCL0][wait]→[NCCL1]...
 *            ^pre-queued^   ^pre-queued^
 */
void queue_flash_overlap_alltoall(
    void* send_ptr,
    void* recv_ptr,
    int waves_per_comm  // Group N waves per NCCL call
) {
    if (g_epi_ctx == nullptr || !g_epi_ctx->nccl_initialized) {
        std::cerr << "Error: NCCL not initialized" << std::endl;
        return;
    }

    int num_waves = g_epi_ctx->num_waves;
    int total_tokens = g_epi_ctx->total_tokens;
    int total_tiles = g_epi_ctx->total_tiles;
    int N = g_epi_ctx->N;
    int world_size = g_epi_ctx->nccl_world_size;

    cudaStream_t comm_stream = g_epi_ctx->comm_stream;
    ncclComm_t comm = g_epi_ctx->nccl_comm;

    half* send = reinterpret_cast<half*>(send_ptr);
    half* recv = reinterpret_cast<half*>(recv_ptr);

    // Queue wait + NCCL pairs in C++ loop (no Python overhead!)
    for (int comm_idx = 0; comm_idx < num_waves; comm_idx += waves_per_comm) {
        // Calculate wave range for this communication
        int wave_end = std::min(comm_idx + waves_per_comm - 1, num_waves - 1);

        // Calculate cumulative target: sum of tiles from wave 0 to wave_end
        int cumulative_target = 0;
        for (int w = 0; w <= wave_end; w++) {
            cumulative_target += g_epi_ctx->h_wave_tile_counts[w];
        }

        // Launch wait kernel for cumulative completion
        // We wait for wave_end's cumulative tile count
        kernel_wait_signal<<<1, 1, 0, comm_stream>>>(
            cumulative_target,
            g_epi_ctx->d_wave_counters + wave_end
        );

        // Calculate token range for this group of waves
        int wave_start_tile = comm_idx * g_epi_ctx->tiles_per_wave;
        int wave_end_tile = std::min(
            (wave_end + 1) * g_epi_ctx->tiles_per_wave,
            total_tiles
        );

        int token_start = (wave_start_tile * total_tokens) / total_tiles;
        int token_end = (wave_end_tile * total_tokens) / total_tiles;
        int token_count = token_end - token_start;

        if (token_count > 0) {
            // Queue NCCL AllToAll for this token range
            // AllToAll: each rank sends token_count/world_size to each other rank
            size_t send_count_per_rank = (token_count / world_size) * N;

            NCCL_CHECK(ncclGroupStart());
            for (int peer = 0; peer < world_size; peer++) {
                size_t send_offset = token_start * N + peer * send_count_per_rank;
                size_t recv_offset = token_start * N + peer * send_count_per_rank;

                NCCL_CHECK(ncclSend(
                    send + send_offset,
                    send_count_per_rank,
                    ncclFloat16,
                    peer,
                    comm,
                    comm_stream
                ));
                NCCL_CHECK(ncclRecv(
                    recv + recv_offset,
                    send_count_per_rank,
                    ncclFloat16,
                    peer,
                    comm,
                    comm_stream
                ));
            }
            NCCL_CHECK(ncclGroupEnd());
        }
    }
}

/**
 * Simplified version: Queue wait + NCCL AllToAllSingle in C++ loop.
 * Uses ncclAllToAll for simpler implementation.
 */
void queue_flash_overlap_alltoall_single(
    void* send_ptr,
    void* recv_ptr,
    int waves_per_comm
) {
    if (g_epi_ctx == nullptr || !g_epi_ctx->nccl_initialized) {
        std::cerr << "Error: NCCL not initialized" << std::endl;
        return;
    }

    int num_waves = g_epi_ctx->num_waves;
    int total_tokens = g_epi_ctx->total_tokens;
    int total_tiles = g_epi_ctx->total_tiles;
    int N = g_epi_ctx->N;
    int world_size = g_epi_ctx->nccl_world_size;

    cudaStream_t comm_stream = g_epi_ctx->comm_stream;
    ncclComm_t comm = g_epi_ctx->nccl_comm;

    half* send = reinterpret_cast<half*>(send_ptr);
    half* recv = reinterpret_cast<half*>(recv_ptr);

    int num_comms = (num_waves + waves_per_comm - 1) / waves_per_comm;

    // Queue wait + NCCL pairs in C++ loop
    for (int c = 0; c < num_comms; c++) {
        int wave_start = c * waves_per_comm;
        int wave_end = std::min(wave_start + waves_per_comm - 1, num_waves - 1);

        // Calculate target: number of tiles completed by wave_end
        int target = g_epi_ctx->h_wave_tile_counts[wave_end];

        // Queue wait kernel
        kernel_wait_signal<<<1, 1, 0, comm_stream>>>(
            target,
            g_epi_ctx->d_wave_counters + wave_end
        );

        // Calculate token range
        int tile_start = wave_start * g_epi_ctx->tiles_per_wave;
        int tile_end = std::min((wave_end + 1) * g_epi_ctx->tiles_per_wave, total_tiles);

        int token_start = (tile_start * total_tokens) / total_tiles;
        int token_end = (tile_end * total_tokens) / total_tiles;
        int token_count = token_end - token_start;

        if (token_count > 0) {
            // Use ncclSend/ncclRecv pairs for AllToAll pattern
            size_t count_per_rank = (token_count * N) / world_size;

            NCCL_CHECK(ncclGroupStart());
            for (int peer = 0; peer < world_size; peer++) {
                half* send_buf = send + token_start * N + peer * count_per_rank;
                half* recv_buf = recv + token_start * N + peer * count_per_rank;

                NCCL_CHECK(ncclSend(send_buf, count_per_rank, ncclFloat16, peer, comm, comm_stream));
                NCCL_CHECK(ncclRecv(recv_buf, count_per_rank, ncclFloat16, peer, comm, comm_stream));
            }
            NCCL_CHECK(ncclGroupEnd());
        }
    }
}

/**
 * Get NCCL comm from PyTorch distributed ProcessGroup.
 * This extracts the ncclComm_t from torch.distributed.
 */
int64_t get_nccl_comm_from_group(int64_t pg_ptr) {
    // Note: This is a placeholder. In practice, you'd need to use
    // torch::distributed's internal APIs to get the ncclComm_t.
    // For now, we accept the comm directly from Python.
    return pg_ptr;
}

// ============================================================================
// Monitoring Mode - Hint Collection (FlashOverlap style)
// ============================================================================

// Global counter for monitoring mode (device memory)
static int* g_monitoring_counter = nullptr;

/**
 * Kernel to reset the monitoring counter
 */
__global__ void kernel_reset_monitoring_counter(int* counter) {
    *counter = 0;
}

/**
 * Run GroupedGEMM in monitoring mode to collect tile completion order.
 * This is used for FlashOverlap-style hint collection.
 *
 * The hint_buffer will contain, for each tile index, the order in which
 * that tile completed (0 = first completed, 1 = second, etc.)
 *
 * @param input       Input tensor [total_tokens, K]
 * @param weight      Weight tensor [num_experts, K, N]
 * @param output      Output tensor [total_tokens, N]
 * @param tokens_per_expert  Number of tokens per expert
 * @param hint_buffer Output buffer to store completion order [total_tiles]
 * @return total_tiles - the number of tiles computed
 */
int grouped_gemm_with_monitoring(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    torch::Tensor hint_buffer
) {
    int total_tokens = input.size(0);
    int K = input.size(1);
    int N = weight.size(2);
    int num_experts = tokens_per_expert.size(0);

    auto tokens_cpu = tokens_per_expert.to(torch::kCPU).to(torch::kInt32);
    auto tokens_ptr = tokens_cpu.data_ptr<int>();

    auto input_ptr = reinterpret_cast<ElementInput*>(input.data_ptr<at::Half>());
    auto weight_ptr = reinterpret_cast<ElementInput*>(weight.data_ptr<at::Half>());
    auto output_ptr = reinterpret_cast<ElementOutput*>(output.data_ptr<at::Half>());
    auto hint_ptr = hint_buffer.data_ptr<int>();

    // Calculate total tiles
    int total_tiles = 0;
    std::vector<int> problem_tile_offsets(num_experts + 1);
    problem_tile_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
        int tiles_e = calculate_expert_tiles_epi(tokens_ptr[e], N);
        total_tiles += tiles_e;
        problem_tile_offsets[e + 1] = total_tiles;
    }

    // Allocate and reset monitoring counter if needed
    if (g_monitoring_counter == nullptr) {
        cudaMalloc(&g_monitoring_counter, sizeof(int));
    }
    kernel_reset_monitoring_counter<<<1, 1>>>(g_monitoring_counter);
    cudaDeviceSynchronize();

    // Initialize context if needed (we need the basic setup)
    int sm_count = get_sm_count_epi();
    int tiles_per_wave = sm_count;  // Use SM count for monitoring

    init_epilogue_signal_context(tokens_ptr, num_experts, K, N, total_tokens,
                                  input_ptr, weight_ptr, output_ptr, tiles_per_wave);

    cudaStream_t stream = g_epi_ctx->compute_stream;

    // Create EpilogueVisitor params with MONITORING MODE enabled
    typename CustomEpilogueVisitor::Params visitor_params;
    visitor_params.elementwise = {ElementAccumulator(1.0f), ElementAccumulator(0.0f)};
    visitor_params.wave_counters = nullptr;  // Don't use wave counters in monitoring mode
    visitor_params.wave_tile_counts = nullptr;
    visitor_params.tiles_per_wave = tiles_per_wave;
    visitor_params.problem_tile_offsets = g_epi_ctx->d_problem_tile_offsets;
    visitor_params.total_problems = num_experts;
    visitor_params.monitor_mode = true;
    visitor_params.hint_buffer = hint_ptr;
    visitor_params.reorder_array = nullptr;
    visitor_params.completion_counter = g_monitoring_counter;

    // Create kernel arguments
    typename GemmGroupedWithSignal::Arguments args(
        g_epi_ctx->d_problem_sizes,
        num_experts,
        sm_count,
        visitor_params,
        g_epi_ctx->d_ptr_A,
        g_epi_ctx->d_ptr_B,
        g_epi_ctx->d_ptr_C,
        g_epi_ctx->d_ptr_D,
        g_epi_ctx->d_lda,
        g_epi_ctx->d_ldb,
        g_epi_ctx->d_ldc,
        g_epi_ctx->d_ldd,
        g_epi_ctx->h_problem_sizes
    );

    // Calculate workspace size
    size_t workspace_bytes = sizeof(int) * num_experts;

    if (workspace_bytes > g_epi_ctx->workspace_size) {
        if (g_epi_ctx->workspace) cudaFree(g_epi_ctx->workspace);
        cudaMalloc(&g_epi_ctx->workspace, workspace_bytes);
        g_epi_ctx->workspace_size = workspace_bytes;
    }

    typename GemmGroupedWithSignal::Params params(args, g_epi_ctx->workspace, total_tiles);

    // Calculate grid and block dimensions
    dim3 grid(sm_count, 1, 1);
    dim3 block(GemmGroupedWithSignal::kThreadCount, 1, 1);

    int smem_size = sizeof(typename GemmGroupedWithSignal::SharedStorage);

    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(
            cutlass::Kernel<GemmGroupedWithSignal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }

    // Launch kernel
    cutlass::Kernel<GemmGroupedWithSignal><<<grid, block, smem_size, stream>>>(params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Monitoring kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure hints are collected
    cudaStreamSynchronize(stream);

    return total_tiles;
}

/**
 * Get the total number of tiles for a given configuration.
 * Useful for allocating hint buffer before calling monitoring.
 */
int get_total_tiles(
    torch::Tensor tokens_per_expert,
    int N
) {
    int num_experts = tokens_per_expert.size(0);
    auto tokens_cpu = tokens_per_expert.to(torch::kCPU).to(torch::kInt32);
    auto tokens_ptr = tokens_cpu.data_ptr<int>();

    int total_tiles = 0;
    for (int e = 0; e < num_experts; e++) {
        total_tiles += calculate_expert_tiles_epi(tokens_ptr[e], N);
    }
    return total_tiles;
}

// ============================================================================
// Fully Fused FC2 + AllToAll Overlap API
// ============================================================================

/**
 * FC2 + AllToAll with Full C++ Overlap (Zero Python Loop)
 *
 * This is the BEST performing version that achieves TRUE overlap between
 * FC2 computation and AllToAll communication, all in C++.
 *
 * Key Optimizations:
 * 1. Single Python call - no Python in the critical path
 * 2. FC2 GEMM with wave signaling on compute_stream
 * 3. Wait + NCCL pre-queued on comm_stream (FlashOverlap style)
 * 4. All synchronization done via CUDA events
 *
 * Timeline:
 *   FC2:      [wave0][wave1][wave2][wave3]...
 *   AllToAll:       [comm0]     [comm1]...
 *                      ↑ Pre-queued, starts immediately when wave signals
 *
 * @param input             Input tensor [total_tokens, K]
 * @param weight            Weight tensor [num_experts, K, N]
 * @param output            Output tensor [total_tokens, N] - FC2 output, also used as AllToAll send
 * @param recv_buffer       Receive buffer [total_tokens, N] - AllToAll receive
 * @param tokens_per_expert Number of tokens per expert
 * @param tiles_per_wave    Tiles per wave (0 = use SM count)
 * @param waves_per_comm    Number of waves to group per communication
 * @return output tensor
 */
// Static event for sync (avoid repeated create/destroy)
static cudaEvent_t g_fc2_comm_done_event = nullptr;

/**
 * FC2 + AllToAll with Full C++ Overlap
 *
 * This implementation EXACTLY matches the Python loop version:
 * 1. Launch FC2 GEMM with wave signaling
 * 2. For each wave group: wait for signal, then start NCCL
 * 3. NCCL operations are queued on comm_stream (overlap with compute)
 */
torch::Tensor fc2_alltoall_overlap(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor recv_buffer,
    torch::Tensor tokens_per_expert,
    int tiles_per_wave,
    int waves_per_comm
) {
    // Validate NCCL is initialized
    if (g_epi_ctx == nullptr || !g_epi_ctx->nccl_initialized) {
        throw std::runtime_error("NCCL not initialized. Call init_fc2_alltoall_nccl first.");
    }

    int total_tokens = input.size(0);
    int K = input.size(1);
    int N = weight.size(2);
    int num_experts = tokens_per_expert.size(0);

    // Use cached token counts if config unchanged (avoid GPU sync)
    std::vector<int> tokens_vec(num_experts);
    int actual_tiles_per_wave = (tiles_per_wave > 0) ? tiles_per_wave : get_sm_count_epi();

    bool can_reuse = g_epi_ctx->initialized &&
                     g_epi_ctx->num_experts == num_experts &&
                     g_epi_ctx->K == K && g_epi_ctx->N == N &&
                     g_epi_ctx->total_tokens == total_tokens &&
                     g_epi_ctx->tiles_per_wave == actual_tiles_per_wave;

    if (can_reuse) {
        for (int e = 0; e < num_experts; e++) {
            tokens_vec[e] = g_epi_ctx->tokens_per_expert_vec[e];
        }
    } else {
        // First call or config changed - need GPU sync (only happens once)
        auto tokens_cpu = tokens_per_expert.to(torch::kCPU).to(torch::kInt32);
        auto tokens_ptr = tokens_cpu.data_ptr<int>();
        for (int e = 0; e < num_experts; e++) {
            tokens_vec[e] = tokens_ptr[e];
        }
    }

    auto input_ptr = reinterpret_cast<ElementInput*>(input.data_ptr<at::Half>());
    auto weight_ptr = reinterpret_cast<ElementInput*>(weight.data_ptr<at::Half>());
    auto output_ptr = reinterpret_cast<ElementOutput*>(output.data_ptr<at::Half>());
    auto recv_ptr = reinterpret_cast<ElementOutput*>(recv_buffer.data_ptr<at::Half>());

    // Initialize/update context (resets wave counters on compute_stream)
    init_epilogue_signal_context(tokens_vec.data(), num_experts, K, N, total_tokens,
                                  input_ptr, weight_ptr, output_ptr, tiles_per_wave);

    cudaStream_t compute_stream = g_epi_ctx->compute_stream;
    cudaStream_t comm_stream = g_epi_ctx->comm_stream;
    int sm_count = g_epi_ctx->sm_count;

    // CRITICAL: Sync comm_stream with compute_stream to ensure wave counters are reset
    // before we start queueing wait kernels
    static cudaEvent_t reset_done_event = nullptr;
    if (reset_done_event == nullptr) {
        cudaEventCreateWithFlags(&reset_done_event, cudaEventDisableTiming);
    }
    cudaEventRecord(reset_done_event, compute_stream);
    cudaStreamWaitEvent(comm_stream, reset_done_event, 0);

    // ========================================
    // Step 1: Launch FC2 GEMM with wave signaling on compute_stream
    // ========================================
    typename CustomEpilogueVisitor::Params visitor_params;
    visitor_params.elementwise = {ElementAccumulator(1.0f), ElementAccumulator(0.0f)};
    visitor_params.wave_counters = g_epi_ctx->d_wave_counters;
    visitor_params.wave_tile_counts = g_epi_ctx->d_wave_tile_counts;
    visitor_params.tiles_per_wave = g_epi_ctx->tiles_per_wave;
    visitor_params.problem_tile_offsets = g_epi_ctx->d_problem_tile_offsets;
    visitor_params.total_problems = num_experts;

    typename GemmGroupedWithSignal::Arguments args(
        g_epi_ctx->d_problem_sizes,
        num_experts,
        sm_count,
        visitor_params,
        g_epi_ctx->d_ptr_A,
        g_epi_ctx->d_ptr_B,
        g_epi_ctx->d_ptr_C,
        g_epi_ctx->d_ptr_D,
        g_epi_ctx->d_lda,
        g_epi_ctx->d_ldb,
        g_epi_ctx->d_ldc,
        g_epi_ctx->d_ldd,
        g_epi_ctx->h_problem_sizes
    );

    size_t workspace_bytes = sizeof(int) * num_experts;
    if (workspace_bytes > g_epi_ctx->workspace_size) {
        if (g_epi_ctx->workspace) cudaFree(g_epi_ctx->workspace);
        cudaMalloc(&g_epi_ctx->workspace, workspace_bytes);
        g_epi_ctx->workspace_size = workspace_bytes;
    }

    typename GemmGroupedWithSignal::Params params(args, g_epi_ctx->workspace, g_epi_ctx->total_tiles);

    dim3 grid(sm_count, 1, 1);
    dim3 block(GemmGroupedWithSignal::kThreadCount, 1, 1);
    int smem_size = sizeof(typename GemmGroupedWithSignal::SharedStorage);

    if (smem_size >= (48 << 10)) {
        cudaFuncSetAttribute(
            cutlass::Kernel<GemmGroupedWithSignal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }

    // Launch FC2 GEMM
    cutlass::Kernel<GemmGroupedWithSignal><<<grid, block, smem_size, compute_stream>>>(params);

    // ========================================
    // Step 2: Queue wait + NCCL on comm_stream (FlashOverlap style)
    // This is pre-queued - NCCL starts immediately when wave completes
    // ========================================
    int num_waves = g_epi_ctx->num_waves;
    int total_tiles = g_epi_ctx->total_tiles;
    int world_size = g_epi_ctx->nccl_world_size;
    ncclComm_t nccl_comm = g_epi_ctx->nccl_comm;

    half* send = reinterpret_cast<half*>(output_ptr);
    half* recv = reinterpret_cast<half*>(recv_ptr);

    // ========================================
    // Step 2: Wave-based overlap - queue wait + NCCL for each wave group
    // Optimized: calculate boundaries on-the-fly to avoid vector allocation
    // Note: actual_tiles_per_wave is already defined above
    // ========================================
    const int* h_wave_tile_counts = g_epi_ctx->h_wave_tile_counts;
    int* d_wave_counters = g_epi_ctx->d_wave_counters;

    for (int comm_idx = 0; comm_idx < num_waves; comm_idx += waves_per_comm) {
        int wave_end = std::min(comm_idx + waves_per_comm - 1, num_waves - 1);

        // Calculate token_start for this comm group
        int wave_start_tile = comm_idx * actual_tiles_per_wave;
        int token_start = (wave_start_tile * total_tokens) / total_tiles;

        // Calculate token_end for this comm group (end of wave_end)
        int wave_end_tile = std::min((wave_end + 1) * actual_tiles_per_wave, total_tiles);
        int token_end = (wave_end_tile * total_tokens) / total_tiles;
        int token_count = token_end - token_start;

        if (token_count <= 0) continue;

        // Wait for this wave group to complete
        int target = h_wave_tile_counts[wave_end];
        kernel_wait_signal<<<1, 1, 0, comm_stream>>>(target, d_wave_counters + wave_end);

        // All-to-all on this chunk (equal split per rank)
        int tokens_per_rank = token_count / world_size;
        if (tokens_per_rank <= 0) continue;

        size_t elements_per_rank = static_cast<size_t>(tokens_per_rank) * N;

        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < world_size; peer++) {
            size_t send_offset = static_cast<size_t>(token_start + peer * tokens_per_rank) * N;
            size_t recv_offset = send_offset;  // Same offset for send and recv

            NCCL_CHECK(ncclSend(send + send_offset, elements_per_rank, ncclFloat16, peer, nccl_comm, comm_stream));
            NCCL_CHECK(ncclRecv(recv + recv_offset, elements_per_rank, ncclFloat16, peer, nccl_comm, comm_stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    // ========================================
    // Step 3: Sync comm_stream to compute_stream (use static event)
    // ========================================
    if (g_fc2_comm_done_event == nullptr) {
        cudaEventCreateWithFlags(&g_fc2_comm_done_event, cudaEventDisableTiming);
    }
    cudaEventRecord(g_fc2_comm_done_event, comm_stream);
    cudaStreamWaitEvent(compute_stream, g_fc2_comm_done_event, 0);

    return recv_buffer;
}

/**
 * Initialize NCCL for FC2+AllToAll overlap.
 * Must be called before fc2_alltoall_overlap.
 */
void init_fc2_alltoall_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id_vec) {
    // First ensure we have a context
    if (g_epi_ctx == nullptr) {
        g_epi_ctx = new EpilogueSignalContext();
    }

    // Skip if already initialized with same config
    if (g_epi_ctx->nccl_initialized &&
        g_epi_ctx->nccl_rank == rank &&
        g_epi_ctx->nccl_world_size == world_size) {
        return;
    }

    // Cleanup old comm if we own it
    if (g_epi_ctx->nccl_initialized && g_epi_ctx->nccl_owns_comm && g_epi_ctx->nccl_comm != nullptr) {
        ncclCommDestroy(g_epi_ctx->nccl_comm);
        g_epi_ctx->nccl_initialized = false;
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
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, rank));

    g_epi_ctx->nccl_comm = comm;
    g_epi_ctx->nccl_rank = rank;
    g_epi_ctx->nccl_world_size = world_size;
    g_epi_ctx->nccl_initialized = true;
    g_epi_ctx->nccl_owns_comm = true;

    // Create streams if not exists
    if (g_epi_ctx->compute_stream == nullptr) {
        cudaStreamCreateWithPriority(&g_epi_ctx->compute_stream, cudaStreamNonBlocking, 0);
    }
    if (g_epi_ctx->comm_stream == nullptr) {
        cudaStreamCreateWithPriority(&g_epi_ctx->comm_stream, cudaStreamNonBlocking, -5);
    }
}

/**
 * Get NCCL unique ID (same as alltoall_fc1 version).
 */
std::vector<int64_t> get_fc2_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    std::vector<int64_t> id_vec(16);
    memcpy(id_vec.data(), &id, sizeof(ncclUniqueId));
    return id_vec;
}

/**
 * Destroy NCCL for FC2+AllToAll.
 */
void destroy_fc2_alltoall_nccl() {
    if (g_fc2_comm_done_event != nullptr) {
        cudaEventDestroy(g_fc2_comm_done_event);
        g_fc2_comm_done_event = nullptr;
    }
    cleanup_epilogue_signal_context();
}

} // namespace fluid
