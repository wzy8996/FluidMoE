/**
 * GemmGroupedWithEpilogueVisitor - CUTLASS GroupedGEMM with custom EpilogueVisitor
 *
 * This is a modified version of CUTLASS's GemmGrouped kernel that uses
 * EpilogueWithVisitor instead of the standard Epilogue. This enables
 * FlashOverlap-style signaling in the Epilogue phase.
 *
 * Key modifications from cutlass/gemm/kernel/gemm_grouped.h:
 * - Uses EpilogueWithVisitor template parameter
 * - Passes EpilogueVisitor to epilogue instead of OutputOp
 * - Adds signal parameters for wave-level communication
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * GemmGroupedWithEpilogueVisitor
 *
 * A GroupedGEMM kernel that supports custom EpilogueVisitor for fused operations
 * like signaling, layernorm, etc.
 */
template <
    typename Mma_,                           ///! Threadblock-scoped matrix multiply-accumulate
    typename EpilogueWithVisitor_,           ///! Epilogue with visitor pattern
    typename EpilogueVisitor_,               ///! The visitor itself
    typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
    GroupScheduleMode GroupScheduleMode_,    ///! Type of scheduling to perform
    bool Transposed = false
>
struct GemmGroupedWithEpilogueVisitor {
public:
    using Mma = Mma_;
    using EpilogueWithVisitor = EpilogueWithVisitor_;
    using EpilogueVisitor = EpilogueVisitor_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
    static bool const kTransposed = Transposed;

    // Optional transpose
    using MapArguments = kernel::detail::MapArguments<
        typename Mma::IteratorA::Element,
        typename Mma::IteratorA::Layout,
        Mma::kTransformA,
        Mma::IteratorA::AccessType::kElements,
        typename Mma::IteratorB::Element,
        typename Mma::IteratorB::Layout,
        Mma::kTransformB,
        Mma::IteratorB::AccessType::kElements,
        typename Mma::LayoutC,
        kTransposed
    >;

    using ElementA = typename MapArguments::ElementA;
    using LayoutA = typename MapArguments::LayoutA;
    using ElementB = typename MapArguments::ElementB;
    using LayoutB = typename MapArguments::LayoutB;
    using ElementC = typename EpilogueVisitor::ElementOutput;
    using LayoutC = typename MapArguments::LayoutC;

    static ComplexTransform const kTransformA = MapArguments::kTransformA;
    static ComplexTransform const kTransformB = MapArguments::kTransformB;

    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = MapArguments::kAlignmentA;
    static int const kAlignmentB = MapArguments::kAlignmentB;
    static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    using ProblemVisitor = GemmGroupedProblemVisitor<
        ThreadblockShape,
        kGroupScheduleMode,
        kThreadCount,
        kThreadCount,
        kTransposed
    >;

    //
    // Structures
    //

    /// Argument structure
    struct Arguments {
        GemmCoord *problem_sizes{nullptr};
        int problem_count{0};
        int threadblock_count{0};

        // EpilogueVisitor arguments
        typename EpilogueVisitor::Params epilogue_visitor{};

        ElementA ** ptr_A{nullptr};
        ElementB ** ptr_B{nullptr};
        ElementC ** ptr_C{nullptr};
        ElementC ** ptr_D{nullptr};

        typename LayoutA::Stride::LongIndex *lda{nullptr};
        typename LayoutB::Stride::LongIndex *ldb{nullptr};
        typename LayoutC::Stride::LongIndex *ldc{nullptr};
        typename LayoutC::Stride::LongIndex *ldd{nullptr};

        // Only used by device-level operator
        GemmCoord *host_problem_sizes{nullptr};

        Arguments() = default;

        CUTLASS_HOST_DEVICE
        Arguments(
            GemmCoord *problem_sizes_,
            int problem_count_,
            int threadblock_count_,
            typename EpilogueVisitor::Params epilogue_visitor_,
            ElementA ** ptr_A_,
            ElementB ** ptr_B_,
            ElementC ** ptr_C_,
            ElementC ** ptr_D_,
            typename LayoutA::Stride::LongIndex *lda_,
            typename LayoutB::Stride::LongIndex *ldb_,
            typename LayoutC::Stride::LongIndex *ldc_,
            typename LayoutC::Stride::LongIndex *ldd_,
            GemmCoord *host_problem_sizes_ = nullptr
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

    /// Parameters structure
    struct Params {
        typename ProblemVisitor::Params problem_visitor{};
        int threadblock_count{0};

        typename EpilogueVisitor::Params epilogue_visitor{};

        ElementA ** ptr_A{nullptr};
        ElementB ** ptr_B{nullptr};
        ElementC ** ptr_C{nullptr};
        ElementC ** ptr_D{nullptr};

        typename LayoutA::Stride::LongIndex *lda{nullptr};
        typename LayoutB::Stride::LongIndex *ldb{nullptr};
        typename LayoutC::Stride::LongIndex *ldc{nullptr};
        typename LayoutC::Stride::LongIndex *ldd{nullptr};

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

        CUTLASS_HOST_DEVICE
        void update(Arguments const &args, void *workspace = nullptr, int tile_count = 0) {
            problem_visitor = typename ProblemVisitor::Params(
                args.problem_sizes, args.problem_count, workspace, tile_count);
            threadblock_count = args.threadblock_count;
            epilogue_visitor = args.epilogue_visitor;
            ptr_A = args.ptr_A;
            ptr_B = args.ptr_B;
            ptr_C = args.ptr_C;
            ptr_D = args.ptr_D;
            lda = args.lda;
            ldb = args.ldb;
            ldc = args.ldc;
            ldd = args.ldd;
        }
    };

    /// Shared memory storage structure
    struct SharedStorage {
        union {
            typename Mma::SharedStorage main_loop;
            struct {
                typename EpilogueWithVisitor::SharedStorage epilogue;
                typename EpilogueVisitor::SharedStorage visitor;
            } epilogue;
        } kernel;

        typename ProblemVisitor::SharedStorage problem_visitor;
    };

public:
    CUTLASS_DEVICE
    GemmGroupedWithEpilogueVisitor() {}

    static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
        return Status::kSuccess;
    }

    static Status can_implement(Arguments const &args) {
        return Status::kSuccess;
    }

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        using ElementA = typename Mma::IteratorA::Element;
        using LayoutA = typename Mma::IteratorA::Layout;
        using ElementB = typename Mma::IteratorB::Element;
        using LayoutB = typename Mma::IteratorB::Layout;
        using ElementC = typename EpilogueVisitor::ElementOutput;
        using LayoutC = layout::RowMajor;

        // Problem visitor
        ProblemVisitor problem_visitor(
            params.problem_visitor,
            shared_storage.problem_visitor,
            blockIdx.x);

        // Outer 'persistent' loop to iterate over tiles
        while (problem_visitor.next_tile()) {
            GemmCoord problem_size = problem_visitor.problem_size();
            int32_t problem_idx = problem_visitor.problem_index();
            int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

            GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

            cutlass::gemm::GemmCoord threadblock_offset(
                int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
                int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
                0);

            // Load element pointers
            ElementA *ptr_A = reinterpret_cast<ElementA *>(
                kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]);
            typename LayoutA::LongIndex ldm_A =
                kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx];

            ElementB *ptr_B = reinterpret_cast<ElementB *>(
                kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]);
            typename LayoutB::LongIndex ldm_B =
                kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx];

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{threadblock_offset.m(), 0};
            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            int thread_idx = threadIdx.x;

            // Construct iterators to A and B operands
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

            int warp_idx = canonical_warp_idx_sync();
            int lane_idx = threadIdx.x % 32;

            // Matrix multiply phase
            Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            __syncthreads();

            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            //
            // Epilogue with Visitor
            //

            // Get output pointers for this problem
            ElementC *ptr_C = params.ptr_C[problem_idx];
            ElementC *ptr_D = params.ptr_D[problem_idx];

            // Create visitor with problem-specific parameters
            // We need to create a modified params that includes the current problem info
            typename EpilogueVisitor::Params visitor_params = params.epilogue_visitor;
            visitor_params.ptr_C = ptr_C;
            visitor_params.ptr_D = ptr_D;
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

            // Construct the epilogue
            EpilogueWithVisitor epilogue(
                shared_storage.kernel.epilogue.epilogue,
                thread_idx,
                warp_idx,
                lane_idx);

            // Execute the epilogue operator
            epilogue(epilogue_visitor, accumulators);

            // Next tile
            problem_visitor.advance(gridDim.x);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
