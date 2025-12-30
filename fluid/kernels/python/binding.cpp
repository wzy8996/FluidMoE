/**
 * PyTorch C++ Extension bindings for Fluid kernels
 *
 * GroupedGEMM operations and Fused FC2+AllToAll
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>

namespace fluid {

// Declarations from grouped_gemm.cu
torch::Tensor grouped_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    bool trans_a,
    bool trans_b
);

torch::Tensor grouped_gemm_dw_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor tokens_per_expert,
    int M,
    int N
);

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

torch::Tensor get_chunk_info(
    torch::Tensor tokens_per_expert,
    int num_chunks,
    int chunk_idx
);

torch::Tensor compute_chunk_boundaries(
    torch::Tensor tokens_per_expert,
    int num_chunks
);

torch::Tensor grouped_gemm_with_gather(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor indices,
    torch::Tensor tokens_per_expert,
    bool trans_b
);

std::vector<torch::Tensor> grouped_gemm_dx_chunked(
    torch::Tensor grad_fc2,
    torch::Tensor probs,
    torch::Tensor act_deriv,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor chunk_indices,
    torch::Tensor chunk_offsets,
    int num_chunks,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
);

std::vector<torch::Tensor> grouped_gemm_dx_fused(
    torch::Tensor grad_fc2,
    torch::Tensor probs,
    torch::Tensor act_deriv,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor chunk_indices,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
);

std::vector<torch::Tensor> grouped_gemm_dx_all_chunks(
    torch::Tensor grad_fc2,
    torch::Tensor probs,
    torch::Tensor act_deriv,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor all_chunk_indices,
    torch::Tensor chunk_offsets,
    int num_chunks,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
);

std::vector<torch::Tensor> grouped_gemm_dx_pipelined(
    torch::Tensor grad_fc2,
    torch::Tensor probs,
    torch::Tensor act_deriv,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor all_chunk_indices,
    torch::Tensor chunk_offsets,
    int num_chunks,
    c10::optional<torch::Tensor> act_val,
    c10::optional<torch::Tensor> x_2
);

void wait_cuda_event(int64_t event_ptr, int64_t stream_ptr);

void destroy_cuda_events(torch::Tensor event_ptrs);

// Declarations from grouped_gemm_epilogue_signal.cu (True Epilogue-Level Signaling)
std::vector<torch::Tensor> grouped_gemm_epilogue_signal(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    int tiles_per_wave,
    int max_sm_usage = 0  // 0 = use all SMs, >0 = limit GEMM to this many SMs
);

void wait_for_wave_signal_pytorch(int wave_idx);
void wait_for_all_waves_pytorch();

cudaStream_t get_comm_stream();

cudaStream_t get_compute_stream();

void sync_comm_to_compute();

torch::Tensor get_wave_boundaries_epi();

void destroy_epilogue_signal_context();

torch::Tensor get_epilogue_signal_info();

// Declarations from alltoall_fc1_signal.cu (AllToAll + FC1 Overlap)
void init_alltoall_fc1_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id);
void destroy_alltoall_fc1_nccl();

// Compute indices for Local-First overlap (call once, reuse)
std::vector<torch::Tensor> compute_localfirst_indices(
    torch::Tensor tokens_per_expert,
    torch::Tensor self_tokens_per_expert,
    torch::Tensor num_global_tokens_per_local_expert,
    int my_rank,
    torch::Device device
);

// Local-First overlap: Self FC1 || AllToAll -> Remote FC1 (BEST for most cases)
std::vector<torch::Tensor> alltoall_fc1_localfirst(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    std::vector<int64_t> input_splits_vec,
    std::vector<int64_t> output_splits_vec,
    int64_t self_input_offset,
    int64_t self_input_count,
    torch::Tensor self_tokens_per_expert,
    torch::Tensor sort_indices,
    torch::Tensor remote_input_indices,
    torch::Tensor self_output_indices,
    torch::Tensor remote_output_indices,
    torch::Tensor remote_tokens_per_expert,
    int my_rank
);

// Per-Peer Pipelined: Finer-grained overlap (useful when AllToAll > Self FC1)
std::vector<torch::Tensor> alltoall_fc1_pipelined(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    std::vector<int64_t> input_splits_vec,
    std::vector<int64_t> output_splits_vec,
    int64_t self_input_offset,
    int64_t self_input_count,
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,
    std::vector<int64_t> peer_token_counts,
    std::vector<int64_t> num_global_flat,
    int num_local_experts,
    int my_rank,
    bool serialize_peer_fc1
);

// Per-Peer Pipelined WITHOUT reordering (for use with FC2+AllToAll pipelined)
std::vector<torch::Tensor> alltoall_fc1_pipelined_no_reorder(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    std::vector<int64_t> input_splits_vec,
    std::vector<int64_t> output_splits_vec,
    int64_t self_input_offset,
    int64_t self_input_count,
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,
    std::vector<int64_t> peer_token_counts,
    int my_rank,
    bool serialize_peer_fc1
);

// Declarations from fc2_alltoall_signal.cu (FC2 + AllToAll Overlap)
void init_fc2_alltoall_pipelined_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id);
void destroy_fc2_alltoall_pipelined_nccl();
std::vector<int64_t> get_fc2_alltoall_pipelined_nccl_unique_id();

torch::Tensor fc2_alltoall_pipelined(
    torch::Tensor input,
    torch::Tensor fc2_weight,
    int64_t self_token_count,
    std::vector<int64_t> peer_token_counts,
    torch::Tensor self_tokens_per_expert,
    std::vector<torch::Tensor> peer_tokens_per_expert_vec,
    std::vector<int64_t> send_splits,
    std::vector<int64_t> recv_splits,
    torch::Tensor reorder_indices,
    int my_rank
);

// Declarations from moe_forward_fused.cu
std::vector<int64_t> get_moe_fused_nccl_unique_id();
void init_moe_fused_nccl(int rank, int world_size, std::vector<int64_t> nccl_id_vec);
void destroy_moe_fused_nccl();

// Direct output GEMM (for overlap without copy overhead)
void grouped_gemm_to_output(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    int64_t output_offset
);

void grouped_gemm_gelu_to_output(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    int64_t output_offset
);

// True GroupedGEMM - single kernel for all experts (drop-in replacement)
torch::Tensor grouped_gemm_true_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor tokens_per_expert,
    int64_t total_tokens,
    bool trans_a,
    bool trans_b
);

// Fused AllToAll + FC1 + Activation with compute-communication overlap
// Returns: (fc1_output, segment_sizes, dispatched_input, fc1_pre_activation)
// Note: probs multiplication happens in Python at unpermute stage (standard Megatron behavior)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_alltoall_fc1_fused(
    torch::Tensor permuted_tokens,
    torch::Tensor fc1_weight,
    int64_t self_input_offset,
    int64_t self_input_count,
    std::vector<int64_t> send_splits,
    std::vector<int64_t> recv_splits,
    std::vector<int64_t> peer_token_counts,
    std::vector<int> h_self_tokens_per_expert,
    std::vector<std::vector<int>> h_peer_tokens_per_expert_all,
    int activation_type = 0  // 0=GELU, 1=SiLU, 2=ReLU, 3=None
);

torch::Tensor moe_fc2_alltoall_fused(
    torch::Tensor fc1_output,
    torch::Tensor fc2_weight,
    torch::Tensor segment_sizes,
    std::vector<int64_t> original_send_splits,
    std::vector<int> h_self_tokens_per_expert,
    std::vector<std::vector<int>> h_peer_tokens_per_expert_all
);

// Fused AllToAll + FC1 + Activation with Expert-Major Output
// Returns: (fc1_output, segment_sizes, dispatched_input, fc1_pre_act, reorder_indices, inverse_indices)
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
    int activation_type = 0  // 0=GELU, 1=SiLU, 2=ReLU, 3=None
);

// Monitoring mode - Hint collection (FlashOverlap style)
int grouped_gemm_with_monitoring(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    torch::Tensor hint_buffer
);

int get_total_tiles(
    torch::Tensor tokens_per_expert,
    int N
);

// FlashOverlap-style NCCL integration
void init_nccl_comm(int64_t nccl_comm_ptr, int rank, int world_size);

// Independent NCCL initialization (FlashOverlap style)
std::vector<int64_t> get_nccl_unique_id();
void init_nccl_comm_with_id(const std::vector<int64_t>& nccl_id, int rank, int world_size);

void queue_flash_overlap_alltoall(
    void* send_ptr,
    void* recv_ptr,
    int waves_per_comm
);

void queue_flash_overlap_alltoall_single(
    void* send_ptr,
    void* recv_ptr,
    int waves_per_comm
);

// Fully fused FC2+AllToAll API
torch::Tensor fc2_alltoall_overlap(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor recv_buffer,
    torch::Tensor tokens_per_expert,
    int tiles_per_wave,
    int waves_per_comm
);
void init_fc2_alltoall_nccl(int rank, int world_size, const std::vector<int64_t>& nccl_id_vec);
std::vector<int64_t> get_fc2_nccl_unique_id();
void destroy_fc2_alltoall_nccl();

// FlashOverlap-style tuning functions
// TODO: Re-implement these functions
// torch::Tensor compute_reorder_from_hints(torch::Tensor hints, int tiles_per_wave);
// torch::Tensor compute_inverse_ra(torch::Tensor ra);
// std::vector<int64_t> get_search_result(
//     int total_tiles,
//     int sm_count,
//     float compute_time_per_tile_us,
//     int elements_per_tile,
//     bool use_predictive
// );
// int get_compute_sms(int total_sms, int nccl_sms);

} // namespace fluid

PYBIND11_MODULE(fluid_kernels, m) {
    m.doc() = "Fluid CUDA kernels - GroupedGEMM for MoE";

    // Grouped GEMM for MoE
    m.def("grouped_gemm", &fluid::grouped_gemm_forward,
          "Grouped GEMM for MoE expert computation\n"
          "C[i] = A[i] @ B[i] (with optional transposes)\n"
          "Args:\n"
          "  A: Input tensor [total_tokens, K] or [K, total_tokens] if trans_a\n"
          "  B: Weight tensor [num_experts, K, N] or [num_experts, N, K] if trans_b\n"
          "  tokens_per_expert: Number of tokens per expert [num_experts]\n"
          "  trans_a: Whether to transpose A\n"
          "  trans_b: Whether to transpose B",
          py::arg("A"),
          py::arg("B"),
          py::arg("tokens_per_expert"),
          py::arg("trans_a") = false,
          py::arg("trans_b") = false);

    // Grouped GEMM for weight gradient (dW)
    m.def("grouped_gemm_dw", &fluid::grouped_gemm_dw_forward,
          "Grouped GEMM for weight gradient computation\n"
          "dW[i] = A[tokens_i]^T @ B[tokens_i]\n"
          "Args:\n"
          "  A: Input activations [total_tokens, M]\n"
          "  B: Gradients [total_tokens, N]\n"
          "  tokens_per_expert: Number of tokens per expert [num_experts]\n"
          "  M: Hidden dimension\n"
          "  N: FFN dimension\n"
          "Returns: Weight gradients [num_experts, M, N]",
          py::arg("A"),
          py::arg("B"),
          py::arg("tokens_per_expert"),
          py::arg("M"),
          py::arg("N"));

    // Single chunk grouped GEMM for pipelined dX + AllToAll
    m.def("grouped_gemm_single_chunk", &fluid::grouped_gemm_single_chunk,
          "Single chunk grouped GEMM for pipelined computation\n"
          "Computes one chunk of the grouped GEMM (each chunk processes portion of each expert's tokens)\n"
          "Args:\n"
          "  A: Input tensor [total_tokens, K]\n"
          "  B: Weight tensor [num_experts, K, N] or [num_experts, N, K] if trans_b\n"
          "  tokens_per_expert: Number of tokens per expert [num_experts]\n"
          "  C: Pre-allocated output tensor [total_tokens, N]\n"
          "  trans_a: Whether to transpose A\n"
          "  trans_b: Whether to transpose B\n"
          "  num_chunks: Total number of chunks\n"
          "  chunk_idx: Index of this chunk (0 to num_chunks-1)\n"
          "Returns: The output tensor C",
          py::arg("A"),
          py::arg("B"),
          py::arg("tokens_per_expert"),
          py::arg("C"),
          py::arg("trans_a") = false,
          py::arg("trans_b") = false,
          py::arg("num_chunks") = 1,
          py::arg("chunk_idx") = 0);

    // Get chunk info for a specific chunk
    m.def("get_chunk_info", &fluid::get_chunk_info,
          "Get chunk info for a specific chunk\n"
          "Args:\n"
          "  tokens_per_expert: Number of tokens per expert [num_experts]\n"
          "  num_chunks: Total number of chunks\n"
          "  chunk_idx: Index of this chunk\n"
          "Returns: Tensor [num_experts, 2] with (global_start, chunk_size) per expert",
          py::arg("tokens_per_expert"),
          py::arg("num_chunks"),
          py::arg("chunk_idx"));

    // Compute all chunk boundaries
    m.def("compute_chunk_boundaries", &fluid::compute_chunk_boundaries,
          "Compute chunk boundaries for all chunks\n"
          "Args:\n"
          "  tokens_per_expert: Number of tokens per expert [num_experts]\n"
          "  num_chunks: Total number of chunks\n"
          "Returns: Tensor [num_chunks, num_experts, 3] with (global_start, chunk_start_in_expert, chunk_size)",
          py::arg("tokens_per_expert"),
          py::arg("num_chunks"));

    // Grouped GEMM with gather (fused index_select + GEMM)
    m.def("grouped_gemm_with_gather", &fluid::grouped_gemm_with_gather,
          "Grouped GEMM with gather for non-contiguous input access\n"
          "Computes C = A[indices] @ B^T where A is accessed via index array\n"
          "This fuses gather (index_select) into GEMM for better performance.\n"
          "Args:\n"
          "  A: Full input tensor [total_tokens, K]\n"
          "  B: Weight tensor [num_experts, N, K] (for trans_b=true)\n"
          "  indices: Index array [chunk_total] specifying which rows of A to use\n"
          "  tokens_per_expert: Number of tokens per expert in chunk [num_experts]\n"
          "  trans_b: Whether B is transposed (currently must be true)\n"
          "Returns: Output tensor [chunk_total, N]",
          py::arg("A"),
          py::arg("B"),
          py::arg("indices"),
          py::arg("tokens_per_expert"),
          py::arg("trans_b") = true);

    // Fused dX kernel with chunked output for pipelining
    m.def("grouped_gemm_dx_chunked", &fluid::grouped_gemm_dx_chunked,
          "Fused dX computation with chunked output for pipeline overlap\n"
          "Computes all chunks of dX in C++ without returning to Python.\n"
          "Sets chunk_ready_flags[i] = 1 when chunk i is complete.\n"
          "Args:\n"
          "  grad_fc2: Gradient from fc2 output [total_tokens, hidden_size]\n"
          "  probs: Expert probabilities [total_tokens] or [total_tokens, 1]\n"
          "  act_deriv: Activation derivative [total_tokens, intermediate_size]\n"
          "  w1: Weight matrix w1 [num_experts, hidden_size, ffn_size]\n"
          "  w2: Weight matrix w2 [num_experts, intermediate_size, hidden_size]\n"
          "  chunk_indices: Index array [total_chunk_tokens]\n"
          "  chunk_offsets: Offset array [num_chunks+1]\n"
          "  num_chunks: Number of chunks\n"
          "  act_val: Optional activation values for GLU\n"
          "  x_2: Optional x_2 for GLU\n"
          "Returns: [output, chunk_ready_flags]",
          py::arg("grad_fc2"),
          py::arg("probs"),
          py::arg("act_deriv"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("chunk_indices"),
          py::arg("chunk_offsets"),
          py::arg("num_chunks"),
          py::arg("act_val") = c10::nullopt,
          py::arg("x_2") = c10::nullopt);

    // Fused dX kernel for single chunk (use in Python loop)
    m.def("grouped_gemm_dx_fused", &fluid::grouped_gemm_dx_fused,
          "Fused dX computation for a single chunk\n"
          "Computes dx = (grad_fc2[indices] @ W2.T * act_deriv * probs) @ W1.T\n"
          "Also returns grad_fc1 for dW computation.\n"
          "Args:\n"
          "  grad_fc2: Gradient from fc2 output [total_tokens, hidden_size]\n"
          "  probs: Expert probabilities [total_tokens] or [total_tokens, 1]\n"
          "  act_deriv: Activation derivative [total_tokens, intermediate_size]\n"
          "  w1: Weight matrix w1 [num_experts, hidden_size, ffn_size]\n"
          "  w2: Weight matrix w2 [num_experts, intermediate_size, hidden_size]\n"
          "  chunk_indices: Index array [chunk_size]\n"
          "  act_val: Optional activation values for GLU\n"
          "  x_2: Optional x_2 for GLU\n"
          "Returns: [dx, grad_fc1] - dx [chunk_size, hidden_size], grad_fc1 [chunk_size, fc1_dim]",
          py::arg("grad_fc2"),
          py::arg("probs"),
          py::arg("act_deriv"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("chunk_indices"),
          py::arg("act_val") = c10::nullopt,
          py::arg("x_2") = c10::nullopt);

    // Fused dX kernel for ALL chunks (C++ loop, eliminates Python overhead)
    m.def("grouped_gemm_dx_all_chunks", &fluid::grouped_gemm_dx_all_chunks,
          "Fused dX computation for all chunks in one C++ call\n"
          "Eliminates Python loop overhead by computing all chunks in C++.\n"
          "Args:\n"
          "  grad_fc2: Gradient from fc2 output [total_tokens, hidden_size]\n"
          "  probs: Expert probabilities [total_tokens] or [total_tokens, 1]\n"
          "  act_deriv: Activation derivative [total_tokens, intermediate_size]\n"
          "  w1: Weight matrix w1 [num_experts, hidden_size, ffn_size]\n"
          "  w2: Weight matrix w2 [num_experts, intermediate_size, hidden_size]\n"
          "  all_chunk_indices: All chunk indices concatenated [total_chunk_tokens]\n"
          "  chunk_offsets: Offset array [num_chunks+1]\n"
          "  num_chunks: Number of chunks\n"
          "  act_val: Optional activation values for GLU\n"
          "  x_2: Optional x_2 for GLU\n"
          "Returns: [full_dx, full_grad_fc1] - concatenated results for all chunks",
          py::arg("grad_fc2"),
          py::arg("probs"),
          py::arg("act_deriv"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("all_chunk_indices"),
          py::arg("chunk_offsets"),
          py::arg("num_chunks"),
          py::arg("act_val") = c10::nullopt,
          py::arg("x_2") = c10::nullopt);

    // Pipelined dX kernel with CUDA events for true overlap
    m.def("grouped_gemm_dx_pipelined", &fluid::grouped_gemm_dx_pipelined,
          "Pipelined dX computation with CUDA events for true overlap\n"
          "Computes all chunks and returns events that can be waited on.\n"
          "The caller can wait on each event and launch AllToAll immediately,\n"
          "achieving true overlap between dX[i+1] and AllToAll[i].\n"
          "Args:\n"
          "  grad_fc2: Gradient from fc2 output [total_tokens, hidden_size]\n"
          "  probs: Expert probabilities [total_tokens] or [total_tokens, 1]\n"
          "  act_deriv: Activation derivative [total_tokens, intermediate_size]\n"
          "  w1: Weight matrix w1 [num_experts, hidden_size, ffn_size]\n"
          "  w2: Weight matrix w2 [num_experts, intermediate_size, hidden_size]\n"
          "  all_chunk_indices: All chunk indices concatenated [total_chunk_tokens]\n"
          "  chunk_offsets: Offset array [num_chunks+1]\n"
          "  num_chunks: Number of chunks\n"
          "  act_val: Optional activation values for GLU\n"
          "  x_2: Optional x_2 for GLU\n"
          "Returns: [full_dx, full_grad_fc1, event_ptrs]",
          py::arg("grad_fc2"),
          py::arg("probs"),
          py::arg("act_deriv"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("all_chunk_indices"),
          py::arg("chunk_offsets"),
          py::arg("num_chunks"),
          py::arg("act_val") = c10::nullopt,
          py::arg("x_2") = c10::nullopt);

    // Helper to wait on a CUDA event from another stream
    m.def("wait_cuda_event", &fluid::wait_cuda_event,
          "Make a CUDA stream wait on an event\n"
          "Args:\n"
          "  event_ptr: int64 pointer to cudaEvent_t\n"
          "  stream_ptr: int64 pointer to cudaStream_t",
          py::arg("event_ptr"),
          py::arg("stream_ptr"));

    // Helper to destroy CUDA events
    m.def("destroy_cuda_events", &fluid::destroy_cuda_events,
          "Destroy CUDA events to prevent memory leak\n"
          "Args:\n"
          "  event_ptrs: Tensor of int64 pointers to cudaEvent_t",
          py::arg("event_ptrs"));

    // ============================================================================
    // GroupedGEMM with Epilogue-Level Signaling (True FlashOverlap Style)
    // ============================================================================

    m.def("grouped_gemm_epilogue_signal", &fluid::grouped_gemm_epilogue_signal,
          "GroupedGEMM with Epilogue-level signaling (FlashOverlap style)\n"
          "This implements true fine-grained signaling where:\n"
          "  - Signals are emitted in the Epilogue phase using atomicAdd\n"
          "  - Each tile's completion triggers an atomic increment\n"
          "  - Communication stream can poll/wait for tile completion\n"
          "Key difference from grouped_gemm_with_wave_signal:\n"
          "  - Previous: Signal after entire GroupedGEMM kernel completes\n"
          "  - This: Signal after each tile's Epilogue completes (true overlap)\n"
          "Args:\n"
          "  input: FC2 input [total_tokens, K]\n"
          "  weight: FC2 weight [num_experts, K, N]\n"
          "  output: Pre-allocated output [total_tokens, N]\n"
          "  tokens_per_expert: Tokens per expert [num_experts]\n"
          "  tiles_per_wave: Tiles per wave (0 = use SM count)\n"
          "Returns: [wave_info, counter_ptr, tile_counts, info]\n"
          "  wave_info: [num_waves, 3] (start_tile, tile_count, tiles_per_wave)\n"
          "  counter_ptr: [1] pointer to wave counters\n"
          "  tile_counts: [num_waves] expected tiles per wave\n"
          "  info: [6] (total_tiles, num_waves, tiles_per_wave, sm_count, num_experts, stream_handle)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("tokens_per_expert"),
          py::arg("tiles_per_wave") = 0,
          py::arg("max_sm_usage") = 0);

    m.def("wait_for_wave_signal", &fluid::wait_for_wave_signal_pytorch,
          "Wait for a wave to complete using atomic counter polling\n"
          "This launches a wait kernel on the communication stream.\n"
          "Args:\n"
          "  wave_idx: Wave index to wait for",
          py::arg("wave_idx"));

    m.def("wait_for_all_waves", &fluid::wait_for_all_waves_pytorch,
          "Wait for all waves to complete in a single kernel launch\n"
          "This eliminates Python loop overhead by processing all waves in C++/CUDA.\n"
          "Launches a single kernel that sequentially waits for each wave.");

    m.def("get_comm_stream_epi", []() {
              return reinterpret_cast<int64_t>(fluid::get_comm_stream());
          },
          "Get the communication stream handle\n"
          "Returns: int64 stream pointer (for use with NCCL)");

    m.def("get_compute_stream_epi", []() {
              return reinterpret_cast<int64_t>(fluid::get_compute_stream());
          },
          "Get the compute stream handle\n"
          "Returns: int64 stream pointer");

    m.def("sync_comm_to_compute", &fluid::sync_comm_to_compute,
          "Synchronize communication stream back to compute stream\n"
          "Call this after all overlapped operations complete.");

    m.def("get_wave_boundaries_epi", &fluid::get_wave_boundaries_epi,
          "Get wave boundaries for epilogue signaling\n"
          "Returns: [num_waves, 3] (token_start, token_count, tile_count)");

    m.def("get_epilogue_signal_info", &fluid::get_epilogue_signal_info,
          "Get epilogue signal context info\n"
          "Returns: [6] (total_tiles, num_waves, tiles_per_wave, sm_count, num_experts, total_tokens)");

    m.def("destroy_epilogue_signal_context", &fluid::destroy_epilogue_signal_context,
          "Clean up epilogue signal context and free resources");

    // ============================================================================
    // Monitoring Mode - Hint Collection (FlashOverlap style)
    // ============================================================================

    m.def("grouped_gemm_with_monitoring", &fluid::grouped_gemm_with_monitoring,
          "Run GroupedGEMM in monitoring mode to collect tile completion order.\n"
          "This is used for FlashOverlap-style hint collection.\n"
          "Args:\n"
          "  input: [total_tokens, K] input tensor\n"
          "  weight: [num_experts, K, N] weight tensor\n"
          "  output: [total_tokens, N] output tensor\n"
          "  tokens_per_expert: [num_experts] token counts\n"
          "  hint_buffer: [total_tiles] buffer to store completion order\n"
          "Returns: total_tiles computed",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("tokens_per_expert"),
          py::arg("hint_buffer"));

    m.def("get_total_tiles", &fluid::get_total_tiles,
          "Get the total number of tiles for a given configuration.\n"
          "Useful for allocating hint buffer before calling monitoring.\n"
          "Args:\n"
          "  tokens_per_expert: [num_experts] token counts\n"
          "  N: output dimension\n"
          "Returns: total number of tiles",
          py::arg("tokens_per_expert"),
          py::arg("N"));

    // ============================================================================
    // FlashOverlap-style NCCL Integration (Queue wait + NCCL in C++ loop)
    // ============================================================================

    m.def("init_nccl_comm", &fluid::init_nccl_comm,
          "Initialize NCCL communicator for FlashOverlap-style overlap\n"
          "This must be called after grouped_gemm_epilogue_signal.\n"
          "Args:\n"
          "  nccl_comm_ptr: int64 pointer to ncclComm_t (from distributed._get_default_group()._get_backend(...)._get_nccl_comm())\n"
          "  rank: Current rank\n"
          "  world_size: Total number of ranks",
          py::arg("nccl_comm_ptr"),
          py::arg("rank"),
          py::arg("world_size"));

    m.def("get_nccl_unique_id", &fluid::get_nccl_unique_id,
          "Get NCCL unique ID for distributed initialization (FlashOverlap style)\n"
          "Rank 0 should call this and broadcast to all ranks.\n"
          "Returns: List of int64 representing ncclUniqueId bytes");

    m.def("init_nccl_comm_with_id", &fluid::init_nccl_comm_with_id,
          "Initialize NCCL communicator with broadcasted ID (FlashOverlap style)\n"
          "This creates an independent NCCL communicator without using PyTorch's.\n"
          "Args:\n"
          "  nccl_id: List of int64 from get_nccl_unique_id()\n"
          "  rank: Current rank\n"
          "  world_size: Total number of ranks",
          py::arg("nccl_id"),
          py::arg("rank"),
          py::arg("world_size"));

    m.def("queue_flash_overlap_alltoall",
          [](int64_t send_ptr, int64_t recv_ptr, int waves_per_comm) {
              fluid::queue_flash_overlap_alltoall(
                  reinterpret_cast<void*>(send_ptr),
                  reinterpret_cast<void*>(recv_ptr),
                  waves_per_comm
              );
          },
          "Queue FlashOverlap-style overlap with AllToAll\n"
          "This is the KEY optimization from FlashOverlap:\n"
          "  - All wait kernels + NCCL operations are queued in a C++ loop\n"
          "  - No Python in the critical path\n"
          "  - NCCL is pre-queued and starts immediately when signal arrives\n"
          "Timeline:\n"
          "  Compute: [GEMM tile0][tile1]...[tileN]\n"
          "                  |signal    |signal\n"
          "                  v          v\n"
          "  Comm:    [wait]→[NCCL0][wait]→[NCCL1]...\n"
          "           ^pre-queued^  ^pre-queued^\n"
          "Args:\n"
          "  send_ptr: int64 pointer to send tensor data\n"
          "  recv_ptr: int64 pointer to recv tensor data\n"
          "  waves_per_comm: Group N waves per NCCL call (4 recommended for 2 GPUs)",
          py::arg("send_ptr"),
          py::arg("recv_ptr"),
          py::arg("waves_per_comm"));

    m.def("queue_flash_overlap_alltoall_single",
          [](int64_t send_ptr, int64_t recv_ptr, int waves_per_comm) {
              fluid::queue_flash_overlap_alltoall_single(
                  reinterpret_cast<void*>(send_ptr),
                  reinterpret_cast<void*>(recv_ptr),
                  waves_per_comm
              );
          },
          "Queue FlashOverlap-style overlap with AllToAllSingle\n"
          "Simplified version using ncclSend/ncclRecv pairs.\n"
          "Args:\n"
          "  send_ptr: int64 pointer to send tensor data\n"
          "  recv_ptr: int64 pointer to recv tensor data\n"
          "  waves_per_comm: Group N waves per NCCL call",
          py::arg("send_ptr"),
          py::arg("recv_ptr"),
          py::arg("waves_per_comm"));

    // ============================================================================
    // Fully Fused FC2 + AllToAll API
    // ============================================================================

    m.def("fc2_alltoall_overlap", &fluid::fc2_alltoall_overlap,
          "FC2 + AllToAll with Full C++ Overlap (Zero Python Loop)\n"
          "Single call that fuses FC2 GEMM with AllToAll communication.\n"
          "Timeline:\n"
          "  FC2:      [wave0][wave1][wave2]...\n"
          "  AllToAll:       [comm0]     [comm1]...\n"
          "Args:\n"
          "  input: [total_tokens, K] input tensor\n"
          "  weight: [num_experts, K, N] weight tensor\n"
          "  output: [total_tokens, N] output/send buffer\n"
          "  recv_buffer: [total_tokens, N] receive buffer\n"
          "  tokens_per_expert: [num_experts] token counts\n"
          "  tiles_per_wave: tiles per wave (0 = use SM count)\n"
          "  waves_per_comm: waves to group per communication\n"
          "Returns: recv_buffer with AllToAll results",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("recv_buffer"),
          py::arg("tokens_per_expert"),
          py::arg("tiles_per_wave"),
          py::arg("waves_per_comm"));

    m.def("init_fc2_alltoall_nccl", &fluid::init_fc2_alltoall_nccl,
          "Initialize NCCL for FC2+AllToAll overlap\n"
          "Args:\n"
          "  rank: local rank\n"
          "  world_size: total ranks\n"
          "  nccl_id: NCCL unique ID vector (from get_fc2_nccl_unique_id)",
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("nccl_id"));

    m.def("get_fc2_nccl_unique_id", &fluid::get_fc2_nccl_unique_id,
          "Get NCCL unique ID for FC2+AllToAll initialization");

    m.def("destroy_fc2_alltoall_nccl", &fluid::destroy_fc2_alltoall_nccl,
          "Destroy NCCL for FC2+AllToAll");

    // ============================================================================
    // FlashOverlap-style Tuning Functions
    // ============================================================================

    // TODO: Re-enable these when implementing the functions
    // m.def("compute_reorder_from_hints", &fluid::compute_reorder_from_hints,
    //       "Compute tile reorder array from hint buffer (FlashOverlap style)\n"
    //       "Args:\n"
    //       "  hints: Tensor of tile completion order [total_tiles]\n"
    //       "  tiles_per_wave: Tiles per wave for grouping\n"
    //       "Returns: reorder_array where reorder_array[new_idx] = old_tile_idx",
    //       py::arg("hints"),
    //       py::arg("tiles_per_wave"));

    // m.def("compute_inverse_ra", &fluid::compute_inverse_ra,
    //       "Compute inverse reorder array\n"
    //       "Args:\n"
    //       "  reorder_array: Tensor [total_tiles]\n"
    //       "Returns: inverse_ra where inverse_ra[old_idx] = new_position",
    //       py::arg("reorder_array"));

    // m.def("get_search_result", &fluid::get_search_result,
    //       "Search for optimal (tiles_per_wave, waves_per_comm) configuration\n"
    //       "Implements FlashOverlap's exhaustive or predictive search.\n"
    //       "Args:\n"
    //       "  total_tiles: Total number of tiles\n"
    //       "  sm_count: Number of SMs\n"
    //       "  compute_time_per_tile_us: Estimated compute time per tile\n"
    //       "  elements_per_tile: Elements per tile (for bandwidth calculation)\n"
    //       "  use_predictive: Use fast predictive search (vs exhaustive)\n"
    //       "Returns: [tiles_per_wave, waves_per_comm, predicted_time_ns]",
    //       py::arg("total_tiles"),
    //       py::arg("sm_count"),
    //       py::arg("compute_time_per_tile_us"),
    //       py::arg("elements_per_tile"),
    //       py::arg("use_predictive") = true);

    // m.def("get_compute_sms", &fluid::get_compute_sms,
    //       "Get effective compute SMs considering NCCL SM usage\n"
    //       "FlashOverlap assumes NCCL uses ~2 SMs for collective operations.\n"
    //       "Args:\n"
    //       "  total_sms: Total number of SMs\n"
    //       "  nccl_sms: Estimated SMs used by NCCL (default 2)\n"
    //       "Returns: Effective compute SMs (total_sms - nccl_sms)",
    //       py::arg("total_sms"),
    //       py::arg("nccl_sms") = 2);

    // ============================================================================
    // AllToAll + FC1 Overlap
    // ============================================================================

    m.def("init_alltoall_fc1_nccl", &fluid::init_alltoall_fc1_nccl,
          "Initialize NCCL communicator for AllToAll+FC1 overlap\n"
          "Args:\n"
          "  rank: Current rank\n"
          "  world_size: Total ranks\n"
          "  nccl_id: NCCL unique ID from get_nccl_unique_id()",
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("nccl_id"));

    m.def("destroy_alltoall_fc1_nccl", &fluid::destroy_alltoall_fc1_nccl,
          "Destroy NCCL communicator for AllToAll+FC1 overlap");

    m.def("compute_localfirst_indices",
          [](torch::Tensor tokens_per_expert,
             torch::Tensor self_tokens_per_expert,
             torch::Tensor num_global_tokens_per_local_expert,
             int my_rank,
             int device_index) {
              return fluid::compute_localfirst_indices(
                  tokens_per_expert,
                  self_tokens_per_expert,
                  num_global_tokens_per_local_expert,
                  my_rank,
                  torch::Device(torch::kCUDA, device_index)
              );
          },
          "Compute indices for Local-First overlap (call ONCE, reuse)\n"
          "Args:\n"
          "  tokens_per_expert: [num_local_experts] - Total token count per expert\n"
          "  self_tokens_per_expert: [num_local_experts] - Self-copy token count per expert\n"
          "  num_global_tokens_per_local_expert: [ep_size, num_local_experts]\n"
          "  my_rank: Current EP rank\n"
          "  device_index: CUDA device index (e.g., 0 for cuda:0)\n"
          "Returns: [sort_indices, remote_input_indices, self_output_indices,\n"
          "          remote_output_indices, remote_tokens_per_expert]",
          py::arg("tokens_per_expert"),
          py::arg("self_tokens_per_expert"),
          py::arg("num_global_tokens_per_local_expert"),
          py::arg("my_rank"),
          py::arg("device_index"));

    m.def("alltoall_fc1_localfirst", &fluid::alltoall_fc1_localfirst,
          "AllToAll + FC1 with Local-First Overlap (BEST for most cases)\n"
          "TRUE OVERLAP: Self FC1 runs in parallel with AllToAll.\n"
          "Timeline:\n"
          "  AllToAll:   [================]  <- comm_stream\n"
          "  Self FC1:   [================]  <- compute_stream (PARALLEL!)\n"
          "  Sort:                          [=]\n"
          "  Remote FC1:                        [===========]\n"
          "Args:\n"
          "  permuted_tokens: [total_tokens, hidden_size]\n"
          "  fc1_weight: [num_local_experts, hidden_size, ffn_hidden_size]\n"
          "  input_splits_vec: List[int] - Token counts to send (pre-computed CPU)\n"
          "  output_splits_vec: List[int] - Token counts to receive (pre-computed CPU)\n"
          "  self_input_offset: Offset of self tokens in permuted_tokens\n"
          "  self_input_count: Number of self tokens\n"
          "  self_tokens_per_expert: [num_local_experts]\n"
          "  sort_indices: [total_output_tokens] - from compute_localfirst_indices\n"
          "  remote_input_indices: [remote_count] - from compute_localfirst_indices\n"
          "  self_output_indices: [self_count] - from compute_localfirst_indices\n"
          "  remote_output_indices: [remote_count] - from compute_localfirst_indices\n"
          "  remote_tokens_per_expert: [num_local_experts] - from compute_localfirst_indices\n"
          "  my_rank: Current EP rank",
          py::arg("permuted_tokens"),
          py::arg("fc1_weight"),
          py::arg("input_splits_vec"),
          py::arg("output_splits_vec"),
          py::arg("self_input_offset"),
          py::arg("self_input_count"),
          py::arg("self_tokens_per_expert"),
          py::arg("sort_indices"),
          py::arg("remote_input_indices"),
          py::arg("self_output_indices"),
          py::arg("remote_output_indices"),
          py::arg("remote_tokens_per_expert"),
          py::arg("my_rank"));

    m.def("alltoall_fc1_pipelined", &fluid::alltoall_fc1_pipelined,
          "AllToAll + FC1 with Per-Peer Pipelined Overlap (Zero-Sync Version)\n"
          "Finer-grained overlap: computes FC1 as each peer's data arrives.\n"
          "Two modes:\n"
          "  serialize_peer_fc1=true:  Serialize peer FC1 (avoids SM competition)\n"
          "  serialize_peer_fc1=false: Parallel peer FC1 (may compete for SMs)\n"
          "Timeline:\n"
          "  AllToAll:    [=======================]  <- comm_stream\n"
          "  Self FC1:    [=========]                <- compute_stream (PARALLEL!)\n"
          "  Peer0 FC1:             [=========]      <- waits for peer0 data\n"
          "  Peer1 FC1:                       [=========]\n"
          "Args:\n"
          "  permuted_tokens: [total_tokens, hidden_size]\n"
          "  fc1_weight: [num_local_experts, hidden_size, ffn_hidden_size]\n"
          "  input_splits_vec: List[int] - Token counts to send (pre-computed CPU)\n"
          "  output_splits_vec: List[int] - Token counts to receive (pre-computed CPU)\n"
          "  self_input_offset: Offset of self tokens in permuted_tokens\n"
          "  self_input_count: Number of self tokens\n"
          "  self_tokens_per_expert: [num_local_experts]\n"
          "  peer_tokens_per_expert_vec: List[Tensor] - Pre-computed on GPU for each peer\n"
          "  peer_token_counts: List[int] - Token count per peer (pre-computed CPU)\n"
          "  num_global_flat: List[int] - Flattened [ep_size * num_local_experts]\n"
          "  num_local_experts: Number of local experts\n"
          "  my_rank: Current EP rank\n"
          "  serialize_peer_fc1: Whether to serialize peer FC1 (default: true)",
          py::arg("permuted_tokens"),
          py::arg("fc1_weight"),
          py::arg("input_splits_vec"),
          py::arg("output_splits_vec"),
          py::arg("self_input_offset"),
          py::arg("self_input_count"),
          py::arg("self_tokens_per_expert"),
          py::arg("peer_tokens_per_expert_vec"),
          py::arg("peer_token_counts"),
          py::arg("num_global_flat"),
          py::arg("num_local_experts"),
          py::arg("my_rank"),
          py::arg("serialize_peer_fc1") = true);

    // ============================================================================
    // AllToAll + FC1 Pipelined WITHOUT Reorder (for FC2+AllToAll symmetric design)
    // ============================================================================

    m.def("alltoall_fc1_pipelined_no_reorder", &fluid::alltoall_fc1_pipelined_no_reorder,
          "AllToAll + FC1 with Per-Peer Pipelined Overlap (NO Reorder)\n"
          "Same as alltoall_fc1_pipelined but skips the final expert reordering.\n"
          "Output format: [self_fc1, peer0_fc1, peer1_fc1, ...]\n"
          "Designed to work with fc2_alltoall_pipelined for symmetric overlap.\n"
          "Timeline:\n"
          "  AllToAll:    [=======================]  <- comm_stream\n"
          "  Self FC1:    [=========]                <- compute_stream (PARALLEL!)\n"
          "  Peer0 FC1:             [=========]      <- waits for peer0 data\n"
          "  Peer1 FC1:                       [=========]\n"
          "  (NO reorder step - deferred to FC2+AllToAll)\n"
          "Args:\n"
          "  permuted_tokens: [total_tokens, hidden_size]\n"
          "  fc1_weight: [num_local_experts, hidden_size, ffn_hidden_size]\n"
          "  input_splits_vec: List[int] - Token counts to send (pre-computed CPU)\n"
          "  output_splits_vec: List[int] - Token counts to receive (pre-computed CPU)\n"
          "  self_input_offset: Offset of self tokens in permuted_tokens\n"
          "  self_input_count: Number of self tokens\n"
          "  self_tokens_per_expert: [num_local_experts]\n"
          "  peer_tokens_per_expert_vec: List[Tensor] - Pre-computed on GPU for each peer\n"
          "  peer_token_counts: List[int] - Token count per peer (pre-computed CPU)\n"
          "  my_rank: Current EP rank\n"
          "  serialize_peer_fc1: Whether to serialize peer FC1 (default: true)\n"
          "Returns: [combined_output, segment_sizes]\n"
          "  combined_output: [total_tokens, ffn_hidden_size] in [self, peer0, peer1, ...] order\n"
          "  segment_sizes: [ep_size] token count for each segment",
          py::arg("permuted_tokens"),
          py::arg("fc1_weight"),
          py::arg("input_splits_vec"),
          py::arg("output_splits_vec"),
          py::arg("self_input_offset"),
          py::arg("self_input_count"),
          py::arg("self_tokens_per_expert"),
          py::arg("peer_tokens_per_expert_vec"),
          py::arg("peer_token_counts"),
          py::arg("my_rank"),
          py::arg("serialize_peer_fc1") = true);

    // ============================================================================
    // FC2 + AllToAll Pipelined Overlap (Symmetric to AllToAll+FC1)
    // ============================================================================

    m.def("init_fc2_alltoall_pipelined_nccl", &fluid::init_fc2_alltoall_pipelined_nccl,
          "Initialize NCCL for FC2+AllToAll pipelined overlap\n"
          "Args:\n"
          "  rank: Current rank\n"
          "  world_size: Total ranks\n"
          "  nccl_id: NCCL unique ID from get_fc2_alltoall_pipelined_nccl_unique_id()",
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("nccl_id"));

    m.def("destroy_fc2_alltoall_pipelined_nccl", &fluid::destroy_fc2_alltoall_pipelined_nccl,
          "Destroy NCCL for FC2+AllToAll pipelined overlap");

    m.def("get_fc2_alltoall_pipelined_nccl_unique_id", &fluid::get_fc2_alltoall_pipelined_nccl_unique_id,
          "Get NCCL unique ID for FC2+AllToAll pipelined initialization\n"
          "Returns: List of int64 representing ncclUniqueId bytes");

    m.def("fc2_alltoall_pipelined", &fluid::fc2_alltoall_pipelined,
          "FC2 + AllToAll with Send-First Pipelined Overlap\n"
          "Symmetric to AllToAll+FC1: compute FC2 for each peer, send immediately.\n"
          "Input format: [self_tokens, peer0_tokens, peer1_tokens, ...]\n"
          "             (from alltoall_fc1_pipelined_no_reorder)\n"
          "Timeline:\n"
          "  FC2(peer0):  [========]\n"
          "  Send(peer0):          [===========]\n"
          "  FC2(peer1):           [========]\n"
          "  Send(peer1):                   [===========]\n"
          "  FC2(self):                     [========]  <- last, no send\n"
          "  Recv(all):            [=========================]\n"
          "  Reorder:                                   [====]\n"
          "Args:\n"
          "  input: [total_tokens, hidden_size] in [self, peer0, peer1, ...] order\n"
          "  fc2_weight: [num_experts, ffn_hidden_size, hidden_size]\n"
          "  self_token_count: Number of self tokens\n"
          "  peer_token_counts: List[int] - Token count for each peer\n"
          "  self_tokens_per_expert: [num_experts] - Expert distribution for self\n"
          "  peer_tokens_per_expert_vec: List[Tensor] - Expert distribution for each peer\n"
          "  send_splits: List[int] - Tokens to send to each peer\n"
          "  recv_splits: List[int] - Tokens to receive from each peer\n"
          "  reorder_indices: [total_output_tokens] - Optional reorder indices\n"
          "  my_rank: Current rank\n"
          "Returns: Output tensor [total_output_tokens, hidden_size] in original order",
          py::arg("input"),
          py::arg("fc2_weight"),
          py::arg("self_token_count"),
          py::arg("peer_token_counts"),
          py::arg("self_tokens_per_expert"),
          py::arg("peer_tokens_per_expert_vec"),
          py::arg("send_splits"),
          py::arg("recv_splits"),
          py::arg("reorder_indices"),
          py::arg("my_rank"));

    // ============================================================================
    // Fully Fused MoE Forward (AllToAll + FC1 + GELU + FC2 + AllToAll)
    // ============================================================================

    m.def("get_moe_fused_nccl_unique_id", &fluid::get_moe_fused_nccl_unique_id,
          "Get NCCL unique ID for fully fused MoE forward\n"
          "Returns: List of int64 representing ncclUniqueId bytes");

    m.def("init_moe_fused_nccl", &fluid::init_moe_fused_nccl,
          "Initialize NCCL for fully fused MoE forward\n"
          "Args:\n"
          "  rank: Current rank\n"
          "  world_size: Total ranks\n"
          "  nccl_id: NCCL unique ID from get_moe_fused_nccl_unique_id()",
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("nccl_id"));

    m.def("destroy_moe_fused_nccl", &fluid::destroy_moe_fused_nccl,
          "Destroy NCCL for fully fused MoE forward");

    m.def("moe_alltoall_fc1_fused", &fluid::moe_alltoall_fc1_fused,
          "Fused AllToAll + FC1 + Activation with compute-communication overlap\n"
          "Timeline:\n"
          "  Self FC1:    [=========] <- compute (starts IMMEDIATELY!)\n"
          "  AllToAll:    [=========] <- communication (TRUE PARALLEL!)\n"
          "  Peer0 FC1:         [===] <- process as data arrives\n"
          "  Peer1 FC1:             [===]\n"
          "Note: probs multiplication happens in Python unpermute (standard Megatron behavior)\n"
          "Args:\n"
          "  permuted_tokens: [total_tokens, hidden_size]\n"
          "  fc1_weight: [num_experts, hidden_size, ffn_hidden_size]\n"
          "  self_input_offset: Offset of self tokens in input\n"
          "  self_input_count: Number of self tokens\n"
          "  send_splits: Token counts to send to each rank (List[int])\n"
          "  recv_splits: Token counts to receive from each rank (List[int])\n"
          "  peer_token_counts: Total tokens from each peer (List[int])\n"
          "  h_self_tokens_per_expert: PRE-COMPUTED HOST list (List[int])\n"
          "  h_peer_tokens_per_expert_all: PRE-COMPUTED HOST nested list (List[List[int]])\n"
          "  activation_type: 0=GELU(default), 1=SiLU, 2=ReLU, 3=None\n"
          "Returns: (fc1_output, segment_sizes, dispatched_input, fc1_pre_activation)\n"
          "  - fc1_output: [total_recv_tokens, ffn_hidden_size] after activation\n"
          "  - segment_sizes: [ep_size] tokens per source rank\n"
          "  - dispatched_input: [total_recv_tokens, hidden_size] FC1 input (for dW1)\n"
          "  - fc1_pre_activation: [total_recv_tokens, ffn_hidden_size] before activation",
          py::arg("permuted_tokens"),
          py::arg("fc1_weight"),
          py::arg("self_input_offset"),
          py::arg("self_input_count"),
          py::arg("send_splits"),
          py::arg("recv_splits"),
          py::arg("peer_token_counts"),
          py::arg("h_self_tokens_per_expert"),
          py::arg("h_peer_tokens_per_expert_all"),
          py::arg("activation_type") = 0);

    m.def("moe_alltoall_fc1_fused_expert_major", &fluid::moe_alltoall_fc1_fused_expert_major,
          "Fused AllToAll + FC1 + Activation with Expert-Major Output\n"
          "Same as moe_alltoall_fc1_fused but:\n"
          "  1. Outputs in expert-major layout (ready for grouped_gemm backward)\n"
          "  2. Returns reorder indices for backward computation\n"
          "Timeline:\n"
          "  Self FC1:    [=========] <- compute (starts IMMEDIATELY!)\n"
          "  AllToAll:    [=========] <- communication (TRUE PARALLEL!)\n"
          "  Peer0 FC1:         [===]\n"
          "  Reorder:               [==]  <- on GPU, can overlap with FC2\n"
          "Benefits:\n"
          "  - No Python-side index construction (~0.65ms saved per layer)\n"
          "  - Expert-major output avoids backward reorder overhead\n"
          "  - Reorder can overlap with FC2 computation\n"
          "Args:\n"
          "  permuted_tokens: [total_tokens, hidden_size]\n"
          "  fc1_weight: [num_experts, hidden_size, ffn_hidden_size]\n"
          "  self_input_offset: Offset of self tokens in input\n"
          "  self_input_count: Number of self tokens\n"
          "  send_splits: Token counts to send to each rank (List[int])\n"
          "  recv_splits: Token counts to receive from each rank (List[int])\n"
          "  peer_token_counts: Total tokens from each peer (List[int])\n"
          "  h_self_tokens_per_expert: PRE-COMPUTED HOST list (List[int])\n"
          "  h_peer_tokens_per_expert_all: PRE-COMPUTED HOST nested list (List[List[int]])\n"
          "  activation_type: 0=GELU(default), 1=SiLU, 2=ReLU, 3=None\n"
          "Returns: (fc1_output, segment_sizes, dispatched_input, fc1_pre_act, reorder_indices, inverse_indices)\n"
          "  - fc1_output: [total_recv_tokens, ffn_hidden_size] RANK-MAJOR (for FC2!)\n"
          "  - segment_sizes: [ep_size] tokens per source rank\n"
          "  - dispatched_input: [total_recv_tokens, hidden_size] EXPERT-MAJOR (for dW1)\n"
          "  - fc1_pre_activation: [total_recv_tokens, ffn_hidden_size] EXPERT-MAJOR\n"
          "  - reorder_indices: [total_recv_tokens] expert->rank position mapping\n"
          "  - inverse_indices: [total_recv_tokens] rank->expert position mapping",
          py::arg("permuted_tokens"),
          py::arg("fc1_weight"),
          py::arg("self_input_offset"),
          py::arg("self_input_count"),
          py::arg("send_splits"),
          py::arg("recv_splits"),
          py::arg("peer_token_counts"),
          py::arg("h_self_tokens_per_expert"),
          py::arg("h_peer_tokens_per_expert_all"),
          py::arg("activation_type") = 0);

    m.def("moe_fc2_alltoall_fused", &fluid::moe_fc2_alltoall_fused,
          "Fused FC2 + AllToAll with TRUE Overlap\n"
          "Takes FC1 output in [self, peer0, peer1, ...] format.\n"
          "Timeline:\n"
          "  Peer0 FC2:  [===]\n"
          "  Peer0 Send:      [=====]\n"
          "  Peer1 FC2:       [===]\n"
          "  Peer1 Send:           [=====]\n"
          "  Self FC2:             [===]    <- Overlaps with sends!\n"
          "  Self copy:                [=]\n"
          "Args:\n"
          "  fc1_output: [total_local_tokens, ffn_hidden_size] from stage 1\n"
          "  fc2_weight: [num_experts, ffn_hidden_size, hidden_size]\n"
          "  segment_sizes: [ep_size] from stage 1\n"
          "  original_send_splits: Original token counts sent to each rank\n"
          "  h_self_tokens_per_expert: PRE-COMPUTED HOST list (List[int])\n"
          "  h_peer_tokens_per_expert_all: PRE-COMPUTED HOST nested list (List[List[int]])\n"
          "Returns: Output [total_output_tokens, hidden_size]",
          py::arg("fc1_output"),
          py::arg("fc2_weight"),
          py::arg("segment_sizes"),
          py::arg("original_send_splits"),
          py::arg("h_self_tokens_per_expert"),
          py::arg("h_peer_tokens_per_expert_all"));

    // ============================================================================
    // Direct Output GEMM (for overlap without copy overhead)
    // ============================================================================

    m.def("grouped_gemm_to_output", &fluid::grouped_gemm_to_output,
          "Grouped GEMM that writes directly to a pre-allocated buffer\n"
          "This avoids tensor allocation + copy overhead in overlap scenarios.\n"
          "Args:\n"
          "  input: [total_tokens, K] input tensor\n"
          "  weight: [num_experts, K, N] weight tensor\n"
          "  output: [output_size, N] PRE-ALLOCATED output tensor\n"
          "  tokens_per_expert: [num_experts] int32 tensor\n"
          "  output_offset: Starting offset in output tensor to write to\n"
          "Returns: None (writes directly to output tensor)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("tokens_per_expert"),
          py::arg("output_offset") = 0);

    m.def("grouped_gemm_gelu_to_output", &fluid::grouped_gemm_gelu_to_output,
          "Grouped GEMM with GELU that writes directly to a pre-allocated buffer\n"
          "This fuses GEMM + GELU and avoids tensor allocation + copy overhead.\n"
          "Args:\n"
          "  input: [total_tokens, K] input tensor\n"
          "  weight: [num_experts, K, N] weight tensor\n"
          "  output: [output_size, N] PRE-ALLOCATED output tensor\n"
          "  tokens_per_expert: [num_experts] int32 tensor\n"
          "  output_offset: Starting offset in output tensor to write to\n"
          "Returns: None (writes directly to output tensor)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("tokens_per_expert"),
          py::arg("output_offset") = 0);

    // ============================================================================
    // True GroupedGEMM (drop-in replacement for grouped_gemm_forward)
    // ============================================================================

    m.def("grouped_gemm_true", &fluid::grouped_gemm_true_forward,
          "TRUE GroupedGEMM using CUTLASS GemmGrouped kernel.\n\n"
          "This is a drop-in replacement for grouped_gemm that uses a SINGLE kernel\n"
          "to process ALL experts, instead of launching separate kernels per expert.\n\n"
          "Key benefits:\n"
          "- Eliminates kernel launch overhead (critical for overlap!)\n"
          "- Better GPU utilization with persistent threads\n"
          "- 2x HALF-GEMM ≈ 1x FULL-GEMM (fixes the split GEMM overhead issue)\n\n"
          "Args:\n"
          "  input: [total_tokens, K] input tensor\n"
          "  weight: [num_experts, K, N] or [num_experts, N, K] weight tensor\n"
          "  tokens_per_expert: [num_experts] int32 tensor\n"
          "  total_tokens: Pre-computed total tokens (avoids GPU sync)\n"
          "  trans_a: Transpose A (must be False for now)\n"
          "  trans_b: Transpose B (if True, weight is [num_experts, N, K])\n"
          "Returns: [total_tokens, N] output tensor",
          py::arg("input"),
          py::arg("weight"),
          py::arg("tokens_per_expert"),
          py::arg("total_tokens"),
          py::arg("trans_a") = false,
          py::arg("trans_b") = false);
}
