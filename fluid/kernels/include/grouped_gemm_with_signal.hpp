/**
 * Grouped GEMM with Expert-Level Atomic Signaling
 *
 * Provides true fine-grained overlap between GEMM computation and AllToAll:
 * - Each expert runs on its own CUDA stream
 * - Expert completion is tracked via device-side atomics
 * - Wave completion is signaled when all experts in a wave finish
 * - Host can poll wave status to overlap AllToAll with computation
 *
 * Timeline:
 *   Stream0: [Expert0 GEMM] -> signal
 *   Stream1: [Expert1 GEMM] -> signal
 *   Stream2: [Expert2 GEMM] -> signal
 *   Stream3: [Expert3 GEMM] -> signal
 *
 *   Wave0 (E0, E2): completes when both E0 and E2 finish
 *   Wave1 (E1, E3): completes when both E1 and E3 finish
 *
 *   Comm stream: [wait wave0] -> [AllToAll_0] -> [wait wave1] -> [AllToAll_1]
 */

#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector>

namespace fluid {

/**
 * Execute Grouped GEMM with expert-level completion signaling
 *
 * Each expert GEMM runs concurrently on its own stream. After each expert
 * completes, a signal kernel updates the wave completion counter.
 *
 * @param input Input tensor [total_tokens, intermediate_size]
 * @param weight Weight tensor [num_experts, intermediate_size, hidden_size]
 * @param output Pre-allocated output tensor [total_tokens, hidden_size]
 * @param tokens_per_expert Token counts per expert [num_experts]
 * @param num_waves Number of waves for pipelining
 * @return [wave_events, wave_boundaries]
 *         - wave_events: [num_waves] CUDA event pointers
 *         - wave_boundaries: [num_waves, 2] (start, size) per wave
 */
std::vector<torch::Tensor> grouped_gemm_with_expert_signal(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor tokens_per_expert,
    int num_waves
);

/**
 * Poll wave completion status (non-blocking)
 *
 * @return Tensor [num_waves] with 1 for completed, 0 for in-progress
 */
torch::Tensor poll_wave_completion_signaled();

/**
 * Wait for a specific wave to complete on a given stream
 *
 * Use this to make the AllToAll stream wait for a wave before starting.
 *
 * @param wave_idx Wave index to wait for
 * @param wait_stream Stream that should wait
 */
void wait_for_wave(int wave_idx, cudaStream_t wait_stream);

/**
 * Clean up resources
 */
void destroy_signaled_context(int64_t context_ptr);

} // namespace fluid
