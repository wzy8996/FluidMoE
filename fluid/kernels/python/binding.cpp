/**
 * PyTorch C++ Extension bindings for Fluid kernels
 *
 * Only GroupedGEMM operations
 */

#include <torch/extension.h>

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
}
