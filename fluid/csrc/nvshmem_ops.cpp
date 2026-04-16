/*
 * FluidMoE NVSHMEM P2P Operations (host-only)
 *
 * Wraps NVSHMEM host-side stream-ordered APIs for P2P communication.
 * No device code (__global__ kernels) — pure C++ compiled by the system
 * compiler, not nvcc. This avoids all PyTorch half-operator conflicts.
 *
 * Links against: libnvshmem_host.so only.
 * Headers:       host/nvshmem_api.h, host/nvshmemx_api.h only.
 */

#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>

#include <torch/extension.h>
#include <cuda_runtime.h>

// =========================================================================
// Lifecycle
// =========================================================================

static bool g_initialized = false;

void nvshmem_init_wrapper() {
    if (g_initialized) return;
    // nvshmem_init() is static inline and calls nvshmemi_init_thread() which
    // is an internal symbol in libnvshmem_device.a (not in host .so).
    // Use nvshmemx_hostlib_init_attr() directly — it IS in libnvshmem_host.so.
    // With flags=0, bootstrap method is selected via NVSHMEM_BOOTSTRAP env var.
    nvshmemx_init_attr_t attr = {};
    int status = nvshmemx_hostlib_init_attr(0, &attr);
    TORCH_CHECK(status == 0, "nvshmemx_hostlib_init_attr failed with status ", status);
    g_initialized = true;
}

void nvshmem_finalize_wrapper() {
    if (!g_initialized) return;
    // Same reason: nvshmem_finalize() calls nvshmemi_finalize() (internal).
    // Use nvshmemx_hostlib_finalize() directly.
    nvshmemx_hostlib_finalize();
    g_initialized = false;
}

int nvshmem_my_pe_wrapper() {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    return nvshmem_my_pe();
}

int nvshmem_n_pes_wrapper() {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    return nvshmem_n_pes();
}

// =========================================================================
// Symmetric heap allocation → PyTorch tensor
// =========================================================================

torch::Tensor nvshmem_malloc_tensor(int64_t numel, c10::ScalarType dtype) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    TORCH_CHECK(numel > 0, "numel must be > 0");

    auto elem_size = c10::elementSize(dtype);
    size_t nbytes = static_cast<size_t>(numel) * elem_size;

    void* ptr = nvshmem_malloc(nbytes);
    TORCH_CHECK(ptr != nullptr, "nvshmem_malloc failed for ", nbytes, " bytes");

    cudaMemset(ptr, 0, nbytes);

    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, nvshmem_my_pe() % torch::cuda::device_count());
    return torch::from_blob(ptr, {numel}, options);
}

void nvshmem_free_tensor(torch::Tensor t) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    nvshmem_free(t.data_ptr());
}

// =========================================================================
// Remote pointer resolution
// =========================================================================

int64_t nvshmem_ptr_wrapper(torch::Tensor t, int pe) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    void* remote = nvshmem_ptr(t.data_ptr(), pe);
    return reinterpret_cast<int64_t>(remote);
}

// =========================================================================
// One-sided write with signal (stream-ordered)
// =========================================================================

void nvshmem_putmem_signal_on_stream_wrapper(
    int64_t dest_ptr,
    torch::Tensor src,
    int64_t nbytes,
    int64_t sig_addr,
    uint64_t sig_val,
    int sig_op,
    int pe,
    int64_t stream_ptr
) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    TORCH_CHECK(dest_ptr != 0, "dest_ptr is null");
    TORCH_CHECK(sig_addr != 0, "sig_addr is null");

    nvshmemx_putmem_signal_on_stream(
        reinterpret_cast<void*>(dest_ptr),
        src.data_ptr(),
        static_cast<size_t>(nbytes),
        reinterpret_cast<uint64_t*>(sig_addr),
        sig_val,
        sig_op,
        pe,
        reinterpret_cast<cudaStream_t>(stream_ptr)
    );
}

// =========================================================================
// Signal wait (stream-ordered)
// =========================================================================

void nvshmem_signal_wait_until_on_stream_wrapper(
    int64_t sig_addr,
    int cmp_op,
    uint64_t cmp_val,
    int64_t stream_ptr
) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    TORCH_CHECK(sig_addr != 0, "sig_addr is null");

    nvshmemx_signal_wait_until_on_stream(
        reinterpret_cast<uint64_t*>(sig_addr),
        cmp_op,
        cmp_val,
        reinterpret_cast<cudaStream_t>(stream_ptr)
    );
}

// =========================================================================
// Quiet (stream-ordered)
// =========================================================================

void nvshmem_quiet_on_stream_wrapper(int64_t stream_ptr) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    nvshmemx_quiet_on_stream(reinterpret_cast<cudaStream_t>(stream_ptr));
}

// =========================================================================
// Constants
// =========================================================================

int get_nvshmem_signal_set() { return NVSHMEM_SIGNAL_SET; }
int get_nvshmem_cmp_ge()     { return NVSHMEM_CMP_GE; }

// =========================================================================
// pybind11 module
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FluidMoE NVSHMEM P2P operations (host-only)";

    m.def("nvshmem_init", &nvshmem_init_wrapper);
    m.def("nvshmem_finalize", &nvshmem_finalize_wrapper);
    m.def("nvshmem_my_pe", &nvshmem_my_pe_wrapper);
    m.def("nvshmem_n_pes", &nvshmem_n_pes_wrapper);

    m.def("nvshmem_malloc_tensor", &nvshmem_malloc_tensor);
    m.def("nvshmem_free_tensor", &nvshmem_free_tensor);
    m.def("nvshmem_ptr", &nvshmem_ptr_wrapper);

    m.def("putmem_signal_on_stream", &nvshmem_putmem_signal_on_stream_wrapper);
    m.def("signal_wait_until_on_stream", &nvshmem_signal_wait_until_on_stream_wrapper);
    m.def("quiet_on_stream", &nvshmem_quiet_on_stream_wrapper);

    m.def("SIGNAL_SET", &get_nvshmem_signal_set);
    m.def("CMP_GE", &get_nvshmem_cmp_ge);
}
