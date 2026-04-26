/*
 * FluidMoE NVSHMEM P2P Operations (host-only)
 *
 * Wraps NVSHMEM host-side stream-ordered APIs for P2P communication.
 * No device code (__global__ kernels) — pure C++ compiled by the system
 * compiler, not nvcc. This avoids all PyTorch half-operator conflicts.
 *
 * Links against: libnvshmem_host.so only.
 * Headers:       host/nvshmem_api.h, host/nvshmemx_api.h, plus
 *                bootstrap_device_host/nvshmem_uniqueid.h for UID bootstrap.
 *
 * Initialization
 * --------------
 * NVSHMEM 3.x exposes two host-only init paths through libnvshmem_host.so:
 *
 *   1. ``nvshmemx_hostlib_init_attr(flags=0, &attr)``
 *      Bootstrap is selected from $NVSHMEM_BOOTSTRAP (PMI/MPI/SHMEM/plugin).
 *      Requires a launcher that exports the corresponding bootstrap server
 *      env vars (PMIX_SERVER_URI for PMI, MPI_Init beforehand for MPI, ...).
 *      The header's static-inline ``nvshmemx_init_attr`` auto-fills the
 *      ``attr.version`` field, but ``nvshmemx_hostlib_init_attr`` does NOT —
 *      we use the ``nvshmemx_init_init_attr_ver_only`` macro to populate it.
 *
 *   2. ``nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr)``
 *      Programmatic bootstrap: rank 0 calls ``nvshmemx_get_uniqueid`` to
 *      mint a 128-byte UID, the application broadcasts it to all PEs through
 *      a side channel (e.g. torch.distributed.broadcast), then every PE calls
 *      ``nvshmemx_set_attr_uniqueid_args(my_pe, n_pes, &uid, &attr)`` and
 *      this init. Required when the launcher is torchrun (no PMI server).
 */

#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <device_host/nvshmem_types.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstring>
#include <string>

// =========================================================================
// Lifecycle
// =========================================================================

static bool g_initialized = false;

static void check_status(int status, const char *what) {
    TORCH_CHECK(status == 0, what, " failed with status ", status);
}

// Env-based init (PMI/MPI/SHMEM via $NVSHMEM_BOOTSTRAP). Caller is responsible
// for setting the env var BEFORE calling this. flags=0 means "use whatever
// $NVSHMEM_BOOTSTRAP says". Populates ``attr.version`` correctly; without that
// hostlib_init_attr can fail or trigger UB on version-checked code paths.
void nvshmem_init_wrapper() {
    if (g_initialized) return;
    nvshmemx_init_attr_t attr;
    nvshmemx_init_init_attr_ver_only(attr);  // sets attr.version + sub-versions
    int status = nvshmemx_hostlib_init_attr(0, &attr);
    check_status(status, "nvshmemx_hostlib_init_attr (env bootstrap)");
    g_initialized = true;
}

// UID bootstrap. ``uniqueid_bytes`` must be exactly sizeof(nvshmemx_uniqueid_t)
// (128 bytes) and must be the same on all ranks (broadcast from rank 0).
// my_pe / n_pes are ranks within the NVSHMEM world team.
void nvshmem_init_with_uniqueid_wrapper(int my_pe, int n_pes,
                                        const std::string &uniqueid_bytes) {
    if (g_initialized) return;
    TORCH_CHECK(uniqueid_bytes.size() == sizeof(nvshmemx_uniqueid_t),
                "uniqueid_bytes must be ", sizeof(nvshmemx_uniqueid_t),
                " bytes (got ", uniqueid_bytes.size(), ")");
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uniqueid_bytes.data(), sizeof(nvshmemx_uniqueid_t));

    nvshmemx_init_attr_t attr;
    nvshmemx_init_init_attr_ver_only(attr);
    int status = nvshmemx_set_attr_uniqueid_args(my_pe, n_pes, &uid, &attr);
    check_status(status, "nvshmemx_set_attr_uniqueid_args");

    status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    check_status(status, "nvshmemx_hostlib_init_attr (UID bootstrap)");
    g_initialized = true;
}

// Mint a UID (rank 0 only). Returned as raw bytes; caller broadcasts.
py::bytes nvshmem_get_uniqueid_wrapper() {
    nvshmemx_uniqueid_t uid;
    int status = nvshmemx_get_uniqueid(&uid);
    check_status(status, "nvshmemx_get_uniqueid");
    return py::bytes(reinterpret_cast<const char *>(&uid),
                     sizeof(nvshmemx_uniqueid_t));
}

// Size of the UID payload — Python side uses this to size the broadcast tensor
// rather than hardcoding 128.
int64_t nvshmem_uniqueid_size() {
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

void nvshmem_finalize_wrapper() {
    if (!g_initialized) return;
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
// Symmetric heap allocation → PyTorch tensor (NON-OWNING)
//
// The returned tensor is a ``torch::from_blob`` view over the symmetric
// allocation; it has NO custom deleter, so dropping the tensor does NOT free
// the underlying NVSHMEM memory. Lifetime must be managed explicitly:
//
//   - The caller is responsible for retaining the master tensor in a cache
//     and calling ``nvshmem_free_tensor(master)`` collectively across all PEs
//     before reallocating or finalizing.
//   - We deliberately do NOT install a deleter that calls ``nvshmem_free``,
//     because ``nvshmem_free`` is a collective with an entry barrier — running
//     it from a non-collective context (Python GC, exception teardown) would
//     deadlock.
//   - We do NOT zero-initialize. ``cudaMemset`` on the default stream right
//     after a collective ``nvshmem_malloc`` races against any remote PE that
//     resolves the symmetric address and issues a put before the memset
//     completes. Recv buffers in production are fully overwritten by the
//     incoming put before being read; if a caller needs zero-init, it must
//     do so explicitly with proper stream synchronization and a barrier
//     before remote PEs can see the zeros.
// =========================================================================

torch::Tensor nvshmem_malloc_tensor(int64_t numel, c10::ScalarType dtype) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    TORCH_CHECK(numel > 0, "numel must be > 0");

    auto elem_size = c10::elementSize(dtype);
    size_t nbytes = static_cast<size_t>(numel) * elem_size;

    // Collective: all PEs in NVSHMEM_TEAM_WORLD must call with identical size.
    // Has entry+exit barrier — return = remote PEs may resolve the symmetric
    // address.
    void* ptr = nvshmem_malloc(nbytes);
    TORCH_CHECK(ptr != nullptr, "nvshmem_malloc failed for ", nbytes, " bytes");

    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, cur_dev);
    return torch::from_blob(ptr, {numel}, options);
}

void nvshmem_free_tensor(torch::Tensor t) {
    TORCH_CHECK(g_initialized, "nvshmem not initialized");
    // Collective with entry barrier — caller must invoke this on all PEs in
    // lockstep, with the master (non-sliced) tensor.
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
//
// Semantic guarantee from NVSHMEM spec:
//   - When the stream operation completes locally, the source data has been
//     read out of ``src``. PyTorch ``record_stream`` is required so the
//     caching allocator does not recycle ``src`` storage early.
//   - The signal write on the remote PE is ordered AFTER the data delivery.
//     Receiver-side ``signal_wait_until_on_stream`` therefore acts as an
//     acquire fence: when the wait completes, the put data is visible to
//     subsequent kernels on the same stream.
//   - For sender-side completion guarantees on RDMA transports (cross-node IB),
//     a stream-ordered ``nvshmemx_quiet_on_stream`` is required before the
//     source buffer is reused. NVLink-only paths satisfy this implicitly.
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
// Signal wait (stream-ordered) — blocks the stream until *sig_addr cmp cmp_val
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
// Quiet (stream-ordered) — drains all outstanding RDMA puts at completion
// of the queued op. Required before reusing send buffers on RDMA transports.
// On pure NVLink with cuMemcpy-backed put_signal it is a no-op but harmless.
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

    // Init / finalize
    m.def("nvshmem_init", &nvshmem_init_wrapper);
    m.def("nvshmem_init_with_uniqueid", &nvshmem_init_with_uniqueid_wrapper,
          py::arg("my_pe"), py::arg("n_pes"), py::arg("uniqueid_bytes"));
    m.def("nvshmem_get_uniqueid", &nvshmem_get_uniqueid_wrapper);
    m.def("nvshmem_uniqueid_size", &nvshmem_uniqueid_size);
    m.def("nvshmem_finalize", &nvshmem_finalize_wrapper);
    m.def("nvshmem_my_pe", &nvshmem_my_pe_wrapper);
    m.def("nvshmem_n_pes", &nvshmem_n_pes_wrapper);

    // Symmetric heap (non-owning tensor — caller manages collective free)
    m.def("nvshmem_malloc_tensor", &nvshmem_malloc_tensor);
    m.def("nvshmem_free_tensor", &nvshmem_free_tensor);
    m.def("nvshmem_ptr", &nvshmem_ptr_wrapper);

    // Stream-ordered RMA + signal
    m.def("putmem_signal_on_stream", &nvshmem_putmem_signal_on_stream_wrapper);
    m.def("signal_wait_until_on_stream", &nvshmem_signal_wait_until_on_stream_wrapper);
    m.def("quiet_on_stream", &nvshmem_quiet_on_stream_wrapper);

    // Constants
    m.def("SIGNAL_SET", &get_nvshmem_signal_set);
    m.def("CMP_GE", &get_nvshmem_cmp_ge);
}
