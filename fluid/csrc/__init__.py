"""
NVSHMEM extension — JIT compiled on first import.

Auto-discovers NVSHMEM installation by searching (in order):
  1. $NVSHMEM_HOME env var
  2. ~/nvshmem (tarball install)
  3. /usr/local/nvshmem (deb/rpm install)
  4. pip-installed nvidia-nvshmem-cu* package in site-packages

No manual env var setup needed in most cases.

If NVSHMEM is not found, importing this module raises ImportError,
which triggers the NCCL fallback in fluid.core.p2p_backend.
"""

import os

_mod = None


def _find_nvshmem_home():
    """Auto-detect NVSHMEM installation path."""

    # 1. Explicit env var (highest priority)
    env = os.environ.get("NVSHMEM_HOME")
    if env and _is_valid_nvshmem(env):
        return env

    # 2. Common user/system paths
    for candidate in [
        os.path.expanduser("~/nvshmem"),
        "/usr/local/nvshmem",
    ]:
        if _is_valid_nvshmem(candidate):
            return candidate

    # 3. pip-installed nvidia-nvshmem-cu* package
    try:
        import nvidia.nvshmem
        pkg_path = list(nvidia.nvshmem.__path__)[0]
        if _is_valid_nvshmem(pkg_path):
            return pkg_path
    except (ImportError, Exception):
        pass

    return None


def _is_valid_nvshmem(path):
    """Check if path contains NVSHMEM headers and host library."""
    return (os.path.isfile(os.path.join(path, "include", "host", "nvshmem_api.h"))
            and os.path.isfile(os.path.join(path, "lib", "libnvshmem_host.so")))


def _get_ops():
    global _mod
    if _mod is not None:
        return _mod

    nvshmem_home = _find_nvshmem_home()
    if nvshmem_home is None:
        raise ImportError(
            "NVSHMEM installation not found. Searched:\n"
            "  1. $NVSHMEM_HOME env var\n"
            "  2. ~/nvshmem\n"
            "  3. /usr/local/nvshmem\n"
            "  4. pip nvidia-nvshmem-cu* package\n"
            "Install NVSHMEM or set NVSHMEM_HOME.")

    include_dir = os.path.join(nvshmem_home, "include")
    lib_dir = os.path.join(nvshmem_home, "lib")

    import torch.utils.cpp_extension as _ext
    cuda_home = getattr(_ext, "CUDA_HOME", None) or os.environ.get(
        "CUDA_HOME", "/usr/local/cuda")
    cuda_include = os.path.join(cuda_home, "include")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_file = os.path.join(src_dir, "nvshmem_ops.cpp")

    from torch.utils.cpp_extension import load

    _mod = load(
        name="fluidmoe_nvshmem_ops",
        sources=[src_file],
        extra_include_paths=[include_dir, cuda_include],
        extra_ldflags=[
            f"-L{lib_dir}",
            "-lnvshmem_host",
            "-Wl,-rpath," + lib_dir,
        ],
        verbose=False,
    )
    return _mod


def __getattr__(name):
    ops = _get_ops()
    return getattr(ops, name)
