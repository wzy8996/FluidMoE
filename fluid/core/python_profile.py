import os
import threading
import time
from contextlib import contextmanager


PYTHON_PROFILE_ENABLED = os.environ.get("FLUID_PROFILE_PYTHON", "0") == "1"

_PROFILE_LOCK = threading.Lock()
_PROFILE_STATS = {}


@contextmanager
def profile_section(name: str):
    if not PYTHON_PROFILE_ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with _PROFILE_LOCK:
            stat = _PROFILE_STATS.get(name)
            if stat is None:
                _PROFILE_STATS[name] = {"total_ms": elapsed_ms, "count": 1}
            else:
                stat["total_ms"] += elapsed_ms
                stat["count"] += 1


def reset_profile_stats():
    with _PROFILE_LOCK:
        _PROFILE_STATS.clear()



def get_profile_stats():
    with _PROFILE_LOCK:
        return {
            name: {"total_ms": stat["total_ms"], "count": stat["count"]}
            for name, stat in _PROFILE_STATS.items()
        }
