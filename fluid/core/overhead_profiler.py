"""Scheduler-overhead profiler.

Distinct from ``python_profile.profile_section`` which wraps whole functions
(useful + overhead). This module only wraps **orchestration** code paths so
the aggregate matches the paper's T_overhead accounting (Table A.3 in the
FluidMoE appendix): launch / synchronization / bookkeeping that would not
exist without FluidMoE's schedule.

Tags use a flat namespace with a bucket prefix:
    ``overhead.fwd_*``    -> Forward tournament bucket
    ``overhead.bwd_*``    -> Backward refinement bucket
    ``overhead.ar_*``     -> Inline AR bookkeeping bucket

Any other tag is lumped into ``other``.
"""

import os
import threading
import time
from typing import Dict


ENABLED = os.environ.get("FLUID_OVERHEAD_PROFILE", "0") == "1"

_LOCK = threading.Lock()
_STATS: Dict[str, Dict[str, float]] = {}


class _NullCtx:
    """Zero-state no-op context. Single module singleton used when ENABLED=False.

    __enter__ / __exit__ each cost one method call (~50 ns) with no attribute
    writes or reads. Faster than ``contextlib.nullcontext`` which stores an
    ``enter_result`` attribute accessed on __enter__.
    """
    __slots__ = ()
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


_NULL_CTX = _NullCtx()


class _MeasureCtx:
    """Real timing context. Only instantiated when ENABLED=True."""
    __slots__ = ("_tag", "_start")

    def __init__(self, tag: str):
        self._tag = tag

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        with _LOCK:
            stat = _STATS.get(self._tag)
            if stat is None:
                _STATS[self._tag] = {"total_ms": elapsed_ms, "count": 1}
            else:
                stat["total_ms"] += elapsed_ms
                stat["count"] += 1
        return False


def measure(tag: str):
    """Return a context manager accumulating wall time for ``tag``.

    Hot paths should prefer the token-based ``enter``/``exit`` API with
    ``if ENABLED:`` guard — it achieves true zero overhead when disabled
    (~20 ns for one attribute read + one branch). ``measure()`` still works
    but costs ~400 ns/call even when disabled due to Python's ``with``
    statement floor.
    """
    if not ENABLED:
        return _NULL_CTX
    return _MeasureCtx(tag)


def enter(tag: str):
    """Begin timing for ``tag``; return a token or ``None``.

    Callers pair this with ``exit(token)`` inside a ``try/finally``. This API
    is preferred over ``with measure(...)`` at hot sites:

        if _oh.ENABLED:
            _t = _oh.enter("overhead.tag")
        try:
            body
        finally:
            if _t is not None:
                _oh.exit(_t)

    Even simpler zero-overhead form (recommended):

        _t = _oh.enter("tag") if _oh.ENABLED else None
        try:
            body
        finally:
            if _t is not None:
                _oh.exit(_t)
    """
    if not ENABLED:
        return None
    return (tag, time.perf_counter())


def exit(token) -> None:
    """Finalize a token returned by ``enter`` and record stats."""
    if token is None:
        return
    tag, start = token
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _LOCK:
        stat = _STATS.get(tag)
        if stat is None:
            _STATS[tag] = {"total_ms": elapsed_ms, "count": 1}
        else:
            stat["total_ms"] += elapsed_ms
            stat["count"] += 1


def set_enabled(enabled: bool) -> None:
    """Enable/disable at runtime; env var is the default at import time."""
    global ENABLED
    ENABLED = bool(enabled)


def reset() -> None:
    with _LOCK:
        _STATS.clear()


def get_stats() -> Dict[str, Dict[str, float]]:
    """Raw per-tag stats: {tag: {total_ms, count}}."""
    with _LOCK:
        return {k: {"total_ms": v["total_ms"], "count": v["count"]}
                for k, v in _STATS.items()}


def summary_by_bucket(num_iters: int = 1) -> Dict[str, float]:
    """Aggregate raw per-tag stats into the four paper-table buckets.

    Args:
        num_iters: Divide each bucket by this to get per-iter ms. Use 1 if
            you want raw totals across measurement window.

    Returns:
        dict with keys ``fwd_tournament``, ``bwd_refinement``,
        ``ar_bookkeeping``, ``other``, ``total`` (all ms, per-iter if
        ``num_iters > 1``).
    """
    n = max(num_iters, 1)
    buckets = {
        "fwd_tournament": 0.0,
        "bwd_refinement": 0.0,
        "ar_bookkeeping": 0.0,
        "other": 0.0,
    }
    with _LOCK:
        for tag, st in _STATS.items():
            ms = st["total_ms"] / n
            if tag.startswith("overhead.fwd"):
                buckets["fwd_tournament"] += ms
            elif tag.startswith("overhead.bwd"):
                buckets["bwd_refinement"] += ms
            elif tag.startswith("overhead.ar"):
                buckets["ar_bookkeeping"] += ms
            else:
                buckets["other"] += ms
    buckets["total"] = sum(buckets.values())
    return buckets


__all__ = [
    "ENABLED",
    "set_enabled",
    "reset",
    "measure",
    "get_stats",
    "summary_by_bucket",
]
