# ------------------------------------------------------------------------------
# Fair GPU peak-memory measurement helpers.
#
# The headline metric of this project is *peak* memory during prefill (the wall
# that decides how long an initial context a method can ingest before OOM).
# `torch.cuda.max_memory_allocated()` is the right number: it is the high-water
# mark of allocator usage since the last reset, so it captures the transient
# attention/activation spikes AND the stored KV cache.
# ------------------------------------------------------------------------------
import contextlib
import torch


def reset_peak(device=0):
    """Empty cache and reset the peak counter so the next measurement is clean."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device=0):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e9


def alloc_gb(device=0):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1e9


def total_gb(device=0):
    _, total = torch.cuda.mem_get_info(device)
    return total / 1e9


@contextlib.contextmanager
def track_peak(device=0):
    """Context manager yielding a dict that gets `peak_gb` / `alloc_gb` filled in."""
    reset_peak(device)
    out = {"peak_gb": None, "alloc_gb": None}
    try:
        yield out
    finally:
        out["peak_gb"] = peak_gb(device)
        out["alloc_gb"] = alloc_gb(device)


def run_oom_safe(fn, device=0):
    """Run `fn` and report OOM gracefully.

    Returns (result, info) where info has keys: status in {ok, oom, error},
    peak_gb, alloc_gb, error (optional message). On OOM we free the cache so the
    caller can continue the length-ladder sweep with the next configuration.
    """
    reset_peak(device)
    info = {"status": "ok", "peak_gb": None, "alloc_gb": None, "error": None}
    try:
        result = fn()
        info["peak_gb"] = peak_gb(device)
        info["alloc_gb"] = alloc_gb(device)
        return result, info
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        info["status"] = "oom"
        info["peak_gb"] = peak_gb(device)
        info["error"] = str(e)[:200]
        torch.cuda.empty_cache()
        return None, info
    except RuntimeError as e:
        # Some OOMs surface as generic RuntimeError ("CUDA out of memory").
        if "out of memory" in str(e).lower():
            info["status"] = "oom"
            info["peak_gb"] = peak_gb(device)
            info["error"] = str(e)[:200]
            torch.cuda.empty_cache()
            return None, info
        info["status"] = "error"
        info["error"] = str(e)[:500]
        raise
