# ABOUTME: NVTX annotation helpers and CUDA profiler capture range control for nsys.
# ABOUTME: Provides context managers for NVTX ranges and batch-count-based profiler start/stop.

from contextlib import contextmanager


@contextmanager
def profiling_range(name):
    """Context manager that pushes/pops an NVTX range.

    Falls back to a no-op if torch.cuda.nvtx is not available.
    """
    try:
        import torch.cuda.nvtx as nvtx

        nvtx.range_push(name)
    except (ImportError, AttributeError):
        yield
        return
    try:
        yield
    finally:
        nvtx.range_pop()


def cuda_profiler_fence(call_count, skip_batches, active_batches, node_ip="",
                        outdir="/mnt/shared_storage/image_embedding_jsonl"):
    """Start/stop the CUDA profiler based on batch call count.

    Used with nsys cudaProfilerApi capture range mode. Starts profiling
    after skip_batches and stops after active_batches more.

    Args:
        call_count: Current batch call count (1-indexed).
        skip_batches: Number of batches to skip before starting capture.
        active_batches: Number of batches to capture.
        node_ip: Node IP for log messages (optional).

    Returns:
        (profiling_active, profiler_done) tuple of booleans.
    """
    import torch

    if call_count == skip_batches + 1:
        torch.cuda.cudart().cudaProfilerStart()
        print(f"[{node_ip}] nsys capture started at batch {call_count}", flush=True)
        return True, False
    elif call_count == skip_batches + active_batches + 1:
        # Verbose for nsys debug: log before+after the stop, then stat the
        # file synchronously so we can tell if `capture-range-end: stop` is
        # actually finalizing on disk.
        import glob
        import os as _os
        print(
            f"[nsys-debug {node_ip}] PRE-cudaProfilerStop @ batch {call_count}",
            flush=True,
        )
        torch.cuda.cudart().cudaProfilerStop()
        print(
            f"[nsys-debug {node_ip}] POST-cudaProfilerStop @ batch {call_count} "
            f"(nsys should finalize now)",
            flush=True,
        )
        try:
            pid = _os.getpid()
            matches = glob.glob(
                f"{outdir}/*/nsys_*_{pid}.nsys-rep"
            )
            for f in matches:
                print(
                    f"[nsys-debug {node_ip}] file_after_stop: {f} "
                    f"size={_os.path.getsize(f)}",
                    flush=True,
                )
            if not matches:
                print(
                    f"[nsys-debug {node_ip}] file_after_stop: no nsys file found "
                    f"matching pid {pid}",
                    flush=True,
                )
        except BaseException as e:
            print(
                f"[nsys-debug {node_ip}] file_after_stop stat failed: {e!r}",
                flush=True,
            )
        print(
            f"[{node_ip}] nsys capture stopped at batch {call_count} "
            f"({active_batches} batches captured)",
            flush=True,
        )
        return False, True
    return None, None
