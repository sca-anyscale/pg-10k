# ABOUTME: Orchestrates all profiling and monitoring based on environment variables.
# ABOUTME: Provides start/stop methods so benchmarks don't need per-profiler boilerplate.

import os
import time

from . import gpu_monitor, net_monitor, nsys, object_store, perf, pyspy, telemetry


class Profiling:
    """Coordinates all profiling and monitoring for a benchmark run.

    Reads configuration from environment variables:
        PROFILER_MODE:          "nsys", "torch", or "none" (default: "none")
        PROFILE_SKIP_BATCHES:   Batches to skip before nsys capture (default: 0)
        PROFILE_ACTIVE_BATCHES: Batches to capture with nsys (default: 150). Must
                                be smaller than the number of batches each Infer
                                actor processes; otherwise cudaProfilerStop never
                                fires and nsys won't finalize the .nsys-rep file
                                before the actor is killed at end of pipeline.
                                The 717K-row benchmark processes ~155–185 batches
                                per actor (varies with load-balance imbalance);
                                150 is just below the observed minimum.
        PYSPY_ENABLED:          "1" to enable py-spy (default: "0")
        PYSPY_NUM_CPU_WORKERS:  CPU worker nodes to py-spy (default: 5)
        PYSPY_NUM_GPU_WORKERS:  GPU worker nodes to py-spy (default: 5)
        PERF_PROFILING_ENABLED: "1" to enable perf record (default: "0")
        PERF_NUM_CPU_WORKERS:   CPU worker nodes to profile (default: 5)
        PERF_NUM_GPU_WORKERS:   GPU worker nodes to profile (default: 5)
        GPU_MONITOR_ENABLED:    "1" to enable nvidia-smi monitoring (default: "0")
        NET_MONITOR_ENABLED:    "1" to enable network I/O monitoring (default: "0")
        OBJECT_STORE_MONITOR_ENABLED: "1" to enable object-store sampling (default: "0")
        OBJECT_STORE_MONITOR_INTERVAL_S: seconds between samples (default: 5)
        OBJECT_STORE_MONITOR_FAST_WINDOW_S: seconds at the start of the run to
            sample at OBJECT_STORE_MONITOR_FAST_INTERVAL_S instead of
            OBJECT_STORE_MONITOR_INTERVAL_S (default: 60). Set to 0 to disable.
        OBJECT_STORE_MONITOR_FAST_INTERVAL_S: tick interval during fast window (default: 1)
        RAY_MAX_LIMIT_FROM_API_SERVER: object-store sampling defaults to a 10k
            object cap per call. Set this env var to e.g. "200000" on the head
            node to fully sample large runs (otherwise samples are truncated).
        PROFILING_STORAGE_BUCKET:    storage bucket for telemetry upload

    Usage:
        profiling = Profiling(outdir="/mnt/shared_storage/my_benchmark/job123",
                              num_gpu_nodes=40)
        profiling.start()

        infer_kwargs["runtime_env"] = profiling.nsys_runtime_env()
        # ... run benchmark ...

        profiling.stop(storage_prefix="my-benchmark/job123")
    """

    def __init__(self, outdir, num_gpu_nodes=0):
        self.outdir = outdir
        self.num_gpu_nodes = num_gpu_nodes

        self.profiler_mode = os.environ.get("PROFILER_MODE", "none")
        self.profile_skip_batches = int(os.environ.get("PROFILE_SKIP_BATCHES", "0"))
        self.profile_active_batches = int(
            os.environ.get("PROFILE_ACTIVE_BATCHES", "150")
        )
        self.pyspy_enabled = os.environ.get("PYSPY_ENABLED", "0") == "1"
        self.pyspy_num_cpu_workers = int(os.environ.get("PYSPY_NUM_CPU_WORKERS", "5"))
        self.pyspy_num_gpu_workers = int(os.environ.get("PYSPY_NUM_GPU_WORKERS", "5"))
        self.perf_enabled = os.environ.get("PERF_PROFILING_ENABLED", "0") == "1"
        self.perf_num_cpu_workers = int(os.environ.get("PERF_NUM_CPU_WORKERS", "5"))
        self.perf_num_gpu_workers = int(os.environ.get("PERF_NUM_GPU_WORKERS", "5"))
        self.gpu_monitor_enabled = os.environ.get("GPU_MONITOR_ENABLED", "0") == "1"
        self.net_monitor_enabled = os.environ.get("NET_MONITOR_ENABLED", "0") == "1"
        self.object_store_monitor_enabled = (
            os.environ.get("OBJECT_STORE_MONITOR_ENABLED", "0") == "1"
        )
        self.object_store_monitor_interval_s = int(
            os.environ.get("OBJECT_STORE_MONITOR_INTERVAL_S", "5")
        )
        self.object_store_monitor_fast_window_s = int(
            os.environ.get("OBJECT_STORE_MONITOR_FAST_WINDOW_S", "60")
        )
        self.object_store_monitor_fast_interval_s = int(
            os.environ.get("OBJECT_STORE_MONITOR_FAST_INTERVAL_S", "1")
        )

        self._worker_perf_actors = []
        self._head_perf_handles = []
        self._worker_pyspy_actors = []

    def is_enabled(self):
        """Return True if any profiling or monitoring is enabled."""
        return (
            self.profiler_mode != "none"
            or self.pyspy_enabled
            or self.perf_enabled
            or self.gpu_monitor_enabled
            or self.net_monitor_enabled
            or self.object_store_monitor_enabled
        )

    def start(self, extra_config=None):
        """Start all enabled profilers and monitors.

        Args:
            extra_config: Optional dict of extra config key/value pairs to print.
        """
        print("Configuration:")
        if extra_config:
            for key, value in extra_config.items():
                print(f"  {key}: {value}")
        print(f"  PROFILER_MODE:          {self.profiler_mode}")
        print(f"  PYSPY_ENABLED:          {self.pyspy_enabled}")
        if self.pyspy_enabled:
            print(
                f"    workers (cpu/gpu):    "
                f"{self.pyspy_num_cpu_workers}/{self.pyspy_num_gpu_workers}"
            )
        print(f"  PERF_PROFILING_ENABLED: {self.perf_enabled}")
        print(f"  GPU_MONITOR_ENABLED:    {self.gpu_monitor_enabled}")
        print(f"  NET_MONITOR_ENABLED:    {self.net_monitor_enabled}")
        print(
            f"  OBJECT_STORE_MONITOR:   {self.object_store_monitor_enabled} "
            f"(interval={self.object_store_monitor_interval_s}s, "
            f"fast_window={self.object_store_monitor_fast_window_s}s @ "
            f"{self.object_store_monitor_fast_interval_s}s)"
        )
        print(f"  SHARED_OUTDIR:          {self.outdir}")
        print()

        os.makedirs(self.outdir, exist_ok=True)

        if self.gpu_monitor_enabled and self.num_gpu_nodes > 0:
            gpu_monitor.start(self.outdir, self.num_gpu_nodes)
        if self.net_monitor_enabled:
            net_monitor.start(self.outdir)
        if self.object_store_monitor_enabled:
            object_store.start(
                self.outdir,
                interval_s=self.object_store_monitor_interval_s,
                fast_window_s=self.object_store_monitor_fast_window_s,
                fast_interval_s=self.object_store_monitor_fast_interval_s,
            )

        if self.pyspy_enabled:
            pyspy.start(self.outdir)
            if self.pyspy_num_cpu_workers > 0 or self.pyspy_num_gpu_workers > 0:
                self._worker_pyspy_actors = pyspy.start_worker_nodes(
                    self.outdir,
                    num_cpu_workers=self.pyspy_num_cpu_workers,
                    num_gpu_workers=self.pyspy_num_gpu_workers,
                )

        if self.perf_enabled:
            self._head_perf_handles = perf.start_head_node(self.outdir)
            self._worker_perf_actors = perf.start_worker_nodes(
                self.outdir,
                num_cpu_workers=self.perf_num_cpu_workers,
                num_gpu_workers=self.perf_num_gpu_workers,
            )

    def nsys_runtime_env(self):
        """Return nsys runtime_env dict if profiler_mode is "nsys", else empty dict."""
        if self.profiler_mode == "nsys":
            return nsys.runtime_env(self.outdir)
        return {}

    def stop(self, storage_prefix=None, storage_bucket=None):
        """Stop all profilers and upload telemetry to cloud storage.

        Args:
            storage_prefix: storage key prefix for telemetry upload. If None, skips upload.
            storage_bucket: storage bucket name. Defaults to PROFILING_STORAGE_BUCKET env var.
        """
        if self.profiler_mode == "nsys":
            print("Waiting 10s for nsys to flush profiling data...")
            time.sleep(10)

        if self.pyspy_enabled:
            # Worker py-spy first: each actor's stop() blocks until its
            # start() has finished attaching (Ray actor methods are serial),
            # then SIGINTs py-spy on the worker to flush output to shared
            # storage before we move on.
            pyspy.stop_workers(self._worker_pyspy_actors)
            pyspy.stop()

        if self.perf_enabled:
            # Each stop_* converts its own .data to collapsed stacks on the
            # node that produced it — worker .data symbolizes on the worker
            # (where /tmp/ray/session_* runtime libs still exist), head .data
            # on the head. A central pass on the head would symbolize worker
            # data against the head's filesystem and lose most user-space
            # symbols.
            perf.stop_workers(self._worker_perf_actors)
            perf.stop_head(self._head_perf_handles)

        if storage_prefix is not None:
            telemetry.upload(self.outdir, storage_prefix, storage_bucket=storage_bucket)
