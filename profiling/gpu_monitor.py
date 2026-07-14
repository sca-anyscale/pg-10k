# ABOUTME: Launches nvidia-smi monitoring on GPU worker nodes as they join the cluster.
# ABOUTME: Polls for new GPU nodes and starts nvidia-smi dmon on each via long-lived Ray actors.

import glob
import os
import threading
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0, num_gpus=0)
class _NvidiaSmiActor:
    """Long-lived actor that owns an nvidia-smi dmon subprocess on its node.

    Holding nvidia-smi inside an actor (rather than launching it from a Ray
    task) keeps the parent process alive for the lifetime of the actor handle.
    Tasks return immediately and the worker is reaped after ~3 min idle,
    taking nvidia-smi with it; the actor never goes idle, so the subprocess
    survives the whole run.
    """

    def __init__(self, outdir):
        import subprocess

        os.makedirs(outdir, exist_ok=True)
        node_ip = os.environ.get("ANYSCALE_NODE_IP", "unknown").replace(".", "_")
        outfile = f"{outdir}/gpu_usage_{node_ip}.txt"
        self._proc = subprocess.Popen(
            ["stdbuf", "-oL", "nvidia-smi", "dmon", "-s", "u", "-o", "T"],
            stdout=open(outfile, "w"),
            stderr=subprocess.STDOUT,
        )

    def get_pid(self):
        return self._proc.pid


def _gpu_monitor_loop(outdir, num_gpu_nodes):
    """Poll for GPU nodes and launch nvidia-smi on each as it joins.

    Monitors continuously until num_gpu_nodes are found or no new nodes
    appear for 60 seconds, whichever comes first. This avoids blocking
    forever when the cluster has fewer GPU nodes than expected.
    """
    monitored_node_ids = set()
    # Hold actor handles so Ray doesn't GC the actors mid-run. Each handle
    # keeps an nvidia-smi subprocess alive on its assigned GPU node.
    actor_handles = []
    stale_polls = 0
    max_stale_polls = 30  # 30 * 2s = 60s with no new nodes

    while len(monitored_node_ids) < num_gpu_nodes:
        gpu_nodes = [
            n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0
        ]
        new_nodes = [n for n in gpu_nodes if n["NodeID"] not in monitored_node_ids]
        if not new_nodes:
            stale_polls += 1
            if monitored_node_ids and stale_polls >= max_stale_polls:
                print(
                    f"nvidia-smi: no new GPU nodes for {max_stale_polls * 2}s, "
                    f"proceeding with {len(monitored_node_ids)}/{num_gpu_nodes}"
                )
                break
        else:
            stale_polls = 0
        for node in new_nodes:
            try:
                handle = _NvidiaSmiActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"], soft=False
                    )
                ).remote(outdir)
                child_pid = ray.get(handle.get_pid.remote())
                actor_handles.append(handle)
                monitored_node_ids.add(node["NodeID"])
                print(
                    f"Started nvidia-smi on GPU node {node['NodeManagerAddress']} "
                    f"(pid={child_pid}, {len(monitored_node_ids)}/{num_gpu_nodes})"
                )
            except Exception as e:
                monitored_node_ids.add(node["NodeID"])
                print(
                    f"Failed to start nvidia-smi on {node['NodeManagerAddress']}: {e}"
                )
        time.sleep(2)
    if len(monitored_node_ids) >= num_gpu_nodes:
        print(f"nvidia-smi monitoring active on all {num_gpu_nodes} GPU nodes")
    _health_check_loop(outdir, expected=len(monitored_node_ids))


def _health_check_loop(outdir, expected, interval_s=60, stale_threshold_s=30):
    """Detect nvidia-smi processes that have died mid-run.

    Each gpu_usage_*.txt file's mtime should advance every ~1s while nvidia-smi
    dmon is sampling. If a file's mtime hasn't moved for more than
    stale_threshold_s, the monitor on that node has stopped. We log once per
    stalled file to avoid spamming.

    Runs forever; the daemon thread exits when the driver process does.
    """
    last_mtime = {}  # path -> last observed mtime
    warned = set()   # paths we've already warned about
    while True:
        time.sleep(interval_s)
        now = time.time()
        try:
            paths = sorted(glob.glob(os.path.join(outdir, "gpu_usage_*.txt")))
        except OSError:
            continue
        if not paths:
            continue
        if len(paths) < expected and not warned:
            print(
                f"gpu_monitor health: only {len(paths)}/{expected} gpu_usage "
                f"files present in {outdir}"
            )
        for path in paths:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            prev = last_mtime.get(path)
            last_mtime[path] = mtime
            if prev is None:
                continue
            if mtime == prev and (now - mtime) > stale_threshold_s and path not in warned:
                age = now - mtime
                print(
                    f"gpu_monitor health: WARNING — {path} has not been written "
                    f"for {age:.0f}s. nvidia-smi on that node is likely dead "
                    f"(or the node was killed). Last sample at "
                    f"{time.strftime('%H:%M:%S', time.localtime(mtime))}."
                )
                warned.add(path)


def start(outdir, num_gpu_nodes):
    """Start nvidia-smi monitoring on GPU nodes in a background thread.

    Args:
        outdir: Shared storage directory for output files.
        num_gpu_nodes: Expected number of GPU nodes to monitor.
    """
    thread = threading.Thread(
        target=_gpu_monitor_loop, args=(outdir, num_gpu_nodes), daemon=True
    )
    thread.start()
    return thread
