"""Hello world on the packed second cluster: one task per node, prove scheduling works.

Run on the workspace head after launch_second_cluster.py printed SECOND_CLUSTER_READY:
    RAY_ADDRESS=$(cat /tmp/ray2nd/gcs_addr) python hello_world.py
Replace this file with your real workload; keep the RAY_ADDRESS line.
"""
import os
import time

import ray

ray.init(address=os.environ["RAY_ADDRESS"])

alive = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
print(f"HELLO_NODES alive={len(alive)}", flush=True)


@ray.remote(num_cpus=2)  # each packed raylet declares 2 CPUs -> at most one task per node
def whoami():
    time.sleep(2)  # hold the CPUs so all tasks must run concurrently, one per node
    return ray.get_runtime_context().get_node_id()


t0 = time.time()
node_ids = set(ray.get([whoami.remote() for _ in alive]))
print(f"HELLO_DONE tasks={len(alive)} distinct_nodes={len(node_ids)} wall_s={time.time() - t0:.1f}",
      flush=True)
assert len(node_ids) == len(alive), "tasks did not spread to every node"
