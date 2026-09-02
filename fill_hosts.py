"""Demand-driven host fill gate for packbench (default provisioning since 2026-08-29).

Used with a "flex" compute config (many instance-type groups, each min=0/max=N):
asserts autoscaler demand for --hosts worker hosts and waits until that many are
alive, letting the platform buy whichever types have capacity. On a static
min=max config the count is already satisfied, so runcell runs this gate
unconditionally and it returns immediately there.

Notes:
- Demand must be expressed as bundles: num_cpus demand above ~10k is silently
  dropped (anyscale-capacity.md). 833 hosts x 16 CPU = 13,328 > 10k.
- The platform head can absorb part of the bundle demand (it has real CPUs), so
  we over-request by a small pad; launch_second_cluster trims to --hosts.
- The demand stays asserted after exit on purpose: it prevents the autoscaler
  from reclaiming idle hosts in the window before the detached Packer actors
  pin them.
"""
import argparse
import sys
import time

import ray
from ray.autoscaler.sdk import request_resources

p = argparse.ArgumentParser()
p.add_argument("--hosts", type=int, required=True)
p.add_argument("--cpus-per-host", type=int, default=16)
p.add_argument("--timeout-s", type=int, default=2400)
p.add_argument("--poll-s", type=int, default=20)
args = p.parse_args()

ray.init(address="auto", log_to_driver=False)
my_node = ray.get_runtime_context().get_node_id()

HEAD_PAD = 4
request_resources(bundles=[{"CPU": args.cpus_per_host}] * (args.hosts + HEAD_PAD))
print(f"DEMAND_ASSERTED bundles={args.hosts + HEAD_PAD}x{{CPU:{args.cpus_per_host}}}",
      flush=True)

t0 = time.time()
last = -1
while True:
    workers = [n for n in ray.nodes() if n["Alive"] and n["NodeID"] != my_node]
    if len(workers) != last:
        comp = {}
        for n in workers:
            g = (n.get("Labels") or {}).get("ray.io/node-group", "?")
            comp[g] = comp.get(g, 0) + 1
        print(f"FILL_POLL have={len(workers)}/{args.hosts} "
              f"comp={dict(sorted(comp.items()))}", flush=True)
        last = len(workers)
    if len(workers) >= args.hosts:
        print(f"FILL_OK n={len(workers)}", flush=True)
        sys.exit(0)
    if time.time() - t0 > args.timeout_s:
        print(f"FILL_TIMEOUT have={len(workers)}/{args.hosts}", flush=True)
        sys.exit(1)
    time.sleep(args.poll_s)
