"""eapg no-pg 味：同一载具去掉 pg——39,960 个 num_cpus=0.5 actor 直接突发提交
（默认 hybrid 调度，无 placement group），精确占满 target×2 CPU。
口径：t0 = 首个 create 前；T_e2e = 最后一个 actor ping 活 - t0。
Emits: NODES / GATE_PLAIN / ACTOR_SUBMIT / PROGRESS / E2E_WALL= or DNF /
       NODE_CENSUS / MISMATCH= / RESULT_WRITTEN
Result json: ~/default/eapgnp_result.json
python eapg_nopg.py --target-workers N --timeout CAP
"""
import argparse
import json
import time
from collections import Counter

import ray

import eapg_core

PER_NODE = eapg_core.PER_NODE  # 4 (= 2 CPU / 0.5 CPU per actor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-workers", type=int, required=True)
    p.add_argument("--settle", type=int, default=60)
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()

    ray.init(address="auto")
    eapg_core.wait_fill(a.target_workers)
    total_cpu = sum(x.get("Resources", {}).get("CPU", 0)
                    for x in ray.nodes() if x.get("Alive"))
    print("GATE_PLAIN total_cpu=%.0f need=%.0f" %
          (total_cpu, a.target_workers * 2.0), flush=True)
    assert total_cpu >= a.target_workers * 2.0
    time.sleep(a.settle)

    n_actors = a.target_workers * PER_NODE
    t0 = time.perf_counter()
    print("ACTOR_SUBMIT n=%d t=%.3f" % (n_actors, time.time()), flush=True)
    actors = [eapg_core.A.remote() for _ in range(n_actors)]
    pend = {act.ping.remote(): i for i, act in enumerate(actors)}
    ready_nodes = {}
    e2e_dnf = False
    last_print = 0.0
    while pend:
        if time.perf_counter() - t0 > a.timeout:
            e2e_dnf = True
            print("DNF phase=e2e ready=%d/%d at=%.1f" %
                  (len(ready_nodes), n_actors, time.perf_counter() - t0), flush=True)
            break
        done, _ = ray.wait(list(pend.keys()), num_returns=min(500, len(pend)), timeout=5)
        for ref in done:
            i = pend.pop(ref)
            try:
                ready_nodes[i] = ray.get(ref)
            except Exception:
                ready_nodes[i] = None
        now = time.perf_counter()
        if now - last_print >= 5:
            print("PROGRESS ready=%d/%d t=%.1f" %
                  (len(ready_nodes), n_actors, now - t0), flush=True)
            last_print = now
    t_e2e = time.perf_counter() - t0
    if not e2e_dnf:
        print("E2E_WALL=%.1f actors=%d" % (t_e2e, n_actors), flush=True)

    # 结构量：全员活 + 每节点 actor 数不越资源账（2 CPU / 0.5 = 4）。
    mismatch = -1
    if not e2e_dnf:
        per_node = Counter(nid for nid in ready_nodes.values() if nid)
        overpacked = sum(1 for c in per_node.values() if c > PER_NODE)
        dead = sum(1 for nid in ready_nodes.values() if nid is None)
        mismatch = dead + overpacked
        print("NODE_CENSUS nodes_used=%d max_per_node=%d overpacked=%d dead=%d" %
              (len(per_node), max(per_node.values(), default=0), overpacked, dead),
              flush=True)
        print("MISMATCH=%d" % mismatch, flush=True)

    result = {
        "experiment": "eapg_nopg", "flavor": "nopg",
        "target_workers": a.target_workers, "racks": 0, "num_actors": n_actors,
        "pg_wall_s": 0.0, "e2e_wall_s": round(t_e2e, 1),
        "pg_dnf": False, "e2e_dnf": e2e_dnf, "pg_ready_at_end": 0,
        "ready_at_end": len(ready_nodes), "mismatch": mismatch,
        "ray_version": ray.__version__,
        "pg_timeout": 0, "e2e_timeout": a.timeout,
    }
    with open("/home/ray/default/eapgnp_result.json", "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT_WRITTEN", flush=True)


main()
