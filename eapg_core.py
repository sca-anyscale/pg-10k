"""eapg 公共核心：PG 就位（并行 ready）→ actor 突发进组（wildcard）→ e2e 停表。

两臂共享全部量测/收割/核验逻辑，只有 build_pg(r) 和 verify_rack(actor_node, r, nodes_map)
由 flavor 侧注入。口径（递单 v2）：
  t0 = 首个 create 前；T_pg = 并行 ready 全返 - t0（里程碑，DNF 3600s）；
  T_e2e = 最后一个 actor ping 活 - t0（headline，DNF 7200s）。
Emits: NODES / GATE_* / PG_SUBMIT / PG_WALL= / ACTOR_SUBMIT / PROGRESS / E2E_WALL= or DNF / MISMATCH= / RESULT_WRITTEN
Result json: ~/default/eapg_result.json
"""
import json
import time

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

RACK = 18
PER_NODE = 4
PER_RACK = RACK * PER_NODE  # 72
PG_TIMEOUT = 3600
E2E_TIMEOUT = 7200


@ray.remote(num_cpus=0.5)
class A:
    def ping(self):
        return ray.get_runtime_context().get_node_id()


def wait_fill(target, worker_cpus=2.0):
    while True:
        n = sum(1 for x in ray.nodes()
                if x.get("Alive") and x.get("Resources", {}).get("CPU") == worker_cpus)
        print("NODES alive_workers=%d target=%d" % (n, target), flush=True)
        if n >= target:
            return
        time.sleep(30)


def run(flavor, build_pg, verify_actor_node, gate_check, target_workers, settle=60, outkey="eapg", e2e_timeout=None, post_verify=None):
    e2e_to = e2e_timeout or E2E_TIMEOUT
    ray.init(address="auto")
    wait_fill(target_workers)
    gate_check()  # flavor 侧闸门（label 可见性 / rack 资源可见性），失败抛异常
    time.sleep(settle)

    racks = target_workers // RACK
    n_actors = target_workers * PER_NODE

    t0 = time.perf_counter()
    print("PG_SUBMIT racks=%d t=%.3f" % (racks, time.time()), flush=True)
    pgs = [build_pg(r) for r in range(racks)]
    readies = [pg.ready() for pg in pgs]
    pg_dnf = False
    pg_ready_end = 0
    pend = list(readies)
    last_print = 0.0
    while pend:
        if time.perf_counter() - t0 > PG_TIMEOUT:
            pg_dnf = True
            print("DNF phase=pg ready=%d/%d at=%.1f" % (racks - len(pend), racks,
                  time.perf_counter() - t0), flush=True)
            break
        done, pend = ray.wait(pend, num_returns=min(50, len(pend)), timeout=5)
        pg_ready_end = racks - len(pend)
        now = time.perf_counter()
        if now - last_print >= 5:
            print("PROGRESS pg_ready=%d/%d t=%.1f" % (racks - len(pend), racks, now - t0), flush=True)
            last_print = now
    t_pg = time.perf_counter() - t0
    if not pg_dnf:
        print("PG_WALL=%.1f racks=%d" % (t_pg, racks), flush=True)

    actors, ready_nodes = [], {}
    e2e_dnf = False
    if not pg_dnf:
        print("ACTOR_SUBMIT n=%d t=%.3f" % (n_actors, time.time()), flush=True)
        actors = [
            A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pgs[k // PER_RACK])).remote()
            for k in range(n_actors)
        ]
        pend = {a.ping.remote(): i for i, a in enumerate(actors)}
        last_print = 0.0
        while pend:
            if time.perf_counter() - t0 > e2e_to:
                e2e_dnf = True
                print("DNF phase=e2e ready=%d/%d at=%.1f" % (len(ready_nodes), n_actors,
                      time.perf_counter() - t0), flush=True)
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
                print("PROGRESS ready=%d/%d t=%.1f" % (len(ready_nodes), n_actors, now - t0), flush=True)
                last_print = now
    t_e2e = time.perf_counter() - t0
    if not (pg_dnf or e2e_dnf):
        print("E2E_WALL=%.1f actors=%d" % (t_e2e, n_actors), flush=True)

    mismatch = -1
    if not (pg_dnf or e2e_dnf):
        nodes_map = {x["NodeID"]: x for x in ray.nodes()}
        mismatch = 0
        for i, nid in ready_nodes.items():
            r = i // PER_RACK
            if nid is None or not verify_actor_node(nodes_map.get(nid, {}), r):
                mismatch += 1
        if post_verify is not None:
            try:
                mismatch += post_verify(pgs, ready_nodes, nodes_map)
            except Exception as ex:
                print("POST_VERIFY_ERR %s" % str(ex)[:200], flush=True)
                mismatch = -2
        print("MISMATCH=%d" % mismatch, flush=True)

    result = {
        "experiment": "eapg", "flavor": flavor,
        "target_workers": target_workers, "racks": racks, "num_actors": n_actors,
        "pg_wall_s": round(t_pg, 1), "e2e_wall_s": round(t_e2e, 1),
        "pg_dnf": pg_dnf, "e2e_dnf": e2e_dnf, "pg_ready_at_end": pg_ready_end,
        "ready_at_end": len(ready_nodes), "mismatch": mismatch,
        "ray_version": ray.__version__,
        "pg_timeout": PG_TIMEOUT, "e2e_timeout": e2e_to,
    }
    with open("/home/ray/default/%s_result.json" % outkey, "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT_WRITTEN", flush=True)
