"""Start a second, platform-blind Ray cluster with N raylets packed per host.

Runs on the workspace head node, connected to the PLATFORM cluster (address="auto").
  1. (--wipe, default) kill any previous second-cluster processes (their cmdlines
     all contain the temp root) and clear the temp root, on head and every host.
  2. Start a second GCS on this head:  ray start --head --port=<gcs-port>, own temp dir.
  3. One DETACHED orchestrator actor per platform worker host (NodeAffinity, hard),
     each spawns --raylets-per-host `ray start --address=<head>:<gcs-port>` joins,
     with per-raylet temp dir, unique dashboard-agent port, and a rack label
     (rack = global_index // rack-size).
     Every subprocess uses start_new_session=True (setsid): the platform raylet's
     process-group cleanup (killpg on worker exit) can never reach the second
     cluster's process tree. The actor also stays alive (detached) as double cover.
  4. Poll the second cluster until all raylets joined; print rack census.

Env hygiene (both the head-side subprocesses and the Packer actors): strip
  RAY_ADDRESS            — subprocess CLIs must not latch onto the platform cluster
  RAY_OVERRIDE_RESOURCES — Anyscale injects it into worker envs; env values take
                           precedence over --num-cpus/--memory in
                           resource_and_label_spec._merge_resources, which silently
                           turned every packed raylet into a CPU:16 node (P0b bug).

Writes /tmp/ray2nd/gcs_addr (ip:port) for later drivers.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

RACK_KEY = "ray.io/gpu-domain"
# Strip ONLY these two. Everything else is deliberately passed through — in
# particular RAY_BACKEND_LOG_JSON (Anyscale-injected), which wait_drain.py's
# JSON-lines parsing of the second GCS's gcs_server.out depends on.
STRIP_ENV = ("RAY_ADDRESS", "RAY_OVERRIDE_RESOURCES")

p = argparse.ArgumentParser()
p.add_argument("--hosts", type=int, required=True)
p.add_argument("--raylets-per-host", type=int, required=True)
p.add_argument("--rack-size", type=int, default=18)
p.add_argument("--gcs-port", type=int, default=6479)
p.add_argument("--cpus-per-raylet", type=int, default=2)
p.add_argument("--mem-per-raylet", type=int, default=8 * 1024**3)
p.add_argument("--objstore-per-raylet", type=int, default=78643200)  # 75MB floor
p.add_argument("--tmp-root", default="/tmp/ray2nd")
p.add_argument("--join-timeout", type=int, default=900)
p.add_argument("--no-wipe", action="store_true")
args = p.parse_args()


def clean_env():
    env = dict(os.environ)
    for k in STRIP_ENV:
        env.pop(k, None)
    return env


def wipe_local(tmp_root):
    # Bracket trick so pkill's own cmdline never matches the pattern.
    frag = tmp_root.lstrip("/")
    pat = frag[:-1] + "[" + frag[-1] + "]"
    subprocess.run(["pkill", "-f", pat], check=False)
    time.sleep(1)
    subprocess.run(["rm", "-rf", tmp_root], check=False)


def second_cluster_nodes(addr):
    """Count/list alive nodes of the second cluster from a throwaway subprocess
    (this process is already connected to the platform cluster)."""
    code = (
        "import ray,json;"
        f"ray.init(address='{addr}',logging_level='error');"
        "ns=[{'ip':n['NodeManagerAddress'],'labels':n.get('Labels',{}),"
        "'cpu':n['Resources'].get('CPU')}"
        " for n in ray.nodes() if n['Alive']];"
        "print(json.dumps(ns))"
    )
    out = subprocess.run([sys.executable, "-c", code], env=clean_env(),
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None, out.stderr[-500:]
    return json.loads(out.stdout.strip().splitlines()[-1]), None


@ray.remote(num_cpus=0)
class Packer:
    """Per-host orchestrator. Lives on one platform worker; starts and shields
    the packed raylets of the second cluster on that host."""

    def wipe(self, tmp_root):
        import subprocess as _sp
        import time as _t
        frag = tmp_root.lstrip("/")
        pat = frag[:-1] + "[" + frag[-1] + "]"
        _sp.run(["pkill", "-f", pat], check=False)
        _t.sleep(1)
        _sp.run(["rm", "-rf", tmp_root], check=False)
        return True

    def start_raylets(self, head_addr, specs):
        import os as _os
        import subprocess as _sp
        env = dict(_os.environ)
        for k in STRIP_ENV:
            env.pop(k, None)
        results = []
        for spec in specs:
            cmd = [
                "ray", "start",
                f"--address={head_addr}",
                f"--num-cpus={spec['cpus']}",
                f"--memory={spec['mem']}",
                f"--object-store-memory={spec['objstore']}",
                f"--temp-dir={spec['tmp']}",
                f"--labels={RACK_KEY}={spec['rack']}",
                "--min-worker-port=0",
                "--max-worker-port=0",
                f"--dashboard-agent-listen-port={spec['agent_port']}",
                "--disable-usage-stats",
            ]
            try:
                r = _sp.run(cmd, env=env, start_new_session=True,
                            capture_output=True, text=True, timeout=240)
                results.append({"tmp": spec["tmp"], "rc": r.returncode,
                                "tail": (r.stdout + r.stderr)[-500:]})
            except Exception as e:  # timeout etc. — report, keep going
                results.append({"tmp": spec["tmp"], "rc": -1, "tail": repr(e)})
        return results

    def sanity(self):
        import socket
        return socket.gethostname()


ray.init(address="auto", namespace="packbench")  # platform cluster

head_ip = ray.util.get_node_ip_address()
gcs_addr = f"{head_ip}:{args.gcs_port}"

# ---- platform worker hosts ----
my_node = ray.get_runtime_context().get_node_id()
hosts = [n for n in ray.nodes() if n["Alive"] and n["NodeID"] != my_node]
assert len(hosts) >= args.hosts, f"platform workers {len(hosts)} < {args.hosts}"
hosts = sorted(hosts, key=lambda n: n["NodeManagerAddress"])[: args.hosts]

packers = []
for j, h in enumerate(hosts):
    name = f"packer-{j}"
    try:
        old = ray.get_actor(name)
        ray.kill(old)
        time.sleep(1)
    except ValueError:
        pass
    a = Packer.options(
        name=name, lifetime="detached",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=h["NodeID"], soft=False),
    ).remote()
    packers.append((j, h["NodeManagerAddress"], a))
ray.get([a.sanity.remote() for _, _, a in packers])
print(f"PACKERS_UP n={len(packers)}", flush=True)

# ---- 1. wipe previous second cluster (idempotent) ----
if not args.no_wipe:
    wipe_local(args.tmp_root)
    ray.get([a.wipe.remote(args.tmp_root) for _, _, a in packers])
    print("WIPE_DONE", flush=True)

os.makedirs(args.tmp_root, exist_ok=True)

# ---- 2. second GCS on this head ----
head_cmd = [
    "ray", "start", "--head",
    f"--port={args.gcs_port}",
    f"--temp-dir={args.tmp_root}/head",
    "--num-cpus=0",
    f"--object-store-memory={args.objstore_per_raylet}",
    "--include-dashboard=false",
    "--no-monitor",
    "--disable-usage-stats",
]
r = subprocess.run(head_cmd, env=clean_env(), start_new_session=True,
                   capture_output=True, text=True, timeout=240)
print(f"SECOND_GCS rc={r.returncode}", flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-2000:], flush=True)
    sys.exit(1)
t = time.time()
ns = None
while time.time() - t < 120:
    ns, err = second_cluster_nodes(gcs_addr)
    if ns is not None:
        break
    time.sleep(3)
assert ns is not None, f"second GCS never became reachable: {err}"
with open(f"{args.tmp_root}/gcs_addr", "w") as f:
    f.write(gcs_addr)
print(f"SECOND_GCS_READY {gcs_addr}", flush=True)

# ---- 3. packed raylets ----
R = args.raylets_per_host
launch = []
for j, ip, a in packers:
    specs = []
    for i in range(R):
        g = j * R + i
        specs.append({
            "cpus": args.cpus_per_raylet,
            "mem": args.mem_per_raylet,
            "objstore": args.objstore_per_raylet,
            "tmp": f"{args.tmp_root}/n{i}",
            "rack": f"rack-{g // args.rack_size}",
            "agent_port": 53000 + i,
        })
    launch.append((j, ip, a.start_raylets.remote(gcs_addr, specs)))

fails = 0
for j, ip, ref in launch:
    for res in ray.get(ref):
        if res["rc"] != 0:
            fails += 1
            print(f"RAYLET_FAIL host{j}({ip}) {res['tmp']} rc={res['rc']} tail={res['tail']}",
                  flush=True)
print(f"RAY_START_DONE fails={fails}", flush=True)

# ---- 4. wait for all joins, verify resources, print census ----
want = args.hosts * R
t = time.time()
n_ok = 0
ns = []
while time.time() - t < args.join_timeout:
    ns, err = second_cluster_nodes(gcs_addr)
    ns = ns or []
    n_ok = sum(1 for n in ns if n["cpu"] == float(args.cpus_per_raylet))
    print(f"JOIN_PROGRESS alive={len(ns)} cpu_ok={n_ok}/{want}", flush=True)
    if n_ok >= want:
        break
    time.sleep(5)

census = {}
for n in ns:
    census[n["labels"].get(RACK_KEY, "NONE")] = census.get(n["labels"].get(RACK_KEY, "NONE"), 0) + 1
print(f"RACK_CENSUS {json.dumps(census, sort_keys=True)}", flush=True)
if n_ok >= want:
    print(f"SECOND_CLUSTER_READY n={n_ok} elapsed={time.time()-t:.1f}s", flush=True)
else:
    print(f"SECOND_CLUSTER_INCOMPLETE cpu_ok={n_ok}/{want}", flush=True)
    sys.exit(2)
