"""wait_drain.py — block until the second cluster's GCS syncer backlog drains.

The packed cluster joins ~8k raylets in ~90s; the resulting NodeAdded broadcast
storm (O(N^2) fan-out) leaves the single-threaded ray_syncer_io_context deeply
backlogged for many minutes. Starting the workload on top of that backlog is
what made packed 8k cells 3x slower than the real cluster (whose slow fill
spreads the same storm over tens of minutes). This waits for the drain:
polls the cumulative RaySyncer.BroadcastMessage counter in the second GCS's
event-stats dump (printed every 60s) until the per-minute delta falls below
--idle-delta twice in a row, then exits 0.
"""
import argparse
import glob
import json
import re
import time

p = argparse.ArgumentParser()
p.add_argument("--log-glob", default="/tmp/ray2nd/head/session_latest/logs/gcs_server.out*")
p.add_argument("--idle-delta", type=int, default=200)  # broadcasts/min considered idle
p.add_argument("--timeout", type=int, default=1800)
args = p.parse_args()


def latest_bc():
    best = None
    for f in glob.glob(args.log_glob):
        for line in open(f, errors="ignore"):
            if "ray_syncer_io_context Event stats" not in line:
                continue
            try:
                d = json.loads(line)
                m = re.search(r"RaySyncer\.BroadcastMessage - (\d+) total", d["message"])
                if m:
                    best = (d["asctime"], int(m.group(1)))
            except Exception:
                pass
    return best


t0 = time.time()
prev = None
calm = 0
while time.time() - t0 < args.timeout:
    cur = latest_bc()
    if cur:
        if prev and cur[0] != prev[0]:  # a new stats dump appeared (60s cadence)
            delta = cur[1] - prev[1]
            print(f"DRAIN_POLL ts={cur[0][11:19]} bc={cur[1]} delta/min={delta}", flush=True)
            calm = calm + 1 if delta < args.idle_delta else 0
            if calm >= 2:
                print(f"SYNCER_DRAINED bc={cur[1]} waited={time.time()-t0:.0f}s", flush=True)
                raise SystemExit(0)
        prev = prev if (prev and cur[0] == prev[0]) else cur
    time.sleep(20)
print(f"DRAIN_TIMEOUT waited={time.time()-t0:.0f}s", flush=True)
raise SystemExit(3)
