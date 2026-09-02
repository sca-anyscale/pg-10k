# 10k-node Ray cluster on 833 hosts

Each host runs 12 raylets that join a second Ray cluster (GCS on this head, port 6479).
Result: 9,996 Ray nodes. For control-plane tests (scheduling, GCS, syncer), not for real compute.
Image must be Ray >= 2.55.0 (older `ray start` ignores `--temp-dir` on worker nodes, and the 12 raylets per host then collide). This workspace uses anyscale/ray:2.57.0.

Run from a terminal in ~/default:

```bash
# 1. get 833 hosts (~5 min; the compute config starts with 0 workers)
nohup python fill_hosts.py --hosts 833 > fill.log 2>&1 &
tail -f fill.log            # FILL_OK n=8xx

# 2. start the 9,996 raylets (~2 min)
nohup python launch_second_cluster.py --hosts 833 --raylets-per-host 12 > launch.log 2>&1 &
tail -f launch.log          # SECOND_CLUSTER_READY n=9996

# 3. wait for the node-join storm to drain (~13 min at 10k, ~2 min at 36 nodes)
nohup python wait_drain.py > drain.log 2>&1 &
tail -f drain.log           # SYNCER_DRAINED

# 4. your workload (hello_world.py = one task per node; replace it)
RAY_ADDRESS=$(cat /tmp/ray2nd/gcs_addr) python hello_world.py
```

Drivers must set `RAY_ADDRESS=$(cat /tmp/ray2nd/gcs_addr)`, otherwise they talk to the platform cluster.
Second-cluster logs: `/tmp/ray2nd/head/session_latest/logs/` (GCS), `/tmp/ray2nd/n<i>/` (raylets).
Step 2 is idempotent; rerun it (then step 3) to rebuild.

Try `--hosts 3` first (36 nodes, <$1). 833 hosts cost ~$350 per 30 min.
**Terminate the workspace when done.** ~/default survives termination; restart takes 2 min.

Step 3 matters: 9,996 raylets joining in ~90 s leave the GCS syncer with minutes of backlog; anything timed before it drains is distorted.
