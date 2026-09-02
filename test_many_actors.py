import argparse
import os
import time

import tqdm
from many_nodes_tests.dashboard_test import DashboardTestAtScale

import ray
import ray._common.test_utils
import ray._private.test_utils as test_utils
from ray._private.state_api_test_utils import summarize_worker_startup_time

BENCHMARK = 'sca-actor-scaling'

parser = argparse.ArgumentParser()
parser.add_argument("--num-actors", type=int, default=20)
parser.add_argument("--num-cpus-per-actor", type=float, default=0)
parser.add_argument(
    "--resource",
    type=str,
    default=None,
    help="Custom resource requirement, e.g. fake_GPU=1",
)
parser.add_argument("--profile", action='store_true')
args = parser.parse_args()

is_smoke_test = True
if "SMOKE_TEST" in os.environ:
    MAX_ACTORS_IN_CLUSTER = 100
else:
    MAX_ACTORS_IN_CLUSTER = 10000
    is_smoke_test = False


def test_max_actors(args):
    # TODO (Alex): Dynamically set this based on number of cores
    cpus_per_actor = args.num_cpus_per_actor

    #@ray.remote(namespace="sca-test-1", num_cpus=cpus_per_actor, scheduling_strategy="SPREAD")
    @ray.remote(namespace="sca-test-1", num_cpus=cpus_per_actor)
    class Actor:
        def foo(self):
            pass

    actors = [
        Actor.remote()
        for _ in tqdm.trange(args.num_actors, desc="Launching actors")
    ]

    done = ray.get([actor.foo.remote() for actor in actors])
    for result in done:
        assert result is None


def no_resource_leaks():
    return test_utils.no_resource_leaks_excluding_node_resources()


addr = ray.init(address="auto")

outdir = os.environ.get("PROFILING_STORAGE_DIR", "/mnt/shared_storage")
if args.profile:
    from profiling.coordinator import Profiling
    job_id = os.environ.get("ANYSCALE_JOB_ID", "unknown")
    if job_id == 'unknown':
        job_id = os.environ.get("ANYSCALE_WORKSPACE_ID", "unknown")

    profiling = Profiling(
        outdir=f"{outdir}/{BENCHMARK}/{job_id}",
        num_gpu_nodes=0,
    )

    profiling.start()

#ray._common.test_utils.wait_for_condition(no_resource_leaks, timeout=60)
monitor_actor = test_utils.monitor_memory_usage()
dashboard_test = DashboardTestAtScale(addr)

start_time = time.time()
test_max_actors(args)
end_time = time.time()

if args.profile:
    profiling.stop(storage_prefix=f"{BENCHMARK}/{job_id}")

ray.get(monitor_actor.stop_run.remote())
used_gb, usage = ray.get(monitor_actor.get_peak_memory_info.remote())
print(f"Peak memory usage: {round(used_gb, 2)}GB")
print(f"Peak memory usage per processes:\n {usage}")
del monitor_actor

# Get the dashboard result
#ray._common.test_utils.wait_for_condition(no_resource_leaks)

rate = args.num_actors / (end_time - start_time)
try:
    summarize_worker_startup_time()
except Exception as e:
    print("Failed to summarize worker startup time.")
    print(e)

print(
    f"Success! Started {args.num_actors} actors in "
    f"{end_time - start_time}s. ({rate} actors/s)"
)

results = {
    "actors_per_second": rate,
    "num_actors": args.num_actors,
    "time": end_time - start_time,
    "_peak_memory": round(used_gb, 2),
    "_peak_process_memory": usage,
}
if not is_smoke_test:
    results["perf_metrics"] = [
        {
            "perf_metric_name": "actors_per_second",
            "perf_metric_value": rate,
            "perf_metric_type": "THROUGHPUT",
        }
    ]
dashboard_test.update_release_test_result(results)
test_utils.safe_write_to_results_json(results)
