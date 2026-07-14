# ABOUTME: Uploads profiling and monitoring artifacts from shared storage to cloud storage
# ABOUTME: Matches files by glob patterns and uploads them under a job-specific prefix.

import glob as globmod
import os
from . import storage


DEFAULT_STORAGE_BUCKET = os.environ.get(
    "PROFILING_STORAGE_BUCKET",
    "anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7",
)

DEFAULT_UPLOAD_PATTERNS = [
    "gpu_usage*",
    "net_counters*",
    "nsys_*",
    "torch_profile_*",
    "driver_gc*",
    "pyspy_*",
    "perf_*",
    "result*.json",
    "object_store_state*",
    "actor_placement*",
    "plasma_stats*",
]


def upload(outdir, storage_prefix, storage_bucket=None, patterns=None):
    """Upload telemetry files from outdir to cloud blob storage.

    Args:
        outdir: Local directory containing profiling/monitoring output.
        storage_prefix: storage key prefix (e.g. "image-embedding-jsonl/<job_id>").
        storage_bucket: storage bucket name. Defaults to PROFILING_STORAGE_BUCKET env var.
        patterns: List of glob patterns to match. Defaults to DEFAULT_UPLOAD_PATTERNS.
    """
    client = storage.Client()
    store = client.get_store()

    if storage_bucket is None:
        storage_bucket = DEFAULT_STORAGE_BUCKET
    if patterns is None:
        patterns = DEFAULT_UPLOAD_PATTERNS

    total_uploaded = 0
    for pattern in patterns:
        files = globmod.glob(os.path.join(outdir, pattern))
        for filepath in files:
            key = f"{storage_prefix}/{os.path.basename(filepath)}"
            print(f"Uploading {filepath} -> {store}{storage_bucket}/{key}")
            client.upload_file(filepath, storage_bucket, key)
            total_uploaded += 1
    print(f"Uploaded {total_uploaded} telemetry files to {store}{storage_bucket}/{storage_prefix}/")
