# ABOUTME: Uploads profiling and monitoring artifacts from shared storage to cloud storage
# ABOUTME: Matches files by glob patterns and uploads them under a job-specific prefix.

import os


class Client():
    """ storage client class """
    def __init__(self):
        store = os.environ.get("PROFILING_STORAGE_TYPE", "s3")

        if store == "s3":
            import boto3
            self._client = boto3.client("s3")
            self._store = "s3://"

        if store == "gcs":
            from google.cloud import storage as gcs
            self._client = gcs.Client()
            self._store = "gs://"

    def get_store(self):
        """ return the storage type """
        return self._store

    def upload_file(self, filepath : str, storage_bucket : str, key : str):
        if self._store == "s3://":
            return self._client.upload_file(filepath, storage_bucket, key)

        if self._store == "gs://":
            try:
                bucket = self._client.get_bucket(storage_bucket)
            except Exception as failed:
                raise failed

            blob = bucket.blob(key)
            try:
                blob.upload_from_filename(filepath)
            except Exception as failed:
                raise failed
