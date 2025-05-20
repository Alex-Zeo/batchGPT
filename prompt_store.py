import os
import glob
from typing import List

try:
    import boto3
except ImportError:  # Optional dependency
    boto3 = None


def _load_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_prompts(path: str, glob_pattern: str = "*.txt") -> List[str]:
    """Load prompts from a local file, directory or S3 path."""
    if path.startswith("s3://"):
        if boto3 is None:
            raise ImportError("boto3 is required for S3 support")
        bucket_key = path[5:]
        bucket, _, key = bucket_key.partition('/')
        s3 = boto3.client('s3')
        if key and not key.endswith('/'):
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj['Body'].read().decode('utf-8')
            return [line.strip() for line in data.splitlines() if line.strip()]
        else:
            prefix = key
            paginator = s3.get_paginator('list_objects_v2')
            prompts: List[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get('Contents', []):
                    obj = s3.get_object(Bucket=bucket, Key=item['Key'])
                    data = obj['Body'].read().decode('utf-8')
                    prompts.extend([line.strip() for line in data.splitlines() if line.strip()])
            return prompts

    if os.path.isdir(path):
        prompts: List[str] = []
        files = glob.glob(os.path.join(path, glob_pattern))
        for file_path in files:
            prompts.extend(_load_file(file_path))
        return prompts

    if os.path.isfile(path):
        return _load_file(path)

    raise FileNotFoundError(f"Prompt source {path} not found")

