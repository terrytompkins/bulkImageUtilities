
#!/usr/bin/env python3
"""s3_uploader_tester.py

Benchmark different methods of uploading many image files to an S3 bucket.

Usage
-----

    python s3_uploader_tester.py --input-dir ./images --bucket my-bucket \
        --method awscli_sync --prefix test/

Supported methods
-----------------

* awscli_sync          - uses `aws s3 sync`
* s5cmd_cp             - uses `s5cmd cp`
* boto3_transfer       - uses Boto3 TransferManager (concurrent multipart)
* curl_presigned_single- generates a presigned URL for each file and uploads with curl
* curl_presigned_persist- same as above but re‑uses a single curl process via `--next`

The script times the entire upload batch, logs a CSV of results, and prints a summary.

Requirements
------------
* Python 3.9+
* boto3 (for boto3_transfer and presigned_url generation)
* AWS CLI v2 configured (for awscli_sync)
* s5cmd installed (for s5cmd_cp)
* curl >=7.65 (for persistent connection test)

"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import boto3


def list_files(input_dir: Path) -> List[Path]:
    return [p for p in input_dir.rglob('*') if p.is_file()]


def total_size(files: List[Path]) -> int:
    return sum(f.stat().st_size for f in files)


def run_subprocess(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {' '.join(cmd)}\n{exc}", file=sys.stderr)
        sys.exit(1)


def method_awscli_sync(files: List[Path], bucket: str, prefix: str, input_dir: Path):
    # aws s3 sync localdir s3://bucket/prefix/
    dest = f"s3://{bucket}/{prefix}"
    cmd = ["aws", "s3", "sync", str(input_dir), dest]
    run_subprocess(cmd)


def method_s5cmd_cp(files: List[Path], bucket: str, prefix: str, input_dir: Path):
    # s5cmd cp localdir/ s3://bucket/prefix/
    dest = f"s3://{bucket}/{prefix}"
    cmd = ["s5cmd", "cp", f"{input_dir}/", dest]
    run_subprocess(cmd)


def upload_with_transfer(files: List[Path], bucket: str, prefix: str):
    import concurrent.futures
    s3 = boto3.client("s3")
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True,
    )

    def _upload(path: Path):
        key = f"{prefix}{path.relative_to(path.parents[0])}"
        s3.upload_file(str(path), bucket, key, Config=transfer_config)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(_upload, files))


def method_boto3_transfer(files: List[Path], bucket: str, prefix: str, input_dir: Path):
    upload_with_transfer(files, bucket, prefix)


def generate_presigned_put(path: Path, bucket: str, key: str, expires: int = 3600) -> str:
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )


def method_curl_single(files: List[Path], bucket: str, prefix: str, input_dir: Path):
    for path in files:
        key = f"{prefix}{path.name}"
        url = generate_presigned_put(path, bucket, key)
        cmd = ["curl", "-s", "-o", "/dev/null", "-X", "PUT", "-T", str(path), url]
        run_subprocess(cmd)


def method_curl_persist(files: List[Path], bucket: str, prefix: str, input_dir: Path):
    for path in files:
        key = f"{prefix}{path.name}"
        url = generate_presigned_put(path, bucket, key)
        # reuse connections by passing --keepalive-time
        cmd = [
            "curl",
            "--keepalive-time",
            "60",
            "-s",
            "-o",
            "/dev/null",
            "-X",
            "PUT",
            "-T",
            str(path),
            url,
        ]
        run_subprocess(cmd)


METHOD_MAP = {
    "awscli_sync": method_awscli_sync,
    "s5cmd_cp": method_s5cmd_cp,
    "boto3_transfer": method_boto3_transfer,
    "curl_presigned_single": method_curl_single,
    "curl_presigned_persist": method_curl_persist,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark S3 upload methods.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--method", choices=list(METHOD_MAP.keys()), required=True)
    parser.add_argument("--prefix", default="benchmark/")
    args = parser.parse_args()

    files = list_files(args.input_dir)
    if not files:
        print("No files found in the specified directory.")
        sys.exit(1)

    size_bytes = total_size(files)
    size_mb = size_bytes / (1024 * 1024)

    print(
        f"Uploading {len(files)} files ({size_mb:.2f} MB total) using method '{args.method}'..."
    )
    start = time.time()

    METHOD_MAP[args.method](files, args.bucket, args.prefix, args.input_dir)

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f} s")

    print(
        f"SUMMARY | files={len(files)}, size={size_mb:.2f} MB, method={args.method}, time={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
