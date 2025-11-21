#!/usr/bin/env python3
"""query_payload_size.py

Query S3 to calculate the total size of image collections matching a structured path pattern.

The script searches for S3 objects matching the pattern:
    <BUCKET>/<PREFIX>/<DATE-PARTITION>/<SERIAL>_<TIMESTAMP>_<RUN_TYPE>_<RUN_UUID>/<IMAGE_SUB_FOLDER>

Usage
-----

    python query_payload_size.py --bucket my-bucket --prefix invue_fna \
        --start-date 2022-11-19 --end-date 2022-11-20 \
        --serial-numbers IVDX009978 --run-type run_fna \
        --image-sub-folder images/raw_imageset/ --output results.csv

Requirements
------------
* Python 3.9+
* boto3
* pandas

"""

import argparse
import csv
import re
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def generate_date_partitions(start_date: datetime, end_date: datetime) -> List[str]:
    """Generate list of date partition paths (year=YYYY/month=MM/day=DD) for the date range."""
    partitions = []
    current = start_date
    while current <= end_date:
        partition = f"year={current.year}/month={current.month:02d}/day={current.day:02d}"
        partitions.append(partition)
        current += timedelta(days=1)
    return partitions


def parse_serial_numbers(serial_str: str) -> Set[str]:
    """Parse comma-separated serial numbers into a set."""
    return {s.strip() for s in serial_str.split(",") if s.strip()}


def extract_run_info_from_key(key: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Extract run information from S3 object key.
    
    Expected pattern: .../<SERIAL>_<YYYYMMDD_HHMMSS>_<RUN_TYPE>_<RUN_UUID>/...
    
    Returns: (serial_number, timestamp, run_type, run_uuid) or None if pattern doesn't match
    """
    # Pattern: serial_timestamp_runtype_uuid
    # Example: IVDX009978_20221119_175005_run_fna_E27A704B-DAB6-4282-BF6B-2EBF7079DEA6
    pattern = r"([A-Z0-9]+)_(\d{8}_\d{6})_([a-z_]+)_([A-Z0-9-]+)"
    
    # Find the run folder in the path
    parts = key.split("/")
    for part in parts:
        match = re.match(pattern, part)
        if match:
            serial, timestamp, run_type, run_uuid = match.groups()
            return (serial, timestamp, run_type, run_uuid)
    
    return None


def list_objects_in_prefix(s3_client, bucket: str, prefix: str) -> List[dict]:
    """List all objects with the given prefix, handling pagination."""
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                objects.extend(page["Contents"])
    except ClientError as e:
        print(f"Error listing objects with prefix '{prefix}': {e}", file=sys.stderr)
        return []
    
    return objects


def calculate_run_size(
    s3_client,
    bucket: str,
    run_prefix: str,
    image_sub_folder: str
) -> int:
    """Calculate total size of all objects in a run's image sub-folder."""
    full_prefix = f"{run_prefix}/{image_sub_folder}"
    if not full_prefix.endswith("/"):
        full_prefix += "/"
    
    objects = list_objects_in_prefix(s3_client, bucket, full_prefix)
    return sum(obj["Size"] for obj in objects)


def find_matching_runs(
    s3_client,
    bucket: str,
    prefix: str,
    date_partition: str,
    serial_numbers: Set[str],
    run_type: str,
    run_uuid: Optional[str],
    image_sub_folder: str
) -> List[dict]:
    """
    Find all matching runs for the given criteria and calculate their sizes.
    
    Returns list of dicts with keys: serial_number, timestamp, run_type, run_uuid, size_bytes
    """
    # Build the search prefix up to the date partition
    search_prefix = f"{prefix}/{date_partition}/"
    
    # List all objects under this date partition
    all_objects = list_objects_in_prefix(s3_client, bucket, search_prefix)
    
    # Group objects by run (identified by the run folder path)
    run_folders = {}
    for obj in all_objects:
        key = obj["Key"]
        run_info = extract_run_info_from_key(key)
        
        if not run_info:
            continue
        
        serial, timestamp, rt, uuid = run_info
        
        # Filter by criteria
        if serial not in serial_numbers:
            continue
        if rt != run_type:
            continue
        if run_uuid and uuid != run_uuid:
            continue
        
        # Store the run folder path (everything up to and including the run UUID folder)
        run_folder_key = f"{prefix}/{date_partition}/{serial}_{timestamp}_{rt}_{uuid}"
        if run_folder_key not in run_folders:
            run_folders[run_folder_key] = {
                "serial_number": serial,
                "timestamp": timestamp,
                "run_type": rt,
                "run_uuid": uuid,
            }
    
    # Calculate size for each matching run
    results = []
    for run_folder_key, run_info in run_folders.items():
        size_bytes = calculate_run_size(s3_client, bucket, run_folder_key, image_sub_folder)
        run_info["size_bytes"] = size_bytes
        results.append(run_info)
    
    return results


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp from YYYYMMDD_HHMMSS to YYYY-MM-DD HH:MM:SS."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str


def main():
    parser = argparse.ArgumentParser(
        description="Query S3 to calculate payload sizes for matching image collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query a single device for a date range
  python query_payload_size.py --bucket my-bucket --prefix invue_fna \\
      --start-date 2022-11-19 --end-date 2022-11-20 \\
      --serial-numbers IVDX009978 --run-type run_fna \\
      --image-sub-folder images/raw_imageset/

  # Query multiple devices with a specific run UUID
  python query_payload_size.py --bucket my-bucket --prefix invue_fna \\
      --start-date 2022-11-19 --end-date 2022-11-19 \\
      --serial-numbers IVDX009978,IVDX009979 --run-type run_fna \\
      --run-uuid E27A704B-DAB6-4282-BF6B-2EBF7079DEA6 \\
      --image-sub-folder images/raw_imageset/ --output results.csv
        """
    )
    
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix path below bucket (e.g., 'invue_fna')"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=str,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=str,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--serial-numbers",
        required=True,
        help="Comma-separated list of IoT device serial numbers (e.g., 'IVDX009978,IVDX009979')"
    )
    parser.add_argument(
        "--run-type",
        required=True,
        help="Run type (e.g., 'run_fna')"
    )
    parser.add_argument(
        "--run-uuid",
        default=None,
        help="Optional: Specific run UUID to filter by"
    )
    parser.add_argument(
        "--image-sub-folder",
        required=True,
        help="Image sub-folder path below run UUID (e.g., 'images/raw_imageset/')"
    )
    parser.add_argument(
        "--output",
        default="query_payload_size.csv",
        help="Output CSV file path (default: query_payload_size.csv)"
    )
    
    args = parser.parse_args()
    
    # Parse and validate dates
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if start_date > end_date:
        print("Error: start-date must be <= end-date", file=sys.stderr)
        sys.exit(1)
    
    # Parse serial numbers
    serial_numbers = parse_serial_numbers(args.serial_numbers)
    if not serial_numbers:
        print("Error: At least one serial number must be provided", file=sys.stderr)
        sys.exit(1)
    
    # Initialize S3 client
    s3_client = boto3.client("s3")
    
    # Generate date partitions
    date_partitions = generate_date_partitions(start_date, end_date)
    
    print(f"Searching for runs in {len(date_partitions)} date partition(s)...")
    print(f"Serial numbers: {', '.join(sorted(serial_numbers))}")
    print(f"Run type: {args.run_type}")
    if args.run_uuid:
        print(f"Run UUID: {args.run_uuid}")
    print()
    
    # Collect all matching runs
    all_results = []
    for date_partition in date_partitions:
        print(f"Processing {date_partition}...", end=" ", flush=True)
        results = find_matching_runs(
            s3_client,
            args.bucket,
            args.prefix,
            date_partition,
            serial_numbers,
            args.run_type,
            args.run_uuid,
            args.image_sub_folder
        )
        all_results.extend(results)
        print(f"Found {len(results)} run(s)")
    
    if not all_results:
        print("\nNo matching runs found.")
        sys.exit(0)
    
    # Write results to CSV
    print(f"\nWriting {len(all_results)} result(s) to {args.output}...")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Date/Time", "IoT Device Serial Number", "Run UUID", "Run Type", "Size (bytes)"]
        )
        writer.writeheader()
        
        for result in all_results:
            writer.writerow({
                "Date/Time": format_timestamp(result["timestamp"]),
                "IoT Device Serial Number": result["serial_number"],
                "Run UUID": result["run_uuid"],
                "Run Type": result["run_type"],
                "Size (bytes)": result["size_bytes"]
            })
    
    # Print summary
    total_size = sum(r["size_bytes"] for r in all_results)
    total_size_gb = total_size / (1024 ** 3)
    
    print(f"\nSummary:")
    print(f"  Total runs found: {len(all_results)}")
    print(f"  Total size: {total_size:,} bytes ({total_size_gb:.2f} GB)")
    print(f"  Results written to: {args.output}")


if __name__ == "__main__":
    main()

