#!/usr/bin/env python3
"""query_payload_size.py

Query S3 to calculate the total size of image collections matching a structured path pattern.

The script searches for S3 objects matching the pattern:
    <BUCKET>/<PREFIX>/<DATE-PARTITION>/<SERIAL>_<TIMESTAMP>_<RUN_TYPE>_<RUN_UUID>/<IMAGE_SUB_FOLDER>

Optionally filters runs based on metadata in JSON files within each run folder.

Usage
-----

    # Basic query without metadata filtering
    python query_payload_size.py --bucket my-bucket --prefix invue_fna \
        --start-date 2022-11-19 --end-date 2022-11-20 \
        --serial-numbers IVDX009978 --run-type run_fna \
        --image-sub-folder images/raw_imageset/ --output results.csv

    # Query all devices (no serial number filter)
    python query_payload_size.py --bucket my-bucket --prefix invue_fna \
        --start-date 2022-11-19 --end-date 2022-11-20 \
        --run-type run_fna \
        --image-sub-folder images/raw_imageset/ --output results.csv

    # Query with metadata filtering
    python query_payload_size.py --bucket my-bucket --prefix invue_fna \
        --start-date 2022-11-19 --end-date 2022-11-20 \
        --serial-numbers IVDX009978 --run-type run_fna \
        --image-sub-folder images/raw_imageset/ \
        --metadata-file metadata/fna_results.json \
        --query-field run_metadata.INST_SW_VERSION=1.14.2 \
        --query-field run_metadata.SPECIES=Canine \
        --output-dir ./metadata-downloads --output results.csv

Requirements
------------
* Python 3.9+
* boto3

"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def parse_serial_numbers(serial_str: Optional[str]) -> Optional[Set[str]]:
    """Parse comma-separated serial numbers into a set. Returns None if input is None or empty."""
    if not serial_str or not serial_str.strip():
        return None
    serials = {s.strip() for s in serial_str.split(",") if s.strip()}
    return serials if serials else None


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
    serial_numbers: Optional[Set[str]],
    run_type: str,
    run_uuid: Optional[str],
    image_sub_folder: str,
    metadata_file_path: Optional[str] = None,
    query_fields: Optional[List[Tuple[str, str]]] = None,
    add_report_fields: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    download_metadata: bool = False,
    verbose: bool = False
) -> List[dict]:
    """
    Find all matching runs for the given criteria and calculate their sizes.
    Optionally filter by metadata and download metadata files.
    
    Returns list of dicts with keys: serial_number, timestamp, run_type, run_uuid, size_bytes,
    metadata_matched, and field values for each query field
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
        if serial_numbers is not None and serial not in serial_numbers:
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
    
    # Calculate size for each matching run and optionally filter by metadata
    results = []
    for run_folder_key, run_info in run_folders.items():
        size_bytes = calculate_run_size(s3_client, bucket, run_folder_key, image_sub_folder)
        run_info["size_bytes"] = size_bytes
        
        # Handle metadata filtering and downloading
        metadata_matched = True
        metadata_field_values = {}
        additional_field_values = {}
        
        if metadata_file_path and (query_fields or download_metadata or add_report_fields):
            # Download metadata file if needed
            if verbose:
                print(f"  Processing run {run_info['run_uuid']}...", file=sys.stderr)
            
            local_metadata_path = download_metadata_file(
                s3_client,
                bucket,
                run_folder_key,
                metadata_file_path,
                output_dir,
                run_info["run_uuid"],
                verbose
            )
            
            if local_metadata_path:
                # Load and check metadata
                metadata = load_metadata_file(local_metadata_path)
                
                if metadata:
                    if query_fields:
                        # Check if metadata matches query fields
                        if verbose:
                            print(f"  Checking metadata for run {run_info['run_uuid']}:", file=sys.stderr)
                        matches, field_values = check_metadata_matches(metadata, query_fields, verbose)
                        metadata_matched = matches
                        metadata_field_values = field_values
                        if verbose:
                            status = "MATCHED" if matches else "NOT MATCHED"
                            print(f"  Run {run_info['run_uuid']}: Overall result - {status}", file=sys.stderr)
                    
                    # Extract additional report fields if specified
                    if add_report_fields:
                        for field_path in add_report_fields:
                            value = get_nested_value(metadata, field_path)
                            additional_field_values[field_path] = value if value is not None else ""
                elif not metadata and query_fields:
                    # Invalid JSON - treat as not matching only if filtering is required
                    metadata_matched = False
                    if verbose:
                        print(f"  Run {run_info['run_uuid']}: Invalid JSON - NOT MATCHED", file=sys.stderr)
            else:
                # Metadata file doesn't exist - treat as not matching only if filtering is required
                if query_fields:
                    metadata_matched = False
                    if verbose:
                        print(f"  Run {run_info['run_uuid']}: Metadata file not found - NOT MATCHED", file=sys.stderr)
        
        # Add metadata match status and field values to result
        run_info["metadata_matched"] = metadata_matched
        run_info["metadata_field_values"] = metadata_field_values
        run_info["additional_field_values"] = additional_field_values
        
        # Include all runs that match cloud-side criteria
        # The metadata_matched flag indicates whether they also matched client-side criteria
        results.append(run_info)
    
    return results


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp from YYYYMMDD_HHMMSS to YYYY-MM-DD HH:MM:SS."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str


def parse_query_field(query_field_str: str) -> Tuple[str, str]:
    """
    Parse a query field string in the format 'key.path=value'.
    
    Returns: (key_path, value) tuple
    Raises: ValueError if format is invalid
    """
    if "=" not in query_field_str:
        raise ValueError(f"Invalid query field format: '{query_field_str}'. Expected 'key.path=value'")
    
    parts = query_field_str.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid query field format: '{query_field_str}'. Expected 'key.path=value'")
    
    key_path = parts[0].strip()
    value = parts[1].strip()
    
    if not key_path:
        raise ValueError(f"Query field key path cannot be empty: '{query_field_str}'")
    
    return (key_path, value)


def get_nested_value(data: dict, key_path: str) -> Optional[str]:
    """
    Get a nested value from a dictionary using dot notation.
    
    Example: get_nested_value(data, "run_metadata.INST_SW_VERSION")
    Returns: The value as a string, or None if not found
    """
    keys = key_path.split(".")
    current = data
    
    for key in keys:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]
    
    # Convert to string for comparison
    return str(current) if current is not None else None


def check_metadata_matches(metadata: dict, query_fields: List[Tuple[str, str]], verbose: bool = False) -> Tuple[bool, Dict[str, str]]:
    """
    Check if metadata matches all query fields (case-insensitive string comparison).
    
    Args:
        metadata: JSON metadata dictionary
        query_fields: List of (key_path, expected_value) tuples
        verbose: If True, print debug information about comparisons
    
    Returns:
        (matches: bool, field_values: dict) - field_values contains actual values for each query field
    """
    field_values = {}
    
    for key_path, expected_value in query_fields:
        actual_value = get_nested_value(metadata, key_path)
        field_values[key_path] = actual_value if actual_value is not None else ""
        
        if actual_value is None:
            if verbose:
                print(f"    Field '{key_path}': NOT FOUND in metadata", file=sys.stderr)
            return (False, field_values)
        
        # Case-insensitive string comparison
        if actual_value.lower() != expected_value.lower():
            if verbose:
                print(f"    Field '{key_path}': Expected '{expected_value}', Found '{actual_value}' - NO MATCH", file=sys.stderr)
            return (False, field_values)
        elif verbose:
            print(f"    Field '{key_path}': Expected '{expected_value}', Found '{actual_value}' - MATCH", file=sys.stderr)
    
    return (True, field_values)


def download_metadata_file(
    s3_client,
    bucket: str,
    run_prefix: str,
    metadata_file_path: str,
    output_dir: str,
    run_uuid: str,
    verbose: bool = False
) -> Optional[str]:
    """
    Download a metadata file from S3 to the output directory.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        run_prefix: S3 prefix for the run folder
        metadata_file_path: Path to metadata file relative to run folder
        output_dir: Local directory to save the file
        run_uuid: Run UUID to use as filename
        verbose: If True, print debug information
    
    Returns:
        Local file path if successful, None if file doesn't exist or error occurred
    """
    # Construct S3 key for metadata file
    # Ensure proper path joining (handle cases where run_prefix may or may not end with /)
    if run_prefix.endswith("/"):
        s3_key = f"{run_prefix}{metadata_file_path}"
    else:
        s3_key = f"{run_prefix}/{metadata_file_path}"
    # Normalize double slashes
    s3_key = s3_key.replace("//", "/")
    
    if verbose:
        print(f"  Attempting to download: s3://{bucket}/{s3_key}", file=sys.stderr)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Local file path: output_dir/run_uuid.json
    local_file_path = os.path.join(output_dir, f"{run_uuid}.json")
    
    try:
        s3_client.download_file(bucket, s3_key, local_file_path)
        if verbose:
            print(f"  Successfully downloaded to: {local_file_path}", file=sys.stderr)
        return local_file_path
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchKey":
            # File doesn't exist - this is expected for some runs
            if verbose:
                print(f"  Metadata file not found: s3://{bucket}/{s3_key}", file=sys.stderr)
            return None
        else:
            print(f"Warning: Error downloading metadata file 's3://{bucket}/{s3_key}': {e}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: Unexpected error downloading metadata file 's3://{bucket}/{s3_key}': {e}", file=sys.stderr)
        return None


def load_metadata_file(file_path: str) -> Optional[dict]:
    """
    Load and parse a JSON metadata file.
    
    Returns: Parsed JSON as dict, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in metadata file '{file_path}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Error reading metadata file '{file_path}': {e}", file=sys.stderr)
        return None


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

  # Query with metadata filtering by firmware version and species
  python query_payload_size.py --bucket my-bucket --prefix invue_fna \\
      --start-date 2022-11-19 --end-date 2022-11-20 \\
      --serial-numbers IVDX009978 --run-type run_fna \\
      --image-sub-folder images/raw_imageset/ \\
      --metadata-file metadata/fna_results.json \\
      --query-field run_metadata.INST_SW_VERSION=1.14.2 \\
      --query-field run_metadata.SPECIES=Canine \\
      --output-dir ./metadata-downloads --output results.csv

  # Download metadata files without filtering
  python query_payload_size.py --bucket my-bucket --prefix invue_fna \\
      --start-date 2022-11-19 --end-date 2022-11-20 \\
      --serial-numbers IVDX009978 --run-type run_fna \\
      --image-sub-folder images/raw_imageset/ \\
      --metadata-file metadata/fna_results.json \\
      --download-metadata --output-dir ./metadata-downloads

  # Query all devices (no serial number filter) with metadata filtering
  python query_payload_size.py --bucket my-bucket --prefix invue_fna \\
      --start-date 2022-11-19 --end-date 2022-11-20 \\
      --run-type run_fna \\
      --image-sub-folder images/raw_imageset/ \\
      --metadata-file metadata/fna_results.json \\
      --query-field run_metadata.INST_SW_VERSION=1.14.2 \\
      --output-dir ./metadata-downloads --output results.csv
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
        default=None,
        help="Optional: Comma-separated list of IoT device serial numbers (e.g., 'IVDX009978,IVDX009979'). "
             "If not specified, searches all serial numbers matching other criteria."
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
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Optional: Path to metadata JSON file relative to run folder (e.g., 'metadata/run_metadata.json')"
    )
    parser.add_argument(
        "--query-field",
        action="append",
        dest="query_fields",
        help="Query field in format 'key.path=value' (can be specified multiple times). "
             "Example: --query-field 'run_metadata.INST_SW_VERSION=1.14.2'"
    )
    parser.add_argument(
        "--add-report-fields",
        default=None,
        help="Comma-separated list of dot-notation key paths from metadata JSON to include in CSV output. "
             "Example: 'run_metadata.PATIENT_ID,run_metadata.SPECIES'"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional: Directory to download metadata files to (required if --query-field, --add-report-fields, or --download-metadata is used)"
    )
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download metadata files even if not filtering (requires --output-dir)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug information (useful for troubleshooting metadata downloads)"
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
    
    # Parse serial numbers (optional)
    serial_numbers = parse_serial_numbers(args.serial_numbers)
    
    # Parse query fields
    query_fields = None
    if args.query_fields:
        query_fields = []
        for qf_str in args.query_fields:
            try:
                key_path, value = parse_query_field(qf_str)
                query_fields.append((key_path, value))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
    
    # Parse add-report-fields
    add_report_fields = None
    if args.add_report_fields:
        add_report_fields = [f.strip() for f in args.add_report_fields.split(",") if f.strip()]
        if not add_report_fields:
            print("Error: --add-report-fields must contain at least one field path", file=sys.stderr)
            sys.exit(1)
    
    # Validate metadata-related parameters
    needs_output_dir = (query_fields is not None) or args.download_metadata or (add_report_fields is not None)
    if needs_output_dir and not args.output_dir:
        print("Error: --output-dir is required when using --query-field, --add-report-fields, or --download-metadata", file=sys.stderr)
        sys.exit(1)
    
    if args.metadata_file and not needs_output_dir:
        print("Warning: --metadata-file specified but no filtering, reporting, or download requested. "
              "Specify --query-field, --add-report-fields, or --download-metadata to use metadata files.", file=sys.stderr)
    
    # Initialize S3 client
    s3_client = boto3.client("s3")
    
    # Generate date partitions
    date_partitions = generate_date_partitions(start_date, end_date)
    
    print(f"Searching for runs in {len(date_partitions)} date partition(s)...")
    if serial_numbers:
        print(f"Serial numbers: {', '.join(sorted(serial_numbers))}")
    else:
        print(f"Serial numbers: (all)")
    print(f"Run type: {args.run_type}")
    if args.run_uuid:
        print(f"Run UUID: {args.run_uuid}")
    if args.metadata_file:
        print(f"Metadata file: {args.metadata_file}")
    if query_fields:
        print(f"Metadata query fields: {len(query_fields)} field(s)")
        for key_path, value in query_fields:
            print(f"  - {key_path} = {value}")
    if add_report_fields:
        print(f"Additional report fields: {len(add_report_fields)} field(s)")
        for field_path in add_report_fields:
            print(f"  - {field_path}")
    if args.output_dir:
        print(f"Metadata output directory: {args.output_dir}")
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
            args.image_sub_folder,
            args.metadata_file,
            query_fields,
            add_report_fields,
            args.output_dir,
            args.download_metadata,
            args.verbose
        )
        all_results.extend(results)
        print(f"Found {len(results)} run(s)")
    
    if not all_results:
        print("\nNo matching runs found.")
        sys.exit(0)
    
    # Build CSV fieldnames
    base_fieldnames = ["Date/Time", "IoT Device Serial Number", "Run UUID", "Run Type", "Size (bytes)"]
    
    # Add metadata match column if metadata filtering is enabled
    if query_fields:
        base_fieldnames.append("Metadata Matched")
        # Add columns for each query field (use the last part of the key path as column name)
        for key_path, _ in query_fields:
            # Extract the last part of the key path for column name
            column_name = key_path.split(".")[-1]
            base_fieldnames.append(column_name)
    
    # Add columns for additional report fields
    if add_report_fields:
        for field_path in add_report_fields:
            # Extract the last part of the key path as column name
            column_name = field_path.split(".")[-1]
            base_fieldnames.append(column_name)
    
    # Write results to CSV
    print(f"\nWriting {len(all_results)} result(s) to {args.output}...")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {
                "Date/Time": format_timestamp(result["timestamp"]),
                "IoT Device Serial Number": result["serial_number"],
                "Run UUID": result["run_uuid"],
                "Run Type": result["run_type"],
                "Size (bytes)": result["size_bytes"]
            }
            
            # Add metadata information if query fields are specified
            if query_fields:
                row["Metadata Matched"] = "Yes" if result.get("metadata_matched", False) else "No"
                field_values = result.get("metadata_field_values", {})
                for key_path, _ in query_fields:
                    column_name = key_path.split(".")[-1]
                    row[column_name] = field_values.get(key_path, "")
            
            # Add additional report fields if specified
            if add_report_fields:
                additional_values = result.get("additional_field_values", {})
                for field_path in add_report_fields:
                    column_name = field_path.split(".")[-1]
                    row[column_name] = additional_values.get(field_path, "")
            
            writer.writerow(row)
    
    # Print summary
    total_size = sum(r["size_bytes"] for r in all_results)
    total_size_gb = total_size / (1024 ** 3)
    
    print(f"\nSummary:")
    print(f"  Total runs found: {len(all_results)}")
    
    # Show metadata match statistics if query fields were used
    if query_fields:
        matched_count = sum(1 for r in all_results if r.get("metadata_matched", False))
        not_matched_count = len(all_results) - matched_count
        print(f"  Runs matching metadata criteria: {matched_count}")
        print(f"  Runs not matching metadata criteria: {not_matched_count}")
    
    print(f"  Total size: {total_size:,} bytes ({total_size_gb:.2f} GB)")
    print(f"  Results written to: {args.output}")


if __name__ == "__main__":
    main()

