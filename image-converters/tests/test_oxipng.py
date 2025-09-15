#!/usr/bin/env python3
"""
Simple test script for PNG compression using oxipng.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import time
import shutil

def compress_png_with_oxipng(source_path, dest_path, effort_level=4):
    """
    Compress a single PNG file using oxipng.
    
    Args:
        source_path: Path to source PNG file
        dest_path: Path for output compressed PNG file
        effort_level: oxipng effort level (0-6, default 4)
    
    Returns:
        dict: Compression result with success status and file sizes
    """
    try:
        # Get source file size
        source_size = source_path.stat().st_size
        
        # Build oxipng command
        # -o effort_level: optimization level (0-6)
        # --strip safe: remove safe-to-remove metadata
        # --backup: create backup of original file
        # --out: specify output file
        cmd = [
            "oxipng",
            "-o", str(effort_level),
            "--strip", "safe",
            "--backup",
            "--out", str(dest_path),
            str(source_path)
        ]
        
        # Run oxipng
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"oxipng failed: {result.stderr}",
                "source_size": source_size,
                "dest_size": 0
            }
        
        # Get destination file size
        dest_size = dest_path.stat().st_size
        
        # Calculate savings
        savings_percent = ((source_size - dest_size) / source_size) * 100 if source_size > 0 else 0
        
        return {
            "success": True,
            "source_size": source_size,
            "dest_size": dest_size,
            "savings_percent": savings_percent,
            "error": None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "oxipng timed out after 5 minutes",
            "source_size": source_path.stat().st_size if source_path.exists() else 0,
            "dest_size": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "source_size": source_path.stat().st_size if source_path.exists() else 0,
            "dest_size": 0
        }

def main():
    parser = argparse.ArgumentParser(description="Compress PNG files using oxipng")
    parser.add_argument("source_dir", help="Directory containing PNG files")
    parser.add_argument("dest_dir", help="Directory to write compressed PNG files")
    parser.add_argument("--effort-level", type=int, default=4, choices=range(7), 
                       help="oxipng effort level 0-6 (default: 4)")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum number of parallel workers (default: 4)")
    parser.add_argument("--summarize-only", action="store_true",
                       help="Suppress per-file logging, show only summary")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files
    png_files = list(source_dir.glob("*.png"))
    if not png_files:
        print(f"Error: No PNG files found in '{source_dir}'")
        sys.exit(1)
    
    print("PNG Compression Test (oxipng)")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Effort level: {args.effort_level}")
    print(f"Max workers: {args.max_workers}")
    print(f"Summarize only: {args.summarize_only}")
    print("-" * 50)
    print(f"Found {len(png_files)} PNG files")
    print()
    
    if not args.summarize_only:
        print("Compressing files...")
        print()
    
    # Process files
    start_time = time.time()
    successful_conversions = 0
    failed_conversions = 0
    total_source_size = 0
    total_dest_size = 0
    
    for i, png_file in enumerate(png_files, 1):
        compressed_file = dest_dir / png_file.name
        
        if not args.summarize_only:
            print(f"  Compressing {i}/{len(png_files)}: {png_file.name}")
        
        result = compress_png_with_oxipng(png_file, compressed_file, args.effort_level)
        
        if result["success"]:
            successful_conversions += 1
            total_source_size += result["source_size"]
            total_dest_size += result["dest_size"]
            
            if not args.summarize_only:
                savings = result["savings_percent"]
                print(f"    ✓ {savings:.1f}% savings ({result['source_size']:,} -> {result['dest_size']:,} bytes)")
        else:
            failed_conversions += 1
            if not args.summarize_only:
                print(f"    ✗ Failed: {result['error']}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate overall statistics
    overall_savings = ((total_source_size - total_dest_size) / total_source_size) * 100 if total_source_size > 0 else 0
    compression_ratio = total_dest_size / total_source_size if total_source_size > 0 else 1.0
    avg_time_per_file = processing_time / len(png_files) if png_files else 0
    
    print()
    print("=" * 50)
    print("COMPRESSION RESULTS")
    print("=" * 50)
    print(f"Total files processed: {len(png_files)}")
    print(f"Successful compressions: {successful_conversions}")
    print(f"Failed compressions: {failed_conversions}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Average time per file: {avg_time_per_file:.2f} seconds")
    print()
    print(f"Total source size: {total_source_size:,} bytes ({total_source_size/1024/1024:.1f} MB)")
    print(f"Total destination size: {total_dest_size:,} bytes ({total_dest_size/1024/1024:.1f} MB)")
    print(f"Overall compression ratio: {compression_ratio:.3f}")
    print(f"Overall space savings: {overall_savings:.1f}%")
    print()
    print(f"Files saved to: {dest_dir}")
    print(f"Average file size reduction: {overall_savings:.1f}%")

if __name__ == "__main__":
    main()
