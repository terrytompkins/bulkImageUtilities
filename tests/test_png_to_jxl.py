#!/usr/bin/env python3
"""
Simple test script for PNG to JPEG XL conversion with lossless compression.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import time
import shutil

def convert_png_to_jxl(source_path, dest_path, effort_level=7):
    """
    Convert a single PNG file to JPEG XL with lossless compression using cjxl command.
    
    Args:
        source_path: Path to source PNG file
        dest_path: Path for output JXL file
        effort_level: JXL encoding effort level (0-9, default 7)
    
    Returns:
        dict: Conversion result with success status and file sizes
    """
    try:
        # Check if cjxl is available
        cjxl_path = shutil.which("cjxl")
        if not cjxl_path:
            return {
                "success": False,
                "error": "cjxl not found in PATH. Please install jpeg-xl tools."
            }
        
        # Get source file size
        source_size = source_path.stat().st_size
        
        # Run cjxl command for lossless compression
        # -d 0 means lossless (distance 0)
        # -e effort_level sets encoding effort
        cmd = [cjxl_path, str(source_path), str(dest_path), "-d", "0", "-e", str(effort_level)]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"cjxl failed: {result.stderr.strip()}"
            }
        
        # Get destination file size
        dest_size = dest_path.stat().st_size
        compression_ratio = dest_size / source_size
        
        return {
            "success": True,
            "source_size": source_size,
            "dest_size": dest_size,
            "compression_ratio": compression_ratio,
            "savings_percent": (1 - compression_ratio) * 100
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Convert PNG files to JPEG XL with lossless compression")
    parser.add_argument("source_dir", help="Directory containing PNG files")
    parser.add_argument("dest_dir", help="Directory to write JXL files")
    parser.add_argument("--effort-level", type=int, default=7, choices=range(10), 
                       help="JXL encoding effort level 0-9 (default: 7)")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum number of parallel workers (default: 4)")
    parser.add_argument("--summarize-only", action="store_true",
                       help="Suppress per-image logging, show only final summary")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        sys.exit(1)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PNG to JPEG XL Conversion Test")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Effort level: {args.effort_level}")
    print(f"Max workers: {args.max_workers}")
    print(f"Summarize only: {args.summarize_only}")
    print("-" * 50)
    
    # Find all PNG files
    png_files = list(source_dir.glob("*.png"))
    if not png_files:
        print("No PNG files found in source directory")
        sys.exit(1)
    
    print(f"Found {len(png_files)} PNG files")
    
    # Convert files
    print("\nConverting files...")
    start_time = time.time()
    
    total_source_size = 0
    total_dest_size = 0
    successful_conversions = 0
    failed_conversions = 0
    
    for i, png_file in enumerate(png_files, 1):
        jxl_file = dest_dir / f"{png_file.stem}.jxl"
        
        if not args.summarize_only:
            print(f"  Converting {i}/{len(png_files)}: {png_file.name} -> {jxl_file.name}")
        
        result = convert_png_to_jxl(png_file, jxl_file, args.effort_level)
        
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
            else:
                print(f"  Failed conversion {i}/{len(png_files)}: {result['error']}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate overall statistics
    if total_source_size > 0:
        overall_compression_ratio = total_dest_size / total_source_size
        overall_savings = (1 - overall_compression_ratio) * 100
    else:
        overall_compression_ratio = 1.0
        overall_savings = 0.0
    
    # Print results
    print("\n" + "=" * 50)
    print("CONVERSION RESULTS")
    print("=" * 50)
    print(f"Total files processed: {len(png_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Average time per file: {elapsed_time/len(png_files):.2f} seconds")
    print()
    print(f"Total source size: {total_source_size:,} bytes ({total_source_size/1024/1024:.1f} MB)")
    print(f"Total destination size: {total_dest_size:,} bytes ({total_dest_size/1024/1024:.1f} MB)")
    print(f"Overall compression ratio: {overall_compression_ratio:.3f}")
    print(f"Overall space savings: {overall_savings:.1f}%")
    
    if successful_conversions > 0:
        print(f"\nFiles saved to: {dest_dir}")
        print(f"Average file size reduction: {overall_savings:.1f}%")

if __name__ == "__main__":
    main()
