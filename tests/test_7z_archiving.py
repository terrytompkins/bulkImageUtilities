#!/usr/bin/env python3
"""
Simple 7z archiving test script.
Creates compressed archives from PNG files with configurable batch sizes and compression levels.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
from datetime import datetime

def create_batches(source_dir, target_size_mb, output_dir):
    """Create batches of files with target size in MB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_size_bytes = target_size_mb * 1024 * 1024
    png_files = list(source_dir.glob("*.png"))
    
    if not png_files:
        print("No PNG files found in source directory.")
        return []
    
    print(f"Found {len(png_files)} PNG files")
    
    batches = []
    current_batch = []
    current_size = 0
    batch_num = 1
    
    for png_file in png_files:
        file_size = png_file.stat().st_size
        
        # Start new batch if adding this file would exceed target
        if current_size + file_size > target_size_bytes and current_batch:
            # Create batch directory and copy files
            batch_dir = output_dir / f"part{batch_num:02d}"
            batch_dir.mkdir(exist_ok=True)
            
            for file_path in current_batch:
                dest_path = batch_dir / file_path.name
                shutil.copy2(file_path, dest_path)
            
            batches.append(batch_dir)
            batch_num += 1
            current_batch = []
            current_size = 0
        
        current_batch.append(png_file)
        current_size += file_size
    
    # Handle remaining files
    if current_batch:
        batch_dir = output_dir / f"part{batch_num:02d}"
        batch_dir.mkdir(exist_ok=True)
        
        for file_path in current_batch:
            dest_path = batch_dir / file_path.name
            shutil.copy2(file_path, dest_path)
        
        batches.append(batch_dir)
    
    return batches

def create_7z_archive(batch_dir, output_path, compression_level, solid=True, multithread=True):
    """Create a 7z archive from a batch directory."""
    try:
        # Get all PNG files in the batch directory
        png_files = list(batch_dir.glob("*.png"))
        if not png_files:
            return {"success": False, "error": "No PNG files found in batch directory"}
        
        # Calculate source size
        source_size = sum(f.stat().st_size for f in png_files)
        
        # Build 7z command
        sevenz_path = shutil.which("7z")
        if not sevenz_path:
            return {"success": False, "error": "7z not found in PATH"}
        
        cmd = [sevenz_path, "a", "-t7z", "-m0=lzma2", f"-mx={compression_level}"]
        if solid:
            cmd.append("-ms=on")
        if multithread:
            cmd.append("-mmt=on")
        
        # Add PNG files directly to avoid directory overhead
        cmd.extend([str(output_path)] + [str(f) for f in png_files])
        
        # Execute command
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            archive_size = output_path.stat().st_size
            compression_ratio = archive_size / source_size
            return {
                "success": True,
                "source_size": source_size,
                "archive_size": archive_size,
                "compression_ratio": compression_ratio,
                "files_count": len(png_files)
            }
        else:
            return {"success": False, "error": process.stderr.strip()}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test 7z archiving on PNG files")
    parser.add_argument("source_dir", help="Directory containing PNG files")
    parser.add_argument("output_dir", help="Directory to write archives")
    parser.add_argument("--batch-size-mb", type=int, default=300, help="Target batch size in MB (default: 300)")
    parser.add_argument("--compression-level", type=int, default=9, choices=range(0, 10), help="7z compression level 0-9 (default: 9)")
    parser.add_argument("--no-solid", action="store_true", help="Disable solid compression")
    parser.add_argument("--no-multithread", action="store_true", help="Disable multithreading")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"7z Archiving Test")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size_mb} MB")
    print(f"Compression level: {args.compression_level}")
    print(f"Solid compression: {not args.no_solid}")
    print(f"Multithreading: {not args.no_multithread}")
    print("-" * 50)
    
    # Create batches
    print("Creating batches...")
    batch_dirs = create_batches(source_dir, args.batch_size_mb, output_dir / "batches")
    
    if not batch_dirs:
        print("No batches created. Exiting.")
        sys.exit(1)
    
    print(f"Created {len(batch_dirs)} batches")
    
    # Create archives
    print("\nCreating archives...")
    total_source_size = 0
    total_archive_size = 0
    successful_archives = 0
    
    for i, batch_dir in enumerate(batch_dirs, 1):
        archive_path = output_dir / f"part{i:02d}.7z"
        
        print(f"  Creating archive {i}/{len(batch_dirs)}: {archive_path.name}")
        
        result = create_7z_archive(
            batch_dir, 
            archive_path, 
            args.compression_level,
            solid=not args.no_solid,
            multithread=not args.no_multithread
        )
        
        if result["success"]:
            successful_archives += 1
            total_source_size += result["source_size"]
            total_archive_size += result["archive_size"]
            
            compression_ratio = result["compression_ratio"]
            reduction_percent = (1 - compression_ratio) * 100
            
            print(f"    ✓ Success: {result['files_count']} files")
            print(f"    Source: {result['source_size']:,} bytes ({result['source_size']/1024/1024:.1f} MB)")
            print(f"    Archive: {result['archive_size']:,} bytes ({result['archive_size']/1024/1024:.1f} MB)")
            print(f"    Ratio: {compression_ratio:.4f} ({reduction_percent:.1f}% reduction)")
        else:
            print(f"    ✗ Failed: {result['error']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successful archives: {successful_archives}/{len(batch_dirs)}")
    
    if successful_archives > 0:
        overall_ratio = total_archive_size / total_source_size
        overall_reduction = (1 - overall_ratio) * 100
        
        print(f"Total source size: {total_source_size:,} bytes ({total_source_size/1024/1024:.1f} MB)")
        print(f"Total archive size: {total_archive_size:,} bytes ({total_archive_size/1024/1024:.1f} MB)")
        print(f"Overall compression ratio: {overall_ratio:.4f}")
        print(f"Overall reduction: {overall_reduction:.1f}%")
    
    # Clean up batch directories
    print("\nCleaning up batch directories...")
    for batch_dir in batch_dirs:
        shutil.rmtree(batch_dir)
    print("Done!")

if __name__ == "__main__":
    main()
