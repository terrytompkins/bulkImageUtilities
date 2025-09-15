#!/usr/bin/env python3
"""
Example usage of the enhanced image_converter.py script with PNG compression and archiving features.

This script demonstrates various ways to use the new command line options for:
1. Advanced PNG compression using oxipng, zopflipng, or pngcrush
2. Creating compressed archives with zstd, 7z, or pigz
3. Batch processing with configurable archive sizes
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Command completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print("❌ Command failed")
            if result.stderr:
                print("Error:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Demonstrate various usage examples."""
    
    # Example 1: Basic PNG compression with oxipng (fast)
    print("📋 Example 1: Fast PNG compression with oxipng")
    cmd1 = [
        "python", "image_converter.py",
        "sample_images",  # source directory
        "output_oxipng",  # destination directory
        "report_oxipng.csv",  # report file
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "4",
        "--strip-metadata",
        "--max-workers", "4"
    ]
    run_command(cmd1, "Fast PNG compression with oxipng (effort level 4)")
    
    # Example 2: Maximum PNG compression with zopflipng (slow but effective)
    print("\n📋 Example 2: Maximum PNG compression with zopflipng")
    cmd2 = [
        "python", "image_converter.py",
        "sample_images",
        "output_zopfli",
        "report_zopfli.csv",
        "--png-compression-tool", "zopflipng",
        "--png-effort-level", "15",  # iterations
        "--max-workers", "2"  # fewer workers due to high CPU usage
    ]
    run_command(cmd2, "Maximum PNG compression with zopflipng (15 iterations)")
    
    # Example 3: PNG compression + zstd archiving (balanced approach)
    print("\n📋 Example 3: PNG compression + zstd archiving")
    cmd3 = [
        "python", "image_converter.py",
        "sample_images",
        "output_zstd",
        "report_zstd.csv",
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "6",
        "--create-archives",
        "--archive-tool", "zstd",
        "--archive-compression-level", "9",
        "--batch-size-mb", "300",
        "--zstd-long-range",
        "--max-workers", "4"
    ]
    run_command(cmd3, "PNG compression + zstd archiving (balanced)")
    
    # Example 4: PNG compression + 7z archiving (maximum compression)
    print("\n📋 Example 4: PNG compression + 7z archiving (maximum compression)")
    cmd4 = [
        "python", "image_converter.py",
        "sample_images",
        "output_7z",
        "report_7z.csv",
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "6",
        "--create-archives",
        "--archive-tool", "7z",
        "--archive-compression-level", "7",
        "--batch-size-mb", "200",
        "--sevenz-solid",
        "--sevenz-multithread",
        "--max-workers", "2"
    ]
    run_command(cmd4, "PNG compression + 7z archiving (maximum compression)")
    
    # Example 5: Fast processing with pigz (speed over compression)
    print("\n📋 Example 5: Fast processing with pigz")
    cmd5 = [
        "python", "image_converter.py",
        "sample_images",
        "output_pigz",
        "report_pigz.csv",
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "4",
        "--create-archives",
        "--archive-tool", "pigz",
        "--archive-compression-level", "9",
        "--batch-size-mb", "500",
        "--max-workers", "8"
    ]
    run_command(cmd5, "Fast processing with pigz (speed optimized)")
    
    # Example 6: Dry run to see what would happen
    print("\n📋 Example 6: Dry run (no actual processing)")
    cmd6 = [
        "python", "image_converter.py",
        "sample_images",
        "output_dry_run",
        "report_dry_run.csv",
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "5",
        "--create-archives",
        "--archive-tool", "zstd",
        "--batch-size-mb", "250",
        "--dry-run"
    ]
    run_command(cmd6, "Dry run - shows planned operations without executing")
    
    # Example 7: With similarity filtering
    print("\n📋 Example 7: With similarity filtering")
    cmd7 = [
        "python", "image_converter.py",
        "sample_images",
        "output_filtered",
        "report_filtered.csv",
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "4",
        "--filter-similar",
        "--similarity-threshold", "0.85",
        "--min-group-size", "2",
        "--selection-method", "best_quality",
        "--create-archives",
        "--archive-tool", "zstd",
        "--batch-size-mb", "300"
    ]
    run_command(cmd7, "PNG compression with similarity filtering + archiving")

if __name__ == "__main__":
    print("🚀 Enhanced Image Converter Examples")
    print("This script demonstrates various usage patterns for the enhanced image_converter.py")
    print("Make sure you have the required tools installed:")
    print("  - oxipng: pip install oxipng")
    print("  - zopflipng: install from your package manager")
    print("  - pngcrush: install from your package manager")
    print("  - zstd: install from your package manager")
    print("  - 7z: install from your package manager")
    print("  - pigz: install from your package manager")
    
    # Check if sample_images directory exists
    if not Path("sample_images").exists():
        print("\n⚠️  Warning: 'sample_images' directory not found.")
        print("   Create a directory with some PNG files to test the examples.")
        print("   You can modify the source directory in the examples above.")
    
    print("\n" + "="*60)
    print("📖 Usage Examples:")
    print("="*60)
    
    main()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)
    print("\n💡 Tips:")
    print("  - Use --dry-run to see what would happen without processing")
    print("  - Adjust --max-workers based on your CPU cores")
    print("  - Use --batch-size-mb to control archive sizes")
    print("  - Try different --png-effort-level values (0-7 for oxipng)")
    print("  - Use --filter-similar to reduce dataset size")
    print("  - Check the generated reports for detailed results")
