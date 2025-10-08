#!/usr/bin/env python3
"""
Quick test script to verify compression is working on a small batch.
"""

import subprocess
import tempfile
from pathlib import Path
import shutil

def test_small_batch():
    """Test compression on just 5 files to see if it works."""
    
    source_dir = Path("./IVDX000877_20250624_100824").expanduser()
    # Use tests/output directory for test data (relative to tests/ directory)
    test_dir = Path("output/test_small_batch")
    
    # Create test directory with just 5 files
    test_dir.mkdir(exist_ok=True)
    
    # Copy first 5 PNG files
    png_files = list(source_dir.glob("*.png"))[:5]
    print(f"📁 Copying {len(png_files)} files for testing...")
    
    for png_file in png_files:
        shutil.copy2(png_file, test_dir / png_file.name)
    
    # Test oxipng on these files
    print("🧪 Testing oxipng compression...")
    report_path = test_dir / "test_report.csv"
    cmd = [
        "python3", "../image_converter.py",
        str(test_dir),
        str(test_dir / "compressed"),
        str(report_path),
        "--png-compression-tool", "oxipng",
        "--png-effort-level", "1",  # Use lower effort for testing
        "--max-workers", "1"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"Return code: {result.returncode}")
    
    # Check results
    if report_path.exists():
        with open(report_path, "r") as f:
            lines = f.readlines()
            print(f"Report has {len(lines)} lines")
            if len(lines) > 1:
                print("First result line:", lines[1])

if __name__ == "__main__":
    test_small_batch()
