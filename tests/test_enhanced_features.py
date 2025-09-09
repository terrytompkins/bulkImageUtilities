#!/usr/bin/env python3
"""
Test script for the enhanced image_converter.py features.

This script tests the new PNG compression and archiving capabilities
to ensure they work correctly with sample data.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def create_test_images(output_dir, num_images=10):
    """Create test PNG images for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🖼️  Creating {num_images} test PNG images...")
    
    for i in range(num_images):
        # Create a simple test image
        width, height = 800, 600
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some patterns to make compression interesting
        img_array[::50, :, :] = 128  # Horizontal lines
        img_array[:, ::50, :] = 128  # Vertical lines
        
        img = Image.fromarray(img_array)
        img_path = output_path / f"test_image_{i:03d}.png"
        img.save(img_path, format="PNG", compress_level=0)  # No compression for testing
    
    print(f"✅ Created {num_images} test images in {output_dir}")
    return output_path

def check_tool_availability():
    """Check if required tools are available."""
    tools = {
        "oxipng": "oxipng",
        "zopflipng": "zopflipng", 
        "pngcrush": "pngcrush",
        "zstd": "zstd",
        "7z": "7z",
        "pigz": "pigz"
    }
    
    available = {}
    for name, command in tools.items():
        result = subprocess.run(["which", command], capture_output=True)
        available[name] = result.returncode == 0
    
    return available

def run_test(test_name, cmd, expected_success=True):
    """Run a test command and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 Test: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and expected_success:
            print("✅ Test PASSED")
            return True
        elif result.returncode != 0 and not expected_success:
            print("✅ Test PASSED (expected failure)")
            return True
        else:
            print("❌ Test FAILED")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Image Converter Test Suite")
    print("="*60)
    
    # Check tool availability
    print("\n🔍 Checking tool availability...")
    available_tools = check_tool_availability()
    
    for tool, is_available in available_tools.items():
        status = "✅" if is_available else "❌"
        print(f"   {status} {tool}")
    
    # Create test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_images_dir = Path(temp_dir) / "test_images"
        output_base = Path(temp_dir) / "output"
        
        # Create test images
        create_test_images(test_images_dir, num_images=5)
        
        # Test 1: Basic oxipng compression
        if available_tools["oxipng"]:
            output_dir = output_base / "test1_oxipng"
            cmd1 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "oxipng",
                "--png-effort-level", "4",
                "--strip-metadata"
            ]
            run_test("Basic oxipng compression", cmd1)
        
        # Test 2: zopflipng compression (if available)
        if available_tools["zopflipng"]:
            output_dir = output_base / "test2_zopflipng"
            cmd2 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "zopflipng",
                "--png-effort-level", "5"  # Lower iterations for testing
            ]
            run_test("zopflipng compression", cmd2)
        
        # Test 3: pngcrush compression (if available)
        if available_tools["pngcrush"]:
            output_dir = output_base / "test3_pngcrush"
            cmd3 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "pngcrush"
            ]
            run_test("pngcrush compression", cmd3)
        
        # Test 4: Archive creation with zstd (if available)
        if available_tools["zstd"]:
            output_dir = output_base / "test4_zstd_archive"
            cmd4 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "oxipng",
                "--png-effort-level", "4",
                "--create-archives",
                "--archive-tool", "zstd",
                "--archive-compression-level", "9",
                "--batch-size-mb", "1",  # Small batches for testing
                "--zstd-long-range"
            ]
            run_test("zstd archive creation", cmd4)
        
        # Test 5: Archive creation with 7z (if available)
        if available_tools["7z"]:
            output_dir = output_base / "test5_7z_archive"
            cmd5 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "oxipng",
                "--png-effort-level", "4",
                "--create-archives",
                "--archive-tool", "7z",
                "--archive-compression-level", "5",
                "--batch-size-mb", "1"
            ]
            run_test("7z archive creation", cmd5)
        
        # Test 6: Archive creation with pigz (if available)
        if available_tools["pigz"]:
            output_dir = output_base / "test6_pigz_archive"
            cmd6 = [
                "python", "image_converter.py",
                str(test_images_dir),
                str(output_dir),
                str(output_dir / "report.csv"),
                "--png-compression-tool", "oxipng",
                "--png-effort-level", "4",
                "--create-archives",
                "--archive-tool", "pigz",
                "--archive-compression-level", "9",
                "--batch-size-mb", "1"
            ]
            run_test("pigz archive creation", cmd6)
        
        # Test 7: Dry run
        output_dir = output_base / "test7_dry_run"
        cmd7 = [
            "python", "image_converter.py",
            str(test_images_dir),
            str(output_dir),
            str(output_dir / "report.csv"),
            "--png-compression-tool", "oxipng",
            "--create-archives",
            "--archive-tool", "zstd",
            "--dry-run"
        ]
        run_test("Dry run mode", cmd7)
        
        # Test 8: Similarity filtering
        output_dir = output_base / "test8_filtering"
        cmd8 = [
            "python", "image_converter.py",
            str(test_images_dir),
            str(output_dir),
            str(output_dir / "report.csv"),
            "--png-compression-tool", "oxipng",
            "--filter-similar",
            "--similarity-threshold", "0.8",
            "--min-group-size", "2"
        ]
        run_test("Similarity filtering", cmd8)
        
        # Test 9: Backward compatibility (original pillow mode)
        output_dir = output_base / "test9_pillow"
        cmd9 = [
            "python", "image_converter.py",
            str(test_images_dir),
            str(output_dir),
            str(output_dir / "report.csv"),
            "--format", "png",
            "--compression-level", "9"
        ]
        run_test("Backward compatibility (pillow)", cmd9)
        
        # Test 10: Error handling (non-existent tool)
        output_dir = output_base / "test10_error"
        cmd10 = [
            "python", "image_converter.py",
            str(test_images_dir),
            str(output_dir),
            str(output_dir / "report.csv"),
            "--png-compression-tool", "nonexistent_tool"
        ]
        run_test("Error handling (invalid tool)", cmd10, expected_success=False)
    
    print("\n" + "="*60)
    print("✅ Test suite completed!")
    print("="*60)
    
    print("\n📊 Summary:")
    print(f"   Available tools: {sum(available_tools.values())}/{len(available_tools)}")
    
    if not any(available_tools.values()):
        print("\n⚠️  No compression tools found!")
        print("   Install tools to test full functionality:")
        print("   pip install oxipng")
        print("   brew install zopflipng pngcrush zstd p7zip pigz")
    
    print("\n💡 Next steps:")
    print("   - Run with real image data")
    print("   - Test with larger datasets")
    print("   - Benchmark performance")
    print("   - Check compression ratios")

if __name__ == "__main__":
    main()
