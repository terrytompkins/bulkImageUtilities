import os
import sys
import csv
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from shutil import which
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def verify_image_integrity(original_png, jxl_file):
    """Decode JXL to PNG, compare pixel arrays (lossless check)."""
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_decoded:
        decoded_png_path = tmp_decoded.name

    result = subprocess.run(
        ["djxl", str(jxl_file), decoded_png_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        os.remove(decoded_png_path)
        return False, "Decode failed"

    try:
        img_orig = Image.open(original_png).convert("RGBA")
        img_decoded = Image.open(decoded_png_path).convert("RGBA")
        os.remove(decoded_png_path)

        np_orig = np.array(img_orig)
        np_decoded = np.array(img_decoded)

        if np.array_equal(np_orig, np_decoded):
            return True, "Pixel match"
        else:
            return False, "Pixel mismatch"
    except Exception as e:
        os.remove(decoded_png_path)
        return False, str(e)

def process_image(png_path_str, dest_dir_str):
    png_path = Path(png_path_str)
    dest_dir = Path(dest_dir_str)
    jxl_path = dest_dir / (png_path.stem + ".jxl")
    
    row = {
        "filename": png_path.name,
        "png_size_bytes": png_path.stat().st_size,
        "conversion_success": False,
        "verification_success": False,
        "compression_ratio": "",
        "conversion_start": "",
        "conversion_end": "",
        "verification_start": "",
        "verification_end": "",
        "error": ""
    }

    # Conversion
    row["conversion_start"] = datetime.utcnow().isoformat()
    result = subprocess.run(
        ["cjxl", str(png_path), str(jxl_path), "-d", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    row["conversion_end"] = datetime.utcnow().isoformat()

    if result.returncode != 0:
        row["error"] = f"Conversion failed: {result.stderr.decode().strip()}"
        return row

    row["conversion_success"] = True
    jxl_size = jxl_path.stat().st_size
    row["jxl_size_bytes"] = jxl_size
    row["compression_ratio"] = f"{jxl_size / row['png_size_bytes']:.4f}"

    # Verification
    row["verification_start"] = datetime.utcnow().isoformat()
    success, message = verify_image_integrity(png_path, jxl_path)
    row["verification_end"] = datetime.utcnow().isoformat()
    row["verification_success"] = success
    if not success:
        row["error"] = f"Verification failed: {message}"

    return row

def convert_and_verify_parallel(source_dir, dest_dir, report_path, max_workers=None):
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not which("cjxl") or not which("djxl"):
        print("Error: cjxl and/or djxl not found in PATH. Please install jpeg-xl.")
        sys.exit(1)

    png_files = list(source.glob("*.png"))
    if not png_files:
        print("No PNG files found.")
        return

    print(f"🧵 Starting parallel processing of {len(png_files)} images...")

    report_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, str(png), str(dest)) for png in png_files]
        for future in as_completed(futures):
            report_rows.append(future.result())

    # Write CSV
    with open(report_path, mode="w", newline="") as csvfile:
        fieldnames = [
            "filename",
            "png_size_bytes",
            "jxl_size_bytes",
            "compression_ratio",
            "conversion_success",
            "verification_success",
            "conversion_start",
            "conversion_end",
            "verification_start",
            "verification_end",
            "error"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    print(f"\n✅ Report written to: {report_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_and_verify_png_to_jxl.py <source_dir> <dest_dir> <report_csv>")
        sys.exit(1)

    convert_and_verify_parallel(sys.argv[1], sys.argv[2], sys.argv[3])
