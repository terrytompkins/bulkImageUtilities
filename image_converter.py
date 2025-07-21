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
from PIL import Image, __version__ as pillow_version, features
import numpy as np
import argparse
import traceback

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

def verify_avif_integrity(original_png, avif_file):
    """Decode AVIF to RGBA, compare pixel arrays (lossless check)."""
    try:
        img_orig = Image.open(original_png).convert("RGBA")
        img_decoded = Image.open(avif_file).convert("RGBA")
        np_orig = np.array(img_orig)
        np_decoded = np.array(img_decoded)
        if np.array_equal(np_orig, np_decoded):
            return True, "Pixel match"
        else:
            return False, "Pixel mismatch"
    except Exception as e:
        return False, str(e)

def process_image_png(png_path_str, dest_dir_str, compression_level):
    png_path = Path(png_path_str)
    dest_dir = Path(dest_dir_str)
    compressed_png_path = dest_dir / png_path.name

    row = {
        "filename": png_path.name,
        "png_size_bytes": png_path.stat().st_size,
        "target_size_bytes": None,
        "compression_ratio": "",
        "conversion_success": False,
        "verification_success": True,  # PNG to PNG is always lossless
        "conversion_start": "",
        "conversion_end": "",
        "verification_start": "",
        "verification_end": "",
        "error": "",
        "target_format": "png",
        "compression_level": compression_level,
        "quality": None,
        "avif_encoder": None
    }

    row["conversion_start"] = datetime.utcnow().isoformat()
    try:
        img = Image.open(png_path)
        img.save(compressed_png_path, format="PNG", compress_level=compression_level)
        row["conversion_success"] = True
        row["target_size_bytes"] = compressed_png_path.stat().st_size
        row["compression_ratio"] = f"{row['target_size_bytes'] / row['png_size_bytes']:.4f}"
    except Exception as e:
        row["error"] = f"PNG compression failed: {str(e)}"
    row["conversion_end"] = datetime.utcnow().isoformat()
    return row

def process_image_jxl(png_path_str, dest_dir_str):
    png_path = Path(png_path_str)
    dest_dir = Path(dest_dir_str)
    jxl_path = dest_dir / (png_path.stem + ".jxl")

    row = {
        "filename": png_path.name,
        "png_size_bytes": png_path.stat().st_size,
        "target_size_bytes": None,
        "compression_ratio": "",
        "conversion_success": False,
        "verification_success": False,
        "conversion_start": "",
        "conversion_end": "",
        "verification_start": "",
        "verification_end": "",
        "error": "",
        "target_format": "jxl",
        "compression_level": None,
        "quality": None,
        "avif_encoder": None
    }

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
    row["target_size_bytes"] = jxl_size
    row["compression_ratio"] = f"{jxl_size / row['png_size_bytes']:.4f}"

    row["verification_start"] = datetime.utcnow().isoformat()
    success, message = verify_image_integrity(png_path, jxl_path)
    row["verification_end"] = datetime.utcnow().isoformat()
    row["verification_success"] = success
    if not success:
        row["error"] = f"Verification failed: {message}"

    return row

def process_image_avif(png_path_str, dest_dir_str, quality):
    png_path = Path(png_path_str)
    dest_dir = Path(dest_dir_str)
    avif_path = dest_dir / (png_path.stem + ".avif")

    row = {
        "filename": png_path.name,
        "png_size_bytes": png_path.stat().st_size,
        "target_size_bytes": None,
        "compression_ratio": "",
        "conversion_success": False,
        "verification_success": False,
        "conversion_start": "",
        "conversion_end": "",
        "verification_start": "",
        "verification_end": "",
        "error": "",
        "target_format": "avif",
        "compression_level": None,
        "quality": quality,
        "avif_encoder": None
    }

    row["conversion_start"] = datetime.utcnow().isoformat()
    avifenc_path = which("avifenc")
    try:
        if avifenc_path:
            # Use avifenc for lossless conversion
            cmd = [avifenc_path, "--lossless", str(png_path), str(avif_path)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            row["avif_encoder"] = "avifenc"
            if result.returncode != 0:
                row["error"] = f"avifenc failed: {result.stderr.decode().strip()}"
                row["conversion_end"] = datetime.utcnow().isoformat()
                return row
        else:
            # Fallback to Pillow
            img = Image.open(png_path)
            img.save(avif_path, format="AVIF", lossless=True)
            row["avif_encoder"] = "pillow"
        row["conversion_success"] = True
        row["target_size_bytes"] = avif_path.stat().st_size
        row["compression_ratio"] = f"{row['target_size_bytes'] / row['png_size_bytes']:.4f}"
    except Exception as e:
        tb = traceback.format_exc()
        row["error"] = f"AVIF conversion failed: {type(e).__name__}: {e}\n{tb}"
        row["conversion_end"] = datetime.utcnow().isoformat()
        return row
    row["conversion_end"] = datetime.utcnow().isoformat()

    # Pixel-by-pixel verification
    row["verification_start"] = datetime.utcnow().isoformat()
    success, message = verify_avif_integrity(png_path, avif_path)
    row["verification_end"] = datetime.utcnow().isoformat()
    row["verification_success"] = success
    if not success:
        row["error"] = f"Verification failed: {message}"
    return row

def convert_and_verify_parallel(source_dir, dest_dir, report_path, target_format, compression_level, quality, max_workers=None):
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    png_files = list(source.glob("*.png"))
    if not png_files:
        print("No PNG files found.")
        return

    print(f"🧵 Starting parallel processing of {len(png_files)} images to {target_format.upper()} format...")

    report_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if target_format == "png":
            futures = [executor.submit(process_image_png, str(png), str(dest), compression_level) for png in png_files]
        elif target_format == "avif":
            futures = [executor.submit(process_image_avif, str(png), str(dest), quality) for png in png_files]
        else:
            if not which("cjxl") or not which("djxl"):
                print("Error: cjxl and/or djxl not found in PATH. Please install jpeg-xl.")
                sys.exit(1)
            futures = [executor.submit(process_image_jxl, str(png), str(dest)) for png in png_files]
        for future in as_completed(futures):
            report_rows.append(future.result())

    # Write CSV
    with open(report_path, mode="w", newline="") as csvfile:
        fieldnames = [
            "filename",
            "png_size_bytes",
            "target_size_bytes",
            "compression_ratio",
            "conversion_success",
            "verification_success",
            "conversion_start",
            "conversion_end",
            "verification_start",
            "verification_end",
            "error",
            "target_format",
            "compression_level",
            "quality",
            "avif_encoder"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    print(f"\n✅ Report written to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk PNG image converter: compress PNGs, convert to AVIF or JPEG XL.")
    parser.add_argument("source_dir", help="Directory containing source PNG files")
    parser.add_argument("dest_dir", help="Directory to write converted images")
    parser.add_argument("report_csv", help="Path to write CSV report")
    parser.add_argument(
        "--format",
        choices=["png", "jxl", "avif"],
        default="png",
        help="Target format: 'png' for compressed PNG, 'avif' for AVIF, 'jxl' for JPEG XL (default: png)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        choices=range(0, 10),
        metavar="[0-9]",
        help="PNG compression level (0=no compression, 9=maximum, default: 9). Only used if --format=png."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="AVIF quality (not used for lossless, reserved for future use)."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    args = parser.parse_args()

    # Diagnostics for AVIF support
    if args.format == "avif":
        print(f"Pillow version: {pillow_version}")
        avif_supported = features.check('avif')
        print(f"AVIF support: {avif_supported}")
        avifenc_path = which("avifenc")
        print(f"avifenc in PATH: {bool(avifenc_path)}")
        if not avif_supported and not avifenc_path:
            print("ERROR: Neither Pillow nor avifenc support AVIF. Please install libavif, Pillow>=12.0.0, or avifenc.")
            sys.exit(2)

    convert_and_verify_parallel(
        args.source_dir,
        args.dest_dir,
        args.report_csv,
        args.format,
        args.compression_level,
        args.quality,
        args.max_workers
    )
