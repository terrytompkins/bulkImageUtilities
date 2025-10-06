#!/usr/bin/env python3
"""
Filter & Report for FNA Runs (One-Pass Upload Workflow)

Analyzes images in an input folder with selectable filters and emits a CSV
report listing:
  - filename
  - include (True/False)
  - scores/metrics (focus variance, mean brightness, % dark/bright pixels)
  - reason (why included/excluded)
Also prints total analysis time and counts.

Filters:
  1) blur/focus: Variance of Laplacian (higher = sharper). Include if >= threshold.
  2) brightness/exposure: Mean brightness within [min,max] AND saturated pixels (% dark or bright) <= limit.

You can require BOTH with --algorithm all (i.e., include only if both pass).

Dependencies: numpy, opencv-python
Install: pip install numpy opencv-python
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
import csv
import sys
import os
import json
from typing import Dict, Tuple, Optional, List, Any
import concurrent.futures

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    files.sort()
    return files


def read_grayscale_resized(img_path: Path, max_side: int) -> Optional[np.ndarray]:
    """Read as grayscale, optionally resize so the longer side <= max_side for speed."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if max_side and max(img.shape[:2]) > max_side:
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def focus_score_variance_of_laplacian(gray: np.ndarray) -> float:
    # Variance of Laplacian as focus/blur metric
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_metrics(gray: np.ndarray, dark_thr: int, bright_thr: int) -> Tuple[float, float, float]:
    """
    Returns: (mean_brightness, pct_dark, pct_bright)
    pct_* are in percent [0..100]. Pixels <= dark_thr are 'dark'; >= bright_thr are 'bright'.
    """
    mean_brightness = float(gray.mean())
    total = gray.size
    if total == 0:
        return mean_brightness, 0.0, 0.0
    pct_dark = float((gray <= dark_thr).sum()) * 100.0 / total
    pct_bright = float((gray >= bright_thr).sum()) * 100.0 / total
    return mean_brightness, pct_dark, pct_bright


def decide_include(
    algo: str,
    focus_val: Optional[float],
    brightness_mean: Optional[float],
    pct_dark: Optional[float],
    pct_bright: Optional[float],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    """
    Returns: (include?, reason)
    """
    reason_parts = []

    pass_focus = True
    pass_brightness = True

    if algo in ("focus", "all"):
        if focus_val is None:
            pass_focus = False
            reason_parts.append("focus:missing")
        else:
            pass_focus = focus_val >= args.focus_threshold
            reason_parts.append(f"focus:{'pass' if pass_focus else 'fail'}(%.2f>=%.2f)" % (focus_val, args.focus_threshold))

    if algo in ("brightness", "all"):
        if brightness_mean is None or pct_dark is None or pct_bright is None:
            pass_brightness = False
            reason_parts.append("brightness:missing")
        else:
            in_range = (args.brightness_min <= brightness_mean <= args.brightness_max)
            under_sat = (pct_dark <= args.saturation_pct_limit) and (pct_bright <= args.saturation_pct_limit)
            pass_brightness = in_range and under_sat
            reason_parts.append("brightness:%s(mean=%.2f in[%g,%g], dark=%.2f%%, bright=%.2f%%, sat<=%g%%)"
                                % ("pass" if pass_brightness else "fail",
                                   brightness_mean, args.brightness_min, args.brightness_max,
                                   pct_dark, pct_bright, args.saturation_pct_limit))

    if algo == "focus":
        include = pass_focus
    elif algo == "brightness":
        include = pass_brightness
    else:  # all
        include = pass_focus and pass_brightness

    return include, "; ".join(reason_parts) if reason_parts else ("pass" if include else "fail")


def analyze_one_composite(img_path: Path, args: argparse.Namespace, metadata_map: Dict[str, Dict[str, Any]]) -> Dict[str, object]:
    """
    Analyze a single image for composite algorithms (includes both quality analysis and metadata).
    """
    out = {"filename": str(img_path.relative_to(args.study_dir / "images" / "raw_imageset"))}
    
    # Get metadata for this image
    filename_key = out["filename"]
    metadata = metadata_map.get(filename_key, {})
    
    # Add metadata fields
    out["ILLUMINATION_MODE"] = metadata.get("illumination_mode", "")
    out["LED_COLOR"] = metadata.get("led_color", "")
    out["Z_OFFSET_MODE"] = metadata.get("z_offset_mode", "")
    out["EXPOSURE_MULTIPLIER"] = metadata.get("exposure_multiplier", "")
    
    # Get file size
    file_size, comments = get_file_size_with_handling(filename_key, args.study_dir)
    out["IMAGE_SIZE"] = file_size
    out["COMMENTS"] = comments
    
    # Perform quality analysis if requested
    focus_val = brightness_mean = pct_dark = pct_bright = None
    
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            out["error"] = "unreadable"
            return out
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for performance if needed
        if args.max_side > 0:
            h, w = gray.shape
            if max(h, w) > args.max_side:
                scale = args.max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h))
        
        # Focus analysis
        if "focus" in args.algorithm or args.algorithm == "all":
            focus_val = focus_score_variance_of_laplacian(gray)
            out["focus_score"] = round(focus_val, 4)
        
        # Brightness analysis
        if "brightness" in args.algorithm or args.algorithm == "all":
            brightness_mean, pct_dark, pct_bright = brightness_metrics(gray, args.dark_threshold, args.bright_threshold)
            out["brightness_mean"] = round(brightness_mean, 4)
            out["pct_dark"] = round(pct_dark, 4)
            out["pct_bright"] = round(pct_bright, 4)
        
        # Determine inclusion based on quality criteria
        include, reason = decide_include(
            args.algorithm, focus_val, brightness_mean, pct_dark, pct_bright, args
        )
        out["include"] = include
        out["reason"] = reason
        
    except Exception as e:
        out["error"] = f"exception:{type(e).__name__}:{e}"
        out["include"] = False
        out["reason"] = "processing_error"
    
    return out


def analyze_one(img_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {
        "filename": str(img_path.name),
        "include": False,
        "focus_score": None,
        "brightness_mean": None,
        "pct_dark": None,
        "pct_bright": None,
        "reason": "",
        "error": "",
    }
    try:
        gray = read_grayscale_resized(img_path, args.max_side)
        if gray is None:
            out["error"] = "unreadable"
            return out

        focus_val = brightness_mean = pct_dark = pct_bright = None

        if args.algorithm in ("focus", "all"):
            focus_val = focus_score_variance_of_laplacian(gray)
            out["focus_score"] = round(focus_val, 4)

        if args.algorithm in ("brightness", "all"):
            brightness_mean, pct_dark, pct_bright = brightness_metrics(gray, args.dark_threshold, args.bright_threshold)
            out["brightness_mean"] = round(brightness_mean, 4)
            out["pct_dark"] = round(pct_dark, 4)
            out["pct_bright"] = round(pct_bright, 4)

        include, reason = decide_include(
            args.algorithm, focus_val, brightness_mean, pct_dark, pct_bright, args
        )
        out["include"] = include
        out["reason"] = reason
        return out

    except Exception as e:
        out["error"] = f"exception:{type(e).__name__}:{e}"
        return out


def write_composite_csv(rows: list[Dict[str, object]], study_dir: Path, csv_path: Path, algorithm: str) -> None:
    """
    Write composite CSV with both quality scores and metadata columns.
    """
    # Extract the study directory name
    study_dir_name = study_dir.name
    
    # Define base columns
    base_columns = ["filename", "include", "reason", "error"]
    
    # Add quality columns based on algorithm
    quality_columns = []
    if "focus" in algorithm or algorithm == "all":
        quality_columns.extend(["focus_score"])
    if "brightness" in algorithm or algorithm == "all":
        quality_columns.extend(["brightness_mean", "pct_dark", "pct_bright"])
    
    # Add metadata columns (always present for composite algorithms)
    metadata_columns = ["IMAGE_DIR", "ILLUMINATION_MODE", "LED_COLOR", "Z_OFFSET_MODE", "EXPOSURE_MULTIPLIER", "IMAGE_SIZE", "COMMENTS"]
    
    # Combine all columns
    fieldnames = base_columns + quality_columns + metadata_columns
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        total_images = len(rows)
        for i, row in enumerate(rows):
            # Ensure all required fields are present
            output_row = {}
            for field in fieldnames:
                output_row[field] = row.get(field, "")
            
            # Add study directory name
            output_row["IMAGE_DIR"] = study_dir_name
            
            writer.writerow(output_row)
            
            # Progress reporting every 100 files
            if (i + 1) % 100 == 0:
                print(f"Written {i + 1}/{total_images} rows...")
    
    print(f"Completed writing {total_images} rows to CSV.")


def write_csv(rows: list[Dict[str, object]], csv_path: Path, algorithm: str) -> None:
    # Define columns based on algorithm
    base_columns = ["filename", "include", "reason", "error"]
    
    if algorithm == "focus":
        fieldnames = base_columns + ["focus_score"]
    elif algorithm == "brightness":
        fieldnames = base_columns + ["brightness_mean", "pct_dark", "pct_bright"]
    elif algorithm == "all":
        fieldnames = base_columns + ["focus_score", "brightness_mean", "pct_dark", "pct_bright"]
    else:
        # Fallback to all columns for unknown algorithms
        fieldnames = base_columns + ["focus_score", "brightness_mean", "pct_dark", "pct_bright"]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: list[Dict[str, object]]) -> Dict[str, int]:
    total = len(rows)
    included = sum(1 for r in rows if r.get("include"))
    excluded = total - included
    errors = sum(1 for r in rows if r.get("error"))
    return {"total": total, "included": included, "excluded": excluded, "errors": errors}


def load_algorithm_dev_json(study_dir: Path) -> Dict[str, Any]:
    """
    Load and parse algorithm_dev.json file with detailed error handling.
    Returns the parsed JSON data.
    Raises SystemExit with detailed error information if parsing fails.
    """
    algorithm_file = study_dir / "images" / "algo_dev_imageset" / "algorithm_dev.json"
    
    if not algorithm_file.exists():
        print(f"[ERROR] algorithm_dev.json not found in {algorithm_file}", file=sys.stderr)
        raise SystemExit(3)
    
    try:
        with algorithm_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"[ERROR] Malformed JSON in algorithm_dev.json:", file=sys.stderr)
        print(f"  Error: {e.msg}", file=sys.stderr)
        print(f"  Line: {e.lineno}, Column: {e.colno}", file=sys.stderr)
        print(f"  Position: {e.pos}", file=sys.stderr)
        raise SystemExit(4)
    except Exception as e:
        print(f"[ERROR] Failed to read algorithm_dev.json: {e}", file=sys.stderr)
        raise SystemExit(5)


def extract_image_metadata(algorithm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract image metadata from algorithm_dev.json data.
    Returns a list of dictionaries containing image information.
    """
    images = []
    
    # Look for image data in the JSON structure
    # The structure might vary, so we'll search for objects with FILENAME
    def find_images_recursive(obj, path=""):
        if isinstance(obj, dict):
            if "FILENAME" in obj:
                # This looks like an image record
                image_info = {
                    "filename": obj.get("FILENAME", ""),
                    "illumination_mode": obj.get("ILLUMINATION_MODE", ""),
                    "led_color": obj.get("LED_COLOR", ""),
                    "z_offset_mode": obj.get("Z_OFFSET_MODE", ""),
                    "exposure_multiplier": obj.get("EXPOSURE_MULTIPLIER", ""),
                }
                images.append(image_info)
            else:
                # Recursively search nested objects
                for key, value in obj.items():
                    find_images_recursive(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            # Search through lists
            for i, item in enumerate(obj):
                find_images_recursive(item, f"{path}[{i}]")
    
    find_images_recursive(algorithm_data)
    
    if not images:
        print("[ERROR] No image records found in algorithm_dev.json", file=sys.stderr)
        raise SystemExit(6)
    
    return images


def get_file_size_with_handling(filename: str, study_dir: Path) -> Tuple[int, str]:
    """
    Get file size for an image file, handling missing files.
    Returns: (file_size_bytes, comments)
    """
    # Extract just the filename from the full path
    image_filename = Path(filename).name
    image_path = study_dir / "images" / "raw_imageset" / image_filename
    
    if not image_path.exists():
        return 0, "file_missing"
    
    try:
        return image_path.stat().st_size, ""
    except OSError as e:
        return 0, f"access_error:{e}"


def write_scantypes_csv(images: List[Dict[str, Any]], study_dir: Path, csv_path: Path) -> None:
    """
    Write scantypes analysis CSV with file size information.
    """
    # Extract the study directory name
    study_dir_name = study_dir.name
    fieldnames = ["FILENAME", "IMAGE_DIR", "ILLUMINATION_MODE", "LED_COLOR", "Z_OFFSET_MODE", "EXPOSURE_MULTIPLIER", "IMAGE_SIZE", "COMMENTS"]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        total_images = len(images)
        for i, image in enumerate(images):
            # Get file size
            file_size, comments = get_file_size_with_handling(image["filename"], study_dir)
            
            # Write row
            row = {
                "FILENAME": image["filename"],
                "IMAGE_DIR": study_dir_name,
                "ILLUMINATION_MODE": image["illumination_mode"],
                "LED_COLOR": image["led_color"],
                "Z_OFFSET_MODE": image["z_offset_mode"],
                "EXPOSURE_MULTIPLIER": image["exposure_multiplier"],
                "IMAGE_SIZE": file_size,
                "COMMENTS": comments
            }
            writer.writerow(row)
            
            # Progress reporting every 100 files
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_images} images...")
    
    print(f"Completed processing {total_images} images.")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze images with composite filters and emit CSV report."
    )
    p.add_argument("--study-dir", required=True, type=Path, help="Top-level study directory containing images/ and algorithm_dev.json")
    p.add_argument("--output-csv", required=True, type=Path, help="Where to write the CSV report")
    p.add_argument("--algorithm", 
                   choices=["focus", "brightness", "scantypes", "focus+brightness", "focus+scantypes", "brightness+scantypes", "all"], 
                   required=True,
                   help="Filter combination: focus (sharpness), brightness (lighting), scantypes (metadata), or combinations (e.g., focus+brightness+scantypes)")

    # Focus params
    p.add_argument("--focus-threshold", type=float, default=120.0,
                   help="Minimum variance-of-Laplacian to include (higher is sharper). Default: 120.0")

    # Brightness params
    p.add_argument("--brightness-min", type=float, default=10.0,
                   help="Minimum acceptable mean brightness (0-255). Default: 10.0")
    p.add_argument("--brightness-max", type=float, default=245.0,
                   help="Maximum acceptable mean brightness (0-255). Default: 245.0")
    p.add_argument("--dark-threshold", type=int, default=5,
                   help="Pixel value (<=) considered dark. Default: 5")
    p.add_argument("--bright-threshold", type=int, default=250,
                   help="Pixel value (>=) considered bright. Default: 250")
    p.add_argument("--saturation-pct-limit", type=float, default=2.0,
                   help="Max allowed %% of dark or bright pixels. Default: 2.0%%")

    # Performance / IO
    p.add_argument("--max-side", type=int, default=512,
                   help="Resize longer side to this many pixels for analysis speed (no change to originals). 0 disables. Default: 512")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subfolders")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto based on CPU).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.study_dir.exists() or not args.study_dir.is_dir():
        print(f"[ERROR] --study-dir does not exist or is not a directory: {args.study_dir}", file=sys.stderr)
        return 2

    # Handle composite algorithms (those that include scantypes analysis)
    if "scantypes" in args.algorithm or args.algorithm == "all":
        return handle_composite_algorithm(args)
    
    # Handle single algorithms (focus, brightness, focus+brightness)
    return handle_single_algorithm(args)


def handle_single_algorithm(args: argparse.Namespace) -> int:
    """
    Handle single algorithms: focus, brightness, or focus+brightness
    """
    # Determine image directory from study directory
    image_dir = args.study_dir / "images" / "raw_imageset"
    
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}", file=sys.stderr)
        return 2
    
    imgs = list_images(image_dir, args.recursive)
    if not imgs:
        print(f"[WARN] No images found in {image_dir} (recursive={args.recursive}). Extensions: {sorted(IMAGE_EXTS)}")
        return 0

    print(f"=== {args.algorithm.upper()} Analysis ===")
    print(f"Study directory: {args.study_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Found {len(imgs)} images")

    start = time.perf_counter()

    rows: list[Dict[str, object]] = []
    worker_count = args.workers or max(1, (os.cpu_count() or 2) - 1)
    # Use ThreadPool; OpenCV releases GIL internally for heavy ops, and IO benefits too.
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {ex.submit(analyze_one, p, args): p for p in imgs}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"Processed {done}/{len(imgs)}...")

    rows.sort(key=lambda r: r["filename"])

    write_csv(rows, args.output_csv, args.algorithm)
    elapsed = time.perf_counter() - start

    summary = summarize(rows)
    print("=== Analysis Summary ===")
    print(f"Total files:      {summary['total']}")
    print(f"Included:         {summary['included']}")
    print(f"Excluded:         {summary['excluded']}")
    print(f"Errors:           {summary['errors']}")
    print(f"Elapsed seconds:  {elapsed:.2f}")

    return 0


def handle_composite_algorithm(args: argparse.Namespace) -> int:
    """
    Handle composite algorithms that include scantypes: focus+scantypes, brightness+scantypes, all
    """
    print(f"=== {args.algorithm.upper()} Analysis ===")
    print(f"Study directory: {args.study_dir}")
    print(f"Output CSV: {args.output_csv}")
    
    start = time.perf_counter()
    
    try:
        # Load and parse algorithm_dev.json
        print("Loading algorithm_dev.json...")
        algorithm_data = load_algorithm_dev_json(args.study_dir)
        
        # Extract image metadata
        print("Extracting image metadata...")
        images_metadata = extract_image_metadata(algorithm_data)
        print(f"Found {len(images_metadata)} image records")
        
        # Create a mapping of filename to metadata for quick lookup
        # Support both full path and just filename as keys
        metadata_map = {}
        for img in images_metadata:
            full_path = img["filename"]
            just_filename = Path(full_path).name
            metadata_map[full_path] = img
            metadata_map[just_filename] = img
        
        # Determine image directory from study directory
        image_dir = args.study_dir / "images" / "raw_imageset"
        
        if not image_dir.exists():
            print(f"[ERROR] Image directory not found: {image_dir}", file=sys.stderr)
            return 2
        
        # Get list of actual image files
        imgs = list_images(image_dir, args.recursive)
        if not imgs:
            print(f"[WARN] No images found in {image_dir} (recursive={args.recursive}). Extensions: {sorted(IMAGE_EXTS)}")
            return 0
        
        print(f"Found {len(imgs)} image files")
        
        # Process images with both quality analysis and metadata
        rows: list[Dict[str, object]] = []
        worker_count = args.workers or max(1, (os.cpu_count() or 2) - 1)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {ex.submit(analyze_one_composite, p, args, metadata_map): p for p in imgs}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                rows.append(fut.result())
                done += 1
                if done % 200 == 0:
                    print(f"Processed {done}/{len(imgs)}...")
        
        rows.sort(key=lambda r: r["filename"])
        
        write_composite_csv(rows, args.study_dir, args.output_csv, args.algorithm)
        elapsed = time.perf_counter() - start
        
        # Summary
        summary = summarize(rows)
        print("=== Analysis Summary ===")
        print(f"Total files:      {summary['total']}")
        print(f"Included:         {summary['included']}")
        print(f"Excluded:         {summary['excluded']}")
        print(f"Errors:           {summary['errors']}")
        print(f"Elapsed seconds:  {elapsed:.2f}")
        
        return 0
        
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"[ERROR] Unexpected error in composite analysis: {e}", file=sys.stderr)
        return 7


def handle_scantypes_algorithm(args: argparse.Namespace) -> int:
    """
    Handle the scantypes algorithm: analyze scan type groupings from algorithm_dev.json
    """
    print("=== Scantypes Analysis ===")
    print(f"Study directory: {args.study_dir}")
    print(f"Output CSV: {args.output_csv}")
    
    start = time.perf_counter()
    
    try:
        # Load and parse algorithm_dev.json
        print("Loading algorithm_dev.json...")
        algorithm_data = load_algorithm_dev_json(args.study_dir)
        
        # Extract image metadata
        print("Extracting image metadata...")
        images = extract_image_metadata(algorithm_data)
        print(f"Found {len(images)} image records")
        
        # Write scantypes CSV
        print("Generating scantypes CSV...")
        write_scantypes_csv(images, args.study_dir, args.output_csv)
        
        elapsed = time.perf_counter() - start
        
        # Summary
        print("=== Scantypes Analysis Summary ===")
        print(f"Total image records: {len(images)}")
        print(f"Output file: {args.output_csv}")
        print(f"Elapsed seconds: {elapsed:.2f}")
        
        return 0
        
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"[ERROR] Unexpected error in scantypes analysis: {e}", file=sys.stderr)
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
