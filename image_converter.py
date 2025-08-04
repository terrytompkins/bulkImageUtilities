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
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def extract_image_features(image_path):
    """
    Extract features from an image for similarity comparison.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        dict: Dictionary containing image features
    """
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for comparison
        gray_resized = cv2.resize(gray, (64, 64))
        
        # Extract features
        features = {
            'path': image_path,
            'histogram': cv2.calcHist([gray_resized], [0], None, [256], [0, 256]).flatten(),
            'mean_intensity': np.mean(gray_resized),
            'std_intensity': np.std(gray_resized),
            'size': img.shape[:2],
            'aspect_ratio': img.shape[1] / img.shape[0]
        }
        
        # Normalize histogram
        features['histogram'] = features['histogram'] / np.sum(features['histogram'])
        
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def calculate_image_similarity(features1, features2):
    """
    Calculate similarity between two images based on their features.
    
    Args:
        features1, features2 (dict): Image feature dictionaries
        
    Returns:
        float: Similarity score (0-1, higher is more similar)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    # Histogram similarity using cosine similarity
    hist_similarity = cosine_similarity(
        features1['histogram'].reshape(1, -1), 
        features2['histogram'].reshape(1, -1)
    )[0][0]
    
    # Intensity similarity
    intensity_diff = abs(features1['mean_intensity'] - features2['mean_intensity']) / 255.0
    intensity_similarity = 1.0 - intensity_diff
    
    # Size similarity (if images are similar size)
    size_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
    size_similarity = 1.0 / (1.0 + size_diff)
    
    # Combined similarity score
    similarity = (hist_similarity * 0.6 + intensity_similarity * 0.3 + size_similarity * 0.1)
    return max(0.0, min(1.0, similarity))

def group_similar_images(image_paths, similarity_threshold=0.85, min_group_size=2):
    """
    Group similar images together using clustering.
    
    Args:
        image_paths (list): List of image file paths
        similarity_threshold (float): Minimum similarity to group images (0-1)
        min_group_size (int): Minimum number of images to form a group
        
    Returns:
        list: List of groups, where each group is a list of similar image paths
    """
    print(f"🔍 Analyzing {len(image_paths)} images for similarity...")
    
    # Extract features from all images
    features_list = []
    valid_paths = []
    
    for path in image_paths:
        features = extract_image_features(path)
        if features is not None:
            features_list.append(features)
            valid_paths.append(path)
    
    if len(features_list) < 2:
        print("⚠️  Not enough valid images for grouping")
        return [[path] for path in valid_paths]
    
    # Create similarity matrix
    n_images = len(features_list)
    similarity_matrix = np.zeros((n_images, n_images))
    
    for i in range(n_images):
        for j in range(i+1, n_images):
            similarity = calculate_image_similarity(features_list[i], features_list[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix
    
    # Use DBSCAN clustering
    clustering = DBSCAN(eps=1-similarity_threshold, min_samples=min_group_size, metric='precomputed')
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Group images by cluster
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(valid_paths[i])
    
    # Convert to list format
    grouped_images = list(groups.values())
    
    # Add single images as individual groups
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Noise points (single images)
            grouped_images.append([valid_paths[i]])
    
    print(f"📊 Found {len(grouped_images)} groups:")
    for i, group in enumerate(grouped_images):
        print(f"   Group {i+1}: {len(group)} images")
    
    return grouped_images

def select_representative_images(image_groups, selection_method='best_quality'):
    """
    Select representative images from each group.
    
    Args:
        image_groups (list): List of image groups
        selection_method (str): Method for selecting representative ('best_quality', 'largest', 'smallest', 'first')
        
    Returns:
        list: List of selected representative image paths
    """
    representatives = []
    
    for group in image_groups:
        if len(group) == 1:
            representatives.append(group[0])
            continue
            
        if selection_method == 'best_quality':
            # Select image with highest average intensity (often indicates better quality)
            best_image = max(group, key=lambda p: np.mean(cv2.imread(str(p))))
        elif selection_method == 'largest':
            # Select largest file
            best_image = max(group, key=lambda p: p.stat().st_size)
        elif selection_method == 'smallest':
            # Select smallest file
            best_image = min(group, key=lambda p: p.stat().st_size)
        elif selection_method == 'first':
            # Select first image in group
            best_image = group[0]
        else:
            best_image = group[0]
            
        representatives.append(best_image)
    
    return representatives

def filter_images_by_similarity(source_dir, similarity_threshold=0.85, min_group_size=2, 
                              selection_method='best_quality', output_filtered_dir=None):
    """
    Filter images by similarity to reduce dataset size.
    
    Args:
        source_dir (str): Directory containing source images
        similarity_threshold (float): Minimum similarity to group images (0-1)
        min_group_size (int): Minimum number of images to form a group
        selection_method (str): Method for selecting representative images
        output_filtered_dir (str): Directory to save filtered images (optional)
        
    Returns:
        tuple: (filtered_image_paths, filtering_report)
    """
    source_path = Path(source_dir)
    image_paths = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
    
    if not image_paths:
        print("❌ No image files found in source directory")
        return [], {}
    
    print(f"🔍 Processing {len(image_paths)} images...")
    
    # Group similar images
    image_groups = group_similar_images(image_paths, similarity_threshold, min_group_size)
    
    # Select representatives
    representatives = select_representative_images(image_groups, selection_method)
    
    # Create filtering report
    filtering_report = {
        'total_images': len(image_paths),
        'filtered_images': len(representatives),
        'reduction_percentage': (1 - len(representatives) / len(image_paths)) * 100,
        'groups_found': len(image_groups),
        'similarity_threshold': similarity_threshold,
        'min_group_size': min_group_size,
        'selection_method': selection_method,
        'groups': [
            {
                'size': len(group),
                'representative': str(representatives[i]) if i < len(representatives) else None
            }
            for i, group in enumerate(image_groups)
        ]
    }
    
    # Copy filtered images to output directory if specified
    if output_filtered_dir:
        output_path = Path(output_filtered_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for rep_path in representatives:
            dest_path = output_path / rep_path.name
            import shutil
            shutil.copy2(rep_path, dest_path)
        
        print(f"📁 Filtered images saved to: {output_filtered_dir}")
    
    print(f"✅ Filtering complete: {len(image_paths)} → {len(representatives)} images ({filtering_report['reduction_percentage']:.1f}% reduction)")
    
    return representatives, filtering_report

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

def verify_webp_integrity(original_png, webp_file):
    """Decode WebP to RGBA, compare pixel arrays (lossless check)."""
    try:
        img_orig = Image.open(original_png).convert("RGBA")
        img_decoded = Image.open(webp_file).convert("RGBA")
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

def process_image_webp(png_path_str, dest_dir_str, quality):
    png_path = Path(png_path_str)
    dest_dir = Path(dest_dir_str)
    webp_path = dest_dir / (png_path.stem + ".webp")

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
        "target_format": "webp",
        "compression_level": None,
        "quality": quality,
        "avif_encoder": None
    }

    row["conversion_start"] = datetime.utcnow().isoformat()
    try:
        img = Image.open(png_path)
        img.save(webp_path, format="WEBP", lossless=True)
        row["conversion_success"] = True
        row["target_size_bytes"] = webp_path.stat().st_size
        row["compression_ratio"] = f"{row['target_size_bytes'] / row['png_size_bytes']:.4f}"
    except Exception as e:
        tb = traceback.format_exc()
        row["error"] = f"WebP conversion failed: {type(e).__name__}: {e}\n{tb}"
        row["conversion_end"] = datetime.utcnow().isoformat()
        return row
    row["conversion_end"] = datetime.utcnow().isoformat()

    # Pixel-by-pixel verification
    row["verification_start"] = datetime.utcnow().isoformat()
    success, message = verify_webp_integrity(png_path, webp_path)
    row["verification_end"] = datetime.utcnow().isoformat()
    row["verification_success"] = success
    if not success:
        row["error"] = f"Verification failed: {message}"
    return row

def convert_and_verify_parallel(source_dir, dest_dir, report_path, target_format, compression_level, quality, max_workers=None, 
                              use_filtering=False, similarity_threshold=0.85, min_group_size=2, selection_method='best_quality'):
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(source.glob("*.png")) + list(source.glob("*.jpg")) + list(source.glob("*.jpeg"))
    
    if not image_files:
        print("No image files found.")
        return
    
    # Apply filtering if requested
    if use_filtering:
        print("🔍 Applying similarity-based filtering...")
        filtered_images, filtering_report = filter_images_by_similarity(
            source_dir, similarity_threshold, min_group_size, selection_method
        )
        
        # Save filtering report
        filtering_report_path = report_path.replace('.csv', '_filtering_report.json')
        with open(filtering_report_path, 'w') as f:
            json.dump(filtering_report, f, indent=2, default=str)
        print(f"📊 Filtering report saved to: {filtering_report_path}")
        
        # Use filtered images for conversion
        png_files = [Path(img) for img in filtered_images if Path(img).suffix.lower() == '.png']
        if not png_files:
            print("No PNG files found after filtering.")
            return
    else:
        png_files = [f for f in image_files if f.suffix.lower() == '.png']
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
        elif target_format == "webp":
            futures = [executor.submit(process_image_webp, str(png), str(dest), quality) for png in png_files]
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
    parser = argparse.ArgumentParser(description="Bulk PNG image converter: compress PNGs, convert to AVIF, JPEG XL, or WebP.")
    parser.add_argument("source_dir", help="Directory containing source PNG files")
    parser.add_argument("dest_dir", help="Directory to write converted images")
    parser.add_argument("report_csv", help="Path to write CSV report")
    parser.add_argument(
        "--format",
        choices=["png", "jxl", "avif", "webp"],
        default="png",
        help="Target format: 'png' for compressed PNG, 'avif' for AVIF, 'jxl' for JPEG XL, 'webp' for WebP (default: png)"
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
        help="Quality parameter for AVIF/WebP (not used for lossless, reserved for future use)."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--filter-similar",
        action="store_true",
        help="Enable similarity-based filtering to reduce dataset size"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for grouping images (0-1, default: 0.85)"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum number of images to form a group (default: 2)"
    )
    parser.add_argument(
        "--selection-method",
        choices=["best_quality", "largest", "smallest", "first"],
        default="best_quality",
        help="Method for selecting representative images from groups (default: best_quality)"
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
    
    # Diagnostics for WebP support
    if args.format == "webp":
        print(f"Pillow version: {pillow_version}")
        webp_supported = features.check('webp')
        print(f"WebP support: {webp_supported}")
        if not webp_supported:
            print("ERROR: Pillow does not support WebP. Please install Pillow with WebP support.")
            sys.exit(2)

    convert_and_verify_parallel(
        args.source_dir,
        args.dest_dir,
        args.report_csv,
        args.format,
        args.compression_level,
        args.quality,
        args.max_workers,
        args.filter_similar,
        args.similarity_threshold,
        args.min_group_size,
        args.selection_method
    )
