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
import glob
import math

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
    
    print("   Extracting image features...")
    for i, path in enumerate(image_paths):
        if i % 100 == 0:  # Progress indicator
            print(f"   Processed {i}/{len(image_paths)} images...")
        features = extract_image_features(path)
        if features is not None:
            features_list.append(features)
            valid_paths.append(path)
    
    if len(features_list) < 2:
        print("⚠️  Not enough valid images for grouping")
        return [[path] for path in valid_paths]
    
    print(f"   Calculating similarity matrix for {len(features_list)} images...")
    
    # For large datasets, use a more efficient approach
    if len(features_list) > 1000:
        print("   Using optimized similarity calculation for large dataset...")
        # Use a sample-based approach for very large datasets
        sample_size = min(1000, len(features_list))
        sample_indices = np.random.choice(len(features_list), sample_size, replace=False)
        sample_features = [features_list[i] for i in sample_indices]
        
        # Calculate similarity for sample
        n_sample = len(sample_features)
        similarity_matrix = np.zeros((n_sample, n_sample))
        
        for i in range(n_sample):
            for j in range(i+1, n_sample):
                similarity = calculate_image_similarity(sample_features[i], sample_features[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Use sample for clustering
        distance_matrix = 1 - similarity_matrix
        clustering = DBSCAN(eps=1-similarity_threshold, min_samples=min_group_size, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Map back to full dataset using nearest neighbor
        print("   Mapping sample clusters to full dataset...")
        grouped_images = []
        processed = set()
        
        for label in set(cluster_labels):
            if label == -1:  # Noise points
                continue
            group_indices = [i for i, l in enumerate(cluster_labels) if l == label]
            if len(group_indices) >= min_group_size:
                # Find similar images in full dataset
                group_paths = []
                for idx in group_indices:
                    sample_feature = sample_features[idx]
                    for i, feature in enumerate(features_list):
                        if i not in processed:
                            similarity = calculate_image_similarity(sample_feature, feature)
                            if similarity >= similarity_threshold:
                                group_paths.append(valid_paths[i])
                                processed.add(i)
                if len(group_paths) >= min_group_size:
                    grouped_images.append(group_paths)
        
        # Add remaining images as individual groups
        for i in range(len(valid_paths)):
            if i not in processed:
                grouped_images.append([valid_paths[i]])
    else:
        # For smaller datasets, use full similarity matrix
        n_images = len(features_list)
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            if i % 50 == 0:  # Progress indicator
                print(f"   Calculating similarities: {i}/{n_images}...")
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

def compress_png_with_oxipng(png_path, effort_level=4, strip_metadata=True):
    """
    Compress PNG using oxipng with specified effort level.
    
    Args:
        png_path (Path): Path to PNG file
        effort_level (int): Compression effort (0-7, higher = smaller but slower)
        strip_metadata (bool): Whether to strip non-essential metadata
        
    Returns:
        dict: Compression results
    """
    if not which("oxipng"):
        return {"success": False, "error": "oxipng not found in PATH"}
    
    original_size = png_path.stat().st_size
    result = {
        "tool": "oxipng",
        "filename": str(png_path),
        "original_size": original_size,
        "compressed_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        cmd = ["oxipng", f"-o{effort_level}", "-T", "0"]
        if strip_metadata:
            cmd.extend(["--strip", "safe"])
        cmd.extend(["-r", str(png_path)])
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            compressed_size = png_path.stat().st_size
            result.update({
                "success": True,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size
            })
        else:
            result["error"] = process.stderr.strip()
            
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
    
    return result

def compress_png_with_zopflipng(png_path, iterations=15, filters="01234mepb"):
    """
    Compress PNG using zopflipng with specified parameters.
    
    Args:
        png_path (Path): Path to PNG file
        iterations (int): Number of optimization iterations
        filters (str): Filter strategies to try
        
    Returns:
        dict: Compression results
    """
    if not which("zopflipng"):
        return {"success": False, "error": "zopflipng not found in PATH"}
    
    original_size = png_path.stat().st_size
    result = {
        "tool": "zopflipng",
        "filename": str(png_path),
        "original_size": original_size,
        "compressed_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        # Create temporary output file
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_output = tmp_file.name
        
        cmd = [
            "zopflipng", "--lossless", "-y",
            f"--iterations={iterations}",
            f"--filters={filters}",
            str(png_path), temp_output
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            compressed_size = Path(temp_output).stat().st_size
            # Replace original with compressed version
            os.replace(temp_output, png_path)
            result.update({
                "success": True,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size
            })
        else:
            result["error"] = process.stderr.strip()
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    return result

def compress_png_with_pngcrush(png_path, brute_force=True, reduce_colors=True):
    """
    Compress PNG using pngcrush with specified parameters.
    
    Args:
        png_path (Path): Path to PNG file
        brute_force (bool): Use exhaustive compression trials
        reduce_colors (bool): Apply lossless bit-depth/color-type reductions
        
    Returns:
        dict: Compression results
    """
    if not which("pngcrush"):
        return {"success": False, "error": "pngcrush not found in PATH"}
    
    original_size = png_path.stat().st_size
    result = {
        "tool": "pngcrush",
        "filename": str(png_path),
        "original_size": original_size,
        "compressed_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        cmd = ["pngcrush"]
        if brute_force:
            cmd.append("-brute")
        if reduce_colors:
            cmd.append("-reduce")
        cmd.extend(["-ow", str(png_path)])
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            compressed_size = png_path.stat().st_size
            result.update({
                "success": True,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size
            })
        else:
            result["error"] = process.stderr.strip()
            
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
    
    return result

def create_archive_with_zstd(source_dir, output_path, compression_level=9, long_range=True, ultra=False, dictionary_path=None):
    """
    Create archive using tar + zstd compression.
    
    Args:
        source_dir (Path): Directory to archive
        output_path (Path): Output archive path
        compression_level (int): zstd compression level (1-22)
        long_range (bool): Enable long-range matching
        ultra (bool): Enable ultra compression mode
        dictionary_path (str): Path to zstd dictionary file
        
    Returns:
        dict: Archive creation results
    """
    if not which("zstd"):
        return {"success": False, "error": "zstd not found in PATH"}
    
    source_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
    result = {
        "tool": "tar+zstd",
        "source_dir": str(source_dir),
        "output_path": str(output_path),
        "source_size": source_size,
        "archive_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        # Build zstd command
        zstd_cmd = f"zstd -T0 -{compression_level}"
        if long_range:
            zstd_cmd += " --long=27"
        if ultra:
            zstd_cmd += " --ultra"
        if dictionary_path:
            zstd_cmd += f" --dict={dictionary_path}"
        
        cmd = ["tar", "-I", zstd_cmd, "-cvf", str(output_path), str(source_dir)]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            archive_size = output_path.stat().st_size
            result.update({
                "success": True,
                "archive_size": archive_size,
                "compression_ratio": archive_size / source_size
            })
        else:
            result["error"] = process.stderr.strip()
            
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
    
    return result

def create_archive_with_7z(source_dir, output_path, compression_level=7, solid=True, multithread=True):
    """
    Create archive using 7z with LZMA2 compression.
    
    Args:
        source_dir (Path): Directory to archive
        output_path (Path): Output archive path
        compression_level (int): 7z compression level (0-9)
        solid (bool): Enable solid compression
        multithread (bool): Enable multithreading
        
    Returns:
        dict: Archive creation results
    """
    if not which("7z"):
        return {"success": False, "error": "7z not found in PATH"}
    
    source_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
    result = {
        "tool": "7z",
        "source_size": source_size,
        "archive_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        cmd = ["7z", "a", "-t7z", "-m0=lzma2", f"-mx={compression_level}"]
        if solid:
            cmd.append("-ms=on")
        if multithread:
            cmd.append("-mmt=on")
        cmd.extend([str(output_path), str(source_dir)])
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            archive_size = output_path.stat().st_size
            result.update({
                "success": True,
                "archive_size": archive_size,
                "compression_ratio": archive_size / source_size
            })
        else:
            result["error"] = process.stderr.strip()
            
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
    
    return result

def create_archive_with_pigz(source_dir, output_path, compression_level=9, parallel=True):
    """
    Create archive using tar + pigz compression.
    
    Args:
        source_dir (Path): Directory to archive
        output_path (Path): Output archive path
        compression_level (int): pigz compression level (1-9)
        parallel (bool): Enable parallel processing
        
    Returns:
        dict: Archive creation results
    """
    if not which("pigz"):
        return {"success": False, "error": "pigz not found in PATH"}
    
    source_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
    result = {
        "tool": "tar+pigz",
        "source_size": source_size,
        "archive_size": None,
        "compression_ratio": None,
        "success": False,
        "error": None,
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        pigz_cmd = f"pigz -{compression_level}"
        if parallel:
            pigz_cmd += " -p 0"
        
        cmd = ["tar", "-I", pigz_cmd, "-cvf", str(output_path), str(source_dir)]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        result["end_time"] = datetime.utcnow().isoformat()
        
        if process.returncode == 0:
            archive_size = output_path.stat().st_size
            result.update({
                "success": True,
                "archive_size": archive_size,
                "compression_ratio": archive_size / source_size
            })
        else:
            result["error"] = process.stderr.strip()
            
    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.utcnow().isoformat()
    
    return result

def create_batches(source_dir, target_size_mb=300, output_dir=None):
    """
    Create batches of files with target size in MB.
    
    Args:
        source_dir (Path): Source directory containing files
        target_size_mb (int): Target batch size in MB
        output_dir (Path): Output directory for batches (optional, uses source_dir/batches if None)
        
    Returns:
        list: List of batch directories
    """
    if output_dir is None:
        output_dir = source_dir / "batches"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_size_bytes = target_size_mb * 1024 * 1024
    png_files = list(source_dir.glob("*.png"))
    
    if not png_files:
        return []
    
    batches = []
    current_batch = []
    current_size = 0
    batch_num = 1
    
    for png_file in png_files:
        file_size = png_file.stat().st_size
        
        # Start new batch if adding this file would exceed target
        if current_size + file_size > target_size_bytes and current_batch:
            # Create batch directory and move files
            batch_dir = output_dir / f"part{batch_num:02d}"
            batch_dir.mkdir(exist_ok=True)
            
            for file_path in current_batch:
                dest_path = batch_dir / file_path.name
                import shutil
                shutil.move(str(file_path), str(dest_path))
            
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
            import shutil
            shutil.move(str(file_path), str(dest_path))
        
        batches.append(batch_dir)
    
    return batches

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

def process_png_compression_parallel(source_dir, compression_tool="oxipng", effort_level=4, 
                                   max_workers=None, report_path=None):
    """
    Process PNG compression in parallel using specified tool.
    
    Args:
        source_dir (Path): Source directory
        compression_tool (str): Tool to use ('oxipng', 'zopflipng', 'pngcrush')
        effort_level (int): Compression effort level
        max_workers (int): Maximum parallel workers
        report_path (str): Path to save compression report
        
    Returns:
        list: List of compression results
    """
    png_files = list(source_dir.glob("*.png"))
    
    if not png_files:
        print("No PNG files found for compression.")
        return []
    
    print(f"🧵 Starting parallel PNG compression with {compression_tool}...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if compression_tool == "oxipng":
            futures = [executor.submit(compress_png_with_oxipng, png, effort_level) for png in png_files]
        elif compression_tool == "zopflipng":
            futures = [executor.submit(compress_png_with_zopflipng, png, effort_level) for png in png_files]
        elif compression_tool == "pngcrush":
            futures = [executor.submit(compress_png_with_pngcrush, png, True, True) for png in png_files]
        else:
            print(f"Unknown compression tool: {compression_tool}")
            return []
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Save report if requested
    if report_path:
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Print summary
    successful = [r for r in results if r["success"]]
    if successful:
        total_original = sum(r["original_size"] for r in successful)
        total_compressed = sum(r["compressed_size"] for r in successful)
        avg_ratio = total_compressed / total_original
        print(f"✅ Compression complete: {len(successful)}/{len(results)} files processed")
        print(f"📊 Average compression ratio: {avg_ratio:.4f} ({((1-avg_ratio)*100):.1f}% reduction)")
    
    return results

def process_archiving_parallel(batch_dirs, archive_tool="zstd", compression_level=9, 
                             output_dir=None, max_workers=None, report_path=None,
                             zstd_long_range=True, zstd_ultra=False, zstd_dictionary=None,
                             sevenz_solid=True, sevenz_multithread=True):
    """
    Process archiving in parallel using specified tool.
    
    Args:
        batch_dirs (list): List of batch directories to archive
        archive_tool (str): Tool to use ('zstd', '7z', 'pigz')
        compression_level (int): Compression level
        output_dir (Path): Output directory for archives
        max_workers (int): Maximum parallel workers
        report_path (str): Path to save archiving report
        zstd_long_range (bool): Enable long-range matching for zstd
        zstd_ultra (bool): Enable ultra compression for zstd
        zstd_dictionary (str): Path to zstd dictionary file
        sevenz_solid (bool): Enable solid compression for 7z
        sevenz_multithread (bool): Enable multithreading for 7z
        
    Returns:
        list: List of archiving results
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Starting parallel archiving with {archive_tool}...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for batch_dir in batch_dirs:
            batch_name = batch_dir.name
            if archive_tool == "zstd":
                output_path = output_dir / f"{batch_name}.tar.zst"
                futures.append(executor.submit(
                    create_archive_with_zstd, 
                    batch_dir, 
                    output_path, 
                    compression_level,
                    zstd_long_range,
                    zstd_ultra,
                    zstd_dictionary
                ))
            elif archive_tool == "7z":
                output_path = output_dir / f"{batch_name}.7z"
                futures.append(executor.submit(
                    create_archive_with_7z, 
                    batch_dir, 
                    output_path, 
                    compression_level,
                    sevenz_solid,
                    sevenz_multithread
                ))
            elif archive_tool == "pigz":
                output_path = output_dir / f"{batch_name}.tar.gz"
                futures.append(executor.submit(
                    create_archive_with_pigz, 
                    batch_dir, 
                    output_path, 
                    compression_level
                ))
            else:
                print(f"Unknown archive tool: {archive_tool}")
                return []
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Save report if requested
    if report_path:
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Print summary
    successful = [r for r in results if r["success"]]
    if successful:
        total_source = sum(r["source_size"] for r in successful)
        total_archive = sum(r["archive_size"] for r in successful)
        avg_ratio = total_archive / total_source
        print(f"✅ Archiving complete: {len(successful)}/{len(results)} archives created")
        print(f"📊 Average compression ratio: {avg_ratio:.4f} ({((1-avg_ratio)*100):.1f}% reduction)")
    
    return results

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
    parser = argparse.ArgumentParser(description="Bulk PNG image converter: compress PNGs, convert to AVIF, JPEG XL, or WebP, and optionally create archives.")
    parser.add_argument("source_dir", help="Directory containing source PNG files")
    parser.add_argument("dest_dir", help="Directory to write converted images")
    parser.add_argument("report_csv", help="Path to write CSV report")
    
    # Format and conversion options
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
    
    # Advanced PNG compression options
    parser.add_argument(
        "--png-compression-tool",
        choices=["pillow", "oxipng", "zopflipng", "pngcrush"],
        default="pillow",
        help="Tool for PNG compression: 'pillow' (built-in), 'oxipng' (fast), 'zopflipng' (max compression), 'pngcrush' (classic) (default: pillow)"
    )
    parser.add_argument(
        "--png-effort-level",
        type=int,
        default=4,
        help="Compression effort level for oxipng (0-7) or zopflipng iterations (default: 4)"
    )
    parser.add_argument(
        "--strip-metadata",
        action="store_true",
        help="Strip non-essential metadata from PNGs (for oxipng)"
    )
    
    # Archiving options
    parser.add_argument(
        "--create-archives",
        action="store_true",
        help="Create compressed archives after image processing"
    )
    parser.add_argument(
        "--archive-tool",
        choices=["zstd", "7z", "pigz"],
        default="zstd",
        help="Archive compression tool: 'zstd' (balanced), '7z' (max compression), 'pigz' (fast) (default: zstd)"
    )
    parser.add_argument(
        "--archive-compression-level",
        type=int,
        default=9,
        help="Archive compression level: zstd (1-22), 7z (0-9), pigz (1-9) (default: 9)"
    )
    parser.add_argument(
        "--batch-size-mb",
        type=int,
        default=300,
        help="Target batch size in MB for archiving (default: 300)"
    )
    parser.add_argument(
        "--archive-output-dir",
        help="Output directory for archives (default: dest_dir)"
    )
    parser.add_argument(
        "--zstd-long-range",
        action="store_true",
        help="Enable long-range matching for zstd (better for similar images)"
    )
    parser.add_argument(
        "--zstd-ultra",
        action="store_true",
        help="Enable ultra compression mode for zstd (slower but smaller)"
    )
    parser.add_argument(
        "--zstd-dictionary",
        help="Path to zstd dictionary file for better compression"
    )
    parser.add_argument(
        "--7z-solid",
        action="store_true",
        default=True,
        help="Enable solid compression for 7z (default: True)"
    )
    parser.add_argument(
        "--7z-multithread",
        action="store_true",
        default=True,
        help="Enable multithreading for 7z (default: True)"
    )
    
    # Parallel processing options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    
    # Filtering options
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
    
    # Report options
    parser.add_argument(
        "--compression-report",
        help="Path to save PNG compression report (JSON format)"
    )
    parser.add_argument(
        "--archive-report",
        help="Path to save archive creation report (JSON format)"
    )
    
    # Dry run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without executing them"
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

    # Check for required tools based on options
    if args.png_compression_tool != "pillow":
        tool_map = {
            "oxipng": "oxipng",
            "zopflipng": "zopflipng", 
            "pngcrush": "pngcrush"
        }
        tool_name = tool_map.get(args.png_compression_tool)
        if tool_name and not which(tool_name):
            print(f"ERROR: {tool_name} not found in PATH. Please install {tool_name}.")
            sys.exit(2)
    
    if args.create_archives:
        archive_tool_map = {
            "zstd": "zstd",
            "7z": "7z",
            "pigz": "pigz"
        }
        tool_name = archive_tool_map.get(args.archive_tool)
        if tool_name and not which(tool_name):
            print(f"ERROR: {tool_name} not found in PATH. Please install {tool_name}.")
            sys.exit(2)

    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - Planned operations:")
        print(f"   Source directory: {args.source_dir}")
        print(f"   Destination directory: {args.dest_dir}")
        print(f"   Target format: {args.format}")
        if args.png_compression_tool != "pillow":
            print(f"   PNG compression tool: {args.png_compression_tool} (effort level: {args.png_effort_level})")
        if args.create_archives:
            print(f"   Archive tool: {args.archive_tool} (compression level: {args.archive_compression_level})")
            print(f"   Batch size: {args.batch_size_mb} MB")
        print("   (No files will be modified)")
        sys.exit(0)

    # Step 1: Convert/compress images
    print("🖼️  Step 1: Processing images...")
    if args.png_compression_tool != "pillow":
        # Use advanced PNG compression tools
        source_path = Path(args.source_dir)
        if args.filter_similar:
            print("🔍 Applying similarity-based filtering...")
            filtered_images, filtering_report = filter_images_by_similarity(
                args.source_dir, args.similarity_threshold, args.min_group_size, args.selection_method
            )
            # Save filtering report
            filtering_report_path = args.report_csv.replace('.csv', '_filtering_report.json')
            with open(filtering_report_path, 'w') as f:
                json.dump(filtering_report, f, indent=2, default=str)
            print(f"📊 Filtering report saved to: {filtering_report_path}")
        
        # Process PNG compression with advanced tools
        compression_results = process_png_compression_parallel(
            source_path,
            args.png_compression_tool,
            args.png_effort_level,
            args.max_workers,
            args.compression_report
        )
        
        # Create basic CSV report for compatibility
        with open(args.report_csv, mode="w", newline="") as csvfile:
            fieldnames = ["filename", "original_size", "compressed_size", "compression_ratio", "tool", "success", "error"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in compression_results:
                if result["success"]:
                    writer.writerow({
                        "filename": Path(result["filename"]).name,
                        "original_size": result["original_size"],
                        "compressed_size": result["compressed_size"],
                        "compression_ratio": f"{result['compression_ratio']:.4f}",
                        "tool": result["tool"],
                        "success": result["success"],
                        "error": result.get("error", "")
                    })
        
        print(f"✅ Image processing complete. Report written to: {args.report_csv}")
        
    else:
        # Use original conversion logic
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

    # Step 2: Create archives if requested
    if args.create_archives:
        print("\n📦 Step 2: Creating archives...")
        
        # Determine source directory for archiving
        if args.png_compression_tool != "pillow":
            # If we used advanced compression, archive from source directory
            archive_source = Path(args.source_dir)
        else:
            # If we used regular conversion, archive from destination directory
            archive_source = Path(args.dest_dir)
        
        # Create batches
        print(f"📁 Creating batches of ~{args.batch_size_mb} MB...")
        batch_dirs = create_batches(archive_source, args.batch_size_mb)
        print(f"✅ Created {len(batch_dirs)} batches")
        
        # Determine archive output directory
        archive_output_dir = Path(args.archive_output_dir) if args.archive_output_dir else Path(args.dest_dir)
        
        # Process archiving
        archive_results = process_archiving_parallel(
            batch_dirs,
            args.archive_tool,
            args.archive_compression_level,
            archive_output_dir,
            args.max_workers,
            args.archive_report,
            args.zstd_long_range,
            args.zstd_ultra,
            args.zstd_dictionary,
            args.sevenz_solid,
            args.sevenz_multithread
        )
        
        print(f"✅ Archive creation complete. Archives saved to: {archive_output_dir}")
        
        # Clean up batch directories
        print("🧹 Cleaning up batch directories...")
        for batch_dir in batch_dirs:
            import shutil
            shutil.rmtree(batch_dir)
        print("✅ Cleanup complete")
