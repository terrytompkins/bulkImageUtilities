#!/usr/bin/env python3
"""
Example script demonstrating the image filtering functionality.

This script shows how to use the similarity-based filtering to reduce
the number of images in a dataset while maintaining representative samples.
"""

import sys
from pathlib import Path
from image_converter import filter_images_by_similarity

def main():
    if len(sys.argv) != 2:
        print("Usage: python filter_example.py <source_directory>")
        print("\nThis script will:")
        print("1. Analyze images in the source directory for similarity")
        print("2. Group similar images together")
        print("3. Select representative images from each group")
        print("4. Save the filtered images to a 'filtered' subdirectory")
        print("5. Generate a detailed filtering report")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = source_path / "filtered"
    
    print(f"🔍 Starting image filtering process...")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Apply filtering with default parameters
    filtered_images, report = filter_images_by_similarity(
        source_dir=source_dir,
        similarity_threshold=0.85,  # 85% similarity threshold
        min_group_size=2,          # Minimum 2 images to form a group
        selection_method='best_quality',  # Select best quality image from each group
        output_filtered_dir=str(output_dir)
    )
    
    print("\n📊 Filtering Summary:")
    print(f"   Total images processed: {report['total_images']}")
    print(f"   Images after filtering: {report['filtered_images']}")
    print(f"   Reduction: {report['reduction_percentage']:.1f}%")
    print(f"   Groups found: {report['groups_found']}")
    print(f"   Similarity threshold: {report['similarity_threshold']}")
    print(f"   Selection method: {report['selection_method']}")
    
    print("\n📁 Filtered images have been saved to:", output_dir)
    print("📄 Detailed filtering report saved to:", source_path / "filtering_report.json")

if __name__ == "__main__":
    main() 