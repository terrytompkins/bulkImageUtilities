#!/usr/bin/env python3
"""
Example script demonstrating the image filtering functionality.

This script shows how to use the similarity-based filtering to reduce
the number of images in a dataset while maintaining representative samples.
"""

import sys
import argparse
from pathlib import Path
from image_converter import filter_images_by_similarity

def main():
    parser = argparse.ArgumentParser(
        description="Filter similar images to reduce dataset size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use conservative settings for medical images
  python filter_example.py /path/to/images --similarity-threshold 0.95 --min-group-size 5

  # Use moderate settings for general images
  python filter_example.py /path/to/images --similarity-threshold 0.90 --min-group-size 3

  # Use aggressive settings for very similar images
  python filter_example.py /path/to/images --similarity-threshold 0.80 --min-group-size 2

  # Preview mode - analyze without copying files
  python filter_example.py /path/to/images --preview-only
        """
    )
    
    parser.add_argument("source_dir", help="Directory containing source images")
    parser.add_argument(
        "--similarity-threshold", 
        type=float, 
        default=0.92,
        help="Similarity threshold for grouping (0-1, default: 0.92)"
    )
    parser.add_argument(
        "--min-group-size", 
        type=int, 
        default=3,
        help="Minimum images to form a group (default: 3)"
    )
    parser.add_argument(
        "--selection-method", 
        choices=["best_quality", "largest", "smallest", "first"],
        default="best_quality",
        help="Method for selecting representatives (default: best_quality)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: source_dir/filtered)"
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Analyze images without copying files"
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory '{args.source_dir}' does not exist")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = source_path / "filtered"
    
    print(f"🔍 Starting image filtering process...")
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Min group size: {args.min_group_size}")
    print(f"Selection method: {args.selection_method}")
    print()
    
    # Apply filtering with user-specified parameters
    filtered_images, report = filter_images_by_similarity(
        source_dir=args.source_dir,
        similarity_threshold=args.similarity_threshold,
        min_group_size=args.min_group_size,
        selection_method=args.selection_method,
        output_filtered_dir=str(output_dir) if not args.preview_only else None
    )
    
    print("\n📊 Filtering Summary:")
    print(f"   Total images processed: {report['total_images']}")
    print(f"   Images after filtering: {report['filtered_images']}")
    print(f"   Reduction: {report['reduction_percentage']:.1f}%")
    print(f"   Groups found: {report['groups_found']}")
    print(f"   Similarity threshold: {report['similarity_threshold']}")
    print(f"   Selection method: {report['selection_method']}")
    
    # Show group details
    print(f"\n📋 Group Details:")
    for i, group_info in enumerate(report['groups'], 1):
        print(f"   Group {i}: {group_info['size']} images")
    
    if not args.preview_only:
        print(f"\n📁 Filtered images have been saved to: {output_dir}")
        print("📄 Detailed filtering report saved to:", source_path / "filtering_report.json")
    else:
        print(f"\n🔍 Preview mode - no files were copied")
        print("📄 Detailed filtering report saved to:", source_path / "filtering_report.json")
    
    # Provide recommendations based on results
    print(f"\n💡 Recommendations:")
    if report['reduction_percentage'] > 95:
        print("   ⚠️  Very high reduction detected! Consider:")
        print("      - Increasing similarity_threshold to 0.95-0.98")
        print("      - Increasing min_group_size to 5-10")
    elif report['reduction_percentage'] > 80:
        print("   ⚠️  High reduction detected! Consider:")
        print("      - Increasing similarity_threshold to 0.90-0.95")
        print("      - Increasing min_group_size to 3-5")
    elif report['reduction_percentage'] < 20:
        print("   ℹ️  Low reduction detected! Consider:")
        print("      - Decreasing similarity_threshold to 0.85-0.90")
        print("      - Decreasing min_group_size to 2-3")
    else:
        print("   ✅ Good reduction achieved!")

if __name__ == "__main__":
    main() 