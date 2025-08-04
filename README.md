# Bulk Image Utilities

A collection of Python utilities for bulk image processing and cloud storage operations. This repository contains two main scripts for converting PNG images to other formats and benchmarking S3 upload methods.

## Scripts Overview

### 1. Image Converter (`image_converter.py`)

A high-performance script for bulk converting PNG images to:
- **Losslessly compressed PNG** (default)
- **AVIF (lossless)**
- **JPEG XL (JXL)**
- **WebP (lossless)**

**Features:**
- Parallel processing of multiple PNG files
- Lossless PNG compression (with configurable compression level)
- Lossless AVIF conversion (using Pillow with libavif)
- Lossless WebP conversion (using Pillow)
- Optional conversion to JPEG XL (JXL) with verification
- **NEW: Similarity-based image filtering** to reduce dataset size
- Detailed CSV reporting with timing and compression metrics
- SHA-256 file integrity checks (for JXL verification)
- Comprehensive error handling

#### Supported Formats
- `png`: Re-save PNGs with lossless compression (compression level 0-9)
- `avif`: Convert PNGs to AVIF (lossless, requires Pillow with AVIF support)
- `webp`: Convert PNGs to WebP (lossless, requires Pillow with WebP support)
- `jxl`: Convert PNGs to JPEG XL (requires `cjxl`/`djxl` tools)

#### Image Filtering Feature
The image converter now includes intelligent filtering capabilities inspired by HDR compression techniques:

- **Similarity-based grouping**: Groups similar images using computer vision features
- **Representative selection**: Chooses the best representative from each group
- **Dataset reduction**: Can reduce image count by 50-80% while maintaining quality
- **Multiple selection methods**: Choose by quality, file size, or order
- **Detailed reporting**: Generates JSON reports with filtering statistics

**Filtering Methods:**
- `best_quality`: Select image with highest average intensity (recommended)
- `largest`: Select largest file by size
- `smallest`: Select smallest file by size  
- `first`: Select first image in group

### 2. S3 Upload Benchmarker (`s3_uploader_tester.py`)

A benchmarking tool that compares different methods for uploading large batches of images to Amazon S3.

**Supported Upload Methods:**
- `awscli_sync` - AWS CLI sync command
- `s5cmd_cp` - High-performance s5cmd utility
- `boto3_transfer` - Boto3 TransferManager with concurrent multipart uploads
- `curl_presigned_single` - Individual presigned URL uploads with curl
- `curl_presigned_persist` - Presigned URL uploads with persistent curl connections

## Requirements

### System Dependencies

- **Python 3.9+**
- **Pillow** (with AVIF/WebP support for AVIF/WebP conversion; see below)
- **libavif** (for AVIF conversion; required by Pillow for AVIF support)
- **JPEG XL tools** (`cjxl` and `djxl`) - Required for PNG to JXL conversion
- **OpenCV** - For image processing and feature extraction (filtering)
- **scikit-learn** - For clustering and similarity calculations (filtering)
- **AWS CLI v2** - For `awscli_sync` method
- **s5cmd** - For `s5cmd_cp` method (optional)
- **curl >=7.65** - For presigned URL uploads (optional)

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The requirements include:
- `boto3>=1.26.0`
- `botocore>=1.29.0`
- `Pillow` (PIL, with AVIF support)
- `numpy`
- `opencv-python>=4.8.0` (for image processing)
- `scikit-learn>=1.3.0` (for clustering)

#### AVIF/WebP Support in Pillow
- Pillow must be built with AVIF support, which requires `libavif` installed on your system.
- Pillow must be built with WebP support (usually included by default).
- **Note**: Pillow 12.0.0+ is recommended for best AVIF/WebP support, but may not be available for Python 3.13+ yet. Use 11.3.0+ as fallback.
- To check if your Pillow supports AVIF/WebP:
  ```python
  from PIL import features
  print(f"AVIF support: {features.check('avif')}")
  print(f"WebP support: {features.check('webp')}")
  ```
- If AVIF support is `False`, install `libavif` and reinstall Pillow.

**macOS (using Homebrew):**
```bash
brew install libavif jpeg-xl
pip install --upgrade --force-reinstall pillow
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libavif-dev libjxl-tools
pip install --upgrade --force-reinstall pillow
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd bulkImageUtilities
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies as above.

## Usage

### Image Conversion and Compression

Bulk convert PNG files in a directory to compressed PNG, AVIF, WebP, or JPEG XL format:

```bash
python image_converter.py <source_dir> <dest_dir> <report_csv> [--format png|avif|webp|jxl] [--compression-level 0-9] [--quality N]
```

#### Examples

**1. Convert uncompressed PNGs to maximally compressed PNGs (default):**
```bash
python image_converter.py ./png_images ./compressed_pngs conversion_report.csv
```

**2. Convert PNGs to lossless AVIF:**
```bash
python image_converter.py ./png_images ./avif_output conversion_report.csv --format avif
```

**3. Convert PNGs to WebP:**
```bash
python image_converter.py ./png_images ./webp_output conversion_report.csv --format webp
```

**4. Convert PNGs to JPEG XL:**
```bash
python image_converter.py ./png_images ./jxl_output conversion_report.csv --format jxl
```

**5. Specify a custom PNG compression level (0 = none, 9 = max, default: 9):**
```bash
python image_converter.py ./png_images ./compressed_pngs conversion_report.csv --compression-level 6
```

#### Image Filtering Examples

**6. Apply similarity-based filtering before conversion:**
```bash
python image_converter.py ./png_images ./compressed_pngs conversion_report.csv --filter-similar
```

**7. Customize filtering parameters:**
```bash
python image_converter.py ./png_images ./compressed_pngs conversion_report.csv \
  --filter-similar \
  --similarity-threshold 0.90 \
  --min-group-size 3 \
  --selection-method best_quality
```

**8. Use filtering with AVIF conversion:**
```bash
python image_converter.py ./png_images ./avif_output conversion_report.csv \
  --format avif \
  --filter-similar \
  --similarity-threshold 0.85
```

#### Filtering Parameters

- `--filter-similar`: Enable similarity-based filtering
- `--similarity-threshold`: Minimum similarity to group images (0-1, default: 0.85)
- `--min-group-size`: Minimum images to form a group (default: 2)
- `--selection-method`: How to select representatives (`best_quality`, `largest`, `smallest`, `first`)

### Standalone Filtering

Use the example script for standalone filtering:

```bash
python filter_example.py <source_directory>
```

This will:
1. Analyze all images in the directory for similarity
2. Group similar images together
3. Select representative images from each group
4. Save filtered images to a `filtered` subdirectory
5. Generate a detailed filtering report

### S3 Upload Benchmarking

Benchmark different S3 upload methods:

```bash
python s3_uploader_tester.py --input-dir <input_directory> --bucket <s3_bucket> --method <upload_method> [--prefix <s3_prefix>]
```

**Examples:**

```bash
# Test AWS CLI sync method
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method awscli_sync --prefix test/

# Test s5cmd method
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method s5cmd_cp --prefix benchmark/

# Test Boto3 transfer manager
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method boto3_transfer --prefix uploads/

# Test presigned URL uploads
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method curl_presigned_persist --prefix presigned/
```

## Configuration

### AWS Credentials

For S3 operations, ensure you have AWS credentials configured:

```bash
# Using AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### JPEG XL Configuration

The PNG to JXL converter uses lossless compression (`-d 0` flag). To modify compression settings, edit the `cjxl` command in the script.

### PNG Compression Level

- The PNG converter uses Pillow's `compress_level` (0-9).
- `0`: No compression (large files, fastest)
- `9`: Maximum compression (smallest files, slowest, default)
- Use `--compression-level` to control this.

### AVIF/WebP Quality

- By default, AVIF and WebP conversion is lossless (`lossless=True`).
- The `--quality` parameter is reserved for future use (for lossy AVIF/WebP).

### Image Filtering Configuration

The filtering system uses computer vision techniques to identify similar images:

- **Feature extraction**: Histogram analysis, intensity statistics, and aspect ratios
- **Similarity calculation**: Cosine similarity on normalized histograms
- **Clustering**: DBSCAN algorithm for grouping similar images
- **Representative selection**: Multiple methods for choosing the best image from each group

**Recommended settings:**
- `similarity_threshold=0.85`: Good balance between reduction and quality
- `min_group_size=2`: Minimum 2 images to form a group
- `selection_method=best_quality`: Selects highest quality image from each group

## Performance Considerations

### Image Conversion
- Uses parallel processing with `ProcessPoolExecutor`
- Number of workers defaults to CPU count
- Each image is processed independently for optimal performance

### Image Filtering
- Feature extraction is computationally intensive but parallelized
- Similarity matrix calculation scales with O(n²) for n images
- Clustering performance depends on image count and similarity distribution
- Typical processing time: 1-5 seconds per image for feature extraction

### S3 Upload Benchmarking
- **awscli_sync**: Good for simple sync operations
- **s5cmd_cp**: Typically fastest for large file batches
- **boto3_transfer**: Good for programmatic control with concurrent uploads
- **curl_presigned_***: Useful for testing direct upload performance

## Output Formats

### Conversion Report CSV
The image converter generates a CSV with the following columns:
- `filename`: Original PNG filename
- `png_size_bytes`: Original file size
- `target_size_bytes`: Compressed PNG, AVIF, or JXL file size
- `compression_ratio`: Compression ratio (target/original)
- `conversion_success`: Boolean success flag
- `verification_success`: Boolean verification flag (always true for PNG/AVIF, pixel-checked for JXL)
- `conversion_start/end`: ISO timestamp of conversion
- `verification_start/end`: ISO timestamp of verification (JXL only)
- `error`: Error message if any
- `target_format`: Output format (`png`, `avif`, `webp`, or `jxl`)
- `compression_level`: PNG compression level (0-9) or blank for AVIF/WebP/JXL
- `quality`: AVIF/WebP quality (not used for lossless, reserved for future use)

### Filtering Report JSON
When filtering is enabled, a detailed JSON report is generated:
- `total_images`: Number of images processed
- `filtered_images`: Number of images after filtering
- `reduction_percentage`: Percentage reduction achieved
- `groups_found`: Number of similarity groups identified
- `similarity_threshold`: Threshold used for grouping
- `selection_method`: Method used for representative selection
- `groups`: Detailed information about each group

### Benchmark Output
The S3 upload benchmarker provides:
- Real-time progress updates
- Total upload time
- Summary statistics (file count, total size, method, duration)

## Error Handling

Both scripts include comprehensive error handling:
- Graceful handling of missing dependencies
- Detailed error messages for failed operations
- Cleanup of temporary files
- Non-zero exit codes for script failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the error messages for common solutions
2. Verify all dependencies are installed
3. Ensure AWS credentials are properly configured
4. Open an issue with detailed error information 