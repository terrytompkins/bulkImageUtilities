# Bulk Image Utilities

A collection of Python utilities for bulk image processing and cloud storage operations. This repository contains two main scripts for converting PNG images to other formats and benchmarking S3 upload methods.

## Scripts Overview

### 1. Image Converter (`image_converter.py`)

A high-performance script for bulk converting PNG images to:
- **Losslessly compressed PNG** (default)
- **AVIF (lossless)**
- **JPEG XL (JXL)**

**Features:**
- Parallel processing of multiple PNG files
- Lossless PNG compression (with configurable compression level)
- Lossless AVIF conversion (using Pillow with libavif)
- Optional conversion to JPEG XL (JXL) with verification
- Detailed CSV reporting with timing and compression metrics
- SHA-256 file integrity checks (for JXL verification)
- Comprehensive error handling

#### Supported Formats
- `png`: Re-save PNGs with lossless compression (compression level 0-9)
- `avif`: Convert PNGs to AVIF (lossless, requires Pillow with AVIF support)
- `jxl`: Convert PNGs to JPEG XL (requires `cjxl`/`djxl` tools)

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
- **Pillow** (with AVIF support for AVIF conversion; see below)
- **libavif** (for AVIF conversion; required by Pillow for AVIF support)
- **JPEG XL tools** (`cjxl` and `djxl`) - Required for PNG to JXL conversion
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

#### AVIF Support in Pillow
- Pillow must be built with AVIF support, which requires `libavif` installed on your system.
- To check if your Pillow supports AVIF:
  ```python
  from PIL import features
  print(features.check('avif'))
  ```
- If `False`, install `libavif` and reinstall Pillow.

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

Bulk convert PNG files in a directory to compressed PNG, AVIF, or JPEG XL format:

```bash
python image_converter.py <source_dir> <dest_dir> <report_csv> [--format png|avif|jxl] [--compression-level 0-9] [--quality N]
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

**3. Convert PNGs to JPEG XL:**
```bash
python image_converter.py ./png_images ./jxl_output conversion_report.csv --format jxl
```

**4. Specify a custom PNG compression level (0 = none, 9 = max, default: 9):**
```bash
python image_converter.py ./png_images ./compressed_pngs conversion_report.csv --compression-level 6
```

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

### AVIF Quality

- By default, AVIF conversion is lossless (`lossless=True`).
- The `--quality` parameter is reserved for future use (for lossy AVIF).

## Performance Considerations

### Image Conversion
- Uses parallel processing with `ProcessPoolExecutor`
- Number of workers defaults to CPU count
- Each image is processed independently for optimal performance

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
- `target_format`: Output format (`png`, `avif`, or `jxl`)
- `compression_level`: PNG compression level (0-9) or blank for AVIF/JXL
- `quality`: AVIF quality (not used for lossless, reserved for future use)

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