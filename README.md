# Bulk Image Utilities

A collection of Python utilities for bulk image processing and cloud storage operations. This repository contains two main scripts for converting PNG images to JPEG XL format and benchmarking S3 upload methods.

## Scripts Overview

### 1. PNG to JPEG XL Converter (`convert_and_verify_png_to_jxl.py`)

A high-performance script that converts PNG images to JPEG XL format with parallel processing and integrity verification.

**Features:**
- Parallel processing of multiple PNG files
- Lossless conversion verification
- Detailed CSV reporting with timing and compression metrics
- SHA-256 file integrity checks
- Comprehensive error handling

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
- `Pillow` (PIL)
- `numpy`

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

3. Install system dependencies:

**macOS (using Homebrew):**
```bash
# Install JPEG XL tools
brew install jpeg-xl

# Install s5cmd (optional)
brew install s5cmd
```

**Ubuntu/Debian:**
```bash
# Install JPEG XL tools
sudo apt-get install libjxl-tools

# Install s5cmd (optional)
curl -L https://github.com/peak/s5cmd/releases/latest/download/s5cmd_$(uname -s)_$(uname -m).tar.gz | tar -xz
sudo mv s5cmd /usr/local/bin/
```

## Usage

### PNG to JPEG XL Conversion

Convert PNG files in a directory to JPEG XL format with verification:

```bash
python convert_and_verify_png_to_jxl.py <source_dir> <dest_dir> <report_csv>
```

**Example:**
```bash
python convert_and_verify_png_to_jxl.py ./png_images ./jxl_output conversion_report.csv
```

**Output:**
- Converted JXL files in the destination directory
- CSV report with conversion metrics including:
  - Original and compressed file sizes
  - Compression ratios
  - Conversion and verification timing
  - Success/failure status
  - Error messages

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

## Performance Considerations

### PNG to JXL Conversion
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
The PNG to JXL converter generates a CSV with the following columns:
- `filename`: Original PNG filename
- `png_size_bytes`: Original file size
- `jxl_size_bytes`: Compressed file size
- `compression_ratio`: Compression ratio (JXL size / PNG size)
- `conversion_success`: Boolean success flag
- `verification_success`: Boolean verification flag
- `conversion_start/end`: ISO timestamp of conversion
- `verification_start/end`: ISO timestamp of verification
- `error`: Error message if any

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