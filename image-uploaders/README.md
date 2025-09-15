# Image Upload Tools

This directory contains tools for uploading images to cloud storage and calculating upload times.

## Contents

### Scripts
- **`s3_uploader_tester.py`** - Benchmark different methods of uploading images to S3
- **`upload_time_app.py`** - Streamlit app for calculating cloud upload times

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. S3 Upload Benchmarking
```bash
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method awscli_sync --prefix test/
```

**Supported upload methods:**
- `awscli_sync` - Uses `aws s3 sync`
- `s5cmd_cp` - Uses `s5cmd cp` 
- `boto3_transfer` - Uses Boto3 TransferManager (concurrent multipart)
- `curl_presigned_single` - Generates presigned URLs and uploads with curl
- `curl_presigned_persist` - Same as above but re-uses a single curl process

### 3. Upload Time Calculator
```bash
streamlit run upload_time_app.py
```

## External Dependencies

Install these tools separately:

**For S3 Upload Tester:**
- AWS CLI v2 (configured with credentials)
- s5cmd (for s5cmd_cp method)
- curl >=7.65 (for presigned URL uploads)

## Features

### S3 Upload Tester
- Times entire upload batches
- Logs CSV results with detailed metrics
- Supports multiple upload methods
- Handles large file collections efficiently

### Upload Time Calculator
- Interactive Streamlit interface
- Calculates upload times for different speeds
- Supports throttling scenarios (70%, 80%, 90% of link)
- Configurable file sizes and speed ranges
- Visual charts and data tables

## Usage Examples

### Benchmark S3 Upload Methods
```bash
# Test AWS CLI sync
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method awscli_sync

# Test s5cmd (often fastest)
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method s5cmd_cp

# Test Boto3 concurrent uploads
python s3_uploader_tester.py --input-dir ./images --bucket my-bucket --method boto3_transfer
```

### Calculate Upload Times
```bash
# Launch the web interface
streamlit run upload_time_app.py
```

Then configure:
- File size (GB)
- Upload speed range (Mbps)
- Throttling levels
- View results in charts and tables

## Performance Tips

- **s5cmd** is often the fastest for large uploads
- **Boto3 TransferManager** provides good Python integration
- **AWS CLI sync** is reliable and well-tested
- Use **presigned URLs** for distributed uploads

## Integration

These tools work well with the other utilities in this repository:
- Use **image-converters** to optimize images before upload
- Use **image-filtering** to reduce dataset size before upload
- Calculate upload times to plan large data transfers
