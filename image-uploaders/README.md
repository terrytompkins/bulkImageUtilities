# Image Upload Tools

This directory contains tools for uploading images to cloud storage and calculating upload times.

## Contents

### Scripts
- **`s3_uploader_tester.py`** - Benchmark different methods of uploading images to S3
- **`upload_time_app.py`** - Streamlit app for calculating cloud upload times
- **`query_payload_size.py`** - Query S3 to calculate payload sizes for matching image collections

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

### 4. Query S3 Payload Sizes
```bash
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --serial-numbers IVDX009978 --run-type run_fna \
    --image-sub-folder images/raw_imageset/ --output results.csv
```

## External Dependencies

Install these tools separately:

**For S3 Upload Tester:**
- AWS CLI v2 (configured with credentials)
- s5cmd (for s5cmd_cp method)
- curl >=7.65 (for presigned URL uploads)

**For Query Payload Size:**
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Appropriate S3 read permissions for the target bucket

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

### Query Payload Size
- Queries S3 objects matching structured path patterns
- Supports date range queries with automatic partition generation
- Filters by IoT device serial numbers, run types, and optional run UUIDs
- Calculates total size for each matching image collection
- Outputs results to CSV with detailed run information
- Handles large date ranges efficiently with pagination

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

### Query S3 Payload Sizes
```bash
# Query a single device for a date range
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --serial-numbers IVDX009978 --run-type run_fna \
    --image-sub-folder images/raw_imageset/

# Query multiple devices with a specific run UUID
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-19 \
    --serial-numbers IVDX009978,IVDX009979 --run-type run_fna \
    --run-uuid E27A704B-DAB6-4282-BF6B-2EBF7079DEA6 \
    --image-sub-folder images/raw_imageset/ --output results.csv

# Using a specific AWS profile
AWS_PROFILE=my-profile python query_payload_size.py --bucket my-bucket ...
```

**Required parameters:**
- `--bucket`: S3 bucket name
- `--prefix`: Prefix path below bucket (e.g., "invue_fna")
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--serial-numbers`: Comma-separated list of IoT device serial numbers
- `--run-type`: Run type (e.g., "run_fna")
- `--image-sub-folder`: Image sub-folder path below run UUID

**Optional parameters:**
- `--run-uuid`: Filter by specific run UUID
- `--output`: Output CSV file path (default: query_payload_size.csv)

**Output CSV columns:**
- Date/Time: Formatted timestamp from the run
- IoT Device Serial Number: Device serial number
- Run UUID: Unique identifier for the run
- Run Type: Type of run operation
- Size (bytes): Total size of the image collection

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
- Use **query_payload_size** to analyze existing S3 data and plan migrations or archival
