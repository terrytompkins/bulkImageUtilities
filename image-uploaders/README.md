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
- **Metadata filtering**: Filter runs by JSON metadata fields using dot notation
- **Additional metadata fields**: Include any metadata fields in CSV output for analysis
- Calculates total size for each matching image collection
- Downloads metadata files for client-side filtering and analysis
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

# Query all devices (no serial number filter) for a date range
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --run-type run_fna \
    --image-sub-folder images/raw_imageset/ --output results.csv

# Query multiple devices with a specific run UUID
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-19 \
    --serial-numbers IVDX009978,IVDX009979 --run-type run_fna \
    --run-uuid E27A704B-DAB6-4282-BF6B-2EBF7079DEA6 \
    --image-sub-folder images/raw_imageset/ --output results.csv

# Query with metadata filtering (filter by firmware version and species)
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --serial-numbers IVDX009978 --run-type run_fna \
    --image-sub-folder images/raw_imageset/ \
    --metadata-file metadata/fna_results.json \
    --query-field run_metadata.INST_SW_VERSION=1.14.2 \
    --query-field run_metadata.SPECIES=Canine \
    --output-dir ./metadata-downloads --output results.csv

# Download metadata files without filtering (for manual inspection)
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --serial-numbers IVDX009978 --run-type run_fna \
    --image-sub-folder images/raw_imageset/ \
    --metadata-file metadata/fna_results.json \
    --download-metadata --output-dir ./metadata-downloads

# Include additional metadata fields in CSV output
python query_payload_size.py --bucket my-bucket --prefix invue_fna \
    --start-date 2022-11-19 --end-date 2022-11-20 \
    --run-type run_fna \
    --image-sub-folder images/raw_imageset/ \
    --metadata-file metadata/fna_results.json \
    --add-report-fields run_metadata.PATIENT_ID,run_metadata.SPECIES,run_metadata.INST_SW_VERSION \
    --output-dir ./metadata-downloads --output results.csv

# Using a specific AWS profile
AWS_PROFILE=my-profile python query_payload_size.py --bucket my-bucket ...
```

**Required parameters:**
- `--bucket`: S3 bucket name
- `--prefix`: Prefix path below bucket (e.g., "invue_fna")
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--run-type`: Run type (e.g., "run_fna")
- `--image-sub-folder`: Image sub-folder path below run UUID

**Optional parameters:**
- `--serial-numbers`: Comma-separated list of IoT device serial numbers (e.g., "IVDX009978,IVDX009979"). If not specified, searches all serial numbers matching other criteria.

- `--run-uuid`: Filter by specific run UUID
- `--output`: Output CSV file path (default: query_payload_size.csv)
- `--metadata-file`: Path to metadata JSON file relative to run folder (e.g., `metadata/fna_results.json`)
- `--query-field`: Query field in format `key.path=value` (can be specified multiple times). Example: `--query-field run_metadata.INST_SW_VERSION=1.14.2`
- `--add-report-fields`: Comma-separated list of dot-notation key paths from metadata JSON to include in CSV output. Example: `run_metadata.PATIENT_ID,run_metadata.SPECIES`
- `--output-dir`: Directory to download metadata files (required when using `--query-field`, `--add-report-fields`, or `--download-metadata`)
- `--download-metadata`: Download metadata files even without filtering (requires `--output-dir`)

**Output CSV columns:**
- Date/Time: Formatted timestamp from the run
- IoT Device Serial Number: Device serial number
- Run UUID: Unique identifier for the run
- Run Type: Type of run operation
- Size (bytes): Total size of the image collection
- Metadata Matched: "Yes" or "No" (only when `--query-field` is used). Shows whether the run matched the client-side metadata criteria. All runs matching cloud-side criteria are included in the output, regardless of metadata match status.
- Query field columns: One column per `--query-field` showing the actual value from metadata (column name is the last part of the dot-notation path, e.g., `INST_SW_VERSION`)
- Additional report field columns: One column per `--add-report-fields` showing the value from metadata (column name is the last part of the dot-notation path, e.g., `PATIENT_ID`). Missing fields show empty strings.

## Metadata Filtering

The `query_payload_size.py` script supports filtering image runs based on metadata stored in JSON files within each run folder. This is useful for isolating runs based on firmware versions, species, or other metadata attributes.

### How It Works

1. **Specify the metadata file location**: Use `--metadata-file` to point to the JSON file relative to each run folder (e.g., `metadata/fna_results.json`)

2. **Define query fields**: Use one or more `--query-field` parameters to specify which metadata fields to match:
   ```bash
   --query-field run_metadata.INST_SW_VERSION=1.14.2
   --query-field run_metadata.SPECIES=Canine
   ```

3. **Provide output directory**: Use `--output-dir` to specify where downloaded metadata files should be stored

4. **Filtering behavior**:
   - All runs matching cloud-side criteria (date, serial numbers, run type, etc.) are included in the CSV output
   - The "Metadata Matched" column indicates which runs also matched the client-side metadata criteria
   - Runs matching ALL specified query fields will have "Yes" in the "Metadata Matched" column
   - Comparisons are case-insensitive (e.g., `Canine` matches `CANINE` or `canine`)
   - Values are normalized to strings for comparison
   - Missing or invalid metadata files result in "No" in the "Metadata Matched" column

### Query Field Syntax

- Use dot notation to access nested JSON keys: `parent.child.grandchild`
- Format: `key.path=value` (no spaces around the `=`)
- Example: `run_metadata.CLINICAL_SIGNS.RUN_SAMPLE_TYPE=FNA`

### Including Additional Metadata Fields in Reports

You can include additional metadata fields in the CSV output without filtering. This is useful for analysis and reporting:

```bash
python query_payload_size.py ... --metadata-file metadata/fna_results.json \
    --add-report-fields run_metadata.PATIENT_ID,run_metadata.SPECIES,run_metadata.INST_SW_VERSION \
    --output-dir ./metadata-downloads
```

The specified fields will appear as additional columns in the CSV output. Column names use the last part of the dot-notation path (e.g., `run_metadata.PATIENT_ID` becomes `PATIENT_ID`). If a field doesn't exist in a metadata file, the cell will be empty.

### Downloading Metadata Without Filtering

To download metadata files for manual inspection without filtering:
```bash
python query_payload_size.py ... --metadata-file metadata/fna_results.json \
    --download-metadata --output-dir ./metadata-downloads
```

Downloaded files are named by run UUID (e.g., `817C998F-FAB3-43D1-8A66-5C06F55F2198.json`) for easy traceability.

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
