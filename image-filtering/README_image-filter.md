# Image Filter & Report Tool

A Python utility for analyzing images in bulk with selectable filters and generating detailed CSV reports. This tool is designed for FNA (Fine Needle Aspiration) image analysis workflows, providing focus and brightness filtering capabilities.

## Features

- **Focus/Blur Detection**: Uses Laplacian variance to measure image sharpness
- **Brightness Analysis**: Evaluates mean brightness and pixel saturation levels
- **Combined Filtering**: Option to require both focus and brightness criteria
- **Parallel Processing**: Multi-threaded analysis for improved performance
- **Detailed Reporting**: CSV output with per-file decisions and metrics
- **Flexible Input**: Supports multiple image formats (JPG, PNG, TIFF, BMP, WebP)

## Installation

Install the required dependencies:

```bash
pip install opencv-python numpy
```

## Quick Start

### 1. Focus Filter Only

Filter images based on sharpness/focus:

```bash
python image_filter.py \
  --input-dir /path/to/images \
  --output-csv /path/to/report_focus.csv \
  --algorithm focus \
  --focus-threshold 120
```

### 2. Brightness Filter Only

Filter images based on brightness and saturation:

```bash
python image_filter.py \
  --input-dir /path/to/images \
  --output-csv /path/to/report_brightness.csv \
  --algorithm brightness \
  --brightness-min 10 --brightness-max 245 \
  --dark-threshold 5 --bright-threshold 250 \
  --saturation-pct-limit 2.0
```

### 3. Combined Filtering (Focus + Brightness)

Require both focus and brightness criteria to pass:

```bash
python image_filter.py \
  --input-dir /path/to/images \
  --output-csv /path/to/report_all.csv \
  --algorithm all \
  --focus-threshold 120 \
  --brightness-min 10 --brightness-max 245 \
  --dark-threshold 5 --bright-threshold 250 \
  --saturation-pct-limit 2.0
```

### 4. Scan Type Analysis

Analyze scan type groupings from algorithm_dev.json metadata:

```bash
python image_filter.py \
  --input-dir /path/to/study \
  --output-csv /path/to/scantype_report.csv \
  --algorithm scantypes
```

**Note**: The scantypes algorithm requires a study directory with the following structure:
```
study_directory/
├── images/
│   ├── algo_dev_imageset/
│   │   └── algorithm_dev.json    # Image metadata
│   └── raw_imageset/             # Actual image files
│       ├── image1.png
│       ├── image2.png
│       └── ...
```

## Command Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `--input-dir` | Directory containing images to analyze |
| `--output-csv` | Path where the CSV report will be written |
| `--algorithm` | Filtering algorithm: `focus`, `brightness`, `all`, or `scantypes` |

### Focus Filter Options

| Option | Default | Description |
|--------|---------|-------------|
| `--focus-threshold` | 120.0 | Minimum Laplacian variance for sharp images |

### Brightness Filter Options

| Option | Default | Description |
|--------|---------|-------------|
| `--brightness-min` | 10.0 | Minimum mean brightness (0-255) |
| `--brightness-max` | 245.0 | Maximum mean brightness (0-255) |
| `--dark-threshold` | 5 | Pixel value threshold for "dark" pixels |
| `--bright-threshold` | 250 | Pixel value threshold for "bright" pixels |
| `--saturation-pct-limit` | 2.0 | Maximum percentage of saturated pixels allowed |

### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-side` | 512 | Resize images so longest side ≤ this value (for speed) |
| `--workers` | 0 | Number of parallel workers (0 = CPU count - 1) |
| `--recursive` | False | Search subdirectories recursively |

## Output Format

### Standard Algorithms (focus, brightness, all)

The CSV report contains the following columns:

- **filename**: Name of the image file
- **include**: Boolean indicating if the image passed all filters
- **focus_variance**: Laplacian variance score (higher = sharper)
- **mean_brightness**: Average pixel brightness (0-255)
- **dark_pct**: Percentage of dark pixels
- **bright_pct**: Percentage of bright pixels
- **reason**: Explanation of why the image was included or excluded

### Scan Type Analysis (scantypes)

The CSV report contains the following columns:

- **FILENAME**: Relative path to the image file
- **IMAGE_DIR**: Study directory name (contains instrument serial, timestamp, run type, and study UUID)
- **ILLUMINATION_MODE**: Illumination method (`bright_field` or `fluorescent`)
- **LED_COLOR**: LED color used (`red`, `green`, `blue`, `uv`, `violet`)
- **Z_OFFSET_MODE**: Z-axis offset mode (`nominal`, `off`, `large_object`)
- **EXPOSURE_MULTIPLIER**: Exposure multiplier setting (numeric value)
- **IMAGE_SIZE**: File size in bytes
- **COMMENTS**: Additional notes (empty if file found, "file_missing" if not found)

This format allows users to:
1. Load the CSV into Excel or other spreadsheet tools
2. Filter by the four key fields to analyze different scan type combinations
3. Sum the IMAGE_SIZE column to get aggregate file sizes for each grouping

## Algorithm Details

### Focus Detection
- Uses the **Laplacian variance** method to measure image sharpness
- Higher variance values indicate sharper images
- Images with variance ≥ threshold are considered in focus

### Brightness Analysis
- Calculates **mean brightness** across all pixels
- Counts pixels below `dark_threshold` and above `bright_threshold`
- Images must have:
  - Mean brightness within `[brightness_min, brightness_max]`
  - Saturated pixels (dark + bright) ≤ `saturation_pct_limit`

### Combined Filtering
- When using `--algorithm all`, images must pass **both** focus and brightness criteria
- Useful for ensuring images meet quality standards for analysis

## Performance Tips

- Use `--max-side` to balance speed vs. accuracy (smaller values = faster processing)
- Adjust `--workers` based on your system's CPU cores
- Use `--recursive` only when needed to avoid processing unnecessary subdirectories

## Algorithm Development JSON Structure

The `scantypes` algorithm requires an `algorithm_dev.json` file that contains metadata about all images in the study. This file follows a specific structure:

### File Location
```
study_directory/
└── images/
    └── algo_dev_imageset/
        └── algorithm_dev.json
```

### Key Structure Elements

The JSON file contains:

1. **Study Metadata**: Patient information, instrument details, run parameters
2. **Clinical Signs**: Clinical information about the patient and sample
3. **IMAGES Array**: The main data structure containing image metadata

### Image Metadata Fields

Each image in the `IMAGES` array contains these key fields for scan type analysis:

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `FILENAME` | string | Relative path to image file | `"images/raw_imageset/image.png"` |
| `ILLUMINATION_MODE` | string | Illumination method | `"bright_field"`, `"fluorescent"` |
| `LED_COLOR` | string | LED color used | `"red"`, `"green"`, `"blue"`, `"uv"`, `"violet"` |
| `Z_OFFSET_MODE` | string | Z-axis offset mode | `"nominal"`, `"off"`, `"large_object"` |
| `EXPOSURE_MULTIPLIER` | number | Exposure multiplier setting | `1.0`, `2.0`, `5.0`, `10.0` |

### Additional Image Fields

Each image also contains technical metadata:

- `ACQUISITION_DATETIME`: ISO 8601 timestamp
- `IMAGE_WIDTH`, `IMAGE_HEIGHT`: Image dimensions in pixels
- `IMAGE_BITDEPTH`: Bit depth of the image
- `IMAGE_ID`: Unique identifier for the image
- `SCAN_TYPE`: Type of scan performed
- `CHANNEL`: Image channel identifier
- `GAIN`: Gain setting used
- `SAMPLE_CHAMBER`: Sample chamber identifier
- `FIELD_XI`, `FIELD_YI`: Field coordinates
- `X_POSITION`, `Y_POSITION`, `Z_OFFSET`: Physical positions in micrometers

### JSON Schema

A complete JSON schema is available in `algorithm_dev_schema.json` for validation and reference.

## Notes

- The tool processes images in grayscale for efficiency
- All metrics are calculated on resized images (if `--max-side` is used)
- Processing time and file counts are displayed at completion
- Supported image formats: JPG, JPEG, PNG, TIFF, TIFF, BMP, WebP
- The scantypes algorithm automatically handles missing image files by reporting "file_missing" in the COMMENTS column
