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

## Command Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `--input-dir` | Directory containing images to analyze |
| `--output-csv` | Path where the CSV report will be written |
| `--algorithm` | Filtering algorithm: `focus`, `brightness`, or `all` |

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

The CSV report contains the following columns:

- **filename**: Name of the image file
- **include**: Boolean indicating if the image passed all filters
- **focus_variance**: Laplacian variance score (higher = sharper)
- **mean_brightness**: Average pixel brightness (0-255)
- **dark_pct**: Percentage of dark pixels
- **bright_pct**: Percentage of bright pixels
- **reason**: Explanation of why the image was included or excluded

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

## Notes

- The tool processes images in grayscale for efficiency
- All metrics are calculated on resized images (if `--max-side` is used)
- Processing time and file counts are displayed at completion
- Supported image formats: JPG, JPEG, PNG, TIFF, TIFF, BMP, WebP
