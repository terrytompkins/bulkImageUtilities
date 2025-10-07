# Image Filtering Tools

This directory contains tools for filtering and analyzing images in bulk, specifically designed for FNA (Fine Needle Aspiration) image analysis workflows.

## Contents

### Scripts
- **`image_filter.py`** - Command-line tool for filtering images based on focus, brightness, and scan type criteria
- **`image_viewer.py`** - Streamlit web application for visually reviewing filtering results
- **`st-dialog-test.py`** - Test script for Streamlit dialog functionality

### Documentation
- **`README_image-filter.md`** - Comprehensive documentation for the image filtering script
- **`README_image-viewer.md`** - Complete guide for the image viewer application

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Use the Tools
1. **Filter Images**: Use `image_filter.py` to analyze and filter your image collection
2. **Review Results**: Use `image_viewer.py` to visually inspect the filtering results
3. **Iterate**: Adjust filtering parameters based on visual feedback

## Usage Examples

### Filter Images
```bash
# Focus filtering
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./focus-report.csv \
  --algorithm focus \
  --focus-threshold 2.0

# Brightness filtering
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./brightness-report.csv \
  --algorithm brightness \
  --brightness-min 12 --brightness-max 195 \
  --saturation-pct-limit 4.0

# Scan type filtering
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./scantypes-report.csv \
  --algorithm scantypes
```

### View Results
```bash
streamlit run image_viewer.py
```

## Dependencies

Install required packages:
```bash
pip install opencv-python numpy streamlit pandas pillow
```

## Integration

These tools work together seamlessly:
1. Generate filtering reports with `image_filter.py`
2. Review results visually with `image_viewer.py`
3. Adjust parameters and iterate until satisfied

See the individual README files for detailed documentation and advanced usage.
