# Image Filter Viewer

A Streamlit web application for visually reviewing the results of image filtering operations. This companion tool works with the `image_filter.py` script to provide an interactive interface for analyzing filtered images.

## Features

### 🎛️ **Interactive Controls**
- **File Selection**: Browse for input directory with helper tools and CSV report file upload
- **Algorithm Selection**: Choose between focus, brightness, scantypes, or combined filtering algorithms
- **Advanced Filtering**: Filter by include status, filename search, focus score ranges, brightness criteria, and scan type metadata
- **Display Options**: Adjust thumbnails per row and images per page

### 🖼️ **Image Display**
- **Thumbnail Grid**: Paginated display with customizable grid layout and visual row separation
- **Modal Dialog**: Click "View" buttons to open full-size images in a modal popup dialog
- **Navigation**: Previous/Next buttons for easy image browsing within the dialog
- **Metadata Display**: Shows focus scores, brightness values, and filtering reasons

### 📊 **Statistics & Analysis**
- **Real-time Statistics**: Total images, filtered count, current page info, aggregate file sizes
- **Visual Indicators**: Clear status indicators for included/excluded images
- **Search & Filter**: Find specific images by filename, scan type metadata, and quality metrics

## Installation

Ensure you have the required dependencies:

```bash
pip install streamlit opencv-python numpy pandas pillow
```

## Usage

### 1. Generate Filter Report
First, run the image filter to generate a CSV report:

```bash
# For focus filtering
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./focus-report.csv \
  --algorithm focus \
  --focus-threshold 2.0

# For brightness filtering  
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./brightness-report.csv \
  --algorithm brightness \
  --brightness-min 12 --brightness-max 195 \
  --saturation-pct-limit 4.0

# For scantypes filtering
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./scantypes-report.csv \
  --algorithm scantypes

# For combined filtering (focus + brightness + scantypes)
python image_filter.py \
  --study-dir /path/to/study \
  --output-csv ./complete-analysis.csv \
  --algorithm all
```

### 2. Launch Image Viewer
Start the Streamlit application:

```bash
streamlit run image_viewer.py
```

### 3. Configure Viewer
1. **Select Input Directory**: Use the browse button (📂) for guidance, or paste the directory path
2. **Upload CSV Report**: Upload the CSV file from step 1
3. **Choose Algorithm**: Select from the available filtering algorithms:
   - `focus`: Focus/sharpness analysis only
   - `brightness`: Brightness and saturation analysis only
   - `focus+brightness`: Combined quality analysis
   - `scantypes`: Scan type metadata analysis only
   - `focus+scantypes`: Focus analysis with scan type metadata
   - `brightness+scantypes`: Brightness analysis with scan type metadata
   - `all`: Complete analysis (focus + brightness + scantypes)
4. **Apply Filters**: Use sidebar controls to refine your view

## Interface Guide

### Left Sidebar Controls

| Control | Description |
|---------|-------------|
| **Input Directory** | Path to folder containing images (with browse helper) |
| **CSV Report File** | Upload the CSV report from image_filter.py |
| **Algorithm** | Select from 7 available algorithms (auto-detected from CSV) |
| **Search Filename** | Filter images by filename (partial match) |
| **Include Filter** | Show all, only included, or only excluded images |
| **Focus Score Range** | Min/max focus scores (only for focus algorithm) |
| **Brightness Filters** | Min/max brightness, dark/bright thresholds (only for brightness algorithm) |
| **Scan Type Filters** | Multi-select filters for illumination mode, LED color, Z offset mode, exposure multiplier (only for scantypes algorithm) |
| **Thumbnails per row** | Number of thumbnails per row (1-20) |
| **Images per page** | Number of images per page (10-200) |

### Main Display Area

- **Thumbnail Grid**: Clickable thumbnails with key metrics and visual row separation
- **Pagination**: Navigate through pages of results
- **View Buttons**: Click "View" buttons to open modal dialogs

### Modal Dialog Features

- **Full-size Image**: Large display of selected image (up to 1200px width)
- **Metadata Panel**: Filename, scores, include status, and filtering reason
- **Navigation**: Previous/Next buttons to browse through images within the dialog
- **Browser Zoom**: Use Ctrl/Cmd + Plus to zoom the entire dialog for detailed examination

## Tips for Effective Use

### 🔍 **Reviewing Filter Results**
1. **Start with excluded images**: Set "Include Filter" to "False" to review what was filtered out
2. **Use focus score ranges**: For focus filtering, try ranges like 0-2.0 to see excluded images
3. **Search by filename**: Use partial filename matches to find specific image types
4. **Adjust page size**: Use smaller page sizes (20-50) for detailed review

### 📊 **Analyzing Results**
1. **Check borderline cases**: Look at images near your threshold values
2. **Review metadata**: Check the "reason" column to understand why images were excluded
3. **Compare similar images**: Use filename search to compare similar image types
4. **Validate thresholds**: Adjust your filtering parameters based on visual review

### ⚡ **Performance Tips**
1. **Use appropriate page sizes**: 50-100 images per page for good performance
2. **Limit thumbnails per row**: 8-12 thumbnails per row for optimal viewing
3. **Filter first**: Use search and filters to reduce the dataset before browsing

## Troubleshooting

### Common Issues

**"No images to display"**
- Check that input directory path is correct
- Verify CSV file was uploaded successfully
- Ensure filter criteria aren't too restrictive

**"Error loading image"**
- Verify image files exist in the input directory
- Check file permissions
- Ensure supported image formats (JPG, PNG, TIFF, BMP, WebP)

**Slow performance**
- Reduce page size (fewer images per page)
- Use filename search to limit results
- Check that images aren't too large

### Supported Formats
- **Images**: JPG, JPEG, PNG, TIFF, TIFF, BMP, WebP
- **Reports**: CSV files from image_filter.py

## Integration with image_filter.py

This viewer is designed to work seamlessly with the `image_filter.py` script:

1. **Generate reports** using image_filter.py with your desired parameters
2. **Review results** using this viewer to validate filtering quality
3. **Adjust parameters** in image_filter.py based on visual feedback
4. **Iterate** until you achieve the desired filtering results

The viewer supports both focus and brightness filtering algorithms and displays all relevant metrics from the CSV reports.
