# Bulk Image Utilities

A comprehensive collection of tools for processing, converting, filtering, and uploading images in bulk. This repository provides specialized utilities for image analysis workflows, particularly suited for FNA (Fine Needle Aspiration) image processing and general bulk image operations.

## 🗂️ Repository Structure

This repository is organized into specialized tool categories:

### 📁 [image-converters/](image-converters/)
**Image conversion, compression, and optimization tools**
- `image_converter.py` - Advanced image conversion with PNG optimization and archiving
- `hdr_compression.ipynb` - HDR compression analysis notebook
- `compression_example.py` - Example usage script for compression features
- `tests/` - Comprehensive test suite for validation and benchmarking
- Comprehensive documentation for compression tools and strategies

### 📁 [image-filtering/](image-filtering/)
**Image filtering and analysis tools**
- `image_filter.py` - Focus and brightness filtering for image quality assessment
- `image_viewer.py` - Streamlit web app for visually reviewing filtering results
- Interactive tools for image quality analysis and dataset curation

### 📁 [image-uploaders/](image-uploaders/)
**Cloud upload and transfer tools**
- `s3_uploader_tester.py` - Benchmark different S3 upload methods
- `upload_time_app.py` - Streamlit app for calculating upload times
- Performance optimization for large-scale data transfers

### 📁 [image-converters/tests/](image-converters/tests/)
**Test scripts and validation tools for image conversion**
- Comprehensive test suite for image conversion utilities
- Performance benchmarks and validation scripts
- Test output artifacts (excluded from version control)

## 🚀 Quick Start

### Choose Your Tool Category

**For Image Conversion & Compression:**
```bash
cd image-converters
pip install -r requirements.txt
python image_converter.py --help
```

**For Image Filtering & Analysis:**
```bash
cd image-filtering
pip install -r requirements.txt
python image_filter.py --help
streamlit run image_viewer.py
```

**For Cloud Upload & Transfer:**
```bash
cd image-uploaders
pip install -r requirements.txt
python s3_uploader_tester.py --help
streamlit run upload_time_app.py
```

## 🔧 Common Workflows

### 1. Image Processing Pipeline
```bash
# Step 1: Filter images for quality
cd image-filtering
python image_filter.py --input-dir /path/to/images --output-csv ./filter-report.csv --algorithm focus

# Step 2: Convert and optimize filtered images
cd ../image-converters
python image_converter.py /path/to/filtered/images ./optimized ./conversion-report.csv --output-format webp

# Step 3: Upload to cloud storage
cd ../image-uploaders
python s3_uploader_tester.py --input-dir ./optimized --bucket my-bucket --method s5cmd_cp
```

### 2. Bulk Dataset Preparation
```bash
# Convert and compress large image collections
cd image-converters
python image_converter.py /path/to/raw/images ./processed ./report.csv \
  --output-format webp --quality 85 \
  --create-archives --archive-format zstd \
  --filter-similar --similarity-threshold 0.95
```

### 3. Quality Assessment Workflow
```bash
# Analyze image quality and review results
cd image-filtering
python image_filter.py --input-dir /path/to/images --output-csv ./quality-report.csv --algorithm brightness
streamlit run image_viewer.py  # Visual review of results
```

## 📋 Requirements

Each tool category has its own requirements file:

- **image-converters**: Pillow, OpenCV, scikit-learn, Jupyter
- **image-filtering**: OpenCV, NumPy, Streamlit, Pandas, Pillow
- **image-uploaders**: Boto3, Streamlit, Matplotlib

External tools may be required (see individual README files for details):
- Compression tools: oxipng, zopflipng, pngcrush, zstd, 7z
- Cloud tools: AWS CLI, s5cmd, curl
- Image tools: avifenc (libavif-tools)

## 🎯 Key Features

### Image Conversion & Compression
- **Multi-format support**: JPG, PNG, TIFF, BMP, WebP, AVIF
- **Advanced PNG optimization**: oxipng, zopflipng, pngcrush
- **Intelligent archiving**: zstd, 7z, pigz with configurable sizes
- **Similarity filtering**: Remove duplicates and similar images
- **Parallel processing**: Multi-threaded for performance

### Image Filtering & Analysis
- **Focus detection**: Laplacian variance for sharpness assessment
- **Brightness analysis**: Mean brightness and saturation evaluation
- **Interactive viewer**: Streamlit web app for visual review
- **Flexible filtering**: Configurable thresholds and criteria
- **Detailed reporting**: CSV output with metrics and decisions

### Cloud Upload & Transfer
- **Multiple upload methods**: AWS CLI, s5cmd, Boto3, presigned URLs
- **Performance benchmarking**: Compare upload speeds and methods
- **Upload time calculator**: Interactive tool for planning transfers
- **Batch processing**: Handle large collections efficiently
- **Progress tracking**: Real-time statistics and ETA

## 📊 Performance Characteristics

### Typical Performance (varies by hardware and data)
- **Image conversion**: 50-200 images/minute
- **PNG optimization**: 20-100 images/minute (depending on tool/effort)
- **S3 uploads**: 100-1000 MB/minute (depending on method and connection)
- **Similarity filtering**: 100-500 images/minute

### Optimization Tips
- Use appropriate compression tools for your speed/quality needs
- Enable parallel processing for multi-core systems
- Use similarity filtering to reduce dataset size before processing
- Choose upload methods based on your network and requirements

## 🔗 Integration

These tools are designed to work together:

1. **Filter** images for quality using image-filtering tools
2. **Convert** and optimize using image-converters
3. **Upload** efficiently using image-uploaders
4. **Analyze** results and iterate on parameters

## 📚 Documentation

Each tool category includes comprehensive documentation:
- Detailed README files with usage examples
- Installation guides for external dependencies
- Performance optimization tips
- Integration examples

## 🤝 Contributing

This repository focuses on practical image processing workflows. Contributions are welcome for:
- Performance optimizations
- Additional format support
- New filtering algorithms
- Enhanced user interfaces
- Documentation improvements

## 📄 License

This project is designed for research and development use. Please ensure compliance with any applicable licenses for external tools and dependencies.

---

**Need help getting started?** Check the README files in each tool category for detailed instructions and examples.