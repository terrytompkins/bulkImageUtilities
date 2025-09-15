# Image Conversion Tools

This directory contains tools for converting, compressing, and optimizing images with advanced features for bulk processing.

## Contents

### Scripts
- **`image_converter.py`** - Main image conversion and compression tool with advanced features
- **`hdr_compression.ipynb`** - Jupyter notebook for HDR image compression analysis
- **`compression_example.py`** - Example usage script demonstrating PNG compression and archiving features

### Testing
- **`tests/`** - Comprehensive test suite for validation and benchmarking

### Documentation
- **`INSTALL_COMPRESSION_TOOLS.md`** - Installation guide for external compression tools
- **`png_compression_playbook.md`** - Comprehensive PNG compression strategies and benchmarks

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install External Tools
See `INSTALL_COMPRESSION_TOOLS.md` for detailed installation instructions for:
- oxipng, zopflipng, pngcrush (PNG optimization)
- zstd, 7z, pigz (compression/archiving)
- avifenc (AVIF encoding)

### 3. Basic Image Conversion
```bash
python image_converter.py input_dir output_dir report.csv
```

## Features

### Image Conversion
- **Format Support**: JPG, PNG, TIFF, BMP, WebP, AVIF
- **Quality Control**: Configurable compression levels
- **Batch Processing**: Process entire directories
- **Progress Tracking**: Real-time progress and statistics

### Advanced PNG Compression
- **Multiple Tools**: oxipng, zopflipng, pngcrush
- **Effort Levels**: Configurable compression effort (1-6)
- **Lossless Optimization**: Maximum compression without quality loss
- **Performance Tuning**: Balance speed vs compression ratio

### Archiving & Compression
- **Multiple Formats**: zstd, 7z, pigz
- **Batch Archiving**: Split large datasets into manageable archives
- **Configurable Sizes**: Set maximum archive sizes
- **Parallel Processing**: Multi-threaded compression

### Similarity Analysis
- **Duplicate Detection**: Find and remove similar images
- **Feature Extraction**: Advanced image feature analysis
- **Clustering**: Group similar images together
- **Quality Filtering**: Remove low-quality images

## Usage Examples

### Basic Conversion
```bash
# Convert images to WebP with 80% quality
python image_converter.py input_dir output_dir report.csv --output-format webp --quality 80

# Convert to AVIF with high quality
python image_converter.py input_dir output_dir report.csv --output-format avif --quality 90
```

### PNG Optimization
```bash
# Optimize PNGs with oxipng (effort level 3)
python image_converter.py input_dir output_dir report.csv --png-compression-tool oxipng --png-effort-level 3

# Use zopflipng for maximum compression
python image_converter.py input_dir output_dir report.csv --png-compression-tool zopflipng
```

### Archiving
```bash
# Create zstd archives (100MB each)
python image_converter.py input_dir output_dir report.csv --create-archives --archive-format zstd --max-archive-size 100

# Create 7z archives with compression
python image_converter.py input_dir output_dir report.csv --create-archives --archive-format 7z --max-archive-size 200
```

### Similarity Filtering
```bash
# Remove similar images (95% similarity threshold)
python image_converter.py input_dir output_dir report.csv --filter-similar --similarity-threshold 0.95

# Preview mode - analyze without copying
python image_converter.py input_dir output_dir report.csv --filter-similar --preview-only
```

### Example Usage Script
```bash
# Run the compression example script to see various usage patterns
python compression_example.py
```

## Advanced Features

### Performance Optimization
- **Parallel Processing**: Multi-threaded conversion
- **Memory Management**: Efficient handling of large images
- **Progress Reporting**: Real-time statistics and ETA
- **Resume Support**: Continue interrupted operations

### Quality Analysis
- **Feature Extraction**: Advanced image analysis
- **Similarity Scoring**: Cosine similarity and clustering
- **Quality Metrics**: Focus detection and brightness analysis
- **Statistical Reports**: Detailed CSV output

### Integration
- **CSV Reports**: Detailed processing logs
- **Error Handling**: Robust error recovery
- **Configuration**: Flexible command-line options
- **Logging**: Comprehensive operation logs

## Performance Tips

### Compression
- Use **oxipng** for balanced speed/compression
- Use **zopflipng** for maximum PNG compression
- Use **zstd** for fast archiving
- Use **7z** for maximum archive compression

### Processing
- Adjust `--max-workers` based on CPU cores
- Use `--max-side` to resize large images for speed
- Enable `--create-archives` for large datasets
- Use `--filter-similar` to reduce dataset size

## Integration

These tools work seamlessly with other utilities:
- Use **image-filtering** tools to pre-filter images
- Use **image-uploaders** to upload processed images
- Generate reports for quality analysis
- Create optimized archives for distribution

## Documentation

- **`INSTALL_COMPRESSION_TOOLS.md`** - Complete installation guide
- **`png_compression_playbook.md`** - PNG optimization strategies
- **`hdr_compression.ipynb`** - HDR compression analysis examples
