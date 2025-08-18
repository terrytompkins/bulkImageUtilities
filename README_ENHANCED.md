# Enhanced Bulk Image Converter

This enhanced version of the image converter provides advanced PNG compression and archiving capabilities, implementing the recommendations from the PNG compression playbook.

## 🚀 New Features

### 1. Advanced PNG Compression Tools

The script now supports multiple PNG compression tools beyond the built-in Pillow compression:

- **oxipng**: Fast, multi-core PNG optimization (recommended default)
- **zopflipng**: Maximum compression (slower but more effective)
- **pngcrush**: Classic PNG optimization tool
- **pillow**: Original built-in compression (default)

### 2. Archive Creation

Create compressed archives of processed images with multiple compression algorithms:

- **zstd**: Balanced compression (recommended)
- **7z**: Maximum compression (slower)
- **pigz**: Fastest compression (least CPU intensive)

### 3. Batch Processing

Automatically split large image collections into manageable archive batches (200-500 MB) for fault-tolerant uploads.

### 4. Enhanced Reporting

Detailed JSON reports for both compression and archiving operations with timing and compression ratios.

## 📋 Installation

### Required Tools

Install the compression tools based on your needs:

```bash
# PNG compression tools
pip install oxipng
# or install via package manager:
# brew install oxipng zopflipng pngcrush (macOS)
# apt install oxipng zopflipng pngcrush (Ubuntu/Debian)

# Archive compression tools
# brew install zstd p7zip pigz (macOS)
# apt install zstd p7zip pigz (Ubuntu/Debian)
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start Examples

### Basic PNG Compression (Fast)

```bash
python image_converter.py sample_images output_oxipng report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 4 \
  --strip-metadata
```

### Maximum PNG Compression (Slow but Effective)

```bash
python image_converter.py sample_images output_zopfli report.csv \
  --png-compression-tool zopflipng \
  --png-effort-level 15
```

### PNG Compression + Archive Creation (Balanced)

```bash
python image_converter.py sample_images output_zstd report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 6 \
  --create-archives \
  --archive-tool zstd \
  --archive-compression-level 9 \
  --batch-size-mb 300 \
  --zstd-long-range
```

### Maximum Compression Pipeline

```bash
python image_converter.py sample_images output_max report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 7 \
  --create-archives \
  --archive-tool 7z \
  --archive-compression-level 7 \
  --batch-size-mb 200 \
  --sevenz-solid \
  --sevenz-multithread
```

## 🔧 Command Line Options

### PNG Compression Options

| Option | Description | Default |
|--------|-------------|---------|
| `--png-compression-tool` | Tool: pillow, oxipng, zopflipng, pngcrush | pillow |
| `--png-effort-level` | Compression effort (0-7 for oxipng, iterations for zopflipng) | 4 |
| `--strip-metadata` | Strip non-essential metadata from PNGs | False |

### Archive Options

| Option | Description | Default |
|--------|-------------|---------|
| `--create-archives` | Enable archive creation | False |
| `--archive-tool` | Tool: zstd, 7z, pigz | zstd |
| `--archive-compression-level` | Compression level (varies by tool) | 9 |
| `--batch-size-mb` | Target archive size in MB | 300 |
| `--archive-output-dir` | Output directory for archives | dest_dir |

### zstd-specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--zstd-long-range` | Enable long-range matching | False |
| `--zstd-ultra` | Enable ultra compression mode | False |
| `--zstd-dictionary` | Path to zstd dictionary file | None |

### 7z-specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sevenz-solid` | Enable solid compression | True |
| `--sevenz-multithread` | Enable multithreading | True |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-workers` | Maximum parallel workers | CPU count |
| `--dry-run` | Show planned operations without executing | False |

### Filtering Options

| Option | Description | Default |
|--------|-------------|---------|
| `--filter-similar` | Enable similarity-based filtering | False |
| `--similarity-threshold` | Similarity threshold (0-1) | 0.85 |
| `--min-group-size` | Minimum group size for filtering | 2 |
| `--selection-method` | Method: best_quality, largest, smallest, first | best_quality |

### Report Options

| Option | Description |
|--------|-------------|
| `--compression-report` | Path for PNG compression report (JSON) |
| `--archive-report` | Path for archive creation report (JSON) |

## 📊 Tool Comparison

### PNG Compression Tools

| Tool | Speed | Compression | CPU Usage | Use Case |
|------|-------|-------------|-----------|----------|
| **oxipng** | Fast | Good | Low | General purpose, edge devices |
| **zopflipng** | Slow | Excellent | High | Maximum compression needed |
| **pngcrush** | Medium | Good | Medium | Classic approach |
| **pillow** | Fast | Basic | Low | Simple compression |

### Archive Tools

| Tool | Speed | Compression | CPU Usage | Use Case |
|------|-------|-------------|-----------|----------|
| **zstd** | Fast | Excellent | Low | Balanced approach |
| **7z** | Slow | Maximum | High | Maximum compression |
| **pigz** | Very Fast | Good | Very Low | Speed critical |

## 🎯 Recommended Workflows

### 1. Fast Edge Processing
```bash
python image_converter.py images output report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 4 \
  --create-archives \
  --archive-tool pigz \
  --batch-size-mb 500 \
  --max-workers 8
```

### 2. Balanced Production
```bash
python image_converter.py images output report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 6 \
  --create-archives \
  --archive-tool zstd \
  --archive-compression-level 9 \
  --batch-size-mb 300 \
  --zstd-long-range \
  --max-workers 4
```

### 3. Maximum Compression
```bash
python image_converter.py images output report.csv \
  --png-compression-tool zopflipng \
  --png-effort-level 15 \
  --create-archives \
  --archive-tool 7z \
  --archive-compression-level 7 \
  --batch-size-mb 200 \
  --max-workers 2
```

### 4. With Similarity Filtering
```bash
python image_converter.py images output report.csv \
  --png-compression-tool oxipng \
  --png-effort-level 4 \
  --filter-similar \
  --similarity-threshold 0.85 \
  --create-archives \
  --archive-tool zstd \
  --batch-size-mb 300
```

## 📈 Performance Tips

### PNG Compression
- Start with **oxipng effort level 4** for good balance
- Use **zopflipng** only for maximum compression needs
- **Strip metadata** for additional size reduction
- Use **parallel processing** with `--max-workers`

### Archive Creation
- **zstd level 9** provides excellent compression/speed balance
- **Long-range matching** helps with similar images
- **Batch sizes 200-500 MB** for fault-tolerant uploads
- **Ultra mode** for maximum zstd compression (slower)

### System Optimization
- Adjust `--max-workers` based on CPU cores
- Use `--dry-run` to test configurations
- Monitor memory usage with large datasets
- Consider SSD storage for temporary files

## 📋 Output Files

### CSV Report
Standard conversion report with compression ratios and timing.

### JSON Reports
Detailed reports for compression and archiving operations:
- File-by-file compression results
- Archive creation details
- Timing information
- Error tracking

### Archive Files
- `part01.tar.zst`, `part02.tar.zst`, etc. (zstd)
- `part01.7z`, `part02.7z`, etc. (7z)
- `part01.tar.gz`, `part02.tar.gz`, etc. (pigz)

## 🔍 Troubleshooting

### Common Issues

1. **Tool not found**: Install required compression tools
2. **Memory errors**: Reduce `--max-workers` or `--batch-size-mb`
3. **Slow processing**: Use faster tools (oxipng + pigz)
4. **Large archives**: Reduce `--batch-size-mb`

### Performance Monitoring

```bash
# Monitor CPU and memory usage
htop

# Check disk space
df -h

# Monitor network upload speed
iftop
```

## 📚 Advanced Usage

### Custom zstd Dictionary
```bash
# Train dictionary on sample images
zstd --train sample_images/*.png -o png.dict

# Use dictionary for compression
python image_converter.py images output report.csv \
  --png-compression-tool oxipng \
  --create-archives \
  --archive-tool zstd \
  --zstd-dictionary png.dict
```

### Parallel Upload Strategy
```bash
# Upload archives in parallel
ls *.tar.zst | parallel -j 4 "aws s3 cp {} s3://bucket/prefix/"
```

### Integration with Existing Workflows
The script maintains backward compatibility with existing CSV reports while adding new capabilities through optional parameters.

## 🤝 Contributing

This enhanced version builds upon the original image converter while adding the advanced compression and archiving features recommended in the PNG compression playbook. The implementation prioritizes:

- **Backward compatibility** with existing workflows
- **Configurable performance** for different use cases
- **Comprehensive reporting** for analysis and optimization
- **Fault tolerance** through batch processing
- **Parallel processing** for efficiency

For questions or improvements, please refer to the original project documentation and the PNG compression playbook.
