# Installation Guide for Compression Tools

This guide explains how to install the external compression tools required for the enhanced PNG compression and archiving features.

## 🚀 Quick Installation

### macOS (using Homebrew)
```bash
# Install all compression tools at once
brew install oxipng zopflipng pngcrush zstd p7zip pigz

# Or install individually
brew install oxipng      # Fast PNG optimization
brew install zopflipng   # Maximum PNG compression
brew install pngcrush    # Classic PNG optimization
brew install zstd        # Fast archive compression
brew install p7zip       # Maximum archive compression
brew install pigz        # Parallel gzip compression
```

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install all compression tools
sudo apt install oxipng zopflipng pngcrush zstd p7zip pigz

# Or install individually
sudo apt install oxipng      # Fast PNG optimization
sudo apt install zopflipng   # Maximum PNG compression
sudo apt install pngcrush    # Classic PNG optimization
sudo apt install zstd        # Fast archive compression
sudo apt install p7zip       # Maximum archive compression
sudo apt install pigz        # Parallel gzip compression
```

### CentOS/RHEL/Fedora
```bash
# Fedora
sudo dnf install oxipng zopflipng pngcrush zstd p7zip pigz

# CentOS/RHEL (enable EPEL first)
sudo yum install epel-release
sudo yum install oxipng zopflipng pngcrush zstd p7zip pigz
```

### Windows
```bash
# Using Chocolatey
choco install oxipng zopflipng pngcrush zstd 7zip pigz

# Using Scoop
scoop install oxipng zopflipng pngcrush zstd 7zip pigz
```

## 📦 Individual Tool Details

### PNG Compression Tools

#### oxipng
- **Purpose**: Fast, multi-core PNG optimization
- **Speed**: Fast
- **Compression**: Good
- **Use case**: General purpose, edge devices
- **Installation**: `brew install oxipng` (macOS) or `apt install oxipng` (Ubuntu)

#### zopflipng
- **Purpose**: Maximum PNG compression using Google's Zopfli algorithm
- **Speed**: Slow
- **Compression**: Excellent
- **Use case**: Maximum compression needed
- **Installation**: `brew install zopflipng` (macOS) or `apt install zopflipng` (Ubuntu)

#### pngcrush
- **Purpose**: Classic PNG optimization tool
- **Speed**: Medium
- **Compression**: Good
- **Use case**: Classic approach, compatibility
- **Installation**: `brew install pngcrush` (macOS) or `apt install pngcrush` (Ubuntu)

### Archive Compression Tools

#### zstd
- **Purpose**: Fast, high-compression ratio archive compression
- **Speed**: Fast
- **Compression**: Excellent
- **Use case**: Balanced approach (recommended)
- **Installation**: `brew install zstd` (macOS) or `apt install zstd` (Ubuntu)

#### 7z (p7zip)
- **Purpose**: Maximum archive compression using LZMA2
- **Speed**: Slow
- **Compression**: Maximum
- **Use case**: Maximum compression needed
- **Installation**: `brew install p7zip` (macOS) or `apt install p7zip` (Ubuntu)

#### pigz
- **Purpose**: Parallel gzip compression
- **Speed**: Very Fast
- **Compression**: Good
- **Use case**: Speed critical applications
- **Installation**: `brew install pigz` (macOS) or `apt install pigz` (Ubuntu)

## 🔍 Verification

After installation, verify that all tools are available:

```bash
# Check if tools are in PATH
which oxipng
which zopflipng
which pngcrush
which zstd
which 7z
which pigz

# Test basic functionality
oxipng --version
zopflipng --version
pngcrush -version
zstd --version
7z --version
pigz --version
```

## 🐍 Python Dependencies

Install the Python dependencies:

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install individually
pip install boto3>=1.26.0
pip install botocore>=1.29.0
pip install pillow>=11.3.0
pip install numpy>=1.21.0
pip install opencv-python>=4.8.0
pip install scikit-learn>=1.3.0
```

## 🧪 Testing Installation

Run the test suite to verify everything works:

```bash
python test_enhanced_features.py
```

This will:
- Check tool availability
- Create test images
- Run compression tests
- Test archive creation
- Verify all features work correctly

## 🔧 Troubleshooting

### Tool Not Found Errors

If you get "tool not found" errors:

1. **Check PATH**: Ensure tools are in your system PATH
2. **Restart terminal**: Sometimes PATH changes require terminal restart
3. **Install manually**: Download and install tools manually if package manager fails
4. **Check versions**: Ensure you have compatible versions

### Permission Errors

```bash
# Fix permission issues (macOS/Linux)
sudo chmod +x /usr/local/bin/oxipng
sudo chmod +x /usr/local/bin/zopflipng
sudo chmod +x /usr/local/bin/pngcrush
sudo chmod +x /usr/local/bin/zstd
sudo chmod +x /usr/local/bin/7z
sudo chmod +x /usr/local/bin/pigz
```

### Version Conflicts

If you have multiple versions installed:

```bash
# Check all installed versions
ls -la /usr/local/bin/ | grep -E "(oxipng|zopflipng|pngcrush|zstd|7z|pigz)"
ls -la /usr/bin/ | grep -E "(oxipng|zopflipng|pngcrush|zstd|7z|pigz)"

# Remove conflicting versions
sudo rm /usr/local/bin/old_version_tool
```

## 📊 Performance Notes

### Recommended Tool Combinations

| Use Case | PNG Tool | Archive Tool | Notes |
|----------|----------|--------------|-------|
| **Fast Processing** | oxipng (level 4) | pigz | Best for edge devices |
| **Balanced** | oxipng (level 6) | zstd | Good compression/speed balance |
| **Maximum Compression** | zopflipng | 7z | Slowest but smallest files |
| **Legacy Compatibility** | pngcrush | pigz | Widest compatibility |

### System Requirements

- **CPU**: Multi-core recommended for parallel processing
- **Memory**: 2GB+ RAM for large image batches
- **Storage**: SSD recommended for temporary files
- **Network**: High bandwidth for archive uploads

## 🎯 Next Steps

After installation:

1. **Test with sample data**: Use the example scripts
2. **Benchmark performance**: Test with your actual image data
3. **Optimize settings**: Adjust compression levels based on your needs
4. **Monitor resources**: Watch CPU and memory usage
5. **Scale up**: Test with larger datasets

## 📚 Additional Resources

- [oxipng Documentation](https://github.com/shssoichiro/oxipng)
- [zopflipng Documentation](https://github.com/google/zopfli)
- [pngcrush Documentation](http://pmt.sourceforge.net/pngcrush/)
- [zstd Documentation](https://github.com/facebook/zstd)
- [7-Zip Documentation](https://7-zip.org/)
- [pigz Documentation](https://zlib.net/pigz/)
