# Segmentation Tool

A comprehensive tool for image segmentation analysis using LandingAI. This tool provides both CLI and library interfaces for processing images, calculating measurements, and exporting results in various formats.

## Features

- **Image Segmentation**: Process single images or batch directories using LandingAI
- **Comprehensive Measurements**: Calculate absolute and relative coverage, area, and instance counts
- **Multiple Output Formats**: Export results to JSON, CSV, or VOC dataset format
- **Interactive Visualization**: View segmentation overlays with customizable color schemes
- **Batch Processing**: Process multiple images with progress tracking
- **Library Interface**: Use as a Python library in your own projects
- **CLI Interface**: Command-line tool for easy integration into workflows

## Installation

### Prerequisites

- Python 3.8 or higher
- LandingAI API key and endpoint ID

### Install from Source

```bash
git clone <repository-url>
cd segmentation_tool
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

**Note**: The tool uses `python-dotenv` for automatic `.env` file loading.

## Quick Start

### 1. Set up API Credentials

**Recommended: Use .env file (easiest method)**

```bash
cp .env.example .env
# Edit .env with your actual API credentials
```

**Alternative: Environment variables**

```bash
export LANDINGAI_API_KEY='your_api_key_here'
export LANDINGAI_ENDPOINT='your_endpoint_id_here'
```

**Alternative: Command-line arguments**

```bash
segmentation-tool ./images/ --api-key YOUR_KEY --endpoint YOUR_ENDPOINT
```

**Credential Precedence (highest to lowest):**
1. CLI arguments (`--api-key`, `--endpoint`) - overrides everything
2. Environment variables (`LANDINGAI_API_KEY`, `LANDINGAI_ENDPOINT`)
3. `.env` file - fallback option

### 2. CLI Usage

Process a single image with measurements:

```bash
segmentation-tool image.jpg --measure
```

Process a directory and save results:

```bash
segmentation-tool ./images/ --measure --output ./results/ --format json
# Creates: ./results/image1.json, ./results/image2.json, ... (individual predictions)
#          ./results/results.json (unified measurements)
```

Process with visualization and VOC export:

```bash
segmentation-tool ./images/ --measure --visualize --output ./results/ --voc
```

### 3. Library Usage

```python
from segmentation_tool import SegmentationProcessor

# Initialize processor
processor = SegmentationProcessor(
    api_key="your_api_key",
    endpoint_id="your_endpoint"
)

# Process single image
result = processor.process_image("image.jpg", calculate_measurements=True)
print(f"Found {len(result.predictions)} detections")

# Process batch
results = processor.process_batch("./images/", calculate_measurements=True)
```

## CLI Reference

### Basic Usage

```bash
segmentation-tool [OPTIONS] INPUT_PATH
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | LandingAI API key (or set `LANDINGAI_API_KEY` env var) | - |
| `--endpoint` | LandingAI endpoint ID (or set `LANDINGAI_ENDPOINT` env var) | - |
| `--output DIR` | Output directory for results | - |
| `--measure` | Calculate absolute and relative coverage measurements | False |
| `--visualize` | Show interactive visualization for each image | False |
| `--format [json|csv]` | Output format for results | json |
| `--voc` | Export VOC dataset format alongside results | False |
| `--timeout INT` | Request timeout in seconds | 60 |
| `--verbose` | Enable detailed output | False |

### Examples

#### Basic Processing
```bash
# Process single image
segmentation-tool image.jpg

# Process directory
segmentation-tool ./images/
```

#### With Measurements
```bash
# Calculate and display measurements
segmentation-tool ./images/ --measure

# Save predictions + measurements to JSON
segmentation-tool ./images/ --measure --output ./results/ --format json
# Creates: individual prediction files + results.json

# Save predictions + measurements to CSV
segmentation-tool ./images/ --measure --output ./results/ --format csv
# Creates: individual prediction files + results.csv
```

#### With Visualization
```bash
# Show visualization for each image
segmentation-tool ./images/ --visualize

# Combine measurements and visualization
segmentation-tool ./images/ --measure --visualize
```

#### VOC Dataset Export
```bash
# Export to VOC format
segmentation-tool ./images/ --measure --output ./results/ --voc

# This creates:
# ./results/sample.json, ./results/image2.json, ... # Individual prediction files
# ./results/results.json (or .csv)                  # Unified measurement file (if --measure used)
# ./results/VOCDataset/Images/                      # Original images
# ./results/VOCDataset/Segmentations/               # Segmentation masks
# ./results/VOCDataset/defect_map.json              # Class mappings
```

## Library Reference

### Core Classes

#### SegmentationProcessor

Main processing engine for image segmentation.

```python
from segmentation_tool import SegmentationProcessor

processor = SegmentationProcessor(
    api_key="your_key",
    endpoint_id="your_endpoint",
    timeout=60  # optional
)

# Process single image
result = processor.process_image(
    image_path="image.jpg",
    calculate_measurements=True
)

# Process batch
results = processor.process_batch(
    input_path="./images/",
    calculate_measurements=True,
    progress_callback=None  # optional callback function
)
```

#### SegmentationResult

Container for processing results.

```python
# Access result data
print(f"Image: {result.image_path}")
print(f"Predictions: {len(result.predictions)}")
print(f"Processing time: {result.processing_time}s")

# Access measurements (if calculated)
if result.measurements:
    measurements = result.measurements
    print(f"Total coverage: {measurements['totals']['total_relative_coverage_percent']:.2f}%")

    for class_name, metrics in measurements['by_class'].items():
        print(f"{class_name}: {metrics['instance_count']} instances")
```

#### VOCConverter

Convert results to Pascal VOC format.

```python
from segmentation_tool import VOCConverter

converter = VOCConverter("./output_directory")
summary = converter.convert(results)

print(f"Converted {summary['converted_images']} images")
print(f"Classes: {', '.join(summary['classes_found'])}")
```

#### Visualization

```python
from segmentation_tool import show_visualization, create_batch_summary

# Show single image visualization
show_visualization(image_path, result)

# Create batch summary plot
create_batch_summary(results, save_path="summary.png")
```

### Convenience Functions

```python
from segmentation_tool import process_image, process_batch

# Quick single image processing
result = process_image("image.jpg", api_key, endpoint_id, calculate_measurements=True)

# Quick batch processing
results = process_batch("./images/", api_key, endpoint_id, calculate_measurements=True)
```

## Output Formats

The tool uses a **dual output system** when `--output` is specified:

1. **Prediction Results**: Individual JSON files per image (`imagefilename.json`)
2. **Measurement Results**: Unified files when `--measure` flag is used (`results.json/csv`)

### Individual Prediction Files (Always Created)

Each processed image gets its own JSON file with predictions:

**File: `sample.json`**
```json
{
  "metadata": {
    "processed_at": "2024-01-15T10:30:00",
    "api_endpoint": "endpoint_id"
  },
  "result": {
    "image_path": "sample.jpg",
    "predictions": [
      {
        "label_name": "defect_class",
        "encoding_map": {...},
        "mask_shape": [1080, 1920],
        "encoded_mask": "..."
      }
    ],
    "processing_time": 2.34
  }
}
```

### Unified Measurement Files (Created with --measure)

Single file containing measurements for all images:

```json
{
  "metadata": {
    "processed_at": "2024-01-15T10:30:00",
    "total_images": 5,
    "api_endpoint": "endpoint_id"
  },
  "results": [
    {
      "image": "image1.jpg",
      "measurements": {
        "summary": {
          "total_instances": 3,
          "unique_classes": 2,
          "total_pixels": 2073600
        },
        "by_class": {
          "defect_class": {
            "absolute_coverage_pixels": 15420,
            "relative_coverage_percent": 0.74,
            "instance_count": 3
          }
        },
        "totals": {
          "total_absolute_coverage_pixels": 15420,
          "total_relative_coverage_percent": 0.74
        }
      }
    }
  ]
}
```

### CSV Output

Single file with measurements for all images:

```csv
Image,Class,Instance Count,Absolute Coverage (pixels),Relative Coverage (%),Image Width,Image Height,Total Pixels
image1.jpg,defect_class,3,15420,0.74,1920,1080,2073600
image1.jpg,TOTAL,3,15420,0.74,1920,1080,2073600
image2.jpg,defect_class,1,5200,0.25,1920,1080,2073600
...
```

### VOC Dataset Format

When using `--voc` option, creates a complete VOC dataset:

```
VOCDataset/
├── Images/              # Original images copied here
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Segmentations/       # PNG segmentation masks
│   ├── image1.png       # Each pixel value represents class ID
│   ├── image2.png
│   └── ...
└── defect_map.json      # Class ID to name mapping
    {
      "0": "background",
      "1": "defect_class",
      "2": "another_class"
    }
```

## Measurement Details

The tool calculates comprehensive measurements using LandingAI's postprocessing functions:

### Per-Class Metrics
- **Absolute Coverage**: Number of pixels covered by each class
- **Relative Coverage**: Percentage of total image area covered by each class
- **Instance Count**: Number of separate instances detected for each class

### Total Metrics
- **Total Absolute Coverage**: Combined pixel count across all classes
- **Total Relative Coverage**: Combined percentage coverage across all classes
- **Total Instances**: Total number of detections across all classes

### Summary Information
- **Image Dimensions**: Width × Height in pixels
- **Total Pixels**: Complete image pixel count
- **Unique Classes**: Number of different classes detected
- **Processing Time**: Time taken to process each image

## Development Setup

### Clone and Install in Development Mode

```bash
git clone <repository-url>
cd segmentation_tool
pip install -e .
```

### Install Development Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

```
segmentation_tool/
├── segmentation_tool/           # Main package
│   ├── __init__.py             # Package initialization and exports
│   ├── __main__.py             # Module entry point
│   ├── cli.py                  # Command-line interface
│   ├── core.py                 # Core processing logic
│   ├── voc_converter.py        # VOC format converter
│   ├── visualizer.py           # Visualization utilities
│   └── config.py               # Configuration management
├── examples/                   # Usage examples
│   ├── basic_usage.py          # Library usage examples
│   └── sample_images/          # Sample test images
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── setup.py                    # Package installation config
└── .env.example                # Environment variables template
```

### Running Examples

```bash
# Set up your API credentials first
export LANDINGAI_API_KEY='your_key'
export LANDINGAI_ENDPOINT='your_endpoint'

# Run the example script
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

#### 1. API Credentials Not Set
```
Error: API key and endpoint are required!
```
**Solution**: Use any of the three credential methods (in order of precedence):

1. **CLI arguments** (highest priority):
   ```bash
   segmentation-tool ./images/ --api-key YOUR_KEY --endpoint YOUR_ENDPOINT
   ```

2. **Environment variables**:
   ```bash
   export LANDINGAI_API_KEY='your_key'
   export LANDINGAI_ENDPOINT='your_endpoint'
   ```

3. **Create .env file** (lowest priority):
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

#### 2. No Images Found
```
ValueError: No image files found in directory
```
**Solution**: Ensure directory contains supported image formats (jpg, png, bmp, tiff)

#### 3. Visualization Not Showing
**Solution**: Ensure you have a display available and matplotlib backend configured:
```bash
export DISPLAY=:0  # On Linux with X11
```

#### 4. Permission Errors
```
IOError: Cannot create output directory
```
**Solution**: Check write permissions or use a different output directory:
```bash
chmod 755 /path/to/output/directory
```

#### 5. .env File Not Loading
**Issue**: Credentials in `.env` file not being recognized
**Solution**:
- Ensure `.env` file is in the current working directory (where you run the command)
- Check file format (no spaces around `=`, no quotes needed):
  ```
  LANDINGAI_API_KEY=your_actual_key_here
  LANDINGAI_ENDPOINT=your_actual_endpoint_here
  ```
- Environment variables and CLI args override `.env` file
- Use `--verbose` flag to check if credentials are being loaded

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Performance Tips

- Use `--timeout` option for slow networks
- Process images in smaller batches for memory efficiency
- Use `--verbose` flag for detailed progress information
- Disable visualization (`--visualize`) for faster batch processing

