# iPhone GNSS Survey Pro

A sophisticated technical surveying tool designed to process video footage with GNSS (Global Navigation Satellite System) data, specifically optimized for iPhone 16 Pro Max.

## Features

- **High-Quality Frame Extraction**: Automatically extracts frames from video with quality analysis
- **Multi-Format GNSS Support**: Processes data from CSV, GPX, and NMEA formats
- **Precise Positioning**: Calculates camera positions with offsets and error estimation
- **Georeferenced Images**: Embeds geographic metadata in extracted frames
- **Comprehensive Reporting**: Generates HTML reports with interactive maps
- **Indoor Positioning**: Enhanced algorithms for indoor surveying
- **Quality Control**: Blur detection, brightness analysis, and contrast evaluation
- **Interactive Mode**: Google Colab integration for easy usage

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- FFmpeg
- ExifTool
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ahmed202020803/iphone-gnss-survey-pro.git
cd iphone-gnss-survey-pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
- FFmpeg: [Installation Guide](https://ffmpeg.org/download.html)
- ExifTool: [Installation Guide](https://exiftool.org/install.html)

## Usage

### Command Line Interface

```bash
python iphone_gnss_survey.py --video path/to/video.mp4 --gnss path/to/gnss.csv --output output_dir
```

Optional arguments:
- `--config`: Path to configuration file
- `--sample-data`: Use sample GNSS data
- `--pole-height`: Camera pole height in meters (default: 1.5)
- `--frame-interval`: Frame extraction interval in seconds (default: 0.5)
- `--indoor`: Enable indoor positioning mode

### Google Colab

The tool can be run interactively in Google Colab. Import the module and use:

```python
from iphone_gnss_survey import run_video_gnss_survey

# Run the survey with default parameters
report_path = run_video_gnss_survey(
    video_path="path/to/video.mp4",
    use_sample_data=True,
    pole_height=1.5,
    frame_interval=0.5,
    indoor_mode=True
)
```

## Configuration

The tool can be configured via JSON file or command-line arguments. Example configuration:

```json
{
    "video": {
        "frame_interval": 0.5,
        "resolution": "1080x1920",
        "quality_threshold": 40
    },
    "gnss": {
        "interpolation_method": "linear",
        "min_satellites": 6
    },
    "camera": {
        "offset_up": 1.5
    }
}
```

## Output

- **Georeferenced Images**: JPEG images with embedded geographic metadata
- **Marked Images**: Visual overlays showing coordinates and quality metrics
- **HTML Report**: Interactive survey report with map and statistics
- **KML File**: For Google Earth visualization
- **CSV Export**: Tabular data of all survey points

## Technical Details

- Optimized for iPhone 16 Pro Max (1080x1920 resolution, 59.97 FPS)
- Uses Laplacian variance for blur detection
- Implements linear, cubic, and nearest-neighbor interpolation for GNSS data
- Calculates camera position with north, east, and up offsets
- Estimates horizontal and vertical errors based on HDOP and environment

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.