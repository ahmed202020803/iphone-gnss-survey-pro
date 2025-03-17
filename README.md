# iPhone GNSS Survey Pro

A professional-grade technical surveying tool optimized for iPhone 16 Pro Max that processes video with GNSS data to create georeferenced images and generate detailed survey reports.

## Features

- **High-Quality Frame Extraction**: Automatically extracts frames from iPhone 16 Pro Max videos at specified intervals with blur detection and quality assessment.
- **Multi-Format GNSS Support**: Works with CSV, GPX, and NMEA GNSS data formats.
- **Precise Positioning**: Applies camera offsets and heading corrections for accurate positioning.
- **Georeferenced Images**: Embeds geographic coordinates in image metadata and adds visual markers.
- **Comprehensive Reporting**: Generates HTML reports with interactive maps, KML files for GIS applications, and CSV exports.
- **Indoor Positioning**: Enhanced algorithms for indoor surveying with improved error estimation.
- **Quality Control**: Automatic blur detection, brightness analysis, and quality scoring for each frame.
- **Interactive Mode**: Seamless integration with Google Colab for easy use without local installation.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- FFmpeg (for video processing)
- ExifTool (for metadata manipulation)
- Additional dependencies in requirements.txt

## Installation

```bash
# Clone the repository
git clone https://github.com/ahmed202020803/iphone-gnss-survey-pro.git
cd iphone-gnss-survey-pro

# Install dependencies
pip install -r requirements.txt

# Install external tools
# FFmpeg and ExifTool are required for full functionality
```

## Usage

### Command Line

```bash
python iphone_gnss_survey.py --video path/to/video.mp4 --gnss path/to/gnss_data.csv --pole-height 1.5 --frame-interval 0.5 --indoor
```

### Google Colab

The tool can be run interactively in Google Colab:

```python
from iphone_gnss_survey import run_video_gnss_survey

# Run with sample GNSS data
report_path = run_video_gnss_survey(
    video_path="path/to/video.mp4",
    use_sample_data=True,
    pole_height=1.5,
    frame_interval=0.5,
    indoor_mode=True
)
```

## Configuration Options

The tool is highly configurable through a JSON configuration file or command-line arguments:

- **Video Settings**: Frame interval, quality thresholds, blur detection sensitivity
- **GNSS Settings**: Data format, time offset, interpolation method
- **Camera Settings**: Offset from GNSS antenna, heading correction
- **Output Settings**: Directory, report generation options, coordinate format

## Output

- **Georeferenced Images**: JPG files with embedded geographic coordinates
- **HTML Report**: Interactive report with maps, tables, and image thumbnails
- **KML File**: For viewing in Google Earth or other GIS applications
- **CSV Export**: Tabular data with coordinates and quality metrics

## Technical Details

- **iPhone 16 Pro Max Optimization**: Tuned for the specific camera characteristics and video formats of the iPhone 16 Pro Max
- **Blur Detection**: Uses Laplacian variance to detect and filter blurry frames
- **GNSS Interpolation**: Linear, cubic, or nearest-neighbor interpolation for precise positioning between GNSS points
- **Error Estimation**: Calculates position error estimates based on HDOP and environment (indoor/outdoor)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.