import os
import cv2
import numpy as np
import datetime
import json
import csv
import math
import logging
import subprocess
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
from dateutil import parser
from tqdm import tqdm
from scipy.interpolate import interp1d
import folium
from google.colab import files
import io
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurveyConfig:
    """Configuration class for the survey settings."""
    
    def __init__(self, config_file=None):
        # Default configuration
        self.config = {
            "video": {
                "file_path": "",
                "frame_interval": 0.5,  # Extract frame every 0.5 seconds (optimized for 59.97 FPS)
                "resolution": "1080x1920",  # iPhone 16 Pro Max portrait mode
                "frame_rate": 59.97,
                "codec": "H.264",
                "quality_threshold": 40,  # Minimum quality score to keep a frame
                "blur_threshold": 100,  # Laplacian variance threshold for blur detection
                "brightness_range": [40, 220]  # Min and max average brightness values
            },
            "gnss": {
                "file_path": "",
                "format": "csv",  # csv, gpx, or nmea
                "time_offset": 0,  # Time offset in seconds between video and GNSS data
                "interpolation_method": "linear",  # linear, cubic, or nearest
                "min_satellites": 6,  # Minimum number of satellites for reliable positioning
                "hdop_threshold": 2.0,  # Horizontal dilution of precision threshold
                "use_sample_data": False  # Whether to use sample GNSS data
            },
            "camera": {
                "offset_north": 0.0,  # Offset in meters north from GNSS antenna
                "offset_east": 0.0,   # Offset in meters east from GNSS antenna
                "offset_up": 1.5,     # Offset in meters up from GNSS antenna (pole height)
                "heading_offset": 0.0, # Heading offset in degrees
                "calibration_file": "" # Camera calibration file path
            },
            "output": {
                "directory": "survey_output",
                "add_coordinates_to_images": True,
                "generate_report": True,
                "generate_kml": True,
                "generate_shapefile": False,
                "coordinate_format": "decimal_degrees"  # decimal_degrees or dms
            },
            "advanced": {
                "control_points": [],  # List of known control points [name, lat, lon, alt]
                "use_external_gnss": False,  # Whether to use external GNSS receiver
                "external_gnss_port": "",  # Serial port for external GNSS
                "feature_detection_algorithm": "SIFT",  # SIFT, SURF, or ORB
                "bundle_adjustment": True,  # Whether to perform bundle adjustment
                "error_estimation": True,  # Whether to estimate measurement errors
                "indoor_positioning": True  # Enable indoor positioning enhancements
            }
        }
        
        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                # Merge custom config with default config
                self.merge_config(self.config, custom_config)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def merge_config(self, default_config, custom_config):
        """Recursively merge custom configuration into default configuration."""
        for key, value in custom_config.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    self.merge_config(default_config[key], value)
                else:
                    default_config[key] = value
            else:
                default_config[key] = value
    
    def save_config(self, config_file):
        """Save current configuration to a JSON file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def update_config(self, section, key, value):
        """Update a specific configuration value."""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            logger.info(f"Updated config: {section}.{key} = {value}")
        else:
            logger.error(f"Invalid configuration key: {section}.{key}")
    
    def get_config(self, section=None, key=None):
        """Get configuration value(s)."""
        if section is None:
            return self.config
        elif section in self.config:
            if key is None:
                return self.config[section]
            elif key in self.config[section]:
                return self.config[section][key]
        return None

def check_video_file(file_path):
    """Check if the video file exists and get its properties."""
    if not os.path.exists(file_path):
        logger.error(f"Video file not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    logger.info(f"Video file found: {file_path} ({file_size:.2f} MB)")
    
    # Get video properties using FFmpeg
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name', 
            '-of', 'json', 
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        
        if 'streams' in video_info and len(video_info['streams']) > 0:
            stream = video_info['streams'][0]
            width = stream.get('width', 'unknown')
            height = stream.get('height', 'unknown')
            
            # Parse frame rate fraction (e.g., "60000/1001")
            r_frame_rate = stream.get('r_frame_rate', 'unknown')
            if r_frame_rate != 'unknown':
                num, den = map(int, r_frame_rate.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = 'unknown'
                
            codec = stream.get('codec_name', 'unknown')
            
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {codec} codec")
        else:
            logger.warning("Could not determine video properties")
    except Exception as e:
        logger.error(f"Error getting video properties: {e}")
    
    return True

def extract_video_creation_time(file_path):
    """Extract the creation time of the video from its metadata."""
    creation_time = None
    
    # Try using exiftool first (more reliable for iPhone videos)
    try:
        cmd = ['exiftool', '-CreateDate', '-json', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            exif_data = json.loads(result.stdout)
            if exif_data and 'CreateDate' in exif_data[0]:
                date_str = exif_data[0]['CreateDate']
                # Convert from "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
                date_str = date_str.replace(':', '-', 2).replace('-', ':', 2)
                creation_time = parser.parse(date_str)
                logger.info(f"Video creation time from EXIF: {creation_time}")
                return creation_time
    except Exception as e:
        logger.warning(f"Could not extract creation time using exiftool: {e}")
    
    # Fallback to ffprobe
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-select_streams', 'v:0', 
            '-show_entries', 'format_tags=creation_time', 
            '-of', 'json', 
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'format' in data and 'tags' in data['format'] and 'creation_time' in data['format']['tags']:
                creation_time = parser.parse(data['format']['tags']['creation_time'])
                logger.info(f"Video creation time from FFprobe: {creation_time}")
                return creation_time
    except Exception as e:
        logger.warning(f"Could not extract creation time using ffprobe: {e}")
    
    # If all else fails, use the file's modification time
    try:
        mtime = os.path.getmtime(file_path)
        creation_time = datetime.fromtimestamp(mtime)
        logger.warning(f"Using file modification time as fallback: {creation_time}")
    except Exception as e:
        logger.error(f"Could not determine video creation time: {e}")
        # Use current time as last resort
        creation_time = datetime.now()
        logger.warning(f"Using current time as last resort: {creation_time}")
    
    return creation_time

def analyze_frame_quality(frame):
    """Analyze the quality of a frame based on blur, brightness, and contrast."""
    if frame is None:
        return 0
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness (average pixel value)
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(gray)
    
    # Normalize and combine metrics into a quality score (0-100)
    blur_score = min(100, laplacian_var / 5)  # Higher variance = less blur
    brightness_score = 100 - abs(brightness - 128) * 100 / 128  # Closer to middle value (128) is better
    contrast_score = min(100, contrast * 2)  # Higher contrast is generally better
    
    # Weighted average of scores
    quality_score = 0.5 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score
    
    return quality_score