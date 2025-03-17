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

def extract_frames_ffmpeg(video_path, output_dir, interval, quality_threshold):
    """Extract frames from video using FFmpeg at specified intervals."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video duration using FFmpeg
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to get video duration: {result.stderr}")
        return []
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        logger.error(f"Invalid duration value: {result.stdout}")
        return []
    
    # Calculate timestamps for frame extraction
    timestamps = [i for i in np.arange(0, duration, interval)]
    
    # Get video creation time
    video_start_time = extract_video_creation_time(video_path)
    if video_start_time is None:
        video_start_time = datetime.now()
        logger.warning("Could not determine video creation time, using current time")
    
    extracted_frames = []
    
    # Extract frames at each timestamp
    for i, timestamp in enumerate(tqdm(timestamps, desc="Extracting frames")):
        output_file = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        
        # Extract frame at specific timestamp
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',  # High quality JPEG
            '-y',  # Overwrite output file
            output_file
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Read the extracted frame for quality analysis
            frame = cv2.imread(output_file)
            if frame is None:
                logger.warning(f"Failed to read extracted frame: {output_file}")
                continue
            
            # Analyze frame quality
            quality = analyze_frame_quality(frame)
            
            # Skip low-quality frames
            if quality < quality_threshold:
                logger.info(f"Skipping low-quality frame {i} (quality: {quality:.2f})")
                os.remove(output_file)
                continue
            
            # Calculate frame timestamp
            frame_time = video_start_time + timedelta(seconds=timestamp)
            
            # Store frame information
            frame_info = {
                'index': i,
                'path': output_file,
                'timestamp': frame_time.isoformat(),
                'video_time': timestamp,
                'quality': quality
            }
            
            extracted_frames.append(frame_info)
            logger.debug(f"Extracted frame {i} at {timestamp:.2f}s (quality: {quality:.2f})")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract frame at {timestamp:.2f}s: {e}")
    
    logger.info(f"Extracted {len(extracted_frames)} frames from video")
    return extracted_frames

def extract_frames_from_video(config):
    """Extract frames from video at specified intervals."""
    video_path = config.get_config('video', 'file_path')
    frame_interval = config.get_config('video', 'frame_interval')
    quality_threshold = config.get_config('video', 'quality_threshold')
    output_dir = os.path.join(config.get_config('output', 'directory'), 'frames')
    
    # Check if video file exists
    if not check_video_file(video_path):
        logger.error(f"Video file not found or invalid: {video_path}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use FFmpeg for frame extraction (more reliable for iPhone videos)
    return extract_frames_ffmpeg(video_path, output_dir, frame_interval, quality_threshold)

def generate_sample_gnss_data(frames, config):
    """Generate sample GNSS data for testing purposes."""
    logger.info("Generating sample GNSS data")
    
    # Base coordinates (can be adjusted as needed)
    base_lat = 24.7136  # Riyadh, Saudi Arabia
    base_lon = 46.6753
    base_alt = 612.0   # Approximate elevation of Riyadh
    
    # For indoor surveying, we'll create a small-scale movement pattern
    # that simulates walking around a room
    
    # Parameters for indoor movement simulation
    room_width = 20.0  # meters
    room_length = 30.0  # meters
    
    # Create a path that moves around the perimeter of the room
    path_points = []
    
    # Number of points to generate (one for each frame plus some extras)
    num_points = len(frames) + 10
    
    # Generate points along a rectangular path
    for i in range(num_points):
        # Normalize i to [0, 1] range representing progress along the path
        t = (i % num_points) / num_points
        
        # Determine which side of the rectangle we're on
        if t < 0.25:
            # Bottom side (moving right)
            x = t * 4 * room_width
            y = 0
        elif t < 0.5:
            # Right side (moving up)
            x = room_width
            y = (t - 0.25) * 4 * room_length
        elif t < 0.75:
            # Top side (moving left)
            x = room_width - (t - 0.5) * 4 * room_width
            y = room_length
        else:
            # Left side (moving down)
            x = 0
            y = room_length - (t - 0.75) * 4 * room_length
        
        # Convert local coordinates to lat/lon
        # Approximate conversion (not geodetically accurate but sufficient for simulation)
        lat_offset = y / 111111.0  # 1 degree latitude is approximately 111111 meters
        lon_offset = x / (111111.0 * math.cos(math.radians(base_lat)))  # Adjust for latitude
        
        lat = base_lat + lat_offset
        lon = base_lon + lon_offset
        
        # Add some noise to simulate GNSS measurement errors
        lat += random.gauss(0, 0.0000005)  # ~5-10cm noise
        lon += random.gauss(0, 0.0000005)
        alt = base_alt + random.gauss(0, 0.05)  # 5cm vertical noise
        
        # Add HDOP (horizontal dilution of precision) - higher for indoor environments
        hdop = 2.5 + random.uniform(0, 1.0)
        
        # Add number of satellites (lower for indoor environments)
        satellites = max(4, int(random.gauss(6, 1)))
        
        # Get timestamp from corresponding frame or generate one
        if i < len(frames):
            timestamp = parser.parse(frames[i]['timestamp'])
        else:
            # For extra points, add time beyond the last frame
            last_time = parser.parse(frames[-1]['timestamp'])
            timestamp = last_time + timedelta(seconds=(i - len(frames) + 1) * config.get_config('video', 'frame_interval'))
        
        point = {
            'timestamp': timestamp.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'hdop': hdop,
            'satellites': satellites
        }
        
        path_points.append(point)
    
    # Save sample data to CSV
    output_dir = config.get_config('output', 'directory')
    os.makedirs(output_dir, exist_ok=True)
    
    sample_gnss_file = os.path.join(output_dir, 'sample_gnss_data.csv')
    
    with open(sample_gnss_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'latitude', 'longitude', 'altitude', 'hdop', 'satellites']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for point in path_points:
            writer.writerow(point)
    
    logger.info(f"Generated sample GNSS data with {len(path_points)} points, saved to {sample_gnss_file}")
    
    # Update config with the sample data file path
    config.update_config('gnss', 'file_path', sample_gnss_file)
    config.update_config('gnss', 'format', 'csv')
    
    return path_points

def import_gnss_data(config):
    """Import GNSS data from file."""
    gnss_file = config.get_config('gnss', 'file_path')
    gnss_format = config.get_config('gnss', 'format')
    
    if not gnss_file or not os.path.exists(gnss_file):
        logger.warning(f"GNSS data file not found: {gnss_file}")
        return []
    
    gnss_data = []
    
    try:
        if gnss_format.lower() == 'csv':
            with open(gnss_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure required fields are present
                    if all(key in row for key in ['timestamp', 'latitude', 'longitude']):
                        # Convert string values to appropriate types
                        point = {
                            'timestamp': row['timestamp'],
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'altitude': float(row.get('altitude', 0)),
                            'hdop': float(row.get('hdop', 1.0)),
                            'satellites': int(row.get('satellites', 0))
                        }
                        gnss_data.append(point)
        
        elif gnss_format.lower() == 'gpx':
            # Simple GPX parsing (for more complex GPX files, consider using a dedicated library)
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(gnss_file)
            root = tree.getroot()
            
            # Define namespace if present in the GPX file
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'} if root.tag.startswith('{') else {}
            
            # Find track points
            xpath = './/gpx:trkpt' if ns else './/trkpt'
            for trkpt in root.findall(xpath, ns):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                
                # Find time element
                time_elem = trkpt.find('.//gpx:time' if ns else './/time', ns)
                if time_elem is not None and time_elem.text:
                    timestamp = parser.parse(time_elem.text).isoformat()
                else:
                    continue  # Skip points without timestamp
                
                # Find elevation element
                ele_elem = trkpt.find('.//gpx:ele' if ns else './/ele', ns)
                altitude = float(ele_elem.text) if ele_elem is not None and ele_elem.text else 0
                
                # Find satellite and hdop elements if available
                sat_elem = trkpt.find('.//gpx:sat' if ns else './/sat', ns)
                satellites = int(sat_elem.text) if sat_elem is not None and sat_elem.text else 0
                
                hdop_elem = trkpt.find('.//gpx:hdop' if ns else './/hdop', ns)
                hdop = float(hdop_elem.text) if hdop_elem is not None and hdop_elem.text else 1.0
                
                point = {
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': altitude,
                    'hdop': hdop,
                    'satellites': satellites
                }
                gnss_data.append(point)
        
        elif gnss_format.lower() == 'nmea':
            # Basic NMEA parsing (for more complex parsing, consider using a dedicated library)
            from datetime import datetime, timedelta
            
            # Initialize variables to store NMEA sentence data
            current_date = None
            current_time = None
            current_lat = None
            current_lon = None
            current_alt = None
            current_hdop = None
            current_satellites = None
            
            with open(gnss_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith('$'):
                        continue
                    
                    # Split the NMEA sentence into fields
                    fields = line.split(',')
                    if len(fields) < 2:
                        continue
                    
                    sentence_type = fields[0]
                    
                    # Process different NMEA sentence types
                    if sentence_type == '$GPRMC' or sentence_type == '$GNRMC':
                        # RMC sentence contains time, date, position
                        if len(fields) >= 10 and fields[2] == 'A':  # 'A' means data valid
                            # Extract time
                            time_str = fields[1]
                            if len(time_str) >= 6:
                                current_time = time_str[:6]  # HHMMSS
                            
                            # Extract date
                            date_str = fields[9]
                            if len(date_str) == 6:
                                current_date = date_str  # DDMMYY
                            
                            # Extract position
                            if fields[3] and fields[5]:
                                lat_str = fields[3]
                                lat_dir = fields[4]
                                lon_str = fields[5]
                                lon_dir = fields[6]
                                
                                # Convert NMEA format to decimal degrees
                                try:
                                    lat_deg = float(lat_str[:2])
                                    lat_min = float(lat_str[2:])
                                    current_lat = lat_deg + lat_min/60.0
                                    if lat_dir == 'S':
                                        current_lat = -current_lat
                                    
                                    lon_deg = float(lon_str[:3])
                                    lon_min = float(lon_str[3:])
                                    current_lon = lon_deg + lon_min/60.0
                                    if lon_dir == 'W':
                                        current_lon = -current_lon
                                except ValueError:
                                    pass
                    
                    elif sentence_type == '$GPGGA' or sentence_type == '$GNGGA':
                        # GGA sentence contains time, position, fix quality, satellites, HDOP, altitude
                        if len(fields) >= 14 and fields[6] != '0':  # Fix quality > 0 means valid fix
                            # Extract time
                            time_str = fields[1]
                            if len(time_str) >= 6:
                                current_time = time_str[:6]  # HHMMSS
                            
                            # Extract position
                            if fields[2] and fields[4]:
                                lat_str = fields[2]
                                lat_dir = fields[3]
                                lon_str = fields[4]
                                lon_dir = fields[5]
                                
                                # Convert NMEA format to decimal degrees
                                try:
                                    lat_deg = float(lat_str[:2])
                                    lat_min = float(lat_str[2:])
                                    current_lat = lat_deg + lat_min/60.0
                                    if lat_dir == 'S':
                                        current_lat = -current_lat
                                    
                                    lon_deg = float(lon_str[:3])
                                    lon_min = float(lon_str[3:])
                                    current_lon = lon_deg + lon_min/60.0
                                    if lon_dir == 'W':
                                        current_lon = -current_lon
                                except ValueError:
                                    pass
                            
                            # Extract satellites, HDOP, altitude
                            try:
                                current_satellites = int(fields[7]) if fields[7] else 0
                                current_hdop = float(fields[8]) if fields[8] else 1.0
                                current_alt = float(fields[9]) if fields[9] else 0.0
                            except ValueError:
                                pass
                    
                    # If we have all the necessary data, create a GNSS point
                    if current_date and current_time and current_lat is not None and current_lon is not None:
                        try:
                            # Parse date and time
                            day = int(current_date[:2])
                            month = int(current_date[2:4])
                            year = 2000 + int(current_date[4:6])  # Assuming 21st century
                            
                            hour = int(current_time[:2])
                            minute = int(current_time[2:4])
                            second = int(current_time[4:6])
                            
                            timestamp = datetime(year, month, day, hour, minute, second).isoformat()
                            
                            point = {
                                'timestamp': timestamp,
                                'latitude': current_lat,
                                'longitude': current_lon,
                                'altitude': current_alt if current_alt is not None else 0.0,
                                'hdop': current_hdop if current_hdop is not None else 1.0,
                                'satellites': current_satellites if current_satellites is not None else 0
                            }
                            
                            gnss_data.append(point)
                            
                            # Reset data for next point
                            current_date = None
                            current_time = None
                            current_lat = None
                            current_lon = None
                            current_alt = None
                            current_hdop = None
                            current_satellites = None
                            
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing NMEA data: {e}")
        
        else:
            logger.error(f"Unsupported GNSS data format: {gnss_format}")
            return []
        
    except Exception as e:
        logger.error(f"Error importing GNSS data: {e}")
        return []
    
    # Sort GNSS data by timestamp
    gnss_data.sort(key=lambda x: x['timestamp'])
    
    logger.info(f"Imported {len(gnss_data)} GNSS data points")
    return gnss_data

def calculate_camera_position(gnss_position, config):
    """Calculate camera position based on GNSS position and camera offsets."""
    # Extract GNSS coordinates
    lat = gnss_position['latitude']
    lon = gnss_position['longitude']
    alt = gnss_position['altitude']
    
    # Get camera offsets
    north_offset = config.get_config('camera', 'offset_north')
    east_offset = config.get_config('camera', 'offset_east')
    up_offset = config.get_config('camera', 'offset_up')
    heading_offset = config.get_config('camera', 'heading_offset')
    
    # Convert heading offset from degrees to radians
    heading_rad = math.radians(heading_offset)
    
    # Apply heading rotation to horizontal offsets
    rotated_north = north_offset * math.cos(heading_rad) - east_offset * math.sin(heading_rad)
    rotated_east = north_offset * math.sin(heading_rad) + east_offset * math.cos(heading_rad)
    
    # Earth radius in meters
    # Earth radius in meters
    earth_radius = 6378137.0
    
    # Calculate new position using Haversine formula (more accurate for surveying)
    # Convert offsets from meters to radians
    north_rad = rotated_north / earth_radius
    east_rad = rotated_east / (earth_radius * math.cos(math.radians(lat)))
    
    # Apply offsets to get new position
    new_lat = lat + math.degrees(north_rad)
    new_lon = lon + math.degrees(east_rad)
    new_alt = alt + up_offset
    
    # For technical surveying, calculate error estimates
    # HDOP-based horizontal error estimate (in meters)
    if 'hdop' in gnss_position:
        h_error = gnss_position['hdop'] * 2.5  # Typical GNSS accuracy factor
    else:
        h_error = 2.5  # Default error estimate
    
    # Vertical error estimate (typically 1.5x horizontal error)
    v_error = h_error * 1.5
    
    # For indoor environments, increase error estimates
    if config.get_config('advanced', 'indoor_positioning'):
        indoor_factor = 2.0  # Indoor environments have higher error
        h_error *= indoor_factor
        v_error *= indoor_factor
    
    # Return camera position with error estimates
    return {
        'latitude': new_lat,
        'longitude': new_lon,
        'altitude': new_alt,
        'horizontal_error': h_error,
        'vertical_error': v_error,
        'timestamp': gnss_position['timestamp']
    }

def match_frames_with_gnss(frames, gnss_data, config):
    """Match video frames with GNSS data based on timestamps."""
    if not frames or not gnss_data:
        logger.error("No frames or GNSS data to match")
        return []
    
    # Extract timestamps from frames and GNSS data
    frame_times = [parser.parse(frame['timestamp']) for frame in frames]
    gnss_times = [parser.parse(point['timestamp']) for point in gnss_data]
    
    # Apply time offset from configuration
    time_offset = config.get_config('gnss', 'time_offset')
    if time_offset != 0:
        frame_times = [t + timedelta(seconds=time_offset) for t in frame_times]
    
    # Convert timestamps to seconds since epoch for interpolation
    frame_seconds = [(t - datetime(1970, 1, 1)).total_seconds() for t in frame_times]
    gnss_seconds = [(t - datetime(1970, 1, 1)).total_seconds() for t in gnss_times]
    
    # Extract GNSS coordinates
    latitudes = [point['latitude'] for point in gnss_data]
    longitudes = [point['longitude'] for point in gnss_data]
    altitudes = [point['altitude'] for point in gnss_data]
    
    # Check if GNSS data covers the frame timestamps
    if min(frame_seconds) < min(gnss_seconds) or max(frame_seconds) > max(gnss_seconds):
        logger.warning("Some frames are outside the GNSS data time range")
    
    # Choose interpolation method
    method = config.get_config('gnss', 'interpolation_method')
    if method not in ['linear', 'cubic', 'nearest']:
        logger.warning(f"Invalid interpolation method: {method}, using linear")
        method = 'linear'
    
    # For cubic interpolation, we need at least 4 points
    if method == 'cubic' and len(gnss_data) < 4:
        logger.warning("Not enough GNSS points for cubic interpolation, using linear")
        method = 'linear'
    
    # Create interpolation functions
    try:
        f_lat = interp1d(gnss_seconds, latitudes, kind=method, bounds_error=False, fill_value="extrapolate")
        f_lon = interp1d(gnss_seconds, longitudes, kind=method, bounds_error=False, fill_value="extrapolate")
        f_alt = interp1d(gnss_seconds, altitudes, kind=method, bounds_error=False, fill_value="extrapolate")
        
        # If HDOP and satellites are available, interpolate them too
        if 'hdop' in gnss_data[0] and 'satellites' in gnss_data[0]:
            hdops = [point.get('hdop', 1.0) for point in gnss_data]
            satellites = [point.get('satellites', 0) for point in gnss_data]
            f_hdop = interp1d(gnss_seconds, hdops, kind='linear', bounds_error=False, fill_value="extrapolate")
            f_sat = interp1d(gnss_seconds, satellites, kind='nearest', bounds_error=False, fill_value="extrapolate")
        else:
            f_hdop = None
            f_sat = None
    except Exception as e:
        logger.error(f"Error creating interpolation functions: {e}")
        return []
    
    # Match frames with interpolated GNSS data
    matched_frames = []
    
    for i, frame in enumerate(frames):
        frame_time = frame_seconds[i]
        
        # Interpolate GNSS coordinates at frame time
        try:
            lat = float(f_lat(frame_time))
            lon = float(f_lon(frame_time))
            alt = float(f_alt(frame_time))
            
            # Interpolate HDOP and satellites if available
            hdop = float(f_hdop(frame_time)) if f_hdop is not None else 1.0
            satellites = int(round(float(f_sat(frame_time)))) if f_sat is not None else 0
            
            # Create GNSS position object
            gnss_position = {
                'timestamp': frame['timestamp'],
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'hdop': hdop,
                'satellites': satellites
            }
            
            # Calculate camera position
            camera_position = calculate_camera_position(gnss_position, config)
            
            # Create matched frame object
            matched_frame = {
                'frame': frame,
                'gnss_position': gnss_position,
                'camera_position': camera_position
            }
            
            matched_frames.append(matched_frame)
            
        except Exception as e:
            logger.warning(f"Error matching frame {i}: {e}")
    
    logger.info(f"Matched {len(matched_frames)} frames with GNSS data")
    return matched_frames

def add_georeference_to_image(image_path, position, config):
    """Add georeference information to image metadata."""
    try:
        # Use exiftool to add GPS metadata
        lat = position['latitude']
        lon = position['longitude']
        alt = position['altitude']
        
        # Convert decimal degrees to degrees, minutes, seconds
        lat_ref = 'N' if lat >= 0 else 'S'
        lon_ref = 'E' if lon >= 0 else 'W'
        
        lat = abs(lat)
        lon = abs(lon)
        
        lat_deg = int(lat)
        lat_min = int((lat - lat_deg) * 60)
        lat_sec = (lat - lat_deg - lat_min/60) * 3600
        
        lon_deg = int(lon)
        lon_min = int((lon - lon_deg) * 60)
        lon_sec = (lon - lon_deg - lon_min/60) * 3600
        
        # Prepare exiftool command
        cmd = [
            'exiftool',
            f'-GPSLatitudeRef={lat_ref}',
            f'-GPSLatitude={lat_deg} {lat_min} {lat_sec}',
            f'-GPSLongitudeRef={lon_ref}',
            f'-GPSLongitude={lon_deg} {lon_min} {lon_sec}',
            f'-GPSAltitude={alt}',
            '-overwrite_original',
            image_path
        ]
        
        # Add error estimates if available
        if 'horizontal_error' in position:
            cmd.insert(-1, f'-GPSHPositioningError={position["horizontal_error"]}')
        
        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Error adding georeference to image: {result.stderr}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error adding georeference to image: {e}")
        return False

def add_visual_marker_to_image(image_path, position, output_path=None):
    """Add visual marker with coordinates to image."""
    try:
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Format coordinates
        lat = position['latitude']
        lon = position['longitude']
        alt = position['altitude']
        
        coord_text = f"Lat: {lat:.6f}° Lon: {lon:.6f}° Alt: {alt:.2f}m"
        
        # Add error estimates if available
        if 'horizontal_error' in position and 'vertical_error' in position:
            h_err = position['horizontal_error']
            v_err = position['vertical_error']
            error_text = f"H.Err: ±{h_err:.2f}m V.Err: ±{v_err:.2f}m"
        else:
            error_text = ""
        
        # Add timestamp if available
        if 'timestamp' in position:
            time_text = f"Time: {position['timestamp']}"
        else:
            time_text = ""
        
        # Draw semi-transparent background for text
        img_width, img_height = img.size
        text_bg_height = 80 if error_text else 60
        draw.rectangle([(0, img_height - text_bg_height), (img_width, img_height)], fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((10, img_height - text_bg_height + 10), coord_text, fill=(255, 255, 255), font=font)
        
        if error_text:
            draw.text((10, img_height - text_bg_height + 35), error_text, fill=(255, 255, 255), font=font)
        
        if time_text:
            draw.text((10, img_height - text_bg_height + 60), time_text, fill=(255, 255, 255), font=font)
        
        # Save image
        if output_path is None:
            output_path = image_path.replace('.jpg', '_marked.jpg')
        
        img.save(output_path)
        return output_path
    
    except Exception as e:
        logger.error(f"Error adding visual marker to image: {e}")
        return None

def process_matched_frame(matched_frame, config):
    """Process a matched frame by adding georeference and visual marker."""
    frame = matched_frame['frame']
    camera_position = matched_frame['camera_position']
    
    # Get output directory
    output_dir = os.path.join(config.get_config('output', 'directory'), 'processed_frames')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    base_filename = os.path.basename(frame['path'])
    georef_path = os.path.join(output_dir, base_filename)
    marked_path = os.path.join(output_dir, base_filename.replace('.jpg', '_marked.jpg'))
    
    # Copy original frame to output directory
    try:
        import shutil
        shutil.copy2(frame['path'], georef_path)
    except Exception as e:
        logger.error(f"Error copying frame: {e}")
        return None
    
    # Add georeference to image if enabled
    if config.get_config('output', 'add_coordinates_to_images'):
        add_georeference_to_image(georef_path, camera_position, config)
    
    # Add visual marker to image
    marked_image_path = add_visual_marker_to_image(georef_path, camera_position, marked_path)
    
    # Return processed frame information
    return {
        'original_path': frame['path'],
        'georeferenced_path': georef_path,
        'marked_path': marked_image_path,
        'position': camera_position,
        'quality': frame['quality'],
        'timestamp': frame['timestamp']
    }

def generate_survey_report(processed_frames, config):
    """Generate a survey report with frame information and map."""
    if not processed_frames:
        logger.error("No processed frames to generate report")
        return None
    
    # Get output directory
    output_dir = config.get_config('output', 'directory')
    report_dir = os.path.join(output_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Create report HTML file
    report_path = os.path.join(report_dir, 'survey_report.html')
    
    # Extract coordinates for map
    coordinates = []
    for frame in processed_frames:
        position = frame['position']
        coordinates.append([position['latitude'], position['longitude']])
    
    # Create map centered on the mean of coordinates
    mean_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    mean_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
    
    map_html = folium.Map(location=[mean_lat, mean_lon], zoom_start=18).get_root().render()
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Survey Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .map-container {{ height: 500px; margin-bottom: 20px; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
            .image-item {{ margin-bottom: 20px; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Technical Survey Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Survey Information</h2>
        <table>
            <tr><th>Total Frames</th><td>{len(processed_frames)}</td></tr>
            <tr><th>Survey Area</th><td>Riyadh, Hittin District (Indoor Theater)</td></tr>
            <tr><th>Camera</th><td>iPhone 16 Pro Max (24mm, f/1.78)</td></tr>
            <tr><th>Resolution</th><td>{config.get_config('video', 'resolution')}</td></tr>
            <tr><th>Frame Rate</th><td>{config.get_config('video', 'frame_rate')} FPS</td></tr>
            <tr><th>Pole Height</th><td>{config.get_config('camera', 'offset_up')} meters</td></tr>
        </table>
        
        <h2>Survey Map</h2>
        <div class="map-container">
            {map_html}
        </div>
        
        <h2>Frame Information</h2>
        <table>
            <tr>
                <th>Frame</th>
                <th>Timestamp</th>
                <th>Latitude</th>
                <th>Longitude</th>
                <th>Altitude</th>
                <th>H.Error</th>
                <th>Quality</th>
            </tr>
    """
    
    # Add rows for each frame
    for i, frame in enumerate(processed_frames):
        position = frame['position']
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{frame['timestamp']}</td>
                <td>{position['latitude']:.8f}°</td>
                <td>{position['longitude']:.8f}°</td>
                <td>{position['altitude']:.2f} m</td>
                <td>±{position.get('horizontal_error', 'N/A')} m</td>
                <td>{frame['quality']:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Processed Images</h2>
        <div class="image-grid">
    """
    
    # Add images
    for i, frame in enumerate(processed_frames):
        if 'marked_path' in frame and frame['marked_path']:
            # Get relative path for HTML
            rel_path = os.path.relpath(frame['marked_path'], report_dir)
            html_content += f"""
            <div class="image-item">
                <h3>Frame {i+1}</h3>
                <img src="{rel_path}" alt="Frame {i+1}">
                <p>Lat: {frame['position']['latitude']:.6f}° Lon: {frame['position']['longitude']:.6f}°</p>
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Generate KML file if enabled
    if config.get_config('output', 'generate_kml'):
        kml_path = os.path.join(report_dir, 'survey_points.kml')
        generate_kml(processed_frames, kml_path)
    
    # Generate CSV file with coordinates
    csv_path = os.path.join(report_dir, 'survey_points.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'timestamp', 'latitude', 'longitude', 'altitude', 'horizontal_error', 'quality']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, frame in enumerate(processed_frames):
            position = frame['position']
            writer.writerow({
                'frame': i+1,
                'timestamp': frame['timestamp'],
                'latitude': position['latitude'],
                'longitude': position['longitude'],
                'altitude': position['altitude'],
                'horizontal_error': position.get('horizontal_error', 'N/A'),
                'quality': frame['quality']
            })
    
    logger.info(f"Survey report generated at {report_path}")
    return report_path

def generate_kml(processed_frames, output_path):
    """Generate KML file from processed frames."""
    try:
        # Create KML header
        kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Survey Points</name>
    <Style id="frameStyle">
        <IconStyle>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
            </Icon>
        </IconStyle>
    </Style>
"""
        
        # Add placemarks for each frame
        for i, frame in enumerate(processed_frames):
            position = frame['position']
            
            # Get relative path for image link
            image_path = frame.get('marked_path', frame.get('georeferenced_path', ''))
            if image_path:
                image_path = os.path.abspath(image_path)
                image_link = f'<a href="file://{image_path}">View Image</a>'
            else:
                image_link = 'No image available'
            
            kml += f"""
    <Placemark>
        <name>Frame {i+1}</name>
        <description>
            <![CDATA[
            Timestamp: {frame['timestamp']}<br/>
            Latitude: {position['latitude']:.8f}°<br/>
            Longitude: {position['longitude']:.8f}°<br/>
            Altitude: {position['altitude']:.2f} m<br/>
            Horizontal Error: ±{position.get('horizontal_error', 'N/A')} m<br/>
            Quality: {frame['quality']:.2f}<br/>
            {image_link}
            ]]>
        </description>
        <styleUrl>#frameStyle</styleUrl>
        <Point>
            <coordinates>{position['longitude']},{position['latitude']},{position['altitude']}</coordinates>
        </Point>
    </Placemark>"""
        
        # Add path connecting all points
        kml += """
    <Placemark>
        <name>Survey Path</name>
        <LineString>
            <tessellate>1</tessellate>
            <coordinates>"""
        
        for frame in processed_frames:
            position = frame['position']
            kml += f"\n                {position['longitude']},{position['latitude']},{position['altitude']}"
        
        kml += """
            </coordinates>
        </LineString>
    </Placemark>
</Document>
</kml>"""
        
        # Write KML to file
        with open(output_path, 'w') as f:
            f.write(kml)
        
        logger.info(f"KML file generated at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating KML file: {e}")
        return False

def process_video_with_gnss(config):
    """Process video with GNSS data to create georeferenced images."""
    # Extract frames from video
    frames = extract_frames_from_video(config)
    if not frames:
        logger.error("No frames extracted from video")
        return None
    
    # Import or generate GNSS data
    if config.get_config('gnss', 'use_sample_data'):
        gnss_data = generate_sample_gnss_data(frames, config)
    else:
        gnss_data = import_gnss_data(config)
    
    if not gnss_data:
        logger.error("No GNSS data available")
        return None
    
    # Match frames with GNSS data
    matched_frames = match_frames_with_gnss(frames, gnss_data, config)
    if not matched_frames:
        logger.error("No frames matched with GNSS data")
        return None
    
    # Process matched frames
    processed_frames = []
    for matched_frame in tqdm(matched_frames, desc="Processing frames"):
        processed_frame = process_matched_frame(matched_frame, config)
        if processed_frame:
            processed_frames.append(processed_frame)
    
    if not processed_frames:
        logger.error("No frames processed successfully")
        return None
    
    # Generate report
    if config.get_config('output', 'generate_report'):
        report_path = generate_survey_report(processed_frames, config)
    else:
        report_path = None
    
    return {
        'frames': processed_frames,
        'report': report_path
    }

def run_in_colab():
    """Run the survey tool interactively in Google Colab."""
    from google.colab import files
    import ipywidgets as widgets
    from IPython.display import display, HTML
    
    print("Welcome to the Technical Surveying Tool for iPhone 16 Pro Max")
    print("This tool will process your video and GNSS data to create georeferenced images.")
    
    # Create configuration
    config = SurveyConfig()
    
    # Set up output directory
    output_dir = "survey_output"
    os.makedirs(output_dir, exist_ok=True)
    config.update_config('output', 'directory', output_dir)
    
    # Ask user if they want to use sample data
    use_sample = input("Do you want to use sample GNSS data? (y/n): ").lower() == 'y'
    config.update_config('gnss', 'use_sample_data', use_sample)
    
    if not use_sample:
        print("Please upload your GNSS data file (CSV, GPX, or NMEA format):")
        uploaded = files.upload()
        
        if not uploaded:
            print("No GNSS file uploaded, using sample data instead.")
            config.update_config('gnss', 'use_sample_data', True)
        else:
            gnss_file = list(uploaded.keys())[0]
            gnss_path = os.path.join(output_dir, gnss_file)
            
            # Save uploaded file
            with open(gnss_path, 'wb') as f:
                f.write(uploaded[gnss_file])
            
            config.update_config('gnss', 'file_path', gnss_path)
            
            # Determine format from file extension
            if gnss_file.lower().endswith('.csv'):
                config.update_config('gnss', 'format', 'csv')
            elif gnss_file.lower().endswith('.gpx'):
                config.update_config('gnss', 'format', 'gpx')
            elif gnss_file.lower().endswith('.nmea') or gnss_file.lower().endswith('.txt'):
                config.update_config('gnss', 'format', 'nmea')
            else:
                print(f"Unknown file format for {gnss_file}, assuming CSV.")
                config.update_config('gnss', 'format', 'csv')
    
    # Ask for camera pole height
    pole_height = float(input("Enter the camera pole height in meters (default: 1.5): ") or "1.5")
    config.update_config('camera', 'offset_up', pole_height)
    
    # Ask for frame extraction interval
    frame_interval = float(input("Enter frame extraction interval in seconds (default: 0.5): ") or "0.5")
    config.update_config('video', 'frame_interval', frame_interval)
    
    # Ask for indoor positioning mode
    indoor_mode = input("Is this an indoor survey? (y/n, default: y): ").lower() != 'n'
    config.update_config('advanced', 'indoor_positioning', indoor_mode)
    
    # Upload video file
    print("Please upload your iPhone 16 Pro Max video file:")
    uploaded = files.upload()
    
    if not uploaded:
        print("No video file uploaded, exiting.")
        return
    
    video_file = list(uploaded.keys())[0]
    video_path = os.path.join(output_dir, video_file)
    
    # Save uploaded file
    with open(video_path, 'wb') as f:
        f.write(uploaded[video_file])
    
    config.update_config('video', 'file_path', video_path)
    
    # Process video with GNSS data
    print("Processing video with GNSS data...")
    result = process_video_with_gnss(config)
    
    if result:
        print(f"Processing complete! {len(result['frames'])} frames processed.")
        if result['report']:
            print(f"Report generated at: {result['report']}")
            
            # Display download links for processed files
            print("\nDownload processed files:")
            
            # Create download links for report files
            report_dir = os.path.dirname(result['report'])
            for filename in os.listdir(report_dir):
                file_path = os.path.join(report_dir, filename)
                if os.path.isfile(file_path):
                    display(HTML(f'<a href="./files/{os.path.relpath(file_path)}" download="{filename}">{filename}</a>'))
            
            # Create download links for processed frames
            frames_dir = os.path.join(output_dir, 'processed_frames')
            if os.path.exists(frames_dir):
                print("\nDownload processed frames:")
                for filename in os.listdir(frames_dir)[:10]:  # Limit to first 10 frames to avoid clutter
                    if filename.endswith('_marked.jpg'):
                        file_path = os.path.join(frames_dir, filename)
                        display(HTML(f'<a href="./files/{os.path.relpath(file_path)}" download="{filename}">{filename}</a>'))
                
                if len(os.listdir(frames_dir)) > 10:
                    print(f"... and {len(os.listdir(frames_dir)) - 10} more frames")
    else:
        print("Processing failed. Check the logs for details.")

# Main function to run the survey
def main():
    """Main function to run the survey."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video with GNSS data for technical surveying.')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--gnss', help='Path to GNSS data file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--sample-data', action='store_true', help='Use sample GNSS data')
    parser.add_argument('--pole-height', type=float, default=1.5, help='Camera pole height in meters')
    parser.add_argument('--frame-interval', type=float, default=0.5, help='Frame extraction interval in seconds')
    parser.add_argument('--indoor', action='store_true', help='Enable indoor positioning mode')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SurveyConfig(args.config)
    
    # Update configuration with command-line arguments
    if args.video:
        config.update_config('video', 'file_path', args.video)
    
    if args.gnss:
        config.update_config('gnss', 'file_path', args.gnss)
        
        # Determine format from file extension
        if args.gnss.lower().endswith('.csv'):
            config.update_config('gnss', 'format', 'csv')
        elif args.gnss.lower().endswith('.gpx'):
            config.update_config('gnss', 'format', 'gpx')
        elif args.gnss.lower().endswith('.nmea') or args.gnss.lower().endswith('.txt'):
            config.update_config('gnss', 'format', 'nmea')
    
    if args.output:
        config.update_config('output', 'directory', args.output)
    
    if args.sample_data:
        config.update_config('gnss', 'use_sample_data', True)
    
    config.update_config('camera', 'offset_up', args.pole_height)
    config.update_config('video', 'frame_interval', args.frame_interval)
    config.update_config('advanced', 'indoor_positioning', args.indoor)

    # Create output directory
    output_dir = config.get_config('output', 'directory')
    os.makedirs(output_dir, exist_ok=True)

    # Check if video file is specified
    video_path = config.get_config('video', 'file_path')
    if not video_path:
        logger.error("No video file specified")
        return

    # Process video with GNSS data
    logger.info("Starting video processing with GNSS data")
    result = process_video_with_gnss(config)

    if result:
        logger.info(f"Processing complete! {len(result['frames'])} frames processed.")
        if result['report']:
            logger.info(f"Report generated at: {result['report']}")
    else:
        logger.error("Processing failed")

# For Google Colab compatibility
def run_video_gnss_survey(video_path, use_sample_data=True, pole_height=1.5, frame_interval=0.5, indoor_mode=True):
    """Run the survey with specified parameters."""
    # Create configuration
    config = SurveyConfig()

    # Set up output directory
    output_dir = "survey_output"
    os.makedirs(output_dir, exist_ok=True)
    config.update_config('output', 'directory', output_dir)

    # Update configuration
    config.update_config('video', 'file_path', video_path)
    config.update_config('gnss', 'use_sample_data', use_sample_data)
    config.update_config('camera', 'offset_up', pole_height)
    config.update_config('video', 'frame_interval', frame_interval)
    config.update_config('advanced', 'indoor_positioning', indoor_mode)

    # Process video with GNSS data
    logger.info("Starting video processing with GNSS data")
    result = process_video_with_gnss(config)

    if result:
        logger.info(f"Processing complete! {len(result['frames'])} frames processed.")
        if result['report']:
            logger.info(f"Report generated at: {result['report']}")
            return result['report']
    else:
        logger.error("Processing failed")
        return None

# Execute the main function if script is run directly
if __name__ == "__main__":
    # Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # Run interactive Colab version
        run_in_colab()
    else:
        # Run command-line version
        main()
