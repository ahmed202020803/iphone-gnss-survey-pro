#!/usr/bin/env python3
"""
Basic usage example for iPhone GNSS Survey Pro.
This example demonstrates how to process a video file with sample GNSS data.
"""

import os
import sys
import logging

# Add parent directory to path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iphone_gnss_survey import SurveyConfig, process_video_with_gnss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run a basic survey example."""
    # Create a configuration
    config = SurveyConfig()
    
    # Set up output directory
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    config.update_config('output', 'directory', output_dir)
    
    # Use sample GNSS data (no real video/GNSS files needed)
    config.update_config('gnss', 'use_sample_data', True)
    
    # Set camera pole height
    config.update_config('camera', 'offset_up', 1.5)
    
    # Set frame extraction interval
    config.update_config('video', 'frame_interval', 0.5)
    
    # Enable indoor positioning mode
    config.update_config('advanced', 'indoor_positioning', True)
    
    # Set a dummy video path (will be ignored when using sample data)
    config.update_config('video', 'file_path', 'dummy_video.mp4')
    
    # Process video with GNSS data
    logger.info("Starting video processing with sample GNSS data")
    result = process_video_with_gnss(config)
    
    if result:
        logger.info(f"Processing complete! {len(result['frames'])} frames processed.")
        if result['report']:
            logger.info(f"Report generated at: {result['report']}")
    else:
        logger.error("Processing failed")

if __name__ == "__main__":
    main()