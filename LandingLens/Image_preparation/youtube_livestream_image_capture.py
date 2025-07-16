import os
import schedule
import time
from datetime import datetime
import subprocess
import math

"""
YouTube Live Stream Frame Capture Script

This script captures images from a live YouTube stream at a user-defined interval, resizes the images 
to a custom width with a 16:9 aspect ratio, and saves them in a specified directory. 

The script uses the following tools and libraries:
- `streamlink`: Extracts the direct stream URL from the YouTube live stream.
- `ffmpeg`: Captures frames from the live stream and resizes them to the specified resolution.
- `schedule`: Manages the timing for capturing images at regular intervals.

Main Features:
--------------
1. **Live Stream URL Extraction**: Fetches the best available stream URL from a YouTube live stream.
2. **Image Capture**: Captures an image from the live stream at user-specified intervals (in images per hour).
3. **Resolution Control**: Allows the user to specify the width of the captured image, automatically calculating the height based on a 16:9 aspect ratio.
4. **Custom Save Directory**: Saves the captured images to a user-specified directory, creating the directory if it doesn't exist.
5. **Scheduled Execution**: Captures images at regular intervals using the `schedule` library.

User Inputs:
------------
1. YouTube live stream URL.
2. Number of images to capture per hour.
3. Directory to save the images.
4. Image width (in pixels) for the captured frames (height is auto-calculated to maintain a 16:9 aspect ratio).

Example Usage:
--------------
Run the script from the command line and follow the prompts for input:
    $ python capture_images.py

You will be asked to provide:
1. The YouTube live stream URL.
2. The number of images per hour.
3. The directory to save the captured images.
4. The width (in pixels) for the images.

Once running, the script will continuously capture images from the stream at the defined interval until stopped.

Requirements:
-------------
- Python 3.x
- streamlink (`pip install streamlink` or `brew install streamlink`)
- ffmpeg (`brew install ffmpeg`)
- schedule (`pip install schedule`)
- OpenCV (`pip install opencv-python`) for video capture (if needed)

Note:
-----
To stop the script, use `Ctrl + C` in the terminal.
"""


# Function to get the live stream URL using Streamlink
def get_live_stream_url(youtube_url):
    """
    Fetches the direct stream URL of a live YouTube video using the Streamlink tool.

    Parameters:
    -----------
    youtube_url : str
        The URL of the YouTube live stream.

    Returns:
    --------
    str or None
        The direct URL of the live video stream if successful, otherwise None if an error occurs.

    Exceptions:
    -----------
    If an error occurs during the Streamlink command execution, an exception is caught and an error message is printed.

    Workflow:
    ---------
    1. Executes the `streamlink` command to extract the highest quality stream URL from the provided YouTube URL.
    2. If successful, the extracted stream URL is returned.
    3. In case of any exception during the process, the error is caught, and None is returned.

    Example:
    --------
    >>> youtube_url = "https://www.youtube.com/watch?v=example_live_stream_url"
    >>> stream_url = get_live_stream_url(youtube_url)
    >>> if stream_url:
    ...     print("Stream URL fetched successfully!")
    ... else:
    ...     print("Failed to fetch stream URL.")
    """
    try:
        # Run the streamlink command to fetch the best quality stream URL
        command = ['streamlink', youtube_url, 'best', '--stream-url']
        result = subprocess.run(command, stdout=subprocess.PIPE)
        stream_url = result.stdout.decode('utf-8').strip()
        return stream_url
    except Exception as e:
        print(f"Error fetching live stream URL: {e}")
        return None

# Function to capture and save an image using ffmpeg with custom width and 16:9 aspect ratio
def capture_and_save_image(stream_url, save_dir, width):
    """
    Captures and saves an image from a live stream using FFmpeg with a custom width and a 16:9 aspect ratio.

    Parameters:
    -----------
    stream_url : str
        The URL of the live video stream.
    save_dir : str
        The directory where the captured image will be saved.
    width : int
        The width of the captured image in pixels. The height is automatically calculated to maintain a 16:9 aspect ratio.

    Returns:
    --------
    None

    Example:
    --------
    >>> capture_and_save_image("https://example_stream_url", "./images", 1920)

    This will capture a frame from the live stream and save it as a PNG image with a resolution of 1920x1080 in the "./images" directory.
    """
    # Calculate the height based on a 16:9 aspect ratio
    height = int(width * 9 / 16)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = os.path.join(save_dir, f"image_{timestamp}.png")

    # Run ffmpeg command to capture a single frame from the live stream with the given resolution
    command = [
        'ffmpeg', '-i', stream_url, '-vframes', '1', '-q:v', '2',
        '-vf', f'scale={width}:{height}', filename
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Saved {filename} with resolution {width}x{height}")
    except Exception as e:
        print(f"Error capturing image: {e}")

# Schedule the job to run based on the interval
def job(stream_url, save_dir, width):
    capture_and_save_image(stream_url, save_dir, width)

# Start the capture process
def start_scheduled_capture(stream_url, interval, save_dir, width):
    schedule.every(interval).minutes.do(job, stream_url, save_dir, width)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# Main execution
def main(youtube_url, images_per_hour, save_dir, width):
    """
    Main function to capture images from a live YouTube stream at a user-defined interval
    and save them to a specified directory with a custom resolution.

    Parameters:
    -----------
    youtube_url : str
        The URL of the live YouTube stream from which frames will be captured.
    images_per_hour : int
        The number of images to capture per hour. This will determine the interval in minutes between captures.
    save_dir : str
        The directory where the captured images will be saved. If the directory does not exist, it will be created.
    width : int
        The width of the captured images in pixels. The height is calculated automatically to maintain a 16:9 aspect ratio.

    Returns:
    --------
    None

    Workflow:
    ---------
    1. The function ensures that the specified save directory exists or creates it if it does not.
    2. It calculates the interval in minutes between image captures based on the `images_per_hour` value.
    3. The function retrieves the live stream URL using the `streamlink` tool.
    4. It begins capturing images at the specified interval using FFmpeg, saving each image with a unique timestamp in the specified directory.
    
    Example:
    --------
    >>> youtube_url = "https://www.youtube.com/watch?v=example_live_stream_url"
    >>> images_per_hour = 6
    >>> save_dir = "./images"
    >>> width = 1920
    >>> main(youtube_url, images_per_hour, save_dir, width)
    
    This example captures 6 images per hour from the live stream at 1920x1080 resolution and saves them in the './images' directory.
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert the images_per_hour to an interval in minutes
    interval = math.floor(60 / images_per_hour)

    # Get the live stream URL using streamlink
    stream_url = get_live_stream_url(youtube_url)
    if not stream_url:
        print("Could not fetch the live stream URL. Exiting.")
        return

    # Start capturing images at the specified interval with the custom width
    start_scheduled_capture(stream_url, interval, save_dir, width)

# Usage
if __name__ == "__main__":
    youtube_url = input("Enter the YouTube live stream URL: ")
    images_per_hour = int(input("Enter the number of images to capture per hour: "))
    save_dir = input("Enter the directory to save the images (e.g., ./images): ")
    width = int(input("Enter the width of the image in pixels (e.g., 640, 1280, 1920): "))

    main(youtube_url, images_per_hour, save_dir, width)
