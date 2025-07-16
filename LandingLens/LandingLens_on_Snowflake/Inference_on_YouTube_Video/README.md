# Image Inference Pipeline Script

## Overview
The Python script in this folder is designed to capture images from a live YouTube stream, perform object detection using a LandingLens (LandingAI) model, and store the prediction results along with the images in a Snowflake database. The workflow includes retrieving a live stream URL, capturing snapshots via `ffmpeg`, executing model inference on each image, and then uploading the images and inserting prediction details into Snowflake.

## Video and Blog Content
There is a video on the LandingAI YouTube channel showing exactly how to use this script and demonstrating the full pipeline from camera feed to dashboard. 
https://www.youtube.com/watch?v=JyI9r3U2HSg

There is a Medium blog post that explains how this scripts fits into a larger solution. It is named [From CCTV to Insights — How Snowflake Customers Can Build Visual AI](https://medium.com/snowflake/from-cctv-to-insights-how-snowflake-customers-can-build-visual-ai-6d2d2813f070)

## Features
- **Live Stream Capture:** Retrieves the best stream URL from a YouTube video using `streamlink` and captures snapshots using `ffmpeg`.
- **Image Processing:** Saves captured images locally with timestamp-based filenames. Images are processed using the Python Imaging Library (PIL).
- **Inference Integration:** Uses a LandingLens (LandingAI) model to perform real-time object detection on captured images.
- **Snowflake Integration:** 
  - Connects securely to a Snowflake database using JWT authentication.
  - Uploads images to a specified Snowflake stage.
  - Inserts prediction results (object labels, confidence scores, bounding boxes, timestamps) into a Snowflake table.
- **Logging:** Implements detailed logging (timestamps, levels, messages) to facilitate troubleshooting.

## Prerequisites
- **Python Version:** Python 3.7 or later.
- **Python Libraries:**
  - Standard: `os`, `re`, `json`, `time`, `logging`, `subprocess`, `datetime`
  - Third-party: `Pillow` (PIL), `snowflake-connector-python`, and `landingai`
- **External Tools:**
  - [Streamlink](https://streamlink.github.io/) – to extract the live stream URL.
  - [FFmpeg](https://ffmpeg.org/) – to capture still images from the stream.

## Installation
1. **Python & Dependencies:**
   - Ensure Python 3.7+ is installed.
   - Install the necessary Python packages:
     ```bash
     pip install pillow snowflake-connector-python landingai
     ```
2. **External Tools:**
   - Install `streamlink` and `ffmpeg` on your system as per their installation instructions.
3. **Snowflake Credentials:**
   - Confirm that you have access to a valid Snowflake account, user credentials, and a RSA private key file for authentication.

## Configuration
Edit the script to customize your environment settings:
- **Logging Setup:** Configured to output timestamps and log levels.
- **Snowflake Settings:**
  - **Account and User Information:** Set variables like `snowflake_account_locator`, `snowflake_user`, and `snowflake_account_identifier`.
  - **Private Key:** Ensure the RSA private key file exists at the given path (`private_key_file`). The script reads this file to authenticate using JWT.
  - **Warehouse, Database, and Schema:** These are defined by the `warehouse`, `database`, and `schema` variables.
- **LandingLens Inference:**
  - Configure the `endpoint_id` for the LandingLens model.
  - Instantiate the `SnowflakeNativeAppPredictor` with the endpoint and Snowflake credentials.
- **Stream and File Settings:**
  - **YouTube URL:** The `youtube_url` variable is the source YouTube stream.
  - **Image Dimensions:** The image width and corresponding height (maintaining a 16:9 ratio) are set.
  - **Iterations & Sleep:** Define the number of iterations (`iterations`) to run and the delay between iterations (`sleep_seconds`).
  - **Local Storage:** Images are saved to an `Inference_Images` directory.
  - **Snowflake Stage and Table:** Specify the stage (`stage_name`) for file uploads and the table (`table_name`) for inserting object detection data.

## Script Workflow
1. **Initialization:**
   - Set up logging to track the script’s execution.
   - Validate the presence of the private key file.
   - Create a connection to Snowflake using the defined connection parameters.
2. **Stream Setup and Inference Preparation:**
   - Retrieve the live stream URL from YouTube using `streamlink` (via the `get_live_stream_url` utility function).
   - Create a local directory for storing captured images if it does not already exist.
3. **Main Inference Loop:**
   - **Iteration Process:** For each loop (as defined by the `iterations` variable):
     - **Image Capture:** Capture a single frame from the live stream using `ffmpeg` (via the `capture_and_save_image` function) and save it with a timestamp-based filename.
     - **Image Loading:** Open the saved image using PIL. The filename is parsed with a regex to extract the timestamp.
     - **Inference:** Perform object detection on the image using the LandingLens predictor.
     - **Snowflake Upload:**
       - **File Upload:** Use a Snowflake `PUT` command to upload the image file to the designated stage.
       - **Data Insertion:** For each detected object, construct and execute an SQL INSERT statement to store details such as the label, confidence, bounding box coordinates, and detection time.
     - **Pause:** Wait for a defined interval before processing the next iteration.
4. **Cleanup:**
   - Close the Snowflake cursor and connection.
   - Log the completion time of the inference loop.

## Usage Instructions
1. **Configure the Script:**  
   Edit the configuration variables (Snowflake credentials, endpoint ID, private key path, YouTube stream URL, etc.) to match your environment.
2. **Run the Script:**  
   Execute the script from the command line:
   ```bash
   python your_script_name.py
