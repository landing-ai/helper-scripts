import os
import subprocess
import streamlit as st
import cv2
from PIL import Image
import glob
import requests
from pathlib import Path
from PIL import ImageDraw, ImageFont
from collections import Counter
import time
import altair as alt
import pandas as pd
import json

# ----- Original functions -----

def download_video_yt_dlp(youtube_url, video_save_directory):
    os.makedirs(video_save_directory, exist_ok=True)
    output_template = os.path.join(video_save_directory, 'downloaded_video.%(ext)s')
    command = ['yt-dlp', 
               '-o', output_template, 
               '--cookies-from-browser', 'chrome',
               '--recode-video', 'mp4', 
               youtube_url]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"yt-dlp failed with error: {e}")
        return None

    video_file = os.path.join(video_save_directory, 'downloaded_video.mp4')
    return video_file if os.path.exists(video_file) else None

def extract_frames_from_youtube(youtube_url, total_frames, video_save_directory, frames_save_directory, width):
    os.makedirs(video_save_directory, exist_ok=True)
    os.makedirs(frames_save_directory, exist_ok=True)

    video_file_path = download_video_yt_dlp(youtube_url, video_save_directory)
    if video_file_path is None:
        st.error("Video download failed; aborting frame extraction.")
        return

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_video_frames / video_fps

    st.write(f"Video duration: {duration_sec:.2f} seconds")

    # Frame indexes to capture (evenly spaced)
    target_indexes = [int(i * total_video_frames / total_frames) for i in range(total_frames)]

    frame_index = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in target_indexes:
            new_height = int(width * 9 / 16)
            resized_frame = cv2.resize(frame, (width, new_height))
            output_filename = os.path.join(frames_save_directory, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_filename, resized_frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    st.success(f"Extracted {saved_count} total frames.")

def call_agentic_api_on_frames(frames_dir, api_key, prompt, json_save_dir, throttle_delay):
    """
    Sends frames to the API and saves the raw JSON responses, with status updates.
    
    Args:
        frames_dir (str): Folder of images to send.
        api_key (str): LandingLens API key.
        prompt (str): Object detection prompt.
        json_save_dir (str): Folder to save raw JSON results.
        throttle_delay (int): Delay in seconds between API calls.
    
    Returns:
        dict: Mapping of filename → response JSON.
    """
    os.makedirs(json_save_dir, exist_ok=True)
    results = {}
    url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    headers = {"Authorization": f"Basic {api_key}"}
    image_files = sorted(Path(frames_dir).glob("*.jpg"))
    
    # Set up a progress bar and a status message placeholder.
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    total_files = len(image_files)
    for idx, img_path in enumerate(image_files):
        status_placeholder.write(f"Sending frame: **{img_path.name}** ({idx+1} of {total_files})")
        with open(img_path, "rb") as image_file:
            files = {"image": image_file}
            data = {"prompts": prompt, "model": "agentic"}
            try:
                response = requests.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                response_json = response.json()
                results[img_path.name] = response_json

                # Save JSON response to file with same base filename as the image.
                json_filepath = os.path.join(json_save_dir, img_path.stem + ".json")
                with open(json_filepath, "w") as f:
                    json.dump(response_json, f, indent=2)
            except Exception as e:
                results[img_path.name] = {"error": str(e)}
        
        # Update progress.
        progress_bar.progress((idx + 1) / total_files)

        # Countdown until the next frame.
        for sec in range(throttle_delay, 0, -1):
            status_placeholder.write(f"Waiting {sec} seconds before sending the next frame...")
            time.sleep(1)
    
    status_placeholder.write("All frames processed.")
    return results

def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on an image using detection data.

    Args:
        image (PIL.Image): Original image.
        detections (list): List of detection dicts. Expects each dict to have 'bounding_box' and 'label'.
    
    Returns:
        image: Annotated image.
        dict: Count of objects by label.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    label_counter = Counter()

    for det in detections:
        if "bounding_box" in det:
            label = det.get("label", "unknown")
            box = det["bounding_box"]
            label_counter[label] += 1

            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), label, fill="red", font=font)

    return image, dict(label_counter)


# ----- Streamlit App -----

st.set_page_config(layout="wide")
st.title("Agentic Object Detection on YouTube Video Frames by LandingAI")

tab1, tab2, tab3 = st.tabs(["Select Video", "Frames and Objects", "Summary Metrics"])

with tab1:
    st.header("User Inputs")

    # Button to clear history and reset for new inputs
    if st.button("Clear History and Reset"):
        st.session_state.pop("detections", None)
        # Increase the run counter (start at 1 if not set)
        if "run_count" not in st.session_state:
            st.session_state["run_count"] = 2
        else:
            st.session_state["run_count"] += 1
        st.success(f"History cleared. New run will use subfolder run{st.session_state['run_count']}.")


    youtube_url = st.text_input("YouTube Video URL")
    prompt = st.text_input("Prompt for Agentic Object Detection", value="Enter the specific object to find here.")
    total_frames = st.number_input("How many total frames to extract?", min_value=1, value=10)
    width = st.number_input("Frame Width (px)", min_value=100, value=640)
    video_dir = st.text_input("Video Save Directory", value="downloaded_video")
    frames_dir = st.text_input("Frames Save Directory", value="frames")
    json_save_dir = st.text_input("JSON Response Save Directory", value="json_responses") 
    api_key = st.text_input("LandingLens API Key (Base64)", type="password")
    throttle_delay = st.number_input("API Throttling Delay (seconds)", min_value=1, value=10)
    
    run_detection = st.checkbox("Run Object Detection after Frame Extraction") 

    # Download & Extract Frames button.
    if st.button("Download & Extract Frames"):
        # Use run_count for unique subdirectories; default to 1 if not set.
        run_count = st.session_state.get("run_count", 1)
        run_label = f"run{run_count}"
        # Create unique subdirectories.
        video_subdir = os.path.join(video_dir, run_label)
        frames_subdir = os.path.join(frames_dir, run_label)
        json_subdir = os.path.join(json_save_dir, run_label)

        extract_frames_from_youtube(youtube_url, total_frames, video_subdir, frames_subdir, width)
        st.success("Extraction complete!")
        if run_detection:
            st.info("Running object detection on frames...")
            st.session_state["detections"] = call_agentic_api_on_frames(
                frames_subdir, api_key, prompt, json_subdir, throttle_delay
            )
            st.success("Detection complete!")
        
        # Store the current run's frames directory for Tab 2 to use.
        st.session_state["current_frames_subdir"] = frames_subdir

with tab2:
    st.header("Extracted Frames with Overlays")
    
    current_frames_dir = st.session_state.get("current_frames_subdir", None)
    if current_frames_dir and os.path.exists(current_frames_dir):
        frame_files = sorted(glob.glob(os.path.join(current_frames_dir, '*.jpg')))
        detections_dict = st.session_state.get("detections", {})

        if frame_files:
            for frame_path in frame_files:
                frame_name = os.path.basename(frame_path)
                image = Image.open(frame_path).convert("RGB")

                col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
                # Column 1: Raw frame.
                with col1:
                    st.image(image, caption=frame_name, use_column_width=True)
                # Column 2: JSON response.
                with col2:
                    if frame_name in detections_dict:
                        st.json(detections_dict[frame_name])
                    else:
                        st.write("No detection result.")
                # Column 3: Overlay with bounding boxes.
                with col3:
                    if frame_name in detections_dict:
                        response = detections_dict[frame_name]
                        if isinstance(response, dict) and "data" in response and isinstance(response["data"], list):
                            frame_detections = response["data"][0]
                        else:
                            frame_detections = []
                        overlay_img, counts = draw_bounding_boxes(image, frame_detections)
                        st.image(overlay_img, caption="Overlay", use_column_width=True)
                    else:
                        st.image(image, caption="(overlay placeholder)", use_column_width=True)
                        counts = {}
                # Column 4: Label count.
                with col4:
                    if counts:
                        st.write(counts)
                    else:
                        st.write("No counts.")
        else:
            st.info("No frames found in the current run. Please extract frames in Tab 1.")
    else:
        st.info("No current run frames directory found. Please run a new extraction in Tab 1.")

with tab3:
    st.header("Counts Summary")

    detections_dict = st.session_state.get("detections", {})
    frame_counts = []

    for frame_name, response in detections_dict.items():
        # Safely extract detection data
        if isinstance(response, dict) and "data" in response and isinstance(response["data"], list):
            frame_detections = response["data"][0]
            count = len(frame_detections)
        else:
            count = 0

        frame_counts.append({"frame": frame_name, "object_count": count})

    df = pd.DataFrame(frame_counts)

    if df.empty:
        st.info("No detection data to summarize. Run detection in Tab 1.")
    else:
        # 1. Metric: Total objects detected
        total_objects = int(df["object_count"].sum())

        # 2. Metric: % of frames with at least one object
        frames_with_objects = df[df["object_count"] > 0].shape[0]
        total_frames = df.shape[0]
        percent_with_objects = (frames_with_objects / total_frames) * 100

        # 3. Metric: Avg objects across all frames
        avg_objects_all = df["object_count"].mean()

        # 4. Metric: Avg objects where objects > 0
        avg_objects_nonzero = df[df["object_count"] > 0]["object_count"].mean() if frames_with_objects else 0

        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🧮 Total Objects", f"{total_objects}")
        col2.metric("📸 % Frames w/ Objects", f"{percent_with_objects:.1f}%")
        col3.metric("📊 Avg Objects (All Frames)", f"{avg_objects_all:.2f}")
        col4.metric("📈 Avg Objects (Non-Zero)", f"{avg_objects_nonzero:.2f}")

        # Bar chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("frame:N", title="Frame", sort=None),
            y=alt.Y("object_count:Q", title="Detected Objects"),
            tooltip=["frame", "object_count"]
        ).properties(
            width=700,
            height=400,
            title="Objects Detected per Frame"
        )
        st.altair_chart(chart, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Frame Count CSV",
            data=csv,
            file_name="object_counts_by_frame.csv",
            mime="text/csv"
        )
