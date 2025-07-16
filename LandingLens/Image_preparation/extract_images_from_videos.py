import cv2
import os

def extract_images_from_videos(input_directory, output_directory, frame_rate = 1):
    """
    Extract frames from video files in a directory at a specified frame rate and save them as JPG images.
    Each frame is named as "video_filename_frame_{frame_number}.jpg", where `video_filename` is the name of the video 
    without the file extension and saved in a corresponding subdirectory within the output directory.

    LandingLens requires image files (not video) for model training and inference. See
    https://support.landing.ai/docs/upload-images

    Args:
        input_directory (str): Path to the directory containing video files (e.g., .mp4 or .avi).
        output_directory (str): Path to the directory where extracted frames will be saved.
        frame_rate (int): The number of frames to extract per second from the video (e.g., 1 frame per second).

    Returns:
        None: The function saves the extracted frames as images in the specified output directory.

    Example Usage:
        extract_frames_from_directory('./input_dir/', './output_dir/', 1)
        # Extracts 1 frame per second from all videos in './input_dir/' and saves them to './output_dir/'.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a video file and get the path to each file
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            input_path = os.path.join(input_directory, filename)

            # Get the video filename without the extension
            video_name = os.path.splitext(filename)[0]

            # Create a subdirectory for each video to store its frames
            video_output_folder = os.path.join(output_directory, video_name)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)

            # Open the video file and get the frame rate
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

            # Calculate the interval between frames to match the desired frame rate
            frame_interval = int(fps / frame_rate)

            # Initialize frame counter
            count = 0

            # Loop through video frames
            while cap.isOpened():
                ret, frame = cap.read()

                # If no more frames are returned, exit the loop
                if not ret:
                    break

                # Save every nth frame according to the frame interval
                if count % frame_interval == 0:
                    # Save the frame with the video filename in the image name
                    output_path = os.path.join(video_output_folder, f"{video_name}_frame_{count}.jpg")
                    cv2.imwrite(output_path, frame)

                # Increment the frame counter
                count += 1

            # Release the video capture object
            cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()




