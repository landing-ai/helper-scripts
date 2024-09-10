import os
import json
from PIL import Image

def crop_images_json(json_directory: str) -> None:
    """
    Processes images based on cropping information from JSON files and saves 
    cropped images in label-specific directories.

    This function iterates over JSON files in the specified directory, reads cropping 
    coordinates from each JSON file, and uses them to crop the associated image. The 
    cropped images are saved into folders named after the labels specified in the JSON 
    files, organized within a 'cropped_images' folder.

    Args:
        json_directory (str): The directory containing JSON files and corresponding images. 
                              Each JSON file must contain image metadata, crop coordinates, 
                              and labels.

    Returns:
        None: This function doesn't return any value. It processes images and saves 
        the cropped output into new directories.

    Raises:
        FileNotFoundError: If the JSON file references an image that does not exist.
        OSError: If there are issues reading or writing the image files or directories.

    Example:
        >>> process_json_images("/path/to/json_directory")
        Cropped image saved to /path/to/json_directory/cropped_images/defect1/image1_cropped.jpg
        Cropped image saved to /path/to/json_directory/cropped_images/defect2/image2_cropped.jpg
    
    Notes:
        - Each JSON file must have the key 'shapes', which contains a list of objects 
          with 'points' (cropping coordinates) and 'label' (defect or object name).
        - Images are cropped using the specified coordinates and stored in directories 
          named after the label within 'cropped_images'.
        - The 'points' in the JSON must contain two coordinates: 
          the top-left (x, y) and bottom-right (x, y) corners of the cropping region.
    """
    # Iterate through each JSON file in the specified directory
    for file_name in os.listdir(json_directory):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(json_directory, file_name)
            
            # Open and load the JSON file
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            # Process each shape in the JSON data
            for shape in data['shapes']:
                image_path = os.path.join(json_directory, data['imagePath'])
                crop_points = shape['points']
                label = shape['label']

                # Error handling for image files that do not exist
                if not os.path.exists(image_path):
                    print(f"Image file does not exist: {image_path}")
                    continue

                # Load and crop the image
                image = Image.open(image_path)
                left, top = crop_points[0]
                right, bottom = crop_points[1]
                cropped_image = image.crop((left, top, right, bottom))

                # Create a directory named after the defect label if it doesn't exist
                output_dir = os.path.join(json_directory, 'cropped_images', label)
                os.makedirs(output_dir, exist_ok=True)

                # Save the cropped image in the designated folder
                image_file_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file_name = f"{image_file_name}_cropped.jpg"
                output_path = os.path.join(output_dir, output_file_name)
                cropped_image.save(output_path)

                print(f'Cropped image saved to {output_path}')
