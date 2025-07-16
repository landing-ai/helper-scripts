from PIL import Image
import os

def convert_jfif_to_jpeg(path: str) -> None:
    """
    Convert all .jfif files in adirectory and its subdirectories to .jpeg format,
    removing any alpha channel (transparency) if present. This function
    walks through the provided directory, finds any files with a .jfif 
    extension, converts them to .jpeg, removes the original .jfif file, and removes any 
    transparency if the image contains an alpha channel.

    LandingLens does not support working with .jfif files directly. Conversion to
    supported image format is required. See https://support.landing.ai/docs/upload-images

    Args:
        path (str): The directory path to search for .jfif files.

    Returns:
        None: This function doesn't return any value. It performs in-place conversions
        and removes the original files.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        PermissionError: If the function does not have permission to read or write files 
        in the directory.
        OSError: If any issues arise during file removal or writing.

    Example:
        >>> convert_jfif_to_jpeg('/path/to/directory')
        Converted and removed: /path/to/directory/image1.jfif
        Converted and removed: /path/to/directory/subdir/image2.jfif
    
    Note:
        The function converts images with an alpha channel (e.g., RGBA, LA) to RGB 
        before saving as JPEG, as the JPEG format does not support transparency.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jfif'):
                file_path = os.path.join(root, file)
                # Open the .jfif image
                img = Image.open(file_path)
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                # Define the new file name with the .jpeg extension
                jpeg_path = file_path.rsplit('.', 1)[0] + '.jpeg'
                # Save the image as .jpeg
                img.save(jpeg_path, "JPEG")
                # Remove the original .jfif file
                os.remove(file_path)
                print(f"Converted and removed: {file_path}")
