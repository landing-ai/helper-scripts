import os
from PIL import Image

def transparent_to_solid(path="./MyProject/Images", fill_color=(255, 255, 255)):
    """
    Convert all images in a directpry with transparency to RGB mode, and save them with a specified background color.

    LandingLens expects all image pixels to be a color (not transparent). 

    Args:
        path (str): Path to the directory containing images. Default is './MyProject/Images'.
        fill_color (tuple): The background color to use for replacing transparency, in RGB format. 
            The default is white (255, 255, 255).

    Description:
        This function iterates through all files in a specified directory and its subdirectories.
        For each image, it checks if the image has an alpha channel (i.e., transparency).
        If transparency is found, it replaces the transparent areas with a solid background color 
        and saves the image in RGB format, effectively removing the alpha channel.

    Returns:
        None: The function saves the converted images back to the directory with their original filenames, 
        overwriting the existing files.

    Example:
        convert_images_to_rgb("./images", fill_color=(255, 255, 255))  # Replaces transparency with white background.
        convert_images_to_rgb("./images", fill_color=(0, 0, 0))  # Replaces transparency with black background.
    """

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # Full path of the image file
            image_path = os.path.join(root, name)
            print(image_path)  # Optional: Print the path of the file being processed

            # Open the image
            im = Image.open(image_path)

            # Convert the image to RGBA mode (which includes an alpha channel)
            im = im.convert("RGBA")
            
            # If the image has transparency (alpha channel), replace it with the specified background color
            if im.mode in ('RGBA', 'LA'):  # LA is for grayscale with alpha
                # Create a new background image with the specified color
                background = Image.new(im.mode[:-1], im.size, fill_color)
                # Paste the original image on top of the background, using the alpha channel as a mask
                background.paste(im, im.split()[-1])  # Omit the transparency
                im = background  # Replace the original image with the new one
            
            # Save the image in RGB mode, which removes any alpha/transparency
            im.convert("RGB").save(image_path)
