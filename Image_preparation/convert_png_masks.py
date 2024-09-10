import os
import shutil
import cv2

def convert_png_masks(mask_path: str, new_path: str, new_value: int) -> None:
    """
    The function converts all masks in a folder (path) converts it to 
    Landing AI-compatible format (8-bit grayscale). More information available at
    https://support.landing.ai/landinglens/docs/upload-labeled-images-seg#segmentation-mask
    
    This function iterates through all PNG files in the specified mask directory, processes 
    each mask by replacing non-zero pixel values with the provided `new_value`, and stores 
    the modified masks in the specified output directory. It handles both grayscale and 
    color (RGB/RGBA) masks, ensuring proper processing depending on the mask type.

    Args:
        mask_path (str): The directory containing the original PNG masks.
        new_path (str): The directory where the modified PNG masks will be saved.
        new_value (int): The value that will replace all non-zero pixel values in the masks.

    Returns:
        None: This function does not return any value. It performs operations in-place and 
        saves the results to the `new_path`.

    Raises:
        FileNotFoundError: If the `mask_path` does not exist.
        PermissionError: If the function does not have permission to read/write the directories.

    Example:
        >>> convert_png_masks("/path/to/original_masks", "/path/to/modified_masks", 255)

    Notes:
        - Grayscale masks are 2D arrays where non-zero values are replaced with `new_value`.
        - RGB masks (3 channels) replace all non-black pixels with `new_value`.
        - RGBA masks (4 channels) replace all non-fully-transparent pixels with `new_value`.
        - If `new_path` exists, it is purged before creating new masks.
    """
    # Checks if the new path exists and purges it if necessary
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)

    # Iterate over all PNG files in the mask directory
    for filename in os.listdir(mask_path):
        if filename.endswith(".png"):
            # Full path to the mask file
            file_path = os.path.join(mask_path, filename)
            # Load the image mask
            mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Process grayscale masks
            if len(mask.shape) == 2:
                mask[mask != 0] = new_value
            # Process RGB or RGBA masks
            elif len(mask.shape) == 3:
                if mask.shape[2] == 3:
                    # RGB mask: replace non-black pixels
                    mask[(mask != [0, 0, 0]).any(axis=-1)] = new_value
                elif mask.shape[2] == 4:
                    # RGBA mask: replace non-fully-transparent pixels
                    mask[(mask != [0, 0, 0, 0]).any(axis=-1)] = new_value

            # Save the modified mask to the new directory
            cv2.imwrite(os.path.join(new_path, filename), mask)
