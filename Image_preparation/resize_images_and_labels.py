from PIL import Image

def resize_images_and_labels(image_paths, label_paths, H, W):
    """
    Resizes and saves corresponding image and label (mask) files with matching filenames. The
    starting filenames must be matching to use this function.

    This function iterates through a list of image files and their corresponding label (mask) files, 
    resizes them to a specified height and width, and saves them in separate directories. The function 
    ensures that the image filenames and the label filenames remain consistent, facilitating the proper 
    pairing of images and their associated segmentation masks.

    Image size affects computation time and credit usage in LandingLens. This function allows you
    to resize images and their masks. LandingLens segmentation projects requires that images and their 
    masks be the same size and have the same filename. See https://support.landing.ai/docs/upload-labeled-images-seg

    Args:
        image_paths (list of str): A list of file paths to the input images.
        label_paths (list of str): A list of file paths to the corresponding label (mask) images.
        H (int): The height to resize the images and labels to.
        W (int): The width to resize the images and labels to.
    
    Returns:
        None: This function does not return any values. It saves the resized images 
        and labels in 'images/' and 'labels/' directories respectively.
    
    File Output:
        - Resized images are saved in the 'images/' directory with filenames in the format `img_<i>.png`.
        - Resized labels are saved in the 'labels/' directory with matching filenames `img_<i>.png`, 
            where `<i>` is the index of the image-label pair.

    Notes:
        - The labels (masks) are resized using nearest-neighbor interpolation (`Image.NEAREST`), 
            which is appropriate for segmentation tasks as it preserves the class values in the mask.
        - Ensure that the `images/` and `labels/` directories exist before running this function, 
            or modify the code to create them if they don't exist.

    Example:
        If `image_paths` contains ['image1.jpg', 'image2.jpg'] and `label_paths` contains 
        ['image1.jpg', 'image2.jpg'], and both images and labels are resized to 256x512, the output will be:
            - 'images/image1_256x512.png', 'images/image2_256x512.png'
            - 'labels/image1_256x512.png', 'labels/image2_256x512.png'

    """
    
     # Ensure the 'images/' and 'labels/' directories exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    
    # Loop through each image and its corresponding label
    for f_img, f_lbl in zip(image_paths, label_paths):
        # Open the image and label files
        img = Image.open(f_img).resize((H, W))  # Resize image
        lbl = Image.open(f_lbl).resize((H, W), Image.NEAREST)  # Resize label (using nearest neighbor)
        
        # Extract the base filename without the directory or extension
        base_name_img = os.path.splitext(os.path.basename(f_img))[0]
        base_name_lbl = os.path.splitext(os.path.basename(f_lbl))[0]
        
        # Construct the new filename with dimensions included
        img_filename = f"{base_name_img}_{H}x{W}.png"
        lbl_filename = f"{base_name_lbl}_{H}x{W}.png"
        
        # Save the resized image and label
        img.save(os.path.join("images", img_filename))
        lbl.save(os.path.join("labels", lbl_filename))