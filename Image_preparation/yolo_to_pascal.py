"""
Module Name: YOLO-to-VOC Annotation Converter

Description:
    This script converts YOLO format annotation files (.txt) to Pascal VOC format annotation 
    files (.xml). It processes a set of images and their corresponding YOLO-style annotations, 
    converts bounding box coordinates from normalized format (xywhn) to absolute coordinates 
    (xyxy), and saves the results as XML files compatible with the Pascal VOC dataset format.

Usage:
    The script can be executed with or without command-line arguments. It accepts the following arguments:
    
    - anno_path (str): Path to the directory containing YOLO annotation files in .txt format.
    - save_path (str): Path where the converted Pascal VOC .xml files will be saved.
    - image_path (str): Path to the directory containing the corresponding image files.

    To run the script, you can provide the paths using command-line arguments, or it will default 
    to predefined paths.

    Example command:
        $ python script_name.py --anno-path ./data/labels/yolo --save-path ./data/convert/voc --image-path ./data/images

Parameters:
    - anno_path (str): Path to the YOLO annotations in .txt format (default: './data/labels/yolo').
    - save_path (str): Directory where the Pascal VOC .xml files will be saved (default: './data/convert/voc').
    - image_path (str): Directory containing the image files (default: './data/images').

Outputs:
    The script will generate XML files corresponding to each image with YOLO annotations, 
    saved in the specified save directory. It also prints the total number of images processed, 
    the number of unique categories, and the total number of bounding boxes.

Dependencies:
    - Python 3.x
    - OpenCV (cv2)
    - lxml
    - tqdm

    Install the required packages via pip:
        $ pip install opencv-python lxml tqdm

Author:
    Landing AI

Date:
    2024-09-09

"""


import argparse
import os
import sys
import shutil
import cv2
from lxml import etree, objectify
from tqdm import tqdm


def save_anno_to_xml(filename, size, objs, save_path):
    """
    Save object annotation data to an XML file in Pascal VOC format.

    This function generates an XML file containing annotations for objects 
    in an image, including the size of the image and bounding boxes for 
    each object, and saves it to the specified path.

    Parameters:
    ----------
    filename : str
        The name of the image file (e.g., 'image.jpg').
    size : tuple
        A tuple representing the size of the image as (height, width, depth).
    objs : list of tuples
        A list of objects to annotate, where each object is represented as a 
        tuple: (class_name, (xmin, ymin, xmax, ymax)). Each object specifies 
        the class name and the bounding box coordinates.
    save_path : str
        The directory where the generated XML file will be saved.

    Returns:
    -------
    None
        This function saves the XML file to the specified path but does not 
        return anything.

    Raises:
    -------
    OSError:
        If there is an error writing to the specified path.

    """

    # Create the root XML tree using ElementMaker for the annotation.
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder("DATA"),
        E.filename(filename),
        E.source(
            E.database("The VOC Database"),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size[1]), # Width of the image
            E.height(size[0]), # Height of the image
            E.depth(size[2]) # Depth of the image (e.g., 3 for RGB)
        ),
        E.segmented(0) # Set 'segmented' to 0 (no segmentation by default)
    )

    # Iterate over each object in the list and add its annotation to the XML tree.
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]), # Object class name (e.g., 'cat', 'dog')
            E.pose("Unspecified"), # Pose of the object (can be extended if needed)
            E.truncated(0), # Truncated flag (0 means the object is not truncated)
            E.difficult(0), # Difficulty flag (0 means the object is not difficult to detect)
            E.bndbox(
                E.xmin(obj[1][0]), # Bounding box left x-coordinate (xmin)
                E.ymin(obj[1][1]), # Bounding box top y-coordinate (ymin)
                E.xmax(obj[1][2]), # Bounding box right x-coordinate (xmax)
                E.ymax(obj[1][3]) # Bounding box bottom y-coordinate (ymax)
            )
        )
        # Append the object annotation to the main annotation tree.
        anno_tree.append(anno_tree2)

    # Define the full file path where the XML will be saved, replacing the image extension with '.xml'.    
    anno_path = os.path.join(save_path, filename[:-3] + "xml")

    # Write the XML tree to the specified path with pretty formatting.
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)


def xywhn2xyxy(bbox, size):
    """
    Convert a bounding box from normalized (x_center, y_center, width, height) 
    format to absolute (xmin, ymin, xmax, ymax) format.

    This function takes a bounding box in normalized coordinates, where the box is 
    represented by its center point (x_center, y_center) and its width and height, 
    and converts it to absolute pixel coordinates based on the image size.

    Parameters:
    ----------
    bbox : list or tuple
        A list or tuple representing the bounding box in normalized coordinates as 
        (x_center, y_center, width, height), where all values are relative to the image 
        size (ranging from 0 to 1).
    size : list or tuple
        A list or tuple representing the image size as (height, width).

    Returns:
    -------
    list
        A list of integers representing the bounding box in absolute pixel coordinates 
        as [xmin, ymin, xmax, ymax].
    
    Example:
    --------
    >>> xywhn2xyxy([0.5, 0.5, 0.2, 0.4], [1000, 2000])
    [900, 300, 1100, 700]
    
    This converts a normalized bounding box (centered at 50% width, 50% height, 
    with width 20% and height 40%) into pixel coordinates based on a 1000x2000 image.  
    """

    # Convert bbox and size to floats to ensure precision for calculations.
    bbox = list(map(float, bbox))
    size = list(map(float, size))

    # Compute xmin and ymin: the top-left corner of the bounding box.
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]

    # Compute xmax and ymax: the bottom-right corner of the bounding box.
    xmax = (bbox[0] + bbox[2] / 2.) * size[1]
    ymax = (bbox[1] + bbox[3] / 2.) * size[0]

    # Create the box as [xmin, ymin, xmax, ymax] and convert to integers.
    box = [xmin, ymin, xmax, ymax]

    # Return the bounding box coordinates as integers.
    return list(map(int, box))


def parseXmlFilse(image_path, anno_path, save_path):
    """
    Parse image and annotation files, convert bounding box coordinates,
    and save them in XML format for object detection datasets.

    Args:
        image_path (str): Path to the directory containing image files.
        anno_path (str): Path to the directory containing annotation files.
        save_path (str): Path where the parsed XML files will be saved.

    Global Variables:
        images_nums (int): Total number of image files processed.
        category_nums (int): Total number of unique categories.
        bbox_nums (int): Total number of bounding boxes processed.

    Raises:
        AssertionError: If the image_path or anno_path does not exist.

    Description:
        This function parses a set of images and their corresponding annotations. It reads
        the image files from the image_path and annotation files from the anno_path. It
        matches the annotation with the corresponding image, converts bounding box
        coordinates from normalized format (xywhn) to absolute coordinates (xyxy), and
        saves the results in XML format to the specified save_path.
    """
    # Declare global variables to track totals
    global images_nums, category_nums, bbox_nums

    # Check if image and annotation paths exist
    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)

    # Remove the existing save directory if it exists and create a new one
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # Initialize category list and load class names from 'classes.txt'
    category_set = []
    with open(anno_path + '/classes.txt', 'r') as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_nums = len(category_set) # Count the number of unique categories
    
    # Create a dictionary that maps class indices to category names
    category_id = dict((k, v) for k, v in enumerate(category_set))

    # Get a list of images and annotations
    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path)]
    
    # Create a dictionary mapping image file names (without extension) to their index
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    images_nums = len(images)

    # Loop through all annotation files
    for file in tqdm(files):
        # Skip non-txt files or the 'classes.txt' file
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        
        # Check if the annotation corresponds to an existing image
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img = cv2.imread(images[index]) # Read the corresponding image
            shape = img.shape # Get image dimensions (height, width, channels)
            filename = images[index].split(os.sep)[-1] # Get the image file name
        else:
            continue

        objects = []
        # Open the annotation file and read each bounding box entry
        with open(file, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split() # Split the line by whitespace
                category = int(i[0]) # Extract the category (class index)
                category_name = category_id[category] # Get the category name
                bbox = xywhn2xyxy((i[1], i[2], i[3], i[4]), shape) # Convert bounding box format
                obj = [category_name, bbox] # Store the object as [category, bbox]
                objects.append(obj) # Add the object to the list
        
        # Update the total number of bounding boxes processed
        bbox_nums += len(objects)
        
        # Save the annotations and bounding boxes to an XML file
        save_anno_to_xml(filename, shape, objects, save_path)


import argparse
import sys

# Initialize global counters
images_nums = 0
category_nums = 0
bbox_nums = 0

if __name__ == '__main__':
    """
    Script Description:
        This script converts YOLO format annotation files (.txt) to VOC format annotation files (.xml).
        
    Parameters:
        - anno_path (str): Path to the directory containing YOLO annotation files in .txt format.
        - save_path (str): Directory where the converted VOC .xml files will be saved.
        - image_path (str): Directory containing the corresponding image files.

    Usage:
        The script can be run with or without command-line arguments.
        If no arguments are provided, default paths will be used for image and annotation files.

    Example:
        python script_name.py --anno-path ./data/labels/yolo --save-path ./data/convert/voc --image-path ./data/images
    """
    
    # Set up the argument parser for command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='./data/labels/yolo', help='yolo txt path')
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/voc', help='xml save path')
    parser.add_argument('--image-path', default='./data/images')

    # Parse the command-line arguments
    opt = parser.parse_args()

    # If arguments are provided via the command line, use them
    if len(sys.argv) > 1:
        print(opt) # Print the parsed options for debugging
        parseXmlFilse(**vars(opt)) # Call the function using the parsed arguments
        # Print global counters showing the total processed images, categories, and bounding boxes
        print("image nums: {}".format(images_nums))
        print("category nums: {}".format(category_nums))
        print("bbox nums: {}".format(bbox_nums))
    else:
        # Default paths if no command-line arguments are provided
        anno_path = './data/labels/yolo'
        save_path = './data/convert/voc1'
        image_path = './data/images'

        # Call the function with the default paths
        parseXmlFilse(image_path, anno_path, save_path)

        # Print the global counters after processing
        print("image nums: {}".format(images_nums))
        print("category nums: {}".format(category_nums))
        print("bbox nums: {}".format(bbox_nums))
