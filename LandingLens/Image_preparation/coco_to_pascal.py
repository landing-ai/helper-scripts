
"""
Module Name: COCO to Pascal VOC Converter

Description:
    This module provides functionality to convert COCO-style annotation files (in JSON format) into Pascal VOC format (in XML).
    It can handle individual JSON files or directories containing multiple JSON files. The annotations for object detection, 
    such as bounding boxes and category names, are extracted from the COCO format and saved in the widely-used Pascal VOC format, 
    making it easier to integrate COCO data into pipelines that require VOC-style annotations.

    Use of LandingLens for object detection requires the Pascal VOC format for annotations. 
    See https://support.landing.ai/docs/upload-labeled-images-od. 

Main Functions:
    - catid2name: Converts category IDs to category names from a COCO dataset.
    - save_anno_to_xml: Saves annotation data (e.g., bounding boxes) to Pascal VOC XML format.
    - load_coco: Loads and converts annotations from a COCO dataset to Pascal VOC format.
    - parseJsonFile: Handles the processing of a directory or single JSON file and converts annotations to XML.

Usage:
    This module can be executed as a standalone script using the following command:

        python coco_to_pascal.py --data-dir /path/to/coco_annotations --save-path /path/to/save_xmls

    Where:
    - `data_dir`: The path to the COCO annotations directory or a single JSON file.
    - `xml_save_path`: The directory where the converted Pascal VOC XML files will be saved.

Examples:
    1. Converting a COCO dataset to Pascal VOC XML format:
        python coco_to_pascal.py --data-dir ./data/labels/coco --save-path ./data/convert/voc

    2. Converting a single JSON file:
        python coco_to_pascal.py --data-dir ./data/labels/coco/train.json --save-path ./data/convert/voc

Dependencies:
    - pycocotools: For handling COCO format data.
    - lxml: For generating XML files.
    - tqdm: For displaying progress bars during the conversion process.
"""

from pycocotools.coco import COCO
import os
from lxml import etree, objectify
import shutil
from tqdm import tqdm
import sys
import argparse


def catid2name(coco):
    """
    Convert category IDs to category names from a COCO dataset.

    Args:
        coco (COCO): A COCO dataset object that contains a list of categories.

    Returns:
        dict: A dictionary mapping category IDs to category names.
    
    Description:
        This function extracts the 'id' and 'name' fields from the 'categories' section 
        of a COCO dataset and returns a dictionary where the keys are category IDs and 
        the values are the corresponding category names.
    
    Example:
        Given a COCO dataset with categories, this function will return:
        {1: 'person', 2: 'bicycle', 3: 'car', ...}
    """
    # Initialize an empty dictionary to store category ID to name mapping
    classes = dict()

    # Loop through each category in the COCO dataset and map the category 'id' to its 'name'
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']  # Assign category name to its corresponding ID

    return classes  # Return the dictionary of category IDs and names


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


def load_coco(anno_file, xml_save_path):
    """
    Load annotations from a COCO dataset and convert them to Pascal VOC XML format.

    Args:
        anno_file (str): Path to the COCO annotation JSON file.
        xml_save_path (str): Directory where the XML files will be saved.

    Description:
        This function reads image and annotation data from a COCO dataset, extracts the necessary
        details (image size, category names, and bounding boxes), and converts them to the Pascal VOC 
        XML format. The resulting XML files are saved to the specified directory.

    Example:
        load_coco('annotations/instances_train2017.json', './xml_annotations')

    """
    # If the save directory exists, remove it to start fresh
    if os.path.exists(xml_save_path):
        shutil.rmtree(xml_save_path)
    os.makedirs(xml_save_path)  # Create the save directory

    # Load the COCO dataset
    coco = COCO(anno_file)
    
    # Map category IDs to category names
    classes = catid2name(coco)
    
    # Get all image IDs and category IDs in the dataset
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()

    # Loop through each image in the dataset
    for imgId in tqdm(imgIds):
        size = {}
        
        # Load image information
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']  # Get the filename of the image
        width = img['width']  # Image width
        height = img['height']  # Image height
        size['width'] = width
        size['height'] = height
        size['depth'] = 3  # Assuming depth is 3 (for RGB images)

        # Load all annotations for the current image
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        objs = []  # List to store object details for the image

        # Loop through each annotation for the current image
        for ann in anns:
            object_name = classes[ann['category_id']]  # Get the object category name
            
            # Extract and convert the bounding box from [x, y, width, height] to [xmin, ymin, xmax, ymax]
            bbox = list(map(int, ann['bbox']))
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]  # xmax = x + width
            ymax = bbox[1] + bbox[3]  # ymax = y + height
            
            # Create the object entry
            obj = [object_name, xmin, ymin, xmax, ymax]
            objs.append(obj)  # Add the object to the list of objects for this image

        # Save the image annotations in XML format
        save_anno_to_xml(filename, size, objs, xml_save_path)


def parseJsonFile(data_dir, xmls_save_path):
    """
    Parse COCO JSON files in the specified directory or a single JSON file, converting them to Pascal VOC XML format.

    Args:
        data_dir (str): Path to the COCO data directory or a single annotation JSON file.
        xmls_save_path (str): Path where the generated XML files will be saved.

    Raises:
        AssertionError: If the specified data_dir does not exist.

    Description:
        This function automatically detects all subdirectories in the given directory (data_dir)
        and processes each one by looking for JSON annotation files. It then converts the annotations 
        to Pascal VOC XML format using the `load_coco` function. If a single JSON file is provided, 
        it processes that file directly.

    Example:
        parseJsonFile('./coco_data', './xml_annotations')
        parseJsonFile('./coco_data/instances_custom.json', './xml_annotations/custom')
    """
    
    # Ensure the specified data directory or file exists
    assert os.path.exists(data_dir), "data dir:{} does not exist".format(data_dir)

    # If the input is a directory, automatically detect subdirectories and process any JSON files
    if os.path.isdir(data_dir):
        # Loop through all subdirectories in the data directory
        for sub_dir in os.listdir(data_dir):
            sub_dir_path = os.path.join(data_dir, sub_dir)
            
            # Ensure we're working with subdirectories only
            if os.path.isdir(sub_dir_path):
                # Look for JSON annotation files in each subdirectory
                for file in os.listdir(sub_dir_path):
                    if file.endswith('.json'):
                        ann_file = os.path.join(sub_dir_path, file)
                        
                        # Define where to save the XML files for this dataset
                        xml_save_subdir = os.path.join(xmls_save_path, sub_dir)
                        
                        # Convert the COCO annotations to XML format
                        load_coco(ann_file, xml_save_subdir)
    
    # If the input path is a file, assume it's a single annotation JSON file
    elif os.path.isfile(data_dir) and data_dir.endswith('.json'):
        anno_file = data_dir  # Set the file path
        
        # Load and convert the COCO annotations to XML format
        load_coco(anno_file, xmls_save_path)

    else:
        raise ValueError("Invalid input: data_dir must be a directory containing subdirectories or a JSON file.")


if __name__ == '__main__':
    """
    Script Description:
        This script is used to convert annotation files from COCO format (JSON) to Pascal VOC format (XML).
        It processes either a single JSON file or a directory containing multiple JSON files.
    
    Parameters:
        - data_dir (str): The path to the JSON file or directory containing the COCO annotations.
        - xml_save_path (str): The directory where the generated XML files in Pascal VOC format will be saved.
    
    Example:
        python script.py --data-dir ./data/labels/coco/train.json --save-path ./data/convert/voc
    """
    
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser()
    
    # Argument for specifying the JSON annotation file or directory path
    parser.add_argument('-d', '--data-dir', type=str, default='./data/labels/coco/train.json', help='Path to the COCO JSON file or directory containing JSON files')
    
    # Argument for specifying where to save the generated XML files
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/voc', help='Path to save the converted XML files')

    # Parse command-line arguments
    opt = parser.parse_args()
    print(opt)  # Print the parsed arguments for debugging
    
    # If additional arguments are provided via the command line
    if len(sys.argv) > 1:
        # Call the parseJsonFile function with the provided arguments
        parseJsonFile(opt.data_dir, opt.save_path)
    else:
        # If no command-line arguments are provided, use the default paths
        data_dir = './data/labels/coco/train.json'  # Default path for JSON file
        xml_save_path = './data/convert/voc'  # Default path to save XML files
        
        # Call the parseJsonFile function with default paths
        parseJsonFile(data_dir=data_dir, xmls_save_path=xml_save_path)
