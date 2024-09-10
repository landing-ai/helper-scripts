import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

def process_coco_for_segmentation(input_dir, output_dir, input_json='annotations.json', output_json='annotations_modified.json', defect_map_file='defect_map.json'):
    """
    Process COCO annotations, modify category IDs, generate a defect map, and create segmentation masks for each image.

    Args:
        input_dir (str): Path to the input directory containing the COCO JSON file and images.
        output_dir (str): Path to the output directory where modified annotations and masks are saved.
        input_json (str): Name of the input COCO annotations JSON file. Default is 'annotations.json'.
        output_json (str): Name of the output modified COCO annotations JSON file. Default is 'annotations_modified.json'.
        defect_map_file (str): Name of the defect map JSON file to be saved. Default is 'defect_map.json'.
    
    Returns:
        None: The function processes the annotations, modifies the JSON, generates masks, and saves them to the output directory.
    """
    
    # Ensure that the output directory exists. If not, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the COCO annotations from the specified input JSON file
    with open(os.path.join(input_dir, input_json), 'r') as f:
        coco_data = json.load(f)

     # Initialize the new category ID counter
    new_category_id = 1

    # Create a mapping of old category IDs to new category IDs
    category_id_mapping = {}

    # Initialize defect mapping dictionary
    defect_map = {0: "ok"}

    # Update category IDs in the 'categories' section and map them to defect names
    for category in coco_data['categories']:
        old_id = category['id']
        category['id'] = new_category_id  # Reset to a sequential new ID
        category_id_mapping[old_id] = new_category_id  # Map old to new ID
        defect_map[new_category_id] = category['name']  # Map new ID to defect name
        new_category_id += 1  # Increment the new category ID for the next category

    # Update category IDs in the 'annotations' section based on the mapping
    for ann in coco_data['annotations']:
        ann['category_id'] = category_id_mapping[ann['category_id']]

    # Save the modified annotations to the specified output JSON file
    with open(os.path.join(input_dir, output_json), 'w') as f:
        json.dump(coco_data, f)

    # Save the defect map to the output directory as a JSON file
    with open(os.path.join(output_dir, defect_map_file), 'w') as f:
        json.dump(defect_map, f)

    # Load the modified COCO dataset
    coco = COCO(os.path.join(input_dir, output_json))

    # Get all image IDs in the dataset
    img_ids = coco.getImgIds()

    # Loop through each image in the dataset
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]  # Load image metadata
        img_file = os.path.join(input_dir, img_info['file_name'])  # Get the full path to the image file

        # Read the image in grayscale mode
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape[:2]  # Get the dimensions of the image

        # Initialize an empty binary mask (0 = background, 1 = object)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Get all annotations for the current image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Loop through each annotation (object) in the image
        for ann in anns:
            category_id = ann['category_id']  # Get the category_id of the object
            segmentation = ann['segmentation']  # Get segmentation data

            # Convert the segmentation to a binary mask
            seg_mask = coco.annToMask(ann)

            # Update the mask: set pixels that belong to the object to 1
            mask[seg_mask == 1] = 1

        # Save the mask as a PNG file in the output directory
        mask_file = os.path.splitext(img_info['file_name'])[0] + '.png'
        mask_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(mask_path, mask)


       
