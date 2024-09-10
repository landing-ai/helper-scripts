# helper-scripts
Scripts and functions to assist with common Visual AI tasks


## Image Preparation

| Script | Description |
|-----------------|-----------------|
| [coco_to_pascal.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/coco_to_pascal.py)   |Convert COCO-style annotation files (.json) into Pascal VOC format annotation files (.xml).|
| [yolo_to_pascal.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/yolo_to_pascal.py)    | Convert YOLO format annotation files (.txt) to Pascal VOC format annotation files (.xml).|
| [process_coco_for_segmentation.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/process_coco_for_segmentation.py)    | Process COCO annotations, modify category IDs, generate a defect map, and create segmentation masks for each image. |
| [transparent_background_to_solid.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/transparent_background_to_solid.py)    | Convert all images in a directory with transparency to RGB mode, and save them with a specified background color. (e.g. white)|
| [extract_images_from_videos.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/extract_images_from_videos.py)    | Extract frames from video files in a directory at a specified frame rate and save them as images.|
|[convert_jfif_to_jpeg.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/convert_jfif_to_jpeg.py)|Converts all .jfif files in the specified directory to .jpeg format, removing any alpha channel (transparency) if present.|
|[find_def_location_grid.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/find_def_location_grid.py)|Determines the grid cell location of a bounding box within an image.|
|[convert_png_masks.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/convert_png_masks.py)|Converts all masks in a folder (path) to Landing AI-compatible format for masks (8-bit grayscale).|
|[crop_images_json.py](https://github.com/landing-ai/helper-scripts/blob/main/Image_preparation/crop_images_json.py)|Crops images based on cropping information in JSON files and saves cropped images in label-specific directories.|