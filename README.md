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

