"""
VOC format converter for segmentation results.

This module provides functionality to convert segmentation results from LandingAI
into Pascal VOC format, which is widely used for computer vision datasets.

The VOC format includes:
- Images/ folder with copied original images
- Segmentations/ folder with PNG segmentation masks
- defect_map.json with class ID mappings
"""

import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from .core import SegmentationResult


class VOCConverter:
    """
    Converter class to export segmentation results to Pascal VOC format.

    The VOC format structure created:
    VOCDataset/
    ├── Images/              # Original images
    ├── Segmentations/       # PNG segmentation masks
    └── defect_map.json      # Class ID mappings
    """

    def __init__(self, output_dir: str):
        """
        Initialize VOC converter.

        Args:
            output_dir: Base directory where VOC dataset will be created
        """
        self.output_dir = Path(output_dir)
        self.voc_dir = self.output_dir / "VOCDataset"
        self.images_dir = self.voc_dir / "Images"
        self.segmentations_dir = self.voc_dir / "Segmentations"
        self.defect_map_path = self.voc_dir / "defect_map.json"

        # Initialize class mapping
        self.class_mapping = {"background": 0}
        self.next_class_id = 1

    def convert(self, results: List[SegmentationResult]) -> Dict:
        """
        Convert segmentation results to VOC format.

        Args:
            results: List of SegmentationResult objects to convert

        Returns:
            Dictionary containing conversion summary

        Raises:
            ValueError: If no valid results provided
            IOError: If file operations fail
        """
        if not results:
            raise ValueError("No results provided for conversion")

        # Setup directory structure
        self._setup_directories()

        # Build class mapping from all results
        self._build_class_mapping(results)

        conversion_summary = {
            "converted_images": 0,
            "total_instances": 0,
            "classes_found": list(self.class_mapping.keys()),
            "class_mapping": self.class_mapping.copy()
        }

        # Process each result
        for result in results:
            try:
                self._convert_single_result(result)
                conversion_summary["converted_images"] += 1
                conversion_summary["total_instances"] += len(result.predictions)

            except Exception as e:
                print(f"Warning: Failed to convert {result.image_path}: {str(e)}")
                continue

        # Save class mapping
        self._save_defect_map()

        return conversion_summary

    def _setup_directories(self):
        """Create necessary directories for VOC format."""
        self.voc_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.segmentations_dir.mkdir(exist_ok=True)

    def _build_class_mapping(self, results: List[SegmentationResult]):
        """
        Build class to ID mapping from all results.

        Args:
            results: List of results to extract classes from
        """
        for result in results:
            for pred in result.predictions:
                if hasattr(pred, 'label_name'):
                    class_name = pred.label_name
                    if class_name not in self.class_mapping:
                        self.class_mapping[class_name] = self.next_class_id
                        self.next_class_id += 1

    def _convert_single_result(self, result: SegmentationResult):
        """
        Convert a single result to VOC format.

        Args:
            result: SegmentationResult to convert

        Raises:
            IOError: If image operations fail
        """
        image_path = Path(result.image_path)

        # Copy original image to Images directory
        dest_image_path = self.images_dir / image_path.name
        shutil.copy2(image_path, dest_image_path)

        # Create segmentation mask
        if result.predictions:
            mask = self._create_voc_mask(result, image_path)

            # Save segmentation mask as PNG
            mask_name = image_path.stem + ".png"
            mask_path = self.segmentations_dir / mask_name

            mask_image = Image.fromarray(mask.astype(np.uint8))
            mask_image.save(mask_path)

    def _create_voc_mask(self, result: SegmentationResult, image_path: Path) -> np.ndarray:
        """
        Create VOC format segmentation mask from prediction results.

        Args:
            result: SegmentationResult containing predictions
            image_path: Path to original image for size reference

        Returns:
            Numpy array representing segmentation mask with class IDs

        Raises:
            IOError: If image cannot be loaded
        """
        # Load original image to get dimensions
        with Image.open(image_path) as img:
            image_size = img.size  # (width, height)

        # Initialize mask with background (0)
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Process each prediction
        for pred in result.predictions:
            try:
                # Get class ID
                class_name = getattr(pred, 'label_name', 'unknown')
                class_id = self.class_mapping.get(class_name, 0)

                # Decode mask
                if hasattr(pred, 'encoded_mask'):
                    decoded_mask = self._decode_prediction_mask(pred)

                    if decoded_mask.size > 0:
                        # Ensure mask dimensions match
                        if decoded_mask.shape != mask.shape:
                            print(f"Warning: Mask shape mismatch for {class_name}")
                            continue

                        # Assign class ID to mask pixels
                        mask[decoded_mask > 0] = class_id

            except Exception as e:
                print(f"Warning: Failed to process prediction for class {class_name}: {e}")
                continue

        return mask

    def _decode_prediction_mask(self, prediction) -> np.ndarray:
        """
        Decode RLE encoded mask from prediction object.

        Args:
            prediction: Prediction object with encoded mask

        Returns:
            Decoded binary mask as numpy array
        """
        try:
            encoding_map = prediction.encoding_map
            mask_shape = prediction.mask_shape
            encoded_mask = prediction.encoded_mask

            # Parse RLE encoded string
            decoded = ""
            parts = encoded_mask.replace("Z", "Z ").replace("N", "N ").split()

            for item in parts:
                if len(item) > 1:
                    count = int(item[:-1])
                    char = item[-1]
                    value = encoding_map.get(char, 0)
                    decoded += str(value) * count

            # Convert to numpy array and reshape
            mask_array = np.array([int(x) for x in decoded])
            mask = mask_array.reshape(mask_shape)

            return mask

        except Exception as e:
            print(f"Warning: RLE decoding failed: {e}")
            return np.array([])

    def _save_defect_map(self):
        """Save class mapping to JSON file."""
        defect_map = {str(class_id): class_name
                     for class_name, class_id in self.class_mapping.items()}

        with open(self.defect_map_path, 'w') as f:
            json.dump(defect_map, f, indent=2)

    def create_from_results(self, results: List[SegmentationResult],
                          output_dir: str) -> Dict:
        """
        Class method to create VOC dataset from results.

        Args:
            results: List of SegmentationResult objects
            output_dir: Directory where VOC dataset will be created

        Returns:
            Conversion summary dictionary
        """
        converter = VOCConverter(output_dir)
        return converter.convert(results)

    def get_dataset_info(self) -> Dict:
        """
        Get information about the created VOC dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.voc_dir.exists():
            return {"error": "VOC dataset not found"}

        info = {
            "dataset_path": str(self.voc_dir),
            "images_count": len(list(self.images_dir.glob("*"))) if self.images_dir.exists() else 0,
            "masks_count": len(list(self.segmentations_dir.glob("*.png"))) if self.segmentations_dir.exists() else 0,
            "class_mapping": self.class_mapping.copy(),
            "defect_map_exists": self.defect_map_path.exists()
        }

        return info

    def validate_dataset(self) -> Dict:
        """
        Validate the created VOC dataset structure.

        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "summary": {}
        }

        # Check directory structure
        required_dirs = [self.voc_dir, self.images_dir, self.segmentations_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation["is_valid"] = False
                validation["issues"].append(f"Missing directory: {dir_path}")

        # Check defect map
        if not self.defect_map_path.exists():
            validation["is_valid"] = False
            validation["issues"].append("Missing defect_map.json")

        # Count files
        if self.images_dir.exists() and self.segmentations_dir.exists():
            image_files = set(f.stem for f in self.images_dir.glob("*"))
            mask_files = set(f.stem for f in self.segmentations_dir.glob("*.png"))

            validation["summary"]["image_count"] = len(image_files)
            validation["summary"]["mask_count"] = len(mask_files)

            # Check for missing pairs
            missing_masks = image_files - mask_files
            missing_images = mask_files - image_files

            if missing_masks:
                validation["issues"].append(f"Images without masks: {missing_masks}")
            if missing_images:
                validation["issues"].append(f"Masks without images: {missing_images}")

        return validation