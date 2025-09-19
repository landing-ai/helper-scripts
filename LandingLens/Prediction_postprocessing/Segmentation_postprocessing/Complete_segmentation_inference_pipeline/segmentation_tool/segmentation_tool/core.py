"""
Core processing module for segmentation analysis.

This module provides the main SegmentationProcessor class that handles:
- Image prediction using LandingAI
- Measurement calculations (absolute and relative coverage)
- Mask decoding and processing
- Batch processing capabilities
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from PIL import Image
from landingai.predict import Predictor
from landingai.postprocess import (
    segmentation_class_pixel_coverage,
    class_counts,
    class_pixel_coverage
)


@dataclass
class SegmentationResult:
    """
    Container for results of a single image processing.

    Attributes:
        image_path: Path to the processed image
        predictions: Raw predictions from LandingAI
        measurements: Calculated measurements (if requested)
        mask_array: Decoded segmentation mask as numpy array (optional)
        processing_time: Time taken to process this image in seconds
    """
    image_path: str
    predictions: List[Dict]
    measurements: Optional[Dict] = None
    mask_array: Optional[np.ndarray] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding numpy arrays for JSON serialization"""
        result = asdict(self)
        # Remove mask_array as it's not JSON serializable
        if 'mask_array' in result:
            del result['mask_array']

        # Convert prediction objects to dictionaries
        if 'predictions' in result and result['predictions']:
            serializable_predictions = []
            for pred in result['predictions']:
                if hasattr(pred, '__dict__'):
                    # Convert prediction object to dict, only including known attributes
                    pred_dict = {}
                    # Known attributes from LandingAI prediction objects
                    known_attrs = ['label_name', 'encoding_map', 'mask_shape', 'encoded_mask',
                                   'confidence', 'score', 'class_id', 'bbox']

                    for attr in known_attrs:
                        if hasattr(pred, attr):
                            value = getattr(pred, attr)
                            # Handle different value types
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                pred_dict[attr] = value
                            elif isinstance(value, tuple):
                                pred_dict[attr] = list(value)  # Convert tuples to lists
                    serializable_predictions.append(pred_dict)
                else:
                    # If it's already a dict or serializable, keep as-is
                    serializable_predictions.append(pred)
            result['predictions'] = serializable_predictions

        return result


class SegmentationProcessor:
    """
    Main processor for image segmentation analysis.

    This class handles the complete pipeline from image prediction to measurement
    calculation using LandingAI's prediction and postprocessing capabilities.
    """

    def __init__(self, api_key: str, endpoint_id: str):
        """
        Initialize the segmentation processor.

        Args:
            api_key: LandingAI API key
            endpoint_id: LandingAI endpoint ID
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.predictor = Predictor(endpoint_id, api_key=api_key)

        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def get_image_files(self, input_path: Union[str, Path]) -> List[Path]:
        """
        Get list of image files from input path.

        Args:
            input_path: Single image file or directory containing images

        Returns:
            List of Path objects for image files

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If no image files found
        """
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if path.is_file():
            if path.suffix.lower() in self.image_extensions:
                return [path]
            else:
                raise ValueError(f"File is not a supported image format: {path.suffix}")

        # Directory case - find all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(path.glob(f'*{ext}'))
            image_files.extend(path.glob(f'*{ext.upper()}'))

        if not image_files:
            raise ValueError(f"No image files found in directory: {input_path}")

        return sorted(image_files)

    def process_image(self, image_path: Union[str, Path],
                     calculate_measurements: bool = False) -> SegmentationResult:
        """
        Process a single image for segmentation analysis.

        Args:
            image_path: Path to the image file
            calculate_measurements: Whether to calculate coverage measurements

        Returns:
            SegmentationResult containing predictions and optional measurements

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If prediction or processing fails
        """
        start_time = datetime.now()
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        try:
            # Load and predict
            image = Image.open(image_path)

            # Debug: Print image info
            print(f"Debug: Image mode: {image.mode}, size: {image.size}")

            predictions = self.predictor.predict(image)

            # Debug: Print prediction info
            print(f"Debug: Predictions count: {len(predictions) if predictions else 0}")

            # Ensure image size is a proper tuple
            image_size = image.size  # PIL returns (width, height)
            if not isinstance(image_size, tuple) or len(image_size) != 2:
                raise ValueError(f"Invalid image size format: {image_size}")

            # Create result object
            processing_time = (datetime.now() - start_time).total_seconds()

            result = SegmentationResult(
                image_path=str(image_path),
                predictions=predictions or [],
                processing_time=processing_time
            )

            # Calculate measurements if requested and predictions exist
            if calculate_measurements and predictions:
                print(f"Debug: Calculating measurements for {len(predictions)} predictions")
                result.measurements = self._calculate_measurements(predictions, image_size)
                result.mask_array = self._create_combined_mask(predictions, image_size)

            return result

        except Exception as e:
            import traceback
            print(f"Debug: Full traceback:")
            traceback.print_exc()
            raise Exception(f"Failed to process image {image_path}: {str(e)}")

    def process_batch(self, input_path: Union[str, Path],
                     calculate_measurements: bool = False,
                     progress_callback: Optional[callable] = None) -> List[SegmentationResult]:
        """
        Process multiple images in batch.

        Args:
            input_path: Directory path or single image path
            calculate_measurements: Whether to calculate measurements
            progress_callback: Optional callback function for progress updates

        Returns:
            List of SegmentationResult objects

        Raises:
            ValueError: If no images found or processing fails
        """
        image_files = self.get_image_files(input_path)
        results = []

        total_files = len(image_files)

        for idx, image_file in enumerate(image_files):
            try:
                result = self.process_image(image_file, calculate_measurements)
                results.append(result)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(idx + 1, total_files, str(image_file))

            except Exception as e:
                print(f"Warning: Failed to process {image_file}: {str(e)}")
                continue

        if not results:
            raise ValueError("No images were successfully processed")

        return results

    def _calculate_measurements(self, predictions: List[Dict], image_size: Tuple[int, int]) -> Dict:
        """
        Calculate comprehensive measurements using LandingAI's postprocessing functions.

        Args:
            predictions: List of prediction objects from LandingAI
            image_size: Tuple of (width, height) in pixels

        Returns:
            Dictionary containing structured measurements
        """
        # Get measurements using LandingAI's built-in functions
        absolute_coverage = segmentation_class_pixel_coverage(
            predictions, coverage_type="absolute"
        )

        relative_coverage = segmentation_class_pixel_coverage(
            predictions, coverage_type="relative"
        )

        counts = class_counts(predictions)
        per_class_coverage = class_pixel_coverage(predictions)

        # Debug: Print what we got from LandingAI functions
        print(f"Debug: absolute_coverage = {absolute_coverage}")
        print(f"Debug: relative_coverage = {relative_coverage}")
        print(f"Debug: counts = {counts}")
        print(f"Debug: per_class_coverage = {per_class_coverage}")

        # Ensure image_size is valid
        width, height = image_size[0], image_size[1]
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

        # Extract values from LandingAI tuple format: (value, label_name)
        def extract_value(item):
            if isinstance(item, tuple) and len(item) >= 1:
                return item[0]  # Get the numeric value
            return item

        def extract_label(item):
            if isinstance(item, tuple) and len(item) >= 2:
                return item[1]  # Get the label name
            return "unknown"

        # Calculate totals safely
        total_absolute_pixels = sum(extract_value(v) for v in absolute_coverage.values()) if absolute_coverage else 0
        total_relative_percent = sum(extract_value(v) for v in relative_coverage.values()) if relative_coverage else 0
        total_instances = sum(extract_value(v) for v in counts.values()) if counts else 0

        # Structure comprehensive measurements
        measurements = {
            'summary': {
                'total_instances': int(total_instances),
                'unique_classes': len(counts) if counts else 0,
                'image_dimensions': f"{int(width)}x{int(height)}",
                'total_pixels': int(width * height),
                'processed_at': datetime.now().isoformat()
            },
            'by_class': {},
            'totals': {
                'total_absolute_coverage_pixels': int(total_absolute_pixels),
                'total_relative_coverage_percent': float(total_relative_percent),
                'total_defect_instances': int(total_instances)
            }
        }

        # Per-class detailed measurements
        # Build class mapping from ID to label name
        class_id_to_name = {}
        for class_id in counts.keys():
            count_data = counts.get(class_id, (0, "unknown"))
            class_id_to_name[class_id] = extract_label(count_data)

        for class_id in counts.keys():
            class_name = class_id_to_name[class_id]

            # Extract values safely from tuples
            abs_coverage = extract_value(absolute_coverage.get(class_id, 0))
            rel_coverage = extract_value(relative_coverage.get(class_id, 0))
            instance_count = extract_value(counts.get(class_id, 0))
            per_instance_cov = extract_value(per_class_coverage.get(class_id, 0))

            measurements['by_class'][class_name] = {
                'absolute_coverage_pixels': int(abs_coverage),
                'relative_coverage_percent': float(rel_coverage),
                'instance_count': int(instance_count),
                'per_instance_coverage': float(per_instance_cov) if isinstance(per_instance_cov, (int, float)) else 0
            }

        return measurements

    def _decode_mask(self, prediction) -> np.ndarray:
        """
        Decode a single prediction mask to numpy array.

        Args:
            prediction: Single prediction object with encoded mask

        Returns:
            Decoded binary mask as numpy array
        """
        if not hasattr(prediction, 'encoded_mask'):
            return np.array([])

        try:
            # Use the decoding logic from the original defect_measurement.py
            encoding_map = prediction.encoding_map
            mask_shape = prediction.mask_shape
            encoded_mask = prediction.encoded_mask

            # Decode RLE string
            decoded = ""
            parts = encoded_mask.replace("Z", "Z ").replace("N", "N ").split()

            for item in parts:
                if len(item) > 1:
                    count = int(item[:-1])
                    char = item[-1]
                    decoded += str(encoding_map.get(char, 0)) * count

            # Convert to numpy array and reshape
            mask = np.array(list(map(int, decoded))).reshape(mask_shape)
            return mask

        except Exception as e:
            print(f"Warning: Failed to decode mask: {e}")
            return np.array([])

    def _create_combined_mask(self, predictions: List[Dict], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Create a combined segmentation mask from all predictions.

        Args:
            predictions: List of prediction objects
            image_size: Tuple of (width, height)

        Returns:
            Combined mask array where each class has a unique ID
        """
        # Ensure image_size contains valid integers
        width, height = int(image_size[0]), int(image_size[1])
        mask = np.zeros((height, width), dtype=np.uint8)

        for class_id, pred in enumerate(predictions, 1):
            decoded_mask = self._decode_mask(pred)
            if decoded_mask.size > 0:
                # Ensure mask is same size as image
                if decoded_mask.shape != mask.shape:
                    continue
                mask[decoded_mask > 0] = class_id

        return mask

    def get_class_mapping(self, results: List[SegmentationResult]) -> Dict[str, int]:
        """
        Extract class to ID mapping from processing results.

        Args:
            results: List of processing results

        Returns:
            Dictionary mapping class names to numeric IDs
        """
        class_mapping = {"background": 0}
        class_id = 1

        for result in results:
            for pred in result.predictions:
                if hasattr(pred, 'label_name'):
                    class_name = pred.label_name
                    if class_name not in class_mapping:
                        class_mapping[class_name] = class_id
                        class_id += 1

        return class_mapping