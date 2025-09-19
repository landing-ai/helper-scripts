"""
Segmentation Tool - A comprehensive tool for image segmentation analysis using LandingAI.

This package provides both CLI and library interfaces for:
- Image segmentation prediction using LandingAI
- Measurement calculations (absolute and relative coverage)
- VOC dataset format conversion
- Interactive visualization
- Batch processing capabilities

Main classes:
    SegmentationProcessor: Core processing engine
    SegmentationResult: Container for processing results
    VOCConverter: VOC format converter
    SegmentationVisualizer: Visualization utilities

CLI usage:
    segmentation-tool ./images/ --measure --output ./results/ --format json

Library usage:
    from segmentation_tool import SegmentationProcessor
    processor = SegmentationProcessor(api_key="...", endpoint_id="...")
    result = processor.process_image("image.jpg", calculate_measurements=True)
"""

from .core import SegmentationProcessor, SegmentationResult
from .voc_converter import VOCConverter
from .visualizer import SegmentationVisualizer, show_visualization, create_batch_summary

__version__ = "1.0.0"
__author__ = "Segmentation Tool Team"
__email__ = "support@segmentation-tool.com"

# Export main classes and functions
__all__ = [
    'SegmentationProcessor',
    'SegmentationResult',
    'VOCConverter',
    'SegmentationVisualizer',
    'show_visualization',
    'create_batch_summary',
    # Convenience functions
    'process_image',
    'process_batch'
]


def process_image(image_path: str, api_key: str, endpoint_id: str,
                 calculate_measurements: bool = False) -> SegmentationResult:
    """
    Convenience function to process a single image.

    Args:
        image_path: Path to the image file
        api_key: LandingAI API key
        endpoint_id: LandingAI endpoint ID
        calculate_measurements: Whether to calculate measurements

    Returns:
        SegmentationResult object
    """
    processor = SegmentationProcessor(api_key, endpoint_id)
    return processor.process_image(image_path, calculate_measurements)


def process_batch(input_path: str, api_key: str, endpoint_id: str,
                 calculate_measurements: bool = False,
                 progress_callback=None) -> list:
    """
    Convenience function to process multiple images.

    Args:
        input_path: Directory path or single image path
        api_key: LandingAI API key
        endpoint_id: LandingAI endpoint ID
        calculate_measurements: Whether to calculate measurements
        progress_callback: Optional progress callback function

    Returns:
        List of SegmentationResult objects
    """
    processor = SegmentationProcessor(api_key, endpoint_id)
    return processor.process_batch(input_path, calculate_measurements, progress_callback)