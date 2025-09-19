#!/usr/bin/env python3
"""
Basic usage examples for the segmentation_tool library.

This script demonstrates how to use the segmentation_tool as a Python library
for various common tasks including single image processing, batch processing,
and VOC format conversion.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import segmentation_tool
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation_tool import (
    SegmentationProcessor,
    VOCConverter,
    show_visualization,
    create_batch_summary
)


def example_single_image():
    """Example: Process a single image with measurements."""
    print("=" * 60)
    print("EXAMPLE 1: Single Image Processing")
    print("=" * 60)

    # Setup API credentials (replace with your actual credentials)
    api_key = os.getenv('LANDINGAI_API_KEY', 'your_api_key_here')
    endpoint_id = os.getenv('LANDINGAI_ENDPOINT', 'your_endpoint_id_here')

    if api_key == 'your_api_key_here' or endpoint_id == 'your_endpoint_id_here':
        print("Please set LANDINGAI_API_KEY and LANDINGAI_ENDPOINT environment variables")
        return

    try:
        # Initialize processor
        processor = SegmentationProcessor(api_key, endpoint_id)

        # Process a single image (replace with actual image path)
        image_path = "sample_images/test_image.jpg"

        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            print("Please add a test image to the sample_images/ directory")
            return

        # Process with measurements
        result = processor.process_image(image_path, calculate_measurements=True)

        # Display results
        print(f"Processed: {Path(result.image_path).name}")
        print(f"Detections found: {len(result.predictions)}")

        if result.measurements:
            measurements = result.measurements
            print("\nMeasurements:")
            print(f"  Total instances: {measurements['summary']['total_instances']}")
            print(f"  Unique classes: {measurements['summary']['unique_classes']}")
            print(f"  Total coverage: {measurements['totals']['total_relative_coverage_percent']:.2f}%")

            # Per-class details
            for class_name, metrics in measurements['by_class'].items():
                print(f"  {class_name}: {metrics['instance_count']} instances, "
                      f"{metrics['relative_coverage_percent']:.2f}% coverage")

        # Show visualization (optional - requires display)
        # show_visualization(image_path, result)

    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example: Process multiple images in batch."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 60)

    # Setup API credentials
    api_key = os.getenv('LANDINGAI_API_KEY', 'your_api_key_here')
    endpoint_id = os.getenv('LANDINGAI_ENDPOINT', 'your_endpoint_id_here')

    if api_key == 'your_api_key_here' or endpoint_id == 'your_endpoint_id_here':
        print("Please set LANDINGAI_API_KEY and LANDINGAI_ENDPOINT environment variables")
        return

    try:
        # Initialize processor
        processor = SegmentationProcessor(api_key, endpoint_id)

        # Process batch of images
        input_directory = "sample_images/"

        if not Path(input_directory).exists():
            print(f"Directory not found: {input_directory}")
            print("Please create sample_images/ directory with test images")
            return

        # Progress callback function
        def progress_callback(current, total, current_file):
            percentage = (current / total) * 100
            print(f"Processing: {current}/{total} ({percentage:.1f}%) - {Path(current_file).name}")

        # Process batch with measurements
        results = processor.process_batch(
            input_directory,
            calculate_measurements=True,
            progress_callback=progress_callback
        )

        # Display batch summary
        print(f"\nBatch processing completed!")
        print(f"Total images processed: {len(results)}")
        print(f"Images with detections: {sum(1 for r in results if r.predictions)}")

        total_detections = sum(len(r.predictions) for r in results)
        print(f"Total detections across all images: {total_detections}")

        # Create summary visualization (optional)
        # create_batch_summary(results, save_path="batch_summary.png")

    except Exception as e:
        print(f"Error: {e}")


def example_voc_conversion():
    """Example: Convert results to VOC format."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: VOC Format Conversion")
    print("=" * 60)

    # Setup API credentials
    api_key = os.getenv('LANDINGAI_API_KEY', 'your_api_key_here')
    endpoint_id = os.getenv('LANDINGAI_ENDPOINT', 'your_endpoint_id_here')

    if api_key == 'your_api_key_here' or endpoint_id == 'your_endpoint_id_here':
        print("Please set LANDINGAI_API_KEY and LANDINGAI_ENDPOINT environment variables")
        return

    try:
        # Process some images first
        processor = SegmentationProcessor(api_key, endpoint_id)

        input_directory = "sample_images/"
        if not Path(input_directory).exists():
            print(f"Directory not found: {input_directory}")
            print("Please create sample_images/ directory with test images")
            return

        results = processor.process_batch(input_directory, calculate_measurements=True)

        if not results:
            print("No results to convert")
            return

        # Convert to VOC format
        output_directory = "output_voc/"
        converter = VOCConverter(output_directory)

        conversion_summary = converter.convert(results)

        print("VOC conversion completed!")
        print(f"Output directory: {output_directory}/VOCDataset")
        print(f"Converted images: {conversion_summary['converted_images']}")
        print(f"Total instances: {conversion_summary['total_instances']}")
        print(f"Classes found: {', '.join(conversion_summary['classes_found'])}")

        # Validate the dataset
        validation = converter.validate_dataset()
        if validation['is_valid']:
            print("✓ VOC dataset validation passed")
        else:
            print("✗ VOC dataset validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")

    except Exception as e:
        print(f"Error: {e}")


def example_convenience_functions():
    """Example: Using convenience functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Convenience Functions")
    print("=" * 60)

    # Setup API credentials
    api_key = os.getenv('LANDINGAI_API_KEY', 'your_api_key_here')
    endpoint_id = os.getenv('LANDINGAI_ENDPOINT', 'your_endpoint_id_here')

    if api_key == 'your_api_key_here' or endpoint_id == 'your_endpoint_id_here':
        print("Please set LANDINGAI_API_KEY and LANDINGAI_ENDPOINT environment variables")
        return

    try:
        # Import convenience functions
        from segmentation_tool import process_image, process_batch

        # Single image using convenience function
        image_path = "sample_images/test_image.jpg"
        if Path(image_path).exists():
            result = process_image(image_path, api_key, endpoint_id, calculate_measurements=True)
            print(f"Convenience function processed: {Path(result.image_path).name}")
            print(f"Detections: {len(result.predictions)}")

        # Batch processing using convenience function
        input_directory = "sample_images/"
        if Path(input_directory).exists():
            results = process_batch(input_directory, api_key, endpoint_id, calculate_measurements=True)
            print(f"Convenience function processed {len(results)} images")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("Segmentation Tool - Library Usage Examples")
    print("==========================================")

    # Check if API credentials are set
    api_key = os.getenv('LANDINGAI_API_KEY')
    endpoint_id = os.getenv('LANDINGAI_ENDPOINT')

    if not api_key or not endpoint_id:
        print("\n⚠️  SETUP REQUIRED:")
        print("Please set your API credentials as environment variables:")
        print("  export LANDINGAI_API_KEY='your_api_key'")
        print("  export LANDINGAI_ENDPOINT='your_endpoint_id'")
        print("\nOr copy .env.example to .env and fill in your credentials")
        print("\nRunning examples with placeholder values (will show errors)...\n")

    # Run examples
    example_single_image()
    example_batch_processing()
    example_voc_conversion()
    example_convenience_functions()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()