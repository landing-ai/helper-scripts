"""
Command-line interface for the segmentation tool.

This module provides the CLI functionality including:
- Command-line argument parsing
- Batch processing with progress indication
- Unified output file generation (JSON/CSV)
- VOC format export
- Interactive visualization
- Formatted measurement display
"""

import os
import sys
import json
import csv
import click
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tabulate import tabulate
from dotenv import load_dotenv

from .core import SegmentationProcessor, SegmentationResult
from .voc_converter import VOCConverter
from .visualizer import show_visualization


def display_measurements(measurements: Dict, image_name: str):
    """
    Display measurements in a formatted, readable way.

    Args:
        measurements: Dictionary containing measurement data
        image_name: Name of the image being processed
    """
    if not measurements:
        click.echo("No measurements available")
        return

    click.echo(f"\n{'='*60}")
    click.echo(f"MEASUREMENTS: {image_name}")
    click.echo(f"{'='*60}")

    # Display summary information
    summary = measurements.get('summary', {})
    click.echo(f"Image Dimensions: {summary.get('image_dimensions', 'N/A')}")
    click.echo(f"Total Pixels: {summary.get('total_pixels', 'N/A'):,}")
    click.echo(f"Total Instances Detected: {summary.get('total_instances', 0)}")
    click.echo(f"Unique Classes: {summary.get('unique_classes', 0)}")

    # Display per-class measurements in table format
    by_class = measurements.get('by_class', {})
    if by_class:
        click.echo(f"\n{'-'*60}")
        click.echo("PER-CLASS MEASUREMENTS")
        click.echo(f"{'-'*60}")

        table_data = []
        for class_name, metrics in by_class.items():
            table_data.append([
                class_name,
                metrics.get('instance_count', 0),
                f"{metrics.get('absolute_coverage_pixels', 0):,}",
                f"{metrics.get('relative_coverage_percent', 0):.2f}%"
            ])

        headers = ["Class", "Instances", "Absolute Coverage (px)", "Relative Coverage (%)"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Display totals
    totals = measurements.get('totals', {})
    click.echo(f"\n{'-'*60}")
    click.echo("TOTAL MEASUREMENTS")
    click.echo(f"{'-'*60}")
    click.echo(f"Total Defect Coverage (Absolute): {totals.get('total_absolute_coverage_pixels', 0):,} pixels")
    click.echo(f"Total Defect Coverage (Relative): {totals.get('total_relative_coverage_percent', 0):.2f}%")
    click.echo(f"Total Defect Instances: {totals.get('total_defect_instances', 0)}")
    click.echo(f"{'='*60}\n")


def progress_callback(current: int, total: int, current_file: str):
    """
    Progress callback for batch processing.

    Args:
        current: Current file number
        total: Total number of files
        current_file: Name of current file being processed
    """
    percentage = (current / total) * 100
    click.echo(f"Progress: {current}/{total} ({percentage:.1f}%) - {Path(current_file).name}")


def save_unified_json(results: List[SegmentationResult], output_file: Path, metadata: Dict):
    """
    Save all results to a single JSON file.

    Args:
        results: List of processing results
        output_file: Output file path
        metadata: Metadata to include in the output
    """
    output_data = {
        "metadata": metadata,
        "results": [result.to_dict() for result in results]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def save_individual_json(result: SegmentationResult, output_file: Path, metadata: Dict):
    """
    Save a single result to a JSON file.

    Args:
        result: Single processing result
        output_file: Output file path
        metadata: Metadata to include in the output
    """
    output_data = {
        "metadata": metadata,
        "result": result.to_dict()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def save_individual_csv(result: SegmentationResult, output_file: Path, metadata: Dict):
    """
    Save a single result to a CSV file.

    Args:
        result: Single processing result
        output_file: Output file path
        metadata: Metadata to include (will be added as header comments)
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # Write metadata as comments
        f.write(f"# Segmentation Analysis Result\n")
        f.write(f"# Generated: {metadata.get('processed_at', 'Unknown')}\n")
        f.write(f"# Image: {Path(result.image_path).name}\n")
        f.write(f"# API Endpoint: {metadata.get('api_endpoint', 'Unknown')}\n")
        f.write("#\n")

        writer = csv.writer(f)

        # Write headers
        writer.writerow([
            'Image',
            'Class',
            'Instance Count',
            'Absolute Coverage (pixels)',
            'Relative Coverage (%)',
            'Image Width',
            'Image Height',
            'Total Pixels',
            'Processing Time (s)'
        ])

        # Write data rows for this single result
        if result.measurements:
            summary = result.measurements.get('summary', {})
            dims = summary.get('image_dimensions', '').split('x')
            width = dims[0] if len(dims) == 2 else ''
            height = dims[1] if len(dims) == 2 else ''
            total_pixels = summary.get('total_pixels', '')
            processing_time = result.processing_time or ''

            by_class = result.measurements.get('by_class', {})

            if by_class:
                # Write row for each class
                for class_name, metrics in by_class.items():
                    writer.writerow([
                        Path(result.image_path).name,
                        class_name,
                        metrics.get('instance_count', 0),
                        metrics.get('absolute_coverage_pixels', 0),
                        metrics.get('relative_coverage_percent', 0),
                        width,
                        height,
                        total_pixels,
                        processing_time
                    ])

                # Write totals row
                totals = result.measurements.get('totals', {})
                writer.writerow([
                    Path(result.image_path).name,
                    'TOTAL',
                    totals.get('total_defect_instances', 0),
                    totals.get('total_absolute_coverage_pixels', 0),
                    totals.get('total_relative_coverage_percent', 0),
                    width,
                    height,
                    total_pixels,
                    processing_time
                ])
            else:
                # No detections case
                writer.writerow([
                    Path(result.image_path).name,
                    'NO_DETECTIONS',
                    0,
                    0,
                    0,
                    width,
                    height,
                    total_pixels,
                    processing_time
                ])
        else:
            # No measurements calculated
            writer.writerow([
                Path(result.image_path).name,
                'NO_MEASUREMENTS',
                len(result.predictions),
                '',
                '',
                '',
                '',
                '',
                result.processing_time or ''
            ])


def save_unified_csv(results: List[SegmentationResult], output_file: Path, metadata: Dict):
    """
    Save all results to a single CSV file.

    Args:
        results: List of processing results
        output_file: Output file path
        metadata: Metadata to include (will be added as header comments)
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # Write metadata as comments
        f.write(f"# Segmentation Analysis Results\n")
        f.write(f"# Generated: {metadata.get('processed_at', 'Unknown')}\n")
        f.write(f"# Total Images: {metadata.get('total_images', 0)}\n")
        f.write(f"# API Endpoint: {metadata.get('api_endpoint', 'Unknown')}\n")
        f.write("#\n")

        writer = csv.writer(f)

        # Write headers
        writer.writerow([
            'Image',
            'Class',
            'Instance Count',
            'Absolute Coverage (pixels)',
            'Relative Coverage (%)',
            'Image Width',
            'Image Height',
            'Total Pixels',
            'Processing Time (s)'
        ])

        # Write data rows
        for result in results:
            if result.measurements:
                summary = result.measurements.get('summary', {})
                dims = summary.get('image_dimensions', '').split('x')
                width = dims[0] if len(dims) == 2 else ''
                height = dims[1] if len(dims) == 2 else ''
                total_pixels = summary.get('total_pixels', '')
                processing_time = result.processing_time or ''

                by_class = result.measurements.get('by_class', {})

                if by_class:
                    # Write row for each class
                    for class_name, metrics in by_class.items():
                        writer.writerow([
                            Path(result.image_path).name,
                            class_name,
                            metrics.get('instance_count', 0),
                            metrics.get('absolute_coverage_pixels', 0),
                            metrics.get('relative_coverage_percent', 0),
                            width,
                            height,
                            total_pixels,
                            processing_time
                        ])

                    # Write totals row
                    totals = result.measurements.get('totals', {})
                    writer.writerow([
                        Path(result.image_path).name,
                        'TOTAL',
                        totals.get('total_defect_instances', 0),
                        totals.get('total_absolute_coverage_pixels', 0),
                        totals.get('total_relative_coverage_percent', 0),
                        width,
                        height,
                        total_pixels,
                        processing_time
                    ])
                else:
                    # No detections case
                    writer.writerow([
                        Path(result.image_path).name,
                        'NO_DETECTIONS',
                        0,
                        0,
                        0,
                        width,
                        height,
                        total_pixels,
                        processing_time
                    ])
            else:
                # No measurements calculated
                writer.writerow([
                    Path(result.image_path).name,
                    'NO_MEASUREMENTS',
                    len(result.predictions),
                    '',
                    '',
                    '',
                    '',
                    '',
                    result.processing_time or ''
                ])


@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path), required=False)
@click.option('--api-key', envvar='LANDINGAI_API_KEY',
              help='LandingAI API key (or set LANDINGAI_API_KEY env var)')
@click.option('--endpoint', envvar='LANDINGAI_ENDPOINT',
              help='LandingAI endpoint ID (or set LANDINGAI_ENDPOINT env var)')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output directory for results')
@click.option('--measure', is_flag=True,
              help='Calculate absolute and relative coverage measurements')
@click.option('--visualize', is_flag=True,
              help='Show visualization for each image')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']),
              default='json', help='Output format for results (default: json)')
@click.option('--voc', is_flag=True,
              help='Export results in VOC dataset format')
@click.option('--timeout', default=60, type=int,
              help='Request timeout in seconds (default: 60)')
@click.option('--verbose', is_flag=True,
              help='Enable verbose output')
def main(input_path: Path, api_key: str, endpoint: str, output: Path,
         measure: bool, visualize: bool, output_format: str, voc: bool,
         timeout: int, verbose: bool):
    """
    Process images for segmentation analysis using LandingAI.

    INPUT_PATH can be a single image file or a directory containing images.

    Examples:
        segmentation-tool ./images/ --measure --output ./results/
        segmentation-tool image.jpg --visualize --measure
        segmentation-tool ./batch/ --measure --output ./results/ --format csv --voc
    """

    # Show help if no input path provided
    if input_path is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit(0)

    # Load .env file (won't override existing environment variables)
    load_dotenv(override=False)

    # Re-read environment variables after loading .env
    if not api_key:
        api_key = os.getenv('LANDINGAI_API_KEY')
    if not endpoint:
        endpoint = os.getenv('LANDINGAI_ENDPOINT')

    # Validate required credentials
    if not api_key or not endpoint:
        click.echo("Error: API key and endpoint are required!", err=True)
        click.echo("", err=True)
        click.echo("You can provide credentials in three ways (in order of precedence):", err=True)
        click.echo("", err=True)
        click.echo("1. Command-line options (highest priority):", err=True)
        click.echo("   segmentation-tool ./image.jpg --api-key YOUR_KEY --endpoint YOUR_ENDPOINT", err=True)
        click.echo("", err=True)
        click.echo("2. Environment variables:", err=True)
        click.echo("   export LANDINGAI_API_KEY='your_key'", err=True)
        click.echo("   export LANDINGAI_ENDPOINT='your_endpoint'", err=True)
        click.echo("", err=True)
        click.echo("3. Create a .env file in the current directory (lowest priority):", err=True)
        click.echo("   cp .env.example .env", err=True)
        click.echo("   # Then edit .env with your credentials", err=True)
        sys.exit(1)

    try:
        # Initialize processor
        click.echo(f"Initializing segmentation processor...")
        processor = SegmentationProcessor(api_key, endpoint)

        # Get list of images to process
        image_files = processor.get_image_files(input_path)
        click.echo(f"Found {len(image_files)} image(s) to process")

        if verbose:
            for img in image_files:
                click.echo(f"  - {img.name}")

        # Setup output directory if specified
        if output:
            output.mkdir(parents=True, exist_ok=True)
            click.echo(f"Output directory: {output}")

        # Process all images
        click.echo(f"\n{'='*60}")
        click.echo("PROCESSING IMAGES")
        click.echo(f"{'='*60}")

        results = processor.process_batch(
            input_path,
            calculate_measurements=measure,
            progress_callback=progress_callback if verbose else None
        )

        # Display results and handle visualization
        for result in results:
            image_name = Path(result.image_path).name
            click.echo(f"\nProcessed: {image_name}")

            if result.predictions:
                click.echo(f"  Detections: {len(result.predictions)}")
                if verbose:
                    classes = [getattr(p, 'label_name', 'unknown') for p in result.predictions]
                    click.echo(f"  Classes: {', '.join(set(classes))}")
            else:
                click.echo("  No detections found")

            # Show measurements
            if measure and result.measurements:
                display_measurements(result.measurements, image_name)
            elif measure:
                click.echo("  No measurements (no detections)")

            # Show visualization if requested
            if visualize:
                try:
                    show_visualization(result.image_path, result)
                    if len(results) > 0:  # ask
                        if not click.confirm(f"Continue to next image?"):
                            break
                except Exception as e:
                    click.echo(f"Warning: Visualization failed: {e}")

        # Save results if output directory specified
        if output:
            click.echo(f"\n{'='*60}")
            click.echo("SAVING RESULTS")
            click.echo(f"{'='*60}")

            # Prepare metadata
            metadata = {
                "processed_at": datetime.now().isoformat(),
                "total_images": len(results),
                "successful_predictions": sum(1 for r in results if r.predictions),
                "api_endpoint": endpoint,
                "measurements_calculated": measure,
                "output_format": output_format
            }

            # Always save individual prediction results as JSON files
            prediction_files = []
            for result in results:
                image_name = Path(result.image_path).stem  # Get filename without extension
                result_file = output / f"{image_name}.json"
                save_individual_json(result, result_file, metadata)
                prediction_files.append(result_file)

            click.echo(f"Prediction results saved to {len(prediction_files)} files:")
            for file_path in prediction_files:
                click.echo(f"  - {file_path.name}")

            # If measurements were calculated, save unified measurements file
            if measure:
                if output_format == 'json':
                    measurements_file = output / "results.json"
                    save_unified_json(results, measurements_file, metadata)
                else:  # csv
                    measurements_file = output / "results.csv"
                    save_unified_csv(results, measurements_file, metadata)

                click.echo(f"Measurement results saved to: {measurements_file.name}")

            # Export VOC format if requested
            if voc:
                try:
                    click.echo("Exporting to VOC format...")
                    converter = VOCConverter(str(output))
                    voc_summary = converter.convert(results)

                    click.echo(f"VOC export completed:")
                    click.echo(f"  - Converted images: {voc_summary['converted_images']}")
                    click.echo(f"  - Total instances: {voc_summary['total_instances']}")
                    click.echo(f"  - Classes found: {', '.join(voc_summary['classes_found'])}")
                    click.echo(f"  - Dataset path: {output}/VOCDataset")

                except Exception as e:
                    click.echo(f"Warning: VOC export failed: {e}")

        # Summary
        click.echo(f"\n{'='*60}")
        click.echo("PROCESSING SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total images processed: {len(results)}")
        click.echo(f"Images with detections: {sum(1 for r in results if r.predictions)}")

        if measure:
            total_instances = sum(len(r.predictions) for r in results)
            click.echo(f"Total instances detected: {total_instances}")

        click.echo("Processing completed successfully!")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()