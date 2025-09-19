"""
Visualization utilities for segmentation results.

This module provides functionality to visualize segmentation results including:
- Original image display
- Segmentation mask overlays
- Interactive matplotlib displays
- Customizable color schemes for different classes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from .core import SegmentationResult


class SegmentationVisualizer:
    """
    Visualizer class for segmentation results.

    Provides methods to create various types of visualizations including
    side-by-side comparisons, overlays, and individual mask displays.
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Initialize the visualizer.

        Args:
            figsize: Figure size for matplotlib plots (width, height)
        """
        self.figsize = figsize
        self.colors = self._get_default_colors()

    def _get_default_colors(self) -> List[Tuple[int, int, int]]:
        """
        Get default color palette for segmentation classes.

        Returns:
            List of RGB color tuples
        """
        return [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (255, 192, 203),  # Pink
            (165, 42, 42),    # Brown
            (128, 128, 128),  # Gray
            (0, 128, 0),      # Dark Green
        ]

    def show_comparison(self, image_path: Union[str, Path], result: SegmentationResult,
                       show_legend: bool = True, save_path: Optional[str] = None):
        """
        Show side-by-side comparison of original image and segmentation overlay.

        Args:
            image_path: Path to the original image
            result: SegmentationResult containing predictions
            show_legend: Whether to show color legend for classes
            save_path: Optional path to save the visualization
        """
        # Load original image
        image = Image.open(image_path)
        image_array = np.array(image)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        fig.suptitle(f'Segmentation Analysis: {Path(image_path).name}', fontsize=14, fontweight='bold')

        # Display original image
        axes[0].imshow(image_array)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # Create and display segmentation overlay
        if result.predictions:
            overlay = self._create_overlay(image_array, result.predictions)
            axes[1].imshow(overlay)
            axes[1].set_title(f'Segmentation Overlay ({len(result.predictions)} detections)', fontsize=12)
            axes[1].axis('off')

            # Display mask only
            mask_only = self._create_mask_visualization(image_array.shape, result.predictions)
            axes[2].imshow(mask_only)
            axes[2].set_title('Segmentation Masks', fontsize=12)
            axes[2].axis('off')

            # Add legend if requested
            if show_legend:
                self._add_legend(fig, result.predictions)

        else:
            # No predictions - show original in all panels
            axes[1].imshow(image_array)
            axes[1].set_title('No Detections', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(np.zeros_like(image_array))
            axes[2].set_title('No Masks', fontsize=12)
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show(block=False)
        plt.pause(0.1)  # Allow GUI to update

    def _create_overlay(self, image: np.ndarray, predictions: List) -> np.ndarray:
        """
        Create segmentation overlay on original image.

        Args:
            image: Original image as numpy array
            predictions: List of prediction objects

        Returns:
            Image with segmentation overlay as numpy array
        """
        overlay = image.copy()
        alpha = 0.6  # Transparency factor

        class_colors = {}
        color_idx = 0

        for pred in predictions:
            # Get class name
            class_name = getattr(pred, 'label_name', 'unknown')

            # Assign color to class if not already assigned
            if class_name not in class_colors:
                class_colors[class_name] = self.colors[color_idx % len(self.colors)]
                color_idx += 1

            # Decode mask
            try:
                mask = self._decode_prediction_mask(pred, image.shape[:2])
                if mask.size > 0:
                    color = class_colors[class_name]
                    # Apply colored overlay where mask is positive
                    mask_indices = mask > 0
                    overlay[mask_indices] = (overlay[mask_indices] * alpha +
                                           np.array(color) * (1 - alpha)).astype(np.uint8)
            except Exception as e:
                print(f"Warning: Failed to overlay mask for {class_name}: {e}")

        return overlay

    def _create_mask_visualization(self, image_shape: Tuple, predictions: List) -> np.ndarray:
        """
        Create visualization showing only the segmentation masks.

        Args:
            image_shape: Shape of the original image (height, width, channels)
            predictions: List of prediction objects

        Returns:
            Colored mask visualization as numpy array
        """
        # Create black background
        mask_vis = np.zeros(image_shape, dtype=np.uint8)

        class_colors = {}
        color_idx = 0

        for pred in predictions:
            # Get class name and assign color
            class_name = getattr(pred, 'label_name', 'unknown')
            if class_name not in class_colors:
                class_colors[class_name] = self.colors[color_idx % len(self.colors)]
                color_idx += 1

            # Decode and apply mask
            try:
                mask = self._decode_prediction_mask(pred, image_shape[:2])
                if mask.size > 0:
                    color = class_colors[class_name]
                    mask_indices = mask > 0
                    mask_vis[mask_indices] = color
            except Exception as e:
                print(f"Warning: Failed to visualize mask for {class_name}: {e}")

        return mask_vis

    def _decode_prediction_mask(self, prediction, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode prediction mask to match target image shape.

        Args:
            prediction: Prediction object with encoded mask
            target_shape: Target shape (height, width)

        Returns:
            Decoded binary mask as numpy array
        """
        if not hasattr(prediction, 'encoded_mask'):
            return np.array([])

        try:
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
                    value = encoding_map.get(char, 0)
                    decoded += str(value) * count

            # Convert to numpy array and reshape
            mask_array = np.array([int(x) for x in decoded])
            mask = mask_array.reshape(mask_shape)

            # Ensure mask matches target shape
            if mask.shape != target_shape:
                print(f"Warning: Mask shape {mask.shape} doesn't match target {target_shape}")
                return np.array([])

            return mask

        except Exception as e:
            print(f"Warning: Mask decoding failed: {e}")
            return np.array([])

    def _add_legend(self, fig, predictions: List):
        """
        Add color legend to the figure.

        Args:
            fig: Matplotlib figure object
            predictions: List of prediction objects
        """
        # Extract unique classes and their colors
        class_colors = {}
        color_idx = 0

        for pred in predictions:
            class_name = getattr(pred, 'label_name', 'unknown')
            if class_name not in class_colors:
                class_colors[class_name] = self.colors[color_idx % len(self.colors)]
                color_idx += 1

        # Create legend patches
        legend_patches = []
        for class_name, color in class_colors.items():
            # Convert to matplotlib color format (0-1 range)
            mpl_color = tuple(c/255.0 for c in color)
            patch = mpatches.Patch(color=mpl_color, label=class_name)
            legend_patches.append(patch)

        # Add legend to figure
        fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.0, 0.5))

    def create_summary_plot(self, results: List[SegmentationResult], save_path: Optional[str] = None):
        """
        Create summary visualization showing statistics across multiple images.

        Args:
            results: List of SegmentationResult objects
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to visualize")
            return

        # Collect statistics
        image_names = []
        detection_counts = []
        class_distributions = {}

        for result in results:
            image_names.append(Path(result.image_path).stem)
            detection_counts.append(len(result.predictions))

            # Count classes
            for pred in result.predictions:
                class_name = getattr(pred, 'label_name', 'unknown')
                if class_name not in class_distributions:
                    class_distributions[class_name] = 0
                class_distributions[class_name] += 1

        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Segmentation Analysis Summary', fontsize=16, fontweight='bold')

        # 1. Detection counts per image
        axes[0, 0].bar(range(len(image_names)), detection_counts)
        axes[0, 0].set_title('Detections per Image')
        axes[0, 0].set_xlabel('Images')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].set_xticks(range(len(image_names)))
        axes[0, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                                   for name in image_names], rotation=45)

        # 2. Class distribution
        if class_distributions:
            classes = list(class_distributions.keys())
            counts = list(class_distributions.values())
            colors = [self.colors[i % len(self.colors)] for i in range(len(classes))]
            colors_mpl = [tuple(c/255.0 for c in color) for color in colors]

            axes[0, 1].pie(counts, labels=classes, colors=colors_mpl, autopct='%1.1f%%')
            axes[0, 1].set_title('Class Distribution')

        # 3. Coverage statistics (if available)
        if any(r.measurements for r in results):
            coverage_data = []
            for result in results:
                if result.measurements:
                    total_coverage = result.measurements.get('totals', {}).get('total_relative_coverage_percent', 0)
                    coverage_data.append(total_coverage)
                else:
                    coverage_data.append(0)

            axes[1, 0].plot(range(len(image_names)), coverage_data, 'o-')
            axes[1, 0].set_title('Coverage Percentage per Image')
            axes[1, 0].set_xlabel('Images')
            axes[1, 0].set_ylabel('Coverage (%)')
            axes[1, 0].set_xticks(range(len(image_names)))
            axes[1, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                                       for name in image_names], rotation=45)

        # 4. Processing statistics
        processing_times = [r.processing_time or 0 for r in results]
        axes[1, 1].bar(range(len(image_names)), processing_times)
        axes[1, 1].set_title('Processing Time per Image')
        axes[1, 1].set_xlabel('Images')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_xticks(range(len(image_names)))
        axes[1, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                                   for name in image_names], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {save_path}")

        plt.show(block=False)
        plt.pause(0.1)


def show_visualization(image_path: Union[str, Path], result: SegmentationResult,
                      save_path: Optional[str] = None):
    """
    Convenience function to show visualization for a single result.

    Args:
        image_path: Path to the original image
        result: SegmentationResult to visualize
        save_path: Optional path to save the visualization
    """
    visualizer = SegmentationVisualizer()
    visualizer.show_comparison(image_path, result, save_path=save_path)


def create_batch_summary(results: List[SegmentationResult], save_path: Optional[str] = None):
    """
    Convenience function to create summary visualization for batch results.

    Args:
        results: List of SegmentationResult objects
        save_path: Optional path to save the summary plot
    """
    visualizer = SegmentationVisualizer()
    visualizer.create_summary_plot(results, save_path=save_path)