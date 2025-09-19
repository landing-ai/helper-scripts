"""
Configuration management for segmentation tool.

This module provides utilities for managing configuration settings,
environment variables, and default values.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for segmentation tool.

    Handles loading settings from environment variables and providing defaults.
    """

    def __init__(self):
        """Initialize configuration with default values."""
        self.defaults = {
            'timeout': 60,
            'max_retries': 3,
            'default_output_format': 'json',
            'visualization_figsize': (15, 8),
            'supported_image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        }

    def get_api_credentials(self) -> tuple:
        """
        Get API credentials from environment variables.

        Returns:
            Tuple of (api_key, endpoint_id)

        Raises:
            ValueError: If required credentials are not found
        """
        api_key = os.getenv('LANDINGAI_API_KEY')
        endpoint_id = os.getenv('LANDINGAI_ENDPOINT')

        if not api_key:
            raise ValueError("LANDINGAI_API_KEY environment variable not set")
        if not endpoint_id:
            raise ValueError("LANDINGAI_ENDPOINT environment variable not set")

        return api_key, endpoint_id

    def get_timeout(self) -> int:
        """Get request timeout from environment or default."""
        return int(os.getenv('SEGMENTATION_TIMEOUT', self.defaults['timeout']))

    def get_max_retries(self) -> int:
        """Get maximum retry count from environment or default."""
        return int(os.getenv('SEGMENTATION_MAX_RETRIES', self.defaults['max_retries']))

    def get_output_format(self) -> str:
        """Get default output format from environment or default."""
        return os.getenv('SEGMENTATION_OUTPUT_FORMAT', self.defaults['default_output_format'])

    def get_supported_extensions(self) -> list:
        """Get list of supported image file extensions."""
        return self.defaults['supported_image_extensions'].copy()

    def get_visualization_figsize(self) -> tuple:
        """Get default figure size for visualizations."""
        width = int(os.getenv('SEGMENTATION_FIG_WIDTH', self.defaults['visualization_figsize'][0]))
        height = int(os.getenv('SEGMENTATION_FIG_HEIGHT', self.defaults['visualization_figsize'][1]))
        return (width, height)

    def validate_output_directory(self, output_dir: str) -> Path:
        """
        Validate and create output directory if needed.

        Args:
            output_dir: Directory path string

        Returns:
            Path object for the validated directory

        Raises:
            ValueError: If directory cannot be created
        """
        path = Path(output_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            raise ValueError(f"Cannot create output directory {output_dir}: {e}")

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all current configuration settings.

        Returns:
            Dictionary containing all configuration values
        """
        settings = self.defaults.copy()

        # Update with environment variables where available
        settings.update({
            'api_key': os.getenv('LANDINGAI_API_KEY', 'NOT_SET'),
            'endpoint_id': os.getenv('LANDINGAI_ENDPOINT', 'NOT_SET'),
            'timeout': self.get_timeout(),
            'max_retries': self.get_max_retries(),
            'output_format': self.get_output_format(),
            'visualization_figsize': self.get_visualization_figsize()
        })

        return settings


# Global configuration instance
config = Config()