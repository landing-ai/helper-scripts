"""
Entry point for running segmentation_tool as a module.

This allows the package to be run using:
    python -m segmentation_tool [options]
"""

from .cli import main

if __name__ == '__main__':
    main()