"""
Setup configuration for segmentation_tool package.

This file configures the package for installation via pip and defines
entry points for the CLI commands.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements = []
req_file = this_directory / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="segmentation_tool",
    version="1.0.0",
    author="Segmentation Tool Team",
    author_email="support@segmentation-tool.com",
    description="A comprehensive tool for image segmentation analysis using LandingAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/segmentation_tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'segmentation-tool=segmentation_tool.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="segmentation, computer vision, landingai, image analysis, defect detection",
    project_urls={
        "Bug Reports": "https://github.com/your-org/segmentation_tool/issues",
        "Source": "https://github.com/your-org/segmentation_tool",
        "Documentation": "https://github.com/your-org/segmentation_tool/wiki",
    },
)