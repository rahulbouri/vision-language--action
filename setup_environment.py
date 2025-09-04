#!/usr/bin/env python3
"""
Setup script for PyBullet Ravens environment and Isaac GR00T data conversion.
This script sets up the environment and installs necessary dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True

def setup_ravens_environment():
    """Set up the PyBullet Ravens environment."""
    print("Setting up PyBullet Ravens environment...")
    
    ravens_path = Path("dataset_for_VLA/ravens")
    
    if not ravens_path.exists():
        print("Error: Ravens directory not found. Please ensure dataset_for_VLA is cloned.")
        return False
    
    # Install system dependencies (for macOS)
    print("Installing system dependencies...")
    if sys.platform == "darwin":  # macOS
        run_command("brew install opencv")
    
    # Create conda environment
    print("Creating conda environment...")
    if not run_command("conda create -n ravens python=3.8 -y"):
        print("Warning: Conda environment creation failed, trying with existing environment")
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    requirements_file = ravens_path / "requirements.txt"
    
    if requirements_file.exists():
        # Install with conda activate
        install_cmd = f"conda run -n ravens pip install -r {requirements_file}"
        if not run_command(install_cmd):
            # Fallback to regular pip
            run_command(f"pip install -r {requirements_file}")
    
    # Install Ravens package
    print("Installing Ravens package...")
    install_cmd = f"conda run -n ravens pip install -e {ravens_path}"
    if not run_command(install_cmd):
        # Fallback to regular pip
        run_command(f"pip install -e {ravens_path}")
    
    return True

def install_additional_dependencies():
    """Install additional dependencies for Isaac GR00T data conversion."""
    print("Installing additional dependencies for data conversion...")
    
    additional_packages = [
        "pandas",
        "pyarrow",  # For parquet files
        "opencv-python",
        "imageio",  # For video processing
        "tqdm",
        "jsonlines",
        "numpy",
        "pillow"
    ]
    
    for package in additional_packages:
        install_cmd = f"conda run -n ravens pip install {package}"
        if not run_command(install_cmd):
            run_command(f"pip install {package}")
    
    return True

def main():
    """Main setup function."""
    print("=" * 60)
    print("VLA Synthetic Data Generation Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("dataset_for_VLA").exists():
        print("Error: Please run this script from the VLA project root directory")
        print("Expected to find 'dataset_for_VLA' directory")
        return False
    
    # Setup Ravens environment
    if not setup_ravens_environment():
        print("Failed to setup Ravens environment")
        return False
    
    # Install additional dependencies
    if not install_additional_dependencies():
        print("Failed to install additional dependencies")
        return False
    
    print("=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Activate the environment: conda activate ravens")
    print("2. Run data generation: python generate_synthetic_data.py")
    print("3. Convert to Isaac GR00T format: python convert_to_groot_format.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
