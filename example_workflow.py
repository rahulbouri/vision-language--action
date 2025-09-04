#!/usr/bin/env python3
"""
Example workflow script demonstrating the complete process of generating
synthetic data and converting it to Isaac GR00T format.

This script provides a simple example of how to use the system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    else:
        print(f"✗ FAILED: {description}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False

def main():
    """Run the complete example workflow."""
    print("=" * 60)
    print("VLA Synthetic Data Generation - Example Workflow")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("dataset_for_VLA").exists():
        print("Error: Please run this script from the VLA project root directory")
        print("Expected to find 'dataset_for_VLA' directory")
        return 1
    
    # Step 1: Setup environment (optional - user can run manually)
    print("\nNote: If you haven't set up the environment yet, run:")
    print("python setup_environment.py")
    
    # Step 2: Generate synthetic data for a simple task
    print("\nGenerating synthetic data for block-insertion task...")
    
    success = run_command(
        "python generate_synthetic_data.py --task block-insertion --train_episodes 10 --test_episodes 5 --display=False",
        "Generate synthetic data for block-insertion task"
    )
    
    if not success:
        print("Failed to generate synthetic data. Please check the setup.")
        return 1
    
    # Step 3: Convert training data to Isaac GR00T format
    success = run_command(
        "python convert_to_groot_format.py --ravens_data_dir ./synthetic_data/block-insertion-train --output_dir ./synthetic_data/block-insertion-train-groot",
        "Convert training data to Isaac GR00T format"
    )
    
    if not success:
        print("Failed to convert training data.")
        return 1
    
    # Step 4: Convert test data to Isaac GR00T format
    success = run_command(
        "python convert_to_groot_format.py --ravens_data_dir ./synthetic_data/block-insertion-test --output_dir ./synthetic_data/block-insertion-test-groot",
        "Convert test data to Isaac GR00T format"
    )
    
    if not success:
        print("Failed to convert test data.")
        return 1
    
    # Step 5: Validate the converted data
    success = run_command(
        "python validate_conversion.py ./synthetic_data/block-insertion-train-groot",
        "Validate converted training data"
    )
    
    if not success:
        print("Training data validation failed.")
        return 1
    
    success = run_command(
        "python validate_conversion.py ./synthetic_data/block-insertion-test-groot",
        "Validate converted test data"
    )
    
    if not success:
        print("Test data validation failed.")
        return 1
    
    # Step 6: Show results
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\nGenerated datasets:")
    print("  - Training: ./synthetic_data/block-insertion-train-groot")
    print("  - Test: ./synthetic_data/block-insertion-test-groot")
    
    print("\nDataset structure:")
    for dataset in ["block-insertion-train-groot", "block-insertion-test-groot"]:
        dataset_path = Path(f"./synthetic_data/{dataset}")
        if dataset_path.exists():
            print(f"\n{dataset}:")
            for item in dataset_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(dataset_path)
                    print(f"  - {rel_path}")
    
    print("\nNext steps:")
    print("1. Use the converted datasets with Isaac GR00T")
    print("2. Train your VLA model")
    print("3. Generate more data for other tasks if needed")
    
    print("\nExample usage with Isaac GR00T:")
    print("""
from lerobot import LeRobotDataset

# Load the dataset
dataset = LeRobotDataset("./synthetic_data/block-insertion-train-groot")

# Access episodes
for episode in dataset.episodes:
    print(f"Episode {episode['episode_index']}: {episode['length']} steps")
    """)
    
    return 0

if __name__ == "__main__":
    exit(main())
