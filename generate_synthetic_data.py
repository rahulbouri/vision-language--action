#!/usr/bin/env python3
"""
Generate synthetic ground truth data using PyBullet Ravens for VLA model training.

This script generates demonstrations for various manipulation tasks using
PyBullet Ravens and saves them in a format that can be converted to
Isaac GR00T format.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def run_ravens_demo(task: str, mode: str, n_episodes: int, data_dir: str, 
                   assets_root: str, display: bool = False) -> bool:
    """
    Run Ravens demonstration collection.
    
    Args:
        task: Task name (e.g., 'block-insertion', 'place-red-in-green')
        mode: 'train' or 'test'
        n_episodes: Number of episodes to generate
        data_dir: Directory to save data
        assets_root: Path to Ravens assets
        display: Whether to show display
    
    Returns:
        True if successful, False otherwise
    """
    
    # Ensure data directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "ravens/demos.py",
        f"--assets_root={assets_root}",
        f"--data_dir={data_dir}",
        f"--task={task}",
        f"--mode={mode}",
        f"--n={n_episodes}",
        "--shared_memory=False"
    ]
    
    if display:
        cmd.append("--disp=True")
    else:
        cmd.append("--disp=False")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Change to the dataset_for_VLA directory
        original_cwd = os.getcwd()
        os.chdir("dataset_for_VLA")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"Successfully generated {n_episodes} episodes for {task} ({mode})")
            return True
        else:
            print(f"Error generating data: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout while generating data for {task}")
        return False
    except Exception as e:
        print(f"Exception while generating data: {e}")
        return False

def generate_all_tasks(data_base_dir: str, assets_root: str, 
                      train_episodes: int = 100, test_episodes: int = 20,
                      display: bool = False) -> Dict[str, bool]:
    """
    Generate data for all available Ravens tasks.
    
    Args:
        data_base_dir: Base directory to save all datasets
        assets_root: Path to Ravens assets
        train_episodes: Number of training episodes per task
        test_episodes: Number of test episodes per task
        display: Whether to show display during generation
    
    Returns:
        Dictionary mapping task names to success status
    """
    
    # Available Ravens tasks
    tasks = [
        "block-insertion",
        "place-red-in-green", 
        "towers-of-hanoi",
        "stack-block-pyramid",
        "align-box-corner",
        "assembling-kits",
        "manipulating-rope",
        "packing-boxes",
        "palletizing-boxes",
        "sweeping-piles"
    ]
    
    results = {}
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Generating data for task: {task}")
        print(f"{'='*60}")
        
        # Create task-specific data directory
        task_data_dir = Path(data_base_dir) / f"{task}-{mode}"
        
        # Generate training data
        train_success = run_ravens_demo(
            task=task,
            mode="train", 
            n_episodes=train_episodes,
            data_dir=str(Path(data_base_dir) / f"{task}-train"),
            assets_root=assets_root,
            display=display
        )
        
        # Generate test data
        test_success = run_ravens_demo(
            task=task,
            mode="test",
            n_episodes=test_episodes, 
            data_dir=str(Path(data_base_dir) / f"{task}-test"),
            assets_root=assets_root,
            display=display
        )
        
        results[task] = train_success and test_success
        
        if results[task]:
            print(f"✓ Successfully generated data for {task}")
        else:
            print(f"✗ Failed to generate data for {task}")
    
    return results

def validate_generated_data(data_dir: str) -> bool:
    """
    Validate that the generated data has the expected structure.
    
    Args:
        data_dir: Directory containing generated data
    
    Returns:
        True if data is valid, False otherwise
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory does not exist: {data_dir}")
        return False
    
    # Check for required subdirectories
    required_dirs = ['action', 'color', 'depth', 'reward', 'info']
    missing_dirs = []
    
    for req_dir in required_dirs:
        if not (data_path / req_dir).exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        print(f"Missing required directories: {missing_dirs}")
        return False
    
    # Check for data files
    action_dir = data_path / 'action'
    action_files = list(action_dir.glob('*.pkl'))
    
    if not action_files:
        print("No action files found")
        return False
    
    print(f"Found {len(action_files)} episodes in {data_dir}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic data using PyBullet Ravens")
    parser.add_argument("--data_dir", type=str, default="./synthetic_data",
                       help="Directory to save generated data")
    parser.add_argument("--assets_root", type=str, 
                       default="./dataset_for_VLA/ravens/environments/assets/",
                       help="Path to Ravens assets directory")
    parser.add_argument("--train_episodes", type=int, default=100,
                       help="Number of training episodes per task")
    parser.add_argument("--test_episodes", type=int, default=20,
                       help="Number of test episodes per task")
    parser.add_argument("--display", action="store_true",
                       help="Show display during data generation")
    parser.add_argument("--task", type=str, default="all",
                       help="Specific task to generate (or 'all' for all tasks)")
    
    args = parser.parse_args()
    
    # Check if dataset_for_VLA exists
    if not Path("dataset_for_VLA").exists():
        print("Error: dataset_for_VLA directory not found")
        print("Please ensure you have cloned the repository and are in the correct directory")
        return 1
    
    # Check if assets directory exists
    assets_path = Path(args.assets_root)
    if not assets_path.exists():
        print(f"Error: Assets directory not found: {args.assets_root}")
        return 1
    
    print("=" * 60)
    print("VLA Synthetic Data Generation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Assets root: {args.assets_root}")
    print(f"Train episodes per task: {args.train_episodes}")
    print(f"Test episodes per task: {args.test_episodes}")
    print(f"Display: {args.display}")
    print(f"Task: {args.task}")
    
    # Create data directory
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    if args.task == "all":
        # Generate data for all tasks
        results = generate_all_tasks(
            data_base_dir=args.data_dir,
            assets_root=args.assets_root,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            display=args.display
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("GENERATION SUMMARY")
        print(f"{'='*60}")
        
        successful_tasks = [task for task, success in results.items() if success]
        failed_tasks = [task for task, success in results.items() if not success]
        
        print(f"Successful tasks ({len(successful_tasks)}): {successful_tasks}")
        if failed_tasks:
            print(f"Failed tasks ({len(failed_tasks)}): {failed_tasks}")
        
        # Validate generated data
        print(f"\n{'='*60}")
        print("VALIDATING GENERATED DATA")
        print(f"{'='*60}")
        
        for task in successful_tasks:
            for mode in ['train', 'test']:
                data_path = Path(args.data_dir) / f"{task}-{mode}"
                if validate_generated_data(str(data_path)):
                    print(f"✓ {task}-{mode}: Valid")
                else:
                    print(f"✗ {task}-{mode}: Invalid")
        
        if len(successful_tasks) > 0:
            print(f"\n{'='*60}")
            print("NEXT STEPS")
            print(f"{'='*60}")
            print("1. Convert to Isaac GR00T format:")
            for task in successful_tasks:
                for mode in ['train', 'test']:
                    print(f"   python convert_to_groot_format.py --ravens_data_dir {args.data_dir}/{task}-{mode} --output_dir {args.data_dir}/{task}-{mode}-groot")
            print("\n2. Train your VLA model with the converted data")
        
        return 0 if len(successful_tasks) > 0 else 1
    
    else:
        # Generate data for specific task
        print(f"Generating data for task: {args.task}")
        
        # Generate training data
        train_success = run_ravens_demo(
            task=args.task,
            mode="train",
            n_episodes=args.train_episodes,
            data_dir=str(Path(args.data_dir) / f"{args.task}-train"),
            assets_root=args.assets_root,
            display=args.display
        )
        
        # Generate test data
        test_success = run_ravens_demo(
            task=args.task,
            mode="test", 
            n_episodes=args.test_episodes,
            data_dir=str(Path(args.data_dir) / f"{args.task}-test"),
            assets_root=args.assets_root,
            display=args.display
        )
        
        if train_success and test_success:
            print(f"✓ Successfully generated data for {args.task}")
            return 0
        else:
            print(f"✗ Failed to generate data for {args.task}")
            return 1

if __name__ == "__main__":
    exit(main())
