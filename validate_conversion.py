#!/usr/bin/env python3
"""
Validation script for Isaac GR00T data conversion.

This script validates that the converted data follows the correct format
and can be loaded properly.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse

def validate_directory_structure(data_dir: str) -> bool:
    """Validate that the directory has the correct Isaac GR00T structure."""
    data_path = Path(data_dir)
    
    required_dirs = ['meta', 'videos', 'data']
    required_meta_files = ['episodes.jsonl', 'tasks.jsonl', 'modality.json', 'info.json']
    
    print(f"Validating directory structure for: {data_dir}")
    
    # Check required directories
    for req_dir in required_dirs:
        if not (data_path / req_dir).exists():
            print(f"✗ Missing directory: {req_dir}")
            return False
        print(f"✓ Found directory: {req_dir}")
    
    # Check required meta files
    for req_file in required_meta_files:
        if not (data_path / 'meta' / req_file).exists():
            print(f"✗ Missing meta file: {req_file}")
            return False
        print(f"✓ Found meta file: {req_file}")
    
    return True

def validate_meta_files(data_dir: str) -> bool:
    """Validate the content of meta files."""
    data_path = Path(data_dir)
    meta_path = data_path / 'meta'
    
    print(f"\nValidating meta files...")
    
    # Validate info.json
    try:
        with open(meta_path / 'info.json', 'r') as f:
            info = json.load(f)
        
        required_info_fields = ['name', 'description', 'version', 'total_episodes', 'total_frames']
        for field in required_info_fields:
            if field not in info:
                print(f"✗ Missing field in info.json: {field}")
                return False
        
        print(f"✓ info.json is valid")
        print(f"  - Dataset: {info['name']}")
        print(f"  - Episodes: {info['total_episodes']}")
        print(f"  - Frames: {info['total_frames']}")
        
    except Exception as e:
        print(f"✗ Error reading info.json: {e}")
        return False
    
    # Validate tasks.jsonl
    try:
        tasks = []
        with open(meta_path / 'tasks.jsonl', 'r') as f:
            for line in f:
                tasks.append(json.loads(line.strip()))
        
        if not tasks:
            print(f"✗ No tasks found in tasks.jsonl")
            return False
        
        print(f"✓ tasks.jsonl is valid ({len(tasks)} tasks)")
        for i, task in enumerate(tasks[:3]):  # Show first 3 tasks
            print(f"  - Task {task['task_index']}: {task['task']}")
        
    except Exception as e:
        print(f"✗ Error reading tasks.jsonl: {e}")
        return False
    
    # Validate episodes.jsonl
    try:
        episodes = []
        with open(meta_path / 'episodes.jsonl', 'r') as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
        
        if not episodes:
            print(f"✗ No episodes found in episodes.jsonl")
            return False
        
        print(f"✓ episodes.jsonl is valid ({len(episodes)} episodes)")
        total_frames = sum(ep['length'] for ep in episodes)
        print(f"  - Total frames: {total_frames}")
        
    except Exception as e:
        print(f"✗ Error reading episodes.jsonl: {e}")
        return False
    
    # Validate modality.json
    try:
        with open(meta_path / 'modality.json', 'r') as f:
            modality = json.load(f)
        
        required_modality_sections = ['state', 'action']
        for section in required_modality_sections:
            if section not in modality:
                print(f"✗ Missing section in modality.json: {section}")
                return False
        
        print(f"✓ modality.json is valid")
        print(f"  - State modalities: {len(modality['state'])}")
        print(f"  - Action modalities: {len(modality['action'])}")
        
    except Exception as e:
        print(f"✗ Error reading modality.json: {e}")
        return False
    
    return True

def validate_data_files(data_dir: str) -> bool:
    """Validate the parquet data files."""
    data_path = Path(data_dir)
    chunk_path = data_path / 'data' / 'chunk-000'
    
    print(f"\nValidating data files...")
    
    if not chunk_path.exists():
        print(f"✗ Data chunk directory not found: {chunk_path}")
        return False
    
    parquet_files = list(chunk_path.glob('*.parquet'))
    if not parquet_files:
        print(f"✗ No parquet files found in {chunk_path}")
        return False
    
    print(f"✓ Found {len(parquet_files)} parquet files")
    
    # Validate first parquet file
    sample_file = parquet_files[0]
    try:
        df = pd.read_parquet(sample_file)
        
        required_columns = [
            'observation.state',
            'action', 
            'timestamp',
            'episode_index',
            'index'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                print(f"✗ Missing column in parquet: {col}")
                return False
        
        print(f"✓ Parquet file is valid")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Episode index: {df['episode_index'].iloc[0]}")
        
        # Check data types
        if not isinstance(df['observation.state'].iloc[0], list):
            print(f"✗ observation.state should be a list")
            return False
        
        if not isinstance(df['action'].iloc[0], list):
            print(f"✗ action should be a list")
            return False
        
        print(f"✓ Data types are correct")
        
    except Exception as e:
        print(f"✗ Error reading parquet file: {e}")
        return False
    
    return True

def validate_video_files(data_dir: str) -> bool:
    """Validate the video files."""
    data_path = Path(data_dir)
    video_path = data_path / 'videos' / 'chunk-000'
    
    print(f"\nValidating video files...")
    
    if not video_path.exists():
        print(f"✗ Video directory not found: {video_path}")
        return False
    
    video_files = list(video_path.glob('*.mp4'))
    if not video_files:
        print(f"✗ No video files found in {video_path}")
        return False
    
    print(f"✓ Found {len(video_files)} video files")
    
    # Check if video files are not empty
    for video_file in video_files[:3]:  # Check first 3 files
        if video_file.stat().st_size == 0:
            print(f"✗ Empty video file: {video_file.name}")
            return False
    
    print(f"✓ Video files are valid")
    
    return True

def validate_episode_consistency(data_dir: str) -> bool:
    """Validate consistency between episodes, data, and videos."""
    data_path = Path(data_dir)
    
    print(f"\nValidating episode consistency...")
    
    # Load episodes metadata
    episodes = []
    with open(data_path / 'meta' / 'episodes.jsonl', 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    
    # Count parquet files
    parquet_files = list((data_path / 'data' / 'chunk-000').glob('*.parquet'))
    
    # Count video files
    video_files = list((data_path / 'videos' / 'chunk-000').glob('*.mp4'))
    
    print(f"  - Episodes in metadata: {len(episodes)}")
    print(f"  - Parquet files: {len(parquet_files)}")
    print(f"  - Video files: {len(video_files)}")
    
    if len(episodes) != len(parquet_files):
        print(f"✗ Mismatch between episodes and parquet files")
        return False
    
    if len(episodes) != len(video_files):
        print(f"✗ Mismatch between episodes and video files")
        return False
    
    print(f"✓ Episode consistency is valid")
    
    return True

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Isaac GR00T data conversion")
    parser.add_argument("data_dir", type=str, help="Path to converted dataset directory")
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return 1
    
    print("=" * 60)
    print("Isaac GR00T Data Validation")
    print("=" * 60)
    
    validation_steps = [
        ("Directory Structure", validate_directory_structure),
        ("Meta Files", validate_meta_files),
        ("Data Files", validate_data_files),
        ("Video Files", validate_video_files),
        ("Episode Consistency", validate_episode_consistency)
    ]
    
    all_passed = True
    
    for step_name, validation_func in validation_steps:
        print(f"\n{step_name}:")
        print("-" * 40)
        
        try:
            if not validation_func(args.data_dir):
                all_passed = False
        except Exception as e:
            print(f"✗ Error during {step_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ VALIDATION PASSED - Dataset is ready for Isaac GR00T!")
    else:
        print("✗ VALIDATION FAILED - Please check the errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
