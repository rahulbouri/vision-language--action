#!/usr/bin/env python3
"""
Convert PyBullet Ravens data to Isaac GR00T LeRobot compatible format.

This script converts the Ravens dataset format to the Isaac GR00T format
which includes:
- Parquet files for state/action data
- MP4 files for video observations
- JSON metadata files
- Proper directory structure
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import cv2
import imageio
from pathlib import Path
from typing import Dict, List, Any, Tuple
import jsonlines
from tqdm import tqdm
import argparse

class RavensToGR00TConverter:
    """Convert Ravens dataset to Isaac GR00T format."""
    
    def __init__(self, ravens_data_dir: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            ravens_data_dir: Path to Ravens dataset directory
            output_dir: Path to output Isaac GR00T format directory
        """
        self.ravens_data_dir = Path(ravens_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create required directories
        (self.output_dir / "meta").mkdir(exist_ok=True)
        (self.output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        
        # Task descriptions for different Ravens tasks
        self.task_descriptions = {
            "block-insertion": "Insert the L-shaped block into the fixture",
            "place-red-in-green": "Place the red block in the green zone",
            "towers-of-hanoi": "Move blocks to solve the towers of hanoi puzzle",
            "stack-block-pyramid": "Stack blocks to form a pyramid",
            "align-box-corner": "Align the box with the corner",
            "assembling-kits": "Assemble the kit by placing objects correctly",
            "manipulating-rope": "Manipulate the rope to achieve the goal",
            "packing-boxes": "Pack boxes in the container",
            "palletizing-boxes": "Palletize boxes in the correct order",
            "sweeping-piles": "Sweep the pile of objects"
        }
        
    def load_ravens_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load a single Ravens episode."""
        episode_data = {}
        
        # Load each modality
        for modality in ['color', 'depth', 'action', 'reward', 'info']:
            modality_dir = self.ravens_data_dir / modality
            if not modality_dir.exists():
                continue
                
            # Find the file for this episode
            files = list(modality_dir.glob(f"*{episode_id:06d}*.pkl"))
            if not files:
                continue
                
            file_path = files[0]
            with open(file_path, 'rb') as f:
                episode_data[modality] = pickle.load(f)
        
        return episode_data
    
    def extract_task_name(self, data_dir: str) -> str:
        """Extract task name from the data directory path."""
        # Try to infer task name from directory structure
        if "block-insertion" in data_dir:
            return "block-insertion"
        elif "place-red-in-green" in data_dir:
            return "place-red-in-green"
        elif "towers-of-hanoi" in data_dir:
            return "towers-of-hanoi"
        elif "stack-block-pyramid" in data_dir:
            return "stack-block-pyramid"
        else:
            return "unknown-task"
    
    def create_video_from_frames(self, color_frames: np.ndarray, output_path: str) -> bool:
        """Create MP4 video from color frames."""
        try:
            # Ensure frames are in the correct format (H, W, C)
            if len(color_frames.shape) == 4:  # (T, H, W, C)
                frames = color_frames
            else:
                frames = color_frames.reshape(-1, *color_frames.shape[-3:])
            
            # Convert to uint8 if needed
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8)
            
            # Write video using imageio
            with imageio.get_writer(output_path, fps=10) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            return True
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def create_parquet_data(self, episode_data: Dict[str, Any], episode_id: int, 
                          task_name: str, global_index: int) -> pd.DataFrame:
        """Create parquet data for an episode."""
        rows = []
        
        # Get the length of the episode
        episode_length = len(episode_data.get('action', []))
        
        for step in range(episode_length):
            row = {}
            
            # State information (concatenated from available modalities)
            state_components = []
            
            # Add action as part of state (current action)
            if 'action' in episode_data and step < len(episode_data['action']):
                action = episode_data['action'][step]
                if isinstance(action, (list, np.ndarray)):
                    state_components.extend(action)
                else:
                    state_components.append(action)
            
            # Add reward as part of state
            if 'reward' in episode_data and step < len(episode_data['reward']):
                reward = episode_data['reward'][step]
                state_components.append(reward)
            
            # Add info as part of state if available
            if 'info' in episode_data and step < len(episode_data['info']):
                info = episode_data['info'][step]
                if isinstance(info, dict):
                    # Extract relevant info fields
                    for key, value in info.items():
                        if isinstance(value, (int, float)):
                            state_components.append(value)
                        elif isinstance(value, (list, np.ndarray)) and len(value) <= 10:
                            state_components.extend(value)
            
            row['observation.state'] = state_components
            
            # Action (next action or current action)
            if 'action' in episode_data and step < len(episode_data['action']):
                action = episode_data['action'][step]
                if isinstance(action, (list, np.ndarray)):
                    row['action'] = action
                else:
                    row['action'] = [action]
            else:
                row['action'] = [0.0]  # Default action
            
            # Timestamp
            row['timestamp'] = step * 0.1  # Assume 10Hz
            
            # Task annotation
            task_index = list(self.task_descriptions.keys()).index(task_name) if task_name in self.task_descriptions else 0
            row['annotation.human.action.task_description'] = task_index
            row['task_index'] = task_index
            row['annotation.human.validity'] = 1  # Valid
            
            # Episode and step indices
            row['episode_index'] = episode_id
            row['index'] = global_index + step
            
            # Next step information
            if step < episode_length - 1:
                next_reward = episode_data.get('reward', [0])[step + 1] if step + 1 < len(episode_data.get('reward', [])) else 0
                row['next.reward'] = next_reward
                row['next.done'] = False
            else:
                row['next.reward'] = 0
                row['next.done'] = True
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_modality_config(self, task_name: str) -> Dict[str, Any]:
        """Create modality configuration for Isaac GR00T format."""
        # This is a simplified configuration - you may need to adjust based on your specific data
        config = {
            "state": {
                "action_state": {
                    "start": 0,
                    "end": 7,  # Assuming 7-DOF action
                    "dtype": "float32"
                },
                "reward_state": {
                    "start": 7,
                    "end": 8,
                    "dtype": "float32"
                },
                "info_state": {
                    "start": 8,
                    "end": 20,  # Additional info fields
                    "dtype": "float32"
                }
            },
            "action": {
                "robot_action": {
                    "start": 0,
                    "end": 7,  # 7-DOF action
                    "dtype": "float32",
                    "absolute": False
                }
            },
            "video": {
                "ego_view": {
                    "original_key": "observation.images.ego_view"
                }
            },
            "annotation": {
                "task_description": {},
                "validity": {}
            }
        }
        
        return config
    
    def convert_dataset(self):
        """Convert the entire Ravens dataset to Isaac GR00T format."""
        print(f"Converting Ravens dataset from {self.ravens_data_dir} to {self.output_dir}")
        
        # Extract task name
        task_name = self.extract_task_name(str(self.ravens_data_dir))
        print(f"Detected task: {task_name}")
        
        # Find all episodes
        action_dir = self.ravens_data_dir / "action"
        if not action_dir.exists():
            print("Error: No action directory found in Ravens dataset")
            return False
        
        episode_files = list(action_dir.glob("*.pkl"))
        episode_ids = []
        
        for file in episode_files:
            # Extract episode ID from filename
            filename = file.stem
            if '-' in filename:
                episode_id = int(filename.split('-')[1])
                episode_ids.append(episode_id)
        
        episode_ids.sort()
        print(f"Found {len(episode_ids)} episodes")
        
        # Create metadata files
        self.create_metadata_files(task_name, episode_ids)
        
        # Convert each episode
        global_index = 0
        for episode_id in tqdm(episode_ids, desc="Converting episodes"):
            episode_data = self.load_ravens_episode(episode_id)
            
            if not episode_data:
                print(f"Warning: No data found for episode {episode_id}")
                continue
            
            # Create video
            if 'color' in episode_data:
                video_path = self.output_dir / "videos" / "chunk-000" / f"episode_{episode_id:06d}.mp4"
                self.create_video_from_frames(episode_data['color'], str(video_path))
            
            # Create parquet data
            parquet_data = self.create_parquet_data(episode_data, episode_id, task_name, global_index)
            
            # Save parquet file
            parquet_path = self.output_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            parquet_data.to_parquet(parquet_path, index=False)
            
            global_index += len(parquet_data)
        
        print(f"Conversion completed! Converted {len(episode_ids)} episodes")
        return True
    
    def create_metadata_files(self, task_name: str, episode_ids: List[int]):
        """Create metadata files for Isaac GR00T format."""
        
        # Create tasks.jsonl
        tasks_data = []
        for i, (task_key, task_desc) in enumerate(self.task_descriptions.items()):
            tasks_data.append({
                "task_index": i,
                "task": task_desc
            })
        
        with open(self.output_dir / "meta" / "tasks.jsonl", 'w') as f:
            for task in tasks_data:
                f.write(json.dumps(task) + '\n')
        
        # Create episodes.jsonl
        episodes_data = []
        for episode_id in episode_ids:
            # Load episode to get length
            episode_data = self.load_ravens_episode(episode_id)
            episode_length = len(episode_data.get('action', []))
            
            task_index = list(self.task_descriptions.keys()).index(task_name) if task_name in self.task_descriptions else 0
            
            episodes_data.append({
                "episode_index": episode_id,
                "tasks": [task_index],
                "length": episode_length
            })
        
        with open(self.output_dir / "meta" / "episodes.jsonl", 'w') as f:
            for episode in episodes_data:
                f.write(json.dumps(episode) + '\n')
        
        # Create modality.json
        modality_config = self.create_modality_config(task_name)
        with open(self.output_dir / "meta" / "modality.json", 'w') as f:
            json.dump(modality_config, f, indent=2)
        
        # Create info.json
        info_data = {
            "name": f"ravens_{task_name}",
            "description": f"PyBullet Ravens {task_name} task converted to Isaac GR00T format",
            "version": "1.0.0",
            "total_episodes": len(episode_ids),
            "total_frames": sum(len(self.load_ravens_episode(ep_id).get('action', [])) for ep_id in episode_ids)
        }
        
        with open(self.output_dir / "meta" / "info.json", 'w') as f:
            json.dump(info_data, f, indent=2)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert Ravens data to Isaac GR00T format")
    parser.add_argument("--ravens_data_dir", type=str, required=True,
                       help="Path to Ravens dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to output Isaac GR00T format directory")
    
    args = parser.parse_args()
    
    converter = RavensToGR00TConverter(args.ravens_data_dir, args.output_dir)
    success = converter.convert_dataset()
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
