# VLA Synthetic Data Generation for Isaac GR00T

This project generates synthetic ground truth data using PyBullet Ravens and converts it to the Isaac GR00T LeRobot compatible format for training Vision-Language-Action (VLA) models.

## Overview

The system consists of three main components:

1. **PyBullet Ravens**: Generates synthetic manipulation demonstrations
2. **Data Converter**: Transforms Ravens data to Isaac GR00T format
3. **Training Pipeline**: Ready-to-use data for VLA model training

## Features

- **Multiple Manipulation Tasks**: Block insertion, stacking, towers of hanoi, and more
- **Rich Visual Data**: RGB and depth images from multiple camera viewpoints
- **Language Annotations**: Natural language task descriptions
- **Isaac GR00T Compatible**: Direct compatibility with NVIDIA's Isaac GR00T framework
- **Scalable**: Generate thousands of demonstrations automatically

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/trustmlyoungscientist/dataset_for_VLA.git

# Run the setup script
python setup_environment.py
```

### 2. Generate Synthetic Data

```bash
# Generate data for all tasks (recommended for first run)
python generate_synthetic_data.py --train_episodes 50 --test_episodes 10

# Or generate data for a specific task
python generate_synthetic_data.py --task block-insertion --train_episodes 100 --test_episodes 20
```

### 3. Convert to Isaac GR00T Format

```bash
# Convert a specific dataset
python convert_to_groot_format.py \
    --ravens_data_dir ./synthetic_data/block-insertion-train \
    --output_dir ./synthetic_data/block-insertion-train-groot

# Convert all datasets (run this for each task/mode combination)
for task in block-insertion place-red-in-green towers-of-hanoi; do
    for mode in train test; do
        python convert_to_groot_format.py \
            --ravens_data_dir ./synthetic_data/${task}-${mode} \
            --output_dir ./synthetic_data/${task}-${mode}-groot
    done
done
```

## Available Tasks

The system supports the following manipulation tasks:

| Task | Description | Complexity |
|------|-------------|------------|
| `block-insertion` | Insert L-shaped block into fixture | Low |
| `place-red-in-green` | Place red block in green zone | Low |
| `towers-of-hanoi` | Solve towers of hanoi puzzle | Medium |
| `stack-block-pyramid` | Stack blocks to form pyramid | Medium |
| `align-box-corner` | Align box with corner | Low |
| `assembling-kits` | Assemble kit by placing objects | High |
| `manipulating-rope` | Manipulate rope to achieve goal | High |
| `packing-boxes` | Pack boxes in container | Medium |
| `palletizing-boxes` | Palletize boxes in correct order | Medium |
| `sweeping-piles` | Sweep pile of objects | Medium |

## Data Format

### Ravens Format (Input)
```
synthetic_data/
├── block-insertion-train/
│   ├── action/          # Action sequences (.pkl files)
│   ├── color/           # RGB images (.pkl files)
│   ├── depth/           # Depth images (.pkl files)
│   ├── reward/          # Reward signals (.pkl files)
│   └── info/            # Additional info (.pkl files)
```

### Isaac GR00T Format (Output)
```
synthetic_data/
├── block-insertion-train-groot/
│   ├── meta/
│   │   ├── episodes.jsonl    # Episode metadata
│   │   ├── tasks.jsonl       # Task descriptions
│   │   ├── modality.json     # Data modality configuration
│   │   └── info.json         # Dataset information
│   ├── videos/
│   │   └── chunk-000/
│   │       └── episode_000001.mp4  # Video observations
│   └── data/
│       └── chunk-000/
│           └── episode_000001.parquet  # State/action data
```

## Configuration

### Modality Configuration

The `modality.json` file defines how data is structured:

```json
{
    "state": {
        "action_state": {
            "start": 0,
            "end": 7,
            "dtype": "float32"
        },
        "reward_state": {
            "start": 7,
            "end": 8,
            "dtype": "float32"
        }
    },
    "action": {
        "robot_action": {
            "start": 0,
            "end": 7,
            "dtype": "float32",
            "absolute": false
        }
    },
    "video": {
        "ego_view": {
            "original_key": "observation.images.ego_view"
        }
    }
}
```

### Task Descriptions

Each task includes natural language descriptions:

```json
{"task_index": 0, "task": "Insert the L-shaped block into the fixture"}
{"task_index": 1, "task": "Place the red block in the green zone"}
{"task_index": 2, "task": "Move blocks to solve the towers of hanoi puzzle"}
```

## Usage with Isaac GR00T

Once converted, the data can be used directly with Isaac GR00T:

```python
from lerobot import LeRobotDataset

# Load the converted dataset
dataset = LeRobotDataset("synthetic_data/block-insertion-train-groot")

# Access episodes
for episode in dataset.episodes:
    print(f"Episode {episode['episode_index']}: {episode['length']} steps")
    
    # Access data
    episode_data = dataset[episode['episode_index']]
    observations = episode_data['observation']
    actions = episode_data['action']
    tasks = episode_data['task']
```

## Advanced Usage

### Custom Task Generation

You can modify the task descriptions in `convert_to_groot_format.py`:

```python
self.task_descriptions = {
    "block-insertion": "Your custom task description here",
    # ... other tasks
}
```

### Custom Modality Configuration

Modify the `create_modality_config` method to match your specific robot configuration:

```python
def create_modality_config(self, task_name: str) -> Dict[str, Any]:
    config = {
        "state": {
            "joint_positions": {"start": 0, "end": 7},
            "joint_velocities": {"start": 7, "end": 14},
            # ... add your specific state modalities
        },
        "action": {
            "joint_commands": {"start": 0, "end": 7},
            # ... add your specific action modalities
        }
    }
    return config
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've activated the conda environment:
   ```bash
   conda activate ravens
   ```

2. **Display Issues**: If you encounter display problems, run without display:
   ```bash
   python generate_synthetic_data.py --display=False
   ```

3. **Memory Issues**: Reduce the number of episodes:
   ```bash
   python generate_synthetic_data.py --train_episodes 10 --test_episodes 5
   ```

4. **Asset Path Issues**: Ensure the assets path is correct:
   ```bash
   python generate_synthetic_data.py --assets_root ./dataset_for_VLA/ravens/environments/assets/
   ```

### Validation

Check if your generated data is valid:

```bash
# The generation script automatically validates data
python generate_synthetic_data.py --task block-insertion

# Manual validation
python -c "
from convert_to_groot_format import RavensToGR00TConverter
converter = RavensToGR00TConverter('./synthetic_data/block-insertion-train', './test_output')
print('Validation passed!' if converter.convert_dataset() else 'Validation failed!')
"
```

## Performance Tips

1. **Batch Processing**: Generate data for multiple tasks in parallel
2. **Storage**: Use SSD storage for faster I/O
3. **Memory**: Monitor memory usage during large dataset generation
4. **Display**: Disable display for faster generation on headless systems

## Contributing

To add new tasks or improve the system:

1. Add new task definitions to the Ravens environment
2. Update task descriptions in the converter
3. Modify modality configuration as needed
4. Test with a small number of episodes first

## License

This project uses the Ravens codebase which is licensed under the Apache License 2.0. See the original repository for details.

## References

- [Ravens: Learning to Grasp the World](https://github.com/google-research/ravens)
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
