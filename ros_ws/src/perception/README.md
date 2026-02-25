# Perception Package

This ROS package handles computer vision for the robot. It includes:
- Data collection from CoppeliaSim camera sensors
- CNN-based semantic segmentation (PyTorch)
- ROS node for real-time inference

## Structure
- `data/`: Collected images and labels
- `model/`: PyTorch model definition and weights
- `scripts/`: Training and inference scripts

## Usage
1. Collect data using the provided ROS node.
2. Train the CNN using `train.py`.
3. Run the perception node for real-time segmentation.
