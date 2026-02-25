# Robotics Research System

This repository implements a full end-to-end autonomous robotics system using ROS, CoppeliaSim, Python, PyTorch, and MATLAB. It is designed for research-quality modularity, reproducibility, and extensibility.

## Project Structure
- `coppeliasim/`: Robot model and simulation scenes (see `README.md` for build instructions)
- `ros_ws/`: ROS workspace with packages for perception, planning, and control
- `perception/`: PyTorch computer vision (semantic segmentation or depth estimation)
- `rl_control/`: PyTorch reinforcement learning (PPO/SAC)
- `mathematics/`: Mathematical documentation (kinematics, losses, RL)
- `experiments/`: Configs, scripts, and evaluation metrics

## Integration Plan
1. **Simulation**: Build and save the robot model and scene in CoppeliaSim (`coppeliasim/`).
2. **Perception**: Collect camera data from simulation, train a CNN (PyTorch) for semantic segmentation or depth estimation (`perception/`).
3. **ROS Integration**: Use ROS nodes to acquire camera data, run perception inference, and publish segmentation/depth results (`ros_ws/src/perception`).
4. **Planning**: Generate occupancy grid from perception output and plan paths using A* (`ros_ws/src/planning`).
5. **RL Control**: Train a PPO/SAC agent using perception outputs as part of the state, and deploy as a ROS node for continuous control (`rl_control/`, `ros_ws/src/control`).
6. **Experimentation**: Use scripts and configs in `experiments/` for reproducible training and evaluation.
7. **Mathematical Documentation**: See `mathematics/` for derivations and loss functions.

## Getting Started
1. Build the robot in CoppeliaSim (see `coppeliasim/README.md`).
2. Collect data and train the perception model (`perception/train.py`).
3. Train the RL controller (`rl_control/train_ppo.py`).
4. Launch ROS nodes for perception, planning, and control.
5. Run experiments using `experiments/run_experiment.sh`.

## Requirements
- ROS (Melodic/Noetic recommended)
- CoppeliaSim (latest)
- Python 3.8+
- PyTorch, torchvision
- OpenCV, numpy, scikit-learn
- MATLAB (for camera/dataset visualization)

See each subfolder for detailed instructions and usage.
