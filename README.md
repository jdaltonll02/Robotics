# Robotics Research System

This repository implements a full end-to-end autonomous robotics system using ROS, CoppeliaSim, Python, PyTorch, and MATLAB. It is designed for research-quality modularity, reproducibility, and extensibility.


## Architecture Overview


```mermaid
graph TD;
	A[CoppeliaSim (Physics & Sensors)] -- /camera, /odom --> B[ROS Sensor Topics];
	B --> C[Perception Layer (CNN, PyTorch)];
	C -- /occupancy_grid --> D[World Representation Layer];
	D -- /occupancy_grid --> E[Planning Layer (A*, Hybrid A*)];
	E -- /planned_path --> F[Control Layer (PPO/PPO-CNN)];
	F -- /cmd_vel --> G[ROS Control Topics];
	G --> H[Robot Actuation];
```

All interfaces are formally documented in `INTERFACES.md`.

## Project Structure
- `coppeliasim/`: Robot model and simulation scenes (see `README.md` for build instructions)
- `ros_ws/`: ROS workspace with packages for perception, planning, and control
- `perception/`: PyTorch computer vision (semantic segmentation or depth estimation)
- `rl_control/`: PyTorch reinforcement learning (PPO/SAC)
- `mathematics/`: Mathematical documentation (kinematics, losses, RL)
- `experiments/`: Configs, scripts, experiment runner, and evaluation metrics

## Integration Plan
1. **Simulation**: Build and save the robot model and scene in CoppeliaSim (`coppeliasim/`).
2. **Perception**: Collect camera data from simulation, train a CNN (PyTorch) for semantic segmentation or depth estimation (`perception/`).
3. **ROS Integration**: Use ROS nodes to acquire camera data, run perception inference, and publish segmentation/depth results (`ros_ws/src/perception`).
4. **Planning**: Generate occupancy grid from perception output and plan paths using A* (`ros_ws/src/planning`).
5. **RL Control**: Train a PPO/SAC agent using perception outputs as part of the state, and deploy as a ROS node for continuous control (`rl_control/`, `ros_ws/src/control`).
6. **Experimentation**: Use scripts and configs in `experiments/` for reproducible training and evaluation.
7. **Mathematical Documentation**: See `mathematics/` for derivations and loss functions.


## Datasets

### Perception (Vision) Datasets
- **Simulation-generated:** Use the provided data collection scripts in `perception/` to generate datasets from CoppeliaSim. See `perception/README.md` for instructions.
- **Example public datasets:**
	- [Cityscapes](https://www.cityscapes-dataset.com/) (urban scenes, semantic segmentation)
	- [CARLA Dataset](https://carla.readthedocs.io/en/latest/datasets/) (autonomous driving simulation)
	- [Synthia](http://synthia-dataset.net/) (synthetic urban scenes)

### RL/Control Datasets
- **Simulation rollouts:** RL agents are trained using data generated in simulation. Use the experiment runner and rosbag to record and replay episodes.
- **Example public RL datasets:**
	- [D4RL](https://github.com/rail-berkeley/d4rl) (benchmark RL datasets)
	- [OpenAI Gym Environments](https://gym.openai.com/)

See each submodule's README for dataset download, generation, and usage instructions.

## Getting Started
1. Build the robot in CoppeliaSim (see `coppeliasim/README.md`).
2. Collect data and train the perception model (`perception/train.py`).
3. Train the RL controller (`rl_control/train_ppo.py`).
4. Launch the full system:
	```bash
	roslaunch ros_ws/launch/full_system.launch config:=config.yaml
	```
5. Run experiments and log metrics:
	```bash
	python experiments/experiment_runner.py --config config.yaml --seed 42
	```
6. Analyze results in `experiments/results/` and logs in `experiments/logs/`.

## Requirements
- ROS (Melodic/Noetic recommended)
- CoppeliaSim (latest)
- Python 3.8+
- PyTorch, torchvision
- OpenCV, numpy, scikit-learn
- MATLAB (for camera/dataset visualization)


## Research Standards
- All interface contracts are enforced and documented.
- All major equations are referenced in code and documented in `mathematics/`.
- Failure handling and logging are implemented for all nodes.
- Config-driven design (see `config.yaml`).
- Suitable for academic or industrial review.

See each subfolder for detailed instructions and usage.
