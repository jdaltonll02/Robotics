# Control Package

This ROS package handles robot control:
- Reinforcement learning (PPO/SAC) for continuous control
- Integration of perception outputs into RL state
- ROS node for real-time control

## Structure
- `scripts/`: RL agent, training, and inference

## Usage
1. Train RL agent using simulation and perception outputs.
2. Run control node to send velocity commands to robot.
