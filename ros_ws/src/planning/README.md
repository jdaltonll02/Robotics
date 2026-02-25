# Planning Package

This ROS package implements classical path planning:
- Occupancy grid generation from perception output
- A* or Dijkstra path planner
- ROS node for path planning

## Structure
- `scripts/`: Planning algorithms and ROS node

## Usage
1. Receive occupancy grid from perception node.
2. Compute path to goal using A* or Dijkstra.
3. Publish planned path for control node.
