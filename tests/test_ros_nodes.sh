#!/bin/bash
# Basic ROS node launch/health check
set -e
source ../ros_ws/devel/setup.bash
roslaunch ros_ws/launch/full_system.launch &
PID=$!
sleep 10
rosnode list | grep /perception_node
rosnode list | grep /astar_planner
rosnode list | grep /rl_control_node
kill $PID
