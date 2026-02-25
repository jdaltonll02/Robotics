#!/bin/bash
# Run perception training
cd ../perception && python3 train.py
# Run RL training
cd ../rl_control && python3 train_ppo.py
# (Add ROS launch commands as needed)
