#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import torch
from ppo import ActorCritic
import numpy as np

class RLControlNode:
    def __init__(self):
        rospy.init_node('rl_control_node')
        self.model = ActorCritic(state_dim=100, action_dim=2)
        self.model.load_state_dict(torch.load('ppo_actor_critic.pth', map_location='cpu'))
        self.model.eval()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/rl_state', ... , self.state_callback)  # Define rl_state message
    def state_callback(self, msg):
        # Convert msg to state vector
        state = ... # TODO: implement state extraction
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action, _ = self.model(state)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)
if __name__ == '__main__':
    RLControlNode()
    rospy.spin()
