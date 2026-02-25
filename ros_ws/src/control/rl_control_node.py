#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import torch
from ppo import ActorCritic
import numpy as np
from rl_state.msg import rl_state
import yaml

class RLControlNode:
    def __init__(self):
        rospy.init_node('rl_control_node')
        config_path = rospy.get_param('~config', '../../config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        state_dim = self.config.get('rl_control', {}).get('state_dim', 100)
        action_dim = self.config.get('rl_control', {}).get('action_dim', 2)
        self.model = ActorCritic(state_dim=state_dim, action_dim=action_dim)
        self.model.load_state_dict(torch.load('ppo_actor_critic.pth', map_location='cpu'))
        self.model.eval()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/rl_state', rl_state, self.state_callback)
        self.action_scale = self.config.get('rl_control', {}).get('action_scale', [1.0, 1.0])

    def state_callback(self, msg):
        try:
            # Contract: RL state vector shape
            state = np.array(msg.state_vector, dtype=np.float32)
            if state.shape[0] != self.model.actor[0].in_features:
                rospy.logwarn(f"RL state vector shape mismatch: got {state.shape[0]}, expected {self.model.actor[0].in_features}")
            state = torch.from_numpy(state).float()
            with torch.no_grad():
                action, _ = self.model(state)
            twist = Twist()
            twist.linear.x = float(action[0]) * self.action_scale[0]
            twist.angular.z = float(action[1]) * self.action_scale[1]
            self.cmd_pub.publish(twist)
            rospy.loginfo(f"RLControl: action=({twist.linear.x:.3f}, {twist.angular.z:.3f})")
        except Exception as e:
            rospy.logerr(f"RLControlNode failure: {e}")

if __name__ == '__main__':
    RLControlNode()
    rospy.spin()
