#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from model import SegmentationNet
import cv2
import numpy as np

import yaml
from nav_msgs.msg import OccupancyGrid
import time

class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')
        self.bridge = CvBridge()
        self.model = SegmentationNet()
        self.model.load_state_dict(torch.load('model/segmentation_net.pth', map_location='cpu'))
        self.model.eval()
        # Load config
        config_path = rospy.get_param('~config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.grid_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=1)
        self.seg_pub = rospy.Publisher('/segmentation', Image, queue_size=1)
        self.sub = rospy.Subscriber('/camera/image_raw', Image, self.callback)
        self.frame_id = self.config.get('occupancy_grid', {}).get('frame_id', 'map')
        self.resolution = self.config.get('occupancy_grid', {}).get('resolution', 0.05)
        self.height = self.config.get('occupancy_grid', {}).get('height', 128)
        self.width = self.config.get('occupancy_grid', {}).get('width', 128)
        self.rate = self.config.get('occupancy_grid', {}).get('rate', 10)
        self.last_pub_time = 0

    def callback(self, msg):
        start_time = time.time()
        try:
            # See /mathematics/perception_loss.md for cross-entropy loss definition
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            img = torch.from_numpy(cv_img.transpose(2,0,1)).unsqueeze(0).float()/255.0
            with torch.no_grad():
                out = self.model(img)
                pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            seg_img = self.bridge.cv2_to_imgmsg(pred, encoding='mono8')
            self.seg_pub.publish(seg_img)
            # Generate occupancy grid (0: free, 100: occupied, -1: unknown)
            grid = np.full((self.height, self.width), -1, dtype=np.int8)
            # Assume pred==1 is occupied, pred==0 is free
            h, w = min(pred.shape[0], self.height), min(pred.shape[1], self.width)
            grid[:h, :w] = np.where(pred[:h, :w] == 1, 100, 0)
            # Publish at configured rate
            now = time.time()
            if now - self.last_pub_time >= 1.0/self.rate:
                occ_msg = OccupancyGrid()
                occ_msg.header.stamp = rospy.Time.now()
                occ_msg.header.frame_id = self.frame_id
                occ_msg.info.resolution = self.resolution
                occ_msg.info.width = self.width
                occ_msg.info.height = self.height
                occ_msg.info.origin.position.x = 0.0
                occ_msg.info.origin.position.y = 0.0
                occ_msg.info.origin.position.z = 0.0
                occ_msg.info.origin.orientation.w = 1.0
                occ_msg.data = grid.flatten().tolist()
                self.grid_pub.publish(occ_msg)
                self.last_pub_time = now
            # Metrics logging
            latency = (time.time() - start_time) * 1000.0
            if not hasattr(self, 'frame_count'):
                self.frame_count = 0
                self.start_fps_time = time.time()
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_fps_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                rospy.loginfo(f"Perception latency: {latency:.2f} ms, FPS: {fps:.2f}")
        except Exception as e:
            rospy.logwarn(f"PerceptionNode failure: {e}")
            # Publish empty grid with all -1
            occ_msg = OccupancyGrid()
            occ_msg.header.stamp = rospy.Time.now()
            occ_msg.header.frame_id = self.frame_id
            occ_msg.info.resolution = self.resolution
            occ_msg.info.width = self.width
            occ_msg.info.height = self.height
            occ_msg.info.origin.position.x = 0.0
            occ_msg.info.origin.position.y = 0.0
            occ_msg.info.origin.position.z = 0.0
            occ_msg.info.origin.orientation.w = 1.0
            occ_msg.data = [-1] * (self.height * self.width)
            self.grid_pub.publish(occ_msg)
            rospy.loginfo("Published empty occupancy grid due to perception failure.")
if __name__ == '__main__':
    PerceptionNode()
    rospy.spin()
