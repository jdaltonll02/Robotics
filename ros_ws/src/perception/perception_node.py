#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from model import SegmentationNet
import cv2
import numpy as np

class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')
        self.bridge = CvBridge()
        self.model = SegmentationNet()
        self.model.load_state_dict(torch.load('model/segmentation_net.pth', map_location='cpu'))
        self.model.eval()
        self.sub = rospy.Subscriber('/camera/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/segmentation', Image, queue_size=1)
    def callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        img = torch.from_numpy(cv_img.transpose(2,0,1)).unsqueeze(0).float()/255.0
        with torch.no_grad():
            out = self.model(img)
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        seg_img = self.bridge.cv2_to_imgmsg(pred, encoding='mono8')
        self.pub.publish(seg_img)
if __name__ == '__main__':
    PerceptionNode()
    rospy.spin()
