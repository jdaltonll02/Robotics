#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import heapq

class AStarPlanner:
    def __init__(self):
        rospy.init_node('astar_planner')
        import yaml
        self.config_path = rospy.get_param('~config', 'config.yaml')
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.expected_shape = (
            self.config.get('occupancy_grid', {}).get('height', 128),
            self.config.get('occupancy_grid', {}).get('width', 128)
        )
        self.expected_frame = self.config.get('occupancy_grid', {}).get('frame_id', 'map')
        self.grid_sub = rospy.Subscriber('/occupancy_grid', OccupancyGrid, self.grid_callback)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.grid = None
    def grid_callback(self, msg):
        # Contract validation
        if (msg.info.height, msg.info.width) != self.expected_shape:
            rospy.logwarn(f"OccupancyGrid shape mismatch: got {(msg.info.height, msg.info.width)}, expected {self.expected_shape}")
        if msg.header.frame_id != self.expected_frame:
            rospy.logwarn(f"OccupancyGrid frame mismatch: got {msg.header.frame_id}, expected {self.expected_frame}")
        import time
        start_time = time.time()
        try:
            self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            if np.all(np.array(msg.data) == -1):
                rospy.logwarn("Received empty occupancy grid (all unknown). Skipping planning.")
                return
            # See /mathematics/kinematics_dynamics.md for coordinate conventions
            start = (0, 0)  # Replace with actual start
            goal = (msg.info.height-1, msg.info.width-1)  # Replace with actual goal
            path = self.astar(self.grid, start, goal)
            latency = (time.time() - start_time) * 1000.0
            if path:
                ros_path = Path()
                ros_path.header = msg.header
                for y, x in path:
                    pose = PoseStamped()
                    pose.pose.position.x = x * msg.info.resolution
                    pose.pose.position.y = y * msg.info.resolution
                    ros_path.poses.append(pose)
                self.path_pub.publish(ros_path)
                rospy.loginfo(f"A* planner succeeded. Planning latency: {latency:.2f} ms")
            else:
                rospy.logwarn(f"A* planner failed to find a path. Planning latency: {latency:.2f} ms")
        except Exception as e:
            rospy.logerr(f"Planning node failure: {e}")
    def astar(self, grid, start, goal):
        h = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = [(0 + h(start, goal), 0, start, [])]
        closed = set()
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in closed:
                continue
            path = path + [current]
            if current == goal:
                return path
            closed.add(current)
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = current[0]+dy, current[1]+dx
                if 0<=ny<grid.shape[0] and 0<=nx<grid.shape[1]:
                    if grid[ny, nx] == 0 and (ny, nx) not in closed:
                        heapq.heappush(open_set, (cost+1+h((ny,nx),goal), cost+1, (ny,nx), path))
        return None
if __name__ == '__main__':
    AStarPlanner()
    rospy.spin()
