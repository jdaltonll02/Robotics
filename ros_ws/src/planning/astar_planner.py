#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import heapq

class AStarPlanner:
    def __init__(self):
        rospy.init_node('astar_planner')
        self.grid_sub = rospy.Subscriber('/occupancy_grid', OccupancyGrid, self.grid_callback)
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.grid = None
    def grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        start = (0, 0)  # Replace with actual start
        goal = (msg.info.height-1, msg.info.width-1)  # Replace with actual goal
        path = self.astar(self.grid, start, goal)
        if path:
            ros_path = Path()
            ros_path.header = msg.header
            for y, x in path:
                pose = PoseStamped()
                pose.pose.position.x = x * msg.info.resolution
                pose.pose.position.y = y * msg.info.resolution
                ros_path.poses.append(pose)
            self.path_pub.publish(ros_path)
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
