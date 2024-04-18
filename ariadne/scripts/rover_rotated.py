#!/usr/bin/env python3
from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rospy
from include.RRT import RRT
from include.RRTStar import RRTStar
from include.AStar import AStar

from include.utils import map_updater, obs2array
from include.parameters import planner_algo
from matplotlib import pyplot as plt

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig


class Rover:

    def __init__(self):
        # publish path messages
        self.obstacles_coordinate_list = None
        self.obstacles_radius_list = None
        self.goal = None
        self.path_publisher = rospy.Publisher('rover_path', Path, queue_size=10)

        # publish map messages if new obstacles are detected
        self.map_publisher = rospy.Publisher('map_rover', AriadneMap, queue_size=10)

        # subscribe to the map topic
        rospy.Subscriber("map", AriadneMap, self.map_callback)

        rospy.Subscriber("heli_pose", PoseStamped, self.heli_pose_update)
        self.map = AriadneMap()
        self.map.obstacles_coordinate_list = []
        self.map.obstacles_radius_list = []

        self.planner = RRT()
        # print('here')
        self.dyn_srv = Server(AriadneConfig, self.dynamic_callback)
        self.pose = np.array([0, 0])
        self.prev_pose = np.array([0, 0])
        self.temp_goal = None

    def heli_pose_update(self, msg):
        heli_pose = msg.pose
        # read pose
        # self.prev_pose=self.pose
        self.pose = np.array([heli_pose.position.x, heli_pose.position.y])

    def dynamic_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {planner_algo}""".format(**config))
        print(planner_algo)
        if planner_algo == 'RRT':
            rospy.loginfo('Using RRT')
            self.planner = RRT()
        elif planner_algo == 'RRTStar':
            rospy.loginfo('Using RRT*')
            self.planner = RRTStar()
        else:
            rospy.loginfo('Using AStar')
            self.planner = AStar()

        return config

    def map_callback(self, msg):
        rospy.loginfo('Received map')
        self.obstacles_coordinate_list = msg.obstacles_coordinate_list
        self.obstacles_radius_list = msg.obstacles_radius_list
        # update the map
        self.map, _tf = map_updater(self.map, msg.obstacles_coordinate_list, msg.obstacles_radius_list)
        if np.linalg.norm(self.prev_pose - self.pose) < 1 or np.linalg.norm(
                self.prev_pose - self.pose) == 0:  # issue with producing duplicated obs if update too fast
            current_pose = self.pose[0:2].astype(int)
            self.goal = np.array([msg.goal.x, msg.goal.y])
            if np.all(abs(current_pose - self.goal) <= 2):
                print('Final Goal is reached!')
                plt.pause(10)
                return
            frame_width = 75
            frame_height = 35

            obstacles = np.array([obs2array(o) for o in self.map.obstacles_coordinate_list])
            x_center = obstacles[:, 0]
            y_center = obstacles[:, 1]
            round_obs = np.array([x_center, y_center, self.map.obstacles_radius_list]).T
            self.temp_goal = self.find_temporary_goal(current_pose, self.goal, frame_width, frame_height, round_obs)
            print("temp goal:", self.temp_goal)
            print("current_pose:", current_pose)
            print(f"obstacles radius list: {self.map.obstacles_radius_list}")

            path = self.planner.plan(self.pose, self.temp_goal, [obs2array(o) for o in self.map.obstacles_coordinate_list],
                                     self.map.obstacles_radius_list, show_animation=True,
                                     map_bounds=[int(current_pose[0] - 55), int(current_pose[1] - 25), int(current_pose[0] + 55),
                                                 int(current_pose[1] + 25)])
            if len(path) < 1:
                print(f'Goal seems to be unreachable or the planner failed to find a path')
                return

            self.prev_pose = path[-1][:2]
            print(f'finished rrt, final pose: {path[-1]}')
            if path.any():
                self.publish_path(path)

    def in_bounds(self, x, y, x_min, x_max, y_min, y_max):
        return x_min <= x <= x_max and y_min <= y <= y_max

    def distance_to_line(self, px, py, ax, ay, bx, by):
        # Line AB represented as a1x + b1y = c1
        a1 = by - ay
        b1 = ax - bx
        c1 = a1 * ax + b1 * ay
        # Perpendicular distance from point P to the line AB
        return abs(a1 * px + b1 * py - c1) / np.sqrt(a1 ** 2 + b1 ** 2)

    def find_temporary_goal(self, current, final, frame_width, frame_height, obstacles):
        cx, cy = current
        fx, fy = final
        obstacles = np.array(obstacles)  # [(ox, oy, radius), ...]

        # Define the view frame relative to the current position
        x_min = cx - frame_width / 2
        x_max = cx + frame_width / 2
        y_min = cy - frame_height / 2
        y_max = cy + frame_height / 2

        # Compute the direction vector towards the final goal
        direction = np.array([fx - cx, fy - cy])
        norm = np.linalg.norm(direction)
        if norm <= 2:
            return None  # Already at the goal
        direction_unit = direction / norm

        # Check for intersections with frame bounds and select the furthest point within frame
        max_distance = min(frame_width, frame_height) / 2  # conservative maximum view

        if norm <= max_distance:
            max_distance = norm - 2
        temp_goal = np.array(current) + direction_unit * max_distance

        # Adjust temporary goal to stay within bounds
        if not self.in_bounds(temp_goal[0], temp_goal[1], x_min, x_max, y_min, y_max):
            # Scale back to bounds
            if temp_goal[0] < x_min or temp_goal[0] > x_max:
                scale = min(abs((x_max - cx) / direction_unit[0]), abs((x_min - cx) / direction_unit[0]))
                temp_goal = np.array(current) + direction_unit * scale
            if temp_goal[1] < y_min or temp_goal[1] > y_max:
                scale = min(abs((y_max - cy) / direction_unit[1]), abs((y_min - cy) / direction_unit[1]))
                temp_goal = np.array(current) + direction_unit * scale

        # Avoid obstacles
        for ox, oy, radius in obstacles:
            if np.linalg.norm(temp_goal - np.array([ox, oy])) <= radius + 2:
                # Move back slightly from the obstacle
                away_vector = temp_goal - np.array([ox, oy])
                temp_goal = np.array([ox, oy]) + (away_vector / np.linalg.norm(away_vector) * (radius + 2))

        return temp_goal

    def publish_path(self, path: np.ndarray):
        """Publish the path to the rover_path topic.

        Args:
            path (list): list of waypoints in the form [x, y, z, qx, qy, qz, qw]
            """
        path_msg = Path()
        for p in path:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = 0.0
            # orientation
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = p[2]
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)

    def add_new_obstacle(self, obs, radius):
        """ Add new obstacle to the map if it does not exist and send it over the map_rover topic.

        Args:
            obs (list): list of x, y, z coordinates of the obstacle
            radius (float): radius of the obstacle
        """
        # update the existing map locally
        self.obstacles_coordinate_list.append(obs)
        self.obstacles_radius_list.append(radius)

        # update the existing map globally
        map_msg = AriadneMap()
        map_msg.header.frame_id = 'map'
        map_msg.goal = []

        map_msg.obstacles_coordinate_list = [obs]
        map_msg.obstacles_radius_list = [radius]

        self.map_publisher.publish(map_msg)


if __name__ == '__main__':
    rospy.init_node('rover')

    node = Rover()

    rospy.spin()
