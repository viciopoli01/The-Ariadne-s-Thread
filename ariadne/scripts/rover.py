#!/usr/bin/env python3
from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

import rospy
from include.RRT import RRT
from include.RRTStar import RRTStar
from include.AStar import AStar

from include.utils import map_updater, obs2array, path2msg, msg2pose

from include.planner import Planner
from include.dubins import dubins_path_planning
from include.parameters import planner_algo
from matplotlib import pyplot as plt

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import tf.transformations

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

        self.map = AriadneMap()
        self.map.obstacles_coordinate_list = []
        self.map.obstacles_radius_list = []

        self.planner = Planner()

        self.dyn_srv = Server(AriadneConfig, self.dynamic_callback)
        
        rospy.wait_for_service('/curiosity_mars_rover/mast_service')

        try:
            mast_service_proxy = rospy.ServiceProxy('/curiosity_mars_rover/mast_service', DeleteModel)
            response = mast_service_proxy(model_name='open')
            
            rospy.loginfo("Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        # subscribe to the odometry topic
        rospy.Subscriber('/curiosity_mars_rover/odom', Odometry, self.odom_callback, queue_size=1)

        self.pose = None

        # publish move to the goal
        self.move_to_the_goal_publisher = rospy.Publisher('move_to_the_goal', Bool, queue_size=10)
        self.heli_move = True
        self.can_update_map = False

        # subscribe to goal_reached
        rospy.Subscriber('goal_reached', Bool, self.goal_reached_callback)

        rospy.Subscriber('rover_reached_goal', Bool, self.rover_reached_goal)
        # 1 Hz
        rospy.Timer(rospy.Duration(1.), lambda event: self.move_to_the_goal_publisher.publish(Bool(self.heli_move)))


        # odometry publisher
        self.odom_publisher = rospy.Publisher('/curiosity_mars_rover/odom_1', Odometry, queue_size=10)


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
        
        self.planner = RRTStar()
        return config

    def map_callback(self, msg):
        if self.pose is None or not self.can_update_map:
            return

        rospy.loginfo('Received map')
        # update the map
        self.map, map_updated = map_updater(self.map, msg.obstacles_coordinate_list, msg.obstacles_radius_list)
        
        if map_updated:
            rospy.loginfo('Map updated')

            self.obstacles_coordinate_list = msg.obstacles_coordinate_list
            self.obstacles_radius_list = msg.obstacles_radius_list

            self.goal = np.array([msg.goal.x, msg.goal.y, msg.goal.z])
            if np.linalg.norm(self.pose[:2] - self.goal[:2]) < 1.:
                rospy.loginfo('Final Goal is reached!')
                plt.pause(10)
                return
            frame_width = 75
            frame_height = 35


            # path = self.planner.plan(self.pose, self.goal, [obs2array(o) for o in self.map.obstacles_coordinate_list],
            #                          self.map.obstacles_radius_list, show_animation=True,
            #                          map_bounds=[int( - 50), 
            #                                     int(- 50), 
            #                                     int(+ 50),
            #                                     int(+ 50)])


            obstacles = np.array([obs2array(o) for o in self.map.obstacles_coordinate_list])
            x_center = obstacles[:, 0]
            y_center = obstacles[:, 1]
            round_obs = np.array([x_center, y_center, self.map.obstacles_radius_list]).T
            # self.temp_goal = self.find_temporary_goal(current_pose, self.goal, frame_width, frame_height, round_obs)
            print("Goal:", self.goal)
            print("current_pose:", self.pose)
            print(f"obstacles radius list: {self.map.obstacles_radius_list}")
            print(f"obstacles coord list: {self.map.obstacles_coordinate_list}")
            
            # rotation from pose to goal
            yaw = np.arctan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
            # rotation matrix
            R = Rotation.from_euler('z', yaw).as_matrix().T 
            R_2d = R[:2, :2]

            print(f"R: {R_2d}")
            # rotate the obstacles
            rotated_obs = np.dot(R_2d, obstacles[:, :2].T).T

            # rotate the goal
            rotated_goal = np.dot(R_2d, self.goal[:2])
            rotated_goal = np.array([rotated_goal[0], rotated_goal[1], self.goal[2]-yaw])
            
            # rotate the current pose
            rotated_pose = np.dot(R_2d, self.pose[:2])
            rotated_pose = np.array([rotated_pose[0], rotated_pose[1], self.pose[2]-yaw])

            x_min = np.min(rotated_obs[:, 0])
            x_max = np.max(rotated_obs[:, 0])
            y_min = np.min(rotated_obs[:, 1])
            y_max = np.max(rotated_obs[:, 1])

            # include the goal and the self.pose in the frame
            x_min = min(x_min, rotated_goal[0])
            x_max = max(x_max, rotated_goal[0])
            y_min = min(y_min, rotated_goal[1])
            y_max = max(y_max, rotated_goal[1])

            x_min = min(x_min, self.pose[0])
            x_max = max(x_max, self.pose[0])
            y_min = min(y_min, self.pose[1])
            y_max = max(y_max, self.pose[1])

            path = self.planner.plan(rotated_pose, rotated_goal, rotated_obs,
                                     self.map.obstacles_radius_list, show_animation=False,
                                     map_bounds=[int(x_min - 5), 
                                                int(y_min - 5), 
                                                int(x_max + 5),
                                                int(y_max + 5)])

            if len(path) < 1:
                rospy.logwarn('Goal seems to be unreachable or the planner failed to find a path')
                return

            # derotate the path. 
            # the path is N x 7, where N is the number of waypoints
            # and the 7 columns are x, y, z, qx, qy, qz, qw
            for i in range(len(path)):
                path[i][0], path[i][1] = np.dot(R_2d.T, path[i][:2])
                R_p = Rotation.from_quat([path[i][3], path[i][4], path[i][5], path[i][6]])
                # derotate the orientation
                R_p = R_p * Rotation.from_euler('z', yaw)
                q = R_p.as_quat()
                path[i][3] = q[0]
                path[i][4] = q[1]
                path[i][5] = q[2]
                path[i][6] = q[3]

            # plot the path
            path = np.array(path)
            # print the first point as a green star
            plt.plot(path[0, 0], path[0, 1], 'g*')
            # scatter the path waypoints
            plt.scatter(path[:, 0], path[:, 1], c='r', s=10)
            plt.plot(rotated_obs[:, 0], rotated_obs[:, 1], 'bo')
            plt.plot(rotated_goal[0], rotated_goal[1], 'go')
            plt.plot(rotated_pose[0], rotated_pose[1], 'mo')
            plt.pause(0.01)
            plt.show()

            rospy.loginfo(f'finished rrt, final pose: {path[-1]}')
            if path.any():
                self.can_update_map = False
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
            pose.pose.position.z = p[2]
            # orientation
            pose.pose.orientation.x = p[3]
            pose.pose.orientation.y = p[4]
            pose.pose.orientation.z = p[5]
            pose.pose.orientation.w = p[6]
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

    def odom_callback(self, msg):
        self.pose = msg2pose(msg)
        
        # self.pose to odom_publisher
        new_msg = Odometry()
        new_msg.header = msg.header
        new_msg.pose.pose.position.x = self.pose[0]
        new_msg.pose.pose.position.y = self.pose[1]
        new_msg.pose.pose.position.z = 0.0
        yaw_2_quat = tf.transformations.quaternion_from_euler(0, 0, self.pose[2])
        new_msg.pose.pose.orientation.x = yaw_2_quat[0]
        new_msg.pose.pose.orientation.y = yaw_2_quat[1]
        new_msg.pose.pose.orientation.z = yaw_2_quat[2]
        new_msg.pose.pose.orientation.w = yaw_2_quat[3]
        self.odom_publisher.publish(new_msg)

    def goal_reached_callback(self, msg):
        if msg.data:
            rospy.loginfo('Heli reached the goal')
            self.heli_move = False    
            self.can_update_map = True

    def rover_reached_goal(self, msg):
        if msg.data:
            rospy.loginfo('Rover reached goal')
            self.heli_move = True
            self.can_update_map = False

    # on node shutdown
    def shutdown(self):
        rospy.loginfo('Shutting down')
        rospy.signal_shutdown('Shutting down')

if __name__ == '__main__':
    rospy.init_node('rover', log_level=rospy.DEBUG)

    node = Rover()

    rospy.spin()
