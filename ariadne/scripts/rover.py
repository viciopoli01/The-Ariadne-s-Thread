#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
import rospy

from include.utils import map_updater, obs2array
from include.planner import Planner

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
import math


class Rover():

    def __init__(self):
        # publish path messages
        self.path_publisher = rospy.Publisher('rover_path', Path, queue_size=10)

        # publish map messages if new obstacles are detected
        self.map_publisher = rospy.Publisher('map_rover', AriadneMap, queue_size=10)

        # subscribe to the map topic
        rospy.Subscriber("map", AriadneMap, self.map_callback)
        self.map = AriadneMap()
        self.map.obstacles = []
        self.map.radius = []

        self.dyn_srv = Server(AriadneConfig, self.dynamic_callback)
        self.planner = Planner()

        self.pose = np.array([0, 0, 0]) # x, y, theta
        self.path = [[3, 3,np.pi/36.], [3, 5, np.pi/36.], [2, 4, np.pi/36.]]

        # PD controller
        self.kp = 4.5
        self.kd = 1.5
        self.last_error = 0
        self.next_point = 0

        rospy.wait_for_service('/curiosity_mars_rover/mast_service')

        try:
            mast_service_proxy = rospy.ServiceProxy('/curiosity_mars_rover/mast_service', DeleteModel)
            response = mast_service_proxy(model_name='open')
            
            rospy.loginfo("Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        # publisher cmd_vel messages
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


    def dynamic_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {planner_algo}""".format(**config))

        if config['planner_algo'] == -1:
            rospy.loginfo('Using MyCustomPlanner')
            self.planner = Planner()
        elif config['planner_algo'] == 0:
            from include.RRT import RRT
            rospy.loginfo('Using RRT')
            self.planner = RRT()
        elif config['planner_algo'] == 1:
            from include.RRTStar import RRTStar
            rospy.loginfo('Using RRT*')
            self.planner = RRTStar()
        elif config['planner_algo'] == 2:
            from include.AStar import AStar
            rospy.loginfo('Using A*')
            self.planner = AStar()
        elif config['planner_algo'] == 3:
            from include.DStar import DStar
            rospy.loginfo('Using D*')
            self.planner = DStar()

        return config


    def map_callback(self, msg):
        rospy.loginfo('Received map')
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        # update the map
        self.map, map_updater(self.map, msg.obstacles, msg.radius)
        self.goal = msg.goal

        # self.path = self.planner.plan(self.pose, self.goal, [obs2array(o) for o in self.map.obstacles], self.map.radius)
        # if self.path:
        #     self.publish_path(self.path)


    def publish_path(self, path):
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
        self.obstacles.append(obs)
        self.radius.append(radius)

        # update the existing map globally
        map_msg = AriadneMap()
        map_msg.header.frame_id = 'map'
        map_msg.goal = []

        map_msg.obstacles = [obs]
        map_msg.radius = [radius]

        self.map_publisher.publish(map_msg)


    def model_states_callback(self, msg):
        model_names = msg.name
        model_positions = msg.pose
        

        # Find the index of 'curiosity_mars_rover' in the list of model names
        try:
            curiosity_index = model_names.index('curiosity_mars_rover')
            # Get the position of 'curiosity_mars_rover'
            curiosity_position = model_positions[curiosity_index].position

            # get the yaw angle of the rover
            curiosity_orientation = model_positions[curiosity_index].orientation
            qx = curiosity_orientation.x
            qy = curiosity_orientation.y
            qz = curiosity_orientation.z
            qw = curiosity_orientation.w
            yaw_rad = np.arctan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy**2 + qz**2))

            yaw_deg = np.degrees(yaw_rad)
            yaw_deg = yaw_deg % np.pi
            yaw_rad = np.radians(yaw_deg)

            self.pose = np.array([curiosity_position.x, curiosity_position.y, yaw_rad])
            
            self.controller(self.pose)
            # rospy.loginfo("'curiosity_mars_rover' position: {}".format(curiosity_position))
        except ValueError:
            rospy.logwarn("'curiosity_mars_rover' not found in model_states")


    def controller(self, current_pose):
        # implement the controller here
        # it has to reach the node position (x, y, theta) given the pose of the rover
        # send out cmd_vel messages to the rover
        
        # check if pose is close enough to the goal
        goal_pose = self.path[self.next_point]
        rospy.loginfo('Goal pose: {}'.format(goal_pose))
        if np.linalg.norm(goal_pose[:2] - current_pose[:2]) < 0.5:
            rospy.loginfo('Reached node, moving to next node')
            self.next_point += 1
            if self.next_point >= len(self.path):
                rospy.loginfo('Goal reached')
                return

        goal_pose = self.path[self.next_point]

        # Calculate errors
        error_x = goal_pose[0] - current_pose[0]
        error_y = goal_pose[1] - current_pose[1]
        error_theta = goal_pose[2] - current_pose[2]

        # Ensure the angle error is within -pi to pi range
        error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))

        # Compute control signals
        control_x = self.kp * error_x + self.kd * (error_x - self.last_error)
        control_y = self.kp * error_y + self.kd * (error_y - self.last_error)
        control_theta = self.kp * error_theta + self.kd * (error_theta - self.last_error)

        # Update last error for next iteration
        self.last_error = error_theta
        
        cmd_vel = Twist()
        cmd_vel.linear.x = control_x
        cmd_vel.angular.z = control_y
        cmd_vel.angular.z = control_theta

        # rospy.loginfo('Sending cmd_vel: {}'.format(cmd_vel))

        self.cmd_vel_publisher.publish(cmd_vel)

if __name__ == '__main__':
    rospy.init_node('rover')

    node = Rover()

    rospy.spin()