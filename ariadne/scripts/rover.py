#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rospy

from include.utils import map_updater, obs2array
from include.planner import Planner

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig



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

        self.pose = np.array([0, 0])
        
    def dynamic_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {planner}""".format(**config))
        if config['planner'] == 0:
            self.planner = RRT()
        elif config['planner'] == 1:
            self.planner = RRTStar()
        elif config['planner'] == 2:
            self.planner = AStar()
        elif config['planner'] == 3:
            self.planner = DStar()

        return config


    def map_callback(self, msg):
        rospy.loginfo('Received map')
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        # update the map
        self.map, map_updater(self.map, msg.obstacles, msg.radius)
        self.goal = msg.goal

        path = self.planner.plan(self.pose, self.goal, [obs2array(o) for o in self.map.obstacles], self.map.radius)
        if path:
            self.publish_path(path)

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


if __name__ == '__main__':
    rospy.init_node('rover')

    node = Rover()

    rospy.spin()