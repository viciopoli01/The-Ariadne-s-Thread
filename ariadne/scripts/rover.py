#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rospy

from include.utils import map_updater, obs2array, path2msg, msg2pose
from include.planner import Planner
from include.dubins import dubins_path_planning

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import tf.transformations


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

        self.pose = np.array([0., 0., 0.]) # x, y, theta
        self.path = []
        
        self.init = False
        
        rospy.wait_for_service('/curiosity_mars_rover/mast_service')

        try:
            mast_service_proxy = rospy.ServiceProxy('/curiosity_mars_rover/mast_service', DeleteModel)
            response = mast_service_proxy(model_name='open')
            
            rospy.loginfo("Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        # subscribe to the odometry topic
        rospy.Subscriber('/curiosity_mars_rover/odom', Odometry, self.odom_callback, queue_size=1)


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
        if not self.init:
            return

        rospy.loginfo('Received map')
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        # update the map
        self.map, updated = map_updater(self.map, msg.obstacles, msg.radius)
        if not updated:
            returns
        self.goal = msg.goal

        # Mook path planning
        
        px, py, pyaw, mode, clen = dubins_path_planning(self.pose[0],self.pose[1],self.pose[2],
         24., -20., np.pi/2., 4.0)
        
        self.path = np.vstack((px, py, pyaw)).T

        # take every 10th point
        self.path = self.path[::10]
        self.path_publisher.publish(path2msg(self.path[1:]))



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

    def odom_callback(self, msg):
        self.pose = msg2pose(msg)

        # Print or use the yaw value
        # print("Yaw: {:.4f} radians".format(yaw))
        # # If you prefer degrees, you can convert it as follows:
        # yaw_deg = self.pose[2] * (180.0 / 3.1415926)
        # print("Yaw: {:.4f} degrees".format(yaw_deg))
        
        self.init = True
        

    # on node shutdown
    def shutdown(self):
        rospy.loginfo('Shutting down')
        rospy.signal_shutdown('Shutting down')

if __name__ == '__main__':
    rospy.init_node('rover', log_level=rospy.DEBUG)

    node = Rover()

    rospy.spin()