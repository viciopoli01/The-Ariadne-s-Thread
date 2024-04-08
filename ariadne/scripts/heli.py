#!/usr/bin/env python3
import json
import os

import numpy as np

from ariadne.msg import AriadneMap
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge


class Heli():

    def __init__(self):
        rospy.loginfo('Heli node started')
        self.map_publisher = rospy.Publisher('map', AriadneMap, queue_size=10)
        self.pose_publisher = rospy.Publisher('heli_pose', PoseStamped, queue_size=10)

        self.map = AriadneMap()
        self.map.header.frame_id = 'map'
        self.map.obstacles = []
        self.map.radius = []
        self.map.goal = Point()

        self.bridge= CvBridge()

        # subscribe to the rover_map topic
        rospy.Subscriber("map_rover", AriadneMap, self.update_map)
        rospy.Subscriber("map_image", Image, self.image_callback)

        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = -10.
        # this is the orientation of the camera, should point downwards
        self.pose.pose.orientation.x = 0.7071068
        self.pose.pose.orientation.y = 0.7071068
        self.pose.pose.orientation.z = 0
        self.pose.pose.orientation.w = 0

        # publish the pose every 10 second
        rospy.Timer(rospy.Duration(1), self.publish_pose)
    
    def publish_pose(self, event):
        rospy.loginfo('Publishing pose')
        self.pose_publisher.publish(self.pose)

    def image_callback(self, msg):
        # read image from ros image msg
        rospy.loginfo('Received image')
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # show image
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def update_map(self, msg):
        new_obs = msg.obstacles
        new_radius = msg.radius
        for obs, radius in zip(new_obs, new_radius):
            if not self.check_obs_exists(obs):
                self.map.obstacles.append(obs)
                self.map.radius.append(radius)


    def check_obs_exists(self, obs, threshold=0.1):
        if any(np.linalg.norm(np.array(obs) - np.array(o)) < threshold for o in self.map.obstacles):
            return True
        return False

    def project_obstacles(self, obs_cam_plane, K, T):
        """ Obstacles are in the camera plane, we need to project them to the world frame
        Args:
            obs_cam_plane: list of obstacles in the camera plane, each obstacle is a tuple Nx2 (u, v)
            K: camera matrix
            T: transformation matrix from camera to world frame
        """
        obs_cam_homogenous = np.hstack([obs_cam_plane, np.ones((len(obs_cam_plane), 1))])
        obs_world_homogenous = np.linalg.inv(T) @ np.linalg.inv(K) @ obs_cam_homogenous
        obs_world = [(obs_world_homogenous[0], obs_world_homogenous[1])]
        return obs_world

    def extract_obstacles(self, cv_image):
        """ Extract obstacles from an image and fit a circle to each obstacle
        Args:
            cv_image: OpenCV image
            Returns:
            list of obstacles in the camera plane, each obstacle is a tuple and obstacle ray (u, v, r)
        """
        # extract obstacles from image

        # fit a circle to each obstacle

        return []

    def publish_map(self):
        self.map.header.stamp = self.get_clock().now().to_msg()
        self.map_publisher.publish(self.map)




if __name__ == '__main__':
    rospy.init_node('heli')

    node = Heli()

    rospy.spin()