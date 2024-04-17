#!/usr/bin/env python3

import json
import os

import numpy as np

from ariadne.msg import AriadneMap
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
import rospy
import cv2
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation
from include.utils import check_obs_exists, map_updater, msg2T, msg2pose


class Heli():

    def __init__(self):
        rospy.loginfo('Heli node started')
        self.map_publisher = rospy.Publisher('map', AriadneMap, queue_size=10)
        self.pose_publisher = rospy.Publisher('heli_pose', PoseStamped, queue_size=10)
        self.goal_reached_publisher = rospy.Publisher('goal_reached', Bool, queue_size=10)

        self.heli_speed = 1.

        self.map = AriadneMap()
        self.map.header.frame_id = 'map'
        self.map.obstacles_coordinate_list = []
        self.map.obstacles_radius_list = []
        self.map.goal = Point(25, -150, 0)  #issue goal location?
        self.bridge = CvBridge()

        # subscribe to the map_image topic
        rospy.Subscriber("map_image", Image, self.image_callback)
        
        # subscribe to the rover_map topic
        rospy.Subscriber("map_rover", AriadneMap, self.update_map)

        # Subscribe to goal topic
        rospy.Subscriber("global_goal", Point, self.update_goal)
        self.goal = []

        # subscribe to rover_pose
        rospy.Subscriber("/curiosity_mars_rover/odom", Odometry, self.update_rover_pose)
        self.rover_pose = []   

        # subscribe to move to the goal topic
        rospy.Subscriber("move_to_the_goal", Bool, self.move_to_the_goal_callback)
        self.move_to_the_goal = False

        # heli starting pose
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 30.
        # this is the orientation of the camera, should point downwards
        rot = Rotation.from_matrix([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])
        quat = rot.as_quat()
        self.pose.pose.orientation.x = quat[0]
        self.pose.pose.orientation.y = quat[1]
        self.pose.pose.orientation.z = quat[2]
        self.pose.pose.orientation.w = quat[3]

        rospy.loginfo('Heli node ok')
        
        self.goal_reached_publisher.publish(Bool(False))

        # TODO load these params from a config file
        self.H, self.W = 480., 720.
        focal = 300.0
        self.K = np.array([[focal, 0, self.W / 2.],
                           [0, focal, self.H / 2.],
                           [0, 0, 1.]])

        self.waiting_for_map = True
        while self.waiting_for_map:
            rospy.sleep(10.)
            self.publish_pose()
            rospy.loginfo('Waiting for map')


    def move_to_the_goal_callback(self, msg):
        self.move_to_the_goal = msg.data
        # rospy.loginfo(f"Move to the goal: {self.move_to_the_goal}")

    def update_rover_pose(self, msg):
        # rospy.loginfo('Received rover pose')
        self.rover_pose = msg2pose(msg)

    def update_goal(self, msg):
        # rospy.loginfo('Received goal')
        # the z is the yaw angle
        self.goal = np.array([msg.x, msg.y, msg.z])


    def update_pose(self):
        if self.rover_pose == [] or self.goal == []:
            return

        heli_pose2d = np.array([self.pose.pose.position.x, self.pose.pose.position.y, 0.])
        # make sure the heli is not far away from the rover
        if np.linalg.norm(self.rover_pose[:2] - heli_pose2d[:2]) > 60 or np.linalg.norm(self.goal[:2] - heli_pose2d[:2]) < 1.:
            # the heli is too far away from the rover
            rospy.loginfo('Heli is too far away from the rover, not moving anymore')
            self.goal_reached_publisher.publish(Bool(True))
            return

        # the drone must move towards the goal
        goal_direction = self.goal[:2] - heli_pose2d[:2]
        goal_direction = goal_direction[:2]
        goal_direction /= np.linalg.norm(goal_direction)

        # speed proportional to the distance to the goal
        dist_to_goal = np.linalg.norm(self.goal[:2] - heli_pose2d[:2])
        goal_direction *= self.heli_speed * (dist_to_goal+1.)/60.
        # move in the direction of the goal
        self.pose.pose.position.x += goal_direction[0]
        self.pose.pose.position.y += goal_direction[1]

        # rospy.loginfo(f"Moving heli to the goal: {self.pose.pose.position.x}, {self.pose.pose.position.y}")


    def publish_pose(self):
        if self.move_to_the_goal:
            # rospy.loginfo('Publishing heli pose')
            self.update_pose()
            self.pose_publisher.publish(self.pose)

            if self.goal != []:
                # check if the goal is reached
                dist_to_goal = np.linalg.norm(self.goal[:2] - np.array([self.pose.pose.position.x, self.pose.pose.position.y]))
                self.goal_reached_publisher.publish(Bool(dist_to_goal<1.0))

    def image_callback(self, msg):
        if self.waiting_for_map:
            # rospy.loginfo('Map image arrived')
            self.waiting_for_map = False
        # read image from ros image msg
        # rospy.loginfo('Received image')
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # extract obstacles from the image
        obstacles, radius = self.extract_obstacles(img)
        # rospy.loginfo(f"Obstacles:\n{obstacles.shape}, {radius.shape}")
        self.map, map_updated = map_updater(self.map, obstacles, radius)

        if map_updated or not self.move_to_the_goal:
            self.map.goal.x = self.pose.pose.position.x
            self.map.goal.y = self.pose.pose.position.y
            #
            heli_pose2d = np.array([self.pose.pose.position.x, self.pose.pose.position.y, 0.])
            goal_direction = self.goal[:2] - heli_pose2d[:2]
            goal_direction = goal_direction[:2]
            goal_direction /= np.linalg.norm(goal_direction)
            yaw = np.arctan2(goal_direction[1], goal_direction[0])
            # the z is the yaw angle
            self.map.goal.z = yaw
            self.publish_map()
        
        # move the heli towards the goal
        self.publish_pose()

    def update_map(self, msg):
        new_obs = msg.obstacles_coordinate_list
        new_radius = msg.obstacles_radius_list
        self.map, new_map = map_updater(self.map, new_obs, new_radius)
        if new_map:
            self.publish_map()

    def project_obstacles(self, obs_cam_plane, K, T):
        """ Obstacles are in the camera plane, we need to project them to the world frame
        Args:
            obs_cam_plane: list of obstacles in the camera plane, each obstacle is a tuple Nx2 (u, v)
            K: camera matrix
            T: transformation matrix from camera to world frame
        """
        obs_cam_homogenous = np.hstack([obs_cam_plane, np.ones((len(obs_cam_plane), 1))]).T

        T_inv = np.linalg.inv(T)
        P_cam = np.linalg.inv(K) @ obs_cam_homogenous
        # We assume all the points lay on a plane, the plane is at the distance of the camera height
        P_cam = P_cam * self.pose.pose.position.z  # z is like the avg depth
        P_cam = np.vstack([P_cam, np.ones((1, P_cam.shape[1]))])
        # rospy.loginfo(f"P_cam:\n{P_cam}")
        obs_world_homogenous = T_inv @ P_cam
        # print("tinv",T_inv)
        # print("homogenous",obs_world_homogenous)
        # rospy.loginfo(f"obs_world_homogenous:\n{obs_world_homogenous}")
        obs_world = (obs_world_homogenous[:3, :] / obs_world_homogenous[3, :]).T
        # rospy.loginfo(f"obs_world:\n{obs_world}")
        return obs_world

    def extract_obstacles(self, image):
        """ Extract obstacles from an image and fit a circle to each obstacle
        Args:
            image: OpenCV image
            Returns:
            list of obstacles in the camera plane, each obstacle is a tuple and obstacle ray (u, v, r)
        """
        # extract obstacles from image

        # fit a circle to each obstacle
        # Convert the image to grayscale
        # cv2.imshow('Original Image', image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # cv2.imshow('blurred', blurred)
        # Use adaptive thresholding to segment the rocks
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)

        # cv2.imshow('thr', thresh)

        # Find contours of the circles
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result = image.copy()

        # fit a circle to each obstacle
        # remove the countor that are too small
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        obstacles = []
        for contour in contours:
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            obstacles.append((center, radius))
            # print("rad1:",radius)

            # Draw the circle
            cv2.circle(result, center, radius, (0, 0, 255), 2)
            # print("center:",center)

        # Display the result
        cv2.imshow('Segmented Rocks', result)
        cv2.waitKey(10)

        # add obstacles to the map
        obs_cam_plane = [obs[0] for obs in obstacles]

        if len(obs_cam_plane) == 0:
            return [], []

        # print("cam frame:",obs_cam_plane)
        obs_world = self.project_obstacles(obs_cam_plane, self.K, msg2T(self.pose))

        # scale the ray according to the pose of the camera
        radius = np.array([self.K[0, 0] * obs[1] / self.pose.pose.position.z / 100 for obs in obstacles])  #issue conversion error?
        # print("rad2:",radius)
        # print("world frame:",obs_world)
        # stack obstacles and radius
        return obs_world, radius

    def publish_map(self):
        rospy.loginfo('Publishing map')
        self.map.header.stamp = rospy.Time.now()
        self.map_publisher.publish(self.map)


if __name__ == '__main__':
    rospy.init_node('heli')

    node = Heli()

    rospy.spin()
