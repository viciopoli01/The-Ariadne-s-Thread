#!/usr/bin/env python3

import json
import os

import numpy as np

from ariadne.msg import AriadneMap
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
import rospy
import cv2
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation
from include.utils import check_obs_exists, map_updater

class Heli():

    def __init__(self):
        rospy.loginfo('Heli node started')
        self.map_publisher = rospy.Publisher('map', AriadneMap, queue_size=10)
        self.pose_publisher = rospy.Publisher('heli_pose', PoseStamped, queue_size=10)

        self.map = AriadneMap()
        self.map.header.frame_id = 'map'
        self.map.obstacles = []
        self.map.radius = []
        self.map.goal = Point(25,-150,0)#issue goal location?
        # self.map.temp_goal=np.array([0,0])
        self.bridge= CvBridge()

        # subscribe to the rover_map topic
        rospy.Subscriber("map_rover", AriadneMap, self.update_map)
        rospy.Subscriber("map_image", Image, self.image_callback)

        

        rospy.Subscriber("rover_path", Path, self.update_pose)

        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 30.
        # this is the orientation of the camera, should point downwards
        rot = Rotation.from_matrix([[1, 0, 0], 
                                    [0, -1, 0], 
                                    [0, 0, -1]])
        # rot = Rotation.from_matrix([[0, 1, 0], 
        #                             [1, 0, 0], 
        #                             [0, 0, -1]])
        quat = rot.as_quat()
        self.pose.pose.orientation.x = quat[0]
        self.pose.pose.orientation.y = quat[1]
        self.pose.pose.orientation.z = quat[2]
        self.pose.pose.orientation.w = quat[3]

        rospy.loginfo('Heli node ok')
        # publish the pose every 10 second
        self.pose_publisher.publish(self.pose)
        
        rospy.Timer(rospy.Duration(0.1), self.publish_pose)
        # rospy.Timer(rospy.Duration(2))

        # TODO load these params from a config file
        self.H, self.W = 480., 720.
        focal = 300.0
        self.K = np.array([[focal, 0, self.W / 2.],
                      [0, focal, self.H / 2.],
                      [0, 0, 1.]])
    
    def msg2T(self, msg):
        T = np.eye(4)
        pose = msg.pose
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        # orientation
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        rot = Rotation.from_quat(np.array([qx, qy, qz, qw]))
        T[:3, :3] = rot.as_matrix()
        T[:3, :3] = np.linalg.inv(rot.as_matrix())
        T[:3, 3] = -np.linalg.inv(rot.as_matrix())@ (T[:3, 3])
        return T
    
    def update_pose(self, path):
        rospy.loginfo('Publishing pose due to new path')
        print(path.poses[-1].pose.position.x)
        self.pose.pose.position.x=path.poses[-1].pose.position.x
        self.pose.pose.position.y=path.poses[-1].pose.position.y
        # self.pose.pose.position.z = 30.
        # self.pose.pose.position.z=path.poses[-1].pose.position.z
        # self.pose_publisher.publish(self.pose)
        # move along a line in the x direction
        # self.pose.pose.position.x -= 10.0
        # self.map.goal.x-=15

    def publish_pose(self, event):
        rospy.loginfo('Publishing pose')
        self.pose_publisher.publish(self.pose)
        # move along a line in the x direction
        # self.pose.pose.position.x -= 10.0
        # self.map.goal.x-=15


    def image_callback(self, msg):
        # read image from ros image msg
        rospy.loginfo('Received image')
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # extract obstacles from the image
        obstacles, radius = self.extract_obstacles(img)  
        # rospy.loginfo(f"Obstacles:\n{obstacles.shape}, {radius.shape}")
        self.map, map_updated = map_updater(self.map, obstacles, radius)
        
        if map_updated:
            self.publish_map()

    def update_map(self, msg):
        new_obs = msg.obstacles
        new_radius = msg.radius
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
        P_cam = P_cam * self.pose.pose.position.z # z is like the avg depth
        P_cam = np.vstack([P_cam, np.ones((1, P_cam.shape[1]))])
        # rospy.loginfo(f"P_cam:\n{P_cam}")
        obs_world_homogenous = T_inv@ P_cam
        # print("tinv",T_inv)
        # print("homogenous",obs_world_homogenous)
        # rospy.loginfo(f"obs_world_homogenous:\n{obs_world_homogenous}")
        obs_world = (obs_world_homogenous[:3, :]/obs_world_homogenous[3, :]).T
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
        
        # print("cam frame:",obs_cam_plane)
        obs_world = self.project_obstacles(obs_cam_plane, self.K, self.msg2T(self.pose))

        # scale the ray according to the pose of the camera
        radius = np.array([self.K[0, 0]*obs[1]/self.pose.pose.position.z/100 for obs in obstacles]) #issue conversion error?
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