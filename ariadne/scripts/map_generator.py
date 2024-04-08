#!/usr/bin/env python3
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from ariadne.msg import AriadneMap
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import rospy
import cv2
import math
from cv_bridge import CvBridge


class MapGenerator():

    def __init__(self):
        rospy.loginfo('Map generator node started')

        self.map_publisher = rospy.Publisher('map_image', Image, queue_size=10)
        self.image = None
        
        # the heli is going to send its position in the map
        rospy.Subscriber("heli_pose", PoseStamped, self.heli_request)

        # TODO load these params from a config file
        self.H, self.W = 800., 800.
        focal = 1000.0
        self.K = np.array([[focal, 0, self.H / 2.],
                      [0, focal, self.W / 2.],
                      [0, 0, 1.]])
        
        # Generate map
        np.random.seed(5)

        self.radius = []
        self.obstacleList = []
        for i in range(25):  # at least 1 obstacle
            # assuming the obstacles are all on a plane, in homogenous coordinates
            self.obstacleList.append([np.random.rand() * 150, np.random.rand() * 150, 0., 1.])
            self.radius.append(np.random.rand() * 2.0 + 1.0)
        start = [np.random.uniform(-2, 15), np.random.uniform(-2, 15), np.deg2rad(np.random.uniform(-math.pi, math.pi))]
        goal = [np.random.uniform(-2, 15), np.random.uniform(-2, 15), np.deg2rad(np.random.uniform(-math.pi, math.pi))]

        # Check they are not in the obstacles
        def check_collision(point):
            for (ox, oy, _, _), size in zip(self.obstacleList, self.radius):
                dx = abs(point[0] - ox)
                dy = abs(point[1] - oy)
                d = math.hypot(dx, dy)
                if d <= size+1.0:
                    return check_collision(
                        [np.random.uniform(-2, 15), np.random.uniform(-2, 15), np.deg2rad(np.random.uniform(-math.pi, math.pi))])
            return point

        self.start = check_collision(start)
        self.goal = check_collision(goal)

        self.bridge= CvBridge()



    def heli_request(self, msg):
        rospy.loginfo('Received heli pose')
        T_wc = np.eye(4)

        pose = msg.pose
        # read pose
        T_wc[3, 0] = pose.position.x
        T_wc[3, 1] = pose.position.y
        T_wc[3, 2] = pose.position.z
        # orientation
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        rot = Rotation.from_quat(np.array([qx, qy, qz, qw]))
        T_wc[:3, :3] = rot.as_matrix()

        self.image = self.project_obstacles(T_wc)
        self.publish_image()


    def project_obstacles(self, T_wc):
        pts_cam = self.K @ (T_wc @ np.array(self.obstacleList).T)[:3, :]
        # discard all the points out of the camera plane
        pts_cam_plane = (pts_cam[0] > 0) & (pts_cam[0] < self.H) & (pts_cam[1] > 0) & (pts_cam[1] < self.W)
        pts_cam = pts_cam[:, pts_cam_plane]
        pts_cam = pts_cam[:2, :].T
        radius_visible = np.array(self.radius)[pts_cam_plane]

        # create an image with the obstacles
        img = np.ones((int(self.H), int(self.W), 3), dtype=np.uint8) * 255
        for pt in pts_cam:
            # point size according to the radius
            cv2.circle(img, (int(pt[0]), int(pt[1])), int(radius_visible), (0, 255, 255), -1)
        
        return img

    
    def publish_image(self):
        if self.image is None:
            return
        img_msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")

        self.map_publisher.publish(img_msg)




if __name__ == '__main__':
    rospy.init_node('map_generator', log_level=rospy.DEBUG)

    node = MapGenerator()

    rospy.spin()