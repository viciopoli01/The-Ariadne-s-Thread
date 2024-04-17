#!/usr/bin/env python3

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from ariadne.msg import AriadneMap
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
import rospy
import cv2
import math
from cv_bridge import CvBridge
from gazebo_msgs.srv import SpawnModel

class MapGenerator():

    def __init__(self):
        rospy.loginfo('Map generator node started')

        self.map_publisher = rospy.Publisher('map_image', Image, queue_size=10)
        self.goal_publisher = rospy.Publisher("global_goal", Point, queue_size=10)
        self.image = None
        
        # the heli is going to send its position in the map
        rospy.Subscriber("heli_pose", PoseStamped, self.heli_request)

        # TODO load these params from a config file
        self.H, self.W = 480., 720.
        focal = 300.0
        self.K = np.array([[focal, 0, self.W / 2.],
                      [0, focal, self.H / 2.],
                      [0, 0, 1.]])
        
        # Generate map
        np.random.seed(5)

        self.radius = []
        self.obstacleList = []
        for i in range(250):  # at least 1 obstacle
            # assuming the obstacles are all on a plane, in homogenous coordinates
            self.obstacleList.append([np.random.uniform(-150, 150), np.random.uniform(-150, 150), 0., 1.])
            self.radius.append(np.random.rand() * 3.0 + 2.0)
        start = [0.,0.,0.] # [np.random.uniform(-150, 150), np.random.uniform(-150, 150), np.deg2rad(np.random.uniform(-math.pi, math.pi))]
        goal = [np.random.uniform(-150, 150), np.random.uniform(-150, 150), np.deg2rad(np.random.uniform(-math.pi, math.pi))]

        # Check they are not in the obstacles
        def check_collision(point):
            for (ox, oy, _, _), size in zip(self.obstacleList, self.radius):
                dx = abs(point[0] - ox)
                dy = abs(point[1] - oy)
                d = math.hypot(dx, dy)
                if d <= size+1.0:
                    return check_collision(
                        [np.random.uniform(-150, 150), np.random.uniform(-150, 150), np.deg2rad(np.random.uniform(-math.pi, math.pi))])
            return point

        self.start = check_collision(start)
        self.goal = check_collision(goal)

        # remove all the obstacles that are 3 meters close to the start and goal
        self.obstacleList = [obs for obs in self.obstacleList if np.linalg.norm(np.array(obs[:2]) - np.array(self.start[:2])) > 3.0]
        self.obstacleList = [obs for obs in self.obstacleList if np.linalg.norm(np.array(obs[:2]) - np.array(self.goal[:2])) > 3.0]

        self.bridge= CvBridge()
        # spawn obstacles
        count = 0
        for (ox, oy, _, _), size in zip(self.obstacleList, self.radius):
            pose = Pose()
            pose.position.x = ox
            pose.position.y = oy
            pose.position.z = 0
            self.spawn_obstacle(pose, size, count)
            count += 1

        self.add_goal(self.goal)
        rospy.loginfo('Map generated')


    def heli_request(self, msg):
        # rospy.loginfo('Received heli pose')
        T_cw = np.eye(4)

        pose = msg.pose
        # read pose
        T_cw[0, 3] = pose.position.x
        T_cw[1, 3] = pose.position.y
        T_cw[2, 3] = pose.position.z
        # orientation
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        rot = Rotation.from_quat(np.array([qx, qy, qz, qw]))
        T_cw[:3, :3] = rot.as_matrix()

        T_cw[:3, :3] = np.linalg.inv(rot.as_matrix())
      
        T_cw[:3, 3] = -np.linalg.inv(rot.as_matrix())@ (T_cw[:3, 3])

        self.image = self.project_obstacles(T_cw)
        self.publish_image()


    def project_obstacles(self, T_cw):
        distances = np.linalg.norm((T_cw @ np.array(self.obstacleList).T)[:3, :], axis=0)
        pts_cam = self.K @ (T_cw @ np.array(self.obstacleList).T)[:3, :]
        pts_cam /= pts_cam[2, :]
        pts_cam = pts_cam[:2, :].astype(int)
        # discard all the points out of the camera plane
        pts_cam_plane = (pts_cam[0] > 0) & (pts_cam[1] < self.H) & (pts_cam[1] > 0) & (pts_cam[0] < self.W)
        pts_cam = pts_cam[:, pts_cam_plane].T
        radius_visible = np.array(self.radius)[pts_cam_plane]
        distances = distances[pts_cam_plane]
        # rospy.loginfo(f'Obstacles selected: {np.array(self.obstacleList)[pts_cam_plane]}')

        # create an image color: (138, 83, 47) with the obstacles
        img = np.zeros((int(self.H), int(self.W), 3), np.uint8)
        img[:] = (47, 138, 83)
        for pt, r, d in zip(pts_cam, radius_visible, distances):
            # rospy.loginfo(f"pt:\n{pt}")
            # point size according to the radius
            cv2.circle(img, (int(pt[0]), int(pt[1])), int(self.K[0,0]*r/d), (111, 173, 136), -1)
        
        # add noise to the image
        noise = np.random.normal(0, 10, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        # blur the image
        img = cv2.GaussianBlur(img, (5, 5), 0)

        return img

    
    def publish_image(self):
        if self.image is None:
            return
        img_msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")

        self.map_publisher.publish(img_msg)
        # the goal[2] is the yaw angle
        goal_msg = Point(self.goal[0], self.goal[1], self.goal[2])
        self.goal_publisher.publish(goal_msg)


    # For visualization
    def spawn_obstacle(self, pose, radius, id):
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            
            # Define cylinder SDF with provided pose and radius
            obstacle_sdf = f"""
            <?xml version="1.0"?>
            <sdf version="1.5">
            <model name="cylinder_obstacle_{id}">
                <pose>{pose.position.x} {pose.position.y} {pose.position.z} {pose.orientation.x} {pose.orientation.y} {pose.orientation.z}</pose>
                <link name="link">
                <collision name="collision">
                    <geometry>
                    <cylinder>
                        <radius>{radius}</radius>
                        <length>{2.}</length>
                    </cylinder>
                    </geometry>
                </collision>
                <visual name="visual">
                    <geometry>
                    <cylinder>
                        <radius>{radius}</radius>
                        <length>{2.}</length>
                    </cylinder>
                    </geometry>
                </visual>
                </link>
            </model>
            </sdf>
            """
            # Spawn the obstacle
            resp = spawn_model(f"cylinder_obstacle_{id}", obstacle_sdf, "", pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr("Spawn service call failed: %s", e)

    def add_goal(self, goal):
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            
            # Define a green cube SDF with provided goal pose
            cube_sdf = f"""
            <?xml version="1.0"?>
            <sdf version="1.5">
            <model name="goal">
                <pose>{goal[0]} {goal[1]} 0. 0 0 0</pose>
                <link name="link">
                <collision name="collision">
                    <geometry>
                    <box>
                        <size>1 1 1</size>
                    </box>
                    </geometry>
                </collision>
                <visual name="visual">
                    <geometry>
                    <box>
                        <size>1 1 1</size>
                    </box>
                    <material>
                        <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Green</name>
                        </script>
                    </material>
                    </geometry>
                </visual>
                </link>
            </model>
            </sdf>
            """
            pose = Pose()
            pose.position.x = goal[0]
            pose.position.y = goal[1]
            pose.position.z = 0.5

            # Spawn the obstacle
            resp = spawn_model(f"goal_gaz", cube_sdf, "", pose, "world")
        except rospy.ServiceException as e:
            rospy.logerr("Spawn service call failed: %s", e)




if __name__ == '__main__':
    rospy.init_node('map_generator')

    node = MapGenerator()

    rospy.spin()