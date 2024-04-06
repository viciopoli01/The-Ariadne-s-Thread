import json
import os

import numpy as np

from ariadne_msgs.msg import AriadneMap
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
import cv2


class Heli(Node):

    def __init__(self, root_dir):
        super().__init__('heli')
        self.map_publisher = self.create_publisher(AriadneMap, 'map', 10)
        self.map = AriadneMap()
        self.map.header.frame_id = 'map'
        self.map.obstacles = []
        self.map.radius = []
        self.map.goal = Point()

        # subscribe to image topic
        # self.create_subscription(Image, 'image', self.image_callback, 10)

        # for now we can read an image from a file
        # read all the images in root_dir
        self.load_blender_data(root_dir)

    def load_blender_data(self, root_dir):
        with open(os.path.join(root_dir, 'transforms_train.json'), 'r') as fp:
            data = json.load(fp)

        H, W = 800., 800.
        camera_angle_x = float(data['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = np.array([[focal, 0, H / 2.],
                      [0, focal, W / 2.],
                      [0, 0, 1.]])

        # read the images and track the feature points
        for i, frame in enumerate(data['frames']):
            fname = os.path.join(root_dir, frame['file_path'] + '.png')
            img = cv2.imread(fname)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            obs = self.extract_obstacles(img)
            if not obs:
                continue
            obs_cam_plane = obs[:, :2]
            rays = obs[:, 2]
            obs_world = self.project_obstacles(obs_cam_plane, K, np.array(frame['transform_matrix']))

            # check if 3D point already in map
            for obs in obs_world:
                # if the distance to the obstacle is less than a threshold, we consider it the same obstacle
                if any(np.linalg.norm(np.array(obs) - np.array(o)) < 0.1 for o in self.map.obstacles):
                    continue
                self.map.obstacles.append(obs)
                self.map.radius.append(rays)

            # publish map
            self.publish_map()

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


def main(args=None):
    rclpy.init(args=args)

    heli_node = Heli(root_dir='/home/viciopoli/datasets/MarsNeRF/sol449')

    rclpy.spin(heli_node)

    heli_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
