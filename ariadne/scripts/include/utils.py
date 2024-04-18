#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf import transformations
from scipy.spatial.transform import Rotation

def map_updater(_map, new_obs, new_radius):
    map_updated = False
    for obs, radius in zip(new_obs, new_radius):
        if isinstance(obs, Point):
            obs = obs2array(obs)
        if not check_obs_exists(_map, obs):
            point = Point()
            point.x = obs[0]
            point.y = obs[1]
            point.z = obs[2]

            _map.obstacles_coordinate_list.append(point)
            _map.obstacles_radius_list.append(radius)
            map_updated = True
    
    return _map, map_updated

def obs2array(_obs):
    return np.array([_obs.x, _obs.y, _obs.z])

def check_obs_exists(_map, obs: np.ndarray, threshold=0.5):
    if any(np.linalg.norm(obs - obs2array(o)) < threshold for o in _map.obstacles_coordinate_list):
        return True
    return False

def path2msg(path):
    path_msg = Path()
    for p in path:
        pose = PoseStamped()
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = 0.

        # yaw to quaternion
        # ensure the yaw angle is within [-pi, pi]
        p[2] = np.arctan2(np.sin(p[2]), np.cos(p[2]))
        q = transformations.quaternion_from_euler(0, 0, p[2])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        path_msg.poses.append(pose)
    return path_msg

def msg2path(msg):
    path = []
    for pose in msg.poses:
        p = [pose.pose.position.x, pose.pose.position.y, transformations.euler_from_quaternion(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])[2]]
        # limit the yaw angle to [-pi, pi]
        p[2] = np.arctan2(np.sin(p[2]), np.cos(p[2]))

        path.append(p)
    return path

def msg2pose(msg):
    pose = msg.pose.pose

    # rotate the pose frame by 180 degrees along the z-axis

    T = np.eye(4)
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

    # rotate the pose frame by 180 degrees along the z-axis
    T_rot = np.array([[-1., 0., 0, 0],
                      [0., -1., 0, 0],
                      [0, 0, 1., 0],
                      [0, 0, 0, 1.]])
    T = T @ T_rot
    # T = np.linalg.inv(T)

    # extract the position and yaw angle
    p = T[:3, 3]
    yaw = np.arctan2(T[1, 0], T[0, 0])

    # limit the yaw angle to [-pi, pi]
    yaw+=np.pi/2.
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    # print('yaw', np.rad2deg(yaw))
    return np.array([p[0], p[1], yaw])

def msg2T(msg):
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

    return T