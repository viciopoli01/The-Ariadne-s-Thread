#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import Point

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

def check_obs_exists(_map, obs: np.ndarray, threshold=2):
    if any(np.linalg.norm(obs - obs2array(o)) < threshold for o in _map.obstacles_coordinate_list):
        return True
    return False
