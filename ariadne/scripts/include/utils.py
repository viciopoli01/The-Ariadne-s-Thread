#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import Point

def map_updater(_map, new_obs, new_radius):
    map_updated = False
    for obs, radius in zip(new_obs, new_radius):
        if not check_obs_exists(_map, obs):
            point = Point()
            point.x = obs[0]
            point.y = obs[1]
            point.z = obs[2]

            _map.obstacles.append(point)
            _map.radius.append(radius)
            map_updated = True
    
    return _map, map_updated


def check_obs_exists(_map, obs: np.ndarray, threshold=0.1):
    def obs2array(_obs):
        return np.array([_obs.x, _obs.y, _obs.z])
    if any(np.linalg.norm(obs - obs2array(o)) < threshold for o in _map.obstacles):
        return True
    return False
