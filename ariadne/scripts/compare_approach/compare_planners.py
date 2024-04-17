#!/usr/bin/env python3

import numpy as np
import math
from typing import Tuple
import matplotlib.pyplot as plt
from ariadne.scripts.include.RRT import RRT
from ariadne.scripts.include.RRTStar import RRTStar
from ariadne.scripts.include.AStar import AStar
import matplotlib.patches as patches

np.random.seed(52)
map_width = 1000
map_height = 1000
num_obstacles = 100
rover_radius = 2
max_obstacle_radius = 5
dupin_curvature = 1 / 3.5
helicopter_servey_width = 250
helicopter_servey_height = 150


def generate_map(map_width: float,
                 map_height: float,
                 num_obstacles: int,
                 rover_radius: float,
                 max_obstacle_radius: float,
                 dupin_curvature: float) -> Tuple[np.ndarray, np.ndarray]:
    start_pose = np.array([0, 0, 0])
    goal_pose = np.array([np.random.uniform(-map_width / 2, map_width / 2), np.random.uniform(-map_height / 2, map_height / 2),
                          np.deg2rad(np.random.uniform(-math.pi, math.pi))])
    min_distance_to_rover = 2 / dupin_curvature + rover_radius
    obstacles = np.zeros((num_obstacles, 3))
    for i in range(num_obstacles):
        obstacles[i] = np.array(
            [np.random.uniform(-map_width / 2, map_width / 2), np.random.uniform(-map_height / 2, map_height / 2), np.random.uniform(0.0, max_obstacle_radius)])
        distance_to_start = np.sum((obstacles[i, :2] - start_pose[:2]) ** 2) ** 0.5
        distance_to_goal = np.sum((obstacles[i, :2] - goal_pose[:2]) ** 2) ** 0.5
        attempts = 0
        while (distance_to_start <= (obstacles[i, 2] + min_distance_to_rover)) or (distance_to_goal <= (obstacles[i, 2] + min_distance_to_rover)):
            if attempts > 10:
                obstacles[i] = np.array([map_width / 2, map_height / 2, 0])
                break
            attempts += 1
            obstacles[i] = np.array([np.random.uniform(-map_width / 2, map_width / 2), np.random.uniform(-map_height / 2, map_height / 2),
                                     np.deg2rad(np.random.uniform(0.0, max_obstacle_radius))])

    return goal_pose, obstacles


def find_temporary_goal(current, final, frame_width, frame_height, obstacles):
    # Define the view frame relative to the current position
    x_min = current[0] - frame_width / 2
    x_max = current[0] + frame_width / 2
    y_min = current[1] - frame_height / 2
    y_max = current[1] + frame_height / 2

    # Compute the direction vector towards the final goal
    direction = np.array([final[0] - current[0], final[1] - current[1]])
    norm = np.linalg.norm(direction)
    if norm <= 2:
        return final  # Already at the goal
    direction_unit = direction / norm

    # Check for intersections with frame bounds and select the furthest point within frame
    max_distance = min(frame_width, frame_height) / 2  # conservative maximum view

    if norm <= max_distance:
        max_distance = norm - 2
    temp_goal = np.array(current) + direction_unit * max_distance

    # Adjust temporary goal to stay within bounds
    if not (x_min <= temp_goal[[0]] <= x_max) and (y_min <= temp_goal[1] <= y_max):
        # Scale back to bounds
        if temp_goal[0] < x_min or temp_goal[0] > x_max:
            scale = min(abs((x_max - current[0]) / direction_unit[0]), abs((x_min - current[0]) / direction_unit[0]))
            temp_goal = np.array(current) + direction_unit * scale
        if temp_goal[1] < y_min or temp_goal[1] > y_max:
            scale = min(abs((y_max - current[1]) / direction_unit[1]), abs((y_min - current[1]) / direction_unit[1]))
            temp_goal = np.array(current) + direction_unit * scale

    # Avoid obstacles
    for ox, oy, radius in obstacles:
        if np.linalg.norm(temp_goal - np.array([ox, oy])) <= radius + 2:
            # Move back slightly from the obstacle
            away_vector = temp_goal - np.array([ox, oy])
            temp_goal = np.array([ox, oy]) + (away_vector / np.linalg.norm(away_vector) * (radius + 2))

    return np.append(temp_goal, final[2])


def plan_with_random_map_global() -> Tuple[bool, bool, bool, float, float, float]:
    rrt_planner = RRT()
    rrt_start_planner = RRTStar()
    astar_planner = AStar()

    global_goal_pose, obstacles = generate_map(map_width, map_height, num_obstacles, rover_radius, max_obstacle_radius, dupin_curvature)
    current_pose = np.array([0, 0, 0])
    local_goal_pose = find_temporary_goal(current_pose[:2], global_goal_pose, helicopter_servey_width, helicopter_servey_height, obstacles)
    rrt_cost = 0
    rrt_star_cost = 0
    astar_cost = 0
    local_map_number = 0
    rrt_failed = False
    rrt_star_failed = False
    astar_failed = False
    # plt.plot(current_pose[0], current_pose[1], "og", label='global start')
    # plt.plot(global_goal_pose[0], global_goal_pose[1], "xr", label='global goal')
    # plt.grid(True)
    # plt.xlim([-map_width/2, map_width/2])
    # plt.ylim([-map_height/2, map_height/2])
    # plt.axis("equal")
    # plt.legend()
    # plt.pause(0.001)
    # RRT
    rrt_course = rrt_planner.plan(current_pose, local_goal_pose, list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=False,
                                  map_bounds=[-helicopter_servey_width / 2, -helicopter_servey_height / 2, helicopter_servey_width / 2, helicopter_servey_height / 2],
                                  search_until_max_iter=False)
    rrt_cost += rrt_planner.cost
    print('rrt_cost', rrt_cost)
    if rrt_planner.cost <= 0:
        rrt_failed = True
        print(f'rrt failed to find path')
    # RRT Star
    rrt_star_course = rrt_start_planner.plan(current_pose, local_goal_pose, list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=True,
                                             map_bounds=[-helicopter_servey_width / 2, -helicopter_servey_height / 2, helicopter_servey_width / 2,
                                                         helicopter_servey_height / 2],
                                             search_until_max_iter=False)
    rrt_star_cost += rrt_start_planner.cost
    print('rrt_star_cost', rrt_star_cost)
    if rrt_start_planner.cost <= 0:
        rrt_star_failed = True
        print(f'rrt star failed to find path')
    # A Star
    astar_course = astar_planner.plan(current_pose[:2], local_goal_pose[:2], list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=False,
                                      map_bounds=[int(-helicopter_servey_width / 2), int(-helicopter_servey_height / 2), int(helicopter_servey_width / 2),
                                                  int(helicopter_servey_height / 2)])
    astar_cost += astar_planner.cost
    print('astar_cost', astar_cost)
    if astar_planner.cost <= 0:
        astar_failed = True
        print(f'a star failed to find path')

    print(f'finished planning global map')
    print(f'did rrt fail? {rrt_failed}')
    print(f'did rrt star fail? {rrt_star_failed}')
    print(f'did a star fail? {astar_failed}')
    print(f'rrt cost: {rrt_cost}')
    print(f'rrt star cost: {rrt_star_cost}')
    print(f'a star cost: {astar_cost}')

    return rrt_failed, rrt_star_failed, astar_failed, rrt_cost, rrt_star_cost, astar_cost


def plan_with_random_map_with_heli() -> Tuple[bool, bool, bool, float, float, float]:
    rrt_planner = RRT()
    rrt_start_planner = RRTStar()
    astar_planner = AStar()

    global_goal_pose, obstacles = generate_map(map_width, map_height, num_obstacles, rover_radius, max_obstacle_radius, dupin_curvature)
    current_pose = np.array([0, 0, 0])
    local_goal_pose = find_temporary_goal(current_pose[:2], global_goal_pose, helicopter_servey_width, helicopter_servey_height, obstacles)
    rrt_cost = 0
    rrt_star_cost = 0
    astar_cost = 0
    local_map_number = 0
    rrt_failed = False
    rrt_star_failed = False
    astar_failed = False
    # plt.plot(current_pose[0], current_pose[1], "og", label='global start')
    # plt.plot(global_goal_pose[0], global_goal_pose[1], "xr", label='global goal')
    # plt.grid(True)
    # plt.xlim([-map_width/2, map_width/2])
    # plt.ylim([-map_height/2, map_height/2])
    # plt.axis("equal")
    # plt.legend()
    # plt.pause(0.001)

    while np.any(local_goal_pose != global_goal_pose):
        if not rrt_failed:
            rrt_course = rrt_planner.plan(current_pose, local_goal_pose, list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=False,
                                          map_bounds=[-helicopter_servey_width / 2, -helicopter_servey_height / 2, helicopter_servey_width / 2, helicopter_servey_height / 2],
                                          search_until_max_iter=False)
            rrt_cost += rrt_planner.cost
            print('rrt_cost', rrt_cost)
            if rrt_planner.cost <= 0:
                rrt_failed = True
                print(f'rrt failed to find path')
        if not rrt_star_failed:
            rrt_star_course = rrt_start_planner.plan(current_pose, local_goal_pose, list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=True,
                                                     map_bounds=[-helicopter_servey_width / 2, -helicopter_servey_height / 2, helicopter_servey_width / 2,
                                                                 helicopter_servey_height / 2],
                                                     search_until_max_iter=False)
            rrt_star_cost += rrt_start_planner.cost
            print('rrt_star_cost', rrt_star_cost)
            if rrt_start_planner.cost <= 0:
                rrt_star_failed = True
                print(f'rrt star failed to find path')
        if not astar_failed:
            astar_course = astar_planner.plan(current_pose[:2], local_goal_pose[:2], list(obstacles[:, :2]), list(obstacles[:, 2]), show_animation=False,
                                              map_bounds=[int(-helicopter_servey_width / 2), int(-helicopter_servey_height / 2), int(helicopter_servey_width / 2),
                                                          int(helicopter_servey_height / 2)])
            astar_cost += astar_planner.cost
            print('astar_cost', astar_cost)
            if astar_planner.cost <= 0:
                astar_failed = True
                print(f'a star failed to find path')

        current_pose = np.array([0, 0, 0])
        global_goal_pose[:2] = global_goal_pose[:2] - local_goal_pose[:2]
        for i in range(len(obstacles)):
            obstacles[i, :2] = obstacles[i, :2] - local_goal_pose[:2]
        local_goal_pose = find_temporary_goal(current_pose[:2], global_goal_pose, helicopter_servey_width, helicopter_servey_height, obstacles)
        print(f'finished planning local map {local_map_number}')
        # plt.grid(True)
        # plt.xlim([-map_width / 2, map_width / 2])
        # plt.ylim([-map_height / 2, map_height / 2])
        # plt.axis("equal")
        # plt.legend()
        # plt.pause(0.001)
        local_map_number += 1

    print(f'finished planning global map with {local_map_number} local maps')
    print(f'did rrt fail? {rrt_failed}')
    print(f'did rrt star fail? {rrt_star_failed}')
    print(f'did a star fail? {astar_failed}')
    print(f'rrt cost: {rrt_cost}')
    print(f'rrt star cost: {rrt_star_cost}')
    print(f'a star cost: {astar_cost}')
    return rrt_failed, rrt_star_failed, astar_failed, rrt_cost, rrt_star_cost, astar_cost

def main():
    # Run for 100 randomly generated maps
    n_random_maps = 2
    local_rrt_failed = 0
    local_rrt_star_failed = 0
    local_astar_failed = 0
    global_rrt_failed = 0
    global_rrt_star_failed = 0
    global_astar_failed = 0
    local_rrt_cost = [-1] * n_random_maps
    local_rrt_star_cost = [-1] * n_random_maps
    local_astar_cost = [-1] * n_random_maps
    global_rrt_cost = [-1] * n_random_maps
    global_rrt_star_cost = [-1] * n_random_maps
    global_astar_cost = [-1] * n_random_maps

    for i in range(n_random_maps):
        rrt_failed, rrt_star_failed, astar_failed, rrt_cost, rrt_star_cost, astar_cost = plan_with_random_map_global()
        local_rrt_failed += rrt_failed
        local_rrt_star_failed += rrt_star_failed
        local_astar_failed += astar_failed
        local_rrt_cost[i] = rrt_cost
        local_rrt_star_cost[i] = rrt_star_cost
        local_astar_cost[i] = astar_cost
        rrt_failed, rrt_star_failed, astar_failed, rrt_cost, rrt_star_cost, astar_cost = plan_with_random_map_with_heli()
        global_rrt_failed += rrt_failed
        global_rrt_star_failed += rrt_star_failed
        global_astar_failed += astar_failed
        global_rrt_cost[i] = rrt_cost
        global_rrt_star_cost[i] = rrt_star_cost
        global_astar_cost[i] = astar_cost

    plt.hist(local_rrt_cost)
    plt.hist(local_rrt_star_cost)
    plt.hist(local_astar_cost)
    plt.hist(global_rrt_cost)
    plt.hist(global_rrt_star_cost)
    plt.hist(global_astar_cost)
    print(f'local rrt failed {local_rrt_failed}')
    print(f'local rrt star failed {local_rrt_star_failed}')
    print(f'local a star failed {local_astar_failed}')
    print(f'global rrt failed {global_rrt_failed}')
    print(f'global rrt star failed {global_rrt_star_failed}')
    print(f'global a star failed {global_astar_failed}')


if __name__ == '__main__':
    main()
