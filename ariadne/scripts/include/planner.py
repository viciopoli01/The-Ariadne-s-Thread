#!/usr/bin/env python3

class Planner:

    def __init__(self, config=None):
        self.config = config

    def plan(self, start, goal, obstacles: list, radius: list) -> list:
        """ Plan a path from start to goal avoiding obstacles.

        Args:
            start (list): list of x, y coordinates of the start point
            goal (list): list of x, y coordinates of the goal point
            obstacles (list): list of obstacles, each obstacle is a tuple (x, y, 0)
            radius (list): list of radius of obstacles
        
        Returns:
            list: This must be a valid list of connected nodes that form
                a path from start to goal node
        """
        try:
            from include.planner_notebook import planner
            return planner(start, goal, obstacles, radius)
        except ImportError:
            raise NotImplementedError