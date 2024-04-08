from planning import Planner

class RRT(Planner):
    def __init__(self, config):
        super(RRT, self).__init__()

    def plan(self, start, goal, obstacles) -> list:
        """ Plan a path from start to goal avoiding obstacles.
        Args:
            start (list): list of x, y, z coordinates of the start point
            goal (list): list of x, y, z coordinates of the goal point
            obstacles (list): list of obstacles, each obstacle is a tuple (x, y, z, radius)
        Returns:
            list: This must be a valid list of connected nodes that form
                a path from start to goal node
        """
        pass