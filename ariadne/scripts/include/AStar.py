#!/usr/bin/env python3

from planner import Planner
import numpy as np
import math
import matplotlib.pyplot as plt
show_animation = True


class AStar(Planner):
    def __init__(self, config=None):
        super(AStar, self).__init__(config)
        self.resolution = 4.0
        self.rr = 1.0
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()

        self.x_center=[]
        self.y_center=[]
        self.radius=[]

        

    
    
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)
        
    def plan(self, start, goal, obstacles: list, radius: list,show_animation=True,mapbound=[-50,-30,50,30] ) -> list:
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
        
        obstacles=np.array(obstacles)
        print("obs:",obstacles)
        print("rad:",radius)
        self.x_center=obstacles[:,0]
        self.y_center=obstacles[:,1]
        self.raius=radius
        obs_points=[]
        round_obs=np.array([self.x_center,self.y_center,self.raius]).T
        for x_center, y_center, radius in round_obs:
            # points = circle_to_points(x_center, y_center, radius, num_points=16)
            points = self.circle_to_grid_cells(x_center, y_center, radius)
            obs_points.extend(points)
        ox,oy=[],[]
        self.min_x, self.min_y, self.max_x, self.max_y = mapbound
        for i in range(self.min_x,self.max_x):
            ox.append(i)
            oy.append(self.min_y)
        for i in range(self.min_y,self.max_y):
            ox.append(self.max_x)
            oy.append(i)
        for i in range(self.min_x,self.max_x):
            ox.append(i)
            oy.append(self.max_y)
        for i in range(self.min_y,self.max_y):
            ox.append(self.min_x)
            oy.append(i)

        for i in obs_points:
            ox.append(i[0])
            oy.append(i[1])
        self.calc_obstacle_map(ox, oy)
        print("number of obs!!!!!!!!!!!:",obstacles.shape)

        

        # sx,sy=start

        if isinstance(start,np.ndarray):

            sx,sy,_theta=start
        else:
            sx=start.x
            sy=start.y
        # print("goal:",goal)
        # gx=goal.x
        # gy=goal.y
        if isinstance(goal,np.ndarray):

            gx,gy,_theta=goal  
        else:
            gx=goal.x
            gy=goal.y


        if show_animation:  # pragma: no cover
            plt.clf()
            # plt.gca().invert_yaxis()
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        
        reversed_path=np.array([rx,ry,np.zeros(len(rx))]).T
        # print(reversed_path)
        path=reversed_path[::-1]
        extended_path= self.add_orientation_to_path(path)
        if show_animation:  # pragma: no cover
            print("showwww")
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            # plt.clf()
            # plt.pause(1)
            # print(rx)
            # print(ry)
            # plt.show()
            self.plot_path_with_orientations(extended_path)
        return extended_path
    
    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry
    def plot_path_with_orientations(self, path_with_orientation):
        fig, ax = plt.subplots()
        for i, (x, y, _, zx, zy, zz, zw) in enumerate(path_with_orientation):
            angle = 2 * np.arctan2(zz, zw)  # Convert quaternion back to angle
            # print(angle)
            ax.quiver(x, y, np.cos(angle), np.sin(angle), color='red' if i == 0 else 'green', scale=20)
            ax.scatter(x, y, color='blue')  # Point

        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
    def add_orientation_to_path(self, path):
    # Function to calculate quaternion from an angle theta
        def angle_to_quaternion(theta):
            return np.array([0, 0, np.sin(theta / 2), np.cos(theta / 2)])
        
        # Initialize the path with orientation
        path_with_orientation = []
        
        # Iterate over the path to calculate the orientation at each step
        for i in range(len(path) - 1):
            x1, y1, _ = path[i]
            x2, y2, _ = path[i + 1]
            
            # Calculate the angle theta between successive points
            theta = np.arctan2(y2 - y1, x2 - x1)
            print(theta)
            
            # Compute quaternion
            quaternion = angle_to_quaternion(theta)
            
            # Append position and orientation to the new path
            path_with_orientation.append((x1, y1, 0) + tuple(quaternion))
            
        
        # Add the last point with the same orientation as the second last point
        if path.any():
            
            path_with_orientation.append(tuple(path[-1]) + tuple(path_with_orientation[-1][3:]))
        
        return path_with_orientation
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

        
        
    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion
    
    def circle_to_grid_cells(self, x_center, y_center, radius, resolution=1):
        """
        Calculate grid cells occupied by a circular obstacle.

        Parameters:
        - x_center, y_center: coordinates of the circle's center in meters.
        - radius: radius of the circle in meters.
        - resolution: size of each grid cell in meters.

        Returns:
        - A list of (x, y) tuples representing occupied grid cells.
        """
        occupied_cells = []

        # Calculate the bounding box of the circle
        x_min = int((x_center - radius) // resolution)
        x_max = int((x_center + radius) // resolution)
        y_min = int((y_center - radius) // resolution)
        y_max = int((y_center + radius) // resolution)

        # Iterate over each cell in the bounding box
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Calculate the center of the cell
                cell_center_x = (x + 0.5) * resolution
                cell_center_y = (y + 0.5) * resolution

                # Check if the center of the cell is inside the circle
                distance = ((cell_center_x - x_center) ** 2 + (cell_center_y - y_center) ** 2) ** 0.5
                if distance <= radius:
                    occupied_cells.append((x, y))

        return occupied_cells                                                           



def main():
    obs=np.array([[5,5,0],
                  [20,10,0],
                  [30,30,0],
                  [60,10,0]])
    radi=[2,3,2,10]
    mapbound=[-10,-20,100,60] # min_x min_y max_x max_y
    planner=AStar()
    path=planner.plan(np.array([10, 10,0]), np.array([50, 50,0]), obs, radi,show_animation,mapbound)
    rx=path[:,0]
    ry=path[:,1]

    # if show_animation:  # pragma: no cover
    #     plt.plot(ox, oy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")

   

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(1)
        print(rx)
        print(ry)
        
        plt.show()

if __name__ == '__main__':
    main()