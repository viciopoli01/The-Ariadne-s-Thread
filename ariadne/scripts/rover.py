#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rospy

from include.utils import map_updater, obs2array
from include.planner import Planner

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig



class Rover():

    def __init__(self):
        # publish path messages
        self.path_publisher = rospy.Publisher('rover_path', Path, queue_size=10)

        # publish map messages if new obstacles are detected
        self.map_publisher = rospy.Publisher('map_rover', AriadneMap, queue_size=10)

        # subscribe to the map topic
        rospy.Subscriber("map", AriadneMap, self.map_callback)

        rospy.Subscriber("heli_pose", PoseStamped, self.helipose_update)
        self.map = AriadneMap()
        self.map.obstacles = []
        self.map.radius = []

        self.planner = Planner()
        self.dyn_srv = Server(AriadneConfig, self.dynamic_callback)
        # self.planner = Planner()
        self.pose = np.array([0,0])
        self.prev_pose=np.array([0,0])

        
    def helipose_update(self,msg):
        heli_pose = msg.pose
        # read pose
        # self.prev_pose=self.pose 
        self.pose = np.array([heli_pose.position.x,heli_pose.position.y ])
        
    def dynamic_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {planner_algo}""".format(**config))
        # print(config)
        if config['planner_algo'] == -1:
            rospy.loginfo('Using MyCustomPlanner')
            self.planner = Planner()
        elif config['planner_algo'] == 0:
            from include.RRT import RRT
            from include.AStar import AStar
            rospy.loginfo('Using RRT')
            rospy.loginfo('Using force Astar') #issue unable to call a star using terminal
            self.planner = AStar()
        elif config['planner_algo'] == 1:
            from include.RRTStar import RRTStar
            rospy.loginfo('Using RRT*')
            self.planner = RRTStar()
        elif config['planner_algo'] == 2:
            from include.AStar import AStar
            rospy.loginfo('Using A*')
            self.planner = AStar()
        elif config['planner_algo'] == 3:
            from include.DStar import DStar
            rospy.loginfo('Using D*')
            self.planner = DStar()

        return config


    def map_callback(self, msg):
        rospy.loginfo('Received map')
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        # update the map
        self.map,_tf= map_updater(self.map, msg.obstacles, msg.radius)
        if  _tf: #np.linalg.norm(self.prev_pose-self.pose)<1 or np.linalg.norm(self.prev_pose-self.pose)==0: # issue with producing duplicated obs if update too fast
            self.goal = msg.goal
            current=self.pose[0:2].astype(int)
            final=[msg.goal.x,msg.goal.y]
            frame_width=75
            frame_height=35
            

            obstacles=np.array([obs2array(o) for o in self.map.obstacles])
            x_center=obstacles[:,0]
            y_center=obstacles[:,1]     
            round_obs=np.array([x_center,y_center,self.map.radius]).T
            self.temp_goal=self.find_temporary_goal(current, final, frame_width, frame_height, round_obs)
            print("temp goal:", self.temp_goal)
            

            path = self.planner.plan(self.pose, self.temp_goal, [obs2array(o) for o in self.obstacles], self.map.radius,show_animation=True,mapbound=[current[0]-55,current[1]-25,current[0]+55,current[1]+25] )
            

            self.prev_pose=path[-1,:2]
            if path.any():
                self.publish_path(path)


    def in_bounds(self,x, y, x_min, x_max, y_min, y_max):
        return x_min <= x <= x_max and y_min <= y <= y_max
    def distance_to_line(self,px, py, ax, ay, bx, by):
        # Line AB represented as a1x + b1y = c1
        a1 = by - ay
        b1 = ax - bx
        c1 = a1 * ax + b1 * ay
        # Perpendicular distance from point P to the line AB
        return abs(a1 * px + b1 * py - c1) / np.sqrt(a1**2 + b1**2)
    def find_temporary_goal(self,current, final, frame_width, frame_height, obstacles):
        cx, cy = current
        fx, fy = final
        obstacles = np.array(obstacles)  # [(ox, oy, radius), ...]
        
        # Define the view frame relative to the current position
        x_min = cx - frame_width / 2
        x_max = cx + frame_width / 2
        y_min = cy - frame_height / 2
        y_max = cy + frame_height / 2

        # Compute the direction vector towards the final goal
        direction = np.array([fx - cx, fy - cy])
        norm = np.linalg.norm(direction)
        if norm <= 2:
            return None  # Already at the goal
        direction_unit = direction / norm

        # Check for intersections with frame bounds and select the furthest point within frame
        max_distance = min(frame_width, frame_height) / 2  # conservative maximum view

        if norm <= max_distance:
            max_distance=norm-2
        temp_goal = np.array(current) + direction_unit * max_distance

        # Adjust temporary goal to stay within bounds
        if not self.in_bounds(temp_goal[0], temp_goal[1], x_min, x_max, y_min, y_max):
            # Scale back to bounds
            if temp_goal[0] < x_min or temp_goal[0] > x_max:
                scale = min(abs((x_max - cx) / direction_unit[0]), abs((x_min - cx) / direction_unit[0]))
                temp_goal = np.array(current) + direction_unit * scale
            if temp_goal[1] < y_min or temp_goal[1] > y_max:
                scale = min(abs((y_max - cy) / direction_unit[1]), abs((y_min - cy) / direction_unit[1]))
                temp_goal = np.array(current) + direction_unit * scale

        # Avoid obstacles
        for ox, oy, radius in obstacles:
            if np.linalg.norm(temp_goal - np.array([ox, oy])) <= radius+2:
                # Move back slightly from the obstacle
                away_vector = temp_goal - np.array([ox, oy])
                temp_goal = np.array([ox, oy]) + (away_vector / np.linalg.norm(away_vector) * (radius + 2))
        
        

        return temp_goal                         

    def publish_path(self, path):
        """Publish the path to the rover_path topic.

        Args:
            path (list): list of waypoints in the form [x, y, z, qx, qy, qz, qw]
            """
        path_msg = Path()
        for p in path:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            # orientation
            pose.pose.orientation.x = p[3]
            pose.pose.orientation.y = p[4]
            pose.pose.orientation.z = p[5]
            pose.pose.orientation.w = p[6]
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)

    def add_new_obstacle(self, obs, radius):
        """ Add new obstacle to the map if it does not exist and send it over the map_rover topic.

        Args:
            obs (list): list of x, y, z coordinates of the obstacle
            radius (float): radius of the obstacle
        """
        # update the existing map locally
        self.obstacles.append(obs)
        self.radius.append(radius)

        # update the existing map globally
        map_msg = AriadneMap()
        map_msg.header.frame_id = 'map'
        map_msg.goal = []

        map_msg.obstacles = [obs]
        map_msg.radius = [radius]

        self.map_publisher.publish(map_msg)


if __name__ == '__main__':
    rospy.init_node('rover')

    node = Rover()

    rospy.spin()