#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rospy

from include.utils import map_updater, obs2array
from include.planner import Planner
from include.dubins import dubins_path_planning
from include.controller import Controller

import numpy as np

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import tf.transformations


class Rover():

    def __init__(self):
        # publish path messages
        self.path_publisher = rospy.Publisher('rover_path', Path, queue_size=10)

        # publish map messages if new obstacles are detected
        self.map_publisher = rospy.Publisher('map_rover', AriadneMap, queue_size=10)

        # subscribe to the map topic
        rospy.Subscriber("map", AriadneMap, self.map_callback)
        self.map = AriadneMap()
        self.map.obstacles = []
        self.map.radius = []

        self.dyn_srv = Server(AriadneConfig, self.dynamic_callback)
        self.planner = Planner()

        self.pose = np.array([0., 0., 0.]) # x, y, theta
        self.path = []
        
        self.init = False
        # PD controller
        self.kp = 9.5
        self.kd = 4.5
        self.ki = 2.1
        
        self.controller = Controller(self.kp, self.kd, self.ki, starting_vel=3.5)
        self.next_point = 0

        rospy.wait_for_service('/curiosity_mars_rover/mast_service')

        try:
            mast_service_proxy = rospy.ServiceProxy('/curiosity_mars_rover/mast_service', DeleteModel)
            response = mast_service_proxy(model_name='open')
            
            rospy.loginfo("Response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        # subscribe to the odometry topic
        rospy.Subscriber('/curiosity_mars_rover/odom', Odometry, self.odom_callback, queue_size=1)
        # rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        self.theta= []

    def dynamic_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {planner_algo}""".format(**config))

        if config['planner_algo'] == -1:
            rospy.loginfo('Using MyCustomPlanner')
            self.planner = Planner()
        elif config['planner_algo'] == 0:
            from include.RRT import RRT
            rospy.loginfo('Using RRT')
            self.planner = RRT()
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
        if self.controller.moving or not self.init:
            return
        rospy.loginfo('Received map')
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        # update the map
        self.map, map_updater(self.map, msg.obstacles, msg.radius)
        self.goal = msg.goal
        # start from the current position 0 0 0
        path = [[self.pose[0],self.pose[1],self.pose[2]],[6, 3,np.pi/36.], [10, 5, np.pi/36.], [2, 24, np.pi/36.]]
        path = np.array(path).astype(np.float64)
        pxs, pys, pyaws= [], [], []
        for i in range(len(path)-1):
            rospy.loginfo('Planning path from {} to {}'.format(path[i], path[i+1]))
            px, py, pyaw, mode, clen = dubins_path_planning(path[i][0], path[i][1], path[i][2], path[i+1][0], path[i+1][1], path[i+1][2], 3.0)
            pxs.append(px)
            pys.append(py)
            pyaws.append(pyaw)
            break
        px = np.concatenate(pxs)
        py = np.concatenate(pys)
        pyaw = np.concatenate(pyaws)
        self.path = np.vstack((px, py, pyaw)).T
        # self.path = self.planner.plan(self.pose, self.goal, [obs2array(o) for o in self.map.obstacles], self.map.radius)
        # if self.path:
        #     self.publish_path(self.path)


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

    def odom_callback(self, msg):
        # get the pose of the rover
        # get the yaw angle of the rover
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        quaternion = (qx, qy, qz, qw)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        
        # yaw is the third element in the Euler angle tuple
        roll, pitch, yaw = euler

        # limit the yaw angle to [-pi, pi]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        pitch = np.arctan2(np.sin(pitch), np.cos(pitch))
        roll = np.arctan2(np.sin(roll), np.cos(roll))

        self.theta.append([roll, pitch, yaw])
        
        # Print or use the yaw value
        # print("Yaw: {:.4f} radians".format(yaw))
        # # If you prefer degrees, you can convert it as follows:
        yaw_deg = yaw * (180.0 / 3.1415926)
        print("Yaw: {:.4f} degrees".format(yaw_deg))

        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        
        self.init = True
        # return 
        # check if the rover is moving
        if len(self.path) == 0:
            return

        # check if pose is close enough to the goal
        goal_pose = self.path[self.next_point]
        if np.linalg.norm(goal_pose[:2] - self.pose[:2]) < .5:
            rospy.loginfo('Reached node, moving to next node')
            self.next_point += 1
            if self.next_point >= len(self.path):
                rospy.loginfo('Goal reached')
                self.controller.stop()
                return
            goal_pose = self.path[self.next_point]
        
        self.controller(self.pose, goal_pose, rospy.get_time())
        rospy.loginfo("'curiosity_mars_rover' position: {}".format(yaw))

    def model_states_callback(self, msg):
        model_names = msg.name
        model_positions = msg.pose
        

        # Find the index of 'curiosity_mars_rover' in the list of model names
        try:
            curiosity_index = model_names.index('curiosity_mars_rover')
            # Get the position of 'curiosity_mars_rover'
            curiosity_position = model_positions[curiosity_index].position

            # get the yaw angle of the rover
            curiosity_orientation = model_positions[curiosity_index].orientation
            qx = curiosity_orientation.x
            qy = curiosity_orientation.y
            qz = curiosity_orientation.z
            qw = curiosity_orientation.w
            
            # quat to rotation matrix
            rot = Rotation.from_quat(np.array([qx, qy, qz, qw]))
            yaw_rad = rot.as_euler('zyx')[0]
            rospy.loginfo('Yaw: {}'.format(yaw_rad))
            rospy.loginfo('Pitch: {}'.format(rot.as_euler('zyx')[1]))
            rospy.loginfo('Roll: {}'.format(rot.as_euler('zyx')[2]))

            return
            self.pose = np.array([curiosity_position.x, curiosity_position.y, yaw_rad])
            
            # check if the rover is moving
            if len(self.path) == 0:
                return

            # check if pose is close enough to the goal
            goal_pose = self.path[self.next_point]
            if np.linalg.norm(goal_pose[:2] - self.pose[:2]) < .5:
                rospy.loginfo('Reached node, moving to next node')
                self.next_point += 1
                if self.next_point >= len(self.path):
                    rospy.loginfo('Goal reached')
                    self.controller.stop()
                    return
                goal_pose = self.path[self.next_point]
            
            self.controller(self.pose, goal_pose, rospy.get_time())
            
            rospy.loginfo("'curiosity_mars_rover' position: {}, {}".format(curiosity_position, yaw_rad))
        except ValueError:
            rospy.logwarn("'curiosity_mars_rover' not found in model_states")

    

    # on node shutdown
    def shutdown(self):
        rospy.loginfo('Shutting down')
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.
        cmd_vel.angular.z = 0.
        self.cmd_vel_publisher.publish(cmd_vel)
        rospy.sleep(1.)

        # plot theta values
        
        plt.plot(self.theta)
        plt.show()

        rospy.signal_shutdown('Shutting down')

if __name__ == '__main__':
    rospy.init_node('rover')

    node = Rover()

    rospy.spin()
    node.theta = np.array(node.theta)
    plt.plot(node.theta[:,0])
    plt.plot(node.theta[:,1])
    plt.plot(node.theta[:,2])
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.show()