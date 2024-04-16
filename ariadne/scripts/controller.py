#!/usr/bin/env python3

from ariadne.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rospy

from include.utils import map_updater, obs2array, path2msg, msg2pose

from dynamic_reconfigure.server import Server
from ariadne.cfg import AriadneConfig
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import tf.transformations

import numpy as np
import math


from include.utils import msg2path, msg2pose

class Controller():

    def __init__(self):
        self.kp = 20.
        self.kd = 7.
        self.ki = 0.
        self.previous_error = 0.0
        self.integral = 0

        self.last_time=None

        self.x_vel = 2.

        # publisher cmd_vel messages
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # subscribe to path topic
        self.path_subscriber = rospy.Subscriber('rover_path', Path, self.path_callback)

        # subscribe to odometry topic
        self.odometry_subscriber = rospy.Subscriber('/curiosity_mars_rover/odom', Odometry, self.odometry_callback)

        self.moving = False
        self.cmd_vel=Twist()

        self.stop()
        self.path = []
        self.pose = None

        self.next_goal = 0

    def stop(self):
        self.cmd_vel.linear.x = 0.
        self.cmd_vel.angular.z = 0.
        self.cmd_vel_publisher.publish(self.cmd_vel)
        self.moving = False

    def odometry_callback(self, msg):
        self.pose = msg2pose(msg)

        if len(self.path)>0:
            self.control(self.pose, self.path[self.next_goal], rospy.get_time())
            
    def path_callback(self, msg):
        if not self.moving:
            self.path = msg2path(msg)
    
    def control(self, current_pose, goal_pose, current_time):
        self.moving = True

        if np.linalg.norm(np.array(current_pose[:2]) - np.array(goal_pose[:2])) < 0.1:
            rospy.loginfo('Next goal')
            if self.next_goal >= len(self.path):
                self.stop()
                rospy.loginfo('Goal reached')
                return
            self.next_goal += 1
        
        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time

        error_theta = goal_pose[2] - current_pose[2]

        rospy.loginfo('Goal theta: {}'.format(goal_pose[2]))
        rospy.loginfo('Current theta: {}'.format(current_pose[2]))
        rospy.loginfo('Error theta: {}'.format(error_theta))
        
        error = np.arctan2(np.sin(error_theta), 
                           np.cos(error_theta))

        # limit the yaw angle to [-pi, pi]

        if error > np.pi:
            error -= 2 * np.pi
        elif error < -np.pi:
            error += 2 * np.pi

        rospy.loginfo('Error: {}'.format(error))

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        if output > 10.0:
            output = 10.0
        elif output < -10.0:
            output = -10.0

        self.cmd_vel.linear.x = self.x_vel
        self.cmd_vel.angular.z = output

        # anti-windup
        if self.integral > 3.0:
            self.integral = 3.0
        elif self.integral < -3.0:
            self.integral = -3.0

        self.previous_error = error
        self.last_time = current_time

        # rospy.loginfo('Sending cmd_vel: {}'.format(cmd_vel))
        self.cmd_vel_publisher.publish(self.cmd_vel)

        # rospy.sleep(.5)


    # on node shutdown
    def shutdown(self):
        rospy.loginfo('Shutting down')
        rospy.signal_shutdown('Shutting down')
        self.stop()


if __name__ == '__main__':
    rospy.init_node('controller', log_level=rospy.DEBUG)

    node = Controller()

    rospy.spin()