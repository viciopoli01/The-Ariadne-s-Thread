#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import numpy as np
import math


class Controller:

    def __init__(self, kp, kd, ki, starting_vel=1.5):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.last_error = 0
        self.integral = 0

        self.x_vel = starting_vel

        # publisher cmd_vel messages
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.moving = False
        self.cmd_vel=Twist()

        self.stop()


    def stop(self):
        self.cmd_vel.linear.x = 0.
        self.cmd_vel.angular.z = 0.
        self.cmd_vel_publisher.publish(self.cmd_vel)
        self.moving = False

    
    def control(self, current_pose, goal_pose):
        self.moving = True
        # implement the controller here
        # it has to reach the node position (x, y, theta) given the pose of the rover
        # send out cmd_vel messages to the rover
        
        # Update last error for next iteration
        error_theta = goal_pose[2] - current_pose[2]

        error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))
        rospy.loginfo('error_theta: {}'.format(error_theta*180/np.pi))
        control_theta = self.kp * error_theta + self.kd * (error_theta - self.last_error)

        self.last_error = error_theta

        self.cmd_vel.linear.x = self.x_vel
        self.cmd_vel.angular.z = control_theta

        # rospy.loginfo('Sending cmd_vel: {}'.format(cmd_vel))
        self.cmd_vel_publisher.publish(self.cmd_vel)

        rospy.sleep(1.)


    def __call__(self, current_pose, goal_pose):
        self.control(current_pose, goal_pose)
        return self.moving