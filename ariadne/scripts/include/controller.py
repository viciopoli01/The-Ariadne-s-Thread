#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import numpy as np
import math


class Controller:

    def __init__(self, kp, kd, ki, starting_vel=3.0):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.previous_error = 0.0
        self.integral = 0

        self.last_time=None

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

    
    def control(self, current_pose, goal_pose, current_time):
        self.moving = True
        
        if self.last_time is None:
            self.last_time = current_time
            return 0

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

        # limit the output to [-4, 4]
        if output > 4.0:
            output = 4.0
        elif output < -4.0:
            output = -4.0


        self.cmd_vel.linear.x = output

        # the roation is +- 1.0 according to the sign of the error
        self.cmd_vel.angular.z = np.sign(error)*10. if abs(error) > 0.1 else 0.

        # anti-windup
        if self.integral > 3.0:
            self.integral = 3.0
        elif self.integral < -3.0:
            self.integral = -3.0

        self.previous_error = error
        self.last_time = current_time

        # rospy.loginfo('Sending cmd_vel: {}'.format(cmd_vel))
        self.cmd_vel_publisher.publish(self.cmd_vel)

        rospy.sleep(1.)


    def __call__(self, current_pose, goal_pose, current_time):
        self.control(current_pose, goal_pose, current_time)
        return self.moving