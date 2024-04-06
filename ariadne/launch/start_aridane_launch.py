from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ariadne',
            executable='heli',
            name='heli'
        ),
        Node(
            package='ariadne',
            executable='rover',
            name='rover'
        ),
    ])