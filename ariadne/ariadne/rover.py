from ariadne_msgs.msg import AriadneMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node


class Rover(Node):

    def __init__(self):
        super().__init__('rover')

        # publish path messages
        self.path_publisher = self.create_publisher(Path, 'rover_path', 10)

        # subscribe to the map topic
        self.create_subscription(AriadneMap, 'map', self.map_callback, 10)
        self.obstacles = []
        self.radius = []
        self.goal = []

    def map_callback(self, msg):
        self.obstacles = msg.obstacles
        self.radius = msg.radius
        self.goal = msg.goal

        path = self.plan()
        if path:
            self.publish_path(path)

    def plan(self):
        pass

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


def main(args=None):
    rclpy.init(args=args)

    heli_node = Rover()

    rclpy.spin(heli_node)

    heli_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
