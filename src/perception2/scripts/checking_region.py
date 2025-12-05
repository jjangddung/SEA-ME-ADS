#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool

class HighwayRegionNode(Node):
    def __init__(self):
        super().__init__('highway_region_node')

        # 예시: 하이웨이 구간 A, B를 좌표 박스로 정의
        # 값은 rviz에서 pose 찍어보면서 잡으면 됨
        self.highway_zones = [
            {'xmin': 6.27, 'xmax': 14.7, 'ymin': 9.08, 'ymax': 12.6}
            # {'xmin': 15.0, 'xmax': 25.0, 'ymin': -0.5, 'ymax': 0.5},
        ]
        # self.parking_zones = [
            # {'xmin': 2.0, 'xmax': 8.0, 'ymin': -1.0, 'ymax': 1.0},
            # {'xmin': 15.0, 'xmax': 25.0, 'ymin': -0.5, 'ymax': 0.5},
        # ]
# 
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/lane_pf_pose',
            self.pose_callback,
            10
        )

        self.flag_pub = self.create_publisher(Bool, '/is_on_highway', 10)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        on_highway = False
        for z in self.highway_zones:
            if z['xmin'] <= x <= z['xmax'] and z['ymin'] <= y <= z['ymax']:
                on_highway = True
                break

        out = Bool()
        out.data = on_highway
        self.flag_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HighwayRegionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
