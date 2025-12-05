#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion

import numpy as np
import math
import cv2
import os


class SimpleMapPublisher(Node):
    def __init__(self):
        super().__init__('simple_map_publisher')

        # 파라미터
        self.declare_parameter('image_path', '/home/dongmin/bfmc/Simulator/src/sim_pkg/config/Track.pgm')
        self.declare_parameter('resolution', 0.002125)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        self.declare_parameter('origin_yaw', 0.0)

        image_path = self.get_parameter('image_path').get_parameter_value().string_value
        res = self.get_parameter('resolution').get_parameter_value().double_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        origin_x = self.get_parameter('origin_x').get_parameter_value().double_value
        origin_y = self.get_parameter('origin_y').get_parameter_value().double_value
        origin_yaw = self.get_parameter('origin_yaw').get_parameter_value().double_value

        if not image_path or not os.path.isfile(image_path):
            self.get_logger().error(f"Invalid image_path: {image_path}")
            raise SystemExit

        # PGM 그대로 읽어서 occupancygrid로 변환
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Failed to load image: {image_path}")
            raise SystemExit

        height, width = img.shape[:2]

        # 맵 좌표계랑 맞추려고 위아래 플립 (필요 없으면 이 줄 지워도 됨)
        img = cv2.flip(img, 0)

        # 0(흰색)~255(검정) → 0~100
        # 흰색 = free(0), 검정 = occupied(100)
        # occ = 100 - (img.astype(np.float32) / 255.0 * 100.0)
        # occ = np.clip(occ, 0, 100).astype(np.int8)

        occ = (img.astype(np.float32) / 255.0 * 100.0)
        occ = np.clip(occ, 0, 100).astype(np.int8)

        self.grid = OccupancyGrid()
        self.grid.header = Header()
        self.grid.header.frame_id = frame_id

        self.grid.info.resolution = res
        self.grid.info.width = width
        self.grid.info.height = height

        q = Quaternion()
        q.w = math.cos(origin_yaw * 0.5)
        q.z = math.sin(origin_yaw * 0.5)

        self.grid.info.origin = Pose(
            position=Point(x=origin_x, y=origin_y, z=0.0),
            orientation=q
        )

        self.grid.data = occ.flatten().tolist()

        self.pub = self.create_publisher(OccupancyGrid, '/map', 1)

        # 1Hz로 계속 퍼블리시 → QoS 꼬여도 RViz가 언젠간 받음
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info(
            f"Publishing map from {image_path} as /map "
            f"({width} x {height}, res={res})"
        )

    def timer_cb(self):
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.grid)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
