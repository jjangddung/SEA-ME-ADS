#!/usr/bin/env python3
# coding: utf-8

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class LaneMaskNode(Node):
    def __init__(self):
        super().__init__("lane_mask_node")

        self.bridge = CvBridge()

        qos_sensor = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        rgb_topic = "/camera/color/image_raw"
        mask_topic = "/lane/white_mask"

        self.sub_rgb = self.create_subscription(
            Image, rgb_topic, self.rgb_cb, qos_sensor
        )
        self.pub_mask = self.create_publisher(Image, mask_topic, 10)

        # 수정된 부분
        self.get_logger().info(
            f"[lane_mask_node] started. Sub: {rgb_topic}, Pub: {mask_topic}"
        )


    def rgb_cb(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        out_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
