#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class DepthRingFilter(Node):
    """
    /camera/depth/color/points 를 구독해서
    (Zc^2 + Xc^2) in [min_r^2, max_r^2] 범위만 남긴
    필터된 PointCloud2를 퍼블리시하는 노드.
    """

    def __init__(self):
        super().__init__("depth_ring_filter")

        # 파라미터: 입력/출력 토픽 + 거리 범위
        self.declare_parameter("input_topic", "/camera/depth/color/points")
        self.declare_parameter("output_topic", "/camera/depth/filtered_points")
        self.declare_parameter("min_radius", 0.1)   # 기존 코드의 0.1
        self.declare_parameter("max_radius", 0.7)   # 기존 코드의 0.7

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.min_r = self.get_parameter("min_radius").get_parameter_value().double_value
        self.max_r = self.get_parameter("max_radius").get_parameter_value().double_value

        self.min_r2 = self.min_r * self.min_r
        self.max_r2 = self.max_r * self.max_r

        qos_sensor = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_sensor
        )

        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        self.get_logger().info(
            f"[init] DepthRingFilter started. "
            f"input={input_topic}, output={output_topic}, "
            f"radius ∈ [{self.min_r:.2f}, {self.max_r:.2f}] m"
        )

    def cloud_callback(self, msg: PointCloud2):
        # PointCloud2 → 반복자
        points_iter = pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )

        filtered_points = []
        min_r2 = self.min_r2
        max_r2 = self.max_r2

        # 원래 lane_odom.py 에 있던 조건:
        # if (Zc**2 + Xc**2) < 0.1**2 or (Zc**2 + Xc**2) > 0.7**2: continue
        # 를 여기로 옮김
        for Xc, Yc, Zc in points_iter:
            r2 = Xc * Xc + Zc * Zc
            if r2 < min_r2 or r2 > max_r2:
                continue
            filtered_points.append((Xc, Yc, Zc))

        if not filtered_points:
            # 남는 포인트 없으면 굳이 퍼블리시 안 해도 됨
            return

        # 새 PointCloud2 만들기 (frame_id, stamp는 그대로 유지)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        filtered_cloud = pc2.create_cloud(msg.header, fields, filtered_points)
        self.pub.publish(filtered_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRingFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
