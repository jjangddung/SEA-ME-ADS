#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseWithCovarianceStamped  # ✅ PF pose 구독용

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
import math


class LanePointProjector(Node):
    def __init__(self):
        super().__init__('lane_point_projector')

        # 🔧 실제 토픽 이름에 맞게 수정하세요
        color_image_topic = '/camera/color/image_raw'
        caminfo_topic = '/camera/color/camera_info'
        # cloud_topic = '/camera/depth/filtered_points'
        cloud_topic = '/camera/depth/color/points'

        qos_sensor = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST     # ✅ 오타 수정 완료
        )

        self.bridge = CvBridge()

        # 카메라 내부 파라미터
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 최신 이미지 / 흰색 마스크
        self.latest_image = None
        self.white_mask = None

        # 로깅 플래그
        self.image_logged = False
        self.cloud_logged = False

        # SDF 기반 카메라→base_link 변환 (고정 값)
        # <pose>0 0 0.3 0 0.436332 0</pose>  (x y z roll pitch yaw)
        self.cam_tx = 0.0
        self.cam_ty = 0.0
        self.cam_tz = 0.3       # 카메라 높이
        self.cam_roll = 0.0
        self.cam_pitch = 0.436332   # 약 25도 (앞으로 숙인 각도)
        self.cam_yaw = 0.0

        # 🔁 PF 기반 map -> base_link (lane_pf_pose) 저장용
        self.map_R = None   # 3x3 회전 행렬 (R_map_base)
        self.map_t = None   # (x, y, z) (t_map_base)
        self.map_received = False

        # ⚙️ lane width 파라미터 (1/10 차량 트랙 기준 대략 0.4m 가정)
        self.declare_parameter('lane_width', 0.35)
        self.lane_width = self.get_parameter('lane_width').get_parameter_value().double_value
        self.get_logger().info(f'[init] lane_width = {self.lane_width:.3f} m')

        self.get_logger().info('LanePointProjector (SDF-based, PF map) node started.')

        # ───────────── 구독자 ─────────────
        self.image_sub = self.create_subscription(
            Image, color_image_topic, self.image_callback, qos_sensor
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, caminfo_topic, self.caminfo_callback, qos_sensor
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback, qos_sensor
        )
        # ✅ PF 결과 포즈 (map 기준 base_link) 구독
        self.pf_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/lane_pf_pose',
            self.pf_callback,
            10
        )

        # ───────────── 퍼블리셔 ─────────────
        self.debug_img_pub = self.create_publisher(
            Image, '/debug/lane_points_image', 10
        )
        # camera_depth_optical_frame 기준 차선 포인트
        self.lane_cloud_cam_pub = self.create_publisher(
            PointCloud2, '/camera/depth/lane_points_camera', 10
        )
        # base_link 기준 차선 포인트 (PF measurement용, 그대로 유지)
        self.lane_cloud_base_pub = self.create_publisher(
            PointCloud2, '/camera/depth/lane_points_base_sdf', 10
        )
        # base_link 기준 중앙선 포인트
        self.center_cloud_base_pub = self.create_publisher(
            PointCloud2, '/camera/depth/lane_center_base_sdf', 10
        )
        # ✅ map 기준 중앙선 포인트 (lane_pf_pose 기준)
        self.center_cloud_map_pub = self.create_publisher(
            PointCloud2, '/camera/depth/lane_center_map', 10
        )
        # ✅ map 기준 차선 포인트 (lane_pf_pose 기준)
        self.lane_cloud_map_pub = self.create_publisher(
            PointCloud2, '/camera/depth/lane_points_map', 10
        )

    # ---------------- CameraInfo ----------------
    def caminfo_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'[caminfo_callback] fx={self.fx:.2f}, fy={self.fy:.2f}, '
            f'cx={self.cx:.2f}, cy={self.cy:.2f}'
        )

    # ---------------- 쿼터니언 → 회전행렬 ----------------
    def quat_to_rot(self, x, y, z, w):
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
        ])
        return R

    # ---------------- lane_pf_pose 콜백 (map -> base_link 포즈) ----------------
    def pf_callback(self, msg: PoseWithCovarianceStamped):
        # lane_pf_pose: frame_id = "map", pose of base_link in map
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        R = self.quat_to_rot(qx, qy, qz, qw)
        t = np.array([px, py, pz])

        self.map_R = R
        self.map_t = t
        self.map_received = True

    # ---------------- 이미지 콜백: 흰 차선 마스크 ----------------
    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image = img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 🔧 흰 차선 threshold (필요시 조정)
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        self.white_mask = mask

        if not self.image_logged:
            self.get_logger().info('[image_callback] First image & white mask generated.')
            self.image_logged = True

    # ---------------- PointCloud 콜백 ----------------
    def cloud_callback(self, msg: PointCloud2):
        if self.fx is None or self.latest_image is None or self.white_mask is None:
            self.get_logger().debug('[cloud_callback] Waiting for camera info / image / mask...')
            return

        if not self.cloud_logged:
            self.get_logger().info('[cloud_callback] First pointcloud received.')
            self.cloud_logged = True

        img = self.latest_image.copy()
        mask = self.white_mask
        h, w = img.shape[:2]

        lane_points_cam = []   # camera_depth_optical_frame 기준
        lane_points_base = []  # base_link 기준

        points_iter = pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        )

        step = 5  # 샘플링 간격 (필요하면 1로 줄이기)
        for i, p in enumerate(points_iter):
            if i % step != 0:
                continue

            Xc, Yc, Zc = p  # frame_id = camera_depth_optical_frame (optical frame)

            # 카메라 intrinsics로 이미지 좌표 투영
            u = self.fx * (Xc / Zc) + self.cx
            v = self.fy * (Yc / Zc) + self.cy

            u_i = int(round(u))
            v_i = int(round(v))

            if 0 <= u_i < w and 0 <= v_i < h:
                if mask[v_i, u_i] > 0:
                    # 이미지에 초록 점 (차선 위 포인트)
                    cv2.circle(img, (u_i, v_i), 2, (0, 255, 0), -1)

                    # 1) optical frame 기준 포인트 저장
                    lane_points_cam.append((Xc, Yc, Zc))

                    # 2) optical → camera_link 변환 (REP-103 optical 규약)
                    # optical: X-right, Y-down, Z-forward
                    # camera_link: X-forward, Y-left, Z-up
                    Xl =  Zc
                    Yl = -Xc
                    Zl = -Yc

                    # 3) camera_link → base_link (SDF pitch 적용)
                    theta = self.cam_pitch  # 0.436332 rad

                    # pitch는 camera_link의 Y축 기준 회전이라고 가정
                    Xb =  math.cos(theta) * Xl + math.sin(theta) * Zl
                    Yb =  Yl
                    Zb = -math.sin(theta) * Xl + math.cos(theta) * Zl

                    # 카메라 높이만큼 z 이동
                    Zb += self.cam_tz  # 0.3 m

                    lane_points_base.append((Xb, Yb, Zb))

        # ---------------- camera frame 기준 lane cloud publish ----------------
        if lane_points_cam:
            header_cam = Header()
            header_cam.stamp = msg.header.stamp
            header_cam.frame_id = msg.header.frame_id  # camera_depth_optical_frame

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            lane_cloud_cam = pc2.create_cloud(header_cam, fields, lane_points_cam)
            self.lane_cloud_cam_pub.publish(lane_cloud_cam)

        # ---------------- base_link 기준 lane cloud publish ----------------
        if lane_points_base:
            header_base = Header()
            header_base.stamp = msg.header.stamp
            header_base.frame_id = 'base_link'  # PF에서 measurement로 사용하는 frame

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            lane_cloud_base = pc2.create_cloud(header_base, fields, lane_points_base)
            self.lane_cloud_base_pub.publish(lane_cloud_base)

            # ✅ lane_pf_pose 기준 map frame으로 올린 lane 포인트 (시각화/비교용)
            if self.map_received and self.map_R is not None and self.map_t is not None:
                lane_points_map = []
                for Xb, Yb, Zb in lane_points_base:
                    p_base = np.array([Xb, Yb, Zb])
                    # p_map = R_map_base * p_base + t_map_base  (여기서 map_base = lane_pf_pose)
                    p_map = self.map_R @ p_base + self.map_t
                    lane_points_map.append(tuple(p_map.tolist()))

                header_map = Header()
                header_map.stamp = msg.header.stamp
                header_map.frame_id = 'map'

                lane_cloud_map = pc2.create_cloud(header_map, fields, lane_points_map)
                self.lane_cloud_map_pub.publish(lane_cloud_map)


        # 디버그 이미지 (차선 포인트만 시각화)
        # debug_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        # debug_msg.header = msg.header
        # self.debug_img_pub.publish(debug_msg)
# 
        # try:
            # cv2.imshow('lane points on image', img)
            # cv2.waitKey(1)
        # except Exception as e:
            # self.get_logger().warn(f'cv2.imshow error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LanePointProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
