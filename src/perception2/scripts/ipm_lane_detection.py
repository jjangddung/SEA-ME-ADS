#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

import cv2
import numpy as np
import math
import struct

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_py')

        # ==========================================
        # 1. 파라미터 설정 (C++ 코드 값 기반)
        # ==========================================
        self.declare_parameter('lane_width', 0.35)
        self.declare_parameter('cam_height', 0.3)      # cam_tz
        self.declare_parameter('cam_pitch', 0.436332)  # 약 25도
        
        self.lane_width = self.get_parameter('lane_width').value
        self.cam_height = self.get_parameter('cam_height').value
        self.cam_pitch  = self.get_parameter('cam_pitch').value

        # BEV 변환을 위한 ROI 설정 (보고 싶은 실제 바닥 영역)
        # base_link 기준 (X: 전방, Y: 좌측)
        self.x_min = 0.2   # 차 앞 0.2m 부터 (보닛 가림 고려)
        self.x_max = 1.5   # 차 앞 1.5m 까지
        self.y_min = -0.8  # 우측 0.8m
        self.y_max = 0.8   # 좌측 0.8m
        
        # 생성할 BEV 이미지 해상도
        self.bev_width = 320
        self.bev_height = 240

        # 내부 변수 초기화
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.H_matrix = None
        self.map_received = False
        self.map_R = np.eye(3)
        self.map_t = np.zeros(3)

        # ==========================================
        # 2. ROS 통신 설정
        # ==========================================
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.cb_cam_info, qos_sensor)
        
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self.cb_image, qos_sensor)
            
        self.sub_pf = self.create_subscription(
            PoseWithCovarianceStamped, '/lane_pf_pose', self.cb_pf, 10)

        # Publishers
        self.pub_lane_base = self.create_publisher(PointCloud2, '/lane_points_base', 10)
        self.pub_lane_map  = self.create_publisher(PointCloud2, '/lane_points_map', 10)
        self.pub_debug_img = self.create_publisher(Image, '/debug/bev_image', 10)

        self.cv_bridge = CvBridge()
        self.get_logger().info("Lane Detection Node (RGB-IPM) Started.")

    # ==========================================
    # 3. 콜백 함수들
    # ==========================================
    def cb_cam_info(self, msg):
        """카메라 내부 파라미터 수신 시 1회 실행하여 Homography 행렬 계산"""
        if self.fx == 0.0:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f"Camera Info Received: fx={self.fx:.1f}, fy={self.fy:.1f}")
            
            # Homography 행렬 계산
            self.compute_homography()

    def cb_pf(self, msg):
        """Particle Filter의 추정 위치 수신 (Map 좌표계 변환용)"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        self.map_t = np.array([p.x, p.y, p.z])
        self.map_R = self.quat_to_rot(q.x, q.y, q.z, q.w)
        self.map_received = True

    def cb_image(self, msg):
        """메인 처리 루프"""
        if self.H_matrix is None:
            return # 캘리브레이션 전이면 대기

        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 1. IPM (Top-view 변환)
        bev_img = cv2.warpPerspective(cv_img, self.H_matrix, (self.bev_width, self.bev_height))

        # 2. 색상 필터링 (HSV)
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
        # C++ 코드와 동일한 Threshold 적용
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 모폴로지 연산 (노이즈 제거)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # 3. 좌표 추출 (이미지 픽셀 -> 실제 미터 좌표)
        # non_zero: (row_indices, col_indices)
        rows, cols = np.where(mask > 0)
        
        if len(rows) == 0:
            return

        # 픽셀 좌표를 미터 좌표로 변환
        # ROI 설정에 따라 선형 보간
        # X: rows 0 -> x_max, rows height -> x_min (이미지 위쪽이 먼 곳)
        # Y: cols 0 -> y_max (Left), cols width -> y_min (Right) (이미지 왼쪽이 ROS +Y)
        
        # Scale 계산 (1회만 해도 되지만 명시적으로 표현)
        scale_x = (self.x_min - self.x_max) / self.bev_height
        scale_y = (self.y_min - self.y_max) / self.bev_width

        # 변환 식
        real_x = self.x_max + rows * scale_x
        real_y = self.y_max + cols * scale_y
        real_z = np.zeros_like(real_x) # 바닥이므로 0

        # C++ 코드의 범위 필터 (X: 0.15 ~ 1.0) 적용
        # 이미 ROI 설정(self.x_min/max)으로 1차 필터링 되었으나, 
        # C++ 코드처럼 엄격하게 자르려면 아래 조건 추가
        valid_idx = (real_x >= 0.15) & (real_x <= 1.0)
        
        if np.sum(valid_idx) == 0:
            return

        real_x = real_x[valid_idx]
        real_y = real_y[valid_idx]
        real_z = real_z[valid_idx]

        # 다운샘플링 (너무 많은 점을 보내면 PF 부하) -> 5개 중 1개
        real_x = real_x[::5]
        real_y = real_y[::5]
        real_z = real_z[::5]

        # 4. PointCloud2 생성 및 발행 (Base Link)
        points_base = np.column_stack((real_x, real_y, real_z))
        pc_msg_base = self.create_pointcloud2(points_base, msg.header.stamp, "base_link")
        self.pub_lane_base.publish(pc_msg_base)

        # 5. Map Frame 변환 및 발행 (시각화용)
        if self.map_received:
            # P_map = R * P_base + T
            # (N, 3) dot (3, 3).T + (3,)
            points_map = np.dot(points_base, self.map_R.T) + self.map_t
            pc_msg_map = self.create_pointcloud2(points_map, msg.header.stamp, "map")
            self.pub_lane_map.publish(pc_msg_map)

        # 디버그 이미지 발행
        self.pub_debug_img.publish(self.cv_bridge.cv2_to_imgmsg(mask, "mono8"))

    # ==========================================
    # 4. 헬퍼 함수들 (계산/변환)
    # ==========================================
    def compute_homography(self):
        """Intrinsic + Extrinsic을 이용해 바닥 평면 투영 행렬(H) 계산"""
        # 1. 바닥(Ground)의 4개 모서리 좌표 정의 (Base_link 기준)
        # 순서: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        # ROS 좌표계: X=Front, Y=Left
        
        src_ground = np.array([
            [self.x_max, self.y_max, 0.0], # Top-Left (멀고 왼쪽)
            [self.x_max, self.y_min, 0.0], # Top-Right (멀고 오른쪽)
            [self.x_min, self.y_max, 0.0], # Bottom-Left (가깝고 왼쪽)
            [self.x_min, self.y_min, 0.0]  # Bottom-Right (가깝고 오른쪽)
        ])

        # 2. Base -> Camera 좌표 변환
        # C++ 코드 로직 역설계:
        # Camera is at (0, 0, cam_height) relative to base, rotated by pitch.
        # Point P_cam = R_pitch * (P_base - T_cam)
        # ROS(Front/Left/Up) -> Optical(Right/Down/Fwd) conversion needed

        pixel_coords = []
        for p in src_ground:
            # A. Translation (카메라 위치만큼 빼기)
            dx = p[0]
            dy = p[1]
            dz = p[2] - self.cam_height

            # B. Rotation (Pitch down)
            # Pitch는 Y축 회전. 차가 앞으로 보는데 카메라가 아래를 보려면 Pitch는 양수(C++값 참조)
            theta = self.cam_pitch
            
            # base_link 기준 회전된 좌표
            rot_x = dx * math.cos(theta) - dz * math.sin(theta)
            rot_y = dy
            rot_z = dx * math.sin(theta) + dz * math.cos(theta)

            # C. Optical Frame 변환 (표준 ROS 카메라 좌표계)
            # Optical X(Right) = -Base Y(Left)
            # Optical Y(Down)  = -Base Z(Up)
            # Optical Z(Fwd)   =  Base X(Front) -> 여기선 rot_z가 위쪽인데?
            
            # C++ 코드의 역변환 로직 참조:
            # Xb = cos(th)*Xl + sin(th)*Zl
            # Zb = -sin(th)*Xl + cos(th)*Zl
            # 이를 역으로 풀면 (Optical 기준):
            # Xl(Optical X) = -Yb (Left의 반대 = Right)
            # Yl(Optical Y) = -Zb (Up의 반대 = Down) 
            # Zl(Optical Z) = Xb? 아니오, Pitch가 적용된 축을 봐야함.
            
            # 간단하게 기하학적 접근:
            # 카메라 앞으로 나가는 축(Optical Z) = rot_x
            # 카메라 오른쪽(Optical X) = -rot_y (Base Left의 반대)
            # 카메라 아래(Optical Y) = -rot_z (Base Up의 반대)
            
            opt_x = -rot_y
            opt_y = -rot_z
            opt_z =  rot_x

            # D. Projection (Intrinsics)
            if opt_z <= 0.01: continue # 카메라 뒤쪽 예외처리

            u = self.fx * (opt_x / opt_z) + self.cx
            v = self.fy * (opt_y / opt_z) + self.cy
            pixel_coords.append([u, v])

        if len(pixel_coords) != 4:
            self.get_logger().error("Error computing BEV coordinates")
            return

        src_pts = np.float32(pixel_coords)

        # 3. Destination (BEV 이미지 상의 좌표)
        # (0,0) -> (x_max, y_max) : Top-Left
        dst_pts = np.float32([
            [0, 0],                         # Top-Left
            [self.bev_width, 0],            # Top-Right
            [0, self.bev_height],           # Bottom-Left
            [self.bev_width, self.bev_height] # Bottom-Right
        ])

        self.H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.get_logger().info("Homography Matrix Calculated Successfully.")

    def quat_to_rot(self, x, y, z, w):
        """쿼터니언 -> 3x3 회전행렬"""
        # C++ implementation ported to numpy
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z

        R = np.array([
            [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy)],
            [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx)],
            [      2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
        ])
        return R

    def create_pointcloud2(self, points, stamp, frame_id):
        """Numpy Points(N,3) -> Sensor_msgs/PointCloud2"""
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        
        # Fast serialization
        msg.data = points.astype(np.float32).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()