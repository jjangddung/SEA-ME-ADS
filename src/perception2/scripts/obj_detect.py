#!/usr/bin/env python3
import os
import math
import threading
from typing import List, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge

from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # side-effect 초기화


# =========================
# 사용자 정의 YOLO 클래스 이름 (C++ DEFAULT_CLASS_NAMES와 동일)
# =========================
# class를 config파일로 따로 빼면 좋을 것 같음
# 시각화하는 부분도 따로 정리하는게 좋아보인다.
CUSTOM_CLASS_NAMES: List[str] = [
    "ONEWAY",
    "HIGHWAYENTRANCE",
    "STOPSIGN",
    "ROUNDABOUT",
    "PARK",
    "CROSSWALK",
    "NOENTRY",
    "HIGHWAYEXIT",
    "PRIORITY",
    "LIGHTS",
    "BLOCK",
    "PEDESTRIAN",
    "CAR",
]


def get_class_name(cls_id: int) -> str:
    """YOLO class index -> 문자열 이름.
    범위를 벗어나면 'class_{id}'로 fallback."""
    if 0 <= cls_id < len(CUSTOM_CLASS_NAMES):
        return CUSTOM_CLASS_NAMES[cls_id]
    return f"class_{cls_id}"


# =========================
# YOLO + TensorRT (TensorRT 10 API)
# =========================

class YoloTRT:
    def __init__(self, engine_path: str,
                 score_thresh: float = 0.25,
                 nms_thresh: float = 0.45,
                 top_k: int = 300):
        self.logger = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # ---- Tensor 이름 기반 API 사용 ----
        nb_tensors = self.engine.num_io_tensors
        if nb_tensors < 2:
            raise RuntimeError(f"Engine IO tensor count is {nb_tensors}, expected at least 2")

        input_name = None
        output_name = None
        for i in range(nb_tensors):
            name = self.engine.get_tensor_name(i)
            role = self.engine.get_tensor_mode(name)  # INPUT / OUTPUT
            if role == trt.TensorIOMode.INPUT and input_name is None:
                input_name = name
            elif role == trt.TensorIOMode.OUTPUT and output_name is None:
                output_name = name

        if input_name is None or output_name is None:
            raise RuntimeError("Failed to find input/output tensor names in engine")

        self.input_name = input_name
        self.output_name = output_name

        # shape 정보 얻기
        input_shape = self.engine.get_tensor_shape(self.input_name)
        input_shape = [int(d) for d in input_shape]

        # dynamic 이면 여기선 안 다룸 (test용이니까 static만 지원)
        if any(d == -1 for d in input_shape):
            raise RuntimeError(
                f"Dynamic shape engine detected for {self.input_name}: {input_shape}. "
                f"Static shape engine만 지원하는 테스트 코드입니다."
            )

        self.context.set_input_shape(self.input_name, input_shape)

        output_shape = self.engine.get_tensor_shape(self.output_name)
        output_shape = [int(d) for d in output_shape]

        self.input_dims = input_shape
        self.output_dims = output_shape

        if len(self.input_dims) != 4:
            raise RuntimeError(f"Unexpected input dims: {self.input_dims} (expected NCHW)")

        self.batch_size = self.input_dims[0]
        self.input_channels = self.input_dims[1]
        self.input_h = self.input_dims[2]
        self.input_w = self.input_dims[3]

        if self.batch_size != 1:
            print(f"[YoloTRT] Warning: batch_size={self.batch_size}, this wrapper assumes 1.")

        def volume(dims):
            v = 1
            for x in dims:
                v *= int(x)
            return v

        self.input_size = volume(self.input_dims)
        self.output_size = volume(self.output_dims)

        # GPU 메모리
        self.d_input = cuda.mem_alloc(self.input_size * 4)   # float32
        self.d_output = cuda.mem_alloc(self.output_size * 4)
        self.stream = cuda.Stream()

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    # ---------- 전처리 (letterbox) ----------
    def preprocess(self, frame_bgr: np.ndarray):
        target_w = self.input_w
        target_h = self.input_h
        orig_h, orig_w = frame_bgr.shape[:2]

        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        pad_w = int((target_w - new_w) * 0.5)
        pad_h = int((target_h - new_h) * 0.5)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        letterboxed[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))  # HWC → CHW
        input_flat = chw.ravel().astype(np.float32).copy()

        return input_flat, ratio, pad_w, pad_h

    # ---------- NMS ----------
    def nms(self, boxes: List[np.ndarray], scores: List[float]) -> List[int]:
        if not boxes:
            return []

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        max_keep = self.top_k if self.top_k > 0 else len(order)

        while order.size > 0 and len(keep) < max_keep:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    # ---------- postprocess (YOLOv8 스타일) ----------
    def postprocess(self, output: np.ndarray, orig_size: Tuple[int, int],
                    ratio: float, pad_w: int, pad_h: int):
        orig_w, orig_h = orig_size
        out = output.flatten()
        dims = list(self.output_dims)

        # 배치 차원 제거
        if len(dims) >= 1 and dims[0] == self.batch_size:
            dims = dims[1:]

        if len(dims) == 2:
            d0, d1 = dims
            channels_first = (d0 < d1)
            features = d0 if channels_first else d1
            boxes = d1 if channels_first else d0
        elif len(dims) == 3 and dims[2] == 1:
            d0, d1, _ = dims
            channels_first = (d0 < d1)
            features = d0 if channels_first else d1
            boxes = d1 if channels_first else d0
        else:
            return []

        if features < 5 or boxes <= 0:
            return []

        num_classes = features - 4
        if features * boxes != out.size:
            return []

        boxes_list = []
        scores_list = []
        classes_list = []

        stride = boxes if channels_first else features

        def access(c, b):
            if channels_first:
                return out[c * boxes + b]
            else:
                base = b * stride
                return out[base + c]

        for b in range(boxes):
            cx = access(0, b)
            cy = access(1, b)
            w = access(2, b)
            h = access(3, b)

            best_score = -1e10
            best_class = -1
            for cls in range(num_classes):
                s = access(4 + cls, b)
                if s > best_score:
                    best_score = s
                    best_class = cls

            if best_class < 0 or best_score < self.score_thresh:
                continue

            half_w = 0.5 * w
            half_h = 0.5 * h
            x1 = cx - half_w
            y1 = cy - half_h
            x2 = cx + half_w
            y2 = cy + half_h

            # letterbox 해제
            x1 = (x1 - pad_w) / ratio
            x2 = (x2 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio
            y2 = (y2 - pad_h) / ratio

            x1 = float(np.clip(x1, 0, orig_w - 1))
            x2 = float(np.clip(x2, 0, orig_w - 1))
            y1 = float(np.clip(y1, 0, orig_h - 1))
            y2 = float(np.clip(y2, 0, orig_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            scores_list.append(float(best_score))
            classes_list.append(int(best_class))

        if not boxes_list:
            return []

        keep_idx = self.nms(boxes_list, scores_list)
        dets = []
        for idx in keep_idx:
            x1, y1, x2, y2 = boxes_list[idx]
            dets.append({
                "box": (int(math.floor(x1)), int(math.floor(y1)),
                        int(math.ceil(x2)), int(math.ceil(y2))),
                "score": scores_list[idx],
                "cls": classes_list[idx]
            })
        return dets

    # ---------- infer (TensorRT 10: set_tensor_address + execute_async_v3) ----------
    def infer(self, frame_bgr: np.ndarray):
        input_flat, ratio, pad_w, pad_h = self.preprocess(frame_bgr)

        # host → device
        cuda.memcpy_htod_async(self.d_input, input_flat, self.stream)

        # Tensor 주소 설정
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # v3 실행
        self.context.execute_async_v3(self.stream.handle)

        # device → host
        host_output = np.empty(self.output_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_output, self.d_output, self.stream)
        self.stream.synchronize()

        h, w = frame_bgr.shape[:2]
        dets = self.postprocess(host_output, (w, h), ratio, pad_w, pad_h)
        return dets


# =========================
# 쿼터니언 → 회전행렬
# =========================

def quat_to_rot(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ], dtype=np.float32)
    return R


# =========================
# 메인 노드: YOLO + depth → base_link → map
# =========================

class YoloDepthMapNode(Node):
    def __init__(self):
        super().__init__("yolo_depth_map_node")

        # ---- 파라미터 ----
        self.declare_parameter("engine_path", "/home/dongmin/bfmc/perception_ws/perception/model/model.engine")
        self.declare_parameter("score_threshold", 0.25)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("top_k", 300)

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("pf_pose_topic", "/lane_pf_pose")

        # SDF camera → base_link (lane_odom.py와 맞춰야 하는 값)
        self.declare_parameter("cam_pitch", 0.436332)  # 약 25도
        self.declare_parameter("cam_height", 0.3)      # z=0.3m 정도

        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        score_th = self.get_parameter("score_threshold").get_parameter_value().double_value
        nms_th = self.get_parameter("nms_threshold").get_parameter_value().double_value
        top_k = self.get_parameter("top_k").get_parameter_value().integer_value

        self.cam_pitch = self.get_parameter("cam_pitch").get_parameter_value().double_value
        self.cam_height = self.get_parameter("cam_height").get_parameter_value().double_value

        self.get_logger().info(f"[init] loading engine: {engine_path}")
        self.yolo = YoloTRT(engine_path, score_th, nms_th, top_k)

        self.bridge = CvBridge()

        # 카메라 intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_rgb_header = None

        # PF 기반 map -> base_link
        self.map_R = None
        self.map_t = None
        self.map_received = False

        self.lock = threading.Lock()

        # 센서용 QoS
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        pf_topic = self.get_parameter("pf_pose_topic").get_parameter_value().string_value

        # ---- subscriber ----
        self.sub_rgb = self.create_subscription(Image, rgb_topic, self.rgb_cb, qos_sensor)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_cb, qos_sensor)
        self.sub_caminfo = self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, qos_sensor)
        self.sub_pf = self.create_subscription(PoseWithCovarianceStamped, pf_topic, self.pf_cb, 10)

        # ---- publisher ----
        self.pc_base_pub = self.create_publisher(PointCloud2, "/yolo/detections_base", 10)
        self.pc_map_pub = self.create_publisher(PointCloud2, "/yolo/detections_map", 10)
        self.annotated_pub = self.create_publisher(Image, "/yolo/annotated", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/yolo/detections_markers", 10)

        self.get_logger().info("YoloDepthMapNode started (A안: depth image 기반, TRT10).")

        self.timer = self.create_timer(0.05, self.process)

    # ---------- 콜백들 ----------

    def caminfo_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]
        self.get_logger().info(
            f"[caminfo] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

    def pf_cb(self, msg: PoseWithCovarianceStamped):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        R = quat_to_rot(qx, qy, qz, qw)
        t = np.array([px, py, pz], dtype=np.float32)
        self.map_R = R
        self.map_t = t
        self.map_received = True

    def rgb_cb(self, msg: Image):
        with self.lock:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb_header = msg.header

    def depth_cb(self, msg: Image):
        with self.lock:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    # ---------- depth ROI → median distance → optical frame 3D ----------
    def depth_to_3d_optical(self, depth_image: np.ndarray, box: Tuple[int, int, int, int]):
        if depth_image is None or self.fx is None:
            return None

        x1, y1, x2, y2 = box
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2

        h, w = depth_image.shape[:2]
        if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
            return None

        region_size = max(2, min(x2 - x1, y2 - y1) // 6)
        rx1 = max(0, center_x - region_size)
        ry1 = max(0, center_y - region_size)
        rx2 = min(w, center_x + region_size)
        ry2 = min(h, center_y + region_size)

        roi = depth_image[ry1:ry2, rx1:rx2]

        if depth_image.dtype == np.uint16:
            depth_m = roi.astype(np.float32) * 0.001  # mm → m
        elif depth_image.dtype == np.float32:
            depth_m = roi.astype(np.float32)
        else:
            return None

        valid = depth_m[(depth_m > 0.1) & (depth_m < 10.0)]
        if valid.size == 0:
            return None

        z = float(np.median(valid))
        u = float(center_x)
        v = float(center_y)

        Xc = (u - self.cx) * z / self.fx
        Yc = (v - self.cy) * z / self.fy
        Zc = z

        return Xc, Yc, Zc

    # ---------- optical → camera_link → base_link ----------
    def optical_to_base(self, Xc, Yc, Zc):
        # optical: X-right, Y-down, Z-forward
        # camera_link: X-forward, Y-left, Z-up
        Xl = Zc
        Yl = -Xc
        Zl = -Yc

        theta = self.cam_pitch
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        Xb = cos_t * Xl + sin_t * Zl
        Yb = Yl
        Zb = -sin_t * Xl + cos_t * Zl

        Zb += self.cam_height  # 카메라 높이

        return Xb, Yb, Zb

    def base_to_map(self, Xb, Yb, Zb):
        if not self.map_received or self.map_R is None or self.map_t is None:
            return None
        p_base = np.array([Xb, Yb, Zb], dtype=np.float32)
        p_map = self.map_R @ p_base + self.map_t
        return float(p_map[0]), float(p_map[1]), float(p_map[2])

    # ---------- 메인 처리 루프 ----------
    def process(self):
        with self.lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            frame = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
            header = self.latest_rgb_header

        dets = self.yolo.infer(frame)
        if not dets:
            return

        points_base = []
        points_map = []
        markers: List[Marker] = []

        annotated = frame.copy()
        marker_id = 0

        for det in dets:
            box = det["box"]
            cls_id = det["cls"]
            score = det["score"]
            x1, y1, x2, y2 = box

            cls_name = get_class_name(cls_id)

            # 디버그용 bbox + 라벨(커스텀 클래스 이름)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_txt = f"{cls_name}:{score:.2f}"
            cv2.putText(annotated, label_txt, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            res = self.depth_to_3d_optical(depth, box)
            if res is None:
                continue
            Xc, Yc, Zc = res

            # optical → base_link
            Xb, Yb, Zb = self.optical_to_base(Xc, Yc, Zc)

            # PointCloud2에 class_id와 score까지 포함
            points_base.append((Xb, Yb, Zb, float(cls_id), float(score)))

            # base_link → map
            map_res = self.base_to_map(Xb, Yb, Zb)
            if map_res is not None:
                Xm, Ym, Zm = map_res
                points_map.append((Xm, Ym, Zm, float(cls_id), float(score)))

                # ---------- Marker 생성 (TEXT_VIEW_FACING) ----------
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = header.stamp
                marker.ns = "yolo_detections"
                marker.id = marker_id
                marker_id += 1

                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD

                marker.pose.position.x = Xm
                marker.pose.position.y = Ym
                marker.pose.position.z = Zm + 0.5  # 바닥에서 조금 위로

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                # TEXT_VIEW_FACING에서 scale.z = 글자 높이
                marker.scale.x = 0.0
                marker.scale.y = 0.0
                marker.scale.z = 0.3

                # 흰색 텍스트
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0

                marker.text = f"{cls_name} {score:.2f}"

                marker.lifetime = Duration(sec=0, nanosec=int(0.5 * 1e9))  # 0.5초

                markers.append(marker)

        # PointCloud2 fields (x, y, z, class_id, score)
        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='class_id',  offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='score',     offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        if points_base:
            header_base = Header()
            header_base.stamp = header.stamp
            header_base.frame_id = "base_link"
            cloud_base = pc2.create_cloud(header_base, fields, points_base)
            self.pc_base_pub.publish(cloud_base)

        if points_map:
            header_map = Header()
            header_map.stamp = header.stamp
            header_map.frame_id = "map"
            cloud_map = pc2.create_cloud(header_map, fields, points_map)
            self.pc_map_pub.publish(cloud_map)

        if markers:
            marker_array = MarkerArray()
            marker_array.markers = markers
            self.marker_pub.publish(marker_array)

        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header = header
        self.annotated_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
