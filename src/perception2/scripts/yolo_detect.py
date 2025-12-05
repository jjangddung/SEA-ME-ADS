#!/usr/bin/env python3
# coding: utf-8

import os
import math
import threading
from typing import List, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


#############################################
# 클래스 이름
#############################################
CUSTOM_CLASS_NAMES = [
    "ONEWAY","HIGHWAYENTRANCE","STOPSIGN","ROUNDABOUT","PARK",
    "CROSSWALK","NOENTRY","HIGHWAYEXIT","PRIORITY","LIGHTS",
    "BLOCK","PEDESTRIAN","CAR"
]

def get_class_name(cls_id: int) -> str:
    if 0 <= cls_id < len(CUSTOM_CLASS_NAMES):
        return CUSTOM_CLASS_NAMES[cls_id]
    return f"class_{cls_id}"


#############################################
# TensorRT Wrapper
#############################################
class YoloTRT:
    def __init__(self,
                 engine_path: str,
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
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # tensors
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT and self.output_name is None:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Failed to find IO tensors in engine")

        input_shape = list(self.engine.get_tensor_shape(self.input_name))
        self.context.set_input_shape(self.input_name, input_shape)
        self.input_dims = input_shape
        self.batch_size, self.input_channels, self.input_h, self.input_w = input_shape

        output_shape = list(self.engine.get_tensor_shape(self.output_name))
        self.output_dims = output_shape

        def vol(dims):
            v = 1
            for d in dims:
                v *= int(d)
            return v

        self.input_size = vol(self.input_dims)
        self.output_size = vol(self.output_dims)

        # GPU buffers
        self.d_input = cuda.mem_alloc(self.input_size * 4)
        self.d_output = cuda.mem_alloc(self.output_size * 4)
        self.stream = cuda.Stream()

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    ##################################
    def preprocess(self, frame_bgr: np.ndarray):
        orig_h, orig_w = frame_bgr.shape[:2]
        target_w, target_h = self.input_w, self.input_h

        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        letterboxed[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return chw.ravel(), ratio, pad_w, pad_h

    ##################################
    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0 and len(keep) < self.top_k:
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

    ##################################
    def postprocess(self, output: np.ndarray,
                    orig_size: Tuple[int, int],
                    ratio: float, pad_w: int, pad_h: int):
        orig_w, orig_h = orig_size
        out = output.flatten()
        dims = list(self.output_dims)
        if dims[0] == 1:
            dims = dims[1:]

        d0, d1 = dims
        features = d0 if d0 < d1 else d1
        boxes = d1 if d0 < d1 else d0
        if features < 5:
            return []

        stride = boxes if d0 < d1 else features

        def access(c, b):
            if d0 < d1:
                return out[c * boxes + b]
            else:
                return out[b * stride + c]

        boxes_list, scores_list, classes_list = [], [], []

        for b in range(boxes):
            cx = access(0, b)
            cy = access(1, b)
            w  = access(2, b)
            h  = access(3, b)

            # find best class
            best_score = -1.0
            best_cls = -1
            for cls in range(features - 4):
                s = access(4 + cls, b)
                if s > best_score:
                    best_score = s
                    best_cls = cls
            if best_score < self.score_thresh:
                continue

            # remove padding + scale back
            x1 = (cx - w/2 - pad_w) / ratio
            y1 = (cy - h/2 - pad_h) / ratio
            x2 = (cx + w/2 - pad_w) / ratio
            y2 = (cy + h/2 - pad_h) / ratio

            x1 = float(np.clip(x1, 0, orig_w - 1))
            y1 = float(np.clip(y1, 0, orig_h - 1))
            x2 = float(np.clip(x2, 0, orig_w - 1))
            y2 = float(np.clip(y2, 0, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(float(best_score))
            classes_list.append(best_cls)

        if not boxes_list:
            return []

        keep = self.nms(boxes_list, scores_list)
        results = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_list[idx]
            results.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "score": scores_list[idx],
                "cls": classes_list[idx],
            })
        return results

    ##################################
    def infer(self, frame_bgr: np.ndarray):
        flat, ratio, pad_w, pad_h = self.preprocess(frame_bgr)

        cuda.memcpy_htod_async(self.d_input, flat, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(self.stream.handle)

        output = np.empty(self.output_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        h, w = frame_bgr.shape[:2]
        return self.postprocess(output, (w, h), ratio, pad_w, pad_h)


#############################################
# YOLO Node
#############################################
class Yolo2DNode(Node):
    def __init__(self):
        super().__init__("yolo_2d_node")

        # parameters
        self.declare_parameter(
            "engine_path",
            "/home/dongmin/bfmc/perception_ws/perception/model/model.engine"
        )
        self.declare_parameter("score_thresh", 0.6)
        self.declare_parameter("nms_thresh", 0.6)
        self.declare_parameter("top_k", 100)
        self.declare_parameter("inference_period", 0.1)
        self.declare_parameter("debug_view", False)

        engine_path = self.get_parameter("engine_path").value
        score_thresh = float(self.get_parameter("score_thresh").value)
        nms_thresh   = float(self.get_parameter("nms_thresh").value)
        top_k        = int(self.get_parameter("top_k").value)
        self.period  = float(self.get_parameter("inference_period").value)
        self.debug   = bool(self.get_parameter("debug_view").value)

        self.get_logger().info(f"[YOLO] engine={engine_path}")

        self.yolo = YoloTRT(
            engine_path=engine_path,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            top_k=top_k
        )

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_header = None
        self.lock = threading.Lock()

        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # subscribe camera
        self.sub_rgb = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_callback,
            qos_sensor
        )

        # output bbox
        self.pub_det = self.create_publisher(
            Detection2DArray,
            "/yolo/detections_2d",
            10
        )

        # timer
        self.timer = self.create_timer(self.period, self.timer_callback)

        self.get_logger().info("[YOLO] node started.")

    ##################################
    def rgb_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return
        with self.lock:
            self.latest_rgb = cv_bgr
            self.latest_header = msg.header

    ##################################
    def timer_callback(self):
        with self.lock:
            if self.latest_rgb is None:
                return
            frame = self.latest_rgb.copy()
            header = self.latest_header

        dets = self.yolo.infer(frame)

        ### ==========================
        ### Detection2DArray publish
        ### ==========================
        det_array = Detection2DArray()
        det_array.header = header

        debug_img = frame.copy()

        for det in dets:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            cls_id = det["cls"]
            cls_name = get_class_name(cls_id)

            # pack message
            d = Detection2D()
            d.header = header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name          # 문자열 저장
            hyp.hypothesis.score = float(score)         # confidence
            d.results.append(hyp)


            bbox = BoundingBox2D()
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w  = (x2 - x1)
            h  = (y2 - y1)
            bbox.center.position.x = float(cx)
            bbox.center.position.y = float(cy)
            bbox.size_x = float(w)
            bbox.size_y = float(h)
            d.bbox = bbox

            det_array.detections.append(d)

            ### ==========================
            ### Debug Draw
            ### ==========================
            if self.debug:
                cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(debug_img, f"{cls_name} {score:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

        self.pub_det.publish(det_array)

        ### ==========================
        ### cv2.imshow debug window
        ### ==========================
        if self.debug:
            cv2.imshow("YOLO Debug", debug_img)
            cv2.waitKey(1)

    ##################################
    def destroy_node(self):
        if self.debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Yolo2DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
