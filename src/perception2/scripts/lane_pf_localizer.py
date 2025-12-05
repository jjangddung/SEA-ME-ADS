#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf_transformations


# ---------------------- 유틸 ------------------------
def normalize_angle(a: float) -> float:
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def quat_to_yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# =====================================================
#       Particle Filter Localizer (Map-matching)
# =====================================================
class LanePFLocalizer(Node):
    def __init__(self):
        super().__init__("lane_pf_localizer_map")

        # ---------------- parameters -----------------
        self.declare_parameter("num_particles", 300)
        # self.declare_parameter("odom_topic", "/automobile/wheel_imu_odom")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("lane_center_topic", "/camera/depth/lane_points_base_sdf")
        self.declare_parameter("pf_pose_topic", "/lane_pf_pose")

        # motion noise
        self.declare_parameter("motion_xy_noise", 0.01)
        self.declare_parameter("motion_yaw_noise_deg", 1.0)

        # lane pointcloud sampling
        self.declare_parameter("max_lane_points", 60)

        # occupancy grid scoring threshold
        self.declare_parameter("occ_high", 70)
        self.declare_parameter("occ_mid", 30)

        # ---------------- load parameters -------------
        self.N = int(self.get_parameter("num_particles").value)
        self.odom_topic = self.get_parameter("odom_topic").value
        self.lane_topic = self.get_parameter("lane_center_topic").value
        self.pf_pose_topic = self.get_parameter("pf_pose_topic").value

        self.motion_xy_noise = float(self.get_parameter("motion_xy_noise").value)
        self.motion_yaw_noise = math.radians(
            float(self.get_parameter("motion_yaw_noise_deg").value)
        )
        self.max_lane_points = int(self.get_parameter("max_lane_points").value)
        self.occ_high = int(self.get_parameter("occ_high").value)
        self.occ_mid = int(self.get_parameter("occ_mid").value)

        # ---------------- internal states -------------
        self.has_map = False
        self.map_res = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None
        self.map_height = None
        self.map_data = None

        self.particles = None
        self.weights = None
        self.initialized = False
        self.last_odom = None

        # ---------------- subscribers -----------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 50
        )
        self.lane_sub = self.create_subscription(
            PointCloud2, self.lane_topic, self.lane_callback, 10
        )

        # ---------------- publishers ------------------
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.pf_pose_topic, 10
        )
        self.particles_pub = self.create_publisher(
            PoseArray, "/lane_pf_particles", 10
        )

        self.get_logger().info(
            f"PF Localizer started: N={self.N}, using OccupancyGrid /map"
        )

    # =====================================================
    #                      MAP CALLBACK
    # =====================================================
    def map_callback(self, msg: OccupancyGrid):
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        arr = np.array(msg.data, dtype=np.int16)
        self.map_data = arr.reshape((self.map_height, self.map_width))
        self.has_map = True

        self.get_logger().info(
            f"[map] loaded occupancy grid: {self.map_width}x{self.map_height}, res={self.map_res}"
        )

    # map frame → grid index
    def world_to_map_index(self, x, y):
        if not self.has_map:
            return None

        u = int((x - self.map_origin_x) / self.map_res)
        v = int((y - self.map_origin_y) / self.map_res)

        if u < 0 or v < 0 or u >= self.map_width or v >= self.map_height:
            return None
        return u, v

    # =====================================================
    #                      ODOM CALLBACK
    # =====================================================
    def odom_callback(self, msg: Odometry):
        if not self.initialized:
            self.init_particles_from_odom(msg)
            self.last_odom = msg
            return

        if self.last_odom is None:
            self.last_odom = msg
            return

        # compute delta
        x_prev = self.last_odom.pose.pose.position.x
        y_prev = self.last_odom.pose.pose.position.y
        yaw_prev = quat_to_yaw(self.last_odom.pose.pose.orientation)

        x_now = msg.pose.pose.position.x
        y_now = msg.pose.pose.position.y
        yaw_now = quat_to_yaw(msg.pose.pose.orientation)

        dx = x_now - x_prev
        dy = y_now - y_prev
        dyaw = normalize_angle(yaw_now - yaw_prev)

        self.last_odom = msg

        self.predict(dx, dy, dyaw)

    # init particles around odom
    def init_particles_from_odom(self, msg: Odometry):
        x0 = msg.pose.pose.position.x
        y0 = msg.pose.pose.position.y
        yaw0 = quat_to_yaw(msg.pose.pose.orientation)

        sigma_xy = 0.05  # 5cm
        sigma_yaw = math.radians(5.0)

        xs = np.random.normal(x0, sigma_xy, self.N)
        ys = np.random.normal(y0, sigma_xy, self.N)
        yaws = np.random.normal(yaw0, sigma_yaw, self.N)

        self.particles = np.stack([xs, ys, yaws], axis=1)
        self.weights = np.ones(self.N) / float(self.N)
        self.initialized = True

        self.get_logger().info(
            f"[init] init around ({x0:.2f},{y0:.2f},{math.degrees(yaw0):.1f} deg)"
        )

    # prediction step
    def predict(self, dx, dy, dyaw):
        if self.particles is None:
            return

        noise_xy = self.motion_xy_noise
        noise_yaw = self.motion_yaw_noise

        self.particles[:, 0] += dx + np.random.normal(0.0, noise_xy, self.N)
        self.particles[:, 1] += dy + np.random.normal(0.0, noise_xy, self.N)
        self.particles[:, 2] += dyaw + np.random.normal(0.0, noise_yaw, self.N)

        self.particles[:, 2] = np.array(
            [normalize_angle(a) for a in self.particles[:, 2]]
        )

    # =====================================================
    #             LANE POINTCLOUD (measurement)
    # =====================================================
    def lane_callback(self, msg: PointCloud2):
        if not (self.initialized and self.has_map):
            return

        # pointcloud → numpy
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((p[0], p[1], p[2]))

        if not pts:
            return

        arr = np.array(pts, dtype=float)
        xs_b = arr[:, 0]
        ys_b = arr[:, 1]
        M = len(xs_b)

        # downsampling
        if M > self.max_lane_points:
            idx = np.linspace(0, M - 1, self.max_lane_points).astype(int)
            xs_b = xs_b[idx]
            ys_b = ys_b[idx]
            M = self.max_lane_points

        new_weights = np.zeros(self.N)

        # for each particle: evaluate likelihood
        for i in range(self.N):
            X_i, Y_i, yaw_i = self.particles[i]
            cos_y = math.cos(yaw_i)x
            sin_y = math.sin(yaw_i)

            # transform lane points → map frame
            Xs_m = X_i + cos_y * xs_b - sin_y * ys_b
            Ys_m = Y_i + sin_y * xs_b + cos_y * ys_b

            score_sum = 0.0
            valid = 0

            for j in range(M):
                idx = self.world_to_map_index(float(Xs_m[j]), float(Ys_m[j]))
                if idx is None:
                    continue
                u, v = idx
                occ = int(self.map_data[v, u])  # -1, 0..100

                # scoring based on occupancy
                if occ == -1:
                    s = 0.3
                elif occ >= self.occ_high:
                    s = 1.0
                elif occ >= self.occ_mid:
                    s = 0.5
                else:
                    s = 0.1

                score_sum += s
                valid += 1

            if valid == 0:
                new_weights[i] = self.weights[i]
            else:
                S_i = score_sum / float(valid)
                new_weights[i] = self.weights[i] * S_i

        # normalize
        sw = float(np.sum(new_weights))
        if sw < 1e-12:
            self.get_logger().warn("all weights ~0, reset")
            new_weights[:] = 1.0 / float(self.N)
        else:
            new_weights /= sw

        self.weights = new_weights

        # resample
        self.resample_if_needed()

        # publish final pose
        self.publish_estimated_pose(msg.header.stamp)

    # resample trigger
    def resample_if_needed(self):
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < self.N * 0.5:
            self.systematic_resample()

    # systematic resample
    def systematic_resample(self):
        N = self.N
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, dtype=int)

        cumsum = np.cumsum(self.weights)
        i = 0
        j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indexes]
        self.weights = np.ones(N) / float(N)

    # =====================================================
    #                 Publish Final Pose
    # =====================================================
    def publish_estimated_pose(self, stamp):
        xs = self.particles[:, 0]
        ys = self.particles[:, 1]
        yaws = self.particles[:, 2]

        x_mean = np.average(xs, weights=self.weights)
        y_mean = np.average(ys, weights=self.weights)

        c = np.average(np.cos(yaws), weights=self.weights)
        s = np.average(np.sin(yaws), weights=self.weights)
        yaw_mean = math.atan2(s, c)

        msg = PoseWithCovarianceStamped()
        msg.header = Header()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp

        msg.pose.pose.position.x = float(x_mean)
        msg.pose.pose.position.y = float(y_mean)
        msg.pose.pose.position.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, yaw_mean)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # simple covariance (optional)
        cov = np.zeros((6, 6))
        cov[0, 0] = 0.05
        cov[1, 1] = 0.05
        cov[5, 5] = math.radians(5.0) ** 2
        msg.pose.covariance = cov.flatten().tolist()

        self.pose_pub.publish(msg)

        # publish particles (debug)
        pa = PoseArray()
        pa.header = msg.header

        for i in range(self.N):
            X_i, Y_i, yaw_i = self.particles[i]
            pi = Pose()
            pi.position.x = float(X_i)
            pi.position.y = float(Y_i)
            q_i = tf_transformations.quaternion_from_euler(0, 0, yaw_i)
            pi.orientation.x = q_i[0]
            pi.orientation.y = q_i[1]
            pi.orientation.z = q_i[2]
            pi.orientation.w = q_i[3]
            pa.poses.append(pi)

        self.particles_pub.publish(pa)


# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = LanePFLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
