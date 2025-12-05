#!/usr/bin/env python3
# coding: utf-8

import math
import heapq
import xml.etree.ElementTree as ET
import copy

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GraphPathPublisher(Node):
    def __init__(self):
        super().__init__('graph_path_publisher')

        # ===== 파라미터 =====
        self.declare_parameter(
            'file_path',
            '/home/dongmin/bfmc/Simulator/src/sim_pkg/path/fixed2.graphml'
            # '/home/dongmin/bfmc/Simulator/src/sim_pkg/path/Competition_track_graph.graphml'
        )
        self.declare_parameter('start_node', '263')
        # self.declare_parameter('goal_node', '225')  # 필요에 따라 바꾸기
        self.declare_parameter('goal_node', '442')  # 필요에 따라 바꾸기

        # MPC용 dense path에서 샘플 간 거리 (m)
        # 값이 작을수록 더 촘촘하고 부드러운 path
        self.declare_parameter('mpc_resolution', 0.03)

        self.file_path = self.get_parameter('file_path').value
        self.start_node = self.get_parameter('start_node').value
        self.goal_node = self.get_parameter('goal_node').value
        self.mpc_resolution = float(self.get_parameter('mpc_resolution').value)

        # 전체 Path publisher (MPC에서 보는 path)
        self.path_pub = self.create_publisher(Path, 'track_path', 10)

        # lane을 강하게 쓸 구간만 따로 모은 Path publisher
        self.lane_strong_pub = self.create_publisher(
            Path, 'track_path_lane_strong', 10
        )

        # Path 데이터 저장 (dense 기준)
        self.path_waypoints = []      # [(x, y, yaw), ...]
        self.path_node_ids = []       # 그래프 상 노드 시퀀스
        self.path_edges = []          # [(src, dst)]

        # 🔥 lane을 강하게 쓸 노드 ID 구간 (GraphML node id 기준)
        self.lane_strong_node_ids = set()
        for nid in range(244, 249):   # 244~248
            self.lane_strong_node_ids.add(str(nid))
        for nid in range(271, 289):   # 271~288
            self.lane_strong_node_ids.add(str(nid))

        self.get_logger().info(
            f"Lane-strong node IDs: {sorted(list(self.lane_strong_node_ids))}"
        )

        # Path 생성
        self.path_msg, self.lane_strong_path = self.build_path()

        if self.path_msg is None:
            self.get_logger().error("Path build failed. No publishing will happen.")
        else:
            self.get_logger().info(
                f"Dense path built with {len(self.path_msg.poses)} poses "
                f"from {self.start_node} to {self.goal_node}. "
                "Will publish periodically on 'track_path'."
            )
            if self.lane_strong_path is not None:
                self.get_logger().info(
                    f"Lane-strong path has {len(self.lane_strong_path.poses)} poses."
                )

            # 🔁 0.5초마다 계속 퍼블리시
            self.timer = self.create_timer(0.5, self.timer_callback)

    # ---------------- 타이머: Path 주기적으로 발행 ----------------
    def timer_callback(self):
        if self.path_msg is None:
            return

        now = self.get_clock().now().to_msg()

        # 전체 dense path
        self.path_msg.header.stamp = now
        self.path_pub.publish(self.path_msg)

        # lane-strong 구간 path
        if self.lane_strong_path is not None:
            self.lane_strong_path.header.stamp = now
            self.lane_strong_pub.publish(self.lane_strong_path)

    # ---------------- Path 생성 ----------------
    def build_path(self):
        # GraphML 읽기
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
        except Exception as e:
            self.get_logger().error(f"Failed to parse GraphML file: {e}")
            return None, None

        nodes_data = self.extract_nodes_data(root)          # [(id, x, y), ...]
        edges_all, edges_true = self.extract_edges_data(root)

        # id -> (x, y) 매핑
        node_pos = {nid: (x, y) for nid, x, y in nodes_data}
        if not node_pos:
            self.get_logger().error("No nodes parsed from GraphML.")
            return None, None

        # 🔴 d2는 지금은 무시하고, 그래프 전체 edge 사용
        edges_use = edges_all
        if not edges_use:
            self.get_logger().error("No edges parsed from GraphML.")
            return None, None

        # adjacency list 구성 (단방향, 필요시 양방향으로 변경 가능)
        adj = {}
        for src, tgt, _ in edges_use:
            adj.setdefault(src, []).append(tgt)
            # adj.setdefault(tgt, []).append(src)

        if not adj:
            self.get_logger().error("Adjacency list is empty. Cannot build path.")
            return None, None

        # 시작/목표 노드 존재 여부 체크
        if self.start_node not in adj:
            self.get_logger().error(
                f"Start node {self.start_node} not found in adjacency."
            )
            sample_keys = list(adj.keys())[:10]
            self.get_logger().info(f"Adjacency sample keys: {sample_keys}")
            return None, None

        if self.goal_node not in adj:
            self.get_logger().error(
                f"Goal node {self.goal_node} not found in adjacency."
            )
            sample_keys = list(adj.keys())[:10]
            self.get_logger().info(f"Adjacency sample keys: {sample_keys}")
            return None, None

        self.get_logger().info(
            f"Running shortest path from {self.start_node} to {self.goal_node}."
        )

        # ===== Dijkstra (edge weight=1, 사실상 BFS) =====
        node_path = self.shortest_path(adj, self.start_node, self.goal_node)
        if node_path is None or len(node_path) == 0:
            self.get_logger().error(
                f"No path found from {self.start_node} to {self.goal_node}."
            )
            return None, None

        self.get_logger().info(
            f"Shortest path length (nodes) = {len(node_path)}"
        )

        self.path_node_ids = node_path
        self.path_edges = list(zip(node_path[:-1], node_path[1:]))

        # ===== 원래 node 기반 좌표 배열 만들기 =====
        coords = []   # [(nid, x, y), ...]
        for nid in node_path:
            if nid not in node_pos:
                self.get_logger().warn(f"Node id {nid} has no position. Skipping.")
                continue
            coords.append((nid, *node_pos[nid]))

        if not coords:
            self.get_logger().error("No valid nodes with positions found on path.")
            return None, None

        # ===== MPC용 dense path (Cubic Hermite spline 기반) 생성 =====
        dense_x, dense_y, dense_strong_flags = self.build_dense_path_from_coords(coords)

        if not dense_x:
            self.get_logger().error("Dense path is empty after resampling.")
            return None, None

        # yaw 계산 (dense path 기준)
        yaws = []
        for i in range(len(dense_x) - 1):
            dx = dense_x[i + 1] - dense_x[i]
            dy = dense_y[i + 1] - dense_y[i]
            yaws.append(math.atan2(dy, dx))
        if yaws:
            yaws.append(yaws[-1])
        else:
            yaws.append(0.0)

        # ===== nav_msgs/Path 생성 =====
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        lane_strong_path = Path()
        lane_strong_path.header.frame_id = 'map'
        lane_strong_path.header.stamp = path_msg.header.stamp

        self.path_waypoints = []

        for idx, (x, y, yaw, is_strong) in enumerate(
            zip(dense_x, dense_y, yaws, dense_strong_flags)
        ):
            self.path_waypoints.append((x, y, yaw))

            pose = PoseStamped()
            pose.header.frame_id = path_msg.header.frame_id
            pose.header.stamp.sec = 0
            pose.header.stamp.nanosec = idx  # dense path index

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(pose)

            # lane-strong 구간이면 별도 Path에도 추가
            if is_strong:
                lane_strong_path.poses.append(copy.deepcopy(pose))

        if not path_msg.poses:
            self.get_logger().error("Path has no poses after building. Check GraphML.")
            return None, None

        self.get_logger().info(
            f"Built dense waypoint list with {len(self.path_waypoints)} entries. "
            f"First waypoint: {self.path_waypoints[0]} | "
            f"Last waypoint: {self.path_waypoints[-1]}"
        )
        self.get_logger().info(f"path edges: {self.path_edges[:10]} ...")
        self.get_logger().info(
            f"Lane-strong dense pose count: {len(lane_strong_path.poses)}"
        )

        # 디버그용: lane-strong path 안의 일부 index 출력
        sample_indices = [
            p.header.stamp.nanosec for p in lane_strong_path.poses[:10]
        ]
        self.get_logger().info(
            f"Lane-strong sample indices (dense idx): {sample_indices}"
        )

        return path_msg, lane_strong_path

    # ---------------- dense path 생성 (Cubic Hermite spline) ----------------
    def build_dense_path_from_coords(self, coords):
        """
        coords: [(nid, x, y), ...]
        mpc_resolution 간격으로 Cubic Hermite spline 기반 dense path 생성.
        각 점이 lane-strong 구간인지 여부도 함께 반환.
        """
        ds = self.mpc_resolution

        n = len(coords)
        if n == 0:
            return [], [], []

        node_ids = [c[0] for c in coords]
        xs = [c[1] for c in coords]
        ys = [c[2] for c in coords]

        # ---- 각 노드에서의 접선 (tx, ty) 계산 (Catmull-Rom 스타일) ----
        tx = []
        ty = []
        for i in range(n):
            if i == 0:
                dx = xs[1] - xs[0]
                dy = ys[1] - ys[0]
            elif i == n - 1:
                dx = xs[-1] - xs[-2]
                dy = ys[-1] - ys[-2]
            else:
                dx = 0.5 * (xs[i + 1] - xs[i - 1])
                dy = 0.5 * (ys[i + 1] - ys[i - 1])
            tx.append(dx)
            ty.append(dy)

        dense_x = []
        dense_y = []
        dense_strong_flags = []

        # 노드가 하나뿐이면 그대로 반환
        if n == 1:
            dense_x.append(xs[0])
            dense_y.append(ys[0])
            dense_strong_flags.append(node_ids[0] in self.lane_strong_node_ids)
            return dense_x, dense_y, dense_strong_flags

        # ---- 각 세그먼트마다 Cubic Hermite spline으로 샘플링 ----
        for i in range(n - 1):
            nid1 = node_ids[i]
            nid2 = node_ids[i + 1]

            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            mx0, my0 = tx[i], ty[i]
            mx1, my1 = tx[i + 1], ty[i + 1]

            dx_seg = x1 - x0
            dy_seg = y1 - y0
            seg_len = math.hypot(dx_seg, dy_seg)
            if seg_len < 1e-6:
                continue

            # 이 세그먼트가 lane-strong 영역인지:
            seg_strong = (
                (nid1 in self.lane_strong_node_ids)
                or (nid2 in self.lane_strong_node_ids)
            )

            # 세그먼트를 ds 간격으로 나눔
            n_steps = max(1, int(seg_len / ds))

            for k in range(n_steps + 1):
                t = k / n_steps  # 0.0 ~ 1.0

                # Cubic Hermite basis functions
                t2 = t * t
                t3 = t2 * t
                h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                h10 = t3 - 2.0 * t2 + t
                h01 = -2.0 * t3 + 3.0 * t2
                h11 = t3 - t2

                x = h00 * x0 + h10 * mx0 + h01 * x1 + h11 * mx1
                y = h00 * y0 + h10 * my0 + h01 * y1 + h11 * my1

                # 세그먼트 경계 중복 방지: 첫 세그먼트 제외,
                # 이후 세그먼트의 k == 0 은 이전 세그먼트 마지막 점과 동일하므로 스킵
                if i > 0 and k == 0:
                    continue

                dense_x.append(x)
                dense_y.append(y)
                dense_strong_flags.append(seg_strong)

        # 마지막 노드가 혹시 안 들어갔으면 추가(안전용)
        last_nid = node_ids[-1]
        lx = xs[-1]
        ly = ys[-1]
        if (not dense_x) or (abs(dense_x[-1] - lx) > 1e-6 or abs(dense_y[-1] - ly) > 1e-6):
            dense_x.append(lx)
            dense_y.append(ly)
            dense_strong_flags.append(last_nid in self.lane_strong_node_ids)

        return dense_x, dense_y, dense_strong_flags

    # ---------------- 노드 파싱 ----------------
    def extract_nodes_data(self, root):
        nodes_data = []
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        for node in root.findall(".//g:node", ns):
            node_id = node.get('id')
            d0 = None
            d1 = None
            for data in node:
                key = data.get('key')
                if key == 'd0':
                    d0 = float(data.text)
                elif key == 'd1':
                    d1 = float(data.text)
            if d0 is not None and d1 is not None:
                nodes_data.append((node_id, d0, d1))
        return nodes_data

    # ---------------- 엣지 파싱 ----------------
    def extract_edges_data(self, root):
        """
        GraphML에서 edge 전체를 읽고:
        - d2 값은 일단 읽어두기만 하고, path 생성에는 사용하지 않는다.
        """
        edges_data = []
        edges_data_true = []
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        for edge in root.findall(".//g:edge", ns):
            source_id = edge.get('source')
            target_id = edge.get('target')

            data_d2 = edge.find("g:data[@key='d2']", ns)
            d2_value = (data_d2 is not None and data_d2.text == 'True')

            if d2_value:
                edges_data_true.append((source_id, target_id, d2_value))

            edges_data.append((source_id, target_id, d2_value))

        return edges_data, edges_data_true

    # ---------------- Dijkstra: 최단 노드 시퀀스 ----------------
    def shortest_path(self, adj, start, goal):
        """
        adj: {node_id: [neighbor_id, ...]}
        start, goal: node_id (string)
        return: [start, ..., goal] 혹은 None
        """
        dist = {nid: float('inf') for nid in adj.keys()}
        prev = {nid: None for nid in adj.keys()}
        dist[start] = 0.0

        pq = [(0.0, start)]  # (cost, node_id)

        while pq:
            cur_d, u = heapq.heappop(pq)
            if cur_d > dist[u]:
                continue
            if u == goal:
                break

            for v in adj.get(u, []):
                nd = cur_d + 1.0   # 모든 edge 가중치 1
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist.get(goal, float('inf')) == float('inf'):
            return None

        path_ids = []
        cur = goal
        while cur is not None:
            path_ids.append(cur)
            cur = prev[cur]
        path_ids.reverse()
        return path_ids


def main(args=None):
    rclpy.init(args=args)
    node = GraphPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
