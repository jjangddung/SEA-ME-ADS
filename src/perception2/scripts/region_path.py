#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import xml.etree.ElementTree as ET
import numpy as np
import math

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32


class GraphPathPublisher(Node):
    def __init__(self):
        super().__init__('graph_path_publisher')

        # === region table ===
        self.parking_nodes = self._make_range_set([(225, 240)])
        self.ramp_nodes = self._make_range_set([(225, 238), (407, 423)])
        self.highway_nodes = self._make_range_set([
            (444, 461),  # 위 오른 차선
            (483, 501),  # 위 왼쪽 차선
            (463, 482),  # 아래 오른 차선
            (502, 520),  # 아래 왼쪽 차선
        ])

        # path 저장
        self.path_waypoints = []
        self.dense_region_ids = []

        # pose 기반 region publish
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/lane_pf_pose',
            self.pose_callback,
            10
        )
        self.region_pub = self.create_publisher(Int32, '/current_region', 10)
        self.last_idx = None

        # publishers
        self.path_pub = self.create_publisher(Path, '/track_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/track_markers', 10)

        # GraphML 읽기
        graphml_file = '/home/dongmin/bfmc/Simulator/src/sim_pkg/path/gercek2.graphml'
        self.get_logger().info(f"Loading GraphML: {graphml_file}")
        self.nodes_data, self.edges_data = self.extract_nodes_data(graphml_file)

        # 목적지 설정(고정)
        start_id = "225"
        goal_id = "520"

        path_node_ids = self.find_path(start_id, goal_id)
        coords = [(nid, *self.nodes_data[nid]) for nid in path_node_ids]

        dense_x, dense_y, dense_region_ids = self.build_dense_path(coords)
        self.dense_region_ids = dense_region_ids

        # Path 메시지 생성
        self.publish_path(dense_x, dense_y)
        self.publish_markers(coords)

        self.get_logger().info("GraphPathPublisher ready.")

    # =====================================================================
    # region helper
    # =====================================================================
    def _make_range_set(self, ranges):
        """[(a,b), (c,d)] → {a..b, c..d} 문자열 형태"""
        s = set()
        for a, b in ranges:
            for nid in range(a, b + 1):
                s.add(str(nid))
        return s

    # =====================================================================
    # GraphML parser
    # =====================================================================
    def extract_nodes_data(self, graphml_file):
        tree = ET.parse(graphml_file)
        root = tree.getroot()

        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        nodes = {}
        edges = []

        for n in root.findall(".//g:node", ns):
            nid = n.attrib['id']
            x = float(n.find("g:data[@key='x']", ns).text)
            y = float(n.find("g:data[@key='y']", ns).text)
            nodes[nid] = (x, y)

        for e in root.findall(".//g:edge", ns):
            u = e.attrib['source']
            v = e.attrib['target']
            edges.append((u, v))

        return nodes, edges

    # =====================================================================
    # Dijkstra (단방향)
    # =====================================================================
    def find_path(self, start_id, goal_id):

        graph = {}
        for nid in self.nodes_data.keys():
            graph[nid] = []

        for u, v in self.edges_data:
            graph[u].append(v)

        dist = {nid: float("inf") for nid in graph.keys()}
        prev = {nid: None for nid in graph.keys()}

        dist[start_id] = 0
        unvisited = set(graph.keys())

        while unvisited:
            current = min(unvisited, key=lambda x: dist[x])
            unvisited.remove(current)
            if current == goal_id:
                break

            for nxt in graph[current]:
                d = dist[current] + self._dist_nodes(current, nxt)
                if d < dist[nxt]:
                    dist[nxt] = d
                    prev[nxt] = current

        # backtrack
        path = []
        cur = goal_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def _dist_nodes(self, n1, n2):
        x1, y1 = self.nodes_data[n1]
        x2, y2 = self.nodes_data[n2]
        return math.hypot(x2 - x1, y2 - y1)

    # =====================================================================
    # Dense path 생성
    # =====================================================================
    def build_dense_path(self, coords):
        dense_x = []
        dense_y = []
        dense_region_ids = []

        node_ids = [nid for (nid, _, _) in coords]
        xs = [x for (_, x, _) in coords]
        ys = [y for (_, _, y) in coords]

        n = len(coords)
        for i in range(n - 1):
            nid1 = node_ids[i]
            nid2 = node_ids[i + 1]

            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i + 1], ys[i + 1]

            seg_len = math.hypot(x2 - x1, y2 - y1)
            n_steps = max(1, int(seg_len / 0.01))

            # region tagging
            if (nid1 in self.highway_nodes) or (nid2 in self.highway_nodes):
                region = 3
            elif (nid1 in self.ramp_nodes) or (nid2 in self.ramp_nodes):
                region = 2
            elif (nid1 in self.parking_nodes) or (nid2 in self.parking_nodes):
                region = 1
            else:
                region = 0

            for k in range(n_steps + 1):
                t = k / n_steps
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t

                if i > 0 and k == 0:
                    continue

                dense_x.append(x)
                dense_y.append(y)
                dense_region_ids.append(region)

        # 마지막 점
        last_nid = node_ids[-1]
        lx, ly = xs[-1], ys[-1]

        if last_nid in self.highway_nodes:
            region = 3
        elif last_nid in self.ramp_nodes:
            region = 2
        elif last_nid in self.parking_nodes:
            region = 1
        else:
            region = 0

        dense_x.append(lx)
        dense_y.append(ly)
        dense_region_ids.append(region)

        return dense_x, dense_y, dense_region_ids

    # =====================================================================
    # Path publish
    # =====================================================================
    def publish_path(self, xs, ys):
        msg = Path()
        msg.header.frame_id = "map"

        for i, (x, y) in enumerate(zip(xs, ys)):
            p = PoseStamped()
            p.header.frame_id = "map"
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)

        self.path_waypoints = [(float(x), float(y), 0.0) for x, y in zip(xs, ys)]
        self.path_pub.publish(msg)

    def publish_markers(self, coords):
        arr = MarkerArray()
        for i, (nid, x, y) in enumerate(coords):
            m = Marker()
            m.header.frame_id = "map"
            m.type = Marker.TEXT_VIEW_FACING
            m.scale.z = 0.25
            m.color.r = 1.0
            m.color.a = 1.0
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.1
            m.text = nid
            m.id = i
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # =====================================================================
    # pose → region 계산
    # =====================================================================
    def pose_callback(self, msg):
        if not self.path_waypoints or not self.dense_region_ids:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # first time → full search
        if self.last_idx is None:
            search_range = range(len(self.path_waypoints))
        else:
            w = 80
            start = max(0, self.last_idx - w)
            end = min(len(self.path_waypoints), self.last_idx + w)
            search_range = range(start, end)

        min_d2 = float("inf")
        best_i = 0
        for i in search_range:
            px, py, _ = self.path_waypoints[i]
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
                best_i = i

        self.last_idx = best_i

        region = self.dense_region_ids[best_i]
        out = Int32()
        out.data = region
        self.region_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GraphPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
