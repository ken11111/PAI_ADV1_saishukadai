#!/usr/bin/env python3
"""
[roundtrip 用] Gazebo 上のブロックの位置を、ROS のトピック /cube_pose(geometry_msgs/PointStamped、world 座標)として 20Hz で流す。
カメラの代わりに「ブロックの位置が分かるセンサー」として使い、rosbag に記録して学習と推論の入力(observation.environment_state)にする。

使い方(コンテナ内、シミュレーション起動中に): python3 /workspace/pick180/cube_pose_pub.py
"""
import subprocess, threading, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

CUBE = 'wood_cube_5cm'


class CubePosePub(Node):
    def __init__(self):
        super().__init__('cube_pose_pub')
        self.pub = self.create_publisher(PointStamped, '/cube_pose', 10)
        self.latest = None
        threading.Thread(target=self.read_gazebo, daemon=True).start()
        self.create_timer(0.05, self.publish)

    def read_gazebo(self):
        # pose/info を流し続け、ブロックの position だけを取り出す(値が 0 の項目は出力に現れない)
        p = subprocess.Popen(['ign', 'topic', '-e', '-t', '/world/default/pose/info'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        in_cube = in_pos = False
        xyz = {}
        for line in p.stdout:
            s = line.strip()
            if s.startswith('name:'):
                in_cube = s == f'name: "{CUBE}"'
            elif in_cube and s == 'position {':
                in_pos, xyz = True, {}
            elif in_pos and s == '}':
                self.latest = (xyz.get('x', 0.0), xyz.get('y', 0.0), xyz.get('z', 0.0))
                in_pos = in_cube = False
            elif in_pos and ':' in s:
                k, v = s.split(':')
                xyz[k.strip()] = float(v)

    def publish(self):
        if self.latest is None:
            return
        m = PointStamped(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'world'
        m.point.x, m.point.y, m.point.z = self.latest
        self.pub.publish(m)


def main():
    rclpy.init(); node = CubePosePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
