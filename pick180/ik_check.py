#!/usr/bin/env python3
"""MoveIt の /compute_ik で、ブロックを掴む各姿勢の関節角度を求める(確認用)"""
import math, sys
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

ARM = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']

def ik(node, cli, x, y, z, roll, pitch, yaw, frame='crane_plus_base'):
    req = GetPositionIK.Request()
    req.ik_request.group_name = 'arm'
    req.ik_request.ik_link_name = 'crane_plus_link_endtip'
    req.ik_request.avoid_collisions = False
    req.ik_request.timeout.sec = 1
    ps = PoseStamped(); ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
    q = quaternion_from_euler(roll, pitch, yaw)
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
    req.ik_request.pose_stamped = ps
    fut = cli.call_async(req); rclpy.spin_until_future_complete(node, fut, timeout_sec=5)
    r = fut.result()
    if r is None or r.error_code.val != 1:
        return None, (r.error_code.val if r else 'timeout')
    js = dict(zip(r.solution.joint_state.name, r.solution.joint_state.position))
    return [round(js[j], 4) for j in ARM], 1

def main():
    rclpy.init(); node = Node('ik_check')
    cli = node.create_client(GetPositionIK, '/compute_ik'); cli.wait_for_service(5)
    d = math.radians
    for name, (x, y, z, r, p, yw) in {
        'above_pick': (0.0, -0.095, 0.17, 0, d(180), d(-90)),
        'grasp':      (0.0, -0.095, 0.14, 0, d(180), d(-90)),
        'grasp_low':  (0.0, -0.095, 0.13, 0, d(180), d(-90)),
        'above_place':(0.0,  0.095, 0.17, 0, d(180), d(90)),
        'place':      (0.0,  0.095, 0.14, 0, d(180), d(90)),
    }.items():
        sol, code = ik(node, cli, x, y, z, r, p, yw)
        print(name, sol, code)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
