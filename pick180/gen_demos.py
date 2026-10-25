#!/usr/bin/env python3
"""
ブロックを掴んで土台を180度回し、反対側に置く「見本の動き」を自動で作り、rosbag に記録する。

- 指令は teleop_keyboard / infer.py と同じ形式(20Hz、time_from_start 0.1秒)で送る
- 関節の並び順も同じ(joint1, joint2, joint4, joint3)
- 毎回ブロックを元の位置に戻してから記録し、終了後にブロックが 180度先に運ばれたかを判定する
- 失敗した回の記録は削除する

使い方(コンテナ内、シミュレーション起動中に):
  python3 /workspace/pick180/gen_demos.py --episodes 15 --out ~/dataset/rosbags/pick180
"""
import argparse, math, os, random, re, shutil, signal, subprocess, time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_PUB_ORDER = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER = 'crane_plus_joint_hand'
GRIPPER_OPEN = -0.5236
GRIPPER_CLOSE = 0.1745
RATE = 20.0

# MoveIt の /compute_ik で求めた姿勢(joint1, joint2, joint3, joint4 の順)
# 手先(endtip=手首)を真下向き(RPY 0,180,±90)にし、真上 z=0.17、掴む z=0.13、置く z=0.14(crane_plus_examples の pick_and_place.cpp を参考)
START = [0.0, 0.0, 0.0, 0.0]
ABOVE_PICK = [-1.5708, 0.0340, -1.3687, -1.7388]
GRASP = [-1.5708, 0.0567, -1.7766, -1.3082]
ABOVE_PLACE = [1.5708, 0.0340, -1.3687, -1.7391]
PLACE = [1.5708, 0.0314, -1.6935, -1.4166]

CUBE = 'wood_cube_5cm'
CUBE_HOME = (0.0, -0.1, 1.015)
TOPICS = ['/joint_states', '/crane_plus_arm_controller/joint_trajectory',
          '/crane_plus_gripper_controller/joint_trajectory']


def ign(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout


def reset_cube():
    x, y, z = CUBE_HOME
    ign("ign service -s /world/default/set_pose --reqtype ignition.msgs.Pose "
        "--reptype ignition.msgs.Boolean --timeout 3000 "
        f"--req 'name: \"{CUBE}\", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{w: 1}}'")


def cube_pose():
    out = ign("timeout 5 ign topic -e -n 1 -t /world/default/pose/info")
    m = re.search(r'name: "%s".*?position \{(.*?)\}' % CUBE, out, re.S)
    if not m:
        return None
    v = dict(re.findall(r'([xyz]): (-?[0-9.e-]+)', m.group(1)))
    return tuple(float(v.get(k, 0.0)) for k in 'xyz')


class Demo(Node):
    def __init__(self):
        super().__init__('pick180_demo')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)

    def send(self, arm, grip):
        j1, j2, j3, j4 = arm
        t = JointTrajectory(); t.joint_names = ARM_PUB_ORDER
        p = JointTrajectoryPoint(); p.positions = [j1, j2, j4, j3]
        p.time_from_start.sec = 0; p.time_from_start.nanosec = int(0.1e9)
        t.points.append(p); self.arm_pub.publish(t)
        g = JointTrajectory(); g.joint_names = [GRIPPER]
        q = JointTrajectoryPoint(); q.positions = [grip]
        q.time_from_start.sec = 0; q.time_from_start.nanosec = int(0.1e9)
        g.points.append(q); self.grip_pub.publish(g)

    def move(self, a, b, ga, gb, dur):
        n = max(1, int(dur * RATE))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # なめらかに加減速
            self.send([x + (y - x) * s for x, y in zip(a, b)], ga + (gb - ga) * s)
            time.sleep(1.0 / RATE)
        return b, gb


def jitter(pose, amp):
    return [p + random.uniform(-amp, amp) for p in pose]


def run_episode(node, speed):
    d = lambda sec: sec / speed
    a, g = START, GRIPPER_OPEN
    above_pick = jitter(ABOVE_PICK, 0.03)
    grasp = jitter(GRASP, 0.01)
    above_place = jitter(ABOVE_PLACE, 0.03)
    place = jitter(PLACE, 0.015)
    a, g = node.move(a, START, g, GRIPPER_OPEN, d(0.5))
    a, g = node.move(a, above_pick, g, GRIPPER_OPEN, d(2.5))
    a, g = node.move(a, grasp, g, GRIPPER_OPEN, d(1.5))
    a, g = node.move(a, grasp, g, GRIPPER_CLOSE, d(1.0))
    a, g = node.move(a, grasp, g, GRIPPER_CLOSE, d(0.5))
    a, g = node.move(a, above_pick, g, GRIPPER_CLOSE, d(1.5))
    a, g = node.move(a, above_place, g, GRIPPER_CLOSE, d(3.0))   # 土台を180度回す
    a, g = node.move(a, place, g, GRIPPER_CLOSE, d(1.5))
    a, g = node.move(a, place, g, GRIPPER_OPEN, d(1.0))
    a, g = node.move(a, above_place, g, GRIPPER_OPEN, d(1.0))
    a, g = node.move(a, START, g, GRIPPER_OPEN, d(2.5))
    node.move(a, START, g, GRIPPER_OPEN, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/rosbags/pick180'))
    ap.add_argument('--max-tries', type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init(); node = Demo()
    ok = len([d for d in os.listdir(args.out) if d.startswith('ep')])
    tries = 0
    while ok < args.episodes and tries < args.max_tries:
        tries += 1
        # 開始姿勢に戻し、ブロックを元の位置へ
        node.move(START, START, GRIPPER_OPEN, GRIPPER_OPEN, 2.5)
        reset_cube(); time.sleep(1.5)
        ep_dir = os.path.join(args.out, f'ep{ok}')
        shutil.rmtree(ep_dir, ignore_errors=True)
        rec = subprocess.Popen(['ros2', 'bag', 'record', '-o', ep_dir] + TOPICS,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(2.0)
        speed = random.uniform(0.85, 1.15)
        run_episode(node, speed)
        os.killpg(os.getpgid(rec.pid), signal.SIGINT); rec.wait(timeout=20)
        time.sleep(1.0)
        p = cube_pose()
        success = p is not None and abs(p[0]) < 0.05 and 0.05 < p[1] < 0.16 and p[2] > 0.98
        print(f'try {tries}: speed={speed:.2f} cube={p} -> {"OK" if success else "NG"}', flush=True)
        if success:
            ok += 1
        else:
            shutil.rmtree(ep_dir, ignore_errors=True)
    node.move(START, START, GRIPPER_OPEN, GRIPPER_OPEN, 1.0)
    reset_cube()
    print(f'done: {ok} episodes in {tries} tries -> {args.out}', flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
