#!/usr/bin/env python3
"""
[roundtrip 用] 次の流れを1回の「見本の動き」として作り、rosbag に記録する。
  1. ブロックを元の位置から掴み、土台を180度回して反対側に置く(pick180 と同じ)
  2. アームを開始姿勢(START)に戻す
  3. 反対側に置いたブロックを掴み、土台を180度戻して元の位置に置き、開始姿勢に戻る

- 1 で置く位置を毎回ばらつかせ(PLACE_X / PLACE_Y の範囲)、3 では実際に置かれたブロックの位置に合わせて掴む姿勢を /compute_ik で求める
- ブロックの位置は /cube_pose(cube_pose_pub.py)から読み、rosbag にも記録する。学習と推論では、これを observation.environment_state に使う
- 1 で反対側に置けなかった回、3 で元の位置に戻せなかった回の記録は削除する
- 動きの速さは --speed-min / --speed-max の範囲でばらつかせる。学習で経過時間を入力に使うときは、ばらつかせると
  同じ時刻に違う段階の動きが混ざるので、1.0 に固定する(既定)

使い方(コンテナ内、シミュレーションと cube_pose_pub.py の起動中に):
  python3 /workspace/pick180/gen_demos_roundtrip.py --episodes 25 --out ~/dataset/rosbags/roundtrip
"""
import argparse, math, os, random, shutil, signal, subprocess, time
import rclpy
from geometry_msgs.msg import PointStamped
from moveit_msgs.srv import GetPositionIK
from gen_demos import (Demo, reset_cube, jitter, START, GRIPPER_OPEN, GRIPPER_CLOSE,
                       ABOVE_PICK, GRASP, PLACE, TOPICS)
from ik_check import ik

ROUNDTRIP_TOPICS = TOPICS + ['/cube_pose']
# 1 で置く位置(crane_plus_base 座標)の範囲。pick180 の推論で置かれた位置のばらつき(x -0.03〜0、y 0.10〜0.125)を含むように取る
PLACE_X = (-0.035, 0.02)
PLACE_Y = (0.085, 0.115)
# 3 で元の位置に置く姿勢(gen_demos の PLACE を元の位置の側に向けたもの)
PUT_HOME = [-1.5708] + PLACE[1:]
# 掴む中心は、ブロックの中心より土台側に 5mm(元の位置 world y=-0.1 を base y=-0.095 で掴んでいることから)
GRASP_OFFSET = 0.005


def on_far_side(p):
    return p is not None and 0.05 < p[1] < 0.16 and p[2] > 0.98


def is_home(p):
    return p is not None and abs(p[0]) < 0.05 and -0.16 < p[1] < -0.05 and p[2] > 0.98


class RoundtripDemo(Demo):
    def __init__(self):
        super().__init__()
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik'); self.ik_cli.wait_for_service(10)
        self.cube = None
        self.create_subscription(PointStamped, '/cube_pose', self.on_cube, 10)

    def on_cube(self, m):
        self.cube = (m.point.x, m.point.y, m.point.z)

    def cube_now(self):
        self.cube = None
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.cube is not None:
                return self.cube
        return None

    def poses_at(self, x, y, z_low):
        """真上(z=0.17)と、下ろした位置(z_low)の関節角度(joint1〜4)。手先は真下向き、向きは土台からの方向"""
        yaw = math.atan2(y, x)
        above, _ = ik(self, self.ik_cli, x, y, 0.17, 0, math.pi, yaw)
        low, _ = ik(self, self.ik_cli, x, y, z_low, 0, math.pi, yaw)
        return above, low


def run_episode(node, speed):
    d = lambda sec: sec / speed
    # 1 で置く位置を決める
    for _ in range(10):
        above_place, place = node.poses_at(random.uniform(*PLACE_X), random.uniform(*PLACE_Y), 0.14)
        if above_place and place:
            break
    else:
        return 'ik_place'
    a, g = START, GRIPPER_OPEN
    above_pick = jitter(ABOVE_PICK, 0.03)
    grasp = jitter(GRASP, 0.01)
    # 1. 掴んで反対側へ運ぶ
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
    # 2. 開始姿勢に戻る
    a, g = node.move(a, START, g, GRIPPER_OPEN, d(2.5))
    a, g = node.move(a, START, g, GRIPPER_OPEN, d(1.0))
    p = node.cube_now()
    if not on_far_side(p):
        return 'far_side'
    # 3. 置かれた位置に合わせて掴み、元の位置へ戻す
    r = math.hypot(p[0], p[1])
    k = (r - GRASP_OFFSET) / r
    above_take, take = node.poses_at(p[0] * k, p[1] * k, 0.13)
    if not (above_take and take):
        return 'ik_take'
    above_home = jitter(ABOVE_PICK, 0.03)
    put = jitter(PUT_HOME, 0.015)
    a, g = node.move(a, above_take, g, GRIPPER_OPEN, d(2.5))
    a, g = node.move(a, take, g, GRIPPER_OPEN, d(1.5))
    a, g = node.move(a, take, g, GRIPPER_CLOSE, d(1.0))
    a, g = node.move(a, take, g, GRIPPER_CLOSE, d(0.5))
    a, g = node.move(a, above_take, g, GRIPPER_CLOSE, d(1.5))
    a, g = node.move(a, above_home, g, GRIPPER_CLOSE, d(3.0))   # 土台を180度戻す
    a, g = node.move(a, put, g, GRIPPER_CLOSE, d(1.5))
    a, g = node.move(a, put, g, GRIPPER_OPEN, d(1.0))
    a, g = node.move(a, above_home, g, GRIPPER_OPEN, d(1.0))
    a, g = node.move(a, START, g, GRIPPER_OPEN, d(2.5))
    node.move(a, START, g, GRIPPER_OPEN, 1.0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=25)
    ap.add_argument('--out', default=os.path.expanduser('~/dataset/rosbags/roundtrip'))
    ap.add_argument('--max-tries', type=int, default=60)
    ap.add_argument('--speed-min', type=float, default=1.0)
    ap.add_argument('--speed-max', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init(); node = RoundtripDemo()
    ok = len([d for d in os.listdir(args.out) if d.startswith('ep')])
    tries = 0
    while ok < args.episodes and tries < args.max_tries:
        tries += 1
        node.move(START, START, GRIPPER_OPEN, GRIPPER_OPEN, 2.5)
        reset_cube(); time.sleep(1.5)
        ep_dir = os.path.join(args.out, f'ep{ok}')
        shutil.rmtree(ep_dir, ignore_errors=True)
        rec = subprocess.Popen(['ros2', 'bag', 'record', '-o', ep_dir] + ROUNDTRIP_TOPICS,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(2.0)
        speed = random.uniform(args.speed_min, args.speed_max)
        mid = run_episode(node, speed)
        os.killpg(os.getpgid(rec.pid), signal.SIGINT); rec.wait(timeout=20)
        time.sleep(1.0)
        p = node.cube_now()
        success = isinstance(mid, tuple) and is_home(p)
        fmt = lambda q: tuple(round(v, 3) for v in q) if isinstance(q, tuple) else q
        print(f'try {tries}: speed={speed:.2f} placed={fmt(mid)} final={fmt(p)} -> {"OK" if success else "NG"}', flush=True)
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
