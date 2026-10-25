#!/usr/bin/env python3
"""
推論試験の前後処理(gen_demos.py の関数を流用)。

  python3 episode_ctl.py reset   # アームを開始姿勢に戻し、ブロックを元の位置へ置く
  python3 episode_ctl.py check   # ブロックが 180度先に運ばれたかを判定し、OK / NG を表示
  python3 episode_ctl.py start   # アームだけを開始姿勢に戻す(ブロックは動かさない。往復の間に使う)
  python3 episode_ctl.py check_home  # ブロックが元の位置に戻ったかを判定し、OK / NG を表示
"""
import sys, time
sys.path.insert(0, '/workspace/pick180')
import rclpy
from gen_demos import Demo, START, GRIPPER_OPEN, reset_cube, cube_pose


def main():
    cmd = sys.argv[1]
    if cmd == 'reset':
        rclpy.init(); node = Demo()
        node.move(START, START, GRIPPER_OPEN, GRIPPER_OPEN, 3.0)
        reset_cube(); time.sleep(1.5)
        rclpy.shutdown()
        print('reset done', cube_pose(), flush=True)
    elif cmd == 'check':
        p = cube_pose()
        ok = p is not None and abs(p[0]) < 0.05 and 0.05 < p[1] < 0.16 and p[2] > 0.98
        print(f'cube={p} -> {"OK" if ok else "NG"}', flush=True)
    elif cmd == 'start':
        rclpy.init(); node = Demo()
        node.move(START, START, GRIPPER_OPEN, GRIPPER_OPEN, 2.0)
        rclpy.shutdown()
        print('start pose', cube_pose(), flush=True)
    elif cmd == 'check_home':
        p = cube_pose()
        ok = p is not None and abs(p[0]) < 0.05 and -0.16 < p[1] < -0.05 and p[2] > 0.98
        print(f'cube={p} -> {"OK" if ok else "NG"}', flush=True)


if __name__ == '__main__':
    main()
