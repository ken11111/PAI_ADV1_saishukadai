#!/usr/bin/env python3
"""
[pick180 用] 教材の scripts/infer.py を元にした推論ノード。
変更点:
- --n-action-steps: ACT が一度に予測する動作の列(chunk)のうち、何ステップぶんを続けて実行するか。
  学習時の設定(1)のままだと、掴んだ姿勢で止まっている見本の区間に引きずられて、その場で止まり続けることがあるため、
  数ステップぶんをまとめて実行して先へ進めるようにする。
- --repeat: 予測した1つの動作を何回(20Hz の周期)続けて送るか。
  学習データの動作の列は 0.1秒間隔(train_config.yaml の delta_timestamps)なので、20Hz で送るときは 2 にすると見本と同じ速さになる。
- --cube: [roundtrip 用] observation.environment_state に、関節角度の代わりにブロックの位置(/cube_pose の x, y, z)を入れる。
  cube_pose_pub.py を起動しておくこと。
- --ensemble: ACT の temporal ensembling(毎回予測した動作の列を、重み exp(-係数×i) で平均する)を使う。--n-action-steps 1 と組み合わせる。
- --time: [roundtrip 用] observation.environment_state の4つ目に、動き出してからの経過時間(秒)を入れる(学習時の add_time_feature に合わせる)。
- --phase: [roundtrip 工程つき] observation.environment_state の4つ目・5つ目に、工程の番号(ブロックを放した回数)と、
  その工程に入ってからの経過時間(秒)を入れる(学習時の add_phase_feature に合わせる。判定は rosbag2lerobot_cube.py の PhaseTracker と同じ)。
- --hold-open SEC / --hold-close SEC: 一度「開く」と判断したら SEC 秒はハンドを開いたまま、「閉じる」と判断したら SEC 秒は閉じたままにする
  (ハンドだけ。腕の動作はモデルのまま)。開く・閉じるの判断が揺れて、放しきれない・つかみきれないのを防ぐ。
- --trace: 20Hz の周期ごとに、経過時間・関節角度・ブロックの位置・送った動作を CSV に記録する(失敗したときの様子を後から見るため)。
"""
import argparse
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'


class PhaseTracker:
    """rosbag2lerobot_cube.py の PhaseTracker と同じ。ハンド(観測)が 0.5秒以上閉じたあと開いたら、工程の番号を1つ進める。"""
    def __init__(self, dt):
        self.dt, self.phase, self.t, self.closed_t = dt, 0, 0.0, 0.0
    def update(self, hand):
        if hand > -0.2:
            self.closed_t += self.dt
        else:
            if self.closed_t >= 0.5:
                self.phase += 1; self.t = 0.0
            self.closed_t = 0.0
        out = [float(self.phase), self.t]
        self.t += self.dt
        return out


class InferenceNode(Node):
    def __init__(self, model_path, device, n_action_steps, repeat, use_cube=False, use_time=False, ensemble=None, trace=None, phase=False, hold_open=0.0, hold_close=0.0):
        super().__init__('crane_plus_act_infer_pick180')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.config.n_action_steps = n_action_steps
        if ensemble is not None:
            self.policy.config.temporal_ensemble_coeff = ensemble
            self.policy.temporal_ensembler = ACTTemporalEnsembler(ensemble, self.policy.config.chunk_size)
        self.policy.reset()
        self.policy.eval()
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.get_logger().info(f'inference device: {self.device}, n_action_steps={n_action_steps}, repeat={repeat}')
        self.repeat = repeat
        self.tick = 0
        self.action = None
        self.latest_joint_state = None
        self.use_cube = use_cube
        self.use_time = use_time
        self.latest_cube = None
        if use_cube:
            self.create_subscription(PointStamped, '/cube_pose', self.cube_callback, 10)
        self.phase = PhaseTracker(0.05) if phase else None
        self.hold_open, self.hold_close = hold_open, hold_close
        self.grip_closed, self.grip_until = False, -1
        self.phase_now = [0.0, 0.0]
        self.trace = None
        if trace:
            self.trace = open(trace, 'w', buffering=1)
            self.trace.write('t,j1,j2,j4,j3,hand,cube_x,cube_y,cube_z,phase,phase_t,a_j1,a_j2,a_j4,a_j3,a_hand\n')
        self.create_timer(0.05, self.publish_action)  # 20Hz

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        self.latest_joint_state = np.array(arm_pos + [name_to_pos.get(GRIPPER_JOINT_NAME, 0.0)], dtype=np.float32)

    def cube_callback(self, msg):
        self.latest_cube = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)

    def publish_action(self):
        if self.latest_joint_state is None or (self.use_cube and self.latest_cube is None):
            return
        if self.phase:
            self.phase_now = self.phase.update(self.latest_joint_state[4])  # 20Hz の周期ごとに数える
        if self.action is None or self.tick % self.repeat == 0:
            s = torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.use_cube:
                env = self.latest_cube if not self.use_time else np.r_[self.latest_cube, self.tick * 0.05].astype(np.float32)
                if self.phase:
                    env = np.r_[self.latest_cube, self.phase_now].astype(np.float32)
                e = torch.tensor(env, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                e = s
            with torch.no_grad():
                self.action = self.policy.select_action(
                    {'observation.state': s, 'observation.environment_state': e}).cpu().numpy().flatten()
        self.tick += 1
        a = self.action.copy()
        if self.hold_open > 0 or self.hold_close > 0:
            # ハンドの「開く/閉じる」が切り替わったら、しばらく(開くとき hold_open 秒、閉じるとき hold_close 秒)その指令を保つ
            want_closed = a[4] > -0.2
            if self.tick >= self.grip_until and want_closed != self.grip_closed:
                self.grip_closed = want_closed
                self.grip_until = self.tick + int((self.hold_close if want_closed else self.hold_open) / 0.05)
            if self.tick < self.grip_until:
                a[4] = max(a[4], 0.1745) if self.grip_closed else min(a[4], -0.52)
        for pub, names, pos in ((self.arm_pub, ARM_JOINT_NAMES, a[:4].tolist()),
                                (self.gripper_pub, [GRIPPER_JOINT_NAME], [float(a[4])])):
            jt = JointTrajectory(); jt.joint_names = names
            pt = JointTrajectoryPoint(); pt.positions = pos
            pt.time_from_start.nanosec = int(0.1 * 1e9)
            jt.points.append(pt); pub.publish(jt)
        print('action:', a[:4], 'gripper:', a[4], flush=True)
        if self.trace:
            c = self.latest_cube if self.latest_cube is not None else [np.nan] * 3
            self.trace.write(','.join(f'{v:.4f}' for v in [(self.tick - 1) * 0.05, *self.latest_joint_state, *c, *self.phase_now, *a[:5]]) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--n-action-steps', type=int, default=8)
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--cube', action='store_true')
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--ensemble', type=float, default=None)
    ap.add_argument('--trace', default=None)
    ap.add_argument('--phase', action='store_true')
    ap.add_argument('--hold-open', type=float, default=0.0)
    ap.add_argument('--hold-close', type=float, default=0.0)
    args = ap.parse_args()
    rclpy.init()
    node = InferenceNode(args.model_path, args.device, args.n_action_steps, args.repeat, args.cube, args.time, args.ensemble, args.trace, args.phase, args.hold_open, args.hold_close)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
