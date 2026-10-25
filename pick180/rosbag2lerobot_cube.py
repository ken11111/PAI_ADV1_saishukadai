#!/usr/bin/env python3
"""
[roundtrip 用] 教材の scripts/rosbag2lerobot.py を元にした変換スクリプト。
変更点:
- rosbag の /cube_pose(ブロックの位置 x, y, z)を読み、observation.environment_state に入れる(元は関節角度の複製)。
- 最後にハンドを開いてから trim_after_release_sec 秒(conversion_config.yaml)で記録を切る。
  見本の最後(開始姿勢に戻り、ブロックは元の位置)は始めと同じ状態なので、残すと「動き出す」と「止まる」の区別がつかなくなるため。
- 短い見本の後ろを最後の状態で埋めない(埋めると「止まる」見本が増えるため)。
- 記録の始めの、動き出す前に止まっている区間を除く(trim_leading_hold)。
- add_time_feature が true なら、observation.environment_state の4つ目に、動き出してからの経過時間(秒)を入れる。
- add_phase_feature が true なら、経過時間の代わりに、工程の番号(ハンドを開いてブロックを放した回数: 0, 1, 2)と、
  その工程に入ってからの経過時間(秒)を入れる。放したことは関節角度のハンド(観測)から判定する(推論の --phase と同じ判定)。
- phase_time_stretch(例 [1.0, 1.25, 1.5])を指定すると、見本ごとに、工程内の経過時間だけをその倍率で引き伸ばしたものを足す。
  推論では見本より動きが遅れて、工程内の経過時間が見本にない値になるため、経過時間がずれても動作を決められるようにする。
- final_hold_sec を指定すると、最後のフレーム(開始姿勢に戻ったところ)を、その秒数ぶん繰り返して足す。
  終わったあとも開始姿勢で止まり続けることを学ばせる(工程の番号で最初と区別できるので、止まっている区間を足しても進めなくならない)。
  入力が関節角度とブロックの位置だけだと、止まっている区間(掴んだあとの一時停止、往復の間の開始姿勢など)や、
  始めと終わりの同じ姿勢で、どこまで進んだかが分からず、止まったり同じ動きを繰り返したりするため。
- drop_hold_frames が true なら、指令が前のフレームから変わらない「止まっている」区間のフレームを除く(試したが、掴む前後で動きが乱れたため既定では使わない)。
使い方: configs/conversion_config.yaml のあるフォルダで python rosbag2lerobot_cube.py
"""

import numpy as np
import torch
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
import importlib.util

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None, trim_after_release_sec=None, drop_hold_frames=False, trim_leading_hold=False, add_time_feature=False, add_phase_feature=False, phase_time_stretch=None, final_hold_sec=0.0):
        self.phase_time_stretch = phase_time_stretch or [1.0]
        self.final_hold_sec = final_hold_sec
        self.trim_leading_hold = trim_leading_hold
        self.add_phase_feature = add_phase_feature
        self.add_time_feature = add_time_feature
        self.trim_after_release_sec = trim_after_release_sec
        self.drop_hold_frames = drop_hold_frames
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq

        # Find all episode directories (containing .db3 and metadata.yaml)
        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir}")

        if config_path.endswith('.py'):
            config = load_config_py(config_path)
            features = config.features
            robot_type = getattr(config, "robot_type", "crane_plus")
            repo_id = getattr(config, "repo_id", "teleop_rosbag_dataset")
            task_name = getattr(config, "task_name", "teleop")
        else:
            # YAML fallback
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            features = config['features']
            robot_type = config.get('robot_type', 'crane_plus')
            repo_id = config.get('repo_id', 'teleop_rosbag_dataset')
            task_name = config.get('task_name', 'teleop')

        self.task_name = task_name

        # --- Dataset creation with user prompt if dir exists ---
        while True:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=target_freq or 20,
                    root=self.output_dir,
                    robot_type=robot_type,
                    features=features,
                    use_videos=True,
                )
                break
            except FileExistsError:
                print(f"[Warning] Directory {self.output_dir} already exists.")
                ans = input("Overwrite? (y: overwrite, n: specify new directory name): ").strip().lower()
                if ans == 'y':
                    # Overwrite: Delete the existing directory and recreate it
                    import shutil
                    shutil.rmtree(self.output_dir)
                    print(f"{self.output_dir} deleted. Recreating...")
                    continue
                else:
                    new_dir = input("Enter new directory name: ").strip()
                    self.output_dir = self.output_dir.parent / new_dir
                    print(f"New output directory: {self.output_dir}")
                    continue

        if "teleop" not in self.dataset.meta.task_to_task_index:
            self.dataset.meta.add_task("teleop")

    def _get_messages(self, episode_dir):
        """Get messages from all required topics in the rosbag."""
        messages = {
            '/joint_states': [],
            '/crane_plus_arm_controller/joint_trajectory': [],
            '/crane_plus_gripper_controller/joint_trajectory': [],
            '/cube_pose': [],
        }
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic in messages.keys()]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))
        return messages

    def _sample_and_hold(self, times, values, target_times):
        values = np.asarray(values)
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
            new_values = np.zeros((len(target_times), values.shape[1]))
        elif len(values.shape) >= 3:
            new_values = np.zeros((len(target_times), *values.shape[1:]), dtype=values.dtype)
        else:
            new_values = np.zeros((len(target_times), values.shape[1]))
        for i, target_time in enumerate(target_times):
            idx = np.searchsorted(times, target_time, side='right') - 1
            if idx < 0:
                idx = 0
            new_values[i] = values[idx]
        if len(values.shape) == 1:
            new_values = new_values.flatten()
        return new_values

    def convert(self):
        max_frames = 0
        episode_lengths = []
        print("\nFirst pass: calculating episode lengths...")
        for episode_dir in self.episode_dirs:
            try:
                messages = self._get_messages(episode_dir)
                j_times = np.array([ts for _, ts in messages['/joint_states']])
                a_times = np.array([ts for _, ts in messages['/crane_plus_arm_controller/joint_trajectory']])
                g_times = np.array([ts for _, ts in messages['/crane_plus_gripper_controller/joint_trajectory']])
                # Use the intersection of available times
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = int((end_time - start_time) * target_freq / 1e9)
                max_frames = max(max_frames, num_frames)
                episode_lengths.append(num_frames)
                print(f"Episode {episode_dir}: {num_frames} frames")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                episode_lengths.append(0)
                continue

        print(f"\nLongest episode: {max_frames} frames")

        for idx, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing episode {idx} from {episode_dir}...")
            try:
                messages = self._get_messages(episode_dir)
                joint_states = messages['/joint_states']
                arm_cmds = messages['/crane_plus_arm_controller/joint_trajectory']
                gripper_cmds = messages['/crane_plus_gripper_controller/joint_trajectory']
                j_times = np.array([ts for _, ts in joint_states])
                j_pos = np.array([msg.position for msg, _ in joint_states])
                a_times = np.array([ts for _, ts in arm_cmds])
                a_pos = np.array([msg.points[0].positions if msg.points else [0,0,0,0] for msg, _ in arm_cmds])
                g_times = np.array([ts for _, ts in gripper_cmds])
                c_times = np.array([ts for _, ts in messages['/cube_pose']])
                c_pos = np.array([[msg.point.x, msg.point.y, msg.point.z] for msg, _ in messages['/cube_pose']])
                g_pos = np.array([msg.points[0].positions[0] if (msg.points and len(msg.points[0].positions)>0) else 0.0 for msg, _ in gripper_cmds])
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                target_freq = self.target_freq or 20
                num_frames = episode_lengths[idx]
                target_times = np.linspace(start_time, end_time, num_frames)
                # Sample and hold for each
                j_pos_sampled = self._sample_and_hold(j_times, j_pos, target_times)  # (num_frames, 5)
                a_pos_sampled = self._sample_and_hold(a_times, a_pos, target_times)  # (num_frames, 4)
                g_pos_sampled = self._sample_and_hold(g_times, g_pos, target_times)  # (num_frames,)
                c_pos_sampled = self._sample_and_hold(c_times, c_pos, target_times)  # (num_frames, 3)
                # Concatenate after sampling
                a_pos_full = np.hstack([a_pos_sampled, g_pos_sampled.reshape(-1,1)])  # (num_frames, 5)
                if self.trim_after_release_sec is not None:
                    closed = np.where(a_pos_full[:, 4] > -0.2)[0]  # ハンドを閉じている(開く途中を含む)フレーム
                    num_frames = min(num_frames, int(closed[-1]) + 1 + int(self.trim_after_release_sec * target_freq))
                    print(f"trimmed to {num_frames} frames")
                frames = list(range(num_frames))
                if self.trim_leading_hold:
                    moved = np.where(np.any(np.abs(a_pos_full[:num_frames] - a_pos_full[0]) > 1e-4, axis=1))[0]
                    first = max(0, int(moved[0]) - 1) if len(moved) else 0
                    frames = frames[first:]
                    print(f"trimmed {first} leading hold frames")
                if self.drop_hold_frames:
                    a = a_pos_full[:num_frames]
                    still = np.r_[True, np.all(np.abs(np.diff(a, axis=0)) < 1e-5, axis=1)]
                    frames = [i for i in frames if not still[i]]
                    print(f"dropped {num_frames - len(frames)} hold frames")
                frames += [frames[-1]] * int(self.final_hold_sec * target_freq)
                for k in (self.phase_time_stretch if self.add_phase_feature else [1.0]):
                    phase = PhaseTracker(1.0 / target_freq)
                    for n, idx_to_use in enumerate(frames):
                        env = c_pos_sampled[idx_to_use]
                        if self.add_phase_feature:
                            ph, pt = phase.update(j_pos_sampled[idx_to_use][4])
                            env = np.r_[env, ph, pt * k]
                        elif self.add_time_feature:
                            env = np.r_[env, n / target_freq]
                        self.dataset.add_frame({
                            "observation.state": torch.tensor(j_pos_sampled[idx_to_use], dtype=torch.float32),
                            "observation.environment_state": torch.tensor(env, dtype=torch.float32),
                            "action": torch.tensor(a_pos_full[idx_to_use], dtype=torch.float32),
                            "task": self.task_name,
                        })
                    print(f"Saving episode {idx} (time x{k})...")
                    self.dataset.save_episode()
                print(f"Episode {idx} saved successfully")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return str(self.dataset.root)

class PhaseTracker:
    """工程の番号(ブロックを放した回数)と、その工程に入ってからの経過時間を数える。infer_pick180.py の --phase でも使う。
    ハンド(観測)が閉じた状態(> -0.2)で 0.5秒以上続いたあと、開いた(<= -0.2)ときを「放した」とする。"""
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


def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    # Load parameters from conversion_config.yaml
    with open("configs/conversion_config.yaml", "r") as f:
        conv_cfg = yaml.safe_load(f)
    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    config = conv_cfg["config"]
    target_freq = conv_cfg.get("target_freq", 20)

    converter = RosbagToLeRobot(bag_dir, output_dir, config, target_freq, conv_cfg.get("trim_after_release_sec"), conv_cfg.get("drop_hold_frames", False),
                                 conv_cfg.get("trim_leading_hold", False), conv_cfg.get("add_time_feature", False),
                                 conv_cfg.get("add_phase_feature", False), conv_cfg.get("phase_time_stretch"), conv_cfg.get("final_hold_sec", 0.0))
    output_path = converter.convert()
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    main() 