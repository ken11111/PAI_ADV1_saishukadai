#!/bin/bash
# [roundtrip 用] 見本の動きの作成 → LeRobot 形式への変換 → 学習 を続けて行う。WSL 側で実行する。
#   使い方: ./prep_roundtrip.sh [見本の数=25]
set -eu
N=${1:-25}
C=${PAI_CONTAINER:-pai_ros2_gpu}
ROS='source /opt/ros/humble/setup.bash; source ~/airobot_ws/install/setup.bash'
PIXI='~/.pixi/bin/pixi run --manifest-path ~/pai_training/pyproject.toml -e gpu'
dx() { docker exec -u ubuntu -e DISPLAY=:0 $C bash -lc "$1"; }
cd "$(dirname "$0")"
# ブロックの位置を流すノードが動いていなければ起動する
dx "pgrep -f '[c]ube_pose_pub.py' >/dev/null" || { docker exec -d -u ubuntu $C bash -lc "$ROS; python3 /workspace/pick180/cube_pose_pub.py > /tmp/cube_pose_pub.log 2>&1"; sleep 4; }
echo "$(date '+%T') 見本の作成"
dx "$ROS; rm -rf ~/dataset/rosbags/roundtrip_prev; [ -d ~/dataset/rosbags/roundtrip ] && mv ~/dataset/rosbags/roundtrip ~/dataset/rosbags/roundtrip_prev; cd /workspace/pick180; python3 gen_demos_roundtrip.py --episodes $N --out ~/dataset/rosbags/roundtrip" 2>&1 | grep -v WARN | tee gen_roundtrip.log
echo "$(date '+%T') 変換"
dx "cd /workspace/pick180/run_roundtrip && rm -rf ~/dataset/lerobot_dataset/crane_plus_roundtrip && $PIXI python /workspace/pick180/rosbag2lerobot_cube.py" > convert_roundtrip.log 2>&1
tail -1 convert_roundtrip.log
echo "$(date '+%T') 学習"
dx "cd /workspace/pick180/run_roundtrip && rm -rf ~/checkpoints/train/crane_plus_roundtrip_act && WANDB_MODE=offline $PIXI python /workspace/pick180/train_pick180.py" > train_roundtrip.log 2>&1; cp train_roundtrip.log train_roundtrip_$(date +%Y%m%d_%H%M%S).log
grep 'step:' train_roundtrip.log | tail -1; tail -1 train_roundtrip.log
echo "$(date '+%T') 完了"
