#!/bin/bash
# [roundtrip 用] 往復の動作(運ぶ → 開始姿勢に戻る → 元の位置へ戻す)を、1つのモデルの推論で繰り返す。画面(noVNC)で見る用。WSL 側で実行する。
#   使い方: ./roundtrip_infer.sh [試行数=10] [1試行の秒数=40(往復1回ぶん。起動の約4秒を含む)]
#   各試行で、途中(反対側に置いたあと)のブロックの位置と、最後に元の位置に戻ったか(OK/NG)を表示する。
#   TRACE_DIR を指定したときは、置いたあとに開始姿勢に戻ったか、終わったあと開始姿勢で止まり続けたか(start_check.py)も判定し、どちらかが NG なら失敗とする。
#   INFER_FEATURE: 経過時間の入れ方(既定 --time。工程つきのモデルは --phase)
set -u
N=${1:-10}; SEC=${2:-40}; MID=${MID_SEC:-21}
C=${PAI_CONTAINER:-pai_ros2_gpu}
MODEL=${MODEL:-/home/ubuntu/checkpoints/train/crane_plus_roundtrip_act}
NAS=${N_ACTION_STEPS:-8}  # 予測した動作の列(chunk)のうち、続けて実行するステップ数
TRACE=${TRACE_DIR:-}      # 指定すると、試行ごとの関節角度・ブロックの位置・動作を <TRACE_DIR>/trial_N.csv に記録する(/workspace/pick180 から見たパス)
ROS='source /opt/ros/humble/setup.bash; source ~/airobot_ws/install/setup.bash'
PIXI='~/.pixi/bin/pixi run --manifest-path ~/pai_training/pyproject.toml -e gpu'
dx() { docker exec -u ubuntu -e DISPLAY=:0 $C bash -lc "$1"; }
dx "pgrep -f '[c]ube_pose_pub.py' >/dev/null" || { docker exec -d -u ubuntu $C bash -lc "$ROS; python3 /workspace/pick180/cube_pose_pub.py > /tmp/cube_pose_pub.log 2>&1"; sleep 4; }
[ -n "$TRACE" ] && mkdir -p "${TRACE/#\/workspace\/pick180/.}"
ok=0; mid_ok=0; st_ok=0
for t in $(seq 1 "$N"); do
  dx "$ROS; python3 /workspace/pick180/episode_ctl.py reset" >/dev/null 2>&1
  docker exec -d -u ubuntu $C bash -lc "$ROS; cd ~/pai_training && $PIXI python /workspace/pick180/infer_pick180.py --model-path $MODEL --n-action-steps $NAS --repeat 2 --cube ${INFER_FEATURE:---time} ${INFER_EXTRA:-} ${TRACE:+--trace $TRACE/trial_$t.csv} > /tmp/roundtrip_infer.log 2>&1"
  sleep "$MID"
  m=$(dx "$ROS; python3 /workspace/pick180/episode_ctl.py check" 2>/dev/null | tail -1)
  sleep $((SEC - MID - 3))
  dx "pkill -INT -f '[i]nfer_pick180.py'; sleep 2; pkill -KILL -f '[i]nfer_pick180.py'"
  sleep 1
  r=$(dx "$ROS; python3 /workspace/pick180/episode_ctl.py check_home" 2>/dev/null | tail -1)
  [[ $m == *OK ]] || r="${r%% ->*} -> NG"   # 反対側へ運べていなければ、元の位置のままでも失敗とする
  st=""
  if [ -n "$TRACE" ]; then
    st=$(python3 start_check.py "${TRACE/#\/workspace\/pick180/.}/trial_$t.csv")
    [[ $st != *NG* ]] && st_ok=$((st_ok+1)) || r="${r%% ->*} -> NG"   # 開始姿勢に戻っていない、または終わったあとに動き出したら失敗とする
    st=" / $st"
  fi
  [[ $m == *OK ]] && mid_ok=$((mid_ok+1)); [[ $r == *OK ]] && ok=$((ok+1))
  echo "$(date '+%T') 試行$t: 反対側に置いた位置 ${m%% ->*} (${m##* }) / 最後 ${r%% ->*} (${r##* })$st"
done
echo "反対側へ運ぶ 成功 $mid_ok/$N、${TRACE:+開始姿勢に戻る・終了後に止まる 成功 $st_ok/$N、}元の位置へ戻す 成功 $ok/$N"
