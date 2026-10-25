#!/bin/bash
# [roundtrip 工程つき] 工程の番号を入力に加えたモデルを、seed を変えて学習し、そのたびに推論10回で評価する。WSL 側で実行する。
#   使い方: [TAG=phase2] ./phase_train_eval.sh [seed ...](既定: 1 2 3)
#   TAG: モデルと記録の名前(既定 phase)。HOLD_OPEN を指定すると、ハンドの開閉をその秒数保つ補助(--hold-open、--hold-close)ありの推論も行う
#   - 見本は今の roundtrip の60回分を使い、run_phase/configs の設定で変換し直す
#   - モデルは seed ごとに ~/checkpoints/train/crane_plus_phase_seed<seed> に残す
#   - 推論は方法ごとに5回ずつ行い、「開始姿勢に戻ったか」「終わったあと止まり続けたか」も判定する。記録は traces/<TAG>_seed<seed>_<方法>/
set -u
cd "$(dirname "$0")"
SEEDS=${*:-1 2 3}
TAG=${TAG:-phase}
SEC=${SEC:-45}
HOLD_OPEN=${HOLD_OPEN:-}
C=${PAI_CONTAINER:-pai_ros2_gpu}
PIXI='~/.pixi/bin/pixi run --manifest-path ~/pai_training/pyproject.toml -e gpu'
OUT=${TAG}_train_eval_$(date +%Y%m%d_%H%M%S).txt
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT"; }
dx() { docker exec -u ubuntu $C bash -lc "$1"; }

log "開始: 工程つき、seed $SEEDS"
dx "cd /workspace/pick180/run_phase && rm -rf ~/dataset/lerobot_dataset/crane_plus_roundtrip_phase && $PIXI python /workspace/pick180/rosbag2lerobot_cube.py" > convert_phase.log 2>&1
log "変換: $(grep -c 'saved successfully' convert_phase.log) 回分 / $(grep -m1 'trimmed to' convert_phase.log)"
for S in $SEEDS; do
  M=/home/ubuntu/checkpoints/train/crane_plus_${TAG}_seed$S
  log "seed $S: 学習"
  dx "cd /workspace/pick180/run_phase && rm -rf $M && TRAIN_SEED=$S TRAIN_OUTPUT=$M WANDB_MODE=offline $PIXI python /workspace/pick180/train_pick180.py" > train_${TAG}_seed$S.log 2>&1
  dx "rm -rf $M/step_*"
  log "seed $S: $(grep 'step:' train_${TAG}_seed$S.log | tail -1) / $(tail -1 train_${TAG}_seed$S.log)"
  inf() {  # $1=名前 $2=N_ACTION_STEPS $3=追加のオプション
    local o; o=$(MODEL=$M INFER_FEATURE=--phase TRACE_DIR=/workspace/pick180/traces/${TAG}_seed${S}_$1 N_ACTION_STEPS=$2 INFER_EXTRA="$3" ./roundtrip_infer.sh 5 $SEC 2>&1 | tail -6)
    echo "[$1]" >> "$OUT"; echo "$o" >> "$OUT"
    log "seed $S 結果 [$1]: $(echo "$o" | tail -1)"
  }
  inf nas8 8 ''
  inf ens 1 '--ensemble 0.01'
  if [ -n "$HOLD_OPEN" ]; then
    inf nas8_hold 8 "--hold-open $HOLD_OPEN --hold-close $HOLD_OPEN"
    inf ens_hold 1 "--ensemble 0.01 --hold-open $HOLD_OPEN --hold-close $HOLD_OPEN"
  fi
done
log "すべて終了"
