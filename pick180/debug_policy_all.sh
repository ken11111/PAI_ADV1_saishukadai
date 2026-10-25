#!/bin/bash
# 失敗例と最終モデルを、見本の観測で debug_policy_check.py にかける
cd "$(dirname "$0")"; mkdir -p debug_policy
T=/home/ubuntu/checkpoints/train; D=/home/ubuntu/dataset/lerobot_dataset
run() {  # $1=見出し $2=モデル $3=データセット $4=時間の倍率 $5=追加
  echo "=== $1"
  docker exec -u ubuntu ${PAI_CONTAINER:-pai_ros2_gpu} bash -lc "cd ~/pai_training && ~/.pixi/bin/pixi run -e gpu python /workspace/pick180/debug_policy_check.py --dataset-root $D/$3 --repo-id $3 --policy-path $T/$2 --episodes 0 1 2 3 4 --time-scale $4 --out /workspace/pick180/debug_policy/$2_x$4.png $5 2>&1" | grep -v -i -E "warn|Loading weights|^ *$|⚠|│|╭|╰|·|help:|glibc|platforms"
}
run "① B3 見本どおりの時間" crane_plus_roundtrip_act_best crane_plus_roundtrip 1.0
run "① B3 時間を1.25倍" crane_plus_roundtrip_act_best crane_plus_roundtrip 1.25
run "② seed1 見本どおりの時間" roundtrip_seed1 crane_plus_roundtrip 1.0
run "② seed1 時間を1.25倍" roundtrip_seed1 crane_plus_roundtrip 1.25
run "③ 工程つき・終了後の区間なし(終了後の8秒を含む見本で)" crane_plus_phase_seed2 crane_plus_roundtrip_phase 1.0 "--from-sec 37"
run "最終 工程つき・終了後の区間あり" crane_plus_phase3_act_best crane_plus_roundtrip_phase 1.0 "--from-sec 37"
run "最終 工程内の時間を1.25倍" crane_plus_phase3_act_best crane_plus_roundtrip_phase 1.25 "--from-sec 37"
