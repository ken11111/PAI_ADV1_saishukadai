#!/bin/bash
# 録画用: 改善前のモデルの失敗例を、モデルごとに2回ずつ推論する
# モデル: crane_plus_roundtrip_act_best と roundtrip_seed1 は時間の特徴量のモデル(prep_roundtrip.sh の学習を seed を変えて行ったもの)、
#        crane_plus_phase_seed2 は工程の特徴量・終了後の区間なしのモデル(final_hold_sec 0、trim_after_release_sec 4.5 で TAG=phase として学習したもの)
T=/home/ubuntu/checkpoints/train
run() {  # $1=見出し $2=モデル $3=特徴量 $4=トレース名
  echo "$(date '+%T') ---- $1 開始"
  MODEL=$T/$2 INFER_FEATURE=$3 TRACE_DIR=/workspace/pick180/traces/fail_$4 N_ACTION_STEPS=8 ./roundtrip_infer.sh 2 52 2>&1 | tail -3
}
run "① 時間の特徴量だけ(保存していたモデル)" crane_plus_roundtrip_act_best --time b3
run "② 時間の特徴量だけ(別の seed)" roundtrip_seed1 --time rs1
run "③ 工程つき・終了後の区間なし" crane_plus_phase_seed2 --phase ph2
echo "$(date '+%T') ---- すべて終了"
