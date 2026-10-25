# CRANE+ V2 によるブロックの往復運搬(ACT 模倣学習)

Gazebo 上のロボットアーム CRANE+ V2 に、次の一連の動作を模倣学習(ACT)で学習させ、推論で実行させる。

1. 手前に置いたブロックをつかみ、土台を180度回して反対側に置く
2. 開始姿勢(腕をまっすぐ上に伸ばした姿勢)に戻る
3. 反対側に置いたブロックをつかみ、元の位置へ戻す
4. 開始姿勢に戻り、そのまま止まる

見本の動作は、逆運動学(MoveIt の /compute_ik)で関節角度を求めるスクリプトで自動生成した(60回分)。

## 結果

| モデル | 「いまどの段階か」の入力 | 結果(運ぶ・開始姿勢に戻る・元に戻す・終わったあと止まる、をすべて満たした回) |
|---|---|---|
| 時間の特徴量 | 動き出してからの経過時間 | 開始姿勢を飛ばす、開始姿勢のあとで迷う(seed による当たり外れ) |
| 工程の特徴量 | ブロックを放した回数 + 工程内の経過時間 | 往復はできるが、終わったあとに動き出す |
| **最終** | 同上 + 見本の最後に「止まっている区間」8秒 | **57/60 回**(3つの seed × 4通りの推論 × 5回)。seed 2 は 20/20、録画時の推論は 10/10 |

詳しい記録は `pick180/results/`。

## 構成

```
run_pai_container.sh       コンテナの作成・起動(このフォルダを /workspace にマウントする)
ign-wrapper.sh             WSL2 + GPU 描画で Gazebo を Ogre1 で起動するためのラッパー
pick180/
  gen_demos.py             見本の作成の共通部分(片道: 反対側へ運ぶ)
  gen_demos_roundtrip.py   往復の見本を作成し、rosbag に記録する
  ik_check.py              MoveIt の /compute_ik で姿勢を求める
  cube_pose_pub.py         Gazebo のブロックの位置を /cube_pose として配信するノード
  episode_ctl.py           推論の前後処理(ブロックとアームを初期位置へ戻す、置けたかを判定する)
  rosbag2lerobot_cube.py   rosbag → LeRobotDataset への変換(ブロックの位置・工程の特徴量を追加)
  train_pick180.py         ACT の学習(seed 固定、出力先の指定に対応)
  infer_pick180.py         推論ノード(工程の特徴量、temporal ensembling、試行ごとの記録)
  start_check.py           記録から「開始姿勢に戻ったか」「終わったあと止まり続けたか」を判定する
  roundtrip_infer.sh       推論を繰り返し、自動で判定する
  prep_roundtrip.sh        見本の作成 → 変換 → 学習(時間の特徴量のモデル)
  phase_train_eval.sh      変換 → seed ごとの学習 → 4通りの推論と判定(工程の特徴量のモデル)
  fail_demo.sh             改善前のモデルで失敗例を再現する
  debug_policy_check.py    見本の観測を与えて、予測と見本の動作を比べる(区間別、時間の値をずらす試験つき)
  debug_policy_all.sh      上の比較を、失敗例と最終モデルについてまとめて行う
  run_roundtrip/configs/   時間の特徴量のモデルの設定(変換・特徴量・学習)
  run_phase/configs/       工程の特徴量のモデル(最終)の設定
  analysis/fk_descent.py   記録から指先の位置を計算し、どこへ下りてつかんだかを調べる
  results/                 学習・推論・デバッグの記録、図、試行ごとの軌跡(CSV)
```

## 動かし方

### 前提

- Windows 11 + WSL2(Ubuntu)+ Docker、NVIDIA GPU(学習に CUDA を使用)
- 講座の Docker イメージ `airobotbook/ros2-desktop-ai-robot-book-humble:latest`
- コンテナ内に、講座のリポジトリ [matsuolab/pai_training](https://github.com/matsuolab/pai_training) を `~/pai_training` に clone し、`pixi install` 済みであること(README に従う)
  - このリポジトリのスクリプトは、pai_training の pixi 環境(LeRobot を含む)で実行する

### 1. コンテナの起動

```bash
cd crane-plus-roundtrip-act
PAI_CONTAINER=pai_ros2_gpu ./run_pai_container.sh   # このフォルダが /workspace にマウントされる
```

ブラウザで http://localhost:6080 を開き、コンテナ内でシミュレーションを起動する。

```bash
source /opt/ros/humble/setup.bash && source ~/airobot_ws/install/setup.bash
ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py
```

以降の `*.sh` は WSL 側(このフォルダの `pick180/`)で実行する。コンテナ名を変えたときは `PAI_CONTAINER` を指定する。

### 2. 見本の作成・変換・学習(時間の特徴量のモデル)

```bash
cd pick180
./prep_roundtrip.sh 60      # 見本60回分を ~/dataset/rosbags/roundtrip に作成し、変換・学習
```

### 3. 工程の特徴量のモデル(最終)の学習と評価

```bash
TAG=phase3 SEC=52 HOLD_OPEN=1.0 ./phase_train_eval.sh 1 2 3
```

seed ごとに学習し、4通りの推論(8ステップずつ実行 / 予測を平均しながら実行、それぞれハンドの補助なし / あり)を5回ずつ行って判定する。

### 4. 推論だけを行う

```bash
MODEL=/home/ubuntu/checkpoints/train/crane_plus_phase3_seed2 INFER_FEATURE=--phase \
  N_ACTION_STEPS=1 INFER_EXTRA='--ensemble 0.01' TRACE_DIR=/workspace/pick180/traces/test \
  ./roundtrip_infer.sh 10 52
```

### 5. 方策のデバッグ

```bash
./debug_policy_all.sh
```

## 元にしたもの・ライセンス

- [matsuolab/pai_training](https://github.com/matsuolab/pai_training)(Physical AI 講座の演習用リポジトリ)
  - `train_pick180.py` は `scripts/train.py`(Apache License 2.0、元の著作権表示を残している)を元にした
  - `infer_pick180.py` は `scripts/infer.py`、`rosbag2lerobot_cube.py` は `scripts/rosbag2lerobot.py`、`debug_policy_check.py` は `scripts/debug_policy.py` を元にした。各ファイルの先頭に変更点を書いている
- [huggingface/lerobot](https://github.com/huggingface/lerobot)(ACT の実装)
- [rt-net/crane_plus](https://github.com/rt-net/crane_plus)、講座の Docker イメージに含まれる Gazebo 環境
- 姿勢の決め方は crane_plus_examples の pick_and_place.cpp を参考にした
- 学習済みモデルは容量が大きいため、このリポジトリには含めていない
