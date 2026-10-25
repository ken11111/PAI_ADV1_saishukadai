#!/bin/bash
# ROS2デスクトップ環境(noVNC: http://localhost:6080)を起動する
#
# 2026-10-04 変更:
#   - GPU描画: WSL2の /dev/dxg を渡し、Mesa の D3D12 ドライバで RTX 5070 Ti を使う
#     (従来は llvmpipe = CPUによるソフトウェア描画で、Gazebo+RViz起動時にCPUを大量に消費していた)
#   - Gazebo は D3D12 で Ogre2 が動かないため、ign-wrapper.sh で Ogre1 に切り替える
#     (従来どおり Ogre2 + CPU描画で起動したいときは、コンテナ内で GZ_USE_OGRE2=1 を付ける)
#   - CPU使用量の上限: CPUS 個分まで(既定 8)
#
# 使い方: ./run_pai_container.sh   (環境変数 PAI_CONTAINER でコンテナ名、CPUS でCPU上限を変更可)
# ※ WSLでは NAME にPC名が入っているため、コンテナ名には PAI_CONTAINER を使う

NAME=${PAI_CONTAINER:-pai_ros2}
CPUS=${CPUS:-8}
IMAGE=airobotbook/ros2-desktop-ai-robot-book-humble:latest
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if docker container inspect "$NAME" >/dev/null 2>&1; then
  # 既存コンテナはデータ保護のため作り直さない。CPU上限だけ反映して起動する
  docker update --cpus="$CPUS" "$NAME" >/dev/null
  docker start "$NAME" >/dev/null
  if ! docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$NAME" | grep -q /dev/dxg; then
    echo "注意: 既存コンテナ '$NAME' はGPU描画なし(CPU描画)で作成されています。CPU上限 ${CPUS} だけ適用しました。"
    echo "      GPU描画を使うには、別名で新規作成してください(例: PAI_CONTAINER=pai_ros2_gpu $0)"
  fi
  echo "起動しました: $NAME  -> http://localhost:6080"
  exit 0
fi

docker run -d --name "$NAME" \
  --cpus="$CPUS" \
  -p 6080:80 -p 9090:9090 -p 9876:9876 \
  --shm-size=512m \
  --device /dev/dxg \
  -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e GALLIUM_DRIVER=d3d12 \
  -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  -v "$(pwd)":/workspace \
  "$IMAGE" || exit 1

# Gazebo を Ogre1(GPU描画で動作する)で起動させるラッパーを配置
docker cp "$SCRIPT_DIR/ign-wrapper.sh" "$NAME":/usr/local/bin/ign
docker exec -u root "$NAME" chmod 755 /usr/local/bin/ign

echo "作成・起動しました: $NAME (GPU描画, CPU上限 ${CPUS})  -> http://localhost:6080"
