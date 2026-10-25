#!/bin/bash
# GPU(D3D12)ではOgre2が動かないため、gazebo起動時はOgre1を使う
# (GZ_USE_OGRE2=1 で無効化して従来どおりOgre2=CPU描画で起動)
if [ "$1" = "gazebo" ] && [ -z "$GZ_USE_OGRE2" ]; then
  shift
  args=()
  while [ $# -gt 0 ]; do
    if [ "$1" = "--gui-config" ] && [ -f "$2" ]; then
      tmp=$(mktemp /tmp/gui-ogre1-XXXXXX.config)
      sed 's#<engine>ogre2</engine>#<engine>ogre</engine>#' "$2" > "$tmp"
      args+=("$1" "$tmp"); shift 2
    else
      args+=("$1"); shift
    fi
  done
  exec /usr/bin/ign gazebo --render-engine ogre "${args[@]}"
fi
exec /usr/bin/ign "$@"
