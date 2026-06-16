#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${MEDIAMTX_CONFIG:-stream/mediamtx.yml}"

if command -v mediamtx >/dev/null 2>&1; then
  exec mediamtx "$CONFIG_PATH"
fi

if command -v docker >/dev/null 2>&1; then
  exec docker run --rm \
    --name mediamtx \
    --network=host \
    -v "$PWD/$CONFIG_PATH:/mediamtx.yml:ro" \
    bluenviron/mediamtx:1
fi

cat >&2 <<'EOF'
MediaMTX is not installed and Docker is not available.

Install one of these on the GPU PC:
  sudo apt install -y docker.io
or download MediaMTX from:
  https://github.com/bluenviron/mediamtx/releases
EOF
exit 1
