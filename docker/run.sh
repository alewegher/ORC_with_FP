#!/usr/bin/env bash
# Run the orc24-gpu image, mounting this repo as a volume.
#
# Usage:
#   docker/run.sh          ephemeral container, removed on exit (default)
#   docker/run.sh keep     persistent named container: created once, reused
#                          (docker start -ai) on later calls instead of recreated
set -euo pipefail

IMAGE="orc24-gpu:latest"
CONTAINER_NAME="orc24_gpu_dev"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT_TARGET="/home/student/shared/ORC_with_FP"

COMMON_ARGS=(
  --gpus all
  -v /tmp/.X11-unix/:/tmp/.X11-unix/
  --env="DISPLAY=$DISPLAY"
  --privileged
  -p 127.0.0.1:7000:7000
  --shm-size 2g
  --user=student
  --workdir="$MOUNT_TARGET"
  -v "$REPO_DIR:$MOUNT_TARGET"
)

if [[ "${1:-}" == "keep" ]]; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Reusing existing container '$CONTAINER_NAME'"
    docker start -ai "$CONTAINER_NAME"
  else
    docker run -it --name "$CONTAINER_NAME" "${COMMON_ARGS[@]}" "$IMAGE" bash
  fi
else
  docker run --rm -it "${COMMON_ARGS[@]}" "$IMAGE" bash
fi
