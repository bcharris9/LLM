#!/usr/bin/env bash
set -euo pipefail

# Start a single local vLLM server on a Jetson and keep it attached to the terminal.
IMAGE="${LAB_CHAT_VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
MODEL="${LAB_CHAT_MODEL:-espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16}"
HOST="${LAB_CHAT_HOST:-127.0.0.1}"
PORT="${LAB_CHAT_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${LAB_CHAT_GPU_MEMORY_UTILIZATION:-0.8}"

if ! sudo docker info >/dev/null 2>&1; then
  echo "Docker is not ready. Start the Docker daemon on the Jetson and retry." >&2
  exit 1
fi

if ! sudo docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  # Pull lazily so repeat boots reuse the cached image.
  echo "Pulling chat server image: ${IMAGE}"
  sudo docker pull "${IMAGE}"
fi

exec sudo docker run -it --rm \
  --runtime=nvidia \
  --network host \
  "${IMAGE}" \
  vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
