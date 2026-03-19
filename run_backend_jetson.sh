#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PY="${LAB_PYTHON:-${ROOT}/.venv312/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="python"
fi

API_HOST="${LAB_API_HOST:-127.0.0.1}"
API_PORT="${LAB_API_PORT:-8000}"
CHAT_HOST="${LAB_CHAT_HOST:-127.0.0.1}"
CHAT_PORT="${LAB_CHAT_PORT:-8001}"
CHAT_BASE_URL_DEFAULT="http://${CHAT_HOST}:${CHAT_PORT}/v1"
export LAB_CHAT_BACKEND="${LAB_CHAT_BACKEND:-openai_compat}"
export LAB_CHAT_BASE_URL="${LAB_CHAT_BASE_URL:-${CHAT_BASE_URL_DEFAULT}}"
export LAB_CHAT_MODEL="${LAB_CHAT_MODEL:-espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16}"
export LAB_CHAT_API_KEY="${LAB_CHAT_API_KEY:-}"
export LAB_MANUAL_VERSION="${LAB_MANUAL_VERSION:-v2}"

VLLM_IMAGE="${LAB_CHAT_VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
GPU_MEMORY_UTILIZATION="${LAB_CHAT_GPU_MEMORY_UTILIZATION:-0.8}"
CHAT_READY_ATTEMPTS="${LAB_CHAT_READY_ATTEMPTS:-1800}"
API_READY_ATTEMPTS="${LAB_API_READY_ATTEMPTS:-300}"
LOG_DIR="${LAB_BACKEND_LOG_DIR:-${ROOT}/_backend_logs}"
mkdir -p "${LOG_DIR}"
CHAT_LOG="${LOG_DIR}/chat_server.log"
API_LOG="${LOG_DIR}/api_server.log"

chat_pid=""
api_pid=""

cleanup() {
  local exit_code=$?
  if [[ -n "${api_pid}" ]] && kill -0 "${api_pid}" 2>/dev/null; then
    kill "${api_pid}" 2>/dev/null || true
    wait "${api_pid}" 2>/dev/null || true
  fi
  if [[ -n "${chat_pid}" ]] && kill -0 "${chat_pid}" 2>/dev/null; then
    kill "${chat_pid}" 2>/dev/null || true
    wait "${chat_pid}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

ensure_docker_ready() {
  if ! sudo docker info >/dev/null 2>&1; then
    echo "Docker is not ready. Start the Docker daemon on the Jetson and retry." >&2
    return 1
  fi
}

ensure_chat_image() {
  if ! sudo docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "Pulling chat server image: ${VLLM_IMAGE}"
    sudo docker pull "${VLLM_IMAGE}"
  fi
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-120}"
  for ((i=0; i<attempts; i++)); do
    if "${PY}" - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as resp:
    if 200 <= resp.status < 500:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "${label} did not become ready in time: ${url}" >&2
  return 1
}

echo "Using Python: ${PY}"
echo "Chat backend: ${LAB_CHAT_BACKEND}"
echo "Chat base URL: ${LAB_CHAT_BASE_URL}"
echo "Chat model: ${LAB_CHAT_MODEL}"
echo "Lab manual version: ${LAB_MANUAL_VERSION}"

ensure_docker_ready
ensure_chat_image

if [[ ! -f "${ROOT}/assets/model_bundle.joblib" ]]; then
  echo "Building runtime assets..."
  "${PY}" "${ROOT}/build_runtime_assets.py"
fi

if [[ ! -f "${ROOT}/assets_hybrid/hybrid_config.json" ]]; then
  echo "Building hybrid assets..."
  "${PY}" "${ROOT}/build_hybrid_assets.py"
fi

echo "Starting local chat server..."
sudo docker run --rm \
  --runtime=nvidia \
  --network host \
  "${VLLM_IMAGE}" \
  vllm serve "${LAB_CHAT_MODEL}" \
    --host "${CHAT_HOST}" \
    --port "${CHAT_PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  >"${CHAT_LOG}" 2>&1 &
chat_pid=$!

wait_for_http "${LAB_CHAT_BASE_URL}/models" "Local chat server" "${CHAT_READY_ATTEMPTS}"
echo "Local chat server is ready."

echo "Starting FastAPI..."
"${PY}" -m uvicorn server:app --host "${API_HOST}" --port "${API_PORT}" >"${API_LOG}" 2>&1 &
api_pid=$!

wait_for_http "http://${API_HOST}:${API_PORT}/health" "FastAPI server" "${API_READY_ATTEMPTS}"
echo "Backend ready:"
echo "  API:  http://${API_HOST}:${API_PORT}"
echo "  Chat: ${LAB_CHAT_BASE_URL}"
echo "Logs:"
echo "  ${CHAT_LOG}"
echo "  ${API_LOG}"

wait "${api_pid}"
