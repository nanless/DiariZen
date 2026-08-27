#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/root/miniforge3/bin/conda}"
ENV_NAME="${ENV_NAME:-diarizen-trt1010}"
SOURCE_ENV="${SOURCE_ENV:-cosyvoice}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${SCRIPT_DIR}/tensorrt_l4_requirements.txt}"
CONDA_ROOT="$(dirname "$(dirname "${CONDA_BIN}")")"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ -e "${ENV_PREFIX}" && ! -d "${ENV_PREFIX}/conda-meta" ]]; then
  echo "Refusing to overwrite non-Conda path: ${ENV_PREFIX}" >&2
  exit 1
fi

if [[ ! -d "${ENV_PREFIX}/conda-meta" ]]; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" --clone "${SOURCE_ENV}"
fi

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --upgrade -r "${REQUIREMENTS_FILE}"
"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python -m pip check
"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import json
import os
import sys

import tensorrt as trt
import torch

result = {
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "tensorrt": trt.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(result, indent=2))
if trt.__version__ != "10.10.0.31":
    raise SystemExit(f"Unexpected TensorRT version: {trt.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
