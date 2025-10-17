#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICES="${GPU_DEVICES:-4,5,6,7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_DEVICES}"

IFS=',' read -r -a _cuda_device_list <<< "${CUDA_VISIBLE_DEVICES}"
DEFAULT_NPROC="${#_cuda_device_list[@]}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$DEFAULT_NPROC}"

# Torchrun will pick the python executable from the active conda env (gfpack).
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=55332 --module src.train "$@"
