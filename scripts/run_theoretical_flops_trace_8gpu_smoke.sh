#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-m1m2}"
if [[ "${MODE}" != "m1" && "${MODE}" != "m1m2" ]]; then
  echo "Usage: $0 [m1|m1m2]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/theoretical_flops_dense_model.sh
source "$SCRIPT_DIR/theoretical_flops_dense_model.sh"

theoretical_flops_preflight_report_flag

export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"

THEORETICAL_FLOPS_OUTPUT_DIR="${THEORETICAL_FLOPS_OUTPUT_DIR:-./flops_analysis}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"
theoretical_flops_dense_pretrain_args

COMMON_LOG_ARGS=(
  --log-throughput
  --timing-log-level "${TIMING_LOG_LEVEL}"
)

PROFILE_ARGS=()
if [[ "${MODE}" == "m1m2" ]]; then
  # Warmup 1-3, one active profile step at 4, then leftover steady steps 5-8.
  TRAIN_ITERS="${TRAIN_ITERS:-8}"
  # shellcheck disable=SC2206
  PROFILE_RANK_ARGS=(${PROFILE_RANKS:-0})
  PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --pytorch-profiler-collect-shapes
    --profile-step-start "${PROFILE_STEP_START:-4}"
    --profile-step-end "${PROFILE_STEP_END:-5}"
    --profile-ranks "${PROFILE_RANK_ARGS[@]}"
  )
else
  TRAIN_ITERS="${TRAIN_ITERS:-2}"
fi

python -m torch.distributed.run \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" \
  --master-port "${MASTER_PORT:-29500}" \
  pretrain_gpt.py \
  "${THEORETICAL_FLOPS_PRETRAIN_ARGS[@]}" \
  "${COMMON_LOG_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  --train-iters "${TRAIN_ITERS}"
