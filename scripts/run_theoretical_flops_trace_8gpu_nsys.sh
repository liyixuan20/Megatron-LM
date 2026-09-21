#!/usr/bin/env bash
# 8-GPU Megatron nsys capture for dense GQA layer/chunk compute+comm.
# Do not combine with --use-pytorch-profiler. This is the M4 path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/theoretical_flops_dense_model.sh
source "$SCRIPT_DIR/theoretical_flops_dense_model.sh"

theoretical_flops_preflight_report_flag

resolve_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return 0
  fi
  local candidate
  for candidate in /opt/nvidia/nsight-systems/*/bin/nsys; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! NSYS_BIN="$(resolve_nsys)"; then
  echo "nsys is not available in this container. Check the NGC PyTorch image." >&2
  exit 1
fi
echo "nsys_bin=${NSYS_BIN}"
"$NSYS_BIN" --version

export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"

THEORETICAL_FLOPS_OUTPUT_DIR="${THEORETICAL_FLOPS_OUTPUT_DIR:-./flops_analysis}"
NSYS_OUTPUT="${NSYS_OUTPUT:-${THEORETICAL_FLOPS_OUTPUT_DIR}/nsys/megatron-dense-gqa}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
PROFILE_STEP_START="${PROFILE_STEP_START:-4}"
PROFILE_STEP_END="${PROFILE_STEP_END:-5}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"

mkdir -p "$(dirname "$NSYS_OUTPUT")"
theoretical_flops_dense_pretrain_args

# Read PROFILE_RANKS as a word list so "0 1" remains two CLI values.
# shellcheck disable=SC2206
PROFILE_RANK_ARGS=(${PROFILE_RANKS})

"$NSYS_BIN" profile \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -x true \
  -o "$NSYS_OUTPUT" \
  python -m torch.distributed.run \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --nnodes 1 \
    --node-rank 0 \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" \
    --master-port "${MASTER_PORT:-29500}" \
    pretrain_gpt.py \
    "${THEORETICAL_FLOPS_PRETRAIN_ARGS[@]}" \
    --log-throughput \
    --timing-log-level "${TIMING_LOG_LEVEL}" \
    --profile \
    --nvtx-ranges \
    --record-shapes \
    --profile-step-start "${PROFILE_STEP_START}" \
    --profile-step-end "${PROFILE_STEP_END}" \
    --profile-ranks "${PROFILE_RANK_ARGS[@]}" \
    --train-iters "${TRAIN_ITERS}"

if [[ -f "${NSYS_OUTPUT}.nsys-rep" ]]; then
  "$NSYS_BIN" export --type sqlite -o "${NSYS_OUTPUT}.sqlite" \
    "${NSYS_OUTPUT}.nsys-rep" \
    || echo "nsys sqlite export failed; keeping ${NSYS_OUTPUT}.nsys-rep" >&2
fi

echo "NSYS_OUTPUT=${NSYS_OUTPUT}"
echo "THEORETICAL_FLOPS_OUTPUT_DIR=${THEORETICAL_FLOPS_OUTPUT_DIR}"
