#!/bin/bash
#SBATCH --job-name=flops-dense-8gpu
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --nodelist=octave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Dense 8xA100 Phase C Chrome smoke. Default is m1m2 only; set SMOKE_MODE=both
# to also run the short M1-only job. Never pulls or builds an image.
#
#   mkdir -p logs
#   sbatch scripts/run_theoretical_flops_trace_slurm.slurm
#   sbatch --export=ALL,NUM_LAYERS=28,SMOKE_MODE=m1m2 \
#     scripts/run_theoretical_flops_trace_slurm.slurm

set -euo pipefail

readonly MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
readonly MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
readonly EXPECTED_BRANCH=codex/theoretical-flops-trace
readonly CONTAINER_CACHE=/home/liyixuan/.cache/megatron-docker

# shellcheck source=scripts/theoretical_flops_slurm_common.sh
source "$MEGATRON_REPO/scripts/theoretical_flops_slurm_common.sh"

cd "$MEGATRON_REPO"
refresh_docker_group_if_needed

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
id
date -Is

SMOKE_MODE="${SMOKE_MODE:-m1m2}"
if [[ "$SMOKE_MODE" != "m1" && "$SMOKE_MODE" != "m1m2" && "$SMOKE_MODE" != "both" ]]; then
  echo "SMOKE_MODE must be m1, m1m2, or both; got: $SMOKE_MODE" >&2
  exit 2
fi

THEORETICAL_FLOPS_OUTPUT_DIR="${THEORETICAL_FLOPS_OUTPUT_DIR:-./flops_analysis}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
export THEORETICAL_FLOPS_OUTPUT_DIR NPROC_PER_NODE PROFILE_RANKS
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"
export TRAIN_ITERS="${TRAIN_ITERS:-8}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-4}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-5}"
export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"
export NUM_LAYERS="${NUM_LAYERS:-4}"

require_expected_branch "$EXPECTED_BRANCH"

run_sha=$(git rev-parse HEAD)
run_tag=$(date +%Y%m%d-%H%M%S)-${run_sha:0:12}-${SLURM_JOB_ID:-manual}
run_root="$MEGATRON_REPO/runs/theoretical-flops/$run_tag"
mkdir -p "$run_root"

if [[ -e flops_analysis ]]; then
  mv flops_analysis "$run_root/preexisting-flops_analysis"
fi
if [[ "$THEORETICAL_FLOPS_OUTPUT_DIR" != "./flops_analysis" && -e "$THEORETICAL_FLOPS_OUTPUT_DIR" ]]; then
  mv "$THEORETICAL_FLOPS_OUTPUT_DIR" "$run_root/preexisting-flops-output"
fi

require_clean_git_worktree "$run_root/git-status.txt"
git show -s --format=fuller "$run_sha" | tee "$run_root/git-commit.txt"
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R" \
  | tee "$run_root/slurm-allocation.txt"
nvidia-smi -L | tee "$run_root/nvidia-smi-L.txt"

require_docker
image_input_sha=$(compute_image_input_sha)
require_prepared_image "$MEGATRON_IMAGE" "$image_input_sha"

docker image inspect "$MEGATRON_IMAGE" \
  --format '{{json .RepoDigests}} {{.Id}} {{.Created}} {{index .Config.Labels "org.megatron.build-input-sha"}}' \
  | tee "$run_root/docker-image.txt"
printf '%s\n' "$image_input_sha" | tee "$run_root/image-input-sha.txt"

set_host_user_docker_opts
docker_gpu_args
set_container_cache_opts
theoretical_flops_training_env_args

docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -e NPROC_PER_NODE="$NPROC_PER_NODE" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  -lc 'python -c "import os, torch; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == int(os.environ[\"NPROC_PER_NODE\"])"' \
  2>&1 | tee "$run_root/gpu-preflight.log"

run_smoke() {
  local mode="$1"
  local master_port="$2"
  local log_path="$3"
  local status_path="$4"

  set +e
  docker run --rm \
    "${DOCKER_GPU_ARGS[@]}" \
    --network=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${HOST_USER_DOCKER_OPTS[@]}" \
    "${CONTAINER_CACHE_DOCKER_OPTS[@]}" \
    "${THEORETICAL_FLOPS_TRAINING_ENV_ARGS[@]}" \
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT="$master_port" \
    -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
    -w /workspace/Megatron-LM \
    "$MEGATRON_IMAGE" \
    scripts/run_theoretical_flops_trace_8gpu_smoke.sh "$mode" \
    2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" | tee "$status_path"
  return "$status"
}

accept_smoke_output() {
  local mode="$1"
  local dest="$run_root/$mode"
  local output_dir="$THEORETICAL_FLOPS_OUTPUT_DIR"
  test -s "$output_dir/theoretical_flops.json"
  if [[ "$mode" == "m1m2" ]]; then
    # shellcheck disable=SC2206
    local ranks=(${PROFILE_RANKS})
    local rank
    for rank in "${ranks[@]}"; do
      test -s "$output_dir/torch_profile/rank-${rank}.json.gz"
      gzip -t "$output_dir/torch_profile/rank-${rank}.json.gz"
      test -s "$output_dir/reconciliation_rank${rank}.json"
    done
  fi
  mv "$output_dir" "$dest"
}

master_port_base=$((20000 + SLURM_JOB_ID % 20000))
collect_logs=()

if [[ "$SMOKE_MODE" == "m1" || "$SMOKE_MODE" == "both" ]]; then
  run_smoke m1 "$master_port_base" \
    "$run_root/m1.log" "$run_root/m1-exit-status.txt"
  accept_smoke_output m1
  collect_logs+=(--log "$run_root/m1.log")
  grep -q 'THEORETICAL FLOPS REPORT' "$run_root/m1.log"
fi

if [[ "$SMOKE_MODE" == "m1m2" || "$SMOKE_MODE" == "both" ]]; then
  local_port="$master_port_base"
  if [[ "$SMOKE_MODE" == "both" ]]; then
    local_port=$((master_port_base + 1))
  fi
  run_smoke m1m2 "$local_port" \
    "$run_root/m1m2.log" "$run_root/m1m2-exit-status.txt"
  accept_smoke_output m1m2
  collect_logs+=(--log "$run_root/m1m2.log")
  grep -q 'THEORETICAL FLOPS REPORT' "$run_root/m1m2.log"
  grep -q 'TRACE RECONCILIATION' "$run_root/m1m2.log"
fi

result_mode="m1m2"
if [[ "$SMOKE_MODE" == "m1" ]]; then
  result_mode="m1"
fi

# shellcheck disable=SC2206
profile_ranks=(${PROFILE_RANKS})
docker run --rm \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -e EXPECTED_RUN_SHA="$run_sha" \
  -e RESULT_MODE="$result_mode" \
  -e PROFILE_RANKS="$PROFILE_RANKS" \
  -v "$run_root:/results:ro" \
  "$MEGATRON_IMAGE" \
  -c 'python -c "
import json, os, pathlib
root = pathlib.Path(\"/results\")
mode = os.environ[\"RESULT_MODE\"]
theory = json.loads((root / mode / \"theoretical_flops.json\").read_text())
assert theory[\"computed_total_flops\"] == theory[\"reference_total_flops\"]
assert theory[\"relative_error\"] == 0.0
assert theory[\"runtime_context\"][\"git_commit\"] == os.environ[\"EXPECTED_RUN_SHA\"]
assert theory[\"runtime_context\"][\"git_dirty\"] is False
print(\"theory entries:\", len(theory[\"entries\"]))
print(\"TE backend:\", theory[\"runtime_context\"].get(\"te_selected_backend\"))
if mode == \"m1m2\":
    for rank in os.environ[\"PROFILE_RANKS\"].split():
        recon = json.loads((root / mode / f\"reconciliation_rank{rank}.json\").read_text())
        assert (root / mode / \"torch_profile\" / f\"rank-{rank}.json.gz\").stat().st_size > 0
        print(f\"rank {rank} matched:\", recon.get(\"matched\"))
"'

log_paths=()
if [[ -f "$run_root/m1.log" ]]; then
  log_paths+=("$run_root/m1.log")
fi
if [[ -f "$run_root/m1m2.log" ]]; then
  log_paths+=("$run_root/m1m2.log")
fi
if grep -E -q 'Traceback|CUDA out of memory' "${log_paths[@]}"; then
  echo "A traceback or CUDA OOM marker was found in the smoke logs." >&2
  exit 1
fi

grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|TRACE RECONCILIATION|Selected backend|Traceback|CUDA out of memory|NCCL' \
  "${log_paths[@]}" \
  | tee "$run_root/log-highlights.txt" || true

collect_theoretical_flops_metrics "$run_root" "${collect_logs[@]}" \
  | tee "$run_root/collect.log"

python3 - <<PY
import json
from pathlib import Path
root = Path("$run_root")
backend = json.loads((root / "metrics" / "te_attention_backend.json").read_text())
throughput = json.loads((root / "metrics" / "throughput.json").read_text())
if not backend.get("te_selected_backend"):
    raise SystemExit(f"TE selected backend is empty: {backend}")
if throughput.get("recommended_throughput_tflops_per_gpu") is None:
    raise SystemExit(f"No throughput samples were collected: {throughput}")
print("TE backend:", backend.get("te_selected_backend_display") or backend["te_selected_backend"])
print("recommended TFLOP/s/GPU:", throughput["recommended_throughput_tflops_per_gpu"])
PY

echo "RUN_ROOT=$run_root"
du -sh "$run_root"
