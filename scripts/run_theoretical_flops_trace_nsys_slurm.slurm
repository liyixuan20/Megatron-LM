#!/bin/bash
#SBATCH --job-name=flops-dense-8gpu-nsys
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

# Dense 8xA100 nsys capture (M4). Separate from the Chrome-trace smoke.
# Consumes megatron-lm:theoretical-flops-dev; never pulls or builds an image.
#
#   mkdir -p logs
#   sbatch scripts/run_theoretical_flops_trace_nsys_slurm.slurm

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

require_expected_branch "$EXPECTED_BRANCH"

run_sha=$(git rev-parse HEAD)
run_tag=$(date +%Y%m%d-%H%M%S)-${run_sha:0:12}-${SLURM_JOB_ID:-manual}
run_root="$MEGATRON_REPO/runs/theoretical-flops/${run_tag}-nsys"
mkdir -p "$run_root/provenance" "$run_root/logs" "$run_root/artifacts"

if [[ -e flops_analysis ]]; then
  mv flops_analysis "$run_root/preexisting-flops_analysis"
fi

require_clean_git_worktree "$run_root/provenance/git-status.txt"
git show -s --format=fuller "$run_sha" | tee "$run_root/provenance/git-commit.txt"
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R" \
  | tee "$run_root/provenance/slurm-allocation.txt"
nvidia-smi -L | tee "$run_root/provenance/nvidia-smi-L.txt"

require_docker
image_input_sha=$(compute_image_input_sha)
require_prepared_image "$MEGATRON_IMAGE" "$image_input_sha"

docker image inspect "$MEGATRON_IMAGE" \
  --format '{{json .RepoDigests}} {{.Id}} {{.Created}} {{index .Config.Labels "org.megatron.build-input-sha"}}' \
  | tee "$run_root/provenance/docker-image.txt"
printf '%s\n' "$image_input_sha" | tee "$run_root/provenance/image-input-sha.txt"

export THEORETICAL_FLOPS_OUTPUT_DIR="${THEORETICAL_FLOPS_OUTPUT_DIR:-./flops_analysis}"
export NSYS_OUTPUT="${NSYS_OUTPUT:-./flops_analysis/nsys/megatron-dense-gqa}"
export TRAIN_ITERS="${TRAIN_ITERS:-8}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-4}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-5}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export TIMING_LOG_LEVEL="${TIMING_LOG_LEVEL:-1}"
export NUM_LAYERS="${NUM_LAYERS:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"

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
  -lc 'python -c "import os, torch; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == int(os.environ[\"NPROC_PER_NODE\"])" && (command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys) && (nsys --version || /opt/nvidia/nsight-systems/*/bin/nsys --version)' \
  2>&1 | tee "$run_root/logs/nsys-preflight.log"

if ! grep -q -E 'NVIDIA Nsight Systems|nsys version|Product version' \
  "$run_root/logs/nsys-preflight.log"; then
  echo "nsys was not found in the container; refusing to occupy 8 GPUs." >&2
  exit 1
fi

master_port=$((20000 + SLURM_JOB_ID % 20000))

set +e
docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --network=host --ipc=host \
  --cap-add=SYS_ADMIN \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  "${CONTAINER_CACHE_DOCKER_OPTS[@]}" \
  "${THEORETICAL_FLOPS_TRAINING_ENV_ARGS[@]}" \
  -e MASTER_ADDR=127.0.0.1 \
  -e MASTER_PORT="$master_port" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  scripts/run_theoretical_flops_trace_8gpu_nsys.sh \
  2>&1 | tee "$run_root/logs/train.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" | tee "$run_root/logs/nsys-exit-status.txt"

if [[ -e "$THEORETICAL_FLOPS_OUTPUT_DIR" ]]; then
  mkdir -p "$run_root/artifacts"
  mv "$THEORETICAL_FLOPS_OUTPUT_DIR" "$run_root/artifacts/flops_analysis"
fi

collect_theoretical_flops_metrics "$run_root" --log "$run_root/logs/train.log" \
  | tee "$run_root/logs/collect.log"

test "$status" -eq 0
test -s "$run_root/artifacts/flops_analysis/theoretical_flops.json"
test -s "$run_root/metrics/te_attention_backend.json"
test -s "$run_root/run_manifest.json"

if ! find "$run_root/artifacts" \( -name '*.nsys-rep' -o -name '*.qdrep' \) | grep -q .; then
  echo "No nsys report was produced under $run_root/artifacts." >&2
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path
root = Path("$run_root")
theory = json.loads((root / "artifacts/flops_analysis/theoretical_flops.json").read_text())
backend = json.loads((root / "metrics/te_attention_backend.json").read_text())
assert theory["computed_total_flops"] == theory["reference_total_flops"]
assert backend.get("te_selected_backend"), backend
throughput = json.loads((root / "metrics/throughput.json").read_text())
assert throughput.get("recommended_throughput_tflops_per_gpu") is not None, throughput
print("TE backend:", backend.get("te_selected_backend_display") or backend["te_selected_backend"])
print("throughput:", throughput.get("recommended_throughput_tflops_per_gpu"))
PY

if grep -E -q 'Traceback|CUDA out of memory' "$run_root/logs/train.log"; then
  echo "A traceback or CUDA OOM marker was found in the nsys training log." >&2
  exit 1
fi

grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|Selected backend|FlashAttention|nsys_bin|NSYS_OUTPUT|Traceback|CUDA out of memory|NCCL' \
  "$run_root/logs/train.log" \
  | tee "$run_root/logs/log-highlights.txt" || true

echo "RUN_ROOT=$run_root"
du -sh "$run_root"
