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

# Dense 8xA100 Phase C: consume a previously prepared CI image, then run the
# M1 and M1+M2 smoke tests. Queue with sbatch so the job can wait offline;
# do not use srun sleep infinity. Image is already on octave from the manual
# or 1-GPU prep build. Validate Docker-in-batch first:
#
#   mkdir -p logs
#   scancel <old-flops-jobs-if-any>
#   PROBE_JOB_ID=$(sbatch --parsable scripts/probe_theoretical_flops_docker_slurm.slurm)
#   # after probe COMPLETED / DOCKER_PROBE_OK:
#   sbatch scripts/run_theoretical_flops_trace_slurm.slurm
#
# The 8-GPU job never pulls or builds an image.

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

current_branch=$(git branch --show-current)
if [[ "$current_branch" != "$EXPECTED_BRANCH" ]]; then
  echo "Expected branch $EXPECTED_BRANCH, found ${current_branch:-detached HEAD}." >&2
  exit 1
fi

run_sha=$(git rev-parse HEAD)
run_tag=$(date +%Y%m%d-%H%M%S)-${run_sha:0:12}-${SLURM_JOB_ID:-manual}
run_root="$MEGATRON_REPO/runs/theoretical-flops/$run_tag"
mkdir -p "$run_root"

# Move an earlier output out of the worktree before checking source cleanliness.
if [[ -e flops_analysis ]]; then
  mv flops_analysis "$run_root/preexisting-flops_analysis"
fi

git status --short | tee "$run_root/git-status.txt"
if [[ -s "$run_root/git-status.txt" ]]; then
  echo "The worktree is dirty; refusing to run a non-reproducible validation." >&2
  exit 1
fi

git show -s --format=fuller "$run_sha" | tee "$run_root/git-commit.txt"
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R" \
  | tee "$run_root/slurm-allocation.txt"
nvidia-smi -q | tee "$run_root/nvidia-smi-q.txt"

require_docker

image_input_sha=$(
  git ls-files -s \
    assets docker README.md pyproject.toml uv.lock \
    megatron/core/__init__.py megatron/core/package_info.py \
    | sha256sum \
    | awk '{print $1}'
)

if ! docker image inspect "$MEGATRON_IMAGE" >/dev/null 2>&1; then
  echo "Image $MEGATRON_IMAGE is missing on octave." >&2
  echo "Run scripts/prepare_theoretical_flops_image_slurm.slurm first." >&2
  exit 1
fi

existing_input_sha=$(
  docker image inspect "$MEGATRON_IMAGE" \
    --format '{{index .Config.Labels "org.megatron.build-input-sha"}}' \
    2>/dev/null || true
)

allow_unverified_image=${ALLOW_UNVERIFIED_IMAGE:-0}
if [[ "$allow_unverified_image" != "0" && "$allow_unverified_image" != "1" ]]; then
  echo "ALLOW_UNVERIFIED_IMAGE must be 0 or 1, got: $allow_unverified_image" >&2
  exit 2
fi

echo "expected image input sha: $image_input_sha"
echo "existing image input sha: ${existing_input_sha:-missing}"
if [[ "$existing_input_sha" != "$image_input_sha" ]]; then
  if [[ "$allow_unverified_image" != "1" ]]; then
    echo "Image $MEGATRON_IMAGE is stale or has no build-input label." >&2
    echo "Run the 1-GPU preparation job; this 8-GPU job will not rebuild it." >&2
    exit 1
  fi
  echo "WARNING: using an unverified image because ALLOW_UNVERIFIED_IMAGE=1." >&2
fi

docker image inspect "$MEGATRON_IMAGE" \
  --format '{{json .RepoDigests}} {{.Id}} {{.Created}} {{index .Config.Labels "org.megatron.build-input-sha"}}' \
  | tee "$run_root/docker-image.txt"
printf '%s\n' "$image_input_sha" | tee "$run_root/image-input-sha.txt"

mkdir -p "$CONTAINER_CACHE"
set_host_user_docker_opts

docker_gpu_args
docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  -lc 'python -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == 8"' \
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
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT="$master_port" \
    -e TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    -e TRITON_CACHE_DIR=/tmp/triton \
    -v "$CONTAINER_CACHE:/tmp/megatron-cache" \
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

# Derive ports from the job ID to avoid colliding with unrelated local jobs.
master_port_base=$((20000 + SLURM_JOB_ID % 20000))

run_smoke m1 "$master_port_base" \
  "$run_root/m1.log" "$run_root/m1-exit-status.txt"
test -s flops_analysis/theoretical_flops.json
mv flops_analysis "$run_root/m1"

run_smoke m1m2 "$((master_port_base + 1))" \
  "$run_root/m1m2.log" "$run_root/m1m2-exit-status.txt"
test -s flops_analysis/theoretical_flops.json
test -s flops_analysis/torch_profile/rank-0.json.gz
test -s flops_analysis/reconciliation_rank0.json
gzip -t flops_analysis/torch_profile/rank-0.json.gz
mv flops_analysis "$run_root/m1m2"

grep -q 'THEORETICAL FLOPS REPORT' "$run_root/m1.log"
grep -q 'THEORETICAL FLOPS REPORT' "$run_root/m1m2.log"
grep -q 'TRACE RECONCILIATION' "$run_root/m1m2.log"

docker run --rm \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -e EXPECTED_RUN_SHA="$run_sha" \
  -v "$run_root:/results:ro" \
  "$MEGATRON_IMAGE" \
  -c 'python -c "import json, os, pathlib
root = pathlib.Path(\"/results\")
for mode in (\"m1\", \"m1m2\"):
    theory = json.loads((root / mode / \"theoretical_flops.json\").read_text())
    assert theory[\"computed_total_flops\"] == theory[\"reference_total_flops\"]
    assert theory[\"relative_error\"] == 0.0
    assert theory[\"runtime_context\"][\"git_commit\"] == os.environ[\"EXPECTED_RUN_SHA\"]
    assert theory[\"runtime_context\"][\"git_dirty\"] is False
recon = json.loads((root / \"m1m2\" / \"reconciliation_rank0.json\").read_text())
assert (root / \"m1m2\" / \"torch_profile\" / \"rank-0.json.gz\").stat().st_size > 0
print(\"theory entries:\", len(theory[\"entries\"]))
print(\"TE backend:\", theory[\"runtime_context\"].get(\"te_selected_backend\"))
print(\"reconciliation keys:\", sorted(recon))"'

if grep -E -q 'Traceback|CUDA out of memory' \
  "$run_root/m1.log" "$run_root/m1m2.log"; then
  echo "A traceback or CUDA OOM marker was found in the smoke logs." >&2
  exit 1
fi

grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|TRACE RECONCILIATION|Traceback|CUDA out of memory|NCCL' \
  "$run_root/m1.log" "$run_root/m1m2.log" \
  | tee "$run_root/log-highlights.txt" || true

collect_theoretical_flops_metrics "$run_root" \
  --log "$run_root/m1.log" \
  --log "$run_root/m1m2.log" \
  | tee "$run_root/collect.log"

echo "RUN_ROOT=$run_root"
du -sh "$run_root"
