#!/bin/bash
#SBATCH --job-name=flops-dense-8gpu
#SBATCH --account=long
#SBATCH --partition=long
#SBATCH --nodelist=octave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Dense 8xA100 Phase C: prepare the CI image if its inputs changed, then run
# the M1 and M1+M2 smoke tests. Submit from the shared worktree on yes:
#
#   mkdir -p logs
#   sbatch --test-only scripts/run_theoretical_flops_trace_slurm.slurm
#   sbatch scripts/run_theoretical_flops_trace_slurm.slurm
#
# The long partition on JA allows the long and priority accounts. This script
# uses the long account and pins the allocation to the 8xA100 octave node.

set -euo pipefail

readonly MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
readonly MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
readonly EXPECTED_BRANCH=codex/theoretical-flops-trace

cd "$MEGATRON_REPO"

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

if ! docker version >/dev/null 2>&1; then
  echo "Docker is not usable in this job. Diagnostic information follows:" >&2
  id >&2
  getent group docker >&2 || true
  stat -c '%A %U %G %n' /var/run/docker.sock >&2 || true
  docker version >&2 || true
  exit 1
fi

megatron_base_image=$(<docker/.ngc_version.dev)
image_input_sha=$(
  git ls-files -s \
    assets docker README.md pyproject.toml uv.lock \
    megatron/core/__init__.py megatron/core/package_info.py \
    | sha256sum \
    | awk '{print $1}'
)
existing_input_sha=$(
  docker image inspect "$MEGATRON_IMAGE" \
    --format '{{index .Config.Labels "org.megatron.build-input-sha"}}' \
    2>/dev/null || true
)

echo "base image: $megatron_base_image"
echo "image input sha: $image_input_sha"

if [[ "$existing_input_sha" != "$image_input_sha" ]]; then
  echo "CI image is missing or stale; pulling the base image and rebuilding."
  docker pull "$megatron_base_image" 2>&1 | tee "$run_root/docker-pull.log"
  docker build \
    --target main \
    --build-arg FROM_IMAGE_NAME="$megatron_base_image" \
    --build-arg IMAGE_TYPE=dev \
    --label "org.megatron.build-input-sha=$image_input_sha" \
    -f docker/Dockerfile.ci.dev \
    -t "$MEGATRON_IMAGE" \
    . 2>&1 | tee "$run_root/docker-build.log"
else
  echo "Reusing $MEGATRON_IMAGE; Docker build inputs are unchanged."
fi

docker image inspect "$MEGATRON_IMAGE" \
  --format '{{json .RepoDigests}} {{.Id}} {{.Created}}' \
  | tee "$run_root/docker-image.txt"

docker run --rm --gpus all \
  --ipc=host \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash -lc 'python -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == 8" && nvidia-smi -L' \
  2>&1 | tee "$run_root/gpu-preflight.log"

docker run --rm \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash -lc 'CUDA_VISIBLE_DEVICES="" uv run pytest --noconftest \
    tests/unit_tests/test_theoretical_flops_usage.py \
    tests/unit_tests/test_te_attention_runtime_context.py \
    tests/unit_tests/test_trace_reconciliation.py \
    -v' \
  2>&1 | tee "$run_root/phase-a.log"

run_smoke() {
  local mode="$1"
  local master_port="$2"
  local log_path="$3"
  local status_path="$4"

  set +e
  docker run --rm --gpus all \
    --network=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT="$master_port" \
    -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
    -w /workspace/Megatron-LM \
    "$MEGATRON_IMAGE" \
    bash scripts/run_theoretical_flops_trace_8gpu_smoke.sh "$mode" \
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
  -e EXPECTED_RUN_SHA="$run_sha" \
  -v "$run_root:/results:ro" \
  "$MEGATRON_IMAGE" \
  python -c 'import json, os, pathlib
root = pathlib.Path("/results")
for mode in ("m1", "m1m2"):
    theory = json.loads((root / mode / "theoretical_flops.json").read_text())
    assert theory["computed_total_flops"] == theory["reference_total_flops"]
    assert theory["relative_error"] == 0.0
    assert theory["runtime_context"]["git_commit"] == os.environ["EXPECTED_RUN_SHA"]
    assert theory["runtime_context"]["git_dirty"] is False
recon = json.loads((root / "m1m2" / "reconciliation_rank0.json").read_text())
assert (root / "m1m2" / "torch_profile" / "rank-0.json.gz").stat().st_size > 0
print("theory entries:", len(theory["entries"]))
print("TE backend:", theory["runtime_context"].get("te_selected_backend"))
print("reconciliation keys:", sorted(recon))'

if grep -E -q 'Traceback|CUDA out of memory' \
  "$run_root/m1.log" "$run_root/m1m2.log"; then
  echo "A traceback or CUDA OOM marker was found in the smoke logs." >&2
  exit 1
fi

grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|TRACE RECONCILIATION|Traceback|CUDA out of memory|NCCL' \
  "$run_root/m1.log" "$run_root/m1m2.log" \
  | tee "$run_root/log-highlights.txt" || true

find "$run_root" -maxdepth 4 -type f -printf '%p\t%s bytes\n' \
  | sort \
  | tee "$run_root/artifact-index.txt"

echo "RUN_ROOT=$run_root"
du -sh "$run_root"
