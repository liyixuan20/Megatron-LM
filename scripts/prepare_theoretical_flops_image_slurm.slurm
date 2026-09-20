#!/bin/bash
#SBATCH --job-name=flops-image-prepare
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --nodelist=octave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Prepare the node-local CI image and run the CPU-only Phase A tests before an
# 8-GPU Phase C allocation is allowed to start. Set FORCE_IMAGE_BUILD=1 to
# rebuild even when the recorded Docker-input fingerprint matches.

set -euo pipefail

readonly MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
readonly MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
readonly EXPECTED_BRANCH=codex/theoretical-flops-trace

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
prep_tag=$(date +%Y%m%d-%H%M%S)-${run_sha:0:12}-${SLURM_JOB_ID:-manual}
prep_root="$MEGATRON_REPO/runs/theoretical-flops-prep/$prep_tag"
mkdir -p "$prep_root"

git status --short | tee "$prep_root/git-status.txt"
if [[ -s "$prep_root/git-status.txt" ]]; then
  echo "The worktree is dirty; refusing to prepare an image for an unreproducible run." >&2
  exit 1
fi

git show -s --format=fuller "$run_sha" | tee "$prep_root/git-commit.txt"
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R" \
  | tee "$prep_root/slurm-allocation.txt"

require_docker

force_image_build=${FORCE_IMAGE_BUILD:-0}
if [[ "$force_image_build" != "0" && "$force_image_build" != "1" ]]; then
  echo "FORCE_IMAGE_BUILD must be 0 or 1, got: $force_image_build" >&2
  exit 2
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
echo "expected image input sha: $image_input_sha"
echo "existing image input sha: ${existing_input_sha:-missing}"

mkdir -p assets

if [[ "$force_image_build" == "1" || "$existing_input_sha" != "$image_input_sha" ]]; then
  echo "Pulling the base image and building $MEGATRON_IMAGE."
  docker pull "$megatron_base_image" 2>&1 | tee "$prep_root/docker-pull.log"
  docker build \
    --target main \
    --build-arg FROM_IMAGE_NAME="$megatron_base_image" \
    --build-arg IMAGE_TYPE=dev \
    --label "org.megatron.build-input-sha=$image_input_sha" \
    -f docker/Dockerfile.ci.dev \
    -t "$MEGATRON_IMAGE" \
    . 2>&1 | tee "$prep_root/docker-build.log"
else
  echo "Reusing $MEGATRON_IMAGE; Docker build inputs are unchanged."
fi

prepared_input_sha=$(
  docker image inspect "$MEGATRON_IMAGE" \
    --format '{{index .Config.Labels "org.megatron.build-input-sha"}}'
)
if [[ "$prepared_input_sha" != "$image_input_sha" ]]; then
  echo "Prepared image fingerprint does not match the current checkout." >&2
  exit 1
fi

docker image inspect "$MEGATRON_IMAGE" \
  --format '{{json .RepoDigests}} {{.Id}} {{.Created}} {{index .Config.Labels "org.megatron.build-input-sha"}}' \
  | tee "$prep_root/docker-image.txt"

set_host_user_docker_opts
docker run --rm \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -e CUDA_VISIBLE_DEVICES="" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  -lc 'python -m pytest --noconftest -p no:cacheprovider \
    tests/unit_tests/test_theoretical_flops_usage.py \
    tests/unit_tests/test_te_attention_runtime_context.py \
    tests/unit_tests/test_trace_reconciliation.py \
    -v' \
  2>&1 | tee "$prep_root/phase-a.log"

printf '%s\n' "$image_input_sha" | tee "$prep_root/image-input-sha.txt"
echo "PREP_ROOT=$prep_root"
du -sh "$prep_root"
