#!/bin/bash
#SBATCH --job-name=flops-docker-probe
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --nodelist=octave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 1-GPU, ~2 minute check that a batch job can use Docker via sg docker.
# Does not require a clean worktree. Submit from yes:
#
#   mkdir -p logs
#   sbatch scripts/probe_theoretical_flops_docker_slurm.slurm
#
# Success: logs/flops-docker-probe-<jobid>.out contains DOCKER_PROBE_OK and
# sacct State=COMPLETED ExitCode=0:0. Then submit the 8-GPU smoke.

set -euo pipefail

readonly MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
readonly MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev

# shellcheck source=scripts/theoretical_flops_slurm_common.sh
source "$MEGATRON_REPO/scripts/theoretical_flops_slurm_common.sh"

cd "$MEGATRON_REPO"
refresh_docker_group_if_needed
require_docker

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
id
docker info | grep -i -E 'Server Version|Operating System|CPUs|Total Memory' || true

if ! docker image inspect "$MEGATRON_IMAGE" >/dev/null 2>&1; then
  echo "Image $MEGATRON_IMAGE is missing; sg docker works, but Phase C needs the image." >&2
  echo "DOCKER_PROBE_OK_NO_IMAGE"
  exit 0
fi

set_host_user_docker_opts
docker_gpu_args
docker run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  -lc 'python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda_device_count\", torch.cuda.device_count())"; echo "---- nsys ----"; if command -v nsys >/dev/null; then nsys --version; else ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null || echo NSYS_MISSING; fi'

echo "DOCKER_PROBE_OK"
