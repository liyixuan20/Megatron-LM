#!/bin/bash
#SBATCH --job-name=flops-help-debug
#SBATCH --account=a100
#SBATCH --partition=a100
#SBATCH --nodelist=octave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Reproduce the 8-GPU smoke --help preflight with stderr visible.
# Submit from yes: sbatch scripts/debug_theoretical_flops_help_slurm.slurm

set -euo pipefail

readonly MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
readonly MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev

# shellcheck source=scripts/theoretical_flops_slurm_common.sh
source "$MEGATRON_REPO/scripts/theoretical_flops_slurm_common.sh"

cd "$MEGATRON_REPO"
refresh_docker_group_if_needed
require_docker
set_host_user_docker_opts

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
id
date -Is

docker run --rm --gpus all \
  "${HOST_USER_DOCKER_OPTS[@]}" \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  scripts/debug_theoretical_flops_help.sh
