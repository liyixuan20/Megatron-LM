#!/bin/bash
# Shared helpers for theoretical-FLOPs SLURM jobs on JA.
# Sourced by the sbatch wrappers; not submitted on its own.

refresh_docker_group_if_needed() {
  if [[ "${DOCKER_GROUP_REFRESHED:-0}" == "1" ]]; then
    return 0
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v sg >/dev/null 2>&1; then
    echo "docker is not usable and sg is not available." >&2
    id >&2
    getent group docker >&2 || true
    stat -c '%A %U %G %n' /var/run/docker.sock >&2 || true
    docker version >&2 || true
    return 1
  fi
  echo "docker is not in this process's groups; re-executing under sg docker."
  export DOCKER_GROUP_REFRESHED=1
  local caller="${BASH_SOURCE[1]}"
  exec sg docker -c "bash $(printf '%q' "$caller")"
}

require_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "Docker is not usable in this job. Diagnostic information follows:" >&2
  id >&2
  getent group docker >&2 || true
  stat -c '%A %U %G %n' /var/run/docker.sock >&2 || true
  docker version >&2 || true
  return 1
}

# Flags for bind-mounting the shared NFS home: run as the submitting user so
# root_squash cannot block writes, and skip uv editable installs.
#
# The CI image has no /etc/passwd entry for cluster UIDs. Torch inductor calls
# getpass.getuser() while importing Transformer Engine; without USER/LOGNAME
# that becomes KeyError: getpwuid(): uid not found. Seen on job 316981.
set_host_user_docker_opts() {
  local host_user
  host_user="$(id -un 2>/dev/null || echo megatron)"
  HOST_USER_DOCKER_OPTS=(
    --user "$(id -u):$(id -g)"
    --entrypoint bash
    -e HOME=/tmp
    -e USER="$host_user"
    -e LOGNAME="$host_user"
    -e PYTHONPATH=/workspace/Megatron-LM
    -e UV_NO_SYNC=1
    -e PYTHONDONTWRITEBYTECODE=1
  )
}

# Prefer SLURM-visible devices so `docker run --gpus all` cannot leak extra GPUs
# (job 316935 allocated 1 GPU and still saw device_count==8).
#
# Docker/NVIDIA CSV parsing requires inner quotes around a multi-device list:
#   --gpus '"device=0,1,2,3,4,5,6,7"'
docker_gpu_args() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DOCKER_GPU_ARGS=(--gpus "\"device=${CUDA_VISIBLE_DEVICES}\"")
  else
    DOCKER_GPU_ARGS=(--gpus all)
  fi
}

# Persist compiler caches on the shared host directory that wrappers bind-mount.
set_container_cache_opts() {
  local host_cache="${CONTAINER_CACHE:-/home/liyixuan/.cache/megatron-docker}"
  mkdir -p \
    "$host_cache/torchinductor" \
    "$host_cache/torch_extensions" \
    "$host_cache/triton"
  CONTAINER_CACHE_DOCKER_OPTS=(
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/megatron-cache/torchinductor
    -e TORCH_EXTENSIONS_DIR=/tmp/megatron-cache/torch_extensions
    -e TRITON_CACHE_DIR=/tmp/megatron-cache/triton
    -v "$host_cache:/tmp/megatron-cache"
  )
}

# Forward experiment knobs so `sbatch --export=ALL,NUM_LAYERS=28,...` reaches
# the inner smoke/nsys script instead of dying at the Docker boundary.
theoretical_flops_training_env_args() {
  local keys=(
    NVTE_DEBUG
    NVTE_DEBUG_LEVEL
    THEORETICAL_FLOPS_OUTPUT_DIR
    NSYS_OUTPUT
    NPROC_PER_NODE
    NUM_LAYERS
    TRAIN_ITERS
    PROFILE_STEP_START
    PROFILE_STEP_END
    PROFILE_RANKS
    TIMING_LOG_LEVEL
    TP_SIZE
    PP_SIZE
    CP_SIZE
    HIDDEN_SIZE
    FFN_HIDDEN_SIZE
    SEQ_LENGTH
    MICRO_BATCH_SIZE
    GLOBAL_BATCH_SIZE
    NUM_ATTENTION_HEADS
    NUM_QUERY_GROUPS
    KV_CHANNELS
    VOCAB_SIZE
  )
  THEORETICAL_FLOPS_TRAINING_ENV_ARGS=()
  local key
  for key in "${keys[@]}"; do
    if [[ -n "${!key+x}" && -n "${!key}" ]]; then
      THEORETICAL_FLOPS_TRAINING_ENV_ARGS+=(-e "${key}=${!key}")
    fi
  done
}

require_expected_branch() {
  local expected_branch="$1"
  local current_branch
  current_branch=$(git branch --show-current)
  if [[ "$current_branch" != "$expected_branch" ]]; then
    echo "Expected branch $expected_branch, found ${current_branch:-detached HEAD}." >&2
    return 1
  fi
}

require_clean_git_worktree() {
  local status_path="$1"
  git status --short | tee "$status_path"
  if [[ -s "$status_path" ]]; then
    echo "The worktree is dirty; refusing to run a non-reproducible validation." >&2
    return 1
  fi
}

compute_image_input_sha() {
  git ls-files -s \
    assets docker README.md pyproject.toml uv.lock \
    megatron/core/__init__.py megatron/core/package_info.py \
    | sha256sum \
    | awk '{print $1}'
}

require_prepared_image() {
  local image_name="$1"
  local expected_sha="$2"
  local allow_unverified="${ALLOW_UNVERIFIED_IMAGE:-0}"

  if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "Image $image_name is missing on octave." >&2
    echo "Run scripts/prepare_theoretical_flops_image_slurm.slurm first." >&2
    return 1
  fi
  if [[ "$allow_unverified" != "0" && "$allow_unverified" != "1" ]]; then
    echo "ALLOW_UNVERIFIED_IMAGE must be 0 or 1, got: $allow_unverified" >&2
    return 2
  fi

  local existing_sha
  existing_sha=$(
    docker image inspect "$image_name" \
      --format '{{index .Config.Labels "org.megatron.build-input-sha"}}' \
      2>/dev/null || true
  )
  echo "expected image input sha: $expected_sha"
  echo "existing image input sha: ${existing_sha:-missing}"
  if [[ "$existing_sha" != "$expected_sha" ]]; then
    if [[ "$allow_unverified" != "1" ]]; then
      echo "Image $image_name is stale or has no build-input label." >&2
      echo "Run the 1-GPU preparation job; this GPU job will not rebuild it." >&2
      return 1
    fi
    echo "WARNING: using an unverified image because ALLOW_UNVERIFIED_IMAGE=1." >&2
  fi
}

collect_theoretical_flops_metrics() {
  local run_root="$1"
  shift
  local repo="${MEGATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local python_bin=python3
  if ! command -v python3 >/dev/null 2>&1; then
    python_bin=python
  fi
  "$python_bin" "$repo/scripts/collect_theoretical_flops_artifacts.py" \
    --run-root "$run_root" \
    "$@"
}
