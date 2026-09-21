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
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
    -e PYTHONPATH=/workspace/Megatron-LM
    -e UV_NO_SYNC=1
    -e PYTHONDONTWRITEBYTECODE=1
  )
}

# Prefer SLURM-visible devices so `docker run --gpus all` cannot leak extra GPUs
# (job 316935 allocated 1 GPU and still saw device_count==8).
docker_gpu_args() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DOCKER_GPU_ARGS=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
  else
    DOCKER_GPU_ARGS=(--gpus all)
  fi
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
