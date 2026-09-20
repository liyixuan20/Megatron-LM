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
set_host_user_docker_opts() {
  HOST_USER_DOCKER_OPTS=(
    --user "$(id -u):$(id -g)"
    --entrypoint bash
    -e HOME=/tmp
    -e PYTHONPATH=/workspace/Megatron-LM
    -e UV_NO_SYNC=1
    -e PYTHONDONTWRITEBYTECODE=1
  )
}
