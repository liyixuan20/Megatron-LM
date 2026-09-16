# Theoretical FLOPs Trace: JA Cluster Instructions

This is the operational runbook for the theoretical-FLOPs and trace-reconciliation
work. It records the source-of-truth Git workflow, the current implementation status,
and the exact commands used on the JA cluster (`yes` + `octave`).

The design and formulas remain in `THEORETICAL_FLOPS_TRACE_PLAN.md`. When this runbook
and an older command in the plan disagree about cluster access, use this runbook.

## 1. Target And Current Status

The immediate target is a Dense GPT smoke run on one `octave` node with all 8 A100s.
The run must produce:

1. Per-operator theoretical shapes and FLOPs.
2. A complete PyTorch Chrome trace for the profiled rank and profile window.
3. A reconciliation report comparing analytical GEMM shapes with trace GEMM events.
4. Logs and immutable copies of the above artifacts for simulator validation.

Dense and MoE are separate tracks. Do not add MoE flags to the Dense smoke script.

| Work item | Status | Evidence / next action |
|---|---|---|
| M1 Dense theoretical report | Implemented | Commit `06a5f407f` |
| M2 trace export and reconciliation | Implemented | Commit `06a5f407f` |
| Offline/lightweight Phase A fixes | Implemented | Commit `5defc03ad` |
| M1/M2 Phase A targeted pytest | Passed previously on server | Re-run in the Docker image before Phase C |
| M1/M2 Phase C, 8 A100 | Not run yet | Run `m1`, then `m1m2` on `octave` |
| M3 MLA/MoE/MTP/THD/PP filtering | Not implemented | Start only after Dense Phase C is understood |
| MoE simulator comparison | Deferred | Simulator-side MoE architecture is not ready |

Current feature branch:

```text
codex/theoretical-flops-trace
```

At the time this runbook was written, the feature HEAD on the local fork was:

```text
5defc03ade33407155fabf778df10d6f797b8000
```

Always record the actual run-time SHA; do not assume the value above is still current.

## 2. Machine And Storage Model

| Location | Purpose | What lives there |
|---|---|---|
| Local WSL | Source editing and commits | Editable Git worktree |
| GitHub personal fork | Source-of-truth transport | Branch `codex/theoretical-flops-trace` |
| `yes` login node | Git sync and SLURM submission | Shared worktree, logs, run artifacts |
| `octave` compute node | Docker build/run and GPU work | Docker image/cache and live containers |
| Docker container | Reproducible Megatron runtime | CUDA, PyTorch, TE, `/opt/venv` |

The server worktree is expected at:

```text
/home/liyixuan/workspace/Megatron-LM
```

`/home` is shared, so the same worktree and its artifacts are visible from `yes` and
`octave`. Docker image layers are node-local to the Docker daemon on `octave`; the Git
repository is bind-mounted into the container and is not copied into the image at run
time.

Never run training on `yes`. Do not put SSH private keys on the cluster. Use the already
configured SSH-agent forwarding for GitHub access.

## 3. Git Update Workflow

The normal loop is:

```text
WSL edit -> local test -> commit -> push personal fork
         -> yes pull exact branch -> octave Docker test
         -> inspect artifacts -> repeat from WSL if code changes are needed
```

### 3.1 Local WSL: edit, commit, and push

The local remote names are intentionally different:

- `fork`: `git@github.com:liyixuan20/Megatron-LM.git`
- `origin`: `git@github.com:NVIDIA/Megatron-LM.git`

Run:

```bash
cd /home/duckie/workspace/Megatron-LM
git switch codex/theoretical-flops-trace
git remote -v
git fetch fork
git status --short
git log --oneline --decorate -3
```

Before editing, the local branch should not be behind the fork. If it is clean and only
needs a fast-forward:

```bash
git pull --ff-only fork codex/theoretical-flops-trace
```

After editing and local checks:

```bash
git status --short
git diff --check
git diff
git add <only-the-files-for-this-change>
git commit -S -s -m "<concise message>"
git push fork codex/theoretical-flops-trace
git rev-parse HEAD
```

`-s` adds the DCO `Signed-off-by` trailer. `-S` signs the commit. For private/internal
experiments, use `-s` alone only when WSL signing is not configured; signed commits are
required before an upstream Megatron-LM PR.

Do not push this branch to NVIDIA's `origin`.

### 3.2 Server `yes`: pull only

The server clone was made from the personal fork, so its `origin` is expected to be
`liyixuan20/Megatron-LM`. Verify rather than relying on the name:

```bash
ssh yes
cd /home/liyixuan/workspace/Megatron-LM
git remote -v
git status --short
git switch codex/theoretical-flops-trace
git pull --ff-only origin codex/theoretical-flops-trace
git log --oneline --decorate -3
git rev-parse HEAD
```

The SHA printed on `yes` must equal the SHA printed after the WSL push. If `git status
--short` is non-empty, stop and inspect those server-side files. Do not reset, overwrite,
or stash them blindly. Permanent source changes are made in WSL and transported through
GitHub; the server worktree does not push changes.

If GitHub SSH is intermittent, diagnose before changing remotes:

```bash
ssh -T git@github.com
git ls-remote origin HEAD
```

## 4. SLURM And Docker Access On `octave`

The JA-specific permission behavior is important: joining the `docker` group does not
refresh an already existing login session. In the observed working sequence, an active
SLURM allocation on `octave` is created first, and a new `ssh octave` session is opened
afterward. That new session sees the Docker group.

Keep the allocation alive for the entire Docker operation. Closing the SSH session does
not release the allocation; cancel it explicitly when finished.

This SSH-after-allocation sequence documents the cluster behavior that has been observed
to work; it is not a way to bypass SLURM. The allocation must reserve every GPU used by
the container. If the administrator requires Docker itself to run as an `srun`/`sbatch`
job step for cgroup accounting, use that site-approved wrapper while keeping the same
image, mount, and commands below.

### 4.1 Terminal A on `yes`: create a 1-GPU access allocation

Use this only for Docker permission checks, pulling/building the image, and Phase A CPU
tests. It does not authorize an 8-GPU training run.

```bash
srun -A a100 -p a100 \
  --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 --mem=96G \
  --time=01:00:00 \
  --job-name=flops-docker-access \
  sleep infinity
```

This command intentionally occupies Terminal A. If the site-provided command is
`infinite sleep` rather than `sleep infinity`, keep using the site-provided form that has
already been verified.

### 4.2 Terminal B on `yes`: verify allocation, then enter `octave`

```bash
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R"
ssh octave
hostname
id
getent group docker
stat -c '%A %U %G %n' /var/run/docker.sock
docker version
docker info | grep -i -E 'runtime|nvidia|root dir|storage'
```

Expected results:

- `hostname` prints `octave`.
- `id` includes the `docker` group.
- `docker version` shows both client and server sections without `permission denied`.
- Docker reports an NVIDIA-capable runtime.

If `id` contains `docker` but the socket is still denied, record all five outputs above
and ask the cluster administrator. Do not use `sudo`, copy private keys, or loosen
`/var/run/docker.sock` permissions.

### 4.3 Formal 8-GPU allocation

End the 1-GPU access job before requesting all GPUs. From `yes`, find and cancel only the
access job:

```bash
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %R"
scancel <flops-docker-access-job-id>
```

Then, in Terminal A on `yes`, request all 8 A100s:

```bash
srun -A a100 -p a100 \
  --nodes=1 --ntasks=1 \
  --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=180G \
  --time=01:00:00 \
  --job-name=flops-dense-8gpu \
  sleep infinity
```

Wait until `squeue` shows `R` and `octave`. Only then open a new session from Terminal B:

```bash
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R"
ssh octave
id
nvidia-smi
```

Do not launch `docker run --gpus all` while holding only the 1-GPU access allocation.
The 8-GPU job may wait in `PD` until all GPUs are available; that is expected.

## 5. Build The Reproducible Docker Image

Use the repository CI image, not a hand-maintained Conda environment. The image pin is
read from `docker/.ngc_version.dev`; the current pin is
`nvcr.io/nvidia/pytorch:26.04-py3`. The CI image contains `uv`, uses `/opt/venv`, and pins
the CUDA/PyTorch/Transformer Engine dependency set expected by the checkout.

The image only needs rebuilding when `docker/`, `pyproject.toml`, or `uv.lock` changes.
Ordinary Python/shell source updates are visible through the bind mount and do not need an
image rebuild.

On `octave`, inside an active allocation:

```bash
cd /home/liyixuan/workspace/Megatron-LM
MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
MEGATRON_BASE_IMAGE=$(<docker/.ngc_version.dev)

docker pull "$MEGATRON_BASE_IMAGE"
docker build \
  --target main \
  --build-arg FROM_IMAGE_NAME="$MEGATRON_BASE_IMAGE" \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev \
  -t "$MEGATRON_IMAGE" \
  .

docker image inspect "$MEGATRON_IMAGE" --format '{{.Id}} {{.Created}}'
```

Do not omit `--target main`: the later `jet` stage requires NVIDIA-internal secrets. Do
not increase compiler parallelism; the cluster explicitly bans jobs that cause host OOM.

## 6. Preflight And Phase A In Docker

Set reusable shell variables after every new `ssh octave` login:

```bash
MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
cd "$MEGATRON_REPO"
```

Confirm the image can see the expected software and all GPUs granted to the formal
8-GPU allocation:

```bash
docker run --rm --gpus all \
  --ipc=host \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash -lc 'python -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() == 8" && nvidia-smi -L'
```

Re-run the targeted Phase A suite in the same image. `--noconftest` is deliberate: these
three tests use lightweight imports and must not load the repository-wide GPU/Triton
fixtures.

```bash
docker run --rm \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash -lc 'CUDA_VISIBLE_DEVICES="" uv run pytest --noconftest \
    tests/unit_tests/test_theoretical_flops_usage.py \
    tests/unit_tests/test_te_attention_runtime_context.py \
    tests/unit_tests/test_trace_reconciliation.py \
    -v'
```

All tests must pass before consuming the full 8-GPU allocation for Phase C. The separate
`test_trace_handler_export_path.py` named in an older plan revision does not exist; its
path behavior is currently tested in `test_trace_reconciliation.py`.

## 7. Dense Phase C Run

The packaged entrypoint is:

```text
scripts/run_theoretical_flops_trace_8gpu_smoke.sh
```

It has two modes:

| Mode | Iterations | Result |
|---|---:|---|
| `m1` | 2 | Startup theory report only |
| `m1m2` | 6 | Theory + rank-0 profiler trace + reconciliation |

The current script profiles rank 0 only, over steps `[2, 4)`. Therefore
`rank-0.json.gz` is the complete Chrome trace for that profiled rank/window, not a trace
for all eight ranks. Profiling all ranks would require an explicit script change and much
more storage; it is not required for the first Dense validation.

### 7.1 Record run identity

Inside the fresh 8-GPU `octave` session:

```bash
MEGATRON_REPO=/home/liyixuan/workspace/Megatron-LM
MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
cd "$MEGATRON_REPO"

git status --short
RUN_SHA=$(git rev-parse HEAD)
RUN_TAG=$(date +%Y%m%d-%H%M%S)-${RUN_SHA:0:12}
RUN_ROOT="$MEGATRON_REPO/runs/theoretical-flops/$RUN_TAG"
mkdir -p "$RUN_ROOT"
set -o pipefail

git show -s --format=fuller "$RUN_SHA" | tee "$RUN_ROOT/git-commit.txt"
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.6D %R" | tee "$RUN_ROOT/slurm-allocation.txt"
nvidia-smi -q | tee "$RUN_ROOT/nvidia-smi-q.txt"
docker image inspect "$MEGATRON_IMAGE" --format '{{json .RepoDigests}} {{.Id}} {{.Created}}' | tee "$RUN_ROOT/docker-image.txt"

if [[ -e flops_analysis ]]; then
  mv flops_analysis "$RUN_ROOT/preexisting-flops_analysis"
fi
```

The `runs/` directory is ignored by Git. `git status --short` should be empty for a clean,
reproducible run. If it is not empty, record and understand the diff before proceeding.

### 7.2 M1 startup-report smoke

```bash
docker run --rm --gpus all \
  --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29501 \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash scripts/run_theoretical_flops_trace_8gpu_smoke.sh m1 \
  2>&1 | tee "$RUN_ROOT/m1.log"

M1_STATUS=${PIPESTATUS[0]}
printf '%s\n' "$M1_STATUS" | tee "$RUN_ROOT/m1-exit-status.txt"
test "$M1_STATUS" -eq 0
test -s flops_analysis/theoretical_flops.json
mv flops_analysis "$RUN_ROOT/m1"
```

M1 passes when the process exits zero, all 8 ranks complete, the log contains
`THEORETICAL FLOPS REPORT`, and `m1/theoretical_flops.json` is non-empty.

### 7.3 M1+M2 trace and reconciliation smoke

```bash
docker run --rm --gpus all \
  --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29502 \
  -v "$MEGATRON_REPO:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  "$MEGATRON_IMAGE" \
  bash scripts/run_theoretical_flops_trace_8gpu_smoke.sh m1m2 \
  2>&1 | tee "$RUN_ROOT/m1m2.log"

M1M2_STATUS=${PIPESTATUS[0]}
printf '%s\n' "$M1M2_STATUS" | tee "$RUN_ROOT/m1m2-exit-status.txt"
test "$M1M2_STATUS" -eq 0
test -s flops_analysis/theoretical_flops.json
test -s flops_analysis/torch_profile/rank-0.json.gz
test -s flops_analysis/reconciliation_rank0.json
gzip -t flops_analysis/torch_profile/rank-0.json.gz
mv flops_analysis "$RUN_ROOT/m1m2"
```

Do not delete or replace the `.json.gz` trace after reconciliation. It is the primary
backup used to investigate disagreements with simulator output.

### 7.4 Artifact self-check

```bash
find "$RUN_ROOT" -maxdepth 4 -type f -printf '%p\t%s bytes\n' | sort

docker run --rm \
  -v "$RUN_ROOT:/results:ro" \
  "$MEGATRON_IMAGE" \
  python -c 'import json, pathlib; root=pathlib.Path("/results/m1m2"); theory=json.loads((root/"theoretical_flops.json").read_text()); recon=json.loads((root/"reconciliation_rank0.json").read_text()); assert theory["computed_total_flops"] == theory["reference_total_flops"]; assert theory["relative_error"] == 0.0; assert (root/"torch_profile/rank-0.json.gz").stat().st_size > 0; print("theory entries:", len(theory["entries"])); print("TE backend:", theory["runtime_context"].get("te_selected_backend")); print("reconciliation keys:", sorted(recon))'

grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|TRACE RECONCILIATION|Traceback|CUDA out of memory|NCCL' \
  "$RUN_ROOT/m1.log" "$RUN_ROOT/m1m2.log"
```

Acceptance criteria for the first Dense Phase C run:

- Both Docker commands exit with status 0.
- `computed_total_flops == reference_total_flops` and `relative_error == 0.0`.
- `runtime_context.git_commit` equals `RUN_SHA` and `git_dirty` is false.
- The TE selected backend is captured, or its absence is explicitly explained from logs.
- `rank-0.json.gz` exists, is non-empty, and passes `gzip -t`.
- `reconciliation_rank0.json` exists and lists matched/unmatched events without parser
  failure. Non-zero unmatched counts are diagnostic data, not automatically a test
  failure, because TE fusion can hide one-to-one GEMM events.
- Logs contain no Python traceback, CUDA OOM, or primary NCCL failure.

The theory is an exact analytical math count for the implemented Dense model formula; it
is not a hardware counter. The reconciliation combines theory with observed trace shapes,
which is the intended comparison basis for the simulator.

## 8. Finish The Allocation And Report Results

From the `octave` session, print a compact result summary before exiting:

```bash
du -sh "$RUN_ROOT"
find "$RUN_ROOT" -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
exit
```

On `yes`, identify and cancel only the 8-GPU holder job:

```bash
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %R"
scancel <flops-dense-8gpu-job-id>
sacct -j <flops-dense-8gpu-job-id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
```

For feedback after a run, provide:

```text
Git SHA:
Docker image ID:
SLURM job ID and sacct line:
M1 exit status:
M1M2 exit status:
Artifact directory:
Trace size:
TE selected backend:
Reconciliation matched / unmatched counts:
First Python traceback, if any:
```

Do not paste the entire Chrome trace into chat. Keep it in the shared artifact directory
and provide its path and size.

## 9. Failure Routing

Use the first real failure, not later cascading NCCL errors.

| Symptom | First checks |
|---|---|
| Docker permission denied | Confirm allocation is `R`; open a new `ssh octave`; run `id` and inspect `/var/run/docker.sock` |
| `docker: command not found` | Confirm hostname is `octave`, not `yes` |
| Image pull/build fails | Preserve build output; check registry/network and disk with `docker system df`; do not prune without review |
| `uv` or package import mismatch | Confirm image tag and ID; rebuild from `docker/Dockerfile.ci.dev --target main` after lock/dependency changes |
| Fewer than 8 GPUs in container | Confirm the formal 8-GPU allocation is `R`; stop rather than train on an incomplete allocation |
| Python traceback on one rank, then NCCL errors | Diagnose the earliest Python traceback first |
| CUDA OOM | Record phase, rank, and peak memory; reduce smoke dimensions only as an explicit new experiment |
| Theory JSON exists but no trace | Confirm mode `m1m2` and all three profiler flags in the script |
| Trace exists but no reconciliation JSON | Inspect the first reconciliation/parser traceback after profiler stop |
| Many unmatched GEMMs | Check recorded shapes, TE fusion/backend, profile window, and count normalization before changing formulas |

Useful read-only cluster diagnostics:

```bash
scontrol show node octave
sinfo -p a100,long,octave
squeue -u "$USER"
df -h /home/liyixuan
du -sh /home/liyixuan/workspace/Megatron-LM/runs/theoretical-flops
docker system df
```

## 10. Next Development Order

After the first 4-layer Dense `m1m2` run passes:

1. Inspect the trace/reconciliation and fix only demonstrated Dense M1/M2 problems.
2. Re-run through the same WSL -> fork -> `yes` pull -> `octave` workflow.
3. Scale the Dense config from 4 to 28 layers in a separate committed script/config
   change and preserve a new artifact directory.
4. Compare Megatron theory + trace results with simulator Dense output.
5. Implement M3 items as separate increments. Add a dedicated MoE script only when the
   simulator MoE architecture is ready.

Do not start M3 merely to unblock the current Dense Phase C run; M3 is not a prerequisite
for validating the implemented Dense path.
