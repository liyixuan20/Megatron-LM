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
5. Structured metrics (`throughput`, TE FlashAttention/FusedAttention backend, timers).
6. Optional nsys report for layer/chunk compute+comm (separate 8-GPU job).

Dense and MoE are separate tracks. Do not add MoE flags to the Dense smoke script.

| Work item | Status | Evidence / next action |
|---|---|---|
| M1 Dense theoretical report | Implemented | Commit `06a5f407f` |
| M2 trace export and reconciliation | Implemented | Commit `06a5f407f` |
| Offline/lightweight Phase A fixes | Implemented | Commit `5defc03ad` |
| M1/M2 Phase A targeted pytest | Passed previously on server | Re-run in the Docker image before Phase C |
| 1-GPU Docker probe | Passed | Job `316935`, `DOCKER_PROBE_OK` |
| 8-GPU Phase C, job `316937` | Failed in 44s | Smoke `--help` preflight; training never started |
| 1-GPU `--help` debug, job `316981` | Completed | `KeyError: getpwuid(): uid not found: 18107` while importing TE/torch inductor. Container `--user` has no `/etc/passwd` entry. Fix: set `USER`/`LOGNAME` in `scripts/theoretical_flops_slurm_common.sh` |
| M1/M2 Phase C, 8 A100 | Passed as smoke | Job `316986`; Chrome reconciliation coverage is still low |
| Artifact collector | Implemented | `scripts/collect_theoretical_flops_artifacts.py` writes `metrics/` + `run_manifest.json` |
| TE FlashAttention backend capture | Code fix landed | Parser now waits for `Selected backend` and records the FA version string |
| nsys 8-GPU layer/chunk trace | Scripts ready | `scripts/run_theoretical_flops_trace_nsys_slurm.slurm`; not yet run on octave |
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

**Unattended runs must use `sbatch`, not `srun sleep infinity`.** Holder jobs expire
when `--time` elapses if you go offline. Batch processes on JA do not start in the
`docker` group; every wrapper re-execs under `sg docker`. Interactive `ssh octave`
after an allocation still works for debugging, but it is not the Phase C path.

### 4.1 Queued sbatch catalog (preferred)

Submit from `yes` in `/home/liyixuan/workspace/Megatron-LM`. Create `logs/` first.
The 8-GPU wrapper refuses a dirty worktree; commit before Phase C.

| Job | Script | GPUs | Time | What it proves |
|---|---|---:|---|---|
| Docker probe | `scripts/probe_theoretical_flops_docker_slurm.slurm` | 1 | 10 min | `sg docker`, image present, `import torch` |
| `--help` debug | `scripts/debug_theoretical_flops_help_slurm.slurm` | 1 | 15 min | argparse flag + real `pretrain_gpt.py --help` stderr |
| Image + Phase A | `scripts/prepare_theoretical_flops_image_slurm.slurm` | 1 | 60 min | Build/reuse `megatron-lm:theoretical-flops-dev`, CPU pytest |
| Dense Phase C | `scripts/run_theoretical_flops_trace_slurm.slurm` | **8** | 60 min | default `m1m2` only; never pulls/builds the image |
| Dense nsys (M4) | `scripts/run_theoretical_flops_trace_nsys_slurm.slurm` | **8** | 60 min | NVTX + nsys; no PyTorch profiler; writes `metrics/` |

```bash
cd /home/liyixuan/workspace/Megatron-LM
mkdir -p logs
git status --short
git log -1 --oneline
squeue -u "$USER" -o "%.18i %.16j %.9P %.8T %.10M %.6D %R"

# 1) Docker-in-batch (skip if a recent probe already printed DOCKER_PROBE_OK)
PROBE_JOB_ID=$(sbatch --parsable scripts/probe_theoretical_flops_docker_slurm.slurm)
echo "PROBE_JOB_ID=${PROBE_JOB_ID}"

# 2) Optional: argparse / --help with stderr visible (1 GPU, not 8)
DEBUG_JOB_ID=$(sbatch --parsable scripts/debug_theoretical_flops_help_slurm.slurm)
echo "DEBUG_JOB_ID=${DEBUG_JOB_ID}"

# 3) Only if the image is missing or docker/uv.lock changed
# PREP_JOB_ID=$(sbatch --parsable scripts/prepare_theoretical_flops_image_slurm.slurm)
# sbatch --export=ALL,FORCE_IMAGE_BUILD=1 \
#   scripts/prepare_theoretical_flops_image_slurm.slurm

# 4) Phase C. Queue behind a successful 1-GPU debug so a failed import
#    cancels the 8-GPU job instead of occupying octave after hours of PD.
sbatch --dependency=afterok:${DEBUG_JOB_ID} \
  scripts/run_theoretical_flops_trace_slurm.slurm

# 5) After Chrome smoke is understood: layer/chunk nsys (do not combine with m1m2)
# Confirm nsys in the 1-GPU probe output first.
sbatch scripts/run_theoretical_flops_trace_nsys_slurm.slurm
```

`sbatch` without `--dependency` is fine if the 1-GPU jobs already completed
successfully and the worktree SHA matches the intended run.

Do **not** run `scripts/run_theoretical_flops_trace_8gpu_smoke.sh` under a 1-GPU
allocation. It is hard-coded to `--nproc-per-node 8`. `docker run --gpus all`
can also ignore SLURM's `CUDA_VISIBLE_DEVICES` (job `316935` allocated 1 GPU
and still saw `torch.cuda.device_count() == 8` inside the container).

### 4.2 Inspecting job output

From `yes` (NFS, so `octave` logs are visible immediately):

```bash
cd /home/liyixuan/workspace/Megatron-LM

# Queue / completion
squeue -u "$USER" -o "%.18i %.16j %.9P %.8T %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P

# SLURM stdout/stderr. nvidia-smi -q makes 8-GPU .out files huge; use tail.
tail -n 80 logs/flops-docker-probe-<jobid>.out
tail -n 80 logs/flops-help-debug-<jobid>.out
tail -n 80 logs/flops-dense-8gpu-<jobid>.out

# Phase C artifacts (created only after m1 starts writing JSON)
ls -ld runs/theoretical-flops/*
ls -la runs/theoretical-flops/<tag>/
cat runs/theoretical-flops/<tag>/m1-exit-status.txt
grep -E 'THEORETICAL FLOPS REPORT|TE ATTENTION BACKEND|TRACE RECONCILIATION|Traceback|getpwuid|preflight_ok' \
  runs/theoretical-flops/<tag>/m1.log \
  runs/theoretical-flops/<tag>/m1m2.log \
  logs/flops-dense-8gpu-<jobid>.out
```

A job leaving `squeue` is not success. Trust `sacct` `COMPLETED` / `0:0` plus
the artifacts above. Job `316937` was `FAILED` / `1:0` after 44s with no
`theoretical_flops.json`.

| File | Success marker |
|---|---|
| `logs/flops-docker-probe-*.out` | `DOCKER_PROBE_OK` |
| `logs/flops-help-debug-*.out` | `has_flag True`, `help_exit=0` |
| `logs/flops-image-prepare-*.out` | `PREP_ROOT=...` and pytest passed |
| `logs/flops-dense-8gpu-*.out` | `RUN_ROOT=...` and both smokes exit 0 |
| `runs/theoretical-flops/<tag>/m1/theoretical_flops.json` | non-empty |
| `runs/theoretical-flops/<tag>/m1m2/torch_profile/rank-0.json.gz` | `gzip -t` passes |
| `runs/theoretical-flops/<tag>/m1m2/reconciliation_rank0.json` | parser succeeded |
| `runs/theoretical-flops/<tag>/metrics/te_attention_backend.json` | `te_selected_backend` is non-null (expect `FlashAttention` + version on A100) |
| `runs/theoretical-flops/<tag>/run_manifest.json` | collector finished |
| `runs/theoretical-flops/<tag>-nsys/artifacts/flops_analysis/nsys/*.nsys-rep` | nsys job produced a report |

### 4.3 Optional interactive holders (legacy)

Use this only when you are at the keyboard and need an interactive `ssh octave`
shell. It is not the Phase C path.

```bash
srun -A a100 -p a100 \
  --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 --mem=96G \
  --time=01:00:00 \
  --job-name=flops-docker-access \
  sleep infinity
```

Then from a second terminal:

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

Expected: `hostname` is `octave`, `id` includes `docker`, `docker version`
shows client and server, NVIDIA runtime is present. If the socket is still
denied, record those outputs and ask the administrator. Do not use `sudo`.

Cancel the 1-GPU holder before requesting 8 GPUs interactively:

```bash
scancel <flops-docker-access-job-id>
srun -A a100 -p a100 \
  --nodes=1 --ntasks=1 \
  --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=180G \
  --time=01:00:00 \
  --job-name=flops-dense-8gpu \
  sleep infinity
```

Wait until `squeue` shows `R` on `octave`, then `ssh octave` and `nvidia-smi`.
Do not `docker run --gpus all` on the 1-GPU holder. `PD` for an 8-GPU job is
expected while octave is busy.

## 5. Build The Reproducible Docker Image

Use the repository CI image, not a hand-maintained Conda environment. The image pin is
read from `docker/.ngc_version.dev`; the current pin is
`nvcr.io/nvidia/pytorch:26.04-py3`. The CI image contains `uv`, uses `/opt/venv`, and pins
the CUDA/PyTorch/Transformer Engine dependency set expected by the checkout.

The image only needs rebuilding when `docker/`, `pyproject.toml`, or `uv.lock` changes.
Ordinary Python/shell source updates are visible through the bind mount and do not need an
image rebuild.

For queued operation, **do not use `srun sleep infinity`**. Submit the wrappers in
§4.1. The image is already on `octave` from the earlier build; skip the 1-GPU
prep job unless the image is missing or `docker/` / `uv.lock` changed.

`ALLOW_UNVERIFIED_IMAGE=1` is a diagnostic escape hatch and should not be used
for acceptance runs. The 8-GPU job never pulls or builds images.

`Dockerfile.ci.dev` copies `assets/`; the public clone does not contain that
directory. The prep script runs `mkdir -p assets` before `docker build`. Do not
omit `--target main`. Do not increase compiler parallelism.

On `octave`, inside an active allocation:

```bash
cd /home/liyixuan/workspace/Megatron-LM
MEGATRON_IMAGE=megatron-lm:theoretical-flops-dev
MEGATRON_BASE_IMAGE=$(<docker/.ngc_version.dev)
IMAGE_INPUT_SHA=$(git ls-files -s \
  assets docker README.md pyproject.toml uv.lock \
  megatron/core/__init__.py megatron/core/package_info.py \
  | sha256sum | awk '{print $1}')

docker pull "$MEGATRON_BASE_IMAGE"
docker build \
  --target main \
  --build-arg FROM_IMAGE_NAME="$MEGATRON_BASE_IMAGE" \
  --build-arg IMAGE_TYPE=dev \
  --label "org.megatron.build-input-sha=$IMAGE_INPUT_SHA" \
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

Interactive `docker run` as root does not hit the `getpwuid` bug. Batch wrappers
run `--user $(id -u)` and **must** set `USER`/`LOGNAME` via
`set_host_user_docker_opts` (job `316981`).

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

The queued SLURM wrappers are listed in §4.1. Phase C default is **m1m2 only**:

```text
scripts/run_theoretical_flops_trace_slurm.slurm      # 8 GPUs: m1m2
# optional: SMOKE_MODE=both to also run the short M1-only job
```

It has two modes:

| Mode | Iterations | Result |
|---|---:|---|
| `m1` | 2 | Startup theory report only |
| `m1m2` (default) | 8 | Theory + profiler trace + reconciliation |

The current script profiles rank 0 only, over step `[4, 5)`. Warmup is steps 1–3;
steps 5–8 are leftover steady-state samples for throughput. `--log-throughput` and
`--timing-log-level 1` are on for both modes. `rank-0.json.gz` is the complete Chrome
trace for that profiled rank/window, not a trace for all eight ranks. Set
`PROFILE_RANKS="0 1"` to export more ranks.

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

### 7.5 Artifact collection

After a smoke or nsys job, structured metrics are written by
`scripts/collect_theoretical_flops_artifacts.py`:

```text
runs/theoretical-flops/<tag>/
├── run_manifest.json
├── metrics/
│   ├── throughput.json              # per-iter ms and TFLOP/s/GPU
│   ├── te_attention_backend.json    # FlashAttention / FusedAttention + version
│   └── timers.json                  # timing_log_level samples, if present
├── m1/ or m1m2/                     # Chrome smoke
└── artifacts/flops_analysis/        # nsys job: theory JSON + nsys/*.nsys-rep
```

Re-run collection on an existing directory without occupying GPUs:

```bash
python3 scripts/collect_theoretical_flops_artifacts.py \
  --run-root runs/theoretical-flops/20260921-052755-3daf37427ee0-316986 \
  --log runs/theoretical-flops/20260921-052755-3daf37427ee0-316986/m1m2.log
```

`te_attention_backend.json` is the place to confirm which TE attention kernel ran.
On the A100 smoke, logs showed `Selected backend = FlashAttention (2.7.4.post1)`.
That version string is now parsed into `te_selected_backend_version` instead of
being dropped.

## 8. Finish The Allocation And Report Results

If Phase C was launched with `sbatch`, there is no holder to cancel; the job
exits on its own. Inspect with the commands in §4.2.

If you used a legacy `srun sleep infinity` holder, cancel only that job:

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
| Docker permission denied | Batch jobs lack the docker group until `sg docker`; check `logs/*.err` for the sg re-exec. Confirm `id` inside the job includes docker after sg. Do not use `srun sleep` holders. |
| `docker: command not found` | Confirm hostname is `octave`, not `yes` |
| `getpwuid(): uid not found` | Container `--user` UID is missing from image `/etc/passwd`. Torch inductor needs `USER`/`LOGNAME` (see `set_host_user_docker_opts`). Job `316981`. |
| `pretrain_gpt.py --help does not contain --report-theoretical-flops` | Often a swallowed import error (was `getpwuid`) or `pipefail`+`grep -q` SIGPIPE. Do not treat as "flag missing" until stderr is visible. |
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
2. Confirm TE `Selected backend` + FlashAttention version in `metrics/te_attention_backend.json`.
3. Run the nsys job (`scripts/run_theoretical_flops_trace_nsys_slurm.slurm`) for
   layer/chunk compute+comm. Do not add nsys flags to the Chrome smoke.
4. Re-run through the same WSL -> fork -> `yes` pull -> `octave` workflow.
5. Scale the Dense config from 4 to 28 layers in a separate committed script/config
   change and preserve a new artifact directory.
6. Compare Megatron theory + trace results with simulator Dense output.
7. Implement M3 items as separate increments. Add a dedicated MoE script only when the
   simulator MoE architecture is ready.

Do not start M3 merely to unblock the current Dense Phase C run; M3 is not a prerequisite
for validating the implemented Dense path.
