#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Collect theoretical-FLOPs run artifacts into a stable metrics layout.

Reads training logs and optional ``theoretical_flops.json`` files under a run
root, then writes:

- ``metrics/throughput.json``
- ``metrics/te_attention_backend.json``
- ``metrics/timers.json``
- ``run_manifest.json``
- ``artifact-index.txt``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PARSE_TE_ATTENTION_RUNTIME_CONTEXT = None


ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
    r"(?:.*?throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+))?",
    re.IGNORECASE,
)
TIMER_LINE_RE = re.compile(
    r"(forward-backward|forward-compute|backward-compute|all-grads-sync|"
    r"optimizer|params-all-gather|forward-recv|forward-send|"
    r"backward-recv|backward-send)\s*[:=]\s*([0-9.]+)"
)
NSYS_NAME_RE = re.compile(r"\.(nsys-rep|qdrep|sqlite)$")


def parse_te_attention_backend_from_text(log_text: str) -> dict[str, Any]:
    """Parse TE DotProductAttention backend lines from a training log."""

    parsed = _parse_te_attention_runtime_context()(log_text)
    selected = parsed.get("te_selected_backend")
    version = parsed.get("te_selected_backend_version")
    return {
        "te_available_backends": parsed.get("te_available_backends"),
        "te_selected_backend": selected,
        "te_selected_backend_version": version,
        "te_fused_sub_backend": parsed.get("te_fused_sub_backend"),
        "te_selected_backend_display": _display_backend(selected, version),
        "source": "log",
    }


def parse_throughput_from_text(log_text: str) -> list[dict[str, Any]]:
    """Extract per-iteration elapsed time and optional throughput."""

    iterations = []
    for match in ITERATION_RE.finditer(log_text):
        throughput = match.group(4)
        iterations.append(
            {
                "iteration": int(match.group(1)),
                "train_iters": int(match.group(2)),
                "elapsed_ms": float(match.group(3)),
                "throughput_tflops_per_gpu": float(throughput) if throughput else None,
            }
        )
    return iterations


def parse_timers_from_text(log_text: str) -> list[dict[str, Any]]:
    """Extract Megatron timer samples that appear in stdout."""

    samples = []
    for line in log_text.splitlines():
        matches = TIMER_LINE_RE.findall(line)
        if not matches:
            continue
        samples.append({name: float(value) for name, value in matches})
    return samples


def collect_run_artifacts(
    run_root: str | Path,
    extra_logs: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Write structured metrics under ``run_root`` and return the manifest."""

    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_paths = _discover_logs(root, extra_logs)
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in log_paths)
    theory_paths = sorted(root.rglob("theoretical_flops.json"))
    trace_paths = sorted(root.rglob("rank-*.json.gz"))
    recon_paths = sorted(root.rglob("reconciliation_rank*.json"))
    nsys_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and NSYS_NAME_RE.search(path.name)
    )

    throughput_iterations = parse_throughput_from_text(log_text)
    timers = parse_timers_from_text(log_text)
    te_backend = _resolve_te_backend(log_text, theory_paths)
    throughput_payload = {
        "iterations": throughput_iterations,
        "recommended_throughput_tflops_per_gpu": _recommended_throughput(throughput_iterations),
        "log_files": [str(path.relative_to(root)) for path in log_paths],
    }
    timer_payload = {
        "samples": timers,
        "log_files": [str(path.relative_to(root)) for path in log_paths],
    }

    _write_json(metrics_dir / "throughput.json", throughput_payload)
    _write_json(metrics_dir / "te_attention_backend.json", te_backend)
    _write_json(metrics_dir / "timers.json", timer_payload)

    manifest = {
        "run_root": str(root),
        "logs": [str(path.relative_to(root)) for path in log_paths],
        "artifacts": {
            "theoretical_flops": [str(path.relative_to(root)) for path in theory_paths],
            "chrome_traces": [str(path.relative_to(root)) for path in trace_paths],
            "reconciliation": [str(path.relative_to(root)) for path in recon_paths],
            "nsys": [str(path.relative_to(root)) for path in nsys_paths],
            "metrics": [
                "metrics/throughput.json",
                "metrics/te_attention_backend.json",
                "metrics/timers.json",
            ],
        },
        "te_attention_backend": te_backend,
        "throughput_summary": {
            "num_iterations": len(throughput_iterations),
            "recommended_throughput_tflops_per_gpu": throughput_payload[
                "recommended_throughput_tflops_per_gpu"
            ],
        },
        "timer_sample_count": len(timers),
    }
    _write_json(root / "run_manifest.json", manifest)
    _write_artifact_index(root)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Run directory to index.")
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Extra log file to parse. May be repeated.",
    )
    args = parser.parse_args(argv)
    manifest = collect_run_artifacts(args.run_root, extra_logs=args.log)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _discover_logs(root: Path, extra_logs: Iterable[str | Path] | None) -> list[Path]:
    discovered = [path for path in root.rglob("*") if path.is_file() and path.suffix == ".log"]
    extra = [Path(path).resolve() for path in extra_logs or [] if Path(path).is_file()]
    unique: dict[Path, None] = {}
    for path in [p.resolve() for p in discovered] + extra:
        unique[path] = None
    return [path for path in unique]


def _resolve_te_backend(log_text: str, theory_paths: list[Path]) -> dict[str, Any]:
    from_log = parse_te_attention_backend_from_text(log_text)
    from_json: dict[str, Any] = {}
    for path in theory_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime_context") or {}
        if runtime.get("te_selected_backend"):
            from_json = {
                "te_available_backends": runtime.get("te_available_backends"),
                "te_selected_backend": runtime.get("te_selected_backend"),
                "te_selected_backend_version": runtime.get("te_selected_backend_version"),
                "te_fused_sub_backend": runtime.get("te_fused_sub_backend"),
                "te_selected_backend_display": _display_backend(
                    runtime.get("te_selected_backend"),
                    runtime.get("te_selected_backend_version"),
                ),
                "attention_backend_cli": runtime.get("attention_backend_cli"),
                "source": str(path),
            }
            break
    if from_json.get("te_selected_backend"):
        if not from_json.get("te_selected_backend_version") and from_log.get(
            "te_selected_backend_version"
        ):
            from_json["te_selected_backend_version"] = from_log["te_selected_backend_version"]
            from_json["te_selected_backend_display"] = _display_backend(
                from_json.get("te_selected_backend"),
                from_json.get("te_selected_backend_version"),
            )
        return from_json
    return from_log


def _recommended_throughput(iterations: list[dict[str, Any]]) -> float | None:
    measured = [
        item["throughput_tflops_per_gpu"]
        for item in iterations
        if item.get("throughput_tflops_per_gpu") is not None
    ]
    if not measured:
        return None
    if len(measured) == 1:
        return measured[0]
    # Skip the first warmup sample when a later steady-state step exists.
    return measured[-1]


def _parse_te_attention_runtime_context():
    global _PARSE_TE_ATTENTION_RUNTIME_CONTEXT
    if _PARSE_TE_ATTENTION_RUNTIME_CONTEXT is not None:
        return _PARSE_TE_ATTENTION_RUNTIME_CONTEXT

    usage_path = (
        Path(__file__).resolve().parents[1] / "megatron" / "training" / "theoretical_flops_usage.py"
    )
    module_name = "theoretical_flops_usage_for_collect"
    spec = importlib.util.spec_from_file_location(module_name, usage_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load theoretical FLOPs usage from {usage_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _PARSE_TE_ATTENTION_RUNTIME_CONTEXT = module.parse_te_attention_runtime_context
    return _PARSE_TE_ATTENTION_RUNTIME_CONTEXT


def _display_backend(selected: str | None, version: str | None) -> str | None:
    if selected is None:
        return None
    if version:
        return f"{selected} ({version})"
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_index(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            lines.append(f"{path.relative_to(root)}\t{path.stat().st_size} bytes")
    (root / "artifact-index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
