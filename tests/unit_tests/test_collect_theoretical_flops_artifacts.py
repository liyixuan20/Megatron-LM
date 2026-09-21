# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase A tests for theoretical-FLOPs artifact collection."""

import importlib.util
import json
import sys
from pathlib import Path


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_theoretical_flops_artifacts_for_test",
        Path("scripts/collect_theoretical_flops_artifacts.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_COLLECTOR = _load_collector()
collect_run_artifacts = _COLLECTOR.collect_run_artifacts
parse_te_attention_backend_from_text = _COLLECTOR.parse_te_attention_backend_from_text
parse_throughput_from_text = _COLLECTOR.parse_throughput_from_text
parse_timers_from_text = _COLLECTOR.parse_timers_from_text


SAMPLE_LOG = """
DEBUG:DotProductAttention:Available backends = {FlashAttention=True (2.7.4.post1), FusedAttention=True (sub-backend 1), UnfusedDotProductAttention=True}
DEBUG:DotProductAttention:Selected backend = FlashAttention (2.7.4.post1).
 [2026-09-20 21:31:05] iteration        1/       6 | elapsed time per iteration (ms): 4802.3 | throughput per GPU (TFLOP/s/GPU): 3.1 |
 [2026-09-20 21:31:06] iteration        3/       6 | elapsed time per iteration (ms): 291.2 | throughput per GPU (TFLOP/s/GPU): 50.7 |
 [2026-09-20 21:31:07] iteration        6/       6 | elapsed time per iteration (ms): 286.6 | throughput per GPU (TFLOP/s/GPU): 51.5 |
(min, max) time across ranks (ms):
    forward-backward ................................: (210.40, 221.30)
    optimizer .......................................: (40.10, 41.20)
"""


def test_parse_te_attention_backend_from_flash_log():
    parsed = parse_te_attention_backend_from_text(SAMPLE_LOG)
    assert parsed["te_selected_backend"] == "FlashAttention"
    assert parsed["te_selected_backend_version"] == "2.7.4.post1"
    assert parsed["te_selected_backend_display"] == "FlashAttention (2.7.4.post1)"


def test_parse_throughput_prefers_later_steady_state():
    iterations = parse_throughput_from_text(SAMPLE_LOG)
    assert [item["iteration"] for item in iterations] == [1, 3, 6]
    assert iterations[-1]["throughput_tflops_per_gpu"] == 51.5


def test_parse_real_minmax_timer_format():
    samples = parse_timers_from_text(SAMPLE_LOG)
    assert samples
    assert samples[0]["forward-backward"] == {"min_ms": 210.4, "max_ms": 221.3}
    assert samples[0]["optimizer"] == {"min_ms": 40.1, "max_ms": 41.2}


def test_collect_run_artifacts_writes_metrics(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "logs").mkdir(parents=True)
    (run_root / "logs" / "train.log").write_text(SAMPLE_LOG, encoding="utf-8")
    theory = {
        "runtime_context": {
            "attention_backend_cli": "auto",
            "te_selected_backend": None,
            "te_selected_backend_version": None,
        }
    }
    (run_root / "artifacts").mkdir()
    (run_root / "artifacts" / "theoretical_flops.json").write_text(
        json.dumps(theory), encoding="utf-8"
    )
    (run_root / "artifacts" / "nsys").mkdir()
    (run_root / "artifacts" / "nsys" / "megatron-dense-gqa.nsys-rep").write_bytes(b"nsys")
    (run_root / "artifacts" / "reconciliation_rank0.json").write_text(
        json.dumps({"flops_budget": {}}),
        encoding="utf-8",
    )

    manifest = collect_run_artifacts(run_root)

    assert (run_root / "metrics" / "throughput.json").is_file()
    assert (run_root / "metrics" / "te_attention_backend.json").is_file()
    assert (run_root / "run_manifest.json").is_file()
    assert (run_root / "artifact-index.txt").is_file()
    assert manifest["te_attention_backend"]["te_selected_backend"] == "FlashAttention"
    assert manifest["te_attention_backend"]["te_selected_backend_version"] == "2.7.4.post1"
    assert manifest["throughput_summary"]["recommended_throughput_tflops_per_gpu"] == 51.5
    assert manifest["artifacts"]["nsys"] == ["artifacts/nsys/megatron-dense-gqa.nsys-rep"]
    recon = json.loads((run_root / "artifacts" / "reconciliation_rank0.json").read_text())
    assert recon["flops_budget"]["measured_throughput_tflops_per_gpu"] == 51.5
