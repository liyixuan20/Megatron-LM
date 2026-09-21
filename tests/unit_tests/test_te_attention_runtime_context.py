# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase A tests for TE attention runtime-context parsing."""

import enum
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_theoretical_flops_usage():
    spec = importlib.util.spec_from_file_location(
        "theoretical_flops_usage_for_te_test",
        Path("megatron/training/theoretical_flops_usage.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_THEORETICAL_FLOPS_USAGE = _load_theoretical_flops_usage()
TeAttentionLogCapture = _THEORETICAL_FLOPS_USAGE.TeAttentionLogCapture
build_runtime_context = _THEORETICAL_FLOPS_USAGE.build_runtime_context
parse_te_attention_runtime_context = (
    _THEORETICAL_FLOPS_USAGE.parse_te_attention_runtime_context
)
set_te_attention_debug_env_if_needed = _THEORETICAL_FLOPS_USAGE.set_te_attention_debug_env_if_needed
_stringify_arg_value = _THEORETICAL_FLOPS_USAGE._stringify_arg_value


def test_te_attention_runtime_context_parse():
    log_path = Path("tests/unit_tests/fixtures/theoretical_flops/te_dot_product_attention_debug.log")
    parsed = parse_te_attention_runtime_context(log_path.read_text(encoding="utf-8"))

    assert parsed["te_available_backends"] == (
        "{FlashAttention=True, FusedAttention=True (sub-backend 1), "
        "UnfusedDotProductAttention=True}"
    )
    assert parsed["te_selected_backend"] == "FusedAttention"
    assert parsed["te_selected_backend_version"] is None
    assert parsed["te_fused_sub_backend"] == 1


def test_te_attention_runtime_context_parse_flashattention_version():
    log_path = Path(
        "tests/unit_tests/fixtures/theoretical_flops/te_dot_product_attention_flash_debug.log"
    )
    parsed = parse_te_attention_runtime_context(log_path.read_text(encoding="utf-8"))

    assert parsed["te_selected_backend"] == "FlashAttention"
    assert parsed["te_selected_backend_version"] == "2.7.4.post1"
    assert parsed["te_fused_sub_backend"] is None
    assert "FlashAttention=True (2.7.4.post1)" in parsed["te_available_backends"]


def test_te_attention_log_capture_waits_for_selected_backend():
    capture = TeAttentionLogCapture()
    try:
        record = type("Record", (), {"getMessage": lambda self: (
            "DEBUG:DotProductAttention:Available backends = "
            "{FlashAttention=True (2.7.4.post1), FusedAttention=True (sub-backend 1)}"
        )})()
        capture.emit(record)
        assert capture.maybe_capture(capture_step=1) is None

        selected = type("Record", (), {"getMessage": lambda self: (
            "DEBUG:DotProductAttention:Selected backend = FlashAttention (2.7.4.post1)."
        )})()
        capture.emit(selected)
        parsed = capture.maybe_capture(capture_step=1)
        assert parsed is not None
        assert parsed["te_selected_backend"] == "FlashAttention"
        assert parsed["te_selected_backend_version"] == "2.7.4.post1"
        assert parsed["capture_step"] == 1
    finally:
        capture.close()


def test_stringify_attn_backend_enum_uses_name():
    class RealAttnBackend(enum.Enum):
        auto = 5

    assert _stringify_arg_value(RealAttnBackend.auto) == "auto"


def test_set_te_attention_debug_env_if_needed(monkeypatch):
    monkeypatch.delenv("NVTE_DEBUG", raising=False)
    monkeypatch.delenv("NVTE_DEBUG_LEVEL", raising=False)
    args = SimpleNamespace(
        report_theoretical_flops=True,
        capture_te_attention_backend=True,
        transformer_impl="transformer_engine",
        attention_backend="auto",
    )

    set_te_attention_debug_env_if_needed(args)
    context = build_runtime_context(args)

    assert context.nvte_debug == "1"
    assert context.nvte_debug_level == "2"
