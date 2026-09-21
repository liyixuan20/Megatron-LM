#!/usr/bin/env bash
# Shared dense GQA flags for theoretical-FLOPs smoke and nsys jobs.
# Sourced by the launch scripts; not submitted on its own.

theoretical_flops_preflight_report_flag() {
  if ! command -v python >/dev/null 2>&1; then
    echo "python is not on PATH. PATH=${PATH}" >&2
    return 1
  fi
  python - <<'PY'
import argparse
import sys

from megatron.training.arguments import add_megatron_arguments

parser = argparse.ArgumentParser(add_help=False)
add_megatron_arguments(parser)
flags = {opt for action in parser._actions for opt in action.option_strings}
if "--report-theoretical-flops" not in flags:
    sys.stderr.write(
        "argparse is missing --report-theoretical-flops; check the synced commit.\n"
    )
    sys.exit(1)
print("preflight_ok --report-theoretical-flops")
PY
}

theoretical_flops_dense_pretrain_args() {
  local output_dir="${THEORETICAL_FLOPS_OUTPUT_DIR:-./flops_analysis}"
  THEORETICAL_FLOPS_PRETRAIN_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --report-theoretical-flops
    --theoretical-flops-output-dir "${output_dir}"
    --tensorboard-dir "${output_dir}/tensorboard"
    --tensor-model-parallel-size "${TP_SIZE:-1}"
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --num-layers "${NUM_LAYERS:-4}"
    --hidden-size "${HIDDEN_SIZE:-2048}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE:-6144}"
    --num-attention-heads "${NUM_ATTENTION_HEADS:-16}"
    --group-query-attention
    --num-query-groups "${NUM_QUERY_GROUPS:-8}"
    --kv-channels "${KV_CHANNELS:-128}"
    --seq-length "${SEQ_LENGTH:-4096}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH:-4096}}"
    --position-embedding-type rope
    --swiglu
    --normalization RMSNorm
    --disable-bias-linear
    --micro-batch-size "${MICRO_BATCH_SIZE:-2}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size "${VOCAB_SIZE:-32000}"
    --bf16
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
  )
}
