#!/usr/bin/env bash
# Run inside the Megatron CI container. Prints PATH, argparse flag presence,
# and the real pretrain_gpt.py --help stdout/stderr.

set -u

echo "PATH=${PATH}"
echo "which python: $(command -v python || true)"
python -V

echo "==== argparse flag check ===="
set +e
python - <<'PY'
import argparse
from megatron.training.arguments import add_megatron_arguments

parser = argparse.ArgumentParser(add_help=False)
add_megatron_arguments(parser)
flags = {opt for action in parser._actions for opt in action.option_strings}
print("has_flag", "--report-theoretical-flops" in flags)
PY
echo "argparse_exit=$?"

echo "==== pretrain_gpt.py --help ===="
python pretrain_gpt.py --help > /tmp/help.out 2> /tmp/help.err
echo "help_exit=$?"
echo "==== stderr ===="
cat /tmp/help.err
echo "==== grep flag in stdout ===="
grep -n -- "--report-theoretical-flops" /tmp/help.out || echo "FLAG_NOT_IN_STDOUT"
echo "==== stdout tail ===="
tail -n 40 /tmp/help.out
