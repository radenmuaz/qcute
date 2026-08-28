#!/bin/bash
# Pulls each active run's log.jsonl and regenerates its loss_curve.png. Meant to be run
# periodically (see the hourly Monitor loop) -- not part of the JAX training scripts themselves.
set -u
cd "$(dirname "$0")/.."

declare -a RUNS=(
  "tpu4:35.186.15.67:medium_paper_match_b8:gpt2_jax"
  "tpu5:35.186.33.7:summformer_medium_ablation:summformer_jax"
  "tpu6:35.186.110.50:small_paper_match:gpt2_jax"
  "tpu7:35.186.34.230:summformer_small_ablation:summformer_jax"
)

for entry in "${RUNS[@]}"; do
  IFS=: read -r name ip run module <<< "$entry"
  mkdir -p "logs/$run"
  scp -o ControlPath="$HOME/.ssh/controlmasters/${name}-%r@%h:%p" \
    "muaz@${ip}:~/qcute/${module}/logs/${run}/log.jsonl" "logs/${run}/log.jsonl" 2>/dev/null
  if [ -f "logs/${run}/log.jsonl" ]; then
    .venv/bin/python3 summformer_jax/lm/scripts/plot_jax_run.py "logs/${run}"
  else
    echo "WARN: failed to pull logs/${run}/log.jsonl"
  fi
done
