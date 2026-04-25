#!/usr/bin/env bash
# Run the full improvement curve for one model class.
# Iterates over feature sets × snapshots, appends one row per call to JSONL.
#
# Usage:
#   ./scripts/run_improvement_curve.sh logistic
#   ./scripts/run_improvement_curve.sh multi-quantile
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-logistic}"
PYTHON="${PYTHON:-.venv/bin/python}"

FEATURE_SETS=(
  naive
  naive+pvi
  naive+pvi+demo
  naive+pvi+demo+fund
  naive+pvi+demo+fund+opp
  full
)
SNAPSHOTS=(T-110 T-60 T-20)

for fs in "${FEATURE_SETS[@]}"; do
  for snap in "${SNAPSHOTS[@]}"; do
    echo
    echo "==============================================================="
    echo "  model=$MODEL  features=$fs  snapshot=$snap"
    echo "==============================================================="
    PYTHONPATH=src "$PYTHON" -m oath_score.backtest \
      --features "$fs" \
      --model "$MODEL" \
      --train 2014 2016 2022 \
      --test 2024 \
      --snapshots "$snap" \
      --universe all \
      --bootstrap-reps 500
  done
done

echo
echo "Improvement curve for $MODEL complete."
