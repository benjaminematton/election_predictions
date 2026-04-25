#!/usr/bin/env bash
# Build features for every (cycle, snapshot) combination.
# Idempotent — re-running skips work that's already on disk (FEC aggregation
# rewrites are cheap; features.py is fast once parquets are cached).
#
# Usage:
#   set -a; source .env; set +a
#   ./scripts/build_all_features.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RAW_DIR=data/raw
PROC_DIR=data/processed

CYCLES=(2014 2016 2022 2024)
SNAPSHOTS=(T-110 T-60 T-20)

snapshot_date() {
  PYTHONPATH=src "$PYTHON" -c "from oath_score.constants import snapshot_date_for; print(snapshot_date_for($1, '$2'))"
}

for cycle in "${CYCLES[@]}"; do
  for snap in "${SNAPSHOTS[@]}"; do
    snap_date=$(snapshot_date "$cycle" "$snap")
    echo
    echo "==============================================================="
    echo "  cycle=$cycle  snapshot=$snap  ($snap_date)"
    echo "==============================================================="

    fec_parquet="$PROC_DIR/fec_${cycle}_${snap_date}.parquet"
    if [ ! -f "$fec_parquet" ]; then
      echo "[fec] aggregating contributions through $snap_date"
      PYTHONPATH=src "$PYTHON" -m oath_score.ingest.fec \
        --cycle "$cycle" --snapshot "$snap_date" \
        --raw-dir "$RAW_DIR" --out-dir "$PROC_DIR" >/dev/null
    else
      echo "[fec] already built: $fec_parquet"
    fi

    # fec_ie has no parquet artifact; features.py calls it directly.
    # The Schedule E source CSV is already staged per cycle.

    cand_parquet="$PROC_DIR/candidates_${cycle}_${snap}.parquet"
    if [ ! -f "$cand_parquet" ]; then
      echo "[features] joining sources for $snap"
      PYTHONPATH=src "$PYTHON" -m oath_score.features \
        --cycle "$cycle" --snapshot "$snap" \
        --raw-dir "$RAW_DIR" --out-dir "$PROC_DIR" 2>&1 | tail -3
    else
      echo "[features] already built: $cand_parquet"
    fi
  done
done

echo
echo "==============================================================="
echo "  DONE. Parquets:"
ls -1 "$PROC_DIR"/candidates_*.parquet 2>/dev/null | sort
