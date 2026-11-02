#!/usr/bin/env bash
# Every score file under $RUNS into one table: $RUNS/score.md (+ score.json). (W4-B)
#   PACK=... [RUNS=$PACK/runs] [TOOLS=$PACK/tools] [PYTHON=python] bash collect_scores.sh
set -euo pipefail
: "${PACK:?set PACK}"
RUNS="${RUNS:-$PACK/runs}"
TOOLS="${TOOLS:-$PACK/tools}"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$TOOLS/vlm_run/vlm_collect.py" table \
  --scores "$RUNS"/*/*/*/score/*.score.json \
  --out "$RUNS/score.md" --json "$RUNS/score.json"
echo "[collect_scores] $RUNS/score.md"
