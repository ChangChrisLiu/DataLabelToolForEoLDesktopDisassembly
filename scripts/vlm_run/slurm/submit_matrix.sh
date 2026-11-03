#!/usr/bin/env bash
# Submit every model x variant of the full evaluation (each an array over the export units).
#   PACK=$SCRATCH/vlmpack_2026-11-02 bash submit_matrix.sh
# Edit the two lists; MODEL_TAG defaults to the model id with / -> _.
set -euo pipefail
: "${PACK:?set PACK}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs

OPEN_MODELS=(
  "Qwen/Qwen3-VL-8B-Instruct"
  "OpenGVLab/InternVL3_5-8B"
)
CLOSED_MODELS=(
  # "<api model name 1>"
  # "<api model name 2>"
)
# the export units: four camera views (perception) and multi (the action tasks, every camera at once)
VIEWS=(scan oak1 oak2 rs multi)
ARRAY="--array=0-$(( ${#VIEWS[@]} - 1 ))"
export VIEW_LIST="${VIEWS[*]}"
VARIANTS=(image blind)          # overlay is meaningful for the scan view only (HANDOFF.md section 8.5)
# the closed models' repeat subset is not shipped: draw it first (HANDOFF.md section 8.5) and point REPEAT_IDS at it
REPEAT_IDS="${REPEAT_IDS:-$PACK/eval_v2/repeat.ids.txt}"
if [[ ${#CLOSED_MODELS[@]} -gt 0 && ! -f "$REPEAT_IDS" ]]; then
  echo "no repeat subset at $REPEAT_IDS: draw it first (HANDOFF.md section 8.5), then REPEAT_IDS=<file>" >&2; exit 2
fi

for v in "${VARIANTS[@]}"; do
  for m in "${OPEN_MODELS[@]}"; do
    sbatch "$ARRAY" --export=ALL,PACK="$PACK",MODEL="$m",VARIANT="$v" "$HERE/open_model.sbatch"
  done
  for m in "${CLOSED_MODELS[@]}"; do
    sbatch "$ARRAY" --export=ALL,PACK="$PACK",MODEL="$m",VARIANT="$v" "$HERE/closed_model.sbatch"
  done
done
# the closed models' repeat subset, image variant only
for m in "${CLOSED_MODELS[@]}"; do
  sbatch "$ARRAY" --export=ALL,PACK="$PACK",MODEL="$m",VARIANT=image,REPEAT=3,SUBSET="$REPEAT_IDS",MODEL_TAG="${m//\//_}_repeat" \
    "$HERE/closed_model.sbatch"
done
