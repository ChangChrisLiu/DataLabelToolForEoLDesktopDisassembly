#!/usr/bin/env bash
# Submit every model x variant of the full evaluation (each an array over the four views).
#   PACK=$SCRATCH/vlmpack_2026-11-05 bash submit_matrix.sh
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
VARIANTS=(image blind)

for v in "${VARIANTS[@]}"; do
  for m in "${OPEN_MODELS[@]}"; do
    sbatch --export=ALL,PACK="$PACK",MODEL="$m",VARIANT="$v" "$HERE/open_model.sbatch"
  done
  for m in "${CLOSED_MODELS[@]}"; do
    sbatch --export=ALL,PACK="$PACK",MODEL="$m",VARIANT="$v" "$HERE/closed_model.sbatch"
  done
done
# the closed models' repeat subset, image variant only
for m in "${CLOSED_MODELS[@]}"; do
  sbatch --export=ALL,PACK="$PACK",MODEL="$m",VARIANT=image,REPEAT=3,SUBSET="$PACK/eval_v1/repeat.ids.txt",MODEL_TAG="${m//\//_}_repeat" \
    "$HERE/closed_model.sbatch"
done
