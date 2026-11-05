#!/bin/bash
# Chain the whole Level-1 run: prepare -> (YOLO, Mask R-CNN) in parallel. Run it from the kit folder:
#   cd <download>/benchmarks/level1 && bash slurm/submit_all.sh          # visible masks
#   MASKS=amodal bash slurm/submit_all.sh                                 # amodal masks
set -euo pipefail
cd "$(dirname "$0")/.."
export MASKS=${MASKS:-visible}
P=$(sbatch --parsable --export=ALL slurm/00_prepare.sbatch)
Y=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/10_yolo.sbatch)
M=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/20_maskrcnn.sbatch)
echo "submitted: prepare $P, yolo $Y, maskrcnn $M (MASKS=$MASKS)"
