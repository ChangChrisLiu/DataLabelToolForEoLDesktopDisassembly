#!/bin/bash
# Run once where there is internet (e.g. a login node) so that offline compute nodes find the weights:
#   cd <download>/benchmarks/level1 && bash slurm/fetch_weights.sh
cd "$(dirname "$0")/.."
source slurm/env.sh
"$PYTHON" fetch_weights.py --yolo-dir "$L1_WEIGHTS_DIR"
