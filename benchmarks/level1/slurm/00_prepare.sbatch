#!/bin/bash
#SBATCH --job-name=l1_prepare
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=l1_prepare.%j.log
##SBATCH --account=TODO(hprc)
# make_coco (train / val / test) -> evaluator self-test (must print SELFTEST PASSED) -> YOLO dataset. CPU only.
# Amodal masks:  sbatch --export=ALL,MASKS=amodal slurm/00_prepare.sbatch
source "${KIT:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$KIT"
$PYTHON make_coco.py --data-root "$DATA" --out "$COCO" --masks "$MASKS"
$PYTHON eval_coco.py --gt "$COCO/test.json" --selftest --out "$COCO/selftest"
$PYTHON coco_to_yolo.py --coco "$COCO" --out "$YDATA"
