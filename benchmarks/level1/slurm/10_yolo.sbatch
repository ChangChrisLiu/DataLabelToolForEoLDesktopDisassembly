#!/bin/bash
#SBATCH --job-name=l1_yolo
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu               # TODO(hprc): GPU partition name
#SBATCH --gres=gpu:a100:1             # TODO(hprc): GPU type / count
#SBATCH --output=l1_yolo.%j.log
##SBATCH --account=TODO(hprc)
# Baseline (a): Ultralytics YOLO segmentation. train -> predict (val, test) -> eval. Needs 00_prepare first.
# Settings (environment): MODEL (yolo11n-seg.pt), IMGSZ (1280), EPOCHS (100), BATCH (8), SEED (0),
#                         STAGES ("train predict eval"), RUN (run folder).
# A job that hit the time limit continues with:  sbatch --export=ALL,RESUME=1 slurm/10_yolo.sbatch
source "${KIT:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$KIT"
MODEL=${MODEL:-yolo11n-seg.pt}; IMGSZ=${IMGSZ:-1280}; EPOCHS=${EPOCHS:-100}; BATCH=${BATCH:-8}; SEED=${SEED:-0}
STAGES=${STAGES:-train predict eval}
RUN=${RUN:-$WORK/runs/yolo_${MODEL%.pt}_${MASKS}_${IMGSZ}_s${SEED}}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
if [[ " $STAGES " == *" train "* ]]; then
  $PYTHON train_yolo.py --data "$YDATA/data.yaml" --run "$RUN" --model "$MODEL" --imgsz "$IMGSZ" \
      --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --workers "$WORKERS" ${RESUME:+--resume}
fi
W="$RUN/train/weights/best.pt"; [[ -f "$W" ]] || W="$RUN/train/weights/last.pt"
for SPLIT in val test; do
  if [[ " $STAGES " == *" predict "* ]]; then
    $PYTHON predict_yolo.py --gt "$COCO/$SPLIT.json" --weights "$W" --cmap "$YDATA/category_map.json" \
        --out "$RUN/preds_$SPLIT.json" --imgsz "$IMGSZ"
  fi
  if [[ " $STAGES " == *" eval "* ]]; then
    $PYTHON eval_coco.py --gt "$COCO/$SPLIT.json" --pred "$RUN/preds_$SPLIT.json" --out "$RUN/eval_$SPLIT" \
        --title "YOLO $MODEL ($MASKS masks, imgsz $IMGSZ, seed $SEED) on $SPLIT"
  fi
done
echo "scores: $RUN/eval_test/metrics.md"
