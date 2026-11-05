#!/bin/bash
#SBATCH --job-name=l1_maskrcnn
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu               # TODO(hprc): GPU partition name
#SBATCH --gres=gpu:a100:1             # TODO(hprc): GPU type / count
#SBATCH --output=l1_maskrcnn.%j.log
##SBATCH --account=TODO(hprc)
# Baseline (b): torchvision Mask R-CNN R50-FPN (v2, COCO weights). train -> predict (val, test) -> eval.
# Settings (environment): SIZE (1280), EPOCHS (24), BATCH (4), LR (0.02 at batch 16, scaled linearly),
#                         SEED (0), ARCH (v2 | v1), STAGES ("train predict eval"), RUN (run folder).
# A job that hit the time limit continues with:  sbatch --export=ALL,RESUME=1 slurm/20_maskrcnn.sbatch
source "${KIT:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$KIT"
SIZE=${SIZE:-1280}; EPOCHS=${EPOCHS:-24}; BATCH=${BATCH:-4}; LR=${LR:-0.02}; SEED=${SEED:-0}; ARCH=${ARCH:-v2}
STAGES=${STAGES:-train predict eval}
RUN=${RUN:-$WORK/runs/maskrcnn_${ARCH}_${MASKS}_${SIZE}_s${SEED}}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
if [[ " $STAGES " == *" train "* ]]; then
  $PYTHON train_maskrcnn.py --train "$COCO/train.json" --val "$COCO/val.json" --run "$RUN" --arch "$ARCH" \
      --size "$SIZE" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --seed "$SEED" --workers "$WORKERS" \
      ${RESUME:+--resume}
fi
for SPLIT in val test; do
  if [[ " $STAGES " == *" predict "* ]]; then
    $PYTHON predict_maskrcnn.py --gt "$COCO/$SPLIT.json" --ckpt "$RUN/best.pt" --out "$RUN/preds_$SPLIT.json"
  fi
  if [[ " $STAGES " == *" eval "* ]]; then
    $PYTHON eval_coco.py --gt "$COCO/$SPLIT.json" --pred "$RUN/preds_$SPLIT.json" --out "$RUN/eval_$SPLIT" \
        --title "Mask R-CNN R50-FPN $ARCH ($MASKS masks, size $SIZE, seed $SEED) on $SPLIT"
  fi
done
echo "scores: $RUN/eval_test/metrics.md"
