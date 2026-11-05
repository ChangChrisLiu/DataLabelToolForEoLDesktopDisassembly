# Sourced by every job in this folder. Edit the TODO(hprc) lines once per cluster account.
# Submit the jobs from the kit folder (<download>/benchmarks/level1) so that $SLURM_SUBMIT_DIR is the kit.
set -euo pipefail

# TODO(hprc): load Python / CUDA the way your cluster provides them, for example
#   module purge; module load GCCcore/13.2.0 Python/3.11.5; source "$L1/venv/bin/activate"
#   or: module load Anaconda3; source activate "$L1/conda/e3d-l1"

export L1=${L1:-$SCRATCH/e3d}                 # TODO(hprc): your project space ($SCRATCH on Grace / FASTER / ACES)
export DATA=${DATA:-$L1/hf}                   # the Hugging Face download: annotations/scanner, splits.json, benchmarks/level1
export KIT=${KIT:-${SLURM_SUBMIT_DIR:-$DATA/benchmarks/level1}}
export WORK=${WORK:-$L1/level1}               # COCO files, YOLO dataset, runs
export MASKS=${MASKS:-visible}                # visible | amodal
export COCO=$WORK/coco_$MASKS
export YDATA=$WORK/yolo_$MASKS
export PYTHON=${PYTHON:-python}

# caches in the project space, not $HOME
export HF_HOME=$L1/cache/hf
export TORCH_HOME=$L1/cache/torch             # Mask R-CNN COCO weights land in $TORCH_HOME/hub/checkpoints/
export YOLO_CONFIG_DIR=$L1/cache/ultralytics_cfg
export L1_WEIGHTS_DIR=$L1/cache/weights       # YOLO weights (yolo11n-seg.pt, ...)
export MPLCONFIGDIR=$L1/cache/mpl
mkdir -p "$WORK" "$HF_HOME" "$TORCH_HOME" "$YOLO_CONFIG_DIR" "$L1_WEIGHTS_DIR" "$MPLCONFIGDIR"
# Compute nodes without internet: run slurm/fetch_weights.sh on a login node first, then uncomment:
# export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1

echo "[env] KIT=$KIT DATA=$DATA WORK=$WORK MASKS=$MASKS host=$(hostname) $(date)"
"$PYTHON" -c "import sys, torch; print('[env] python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
