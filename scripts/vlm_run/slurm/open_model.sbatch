#!/bin/bash
# One open model x one input variant, an array over the four views; vllm offline on one GPU.
#   sbatch --export=ALL,PACK=$SCRATCH/vlmpack_2026-11-02,MODEL=Qwen/Qwen3-VL-8B-Instruct,VARIANT=image open_model.sbatch
#   sbatch --export=ALL,PACK=...,MODEL=OpenGVLab/InternVL3_5-8B,VARIANT=blind open_model.sbatch
# Pilot (100 items, one view): add SUBSET=$PACK/pilot_ids.txt and --array=0.
#SBATCH --job-name=vlm_open
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
# TODO(hprc): the GPU partition / account of the cluster you are on, e.g.
##SBATCH --partition=gpu
##SBATCH --account=XXXXXXXXX
# A 70B-class model: --gres=gpu:a100:4 (or h100:2) and RUN_ARGS="--tensor-parallel 4".
set -euo pipefail

# TODO(hprc): module loads matching the wheels in requirements-vllm.txt (CUDA 12.8 for vllm 0.11)
# module purge
# module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
# module load WebProxy        # only if the compute nodes need it to reach the Hub (weights are better pre-fetched)
source "${VENV:-$SCRATCH/venvs/vlmpack}/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"           # weights live on scratch, never in $HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"       # pre-fetch with `hf download <model>` on a login node
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

VIEWS=(scan oak1 oak2 rs)
export PACK="${PACK:?set PACK}"
export MODEL="${MODEL:?set MODEL to the HF id or local path}"
export VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
export BACKEND="${BACKEND:-vllm}"                  # transformers = the fallback, one request at a time
export VARIANT="${VARIANT:-image}"
export SUBSET="${SUBSET:-eval}"
export CHAT_FORMAT="${CHAT_FORMAT:-openai}"
export MAX_SIDE="${MAX_SIDE:-1568}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export RUN_ARGS="${RUN_ARGS:---dtype bf16 --batch-size 16 --max-images 2 --gpu-mem 0.90}"
bash "$PACK/tools/vlm_run/run_eval.sh"
