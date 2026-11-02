#!/bin/bash
# One closed model (OpenAI-compatible endpoint) x one input variant, an array over the views; CPU only.
#   sbatch --export=ALL,PACK=$SCRATCH/vlmpack_2026-11-05,MODEL=<api model name>,VARIANT=image closed_model.sbatch
# The repeat subset (3 answers per question):
#   sbatch --export=ALL,PACK=...,MODEL=...,SUBSET=$PACK/eval_v1/repeat.ids.txt,REPEAT=3,MODEL_TAG=<model>_repeat closed_model.sbatch
#
# The key: a file only you can read, sourced below, never in this repository:
#   echo 'export OPENAI_API_KEY=...' > $HOME/.vlm_keys.env && chmod 600 $HOME/.vlm_keys.env
# API_KEY_ENV names the variable; vlm_run.py reads it and never prints or writes it.
#SBATCH --job-name=vlm_api
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

# TODO(hprc): module loads + proxy -- compute nodes usually need one to reach an API
# module purge
# module load GCCcore/13.3.0 Python/3.12.3
# module load WebProxy
source "${VENV:-$SCRATCH/venvs/vlmpack}/bin/activate"
source "${KEYS:-$HOME/.vlm_keys.env}"

VIEWS=(scan oak1 oak2 rs)
export PACK="${PACK:?set PACK}"
export MODEL="${MODEL:?set MODEL to the API model name}"
export VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
export BACKEND=openai
export BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
export API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
export VARIANT="${VARIANT:-image}"
export SUBSET="${SUBSET:-eval}"
export REPEAT="${REPEAT:-1}"
export MAX_SIDE="${MAX_SIDE:-1568}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export RUN_ARGS="${RUN_ARGS:---workers 4 --max-retries 6 --backoff 2}"
bash "$PACK/tools/vlm_run/run_eval.sh"
