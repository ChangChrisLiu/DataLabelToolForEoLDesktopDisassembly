#!/usr/bin/env bash
# One GPU job at a time: wait for the running F13 training, then evaluate F13,
# draw the baseline overlays, train F5 and evaluate it.  Stops at the STOP file.
#   bash experiments/l5_sam/run_queue.sh <pid-of-running-f13-training>
set -u
PY=/d/Anaconda/envs/tda_l3/python.exe
OUT=/d/DataSet/experiments_out/l5_sam
W=D:/DataSet/models/weights
STOP=/d/DataSet/.cache/tmp/l5/STOP
cd "$(dirname "$0")/../.."

stopped() { [ -e "$STOP" ] && echo "[queue] STOP file present" && exit 0; }

while tasklist //FI "PID eq $1" 2>/dev/null | grep -qi python; do sleep 20; done
echo "[queue] f13 training finished"; stopped
[ -f "$W/sam2.1_hiera_large_tda_f13.pt" ] || { echo "[queue] no f13 checkpoint"; exit 1; }

$PY -m experiments.l5_sam.evaluate --model stock=$W/sam2.1_hiera_large.pt \
    --model f13=$W/sam2.1_hiera_large_tda_f13.pt --desktops 13 --out f13 > $OUT/eval/f13.log 2> $OUT/eval/f13.err
echo "[queue] f13 evaluated: $(tail -1 $OUT/eval/f13.log)"; stopped

$PY -m experiments.l5_sam.overlays --csv baseline_all --prompt point_box \
    --model stock=$W/sam2.1_hiera_large.pt --k 8 > $OUT/overlays_baseline.log 2>&1
$PY -m experiments.l5_sam.overlays --csv f13 --prompt point_box --desktop 13 \
    --model stock=$W/sam2.1_hiera_large.pt --model f13=$W/sam2.1_hiera_large_tda_f13.pt --k 8 > $OUT/overlays_f13.log 2>&1
echo "[queue] overlays done"; stopped

$PY -m experiments.l5_sam.train --fold f5 > $OUT/train/f5.log 2> $OUT/train/f5.err
echo "[queue] f5 trained: $(tail -1 $OUT/train/f5.log)"; stopped
[ -f "$W/sam2.1_hiera_large_tda_f5.pt" ] || { echo "[queue] no f5 checkpoint"; exit 1; }

$PY -m experiments.l5_sam.evaluate --model stock=$W/sam2.1_hiera_large.pt \
    --model f5=$W/sam2.1_hiera_large_tda_f5.pt --desktops 5 --out f5 > $OUT/eval/f5.log 2> $OUT/eval/f5.err
echo "[queue] f5 evaluated: $(tail -1 $OUT/eval/f5.log)"
$PY -m experiments.l5_sam.overlays --csv f5 --prompt point_box --desktop 5 \
    --model stock=$W/sam2.1_hiera_large.pt --model f5=$W/sam2.1_hiera_large_tda_f5.pt --k 8 > $OUT/overlays_f5.log 2>&1
echo "[queue] done"
