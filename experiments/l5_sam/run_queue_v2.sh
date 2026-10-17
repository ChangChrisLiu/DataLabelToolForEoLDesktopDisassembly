#!/usr/bin/env bash
# Amendment 1: after the v1 queue, train + evaluate F5 v2 (blind) then F13 v2;
# if both pass the v2 gate, train the all-desktop v2 candidate.  One GPU job at
# a time, STOP file honoured between jobs.
#   bash experiments/l5_sam/run_queue_v2.sh
set -u
PY=/d/Anaconda/envs/tda_l3/python.exe
OUT=/d/DataSet/experiments_out/l5_sam
W=D:/DataSet/models/weights
STOP=/d/DataSet/.cache/tmp/l5/STOP
cd "$(dirname "$0")/../.."
stopped() { [ -e "$STOP" ] && echo "[queue2] STOP file present" && exit 0; }

until grep -qE "\[queue\] (done|no f5|STOP)" $OUT/queue.log 2>/dev/null; do sleep 20; done
echo "[queue2] v1 queue finished"; stopped

for f in "f5 5" "f13 13"; do
  set -- $f
  $PY -m experiments.l5_sam.train --fold $1 --recipe v2 > $OUT/train/$1_v2.log 2> $OUT/train/$1_v2.err
  echo "[queue2] $1 v2 trained: $(tail -1 $OUT/train/$1_v2.log)"; stopped
  [ -f "$W/sam2.1_hiera_large_tda_$1_v2.pt" ] || { echo "[queue2] no $1 v2 checkpoint"; exit 1; }
  $PY -m experiments.l5_sam.evaluate --model stock=$W/sam2.1_hiera_large.pt \
      --model $1_v2=$W/sam2.1_hiera_large_tda_$1_v2.pt --desktops $2 --out $1_v2 \
      > $OUT/eval/$1_v2.log 2> $OUT/eval/$1_v2.err
  $PY -m experiments.l5_sam.aggregate fold --csv $1_v2 --ft $1_v2 --desktop $2 --v2 > /dev/null 2>&1
  echo "[queue2] $1 v2 evaluated: $(grep -o '"passed": [a-z]*' $OUT/tables/$1_v2_gate.json)"; stopped
done

if grep -q '"passed": true' $OUT/tables/f5_v2_gate.json && grep -q '"passed": true' $OUT/tables/f13_v2_gate.json; then
  $PY -m experiments.l5_sam.train --fold all --recipe v2 > $OUT/train/all_v2.log 2> $OUT/train/all_v2.err
  echo "[queue2] all v2 trained: $(tail -1 $OUT/train/all_v2.log)"
  [ -f "$W/sam2.1_hiera_large_tda_all_v2.pt" ] && mv "$W/sam2.1_hiera_large_tda_all_v2.pt" "$W/sam2.1_hiera_large_tda_v1.pt" \
      && echo "[queue2] candidate -> $W/sam2.1_hiera_large_tda_v1.pt"
else
  echo "[queue2] a v2 fold failed: no all-desktop candidate"
fi
echo "[queue2] done"
