# L4 — the small-part detector against the annotator's own verified masks

Written up by the controller from the worker's hand-back (2026-10-16; the worker could not
write this file itself). Code: this folder. Outputs: `experiments_out/l4_detector/` (3.3 GB:
`compare.md`, `decision.json`, `decision_rule.txt`, `eval/`, `dets/`, `logs/`, `overlays/`,
`bands/`, the three `rfdetr_small/runs/*/last_ema.pth`).

## 0. Controller decision (overrides the a-priori gate, with the reason on record)

The gate written before any result (`decision_rule.txt`, commit d947513) fails on **G2 by one
event**: the guess's rank-1 hits on the held-out screw removals are 10/12 for v2 against 11/12
for the comparators. Twelve events cannot tell 10 from 11; the lost event (D5 k=41) scored
between 0.05 and 0.10 and ranks first at conf 0.05, which the gate's own operating-point rule
would have chosen. Every other measurement favours v2: screw hit@0.10 held out +1.6 (D13) and
+5.3 (D5, CI +2.5..+7.9); OAK screw hit +8.8 (oak1) and +17.8 (oak2) with no OAK training data;
connectors, RAM latches, the socket lever and drive latches up by 30–100 points on D5.

**Ruling:** publish the all-desktop model as `tda_det_rfdetr_small_all_v2.pt`, ship it with
`conf: 0.05` / `merge_floor: 0.05`, and arm the guess for `screw, connector, ram_module,
psu_latch, cpu`. Connectors are enabled although their D13 hit (41.6 %) is under the 60 % bar:
the bar was set when recall looked like the problem, and on the held-out connector removals the
detector's rank 1 is right 11/30 times against 1/30 for the difference map (production on
D2–D5: 25/64 against 1/64). A wrong box costs the annotator one click outside it (U2h: the
first click outside the box makes the prompt point-only). `drive_latch` stays off (one event).
The gate is to be re-run when more verified desktops exist.

The worker's report follows unchanged.

---

L4 follows L3: the same RF-DETR small, tiles, merge, recipe and L2 metrics. The ground truth is
new: the **230 verified scan frames of D2, D3, D4, D5, D13** in the controller's DB copy
(`D:\DataSet\.cache\tmp\l4\tda_l4.sqlite`, opened `mode=ro`). The live DB was never opened.

Code is in `experiments/l4_detector/`, which imports L2/L3 and edits neither. The decision rule
was committed (d947513) before any detection, evaluation or training result.

**Verdict (worker, by the gate): the gate fails by one screw event. Nothing was published and no
config changed.**
- **Screw hit@0.10, held out:** D13 79.4 → 81.1 (+1.6, bootstrap CI −0.4..+3.8); D5 74.8 → 80.1
  (+5.3, CI +2.5..+7.9).
- **OAK views, with no OAK training data:** oak1 36.2 → 45.0, oak2 52.5 → 70.3.
- **Other classes improve a lot:** connector on D5 52 → 86; ram_latch D13 33 → 73, D5 44 → 95;
  cpu_socket_lever on D5 37 → 100; drive_latch on D5 0 → 100.
- **G2 fails.** Rank-1 hits on screw events, pooled over both held-out desktops: v2 10/12,
  comparators 11/12. The lost event is D5 k=41 (scored 0.05–0.10; first at conf 0.05 → 11/12).

## 1. Ground truth and tiles (`gt.py`, `build_datasets.py`)

**Frames.** D2 48, D3 48, D4 46, D5 46, D13 42, read from the file the app shows (the local
cache copy, pixel-identical to the raw `P_0.png`, checked on D2 s5).

**Boxes.** Each box is the bbox of the compiled visible mask, `in_chassis` only. Kept
visibilities: visible 5 243, occluded_partial 664, visible_tiny 98, too_small 15. Skipped:
occluded_full 909. No bench or out_of_view rows on these frames.

**8-class boxes (frame-boxes):** screw 1 368 (D2 231, D3 363, D4 265, D5 266, D13 243),
connector 1 528, ram_latch 1 439, psu_latch 188, cpu_socket_lever 161, drive_latch 81,
ram_module 134, cpu 66. Other classes counted but not trained (the head stays at 8 classes):
chassis 230, motherboard 222, psu 193, cover 180, cpu_cooler 86, expansion_card 57,
optical_drive 34, storage_drive 27, drive_cage 23, cooler_latch 3.

**Tiles.** The stored pose-segment ROI with L2's tiling gives 4 tiles per frame: 920 tiles,
30 unlabelled. Verified tiles hash `d56a574b…d029`.

| fold | init | LS tiles | verified | total | sha256 |
|---|---|---|---|---|---|
| F13 | L3 hold13 | 4 015 | 752 (D2–5) | 4 767 | abc922b8… |
| F5 | production v1 | 4 627 | 736 (D2/3/4/13) | 5 363 | 7419d86a… |
| all | production v1 | 4 627 | 920 | 5 547 | 879338c4… |

F13 does not start from production, because production was trained on D13's LS drafts of these
same frames; D13's comparator is therefore L3 hold13. In F5 and `all` the verified D13 labels
replace L2's D13 scan tiles. Each verified frame is used once.

**Removal events: 174** — screw 36, connector 77, ram_module 10, drive_latch 6, psu_latch 5,
cpu 5, large classes 35. Dropped: 30 `open` actions, 20 targets still visible on k, 1 target not
visible on j.

## 2. Baseline: production on clean ground truth (hit@0.10)

D2–D5 are held out for production; D13 was in its training set.

| class | prod D2–5 pooled | D2 | D3 | D4 | D5 | prod D13 (seen) | L3 hold13 D13 |
|---|---|---|---|---|---|---|---|
| screw | **66.8** | 60.6 | 62.8 | 69.8 | 74.8 | 89.7 | 79.4 |
| connector | 50.4 | 52.5 | 32.0 | 61.7 | 52.2 | 42.0 | 29.6 |
| ram_latch | 44.1 | 44.4 | 41.2 | 48.9 | 43.8 | 46.1 | 33.4 |
| psu_latch | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cpu_socket_lever | 52.3 | 78.4 | 58.8 | 29.0 | 36.7 | 93.1 | 96.6 |
| drive_latch | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ram_module | 67.9 | 100 | 100 | 10.3 | 100 | 100 | 100 |
| cpu | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

**Screw recall on new machines is 67 % against clean GT, not L3's 88 % against the drafts.**
The misses are motherboard screws (60 % on D2–5; cooler screws 93 %): dark screws on black
plastic (`overlays/production_vs_l3hold13_screw_D2_s030.jpg`), screws under 10 px (0/24) and
partly occluded screws (25 %). **The guess still works on new machines:** rank-1 hits on D2–D5
removal events, production vs difference map — screws 24/30 vs 3/30, connectors 25/64 vs 1/64.

## 3. Held-out folds (hit@0.10 / AP50 / FP per frame @0.10)

| class | L3 hold13 D13 | prod D13 (seen) | **v2 F13** | production D5 | **v2 F5** |
|---|---|---|---|---|---|
| screw | 79.4/70.2/2.81 | 89.7/74.3/1.62 | **81.1**/76.8/2.50 | 74.8/73.0/1.26 | **80.1**/66.1/3.20 |
| connector | 29.6/18.1/3.90 | 42.0/23.2/5.31 | 41.6/25.8/3.29 | 52.2/38.0/5.33 | 86.3/64.9/3.46 |
| ram_latch | 33.4/43.1/0.14 | 46.1/41.9/0.55 | 73.4/66.1/2.38 | 43.8/46.1/0.30 | 95.4/94.2/0.52 |
| cpu_socket_lever | 96.6/93.0/0.24 | 93.1/95.7/0 | 100/100/0.02 | 36.7/86.0/0 | 100/100/0 |
| drive_latch | 0/0/0 | 0/3.6/0 | 0/7.5/0 | 0/23.8/0 | 100/50.0/0.28 |
| psu_latch, ram_module, cpu | 100 | 100 | 100 | 100 | 100 |

v2's screw precision is lower on D5 (FP per frame 1.26 → 3.20, precision at 0.10 77.5 → 59.2 %);
part of that is real but unannotated screws, since the GT lists task instances only. Those extra
boxes compete in the ΔE ranking, not in recall.

Screw hit / precision over conf:

| | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|
| D13 L3 hold13 | 81.5/38.3 | 79.4/62.2 | 71.2/78.3 | 57.2/86.3 |
| D13 v2 F13 | 81.9/45.5 | 81.1/65.2 | 74.5/79.7 | 67.5/90.1 |
| D5 production | 74.8/52.8 | 74.8/77.5 | 74.4/93.8 | 68.4/97.3 |
| D5 v2 F5 | 82.7/46.1 | 80.1/59.2 | 79.3/68.7 | 78.6/76.6 |

## 4. The guess on the held-out events

The app's own pipeline: conf ≥ 0.10, inside the ROI, the app's `already_drawn` skip, ranked by
the app's `box_change(j, k)`.

| desktop | class | n | diff map | comparator top-1/top-3/ceiling | v2 top-1/top-3/ceiling |
|---|---|---|---|---|---|
| D13 | screw | 6 | 1 | 5/6/6 | 5/6/6 |
| D5 | screw | 6 | 0 | 6/6/6 | **5/5/5** |
| D13 | connector | 13 | 0 | 2/4/5 | 4/5/7 |
| D5 | connector | 17 | 1 | 6/9/10 | 7/11/11 |
| both | psu_latch | 2 | 0 | 2 | 2 |
| both | ram_module | 4 | 0 | 3 | 4 |
| both | cpu | 2 | 2 | 2 | 2 |
| D5 | drive_latch | 1 | 1 | 0 | 1 |

Screw rank-1 hits pooled, by conf: comparators 11 / 11 / 11 at 0.05 / 0.10 / 0.25; v2 11 / 10 /
10. Over all small-class events where the detector would apply (screw, connector, psu_latch,
drive_latch) v2 gets 24 against 21.

## 5. OAK regression check (D13, LS drafts, L3's protocol)

| view | class | L3 hold13 hit/AP50/FP | v2 F13 |
|---|---|---|---|
| oak1 | screw (80) | 36.2/30.0/0.87 | **45.0**/41.8/0.26 |
| oak2 | screw (101) | 52.5/25.9/1.26 | **70.3**/31.9/1.26 |
| oak1 | connector | 27.6 | 29.3 |
| oak2 | connector | 39.5 | 56.6 |
| oak1 | ram_latch | 55.7 | 98.9 |
| oak2 | ram_latch | 3.9 | 71.4 |

Caveat: F13 is L3 hold13 trained for 30 more epochs; part of every F13 gain may be the extra
training rather than the new labels. No arm separates the two.

## 6. Decision by the gate (`decision.json`)

| test | rule | result |
|---|---|---|
| G1 | F13 screw hit ≥ L3 hold13 − 2; F5 ≥ production − 2 | 81.1 vs 79.4 ✓; 80.1 vs 74.8 ✓ |
| G2 | screw-event rank-1 hits, v2 ≥ comparator, pooled | **10 vs 11 ✗** |
| G3 | OAK screw hit ≥ L3 hold13 − 3, each view | 45.0 vs 36.2 ✓; 70.3 vs 52.5 ✓ |

Class rule (hit ≥ 60 % on both folds, and detector rank-1 ≥ difference-map rank-1 on the class's
removal events), had the gate passed:

| class | hit D13 / D5 | rank-1 v2 vs diff map | added? |
|---|---|---|---|
| ram_module | 100 / 100 | 4/4 vs 0/4 | yes |
| psu_latch | 100 / 100 | 2/2 vs 0/2 | yes |
| cpu | 100 / 100 | 2/2 vs 2/2 | yes |
| ram_latch | 73.4 / 95.4 | no removal events (only opened) | no |
| cpu_socket_lever | 100 / 100 | no removal events (only opened) | no |
| connector | **41.6** / 86.3 | 11/30 vs 1/30 | no (D13 hit < 60) — overridden, see §0 |
| drive_latch | **0** / 100 | 1/1 vs 1/1 | no |

The operating-point rule would have moved conf to 0.05 (screw rank-1 11 vs 10, hit holds).

## 7. The `all` model

`experiments_out/l4_detector/rfdetr_small/runs/all/last_ema.pth` (sha1 5bbee8459dce), trained in
3 854 s, not stopped early; smoke test through `tda.models.detector.RFDetrTileDetector`: loads
with the 8 classes, 0.024 s median / 0.030 s p90 per scan frame at the 0.10 floor. Published by
the controller as `tda_det_rfdetr_small_all_v2.pt` (§0).

Recommendations (worker): re-run the gate when more verified desktops exist; reconsider
connectors (done in §0).

## 8. Side task: scan area bands — not applied

`refit_bands.py` reuses `fit_bands.band_of` / `render`; candidate
`bands/prompt_box_bands_verified.yaml`, diff `bands/scan_band_diff.json`. Scan changes (old LS
fit → verified): screw 164–433 → 110–502 px; connector 371–3 087 → 238–1 846; ram_latch
211–1 246 → 305–659; cpu_socket_lever 18 311–23 090 → 10 585–14 553 (no overlap); cpu_cooler
31 339–130 779 → 42 848–49 322; cover 5 000–17 004 → 11 118–43 047; motherboard 245 767–412 597
→ 197 785–275 105; storage_drive 14 865–83 974 → 36 258–80 544; cpu, psu, psu_latch within
±10 %; newly banded chassis and drive_latch; newly unbanded drive_cage and optical_drive.

Not applied: `tests/test_prompt_gate.py::test_one_long_lived_part_is_not_a_class` fails with the
candidate (it pins U2i's 1-part expansion card), and replaying the gate on the 173 verified events
shows no gain (wrong boxes cut 63.7 % shipped vs 66.7 % leave-one-out, good boxes withheld 0/38 vs
2/38 — two drive-latch removals whose difference blob covers the whole drive).

## 9. Runtime and footprint

Detection per scan frame (4 tiles, fp16, the annotator's app running): production 0.023 /
0.026 s (median / p90) at the 0.10 floor; v2 fold models 0.024–0.026 s; L2's 0.001 protocol
0.040–0.051 s; inference peak CUDA 356 MB (scan), 676 MB (OAK). Training: peak allocated 7.97 GB,
reserved 10.5 GB; the card stayed at 13.3–17.5 GB of 32.6 GB; GPU utilisation 25–35 % (CPU-bound
on SciPy Hungarian matching); 4 threads, 4 loader workers, BELOW_NORMAL priority, one GPU job at a
time; the STOP file was never created. Run times: F13 3 404 s, F5 3 776 s, all 3 854 s; wall clock
about 3 h 25 min. No downloads, no new packages; caches under `D:\DataSet\.cache\l4\`; nothing on
C: or G:.

## 10. Files

`experiments/l4_detector/`: `gt.py`, `build_datasets.py`, `train_rfdetr.py` (L3's recipe plus
the stop file and the VRAM log), `infer.py`, `evaluate.py`, `oak_eval.py`, `compare.py`,
`overlays.py`, `refit_bands.py`, `publish_production.py`, `run_queue.py`, `decision_rule.txt`.
Checkpoints: `experiments_out/l4_detector/rfdetr_small/runs/{hold13,hold5,all}/last_ema.pth`
(each run also keeps a 513 MB `last.ckpt`).
