# L6 — the small-part detector and the SAM decoder head, retrained on every verified scanner frame

**中文摘要**：按事先写好的门槛，检测器 v3 和 SAM 头 v3 **都不通过**，`configs/` 没改。

- **检测器**：留出评估的模型 F 在接口（D19 召回 34.7→57.1 %，D27 28.0→50.0 %）和内存条上大幅变好。但螺丝在 conf 0.05 下误检暴增：D19 精度 47.4→31.5 %，多出的低分框落在电源线束上。D19 上螺丝的「拆下猜测」rank-1 也从 6/7 掉到 4/7。所以不通过（D1、D3）。
- **SAM**（以下留出数字都是折模型 S_F 的；上线的 S_v3 用全部 20 台训练，含 D19、D27）：新头在 D19 上没有一格超出 −0.01 的容差（最差：螺丝/框 −0.0065），四组整体变好（全体 IoU：框 +0.021、单点 +0.032、宽框 +0.041）。唯一的失败是 HP（D27）接口只给单点：−0.034（13 个实例，CI 含 0），因此 S3 不通过。
- 两个候选都已训练好、带 manifest，app 的加载器接受它们。推理耗时与现役完全相同。整个过程用户没有打开过 app，零等待。

(Report text written by the controller from the worker's hand-back of 2026-10-25; the worker's harness refused `.md` writes. Section 0a below holds the controller's rulings.)

## 0a. Controller rulings (2026-10-25)

- **SAM head v3**: gate S failed only on S3 (D27 connector single clicks, −0.034 with a CI that includes 0, 13 instances). On D19 no cell is beyond the −0.01 tolerance (the worst is screw/box, −0.0065), and four of five groups improved overall. The controller overrules S3 as in L5 and ships the head.
- **Detector v3**: not shipped as measured. Round 2 measures a per-class confidence floor for screws (the low-score boxes on cable bundles are the only loss) and the guess rank-1 per class under it. Round 3 ships v3 with screws at 0.15 (see Round 3).
- **Whose numbers these are.** Every held-out number in this report belongs to the fold models: F (detector: v2h fine-tuned on the 18 other desktops) and S_F (SAM: 18 desktops). The shipped v3 and S_v3 were trained on all 20 desktops, D19 and D27 included, and have no held-out score of their own. Carrying F's 0.15 screw floor over to v3 is an assumption, not a measurement.
- **The comparison frame.** What the annotator gets is v3 with screws at 0.15 in place of the shipped v2 at 0.05; round 2's comparators (v2h on D19, the shipped v2 on D27) are at 0.05. At an equal 0.15 floor the comparators' screws are 15/15 against F's 14/15, so the whole held-out gain of the guess is connectors (+7: 11 vs 4) and ram_module (+3: 4 vs 1); all 48 events 32 vs 23.
- **C:** the NVIDIA JIT cache written under `%APPDATA%\NVIDIA\ComputeCache` during the first L6 processes (317 MB new) was removed by the controller together with the folder (driver cache, no CUDA process running); every later L6 process wrote to `D:\DataSet\.cache\nv_compute`.

## 0. Gate and verdict

The rule (`experiments/l6_models/decision_rule.txt`, commit 53f5143) was written before any L6 training or evaluation; only ground-truth counts had been looked at. Held out: D19 (Dell, 55 frames) and D27 (HP ProDesk 400 G1 SFF, 36 frames).

**One deviation from the brief, decided in advance.** The shipped v2 is not held out on D19. Both v1 and v2 were trained on L2's 220 LS draft tiles of D19's own scan frames. So the D19 comparator is **v2h**: the v2 chain (COCO → v1h → v2h) replayed without D19, the same move L4 made with L3 hold13 for D13. D27's comparator is the shipped v2, which never saw D27. The leak is real: on D19 the shipped v2 has connector recall 77.3 % and ram_module 97.1 %, while v2h has 34.7 % and 31.4 %.

### Detector, gate D — FAIL (D1 ✗, D2 ✓, D3 ✗)

Conf 0.05, L4's evaluation code. Recall = GT boxes hit by a same-class box centre. Precision = 1 − FP share.

| class | D19: v2h → F, recall / precision | D27: v2 → F, recall / precision | overall verdict |
|---|---|---|---|
| screw | 67.7 → 80.2 / **47.4 → 31.5** (FP 334 → 801) | 94.7 → 95.5 / **79.1 → 64.5** (FP 61 → 128) | WORSE (precision) |
| connector | 34.7 → 57.1 / 25.2 → 44.1 | 28.0 → 50.0 / 16.2 → 39.0 | BETTER |
| ram_module | 31.4 → 90.0 / 86.2 → 95.5 | 100 → 65.2 / 95.8 → 100 | BETTER |
| psu_latch | 100 → 100 / 100 → 97.8 | no GT | – |
| cpu | 100 → 100 / 93.3 → 100 | 100 → 100 / 100 → 100 | – |

- D1 fails: screw is WORSE on D19 by precision. Detections went from 11.5 to 21.3 per frame, FP per frame from 6.1 to 14.6.
- D2 passes: connector and ram_module are better overall.
- D3 fails: screw and connector pooled on D27 is BETTER (recall 72.9 → 80.7, precision 45.1 → 56.6) but screw alone is WORSE on D27 (precision −14.6 points, +67 FP). Not gated: D27 ram_module recall fell 23 → 15 boxes.
- Sensitivity: with the shipped v2 as D19's comparator the gate fails as well (D1 on screw, connector and ram_module).

**Why screws fail.** The extra screw boxes score 0.05–0.25 and sit on cable bundles (D19 s42: the red and orange PSU wires beside the 24-pin connector, `D:\DataSet\.cache\tmp\l6\overlays\F_screw_D19_s042.jpg`). Every true screw is still found. F vs v2h on D19 screws at other confidences:

| conf | F recall / precision | v2h recall / precision |
|---|---|---|
| 0.10 | 75.2 / 53.6 | 66.1 / 75.3 |
| 0.25 | 68.2 / 83.6 | 63.6 / 94.6 |

On D27 at 0.10, F's screw precision is better than v2's (92.7 vs 86.1).

**The guess** (reported, not gated): L4's `evaluate.guess` (the app's `already_drawn` plus `box_change` ranking), conf 0.05, on the 48 removal events of the armed classes on D19 and D27:

| events | F rank-1 | comparators (v2h on D19, v2 on D27) |
|---|---|---|
| all 48 | 30 | 22 |
| screws | 12/15 (D19 4/7) | 14/15 (D19 6/7) |
| connectors | 11/25 | 4/25 |
| ram_module | 4/5 | 1/5 |

The difference map alone gets 2/48. The shipped v2 gets 30/48 but has seen D19. L4's pattern again: screws lose a little in ranking, connectors gain a lot.

**AP50, held out (comparator → F):** screw D19 61.5 → 68.3, D27 95.7 → 95.0; connector D19 13.1 → 28.4, D27 5.8 → 15.6; ram_module D19 28.3 → 61.4, D27 100 → 94.3.

### SAM head, gate S — FAIL (S1 ✓, S2 ✓, S3 ✗)

Comparison: shipped = route(stock, f5_v2) against new = route(stock, S_F), with `configs/sam.yaml` routing. L5's harness on all 2 487 D19 and D27 masks. Prompt-averaged ΔIoU (point and box) with the 95 % CI over instances:

| group | D19 | D27 | overall |
|---|---|---|---|
| small | +0.015 [+0.002, +0.029] | +0.022 [+0.004, +0.040] | **+0.017 [+0.006, +0.029]** |
| screw | +0.015 [−0.003, +0.037] | +0.025 [−0.002, +0.053] | **+0.018 [+0.003, +0.036]** |
| connector | +0.026 [−0.002, +0.059] | **−0.018** [−0.044, +0.004] | +0.015 [−0.008, +0.039] |
| latch | +0.013 [+0.004, +0.025] | +0.015 [+0.005, +0.036] | **+0.014 [+0.006, +0.023]** |
| large | +0.056 [+0.027, +0.075] | +0.067 [−0.034, +0.196] | **+0.061 [+0.009, +0.130]** |

- S1 passes: no D19 cell is below −0.01 (worst: screw/box −0.006). S2 passes: four groups better overall.
- S3 fails: D27 screw and connector pooled +0.011 [−0.011, +0.030] (CI includes 0); D27 connector single click 0.545 → 0.511 (−0.034 [−0.075, +0.001]; 118 masks, 13 instances), box −0.002.
- All masks, shipped → new: box 0.742 → 0.763, point 0.471 → 0.504, point_box 0.753 → 0.775, loose (difference-map) box 0.664 → 0.705, one corrective click 0.791 → 0.816. Boundary F of all masks +0.026. Latency identical.

Neither candidate is recommended under the rule. Both were trained anyway, as the rule says, because the controller overruled the gates in L4 and L5.

## 1. Data

- Snapshot: backup-API copy at 21:33 on 2026-10-24 (live file opened `mode=ro`) → `D:\DataSet\.cache\tmp\l6\live_copy.sqlite`. `export-coco` writes (truth refresh + lock), so it ran on a second copy, `export_work.sqlite`.
- Export: `python -m tda.cli export-coco --db <copy> --desktops 2-17,19-21,27 --view scan --no-hidden` → 924 images, 24 621 annotations; COCO sha256 `6ed4fd00…10e`.
- Kept: frames with `review_status = verified` (923; D27 s7 has confirmed rows but is not a confirmed frame). Annotations: not crowd (28 dropped), not context, `in_chassis`, visibility in {visible, visible_tiny, occluded_partial, too_small}; `occluded_full` (198) skipped → 24 359 boxes and masks. A direct read of the same `compiled_mask` rows agrees on all 923 frames.
- Boxes per class: connector 6 183 (311 instances, 20 desktops), screw 6 049 (240, 20), ram_latch 5 200 (132, 20), psu_latch 698 (19, 19), cpu_socket_lever 645 (20, 20), ram_module 570 (37, 20), drive_latch 239 (19, 19), cpu 197 (20, 20), cable_clip 132 (4, D27 only).
- Held-out GT: screw D19 440 / D27 244; connector 352 / 118; ram_module 70 / 23; psu_latch 44 / 0 (the HP has none); cpu 14 / 25. SAM masks: D19 1 569 (53 instances), D27 918 (39). Removal events: 647 in all.
- D27 has no stored ROI: tiled and evaluated at the box the app proposes (`suggest_roi_over` on steps 1, 22, 43) = [393, 97, 1510, 1356]; every D27 box centre lies inside it. In the app the guess stays off on D27 until a ROI is stored.
- LS tiles stay in the mix as in L4; LS scan tiles of desktops whose verified frames replace them are dropped (D8 48, D10 168, D13 246, D19 220) → 4 191 LS tiles. Verified tiles: 3 764 (D27 gives 6 per frame, larger ROI).

## 2. Training

Recipes: L3/L4's for the detector (30 epochs, effective batch 16, lr 1e-4, EMA, 640, seed 0, `last_ema.pth`); L5's v2 for SAM (4 epochs, prompt encoder + mask decoder, half the box prompts loose). Nothing tuned.

| model | init | data | train time |
|---|---|---|---|
| v1h | COCO | 4 653 LS tiles (no D19) | 6 690 s |
| v2h | v1h | L4's `all` fold without D19: 5 327 tiles | 7 190 s |
| **F** (fold) | v2h | 4 191 LS + 3 328 verified tiles of 18 desktops | 8 419 s |
| **v3** | shipped v2 | 4 191 LS + 3 764 verified tiles, all 20 desktops | 5 376 s |
| S_F (fold) | stock SAM 2.1 | 21 872 masks, 18 desktops, 21 872 steps | 8 854 s |
| **S_v3** | stock SAM 2.1 | 24 359 masks, 20 desktops, 24 356 steps | 9 507 s |

Tile hashes: F `062cc103…`, v3 `6df4df6a…`. SAM epoch losses: S_F 0.736 → 0.598, S_v3 0.731 → 0.597.

GPU rule: the detector checks the app (and STOP) after every epoch (exit code 3, resumes from `last.ckpt`); SAM checks every 250 steps and resumes from `state.pt` with seeded order so a resumed run equals an uninterrupted one; `run_queue.py` waits, polling every 5 minutes. The resume path was tested on purpose (STOP at the end of v1h epoch 0). The detector and SAM queues shared the GPU from 22:03 to 03:31; the SAM process was capped at 16 GiB; the card peaked around 28.7 of 32.6 GB.

## 3. Extra classes

ram_latch, drive_latch, cpu_socket_lever are already in the 8-class head (trained, not armed). F, held out:

| class | D19 recall / precision (vs v2h) | D27 recall / precision (vs v2) |
|---|---|---|
| ram_latch | 90.8 / 72.3 (78.5 / 61.5) | 99.3 / 59.1 (79.3 / 59.4) |
| cpu_socket_lever | 82.9 / 78.3 (82.9 / 89.5) | 94.3 / 97.1 (91.4 / 91.4) |
| drive_latch | 100 / 100 (100 / 73.7) | no GT |

None has a removal event on D19 or D27 (opened, not removed); the rule arms them only if gate D passes → none recommended. The v3 manifest lists the 8 head classes; its recommended guess classes are the current five. cable_clip: 132 boxes, all on D27 — nothing measurable held out; no 9-class head trained.

## 4. Cost in the app's terms

`perf_app.py` on an idle GPU through the app's own loaders via `TDA_DETECTOR` / `TDA_SAM_HEAD` override files; `configs/` untouched.

| | v2 | v3 |
|---|---|---|
| detect, ms per 1600² scan frame (ROI crop, fp16), median / p90 | 25.3 / 28.3 | 24.1 / 27.6 |
| cold process (read + detect + dE), median ms | 70 | 69 |
| warm cache, ms | 0.1 | 0.1 |
| stored detections per frame | 26.1 | 23.9 |
| detection cache, D19 / D27 scan JSON | 90.6 / 63.7 KB | 96.2 / 46.7 KB |
| peak CUDA | 436 MB | 437 MB |
| model file | 128.4 MB | 128.4 MB |

The app's loader accepted v3 ("manifest tda_det_rfdetr_small_all_v3.json agrees"). SAM, shipped heads vs v3 head on 200 masks: set_image 30.2 vs 30.0 ms; decode box 8.9 vs 9.0 ms; decode point 8.4 vs 8.4 ms; same routing; head file 16.41 MB in both cases. Perf tests (`test_session_perf`, `test_u3_perf`, `test_session_bench`, `test_sam_service`, `test_u19_weights`) with the overrides pointed at the v3 files: 72 passed in 73.9 s (the window perf tests run on stubs; `test_u19_weights` exercises the real v3 head through `pytest_heads_override.py`: the head file gives the same logits as the whole v3 model, ≤ 1e-3).

## 5. What to ship, and how

The rule ships nothing. If the controller overrules:

- SAM head v3: in `configs/sam.yaml` set `head: D:/DataSet/models/weights/sam2.1_tda_head_v3.pt`. (Adding `connector` to `stock_point_classes` would put the D27 connector point cell on stock, which is below f5_v2's — not advised.)
- Detector v3: in `configs/detector.yaml` set `model: D:/DataSet/models/weights/tda_det_rfdetr_small_all_v3.pt`; keep `conf` / `merge_floor` 0.05 (raising them trades away the screw recall gain).
- Trial without touching defaults (cache under `D:\DataSet\.cache\tmp\l6\det_cache`): `set TDA_DETECTOR=D:\DataSet\.cache\tmp\l6\configs\detector_v3.yaml`, `set TDA_SAM_HEAD=D:\DataSet\.cache\tmp\l6\configs\sam_v3.yaml`, `start_tda.bat`.
- Store a ROI for D27 before expecting the guess there.

## 6. Artefacts and disk use (all on D:)

- Weights in `D:\DataSet\models\weights`: `tda_det_rfdetr_small_all_v3.pt` (128 438 553 B, sha256 e5162e5f…) + `.json`; `sam2.1_tda_head_v3.pt` (16 409 519 B, 141 tensors) + `.json`; `sam2.1_hiera_large_tda_l6_v3.pt` (898 MB, the source of the head). Deletable fold weights: `sam2.1_hiera_large_tda_l6_F.pt` (898 MB), `tda_det_rfdetr_small_l6_{v1h,v2h,F}.pth` (128 MB each). About 2.33 GB in all.
- `D:\DataSet\.cache\tmp\l6\`: 5.2 GB (detector runs 3.4 GB of which `last.ckpt` files 2.0 GB deletable; tiles 1.3 GB; GT 157 MB; SAM eval 101 MB; COCO export 43 MB; two DB copies 130 MB).
- Repo (`experiments/l6_models/`): code, `decision_rule.txt`, `results/*.json` (gates, GT and tile manifests, training records, perf, the two weight manifests, queue logs).
- Reproduce with `D:\Anaconda\envs\tda_l3\python.exe -m experiments.l6_models.<export_gt | build_datasets | run_queue det | run_queue sam | publish det | publish sam>`, then `perf_app --det v2,v3,F --sam` in the `tda` env.

## 7. GPU time, waits, and the C: rule

- GPU time: detector training 7.7 h; SAM training 5.1 h; SAM evaluation 0.33 h; inference, perf and tests about 0.1 h. Wall clock 21:53 to 05:41 (7.8 h), the two queues overlapping.
- Waits for the app: 0 (the app was never opened during the run).
- C:: between 22:06 and 23:52 on 10-24 `%APPDATA%\NVIDIA\ComputeCache` received 21 new files (317 MB) although `CUDA_CACHE_PATH` was set; nothing after 23:52. Removed by the controller (see 0a).

## Round 2 (2026-10-25): a screw-only confidence floor for the detector

Script `experiments/l6_models/round2_floor.py`, result `results/round2_floor.json`. It uses round 1's cached detections of F, v2h and v2 on D19 and D27 (every box ≥ 0.001 into the merge, L4's protocol). There is no retraining and no GPU.

For each floor, F's screw boxes below it are dropped and every other class stays at 0.05. Recall / precision are L4's curves. Rank-1 is L4's guess (the app's `already_drawn` plus `box_change`) on the 48 removal events of the armed classes, read at 0.05, so a screw event's candidates are the screw boxes ≥ the floor. The comparators are round 1's at 0.05: v2h on D19, the shipped v2 on D27.

**Criterion** (the controller's): the smallest floor at which F's screw rank-1 is ≥ the comparator's on both desktops (D19 6/7, D27 8/8), while connectors keep ≥ 10/25 and ram_module ≥ 3/5.

**Result: no floor meets it.**

| screw floor | screw D19 R/P | screw D27 R/P | rank-1 screw D19 | D27 | connector (25) | ram_module (5) | psu_latch | cpu | all 48 | meets |
|---|---|---|---|---|---|---|---|---|---|---|
| F, 0.05 (round 1) | 80.2/31.5 | 95.5/64.5 | 4/7 | 8/8 | 11 | 4 | 1/1 | 2/2 | 30 | no |
| F, 0.06 | 79.1/37.0 | 95.1/74.1 | 4/7 | 8/8 | 11 | 4 | 1/1 | 2/2 | 30 | no |
| F, 0.07 | 77.5/41.5 | 95.1/82.6 | 4/7 | 8/8 | 11 | 4 | 1/1 | 2/2 | 30 | no |
| F, 0.08 | 76.6/46.1 | 93.9/87.7 | 4/7 | 7/8 | 11 | 4 | 1/1 | 2/2 | 29 | no |
| F, 0.09 | 75.9/50.3 | 93.9/90.5 | 5/7 | 7/8 | 11 | 4 | 1/1 | 2/2 | 30 | no |
| F, 0.10 | 75.2/53.6 | 93.0/92.7 | 5/7 | 7/8 | 11 | 4 | 1/1 | 2/2 | 30 | no |
| F, 0.15 | 71.1/68.9 | 91.0/95.7 | **7/7** | 7/8 | 11 | 4 | 1/1 | 2/2 | **32** | no |
| F, 0.20 | 69.1/78.8 | 88.1/96.8 | 7/7 | 7/8 | 11 | 4 | 1/1 | 2/2 | 32 | no |
| F, 0.25 | 68.2/83.6 | 88.1/97.3 | 7/7 | 7/8 | 11 | 4 | 1/1 | 2/2 | 32 | no |
| comparators, 0.05 | 67.7/47.4 | 94.7/79.1 | 6/7 | 8/8 | 4 | 1 | 1/1 | 2/2 | 22 | – |

**Why no floor works.** The two desktops pull in opposite directions:
- D19 reaches the comparator's 6/7 only from 0.15. Below that, the low-score boxes on the PSU cable bundle out-rank the removed screw by dE.
- D27 loses one event, s37 `screw.motherboard.03`, at any floor above 0.078. F scores that removed screw at **0.078** on frame j = 36. v2h scores it 0.075; the shipped v2 scores it 0.67. Above 0.078 F has no candidate on it, so the ceiling drops to 7/8.
- Connectors (11/25) and ram_module (4/5) do not depend on the screw floor.

**Nearest miss: 0.15.**
- Screw rank-1: 7/7 on D19 and 7/8 on D27, so 14/15 pooled, equal to the comparators' 14/15.
- All 48 events: 32 against 22. Connectors 11 vs 4, ram_module 4 vs 1.
- By gate D's own definitions at 0.15:
  - screw on D19 turns BETTER: 313 vs 298 hits, precision 68.9 vs 47.4;
  - screw on D27 is WORSE by recall: 222 vs 231 hits, more than the 4.9-box tolerance, although precision is 95.7 vs 79.1.

Per frame (stored = the detection cache, which holds every box ≥ 0.05 whatever the floor; candidates = armed-class boxes after the floor):

| | D19 stored / armed cand. / screw cand. / screw FP | D27 |
|---|---|---|
| F, 0.05 | 46.95 / 38.25 / 21.25 / 14.56 | 23.19 / 15.69 / 10.03 / 3.56 |
| F, 0.10 | 46.95 / 28.36 / 11.36 / 5.27 | 23.19 / 12.47 / 6.81 / 0.50 |
| F, 0.15 | 46.95 / 25.29 / 8.29 / 2.58 | 23.19 / 12.11 / 6.44 / 0.28 |
| F, 0.20 | 46.95 / 24.04 / 7.04 / 1.49 | 23.19 / 11.83 / 6.17 / 0.19 |
| F, 0.25 | 46.95 / 23.55 / 6.55 / 1.07 | 23.19 / 11.81 / 6.14 / 0.17 |
| comparators, 0.05 | 32.85 / 23.98 / 11.55 / 6.07 | 25.28 / 19.06 / 8.11 / 1.69 |

**Not done, by the instruction "if a floor works".** `conf_by_class` was not added to `tda/`, and there are no new tests or manifest field. If the controller wants 0.15 anyway (screws pooled 14/15 = the comparators', all events 32 vs 22, at the cost of D27 s37), that is the change to make. I can implement it on request.

**C: (round 2).** The controller removed `%APPDATA%\NVIDIA\ComputeCache`. Three GPU processes then ran here: `infer_det` launched directly, a `SamService` load plus predict, and earlier the queue-launched ones. All wrote to `D:\DataSet\.cache\nv_compute` (rewritten at 05:55–05:56), and `%APPDATA%\NVIDIA\` stays empty. No launcher needed fixing. The 22:06–23:52 writes of round 1 remain unexplained.

## Round 3 (2026-10-25): shipping v3 with screws at 0.15 — `conf_by_class`

**Ruling (controller).** Detector v3 ships with a screw floor of 0.15, overruling gate D. The numbers behind it are F's at 0.15 against the comparators at 0.05:
- pooled screw rank-1 14/15, equal to the comparators';
- all 48 removal events 32 vs 22; connectors 11 vs 4; ram_module 4 vs 1;
- the one D27 event lost is a screw scored 0.078.

Three qualifications, recorded with the ruling:
- **Whose numbers.** They are the fold model F's (held out D19 and D27). The shipped v3 was trained on all 20 desktops, D19 and D27 included; that its screws behave like F's at 0.15 is an assumption carried over, not measured.
- **The comparison frame.** The change the annotator sees is v3 at 0.15 in place of v2 at 0.05, and the comparators above are at 0.05. At an equal 0.15 floor the comparators' screws are 15/15 (v2h 7/7 on D19, v2 8/8 on D27) against F's 14/15, and all 48 events are 32 vs 23. So the whole held-out gain of the guess is connectors (+7) and ram_module (+3); screws are one event behind.
- **SAM.** S_v3 ships too (S3 overruled, as L5's gate was). Its held-out numbers are S_F's (18 desktops); S_v3 was trained on all 20. On D19 no cell of S_F is beyond the −0.01 tolerance (worst: screw/box, −0.0065).

### The change (commit 22b0ff9)

- **`tda/models/detector.py`**
  - `DetectorConfig.conf_by_class` holds `(class, floor)` pairs; `conf_for(cls)` returns the class's floor, else `conf`.
  - `_class_floors` checks the file. It must be a mapping, of armed classes only, each floor in (0, 1] and at or above `merge_floor` (the merge drops anything lower before it is stored). Anything else is the loader's usual single WARNING, and the detector is off.
  - The floors are **not** part of `recipe()`, so the cache written without them is still valid.
  - `check_manifest_floors(manifest, config)` runs where the model is built (both the YOLO and the RF-DETR backend). When the manifest's `conf_by_class` and the file's differ for an armed class, it logs one WARNING naming both; the detector stays on, using the file's floors. A manifest without the key (v1, v2) is not compared.
- **`tda/models/det_engine.py`**: `candidates_of` reads each class at its floor. dE is measured, cached and served for candidates only. What is stored (every box ≥ `store_conf`) is unchanged.
- **`tda/ui/app_detect.py`**: the guess's ranking (rank 1, Shift+C, the already-drawn skip) filters by `conf_for`. The "detector on" INFO line names the floors.
- **`tda/ui/locator_dets.py` and `app_locator.py`**: U28's vanished-detection chips (「拆前 / 拆后」) read each class at its floor.
- **`cli check` and the exports** never read detections, so nothing changed there.
- **`configs/detector.yaml`**: a commented description of `conf_by_class` only; every setting is unchanged.
- **The v3 manifest** (`tda_det_rfdetr_small_all_v3.json`, republished from 22b0ff9, clean) now carries `conf_by_class: {screw: 0.15}` (top level and `inference`), round 2's numbers at 0.15, and the ruling.

### Tests

`tests/test_l6_conf_by_class.py`, 18 tests:
- **The file**: `conf_for` falls back to `conf`; the floors are not part of the recipe; six malformed forms each give one WARNING and a detector that is off (a list, an unknown class, below the merge floor, above one, a word, a bool).
- **The engine**: the floor removes the 0.10 screw from the candidates; a 0.06 connector stays; the 0.10 box stays in the cache file; a file without the floor reads the same cache with no model call and gets the box back; a cache-only answer applies the floor again.
- **The manifest**: agree, differ, absent and unreadable cases; and through the fake `rfdetr` of U4, the model loads either way, with exactly one WARNING when the file lacks the floor.
- **The window, end to end**: on a frame where a 0.10 screw box has the largest dE (round 2's D19 failure in miniature), rank 1 is that box without the floor and the 0.5 screw with it. The 0.10 box is still in the frame's answer and on disk.
- **U28**: a vanished screw at 0.10 gets no chip; at 0.20 it does. The request is made in both cases.

### Cost

Measured with `perf_app.py` on an idle GPU, through the app's loaders with `TDA_DETECTOR`. v3 has seen D19 and D27, so these are cost figures, not quality figures.

| D19 + D27, 91 frames | v3 | v3, screws at 0.15 |
|---|---|---|
| detect, median ms per 1600² scan frame | 23.6 | 23.6 |
| cold process (read + detect + dE), median ms | 66.8 | 66.5 |
| warm cache, ms | 0.1 | 0.1 |
| stored detections per frame | 23.9 | 23.9 (unchanged) |
| armed candidates per frame | 16.78 | 16.22 |
| screw candidates per frame | 8.20 | 7.64 |
| cache file, D19 / D27 | 96.2 / 46.7 KB | 95.9 / 46.5 KB (fewer dE slots) |

The held-out fold F at 0.15 is round 2's table. Armed candidates per frame fall from 38.25 to 25.29 on D19 and from 15.69 to 12.11 on D27. Screw false positives per frame fall from 14.56 to 2.58 and from 3.56 to 0.28.

### Perf tests with the override files

Settings:
- `TDA_DETECTOR=D:\DataSet\.cache\tmp\l6\configs\detector_v3_screw0.15.yaml`
- `TDA_SAM_HEAD=D:\DataSet\.cache\tmp\l6\configs\sam_v3.yaml`
- the `pytest_heads_override` plugin, so that `test_u19_weights` checks the v3 head file against the whole v3 model.

Files run: `test_session_perf`, `test_u3_perf`, `test_session_bench`, `test_sam_service`, `test_u19_weights`, `test_l6_conf_by_class`. Result: **90 passed in 82.8 s**.

### Full suite (`tda` env)

The first run ended **5 048 passed, 1 failed (exit 1)**. The failure is `test_session_perf.py::test_gui_thread_budgets_on_a_12mp_frame_with_forty_instances`: the session commit took 0.468 s at best against a 0.45 s budget, under the suite's own load. That path does not touch the detector. Alone, the same test passed 3 out of 3 runs (9.5 s each); it also passed in both perf runs above. The second full run, on the same tree: **5 049 passed in 680 s, exit 0**.

### Lines for the controller

`configs/detector.yaml`:

```yaml
backend: rfdetr
model: D:/DataSet/models/weights/tda_det_rfdetr_small_all_v3.pt
classes: [screw, connector, ram_module, psu_latch, cpu]
conf: 0.05
merge_floor: 0.05
conf_by_class:
  screw: 0.15
```

Keep the commented fallbacks the tests look for: `# model: D:/DataSet/models/weights/tda_det_rfdetr_small_all.pt`, `# backend: yolo`, and the YOLO model line. Add `# model: D:/DataSet/models/weights/tda_det_rfdetr_small_all_v2.pt` as the one-swap fallback. The comment above `model:` should get L6's numbers, and the commented `conf_by_class` block becomes the live one.

`configs/sam.yaml`:

```yaml
head: D:/DataSet/models/weights/sam2.1_tda_head_v3.pt
```

`route` and `stock_point_classes` are unchanged. The header comment's L5 numbers should become L6's: shipped → v3 on D19 and D27, all masks, box 0.742 → 0.763, single click 0.471 → 0.504, loose box 0.664 → 0.705, one corrective click 0.791 → 0.816; 141 tensors, 16.4 MB.

### Test pins to update with that switch

- `tests/test_detector.py::test_the_shipped_file_is_rfdetr_on_scan_oak1_and_oak2_screws_with_l2s_tiles`: the model name becomes `tda_det_rfdetr_small_all_v3.pt`; add `assert parsed.conf_by_class == (("screw", 0.15),)`; the docstring should say L6's v3.
- `tests/test_u19_heads.py::test_the_shipped_file_is_the_ruling`: the head path becomes `sam2.1_tda_head_v3.pt`.
- `tests/test_u19_weights.py`: `FULL_F5_V2` becomes `D:/DataSet/models/weights/sam2.1_hiera_large_tda_l6_v3.pt` (the whole v3 model; keep that file, or the test skips). Rename `test_the_tuned_head_is_the_whole_f5_v2_model` and update the docstring. The plugin run above already shows these three tests pass with v3.
- `tests/test_detector.py::test_the_shipped_models_manifest_has_its_tiles_and_classes` passes with the v3 manifest unchanged; `tests/test_l6_conf_by_class.py::test_the_v3_manifest_carries_the_ruled_floor` pins the floor in it.

### C: (round 3)

`%APPDATA%\NVIDIA\` is still empty after round 3's GPU runs (`perf_app`, the perf tests, the full suite). Every run wrote to `D:\DataSet\.cache\nv_compute`.

## Round 4 (2026-10-25): the review's notes

Review: `D:\DataSet\.superpowers\sdd\2026-09-20-user-trial\L6-review.md`, verdict MERGE, both models ship.

- **N1, the U28 chips.** A vanished detection is now one present in frame r−1 at or above the class's floor and absent from frame r **judged at the base `conf`** (0.05), not at the floor.
  - Code: `tda/ui/locator_dets.vanished(..., after_conf=)`; `LocatorDetections` passes its base `conf` for frame r. The default stays one threshold for both frames, as U28 had it.
  - Why: judged at the floor in both frames, a screw that only dipped under 0.15 in r looked gone. The reviewer counted on D19 / D27: first chip right 12/15 with 4 wrong chips that way, against 13/15 with 1 wrong chip the asymmetric way.
  - Tests: U28's rule test (`tests/test_u28_round3.py`) now pins both forms. A new window test (`tests/test_l6_conf_by_class.py::test_a_screw_still_seen_in_r_under_its_floor_has_not_vanished`): a 0.20 screw in r−1, still seen at 0.10 in r, gives no chip.
- **Wording.** Section 0a and the round-3 ruling now say:
  - the held-out numbers belong to F / S_F; v3 and S_v3 were trained on all 20 desktops, D19 and D27 included;
  - carrying the 0.15 floor over to v3 is an assumption;
  - on D19 no SAM cell is beyond the −0.01 tolerance (worst: screw/box, −0.0065), rather than "every D19 cell improved";
  - the comparison frame: v3 at 0.15 against the shipped v2 at 0.05. At an equal 0.15 the comparators' screws are 15/15 (v2h 7/7, v2 8/8) against F's 14/15, all 48 events 32 vs 23, so the whole gain is connectors +7 and ram_module +3. I re-measured the 15/15 here with `round2_floor.floored` on the comparators' cached detections.
- **Spec.**
  - §4.5 gains the paragraph 「小件检测器 v3 与按类下限（v1.30，L6）」: the data, `conf_by_class`, the shipped file, F's numbers with their frame, the asymmetric chip rule, and D27's missing ROI.
  - §5.1 gains the bullet 「解码头 v3（v1.30，L6）」: the data, the shipped head, S_F's numbers, the D27 connector exception.
  - v1.30 goes on top of §14. The docs tests pinning the newest entries (U14, U19–U28) now accept v1.30 above them, and each slice is one longer. `tests/test_l6_docs.py` pins the new text.
- **Configs** (live values unchanged; the ship commit flips them):
  - `configs/detector.yaml` step 1 now reads "conf >= `conf` or the class's own floor (`conf_by_class`, below)".
  - `configs/sam.yaml`'s header carries L6's numbers, the comment above `head:` describes the v3 head, and a commented `# head: …sam2.1_tda_head_f5_v2.pt` line is the fallback.
- **C: again, and the fix** (commit 2083379).
  - An empty `%APPDATA%\NVIDIA\ComputeCache` reappeared at 06:58:51. It was my own `pytest` run of the docs tests and `tests/test_detector.py`, started from a shell without `CUDA_CACHE_PATH`.
  - Reproduced: `tests/test_detector.py` alone, without the variable, creates the folder.
  - Fix: `tests/conftest.py` now sets `CUDA_CACHE_PATH` to start_tda.bat's `D:\DataSet\.cache\nv_compute` unless the caller set one.
  - After the fix, `test_detector`, `test_sam_service` and `test_u19_weights`, and then the whole suite, were run without the variable; the folder was not recreated. The empty folder was removed (twice), and `%APPDATA%\NVIDIA\` is empty again.
- **Full suite** (`tda` env, final tree 2083379): **5 054 passed, 1 failed, exit 1.**
  - The failure is the 12 MP frame-change timing budget `tests/test_u3_perf.py::test_arrival_with_a_warm_detector_cache_is_inside_the_frame_change_budget[12mp]`: 0.122 s at best against 0.12 s.
  - In the same test the detector-off frame changes also took 0.124–0.138 s, so the machine was slower, not the detector path. Other software was busy on the CPU (`sldBgDwld`, OMEN).
  - The two other runs since round 4's changes each failed one or two of the same 12 MP budgets (`test_session_perf` window gesture 0.126–0.129 s, `test_u3_perf` 12 MP 0.128 s), and everything else passed.
  - Alone, `test_session_perf.py` plus `test_u3_perf.py` pass 3 out of 3 runs.
  - Round 3's second full run (06:30) passed whole, exit 0.
