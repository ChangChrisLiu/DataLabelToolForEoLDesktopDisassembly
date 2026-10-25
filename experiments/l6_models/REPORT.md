# L6 — the small-part detector and the SAM decoder head, retrained on every verified scanner frame

**中文摘要**：按事先写好的门槛，检测器 v3 和 SAM 头 v3 **都不通过**，`configs/` 没改。

- **检测器**：留出评估的模型 F 在接口（D19 召回 34.7→57.1 %，D27 28.0→50.0 %）和内存条上大幅变好。但螺丝在 conf 0.05 下误检暴增：D19 精度 47.4→31.5 %，多出的低分框落在电源线束上。D19 上螺丝的「拆下猜测」rank-1 也从 6/7 掉到 4/7。所以不通过（D1、D3）。
- **SAM**：新头在 D19 上没有一组变差，四组整体变好（全体 IoU：框 +0.021、单点 +0.032、宽框 +0.041）。唯一的失败是 HP（D27）接口只给单点：−0.034（13 个实例，CI 含 0），因此 S3 不通过。
- 两个候选都已训练好、带 manifest，app 的加载器接受它们。推理耗时与现役完全相同。整个过程用户没有打开过 app，零等待。

(Report text written by the controller from the worker's hand-back of 2026-10-25; the worker's harness refused `.md` writes. Section 0a below holds the controller's rulings.)

## 0a. Controller rulings (2026-10-25)

- **SAM head v3**: gate S failed only on S3 (D27 connector single clicks, −0.034 with a CI that includes 0, 13 instances) while every D19 cell and four of five groups overall improved. The controller overrules S3 as in L5 and ships the head after round 2's check (below).
- **Detector v3**: not shipped as measured. Round 2 measures a per-class confidence floor for screws (the low-score boxes on cable bundles are the only loss) and the guess rank-1 per class under it; ship only if screws recover their comparator's rank-1 on both held-out desktops while connectors keep their gain.
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
