# E3D Level-1 benchmark kit: detection and instance segmentation on the scanner view

This folder is self-contained: with it, the scanner annotation (`annotations/scanner/`) and `splits.json` from
the Hugging Face dataset `ChangChrisLiu/E3D-Desktop-Disassembly`, you can build standard COCO files, run two
standard baselines (Ultralytics YOLO segmentation, torchvision Mask R-CNN) and score any COCO-format
predictions, on a workstation or on a SLURM cluster. Nothing here writes into the downloaded pack.

## What is measured

* **Task.** Instance segmentation (and the boxes around the masks) of the parts of a desktop PC on the
  structured-light scanner's top-down RGB frames (1600 x 1600, 2 800 frames of 66 desktops, every frame
  confirmed by a human annotator). 24 categories with the pack's own ids (1-23 and 26; `loose_cable` exists only
  in the components gallery and is not in this view).
* **Split.** `splits.json`: **test = D04 D10 D15 D20 D24 D30 D31 D32 D46 D48 D49 D65** (12 desktops,
  platform-disjoint: a model platform is wholly on one side), train = the other 54. `make_coco.py` holds out a
  fixed **val** set of 6 train desktops, `random.Random(0).sample(sorted(train), 6)` =
  **D03 D21 D33 D35 D41 D60**; train = the remaining 48. Val shares platforms with train (unlike test), so val
  scores are optimistic: use val only to pick checkpoints, report test.
* **Metric.** COCO AP from pycocotools, for `segm` (masks) and `bbox`: **AP = mean over IoU 0.50:0.05:0.95**
  (the primary number), **AP50 (IoU >= 0.5) as the headline-compatible number**, AP75, APs / APm / APl (COCO
  area ranges, on the ground-truth mask area), AR1 / AR10 / AR100, at most 100 detections per image.
  Mean AP averages the classes that have test instances (21 of 24: `cooler_bracket`, `misc_part` and
  `screw_hole` have none on the test desktops; `case_fan` has 2 and `cooler_latch` 1, so their per-class AP is
  noise). `eval_coco.py` also writes per-class AP / AP50 and AP / AP50 binned by the ground truth's
  `occlusion_ratio` (other ground truth ignored in each bin, the way COCO handles area ranges).
* **Masks.** `--masks visible` (default): the confirmed visible masks; parts that are in place but completely
  covered on a frame (the pack's crowd rows) are **ignore regions** (`iscrowd = 1`). `--masks amodal`: every
  part's full silhouette (`amodal_segmentation`), crowd rows included as ordinary ground truth; the amodal
  evaluation adds a row "fully covered (= 1)" for exactly those parts.

## Download

```bash
pip install "huggingface_hub>=1.0"     # provides the `hf` command
export DATA=$PWD/e3d                   # any folder; ~9 GB for the scanner pack
hf download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset --local-dir "$DATA" \
    --include "annotations/scanner/*" --include "splits.json" --include "benchmarks/level1/*"
# the 16-bit label maps (labels/, 34 MB) are not used by this kit; --exclude "annotations/scanner/*/labels/*" skips them
```

With huggingface_hub < 1.0 the same download is
`huggingface-cli download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset --local-dir "$DATA" --include "annotations/scanner/*" "splits.json" "benchmarks/level1/*"`
(`huggingface-cli` no longer works in huggingface_hub 1.x). The repository is public; no token is needed to read it.

The layout the scripts expect (`--data-root $DATA`):

```
$DATA/splits.json
$DATA/annotations/scanner/D01 ... D66/{images/S<nn>.png, annotations.json, ...}   # read its README.md first
$DATA/benchmarks/level1/          # this kit
```

The components gallery (`annotations/components/`) is not used by these baselines.

## Environment

Python 3.11 with the pinned versions in `requirements.txt` (tested: Windows 11, RTX 5090, CUDA 12.8):

```bash
python -m venv $HOME/venvs/e3d-l1 && source $HOME/venvs/e3d-l1/bin/activate   # or a conda env
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r $DATA/benchmarks/level1/requirements.txt
```

Keep caches off a small home disk: `HF_HOME`, `TORCH_HOME` (Mask R-CNN COCO weights land in
`$TORCH_HOME/hub/checkpoints/`), `YOLO_CONFIG_DIR` (Ultralytics settings) and `L1_WEIGHTS_DIR` (YOLO weights;
`train_yolo.py --weights-dir` overrides it). `slurm/env.sh` sets all four under `$L1/cache`.

**Offline nodes.** Run `python fetch_weights.py --yolo-dir "$L1_WEIGHTS_DIR"` (or `bash slurm/fetch_weights.sh`)
once where there is internet; it fills `$TORCH_HOME/hub/checkpoints/` with
`maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` (and the v1 file) and `$L1_WEIGHTS_DIR` with `yolo11n-seg.pt`,
`yolo11s-seg.pt` and the small models Ultralytics' AMP self-check may load. Then set `HF_HUB_OFFLINE=1
YOLO_OFFLINE=1`. Alternatively pass a local file: `train_yolo.py --model /path/x.pt`,
`train_maskrcnn.py --weights /path/state_dict.pth` (a torchvision Mask R-CNN state dict with this head size, or a
checkpoint of this script).

## Files

| file | what it does |
|---|---|
| `make_coco.py` | per-desktop `annotations.json` -> COCO `train.json` / `val.json` / `test.json`, `stats.json`, `stats.md` |
| `eval_coco.py` | pycocotools bbox / segm scoring of COCO results -> `metrics.json`, `metrics.md`; `--selftest` |
| `coco_to_yolo.py` | COCO split -> Ultralytics dataset (linked images, polygon labels, `data.yaml`, `category_map.json`) |
| `train_yolo.py`, `predict_yolo.py` | baseline (a): YOLO segmentation training; predictions -> COCO JSON |
| `train_maskrcnn.py`, `predict_maskrcnn.py`, `maskrcnn_lib.py` | baseline (b): Mask R-CNN R50-FPN training; predictions -> COCO JSON |
| `smoke.py` | the whole chain on 2 train / 1 test desktop for 1 epoch (minutes) |
| `fetch_weights.py` | pretrained weights for offline machines |
| `l1common.py` | shared helpers |
| `slurm/` | `env.sh` (edit the `TODO(hprc)` lines), `00_prepare.sbatch`, `10_yolo.sbatch`, `20_maskrcnn.sbatch`, `submit_all.sh`, `fetch_weights.sh` |
| `requirements.txt` | pinned versions |

### `make_coco.py`

* Image ids are the pack's (unique across desktops; checked). Annotation ids are renumbered 1..N per file (the
  pack numbers them per desktop, so they collide across desktops).
* `file_name` = `D<nn>/images/S<nn>.png`, relative to `info.image_root` (the absolute pack path at build time).
  Every script takes `--image-root` to override it, so the COCO files move with the data.
* Each annotation keeps, besides the COCO fields: `desktop`, `instance_key` (the part's stable name),
  `instance_id`, `pack_ann_id`, `visibility`, `occlusion_ratio`, `clean`, `shared_px`; each image `desktop`,
  `step`, `roi`.
* Options: `--masks visible|amodal`; `--clean-only` (only rows with the pack's `clean` flag are positives;
  checked against `clean_parts.csv`); `--min-area N` (on the chosen mask); `--filtered ignore|drop` (a row removed
  by `--clean-only` / `--min-area` becomes an ignore region, the default, or is deleted); `--classes a,b,c`
  (category subset; ids unchanged); `--val-n`, `--val-seed`, `--val-desktops`; `--train-desktops`,
  `--test-desktops` (smoke tests and ablations only).

## Exact commands (one workstation)

```bash
cd $DATA/benchmarks/level1
W=$DATA/../level1_work                                # outputs; anywhere with ~5 GB free

python make_coco.py --data-root $DATA --out $W/coco_visible                    # 3 s
python eval_coco.py --gt $W/coco_visible/test.json --selftest                  # must print SELFTEST PASSED

# (a) YOLO segmentation
python coco_to_yolo.py --coco $W/coco_visible --out $W/yolo_visible           # ~7 min (polygons)
python train_yolo.py --data $W/yolo_visible/data.yaml --run $W/runs/yolo11n_visible \
    --model yolo11n-seg.pt --imgsz 1280 --epochs 100 --batch 8 --seed 0
python predict_yolo.py --gt $W/coco_visible/test.json --weights $W/runs/yolo11n_visible/train/weights/best.pt \
    --cmap $W/yolo_visible/category_map.json --out $W/runs/yolo11n_visible/preds_test.json --imgsz 1280
python eval_coco.py --gt $W/coco_visible/test.json --pred $W/runs/yolo11n_visible/preds_test.json \
    --out $W/runs/yolo11n_visible/eval_test

# (b) Mask R-CNN R50-FPN v2 (COCO weights)
python train_maskrcnn.py --train $W/coco_visible/train.json --val $W/coco_visible/val.json \
    --run $W/runs/maskrcnn_visible --size 1280 --epochs 24 --batch 4 --seed 0
python predict_maskrcnn.py --gt $W/coco_visible/test.json --ckpt $W/runs/maskrcnn_visible/best.pt \
    --out $W/runs/maskrcnn_visible/preds_test.json
python eval_coco.py --gt $W/coco_visible/test.json --pred $W/runs/maskrcnn_visible/preds_test.json \
    --out $W/runs/maskrcnn_visible/eval_test

# amodal: the same with --masks amodal and other folder names
python make_coco.py --data-root $DATA --out $W/coco_amodal --masks amodal

# smoke test of everything (2 train desktops, 1 test desktop, 1 epoch): ~3 min on one GPU
python smoke.py --data-root $DATA --work $W/smoke
```

Your own model: write a COCO results list (`image_id`, `category_id` = the pack's category id, `score`, `bbox`
`[x, y, w, h]` and/or `segmentation` as COCO RLE, full 1600 x 1600 frame) and run `eval_coco.py --pred`.

## SLURM (TAMU HPRC or any cluster)

```bash
cd $DATA/benchmarks/level1
# 1. edit slurm/env.sh: the module / env lines and L1 (all marked TODO(hprc)); in the .sbatch files the
#    partition, GPU type and account lines (TODO(hprc)).
# 2. offline compute nodes: bash slurm/fetch_weights.sh   (on a login node)
bash slurm/submit_all.sh                    # prepare -> YOLO and Mask R-CNN in parallel, visible masks
MASKS=amodal bash slurm/submit_all.sh       # the amodal run
# single stages / other settings, e.g.
sbatch --export=ALL,MODEL=yolo11s-seg.pt,SEED=1 slurm/10_yolo.sbatch
sbatch --export=ALL,STAGES="predict eval" slurm/20_maskrcnn.sbatch
sbatch --export=ALL,RESUME=1 slurm/20_maskrcnn.sbatch          # continue after the time limit
```

Submit from the kit folder: the jobs find the kit through `$SLURM_SUBMIT_DIR`. Outputs go to `$WORK`
(default `$SCRATCH/e3d/level1`): `coco_<masks>/`, `yolo_<masks>/`, `runs/<baseline>_<masks>_<size>_s<seed>/`.

## Settings of the baselines

| | (a) YOLO seg | (b) Mask R-CNN |
|---|---|---|
| model | `yolo11n-seg.pt` (COCO-pretrained; any Ultralytics seg model via `--model`) | torchvision `maskrcnn_resnet50_fpn_v2`, COCO weights, new 25-way box / mask heads |
| input | `--imgsz 1280` (letterboxed, Ultralytics augmentation incl. mosaic) | frame scaled to `--size 1280`, horizontal flip |
| schedule | `--epochs 100`, `--batch 8`, Ultralytics defaults (SGD/auto, warm-up 3 epochs), best.pt by its val fitness | `--epochs 24`, `--batch 4`, SGD lr 0.02 x batch / 16, momentum 0.9, wd 1e-4, warm-up, x0.1 at 2/3 and 8/9; AMP; best.pt by val segm AP (every 4 epochs) |
| seed | `--seed 0`, `deterministic=True` | `--seed 0` (torch, numpy, random, the loader's shuffle) |
| ignore regions | not used in training (background) | not used in training |
| inference | conf 0.001, NMS IoU 0.7, max 100 / image, masks mapped back to 1600 x 1600 | score >= 0.001, max 100 / image, masks bilinearly back to 1600 x 1600, threshold 0.5 |
| run folder | `run.json`, `train.log`, `train/` (Ultralytics: `weights/best.pt`, `results.csv`) | `run.json`, `train.log`, `metrics.jsonl`, `last.pt` (resumable), `best.pt` |

Mask R-CNN at 1280 x 1280 with batch 4 peaked at 7 GiB of GPU memory (AMP) and ran at about 0.25 s per
iteration on an RTX 5090 (512 iterations per epoch; 24 epochs about 1 h there, more on slower GPUs or when the
data loader is the bottleneck); `train.log` prints the peak and the speed. Seeds are fixed, but GPU training is
not bit-exact (cuDNN, AMP, loader workers): two runs with the same seed differ slightly. YOLO polygons cannot hold holes (the chassis' visible mask is full of them) and
a mask in several pieces is joined into one polygon as Ultralytics' COCO converter does; this caps YOLO's
`chassis` segm AP and is a property of the baseline, not of the data.

## Expected outputs

* `make_coco.py`: the three JSON files and `stats.md`. Default (visible) split statistics:

  | | train | val | test |
  |---|---|---|---|
  | desktops | 48 | 6 | 12 |
  | images | 2 051 | 257 | 492 |
  | instances (`iscrowd 0`) | 57 647 | 6 785 | 12 930 |
  | ignore regions (fully covered, `iscrowd 1`) | 4 981 | 652 | 1 230 |
  | max instances per image | 53 | 46 | 47 |

  With `--masks amodal`: 62 628 / 7 437 / 14 160 instances and no ignore regions. Together the three splits hold
  all 84 225 annotations of the pack (77 362 visible + 6 863 fully covered). Per-class counts are in `stats.md`.
* `eval_coco.py --selftest` on test: AP = AP50 = AP75 = APs = APm = APl = 1.000 for bbox and segm (visible and
  amodal), and a control with half the predictions removed scoring below 1.
* `eval_coco.py`: `metrics.md` with the summary row per IoU type, the per-class table and the occlusion bins;
  `metrics.json` with the same numbers (0..1) and pycocotools' summary text.
* Smoke test (`smoke.py`, RTX 5090): every stage exits 0 and `smoke_summary.md` lists the scores and seconds per
  stage. The scores are meaningless (one epoch on two desktops; YOLO is still in its warm-up after one epoch, so
  the smoke test predicts with conf 1e-5).

## What to report

For each baseline and mask type (visible, amodal), on the **test** split:

1. segm AP, AP50, AP75, APs / APm / APl, and bbox AP / AP50 (the summary row of `metrics.md`);
2. per-class segm AP (the small parts -- `screw`, `connector`, `ram_latch`, `cpu_socket_lever` -- dominate the
   instance count, the large ones -- `chassis`, `motherboard`, `cover`, `psu` -- the area);
3. AP by occlusion bin, and for amodal the "fully covered (= 1)" row;
4. the settings: model, image size, epochs, batch, seed (one seed is a baseline; three seeds give a spread),
   `--masks`, any `--clean-only` / `--min-area` / `--classes`, and the kit's commit or download date;
5. the val desktops used for checkpoint selection (D03 D21 D33 D35 D41 D60 unless changed).

## Known caveats

* **Crowd rows.** A part that is in place but fully covered has an empty visible mask, `area` 0, `bbox` = its
  amodal box and `iscrowd 1` in the pack. In visible evaluation they are ignore regions: in `bbox` a detection of
  that class lying inside the amodal box is ignored; in `segm` the ignore mask is empty, so a predicted visible
  mask of a hidden part counts as a false positive. Neither baseline trains on them. In amodal evaluation they
  are ordinary ground truth.
* **D63 instance ids.** The final pack added `cover.06` (a broken slot-cover piece) on D63, and the instance ids
  after it shift by +1 on that desktop compared with older copies. COCO annotation ids are renumbered here
  anyway; to join with other files use `instance_key` (+ desktop, step), never an id from an older copy.
* **QA numbers of the pack (its README).** 2 800 frames, 3 217 parts, 84 225 annotations, 6 863 fully covered
  (crowd) rows, 54 872 clean rows. On 37 frames two confirmed visible masks overlap (66 042 px; the
  `segmentation` is kept as confirmed, so ground-truth masks may overlap there; `shared_px > 0`). 134 amodal
  shapes were widened to hold the visible mask. 442 `occluded_full` rows keep a sliver of visible pixels (they are
  ordinary instances here). On 85 rows the stored occlusion ratio differs from the one recomputed from today's
  masks (the bins use the recomputed top-level `occlusion_ratio`). On 414 rows the amodal box reaches past the
  frame's ROI.
* **Visible vs amodal.** They are different tasks: visible = what the camera sees (fully covered parts ignored);
  amodal = the whole silhouette, including parts with no visible pixel. Report them separately; never mix the
  files of one with the predictions of the other.
* **ROI.** The evaluation uses full frames. `frames.json` has no ROI on any frame of D49 and D64 and on all but
  one frame of D48 and D63; D48 and D49 are test desktops, so do not crop to the ROI unless you say so.
* **Classes.** `chassis` and `motherboard` are on almost every frame and nearly never unoccluded (8 and 1 clean
  rows), so `--clean-only` keeps almost none of them; it is meant for ablations, not as the main setting.
* **Pretrained models.** The detector and SAM head used while annotating were trained on frames that include
  test desktops; they are not in this kit and must not be used as baselines as they are.

## 中文说明

本目录是扫描仪视角的 Level-1 基线工具包（目标检测 + 实例分割），只依赖 Hugging Face 上的
`annotations/scanner/` 和 `splits.json`，不会写入下载的数据包。

* 划分：test 为 `splits.json` 的 12 台（D04 D10 D15 D20 D24 D30 D31 D32 D46 D48 D49 D65，按机型平台互斥）；
  val 为 train 中用种子 0 抽出的 6 台（D03 D21 D33 D35 D41 D60，与 train 共享平台，只用于选 checkpoint）；
  其余 48 台为 train。
* 指标：pycocotools 的 COCO AP（IoU 0.50:0.05:0.95，主指标），AP50 作为可对照的 headline 数字，另有 AP75、
  APs/m/l、AR、逐类 AP 和按遮挡率分箱的 AP。`segm` 与 `bbox` 都报。
* 掩码：`--masks visible`（默认，完全被遮挡的零件 = crowd 行，作为忽略区域 `iscrowd=1`）或 `--masks amodal`
  （完整轮廓，crowd 行作为普通真值，另报 "fully covered" 一行）。两者是不同任务，分开报告。
* 流程：`make_coco.py` → `eval_coco.py --selftest`（必须 AP=1.0）→ YOLO（`coco_to_yolo.py`、`train_yolo.py`、
  `predict_yolo.py`）或 Mask R-CNN（`train_maskrcnn.py`、`predict_maskrcnn.py`）→ `eval_coco.py`。
  集群上改好 `slurm/env.sh` 和 `.sbatch` 中标 `TODO(hprc)` 的行后运行 `bash slurm/submit_all.sh`。
* 注意：D63 的 instance id 在加入 `cover.06` 后整体 +1，跨版本请用 `instance_key` 关联；37 帧上两张确认的可见
  掩码有重叠；评测用整帧，不按 ROI 裁剪。
