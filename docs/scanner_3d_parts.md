# How to take each part's 3D points out of the scanner reconstructions

For the 3D collaborators. Release round 3 (2026-11-02): all **66 desktops**, 2 800 scanner frames, 3 209 parts — the
same recipe as the 10-desktop test package of 2026-10-26, now for every desktop, with a ready-made plan per part and a
script that does the selection.

## 中文摘要（先读这一段）

1. **对应关系**：每个标注帧都是在扫描仪的一张 RGB 图（1600×1600）上画的。那张图在本仓库的
   `raw/scanner/<source_folder 去掉开头的 "UGA DATA/">/<file_name>`（`frames.json` 里逐帧写着；一般是 `P_0.png`，
   少数帧因为 `P_0` 太暗用了别的曝光，见 `burst_choice`）。你们的三维重建只要能做到「每个 3D 点 ↔ 这张图上的一个像素」
   （有组织点云，或与这张 RGB 配准的深度图），零件的点就是 `labels/S<nn>.png == instance_id` 的像素下的那些点。
2. **先看计划表** `parts_3d_plan.csv`：66 台、3 209 个零件，每个零件一行，告诉你用哪条路、用哪一帧：
   - `clean`（2 832 个）：在 `best_frame` 这一帧上它完全没被挡，掩码下的点就是扫描仪看到的它的全部表面——**直接用**；
   - `diff`（259 个）：它从没完全露出来过（散热器在风扇下面、主板在所有东西下面……），就用拆卸顺序：它在第 r 步离开，
     取第 r−1 帧重建里有、第 r 帧里没有（或移动了）、且落在它第 r−1 帧完整轮廓（amodal）里的点；
   - `visible`（99 个）：既没露全、也没有同位姿的离开前后两帧（机箱本身从不离开、换过相机位姿的台），只能取可见部分；
   - `none`（19 个）：每一帧都被完全挡住（全是看不见的接头），没有点。
3. **脚本** `tools/parts3d.py`（只要 numpy + Pillow）：`plan` 生成计划表，`where` 给出原图路径（可下载），
   `extract` 按标签图从有组织点云取点并输出每个零件一个 PLY，`diff` 按拆卸顺序做差。
4. **注意**：一起离开的零件（`left_with`，825 个，比如卡扣随主板、线夹随导风罩）用 `diff` 会一起被取出，需要再按
   它们自己的轮廓分开；挪开（displaced）的零件 `diff` 得到的是它原来位置上的点；D48、D63、D64 中途换过相机位姿，
   跨位姿的两帧不能逐像素比较。完整轮廓（amodal）是轮廓不是点——被挡住的表面扫描仪看不到。

## 1. What you need

| item | where | note |
|---|---|---|
| the annotations | `annotations/scanner/D<NN>/` | `labels/S<nn>.png` (16-bit id map), `annotations.json` (COCO, visible + amodal), `instances.json`, `frames.json`, `sequence.json` |
| the picture each frame was drawn on | `raw/scanner/...` | path below; not repeated in `annotations/` |
| your 3D reconstruction of that capture | yours | one per scanner capture (desktop, step) |
| the plan | `annotations/scanner/parts_3d_plan.csv` | one row per part, 66 desktops |
| the script | `annotations/scanner/tools/parts3d.py` | `pip install numpy pillow` (and `huggingface_hub` to download pictures) |

**Step numbers.** `S<nn>` is the step. `S01` is the untouched machine; every other picture is taken **after** that
step's action (on `S05` the part removed at step 5 is already gone). Frames are 1600 × 1600; coordinates are image
pixels `(x, y)`, x = column, y = row, origin top-left; masks, boxes `[x, y, w, h]`, label maps and the ROI all use it.

## 2. Find the picture and line your cloud up with it

`frames.json` → `frames["<step>"]` gives `source_folder` and `file_name`. On this repository the file is

```
raw/scanner/<source_folder without the leading "UGA DATA/">/<file_name>
e.g. D22 S01 -> raw/scanner/TAMU_B2.3_20-29_RGB/TAMU_B2.3_20-29_RGB/22/RGB11/P_0.png
```

`python tools/parts3d.py where --pack . --desktop 22 --frames 1,12 [--download pictures/]` prints (and fetches) them.
The scanner writes a burst of exposures `P_0 … P_9` per capture; the masks were drawn on the one `file_name` names —
`P_0` on 2 735 frames; on 65 frames `P_0` was dark, missing or an outlier and another exposure was used
(`burst_choice.reason`, `burst_choice.p_index`). All exposures of a burst come from the same camera pose, so the pixel
grid is the same whichever you use.

**Organized cloud.** The script expects, per frame, an array `cloud[y, x] = (X, Y, Z)` (or `(X, Y, Z, R, G, B)`) of
shape 1600 × 1600 × 3 (or × 6), NaN where there is no point — i.e. the point seen at pixel `(x, y)` of that picture.

- If your reconstruction is computed in the scanner camera's own pixel grid (a depth or XYZ map per pixel), it already
  is organized: save it as `.npy`.
- If you have an unorganized point cloud: project every point into the picture with the scanner's calibration,
  `[u, v, 1]ᵀ ∝ K (R X + t)`, keep for each pixel the nearest point (a z-buffer: a pixel sees only the front surface),
  and write it at `cloud[round(v), round(u)]`. Points that project outside the picture or behind a nearer point get no
  pixel and therefore no label.
- If the 3D sensor is a different camera from the RGB one, register them first (the same projection, with the
  RGB camera's K, R, t).

## 3. The plan: which route for which part

`parts_3d_plan.csv` (made by `tools/parts3d.py plan`): one row per part.

| column | meaning |
|---|---|
| `desktop`, `instance_id`, `key`, `class`, `of` | the part (`instance_id` is the value in the label maps; `key` its name, e.g. `screw.cpu_cooler.03`) |
| `route` | `clean` / `diff` / `visible` / `none` (below) |
| `best_frame`, `best_frame_px` | the frame to use and the part's visible pixels there |
| `clean_frames` | every frame on which the part is not covered at all |
| `frames_seen`, `frames_crowd` | frames with visible pixels / frames where it is present but fully covered |
| `leaving_step`, `leaving_kind`, `leaving_verb`, `left_with` | when and how it leaves (`removed` / `displaced` / `changed`), and the part it leaves inside, if any |
| `before_frame`, `after_frame`, `same_pose_segment` | the two frames route 2 compares, and whether the camera pose is the same |
| `raw_best_frame` | the picture of `best_frame` on this repository |
| `note` | special cases in words |

| route | parts | how |
|---|---|---|
| **clean** | 2 832 | On `best_frame` nothing covers the part (`clean: true`: visible mask = whole outline, label `visible`, inside the ROI, no shared pixels). Its points there are all of it the scanner saw. `best_frame` is the clean frame with the most pixels. |
| **diff** | 259 | Never fully uncovered — heatsinks under fans (71), motherboards under everything (60), drive cages, covers, RAM under shrouds, screws under covers. Take the points that disappear when it leaves: present on `before_frame` inside its amodal outline there, gone (or moved by more than a tolerance) on `after_frame`. Needs `same_pose_segment = true`. |
| **visible** | 99 | Never uncovered and no same-pose leaving pair: the chassis (57 — it never leaves), parts on D48 / D63 / D64 whose leaving crosses a camera-pose change, connectors that stay hidden. Take the visible pixels of `best_frame` (the frame where most of it shows): a partial surface. |
| **none** | 19 | Covered on every frame (connectors behind the drive cage or under cables). No points from this view. |

Per class (all 66 desktops):

| class | parts | clean | diff | visible | none |
|---|---|---|---|---|---|
| connector | 1 015 | 965 | 0 | 31 | 19 |
| screw | 791 | 769 | 22 | 0 | 0 |
| ram_latch | 494 | 492 | 1 | 1 | 0 |
| ram_module | 127 | 105 | 22 | 0 | 0 |
| cpu_cooler | 75 | 4 | 71 | 0 | 0 |
| chassis | 66 | 9 | 0 | 57 | 0 |
| motherboard | 66 | 1 | 60 | 5 | 0 |
| cpu | 65 | 59 | 6 | 0 | 0 |
| cpu_socket_lever | 65 | 65 | 0 | 0 | 0 |
| psu | 65 | 61 | 4 | 0 | 0 |
| storage_drive | 65 | 64 | 1 | 0 | 0 |
| cover | 63 | 41 | 20 | 2 | 0 |
| optical_drive | 39 | 39 | 0 | 0 | 0 |
| drive_cage | 38 | 16 | 22 | 0 | 0 |
| expansion_card | 36 | 31 | 5 | 0 | 0 |
| screw_hole | 35 | 14 | 21 | 0 | 0 |
| cable_clip | 35 | 32 | 0 | 3 | 0 |
| others (latches, case fans, brackets, misc) | 69 | 65 | 4 | 0 | 0 |

## 4. Route 1 / clean: the points under the label map

```
python tools/parts3d.py extract --pack . --desktop 22 --frame 1 --cloud D22_S01_cloud.npy \
    --instances psu.01,connector.04 --out parts/            # or leave out --instances: every part on the frame
```

writes `parts/D22_S01_<key>.ply` (binary PLY, with colour if the cloud has it) and `D22_S01.extract.json` (pixels,
points, clean or not, visibility, occlusion ratio per part). In code:

```python
import json, numpy as np
from PIL import Image
labels = np.array(Image.open("D22/labels/S01.png"))          # uint16 instance ids, 0 = background
inst = json.load(open("D22/instances.json"))["instances"]       # keys are strings of instance_id
iid = next(int(k) for k, v in inst.items() if v["key"] == "psu.01")
cloud = np.load("D22_S01_cloud.npy")                            # 1600 x 1600 x 3, NaN = no point
pts = cloud[labels == iid]
pts = pts[np.isfinite(pts).all(axis=1)]                         # the part's points on S01
```

The label map gives every pixel to at most one part; it equals the union of the confirmed visible masks
(`segmentation` in `annotations.json`) everywhere except 3 frames of D02 that keep a small hand-drawn overlap (the label
map then gives the pixel to the part higher in the layer order).

## 5. Route 2 / diff: the points that disappear when the part leaves

```
python tools/parts3d.py diff --pack . --desktop 22 --part psu.01 \
    --cloud-before D22_S11_cloud.npy --cloud-after D22_S12_cloud.npy --tol 0.002 --out parts/
```

`instances.json` → `leaving` gives `step` r, `kind`, `before_frame` (r−1 or the nearest earlier frame in the package),
`after_frame` (r), `same_pose_segment`, `left_with`. The script takes the amodal mask of the part on `before_frame`,
keeps the pixels that have a point on `before_frame` and on `after_frame` either no point or a point farther than
`--tol` (in your cloud's units), and writes those `before_frame` points. Things to know:

- **Parts that leave together.** A part that leaves inside another one (`left_with`: 825 parts — RAM latches with the
  board, the HP fan-cable clip with the shroud, screws in the fan they hold) disappears in the same step as its carrier,
  and so does anything else that changes at step r (the script lists them). Subtract their own outlines (their amodal
  masks on `before_frame`) if they overlap the part's.
- **Displaced parts** (`kind = displaced`: the PSU module folded open, the HP optical-drive bracket lifted, a drive slid
  out): the part is still in the picture on `after_frame`, elsewhere. Route 2 gives its points at the old place, which
  is what you want; do not also take its new place.
- **Changed but never leaving** (`kind = changed`: a connector unplugged, a latch opened, a lever lifted): only a few
  pixels change; prefer route 1 on a clean frame (the plan already does).
- **Camera pose changes** (`same_pose_segment = false`): D48 (2 segments), D63 (4), D64 (8 — the machine is turned over
  several times). Across a pose change compare the two clouds after registering them (nearest neighbour), not pixel by
  pixel; the script refuses those.
- The amodal mask is the part's full **silhouette** on the image, not a set of points: under a cover, the scanner saw the
  cover's surface. Route 2 is what recovers the hidden part's own surface — on the frame where whatever covered it has
  already left, or from what vanishes when the part itself leaves.

## 6. Special cases worth knowing

- **Fully covered on a frame** (`iscrowd: 1`, empty `segmentation`, `area` 0): present but invisible; no points there.
- **Unplugged connectors are not drawn** — from the step a connector is unplugged it has no mask. A connector hidden by
  cables carries `visibility = occluded_full` with `attributes.occluder = "cable"`.
- **Optical drives that never leave their bracket** (HP SFF D22–D33, Dell D38–D41) are one instance `drive_cage` with
  `attrs.of = "optical"` — the drive and its bracket together; `optical_drive` is used where the drive is removed on its
  own. The two drive-end plugs on D23–D27 (`connector.sata_data.01`, `connector.sata_power.01`) ride on that bracket.
- **`screw_hole`** (35 parts, 7 desktops) is the hole a removed fan screw leaves — a detection class, not a part; skip it
  for 3D.
- **The ROI** (`frames.json` → `roi`, `[x0, y0, x1, y1]`) is the rectangle the annotation concentrated on; a few large
  parts (chassis, covers, PSU) reach past it and are then not `clean`.
- **Visibility labels** come from the occlusion ratio: a `visible` part may still be up to 30 % covered — `clean` is the
  test for "not covered at all".
- **Layer order** follows the physical stacking (the socket frame above the CPU, covers above what they hide, the HP
  optical bracket above the board-side parts under it), so a visible mask is what the scanner really saw of that part.

## 7. Quick start

```
pip install numpy pillow huggingface_hub
huggingface-cli download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset \
    --include "annotations/scanner/*" --local-dir e3d
cd e3d/annotations/scanner
python tools/parts3d.py where --pack . --desktop 22 --frames 1,11,12 --download ../../pictures
# reconstruct those captures; save each as an organized cloud D22_S<nn>_cloud.npy (section 2)
python tools/parts3d.py extract --pack . --desktop 22 --frame 1 --cloud D22_S01_cloud.npy --out parts/
python tools/parts3d.py diff --pack . --desktop 22 --part psu.01 --cloud-before D22_S11_cloud.npy \
    --cloud-after D22_S12_cloud.npy --out parts/
```

Then go down `parts_3d_plan.csv` desktop by desktop: `clean` rows with `extract` on `best_frame`, `diff` rows with
`diff`, `visible` rows with `extract` on `best_frame` (partial), skip `none`. `annotations/scanner/README.md` documents
every field in full; questions to Chang Liu (ChangChrisLiu on Hugging Face / GitHub).
