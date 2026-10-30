# Teardown Annotator (`tda`)

A local, single-user annotation tool for a desktop-PC disassembly dataset: per-step
images from four cameras (a structured-light scanner, two OAK-D cameras, a RealSense),
the capture logs, and the labels built on top of them — instance masks with layering,
per-step part states, component relations and disassembly constraints, and the
question/answer records derived from all of that for vision-language models.

This repository holds the tool, its tests, its design documents and the experiment
code. It holds **no data**: images, the annotation database, model weights and
experiment outputs live outside the checkout (see `configs/paths.yaml`) and are
ignored by git.

## Running it

Windows, Python environment `tda` under `D:\Anaconda\envs\tda` (see
`envs/environment-tda.md` for how it was built and verified):

```bat
start_tda.bat --desktop 13 --view scan
```

`start_tda.bat` sets the cache locations and opens the annotator on one machine and
one camera view; without arguments it reopens where the annotator last stopped. The
command-line entry point behind it is `python -m tda.cli` (`app`, `status`, `check`,
`import-logs`, `export-coco`, `export-vlm`, `constraints`, …).

Tests:

The standalone dual-camera video refiner opens with `start_video_refiner.bat`.
It loads preliminary action segments, edits original-video boundaries, aligns
OAK and RealSense with manually selected source-frame anchors, and exports paired
clips. Its JSON projects are separate from the annotation database. See the
[Chinese video refinement guide](docs/video_refiner_guide.md).

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests -q -o addopts= --basetemp=<a folder on D:>
```

## Data

The images are not in this repository. They are published as a Hugging Face dataset,
**E3D: End-of-Life Desktop Disassembly Dataset** —
https://huggingface.co/datasets/ChangChrisLiu/E3D-Desktop-Disassembly
(public; raw captures of all 66 desktops; the annotations and the edited operator
videos follow later under `annotations/` and `videos/`).

```
raw/scanner/<batch>/<batch>/<NN>/RGB<n>1/P_0..P_9.png   structured-light scanner, top-down 1600x1600  (103 GB, 33k files)
raw/scanner/<batch>/<batch>_allcompoentsrgb/...           scanner all-components scans
raw/oak/Desktop NN/Disassemble|Components/Camera_1|2/NNN/ OAK-D: 12 MP jpg, aligned png, depth, ply   (212 GB, 35k files)
raw/realsense/Desktop NN/...                              RealSense: colour, depth, ply                (57 GB, 21k files)
taxonomy.yaml                                             the part taxonomy (same file as configs/taxonomy.yaml)
annotations/scanner/D<NN>/                                scanner-view labels and metadata of the finished desktops (57 on 2026-10-29)
```

### 1. Install the tool

```bash
git clone git@github.com:ChangChrisLiu/DataLabel.git D:\DataSet
# Python 3.11 environment: see envs/environment-tda.md (torch, PySide6, SAM 2.1 …; envs/requirements-lock.txt)
```

### 2. Pick what to download

Everything together is ~370 GB, so download by view and by desktop. The `hf` CLI comes
with `huggingface_hub` (already in the `tda` environment; `pip install huggingface_hub`
elsewhere); no login is needed for a public dataset.

```bash
# the scanner view of every desktop (the main view)
hf download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset    --include "raw/scanner/**" --local-dir D:/E3D

# one desktop, all views (desktop 13: the scanner keeps it under its batch folder)
hf download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset    --include "raw/scanner/*/*/13/**" "raw/oak/Desktop 13/**" "raw/realsense/Desktop 13/**" --local-dir D:/E3D

# the taxonomy only
hf download ChangChrisLiu/E3D-Desktop-Disassembly --repo-type dataset --include taxonomy.yaml --local-dir D:/E3D
```

The same in Python: `huggingface_hub.snapshot_download(repo_id, repo_type="dataset",
allow_patterns=[...], local_dir=...)`. Downloads resume; re-running a command only
fetches what is missing. Keep `--local-dir` on a data drive (the tool keeps every cache
and temp file on D: as well).

### 3. Point the tool at the data

`configs/paths.yaml` names the three capture roots. For a download under `D:/E3D`:

```yaml
f_root:        "D:/E3D/raw"
scanner_root:  "D:/E3D/raw/scanner"
oak_root:      "D:/E3D/raw/oak"
rs_root:       "D:/E3D/raw/realsense"
raw_marker:    "E3D/raw"          # the folder the drive resolver looks for when the drive letter changes
```

(The lab's own copy keeps the capture rigs' original folder names, `UGA DATA`,
`OAKD Capture/Desktop_Datacollection/DesktopData` and
`Realsense Capture/Dataset Information/Exp`; the three roots above are the same
folders renamed, so only the roots change.)

### 4. Build the index and the frame cache

```bash
python -m tda.cli build-index              # scans the roots, writes cache/index.json (which capture is which step)
python -m tda.cli load-index               # frame rows and pose segments into the database
python -m tda.cli build-cache --views scan # local copies of the chosen frames (one per step)
python -m tda.cli app --annotator <name> --desktop 13 --view scan
```

`build-index` only needs the images; `configs/index_fixes.yaml` (in this repository)
carries the per-desktop capture-to-step corrections. The published annotations
(`annotations/scanner/D<NN>/`) are the scanner packages `scripts/export_scanpack.py`
writes from the annotation database: label maps, COCO with amodal fields, the parts
and relations, the step sequence and `frames.json`, whose `source_folder` +
`file_name` name the picture under `raw/scanner/`. The database itself and the step
logs stay private; the packages are what a consumer needs.

## Where things are

| Path | What |
|---|---|
| `tda/core/` | the data model, the database, the mask compiler, the truth table, states, the constraint graph, exports |
| `tda/ui/` | the PySide6 window: canvas and tools, task card, guide, Steps and Review modes |
| `tda/models/` | SAM 2.1 service, the small-part detector (RF-DETR / YOLO backends), the loader gate |
| `configs/` | taxonomy, paths, SAM decoder heads, detector and prompt-gate settings, area bands |
| `docs/HANDOFF_2026-10-28.md` | **start here**: the hand-off for whoever takes over (data locations, how to read the database, exports, state, rules; Chinese) |
| `docs/annotation_guide.md` | the annotator's manual (Chinese) |
| `docs/trial_quickstart.md` | the step-by-step walk-through for the first machine |
| `docs/superpowers/specs/` | the design specification (v1.7) |
| `docs/superpowers/plans/` | the implementation plans |
| `experiments/` | measure-first experiments with their reports (`*/REPORT.md`) |
| `envs/` | environment build notes and the verification script |
| `tests/` | the test suite (synthetic scenes; no real data needed) |

## Working rules

- The annotation database (`annotations/tda.sqlite`) is opened only by the annotator's
  own session; every script, worker or review runs on a copy under `.cache/tmp/`.
- The raw image drive is read-only; the only thing written there is `TDA_backups`.
- Everything the tool produces — caches, logs, backups, temp files — stays on D:.

## Status

Private, pre-publication. The design, the documents and the data samples under
`docs/img` and `tests/fixtures` are part of unpublished research; do not redistribute.
