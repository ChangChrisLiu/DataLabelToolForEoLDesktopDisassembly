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

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests -q -o addopts= --basetemp=<a folder on D:>
```

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
