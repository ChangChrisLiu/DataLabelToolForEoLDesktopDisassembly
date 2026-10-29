# Video Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a local dual-camera video editor using existing preliminary segments, manual synchronization, precise boundary adjustment and paired exports.

**Architecture:** Independent `tda.video` package; pure project/synchronization model, PyAV timestamp decoder, FFmpeg exporter, Qt UI. OAK is the master clock; project storage never touches the annotation DB.

**Tech Stack:** Python 3.11, PySide6, PyAV, FFmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-video-refiner-design.md`

## Global Constraints

- All outputs/cache/temp on D:, raw media read only, no annotation database access.
- Native source PTS; half-open ranges; no inferred sync until user marks an anchor pair.
- Standalone launcher; default 58 Desktop 7 segments; correction 275.5–288.5 retained.
- 30fps silent H.264 exports; affine correction retimes the secondary output to master duration.

## Review Focus

- Variable-rate frames: step to actual previous/next PTS, not average-rate arithmetic (Task 2).
- Missing media or out-of-range mapped cuts: explain and block export, never clamp silently (Tasks 1, 3).
- Crash/save interruption and repeated launch: atomic JSON/backup and one writer per project (Tasks 1, 4).
- Input boundary edit: reject zero/reversed cuts and invalidate prior confirmation (Tasks 1, 4).
- Cancellation/failure during render: no completed marker or overwriting previous export (Task 3).

## Task 1: Project and synchronization model

Files: `tda/video/project.py`, `tests/test_video_project.py`.
Interfaces: `Project.from_manifest(path, secondary_path)`, `Project.load(path)`, `save(path)`, `mapping()` -> `(a,b)`, `mapped_range(segment)`, `validate_segment(segment)`. `History` snapshots support undo/redo.
- [x] Write tests: import preserves metadata and correction; anchor (74,70) maps 115–121 to 111–117; second anchor (1074,1071) produces a=1.001; ranges outside secondary duration rejected; invalid cuts rejected; save/load and undo preserve manual changes and source baseline.
- [x] Run tests RED; implement pure model and atomic storage; run GREEN.
- [x] Commit model and tests.

## Task 2: Source timestamp decoder

Files: `tda/video/media.py`, `tests/test_video_media.py`.
Interfaces: `probe(path)`, `FrameReader.frame_at(seconds)` -> frame with `pts`, `end`, `rgb`; `previous(pts)`, `next(pts)`; `close()`.
- [x] Generate short deterministic VFR fixture with PyAV; test floor selection, next/previous, seek backward, start/end, corrupt source error.
- [x] Run RED; implement sequential decode with keyframe seek, actual PTS and bounded frame caching; run GREEN.
- [x] Commit decoder and tests.

## Task 3: Paired export

Files: `tda/video/export.py`, `tests/test_video_export.py`.
Interfaces: `export_project(project, output_dir, ids, paired, cancel, progress)` -> manifest path; validate before creating output; accepts callback and cancellation event.
- [x] Test synthetic media: offset pairing, drift-retimed duration, ROI size, 30fps, source timestamp metadata, no-sync rejection, range rejection, destination reuse rejection, canceled job incomplete.
- [x] Run RED; implement FFmpeg accurate re-encode and probe validation with JSON/CSV manifests and per-job status; run GREEN.
- [x] Commit exporter and tests.

## Task 4: Desktop editor and delivery

Files: `tda/video/app.py`, `tda/video/widgets.py`, `tda/video/workers.py`, `tda/video/__main__.py`, `start_video_refiner.bat`, `docs/video_refiner_guide.md`, `tests/test_video_ui.py`; update README, dependency declaration, spec changelog and existing changelog acceptance tests.
Interfaces: `VideoRefiner(project, project_path)` owns project state, lock, history, workers and autosave. `main(argv)` defaults to Desktop 7 manifest/project under exports.
- [x] Test UI workflow on real widgets with synthetic media: edit/undo/save/reload, confirmation invalidation, split and exclude, sync pair capture, ROI mapping, lock conflict, export enablement.
- [x] Run RED; implement independent two-camera canvas/player, list, time controls, synchronization panel, ROI selection, storage and threaded export; run GREEN.
- [x] Launch actual UI with Desktop 7; inspect screenshot and exercise controls; export real short paired preview under temporary explicit test sync, do not persist guessed alignment in user's project.
- [x] Run relevant tests and full project regression command; independent final code review, fix important findings, record verification; commit and local no-ff merge per repository convention.

## Execution ledger

User delegated all details and explicitly opted out of further design questions. Execute natively in this session. Work on `codex/video-refiner` in the clean D: checkout to honor its fixed D: runtime/data configuration; no remote push. Progress and verification are recorded here and in the task's D: scratch directory.

### Verification ledger (2026-10-29)

- Task 1 complete: import/validation/atomic backup/history/synchronization tests RED→GREEN, 16 tests initially; later invalidation regression added. Commit 35a8f67.
- Task 2 complete: synthetic VFR PTS navigation, backward seek and end boundaries RED→GREEN. Commit 45ff41d. Real OAK and RS source timestamp checks passed. Added 48-frame decode cache after real RS backward seek took seconds; reverse step now 0.0033 s in observed run.
- Task 3 complete: real FFmpeg synthetic paired offset and affine exports, duration/ROI/native provenance and cancellation tests RED→GREEN. Commit 45ff41d. Real 1-second OAK+RS pair exported and decoded to 720×480; artificial sync confined to `.cache/tmp/video-real-export`, never written to user's project.
- Task 4 implementation complete: actual Qt UI opened with both full source files and 58 Desktop 7 cuts. OAK native frame step observed 74.000 → 74.0166667 seconds. Standalone UI workflow tests RED→GREEN, launcher and Chinese guide added. Commit d48a5b0.
- Final review: fresh gpt-6-astra read-only reviewer; no Critical findings. Important fixes: preserve exact source PTS when untouched form fields round for display; invalidate confirmations for changed sync/ROI with undo restoration; honor cancellation when final encoder finishes. Each regression observed RED then GREEN. Extra usability regression: synchronization tab plays full original video without loop-to-current-cut reset. Current video tests: 33 passed.
- Final minor (deferred): changing start times does not reorder the existing list; original list order is retained, while export sorts by source start and retains UID/log step. Documented in guide.
- Ruling: use independent local PySide6 tool with PyAV and original-source JSON edits — user delegated details and explicitly required original-video synchronization — costs a separate window from image annotation.
- Ruling: all work in clean D: checkout on codex/video-refiner; all output/temp on D:, raw drive read-only, no database access — respects this workstation's fixed data/runtime paths — no separate worktree copy.
- Review items the reviewer declined to judge: actual chosen physical synchronization points (left unset for the user), power-loss durability (fsync + atomic replace + previous backup, no absolute hardware guarantee), full regression run (pending recorded result below).

- Full regression completed: 5,558 passed, 20 skipped, 24 performance tests deselected per repo command; 16 failures were historical changelog assertions held in the process that began before those files were updated. Failure names: test_l6_docs::test_v1_30_is_on_top_of_14; test_u14_docs::test_the_spec_has_the_paragraph_and_the_entry; test_u19_docs::test_the_spec_has_the_paragraph_and_v1_19_on_top; test_u20_docs::test_the_spec_has_the_paragraph_in_3_2_and_v1_20_on_top; test_u21_docs::test_the_spec_has_the_paragraph_in_3_2_and_v1_21_on_top; test_u22_keys::test_the_spec_has_v1_22_on_top_with_the_whole_table; test_u23_docs::test_the_spec_has_the_paragraph_in_4_5_and_v1_23_on_top; test_u24_docs::test_the_spec_has_the_fifth_queue_viewpoint_and_v1_24_on_top; test_u25_docs::test_v1_25_is_on_top_of_14; test_u26_docs::test_v1_26_is_in_the_changelog_above_v1_24; test_u27_docs::test_the_spec_has_the_paragraph_in_3_2_and_v1_27_on_top; test_u28_docs::test_the_spec_has_the_paragraph_after_u23_and_v1_28_on_top; test_u29_docs::test_v1_29_is_on_top_of_14; test_u32_docs::test_v1_31_is_on_top_of_14; test_u33_docs::test_v1_32_is_near_the_top_of_14; test_u38_docs::test_v1_37_is_on_top_of_14. Fresh-process rerun of all affected document modules: 175 passed, including every failed case. Fresh final video tests after fixes: 33 passed. Logs: `.cache/tmp/video-regression.log`, `.cache/tmp/video-doc-tests.log`. No unresolved functional failures.
- Launcher re-tested through start_video_refiner.bat: visible pythonw window opens 58 source cuts, source timestamps and ROI display correctly; synchronization page opens independently and disables cut looping. User project retains corrected sixth cut end 288.5 and no inferred sync points.
