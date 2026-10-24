"""Per-step picture evidence for the data audit (task U25).

For every logged step ``s`` of a desktop: how much the scanner frame changed from
step ``s-1`` to step ``s`` (and from the first step to the last), on the
chassis ROI, at most 256 px. The later frame is matched to the earlier one in
brightness and contrast (10th / 90th percentiles), so an exposure change is not a change;
a pixel changed when the 5x5-blurred absolute difference exceeds
:data:`CHANGE_GRAY` grey levels, and specks are opened away. The result per
step: the changed fraction, and the bounding box (full-resolution scanner
pixels) and fraction of the largest changed blob. No registration: the scanner
is fixed to the bench (one pose segment on every desktop).

The shot of each burst is the one the annotator shows: the local cache's
choice (:func:`tda.core.cache.choose_scan_image`, re-run on the burst metrics
in ``<cache>/scan/D<nn>/manifest.json``, read only), so an unexposed, blown-out
or hand-in-frame ``P_0`` is not compared. A frame the manifest does not know
falls back to ``P_0``, or to the burst's first exposed shot when ``P_0`` is black.

``python scripts/step_changes.py --index <index.json> --db <snapshot> --out <dir>``
caches the frames and writes ``<out>/D<nn>.json``; :func:`load_or_compute` is
what ``scripts/data_audit.py`` calls.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tda.core import rawroot  # noqa: E402
from tda.core import scan_align as A  # noqa: E402

#: A pixel changed when its blurred |difference| exceeds this many grey levels
#: (after the brightness/contrast match). A removed motherboard, PSU or drive
#: clears it everywhere; the scanner's own frame-to-frame noise does not.
CHANGE_GRAY = 25.0
#: Bump this when the measure changes, so cached results are recomputed.
VERSION = 5
#: A small frame darker than this on average is an unexposed scan (a black
#: ``P_0``); the next shot of the burst that is not stands in for it, as
#: :func:`tda.core.cache.choose_scan_image` does for the local cache.
DARK_MEAN = 45.0


def _norm(path) -> str:
    return str(path).replace("\\", "/")


def cache_choices(cache_dir: Optional[str], desktop: int) -> dict[str, tuple[str, str]]:
    """``{burst shot path: (chosen shot path, reason)}`` from the local cache's manifest.

    Re-decided with today's rule on the metrics the manifest holds, so a manifest
    written under an older rule answers as a fresh ``build-cache`` would.
    """
    from tda.core.cache import MANIFEST_NAME, choose_scan_image

    if not cache_dir:
        return {}
    path = os.path.join(cache_dir, "scan", f"D{desktop:02d}", MANIFEST_NAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    out = {}
    for rec in manifest.values():
        metrics = rec.get("metrics") if isinstance(rec, dict) else None
        if not metrics:
            continue
        chosen, reason = choose_scan_image(metrics)
        for m in metrics:
            out[_norm(m["path"])] = (_norm(metrics[chosen]["path"]), reason)
    return out


def _short(shot: str) -> str:
    return "/".join(_norm(shot).rsplit("/", 2)[-2:])


def _exposed(small: Optional[np.ndarray], ff, roi,
             choices: Optional[dict] = None) -> tuple[Optional[np.ndarray], Optional[str]]:
    """``(frame, why another shot of the burst stands in)``: the cache's shot when the
    manifest knows the burst, else ``P_0`` unless it is unexposed."""
    chosen = (choices or {}).get(_norm(ff.path))
    if chosen is not None:
        src, reason = chosen
        if src == _norm(ff.path):
            return small, None
        alt = A.load_frames([rawroot.resolve_raw(src) or src], "scan", roi, None)[0]
        if alt is not None:
            return alt, f"{_short(src)} ({reason.replace('p0_', 'P_0 ')})"
    if small is None or float(small.mean()) >= DARK_MEAN:
        return small, None
    for shot in (ff.aux or {}).get("burst") or []:
        if shot == ff.path:
            continue
        alt = A.load_frames([rawroot.resolve_raw(shot) or shot], "scan", roi, None)[0]
        if alt is not None and float(alt.mean()) >= DARK_MEAN:
            return alt, f"{_short(shot)} (P_0 dark)"
    return small, None


def _matched(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``x`` mapped linearly so its 10th and 90th percentiles are ``ref``'s.

    Percentiles, not the median and MAD: a scanner frame is two- or three-toned
    (board, floor, background), and when a removed part moves the balance across
    50 % the MAD jumps and the whole frame would read as changed.
    """
    x10, x90 = np.percentile(x, (10, 90))
    r10, r90 = np.percentile(ref, (10, 90))
    gain = (r90 - r10) / max(float(x90 - x10), 1.0)
    return (x - x10) * gain + r10


def measure(a: np.ndarray, b: np.ndarray, roi: Optional[list[int]], shape) -> dict:
    """Changed fraction and the largest changed blob between two small frames."""
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    if fa.shape != fb.shape:
        fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.blur(np.abs(_matched(fb, fa) - fa), (5, 5))
    mask = (diff > CHANGE_GRAY).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = {"frac": round(float(mask.mean()), 5), "blob": None, "blob_frac": 0.0}
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = (int(v) for v in stats[k])
        # back to full-resolution scanner pixels (the ROI was cut, then scaled)
        x0, y0, x1, y1 = roi if roi else (0, 0, shape[1], shape[0])
        sx, sy = (x1 - x0) / mask.shape[1], (y1 - y0) / mask.shape[0]
        out["blob"] = [int(x0 + x * sx), int(y0 + y * sy), int(x0 + (x + w) * sx),
                       int(y0 + (y + h) * sy)]
        out["blob_frac"] = round(area / mask.size, 5)
    return out


def load_or_compute(desktop: int, di, roi: Optional[list[int]], out_dir: str,
                    cache_dir: Optional[str] = None) -> dict:
    """``{"steps": {s: measure(s-1 -> s)}, "first_last": measure(1 -> n), ...}``.

    ``cache_dir`` is the local frame cache whose burst choices are followed
    (``None``: ``P_0``, or the first exposed shot when it is black).
    """
    path = os.path.join(out_dir, f"D{desktop:02d}.json")
    frames = sorted(((k.step, ff) for k, ff in di.frames.items() if k.view == "scan"),
                    key=lambda kv: kv[0])
    paths = [rawroot.resolve_raw(ff.path) or ff.path for _s, ff in frames]
    choices = cache_choices(cache_dir, desktop)
    key = {"version": VERSION, "roi": roi, "paths": [ff.path for _s, ff in frames],
           "chosen": [choices.get(_norm(ff.path), ("", ""))[0] for _s, ff in frames]}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            held = json.load(fh)
        if held.get("key") == key:
            return held
    small = A.load_frames(paths, "scan", roi, out_dir, f"D{desktop:02d}_steps")
    shape = (1600, 1600)
    by_step, dark = {}, {}
    for (s, ff), img in zip(frames, small):
        by_step[s], used = _exposed(img, ff, roi, choices)
        if used:
            dark[str(s)] = used
        elif by_step[s] is not None and float(by_step[s].mean()) < DARK_MEAN:
            dark[str(s)] = "P_0 dark, no exposed shot in the burst"
    steps = {}
    for s in sorted(by_step):
        if s - 1 in by_step and by_step[s] is not None and by_step[s - 1] is not None:
            steps[str(s)] = measure(by_step[s - 1], by_step[s], roi, shape)
    result = {"key": key, "steps": steps, "first_last": None, "dark_p0": dark,
              "first": min(by_step) if by_step else None, "last": max(by_step) if by_step else None}
    if by_step:
        result["first_last"] = measure(by_step[min(by_step)], by_step[max(by_step)], roi, shape)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return result


def main(argv=None) -> int:
    import argparse

    from tda.cli_common import load_paths
    from tda.core.index import load_index

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--desktops", default=None)
    ap.add_argument("--cache", default=None,
                    help="local frame cache whose scanner burst choices to follow "
                         "(default: cache_dir of paths.yaml; read only)")
    args = ap.parse_args(argv)
    cfg = load_paths(str(REPO / "configs" / "paths.yaml"))   # also finds the raw drive
    cache_dir = args.cache or cfg.get("cache_dir")
    index = load_index(args.index)
    wanted = sorted(index) if not args.desktops else [int(d) for d in args.desktops.split(",")]
    scanner_root = rawroot.resolve_raw(cfg["scanner_root"])
    for d in wanted:
        di = index[d]
        # the ROI scripts/data_audit.py uses (scan_align.analyse's), so both share the cache
        folders = [p for _n, _name, p in A.scanner_folders(scanner_root, d)]
        roi = A._roi(args.db, d, "scan", folders, True)[0]
        res = load_or_compute(d, di, roi, args.out, cache_dir)
        print(f"D{d:02d}: {len(res['steps'])} steps, first->last {res['first_last']['frac']:.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
