"""Per-desktop data audit over all 66 desktops (task U25) -> a Markdown report.

::

    python scripts/data_audit.py --index <index.json> --db <snapshot.sqlite> \
        --out docs/data_audit_<date>.md --render <dir> [--cache <dir>] [--desktops 1-66]

Read-only on the raw drive and the database (opened ``mode=ro`` for the ROIs);
writes the report, the frame cache and the rendered frames only. For every
desktop: the sheet (rows, ``dupli`` rows, collapsed steps), the folder counts on
the drive, the gallery shots, the known anomalies (``raw_logs/oak_survey``), the
``configs/index_fixes.yaml`` entries, the index's issues, the sheet notes that
talk about rescans or missing shots, and :mod:`tda.core.scan_align`'s verdict for
the scanner against OAK Camera_1 -- and against the RealSense as a second
opinion when that sequence is complete. Flagged desktops get their decisive
frames rendered under ``<render>/D<nn>/``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tda import pipeline as P  # noqa: E402
from tda.cli_common import load_paths  # noqa: E402
from tda.core import rawroot  # noqa: E402
from tda.core import scan_align as A  # noqa: E402
from tda.core.index import load_fixes, load_index  # noqa: E402
from tda.core.logs import sheet_rows_by_desktop  # noqa: E402
from tda.core.model import COMPONENT_VIEWS  # noqa: E402
from tda.core.sources import DEFAULT_FOLDER_ALIASES, STEP_DIR_RE, dirs, first_existing  # noqa: E402

SURVEY_FIELDS = ("n_steps_cam1", "n_steps_cam2", "incomplete_steps", "ts_mismatch_steps",
                 "steps_only_one_cam")

#: What was settled by looking at the frames (U25, 2026-10-24): the decisive
#: crops are the ``where``-style "scanner changes" images; numbers are scanner
#: folders (RGB<n>1) and OAK rows (the sheet's numbering, dupli rows included).
EYES: dict[int, str] = {
    1: "Folders are named by row; both dupli rows were photographed (RGB151, RGB331). "
       "**Camera_1/Camera_2/RealSense capture 40 -- the 40th, which the 39-row sheet does not "
       "have -- is the removed parts laid out on a monitor box, not a teardown step** (the "
       "`rs_extra_from Components/001` comment says the opposite): it belongs in the "
       "components gallery, and the index's step 38 should not exist.",
    18: "RGB91 = row 9 (optical drive out) but RGB391 = row 40 (PSU out, OAK row 39 still "
        "has it): dupli row 30 skipped. RGB411 -> RGB421 changes nothing on the board while "
        "every later folder loses one motherboard screw: RGB421 re-shoots dupli row 43. "
        "Declared `duplicate_folders: [42]`.",
    22: "NOT declared. The magnitudes tie 'skipped dupli 27' (0.616) and 'skipped dupli 32' "
        "(0.614). The RAM latches decide it: RGB271 has all four closed, RGB281 one open, "
        "RGB291 two (rows 28, 29 = RAM clip 1, 2), and RGB321 already has RAM 1 out (row 33) "
        "-- so folders 27-31 are rows 27-31 and the scanner skipped dupli row 32 "
        "('replacement of 31'), having photographed dupli 27 as RGB271. Proposed "
        "`scan.22.duplicate_folders: [27]` -- please confirm on `D22/ram_latches.png`.",
    24: "RGB421 is the empty chassis of row 43: one row was skipped before it; the sheet's "
        "dupli 37 ('replace the result of 36') is it. Declared `[]`.",
    28: "RGB91 already has the heatsink out, which OAK shows from row 10: one row skipped "
        "before folder 9 -- dupli 6. Declared `[]`. RGB41's P_0 is unexposed.",
    31: "The sheet says 'dupli (scanner also scans)' and the folders follow OAK one to one "
        "(RGB101 = row 10 PSU opened; RGB121/131/141 = SSD lifted out / back / removed, like "
        "OAK rows 12/13/14). The magnitudes alone preferred a skip+re-shot pair the frames "
        "refute. Declared `[13]`, and row 13 / RGB131 replace the bad row-12 captures.",
    36: "RGB421 = row 43 (PSU out, OAK row 42 still has it); the sheet: 'Dupli (scanner does "
        "not scan)'. Declared `[]`.",
    40: "RGB371 = row 38 (motherboard out); the sheet: 'dupli (scanner does not scan)'. "
        "Declared `[]`.",
    41: "**Not a missing last capture**: RGB341 is the empty chassis (row 35). Aligned "
        "through row 25 (SSD at 19->20, PSU at 22->23, CPU at 24->25 as in OAK); RGB331 "
        "already has row 34's connector unplugged. So the scanner skipped one of the "
        "motherboard-screw steps, rows 26-33 -- a real step, which `duplicate_folders` "
        "cannot express (it needs a skipped-step rule). No rule yet.",
    48: "RGB241 = row 24 (PSU out), RGB251 = row 26 (a RAM clip open): the skip is dupli "
        "row 25 exactly, as the sheet says ('Dupli (scanner does not scan)'). The "
        "magnitudes put it at rows 18-23. Declared `[]`.",
    52: "RGB51 -> RGB61 changes nothing on the board (the re-shot of dupli 6, 'scanner scans "
        "too'); RGB41 -> RGB51 is the CPU fan. Declared `[6]`.",
    55: "RGB481 = row 49 (motherboard out); the sheet: 'dupli(Scanner does not scan)'. "
        "Declared `[]`.",
    60: "NOT declared. RGB31 already has the case fan out (row 4): dupli row 3 ('dupli "
        "scan') was skipped. 47 folders for 46 steps, so one folder re-shoots a state: the "
        "magnitudes say RGB281 (next RGB291, bootstrap 0.82); 'Scanner rescan' notes sit on "
        "rows 29/30/33. RGB271 -> RGB281 shows no part change in its two strongest "
        "differences. Probably `duplicate_folders: [28]` -- please confirm. OAK row 2 shows "
        "the case fan lifted mid-operation and row 3 has it back in place (a re-take?).",
    61: "**Not a missing last capture**: RGB391 is the empty chassis (row 40). Aligned at row "
        "11 (SSD), shifted by row 28 (the CPU is already out in RGB271): one step of rows "
        "12-27 was never scanned (the PSU connectors 14/15 or the case connectors 20-26; "
        "the magnitudes favour ~23). Needs a skipped-step rule. No rule yet.",
    62: "RGB271 -> RGB281 removes SATA 3 (row 29), RGB281 -> RGB291 the card cover (row 30), "
        "then one card per folder: the skip is dupli row 28 exactly, as the sheet says. The "
        "magnitudes favour a skip near row 18 -- wrong. Declared `[]`; OAK row 27 was shot "
        "under the scanner's blue light, so row 28 replaces it.",
    65: "Aligned: RGB271 -> RGB281 and RGB281 -> RGB291 each open a RAM clip (rows 28, 29), "
        "RGB301 -> RGB311 the CPU locker (row 31), RGB311 -> RGB321 the CPU (row 32). The flag "
        "comes from **RGB331, whose P_0 is black (unexposed)**: an outlier change, not a "
        "misalignment -- check that the cache chose another shot of that burst.",
    9: "Counts match and the end is in step (RGB461 and OAK row 46 are the first without the "
       "motherboard). The low score is the OAK view: no chassis box found, so the whole "
       "frame (operator, tools) is compared.",
    37: "Counts match; the last two folders and the last two rows are all the empty chassis. "
        "The OAK chassis detector finds no box on D37-D63 (the whole frame is compared), "
        "which is what keeps the score low.",
    54: "Counts match and the end is in step (RGB461 and OAK row 46 are the first without the "
        "motherboard); the OAK view is compared as a whole frame.",
    66: "Listed only for its scanner gap: step 13 is the sheet's 'All components' row, the "
        "parts laid out (U10 moved its OAK/RealSense photos into the components gallery); the "
        "scanner never photographed it. Aligned otherwise (0.92). Nothing to fix.",
}

#: A declared mapping that correlates below this on a desktop whose counts match
#: is listed separately: a skip compensated by an extra shot has no count symptom.
POOR_DECLARED = 0.35

# --------------------------------------------------------------------------- #
# picture evidence per step (scripts/step_changes.py)
# --------------------------------------------------------------------------- #
#: (a) a part that leaves or moves should show: these classes are big enough to
#: see from above. RAM modules stand on edge and barely show at 256 px (69 of the
#: 80 removals under 0.5 % were RAM), so they are not held to it; nor are CPUs
#: (about 8 px square at 256 px: D44 step 28 is a visible removal at 0.4 %) and
#: expansion cards (edge-on in a chassis lying on its side, like RAM: D39 step 44,
#: D62 step 30).
VISIBLE_PARTS = frozenset({"motherboard", "psu", "storage_drive", "optical_drive", "case_fan",
                           "cpu_cooler", "drive_cage", "cover", "cooler_bracket"})
NO_CHANGE = 0.005
#: (b) the most a step may change when nothing leaves: per verb (opening a
#: latch, not a cover -- a cover swinging open is a large change by design).
QUIET_VERBS = {"unscrew": 0.15, "release": 0.15, "disconnect": 0.25, "open": 0.20}
LATCHES = frozenset({"ram_latch", "cpu_socket_lever", "psu_latch", "drive_latch", "card_latch",
                     "cooler_latch", "cable_clip"})
#: (c) the first and the last scanner frame of a teardown must differ this much.
FIRST_LAST = 0.15


def picture_flags(rec: dict, changes: dict, li) -> list[dict]:
    """The steps whose scanner change contradicts what the sheet says happened."""
    out = []
    acts: dict[int, list[tuple[str, str, str]]] = {}
    for a in li.actions:
        cls = li.instances[a.target].cls if a.target in li.instances else str(a.target)
        acts.setdefault(a.step, []).append((a.verb, cls, a.target))
    names = {s.step: s.raw_name for s in li.steps}
    for key, m in sorted(changes.get("steps", {}).items(), key=lambda kv: int(kv[0])):
        s = int(key)
        here = acts.get(s, [])
        verbs = {v for v, _c, _t in here}
        leaving = [(v, c) for v, c, _t in here if v in ("remove", "displace") and c in VISIBLE_PARTS]
        if leaving and m["frac"] < NO_CHANGE:
            out.append({"step": s, "kind": "a", "name": names.get(s, ""), "m": m,
                        "why": f"{'/'.join(sorted({v for v, _ in leaving}))} "
                               f"{', '.join(sorted({c for _, c in leaving}))} but no visible change"})
            continue
        quiet = here and all(v in QUIET_VERBS and (v != "open" or c in LATCHES)
                             for v, c, _t in here)
        if quiet:
            limit = max(QUIET_VERBS[v] for v in verbs)
            if m["frac"] > limit:
                out.append({"step": s, "kind": "b", "name": names.get(s, ""), "m": m,
                            "why": f"only {'/'.join(sorted(verbs))} but {m['frac']:.0%} of the "
                                   f"chassis changed (limit {limit:.0%})"})
    fl = changes.get("first_last")
    if fl is not None and fl["frac"] < FIRST_LAST:
        out.append({"step": changes.get("last"), "kind": "c", "name": "first vs last", "m": fl,
                    "why": f"the first and the last scanner frame differ by only {fl['frac']:.0%}"})
    return out


# --------------------------------------------------------------------------- #
# facts about one desktop
# --------------------------------------------------------------------------- #
def folder_counts(roots: dict, desktop: int) -> dict[str, Optional[int]]:
    """Step folders on the drive: Camera_1, Camera_2, RealSense, scanner."""
    out: dict[str, Optional[int]] = {}
    oak = first_existing(os.path.join(roots["oak_root"], f"Desktop {desktop}"),
                         list(DEFAULT_FOLDER_ALIASES))
    for cam in ("Camera_1", "Camera_2"):
        path = os.path.join(oak, cam) if oak else None
        out[cam] = (sum(1 for e in dirs(path) if STEP_DIR_RE.match(e.name))
                    if path and os.path.isdir(path) else None)
    rs = first_existing(os.path.join(roots["rs_root"], f"Desktop {desktop}"),
                        list(DEFAULT_FOLDER_ALIASES))
    out["rs"] = sum(1 for e in dirs(rs) if STEP_DIR_RE.match(e.name)) if rs else None
    out["scan"] = len(A.scanner_folders(roots["scanner_root"], desktop))
    return out


def survey(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {int(r["desktop"].split()[-1]): r for r in csv.DictReader(fh)}


def anomalies(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            head, _, text = line.partition(":")
            if head.startswith("Desktop "):
                out.setdefault(int(head.split()[1]), []).append(text.strip())
    return out


def fix_entries(fixes: dict, desktop: int) -> list[str]:
    out = []
    for section in ("oak", "scan", "rs"):
        entry = (fixes.get(section) or {}).get(desktop)
        if isinstance(entry, dict):
            keys = {k: v for k, v in entry.items() if k not in ("reason", "note")}
            out.append(f"{section}: " + ", ".join(f"{k}={v}" for k, v in keys.items()))
    return out


# --------------------------------------------------------------------------- #
# the verdict of one view
# --------------------------------------------------------------------------- #
def verdict_cell(a: Optional[A.Analysis]) -> str:
    if a is None:
        return "-"
    j = a.judgement
    if j.verdict == "aligned":
        text = f"aligned {j.declared.score:.2f}"
        if j.runner_up is not None:
            text += f" (vs '{short(j.runner_up.hyp)}' {j.margin:+.2f})"
    elif j.verdict == "unknown":
        text = f"unknown (best {j.best.score:.2f})"
    else:
        where = sorted({r.hyp.skip_row for r in j.positions() if r.hyp.skip_row} |
                       ({j.answer.hyp.skip_row} if j.answer.hyp.skip_row else set()))
        text = (f"{short(j.answer.hyp)} {j.answer.score:.2f} vs declared "
                f"{j.declared.score:.2f}" + (f"; skip rows {span(where)}" if len(where) > 1 else ""))
    return text + (" **FLAG**" if j.flagged else "")


def short(h: A.Hypothesis) -> str:
    if h.kind == A.DECLARED:
        return "declared"
    parts = []
    if h.skip_row is not None:
        parts.append(f"skip row {h.skip_row}")
    if h.dup_folder is not None:
        parts.append(f"re-shot RGB{h.dup_folder}1")
    return " + ".join(parts)


def span(values: list[int]) -> str:
    return f"{min(values)}-{max(values)}" if values else ""


def rs_complete(di) -> bool:
    paths, _m, _e = A.rows_of_index(di, "rs")
    return bool(paths) and all(paths)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--db", required=True, help="database snapshot (opened read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--paths", default=str(REPO / "configs" / "paths.yaml"))
    ap.add_argument("--desktops", default="1-66")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--diffs", default=r"D:\DataSet\.cache\tmp\u25\diffs",
                    help="where the per-step picture check keeps its frames and numbers")
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    roots = {k: rawroot.resolve_raw(paths[k]) for k in ("oak_root", "scanner_root", "rs_root")}
    if any(v is None for v in roots.values()):
        print("[audit] raw data drive not connected")
        return 1
    index = load_index(args.index)
    sheets = sheet_rows_by_desktop(P.drive_dir(paths))
    fixes = load_fixes(str(REPO / "configs" / "index_fixes.yaml"))
    raw_logs = Path(paths.get("raw_logs_dir", "raw_logs"))
    surv = survey(raw_logs / "oak_survey" / "desktop_survey.csv")
    anom = anomalies(raw_logs / "oak_survey" / "anomalies.txt")
    desktops = sorted(P.parse_desktops(args.desktops) or index)

    from step_changes import load_or_compute

    from tda.core.logs import import_log, iter_desktop_csvs, read_desktop_csv
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sheet_files = dict(iter_desktop_csvs(P.drive_dir(paths)))
    records = []
    for d in desktops:
        di = index.get(d)
        sheet = sheets.get(d)
        rec = {"d": d, "di": di, "sheet": sheet, "counts": folder_counts(roots, d),
               "survey": surv.get(d, {}), "anom": anom.get(d, []), "fixes": fix_entries(fixes, d),
               "oak": None, "rs": None, "errors": [], "pics": None}
        for view in ("oak1", "rs"):
            if di is None or (view == "rs" and not rs_complete(di)):
                continue
            try:
                rec[view if view == "rs" else "oak"] = A.analyse(
                    d, di, scanner_root=roots["scanner_root"], sheet=sheet, db_path=args.db,
                    view=view, cache_dir=args.cache, n_boot=args.boot)
            except (OSError, ValueError) as exc:
                rec["errors"].append(f"{view}: {exc}")
        if di is not None and d in sheet_files:
            roi = rec["oak"].roi["scan"][0] if rec["oak"] is not None else A.db_roi(args.db, d, "scan")
            try:
                changes = load_or_compute(d, di, roi, args.diffs, paths.get("cache_dir"))
                rows, meta = read_desktop_csv(sheet_files[d])
                li = import_log(d, rows, meta, tax)
                rec["pics"] = {"flags": picture_flags(rec, changes, li),
                               "dark": changes.get("dark_p0") or {},
                               "first_last": (changes.get("first_last") or {}).get("frac")}
            except (OSError, ValueError) as exc:
                rec["errors"].append(f"pictures: {exc}")
        records.append(rec)
        o = rec["oak"]
        print(f"[audit] D{d:02d}: {verdict_cell(o)} | rs {verdict_cell(rec['rs'])}", flush=True)

    flagged = [r for r in records if needs_look(r)]
    for rec in flagged:
        rec["renders"] = render(rec, Path(args.render) / f"D{rec['d']:02d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report(records, flagged, args), encoding="utf-8")
    print(f"[audit] wrote {args.out}; {len(flagged)} desktops need a look")
    return 0


def needs_look(rec: dict) -> bool:
    o = rec["oak"]
    if o is None or rec["d"] in EYES:
        return True
    j = o.judgement
    di = rec["di"]
    scan_missing = [k for k in di.missing if k.view == "scan"] if di else []
    leftover = any("MISMATCH" in i for i in (di.issues if di else []))
    return (j.flagged or j.verdict != "aligned" or leftover
            or bool(scan_missing and not rec["fixes"]) or poor(rec))


def counts_match(rec: dict) -> bool:
    """Every step has a scanner frame and every folder found a step."""
    di = rec["di"]
    return bool(di) and not [k for k in di.missing if k.view == "scan"] \
        and not any("have no step" in i for i in di.issues)


def poor(rec: dict) -> bool:
    o = rec["oak"]
    return o is not None and counts_match(rec) and o.judgement.declared.score < POOR_DECLARED


#: The pairs that settled a desktop by eye: (view, a, b) -- scanner folders or
#: OAK rows -- rendered as "what changed" crops into ``eyes.png``.
EVIDENCE_PAIRS: dict[int, list[tuple[str, int, int]]] = {
    18: [("scan", 38, 39), ("oak1", 39, 40), ("scan", 41, 42), ("scan", 42, 43),
         ("scan", 44, 45)],
    22: [("scan", 26, 27), ("scan", 27, 28), ("scan", 28, 29), ("scan", 31, 32)],
    31: [("scan", 11, 12), ("scan", 12, 13), ("scan", 13, 14), ("oak1", 12, 13)],
    41: [("scan", 19, 20), ("scan", 22, 23), ("scan", 24, 25), ("scan", 32, 33),
         ("scan", 33, 34)],
    48: [("scan", 23, 24), ("scan", 24, 25), ("scan", 27, 28)],
    52: [("scan", 4, 5), ("scan", 5, 6)],
    60: [("scan", 2, 3), ("oak1", 2, 3), ("scan", 26, 27), ("scan", 27, 28), ("scan", 28, 29)],
    61: [("scan", 10, 11), ("scan", 15, 16), ("scan", 26, 27), ("scan", 30, 31)],
    62: [("scan", 26, 27), ("scan", 27, 28), ("scan", 28, 29), ("scan", 29, 30),
         ("oak1", 27, 28)],
    65: [("scan", 27, 28), ("scan", 28, 29), ("scan", 30, 31), ("scan", 31, 32),
         ("scan", 32, 33)],
}
#: Fixed crops (view, items, full-resolution box) for what the strongest
#: differences do not show: D22's RAM latches.
EVIDENCE_CROPS: dict[int, tuple[str, list[int], tuple[int, int, int, int]]] = {
    22: ("scan", [26, 27, 28, 29, 30, 31, 32, 33], (880, 700, 1060, 860)),
}
#: Whole frames side by side (view, items) -- the last folders against the last rows.
EVIDENCE_FRAMES: dict[int, list[tuple[str, list[int]]]] = {
    1: [("oak1", [39, 40]), ("oak2", [39, 40]), ("rs", [39, 40])],
    24: [("scan", [41, 42]), ("oak1", [42, 43])],
    28: [("scan", [8, 9]), ("oak1", [9, 10])],
    36: [("scan", [41, 42]), ("oak1", [42, 43])],
    40: [("scan", [36, 37]), ("oak1", [37, 38])],
    41: [("scan", [33, 34]), ("oak1", [34, 35])],
    55: [("scan", [47, 48]), ("oak1", [48, 49])],
    61: [("scan", [38, 39]), ("oak1", [39, 40]), ("scan", [26, 27]), ("oak1", [27, 28])],
    9: [("scan", [45, 46]), ("oak1", [45, 46])],
    37: [("scan", [44, 45]), ("oak1", [44, 45])],
    54: [("scan", [45, 46]), ("oak1", [45, 46])],
    65: [("scan", [32, 33, 34]), ("oak1", [32, 33, 34])],
}


def _source(a: A.Analysis, view: str, item: int) -> Optional[str]:
    if view == "scan":
        return dict(zip(a.folders, a.folder_paths)).get(item)
    paths, _m, _e = A.rows_of_index(a.di, view)
    p = paths[item - 1] if 1 <= item <= len(paths) else None
    return rawroot.resolve_raw(p) if p else None


def _label(view: str, item: int) -> str:
    return f"RGB{item}1" if view == "scan" else f"{view} row {item}"


def render_evidence(rec: dict, out: Path) -> list[str]:
    import cv2
    import numpy as np

    a, d, files = rec["oak"], rec["d"], []
    if a is None:
        return files
    a.di = rec["di"]
    pairs = []
    for view, x, y in EVIDENCE_PAIRS.get(d, []):
        pa, pb = _source(a, view, x), _source(a, view, y)
        if pa and pb:
            roi = a.roi.get("scan" if view == "scan" else a.view, (None, ""))[0] \
                if view in ("scan", a.view) else None
            files.append(A.render_changes([(_label(view, x), pa, _label(view, y), pb)], view,
                                          roi, str(out / f"eyes_{view}_{x}-{y}.png")))
            pairs.append(files[-1])
    tiles_rows = []
    for view, items in EVIDENCE_FRAMES.get(d, []):
        tiles = []
        for item in items:
            p = _source(a, view, item)
            img = A._read(p, A.DECODE_REDUCE.get(view, 1), colour=True) if p else \
                np.full((240, 320, 3), 40, np.uint8)
            img = cv2.resize(img, (int(img.shape[1] * 240 / img.shape[0]), 240))
            cv2.putText(img, _label(view, item), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles.append(np.pad(img, ((0, 4), (0, 4), (0, 0))))
        tiles_rows.append(np.hstack(tiles))
    if tiles_rows:
        width = max(t.shape[1] for t in tiles_rows)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "frames.png"
        cv2.imwrite(str(path), np.vstack([np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0)))
                                          for t in tiles_rows]))
        files.append(str(path))
    if d in EVIDENCE_CROPS:
        view, items, box = EVIDENCE_CROPS[d]
        tiles = []
        for item in items:
            p = _source(a, view, item)
            if not p:
                continue
            img = cv2.imread(p, cv2.IMREAD_COLOR)[box[1]:box[3], box[0]:box[2]]
            img = cv2.resize(img, (int(img.shape[1] * 260 / img.shape[0]), 260))
            cv2.putText(img, _label(view, item), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            tiles.append(np.pad(img, ((0, 4), (0, 4), (0, 0))))
        out.mkdir(parents=True, exist_ok=True)
        path = out / ("ram_latches.png" if d == 22 else "crops.png")
        cv2.imwrite(str(path), np.hstack(tiles))
        files.append(str(path))
    return files


def render(rec: dict, out: Path) -> list[str]:
    a = rec["oak"]
    if a is None:
        return []
    extra = render_evidence(rec, out)
    return render_model(rec, out) + extra


def render_model(rec: dict, out: Path) -> list[str]:
    a = rec["oak"]
    j = a.judgement
    other = j.answer if j.answer is not j.declared else (j.runner_up or j.answer)
    phis = {"declared": j.declared.hyp.phi, short(other.hyp) or "best": other.hyp.phi}
    files = [A.render_decisive(a, str(out / "decisive.png"), phis)]
    pairs = sorted(A.disputed_pairs(a, other.hyp.phi, limit=6))
    crops = [(f"scan {a.folder_names[i]}", a.folder_paths[i], a.folder_names[i + 1],
              a.folder_paths[i + 1]) for i in pairs]
    if crops:
        files.append(A.render_changes(crops, "scan", a.roi["scan"][0],
                                      str(out / "scanner_changes.png")))
    lo, hi = min(pairs), max(pairs) + 1
    folders = a.folders[max(0, lo - 1):min(len(a.folders), hi + 2)]
    rows = range(max(1, min(a.declared[max(0, lo - 1)], other.hyp.phi[max(0, lo - 1)])),
                 min(a.n_rows, max(a.declared[min(len(a.declared) - 1, hi + 1)],
                                   other.hyp.phi[min(len(a.declared) - 1, hi + 1)])) + 1)
    files.append(A.render_strip(a, folders, rows, str(out / "strip.png"), height=200))
    return files


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def report(records: list[dict], flagged: list[dict], args) -> str:
    lines = [
        f"# Data audit {date.today().isoformat()}: every desktop, every source (U25)",
        "",
        "Generated by `python scripts/data_audit.py` over the index "
        f"`{Path(args.index).name}` (collapsed steps, U25 declarations), the Drive sheets, "
        "`raw_logs/oak_survey` and `configs/index_fixes.yaml`; the scanner check is "
        "`tda.core.scan_align` (method at the end). Read-only on the raw drive and the "
        "database.",
        "",
        "## 1. Per desktop",
        "",
        "| D | sheet rows / dupli / steps | cam1 / cam2 / RS / scan folders | gallery oak1/oak2/rs "
        "| scanner vs OAK cam1 | scanner vs RealSense | scanner pictures vs the sheet "
        "| known anomalies, fixes, index issues | sheet notes (rescan / missing / skip / redo) |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        lines.append(row(rec))
    lines += ["", "## 2. Desktops that need a human look", "",
              "Every desktop the scanner check flags, every one with a count symptom or an "
              "undeclared rule, and every one whose declaration was settled by looking "
              "(**by eye** -- read those first). Frames are under "
              f"`{Path(args.render)}\\D<nn>\\`: `decisive.png` (the two scanner folders the "
              "declared mapping and its strongest rival disagree on most, with the OAK rows each "
              "pairs them with), `scanner_changes.png` / `eyes_*.png` (where two frames differ "
              "most: a part gone, or only light and background = a re-shot), `strip.png`, "
              "`frames.png`.", ""]
    for rec in flagged:
        lines += look(rec)
    low = [rec for rec in records if poor(rec)]
    lines += ["### 2b. Counts match, but the scanner and OAK barely agree", "",
              f"A skip compensated by an extra shot leaves no count symptom, so these "
              f"desktops -- every scanner folder on a step, declared score below "
              f"{POOR_DECLARED:.2f} -- are listed even when no single alternative wins:", ""]
    lines += [f"- D{rec['d']:02d}: declared {rec['oak'].judgement.declared.score:.2f}, OAK ROI "
              f"{rec['oak'].roi.get('oak1', (None, '?'))[1]}"
              + (f" -- {EYES[rec['d']]}" if rec["d"] in EYES else "") for rec in low] or ["- none"]
    lines += [""] + picture_section(records, args)
    lines += ["", "## 3. Method", ""] + METHOD
    return "\n".join(lines) + "\n"


#: Desktops whose scanner frames are known to sit on the wrong step after some
#: point (no rule yet), or whose sheet and cameras disagree in count: their
#: picture flags past that point say more about the mapping than the log.
SHIFTED = {22: "scanner one step off from row 27/32 on (D22 undeclared)",
           41: "scanner one step off from rows 26-33 on (a skipped real step)",
           60: "scanner one step off from about folder 28 on (D60 undeclared)",
           61: "scanner one step off from rows 12-27 on (a skipped real step)",
           43: "sheet 47 rows, Camera_1 48 captures", 56: "sheet 45 rows, Camera_1 46 captures",
           63: "sheet 37 rows, Camera_1 34 captures",
           1: "sheet 39 rows, Camera_1 40 captures (the 40th is the parts layout)"}

#: What the flagged step pairs show when looked at (frames s-1 | s at full size
#: with the blob boxed, rendered to ``D:\DataSet\.cache\tmp\u25\pics\``). The
#: lighting ones are the measure's known blind spot, not the data's.
PICTURE_EYES = {
    (2, 3): "lighting: the drive-bay half of RGB31 is lit brighter (all ten shots of the "
            "burst); no part moved. Not a data problem",
    (2, 4): "lighting: the same glare gone again in RGB41. Not a data problem",
    (2, 33): "lighting: a blue cast on RGB331, plus the cables the disconnect moved. Not a "
             "data problem",
    (2, 47): "real: the black part standing above the chassis and the blue part at its top "
             "right in RGB461 are gone in RGB471, but row 47 only opens the power-module "
             "blocker -- check the sheet for a missing removal",
    (35, 7): "a real removal: the black bracket left of the CPU is gone in RGB71 (its screw "
             "post shows) -- black plastic on a dark board stays under 25 grey levels. Not a "
             "data problem",
    (37, 45): "a real displace, but the drive slides out of the top edge of the chassis, "
              "outside the ROI. Not a data problem",
    (43, 47): "the motherboard is still in RGB471: the scanner has no frame of the empty "
              "chassis; its last frame is a step behind the sheet",
    (45, 11): "real: the PSU is pushed sideways between RGB101 and RGB111, but row 11 only "
              "disconnects -- a `displace psu` is missing from the sheet (or belongs here)",
    (56, 45): "the motherboard is still in RGB451: as D43, the last scanner frame is a step "
              "behind the sheet",
    (59, 41): "lighting: a violet glare over the lower half of RGB411 (the whole burst); no "
              "part moved. Not a data problem",
    (59, 42): "lighting: the same glare gone again in RGB421. Not a data problem",
    (60, 46): "the motherboard is still in RGB461: the known D60 one-step shift (undeclared)",
    (63, 32): "real: the chassis is turned about 90 degrees between RGB311 and RGB321 (and "
              "again for 33 and 34) to reach the motherboard screws; the sheet has no "
              "reorient row",
    (63, 33): "real: turned again (see step 32)",
    (63, 34): "real: turned again (see step 32)",
}


def picture_section(records: list[dict], args) -> list[str]:
    flagged = [(rec, f) for rec in records if rec.get("pics") for f in rec["pics"]["flags"]]
    lines = [
        "### 2c. Steps whose scanner pictures contradict the sheet", "",
        "Per step, the change between the scanner frame of step s-1 and of step s (chassis ROI, "
        "256 px, brightness-matched, > 25 grey levels after a 5x5 blur, specks removed): the "
        "changed fraction and the largest changed blob (full-resolution scanner pixels). (a) a "
        "large part is removed or displaced and nothing changes -- the sheet row may name the "
        "wrong part or the capture repeats the one before; (b) only unscrew / release / "
        "disconnect / a latch, yet a large change -- something else happened (the chassis moved, "
        "a part came out unlogged, or the scanner frame sits on another step) -- or the light "
        "changed on part of the chassis; (c) the first and the last frame barely differ. "
        f"Numbers are cached under `{Path(args.diffs)}\\D<nn>.json`. Only the flagged steps are "
        "listed; every one was looked at (frames s-1 | s with the blob boxed under "
        "`D:\\DataSet\\.cache\\tmp\\u25\\pics\\`), and the **by eye** note says what it is:", "",
        "| D | step | sheet row | flag | changed | largest blob (x0, y0, x1, y1) | note |",
        "|---:|---:|---|---|---:|---|---|",
    ]
    for rec, f in flagged:
        m = f["m"]
        blob = f"{m['blob']} ({m['blob_frac']:.1%})" if m.get("blob") else "-"
        eyes = PICTURE_EYES.get((rec["d"], f["step"]))
        note = "; ".join(x for x in (SHIFTED.get(rec["d"], ""),
                                     f"**by eye:** {eyes}" if eyes else "") if x)
        name = str(f["name"]).replace("|", "/")
        lines.append(f"| {rec['d']} | {f['step']} | {name} | ({f['kind']}) {f['why']} | "
                     f"{m['frac']:.1%} | {blob} | {note} |")
    if not flagged:
        lines.append("| - | - | - | none | - | - | - |")
    dark = [(rec["d"], rec["pics"]["dark"]) for rec in records
            if rec.get("pics") and rec["pics"]["dark"]]
    lines += ["", "### 2d. Scanner shots replaced within the burst", "",
              f"The first shot (`P_0`) of these scanner folders is unexposed (`dark`), blown out "
              f"(`saturated`) or unlike the rest of its burst (`outlier`: a hand, a flash of "
              f"glare); the check above used the shot the local cache shows instead "
              f"(`tda.core.cache.choose_scan_image`; {sum(len(v) for _d, v in dark)} folders on "
              f"{len(dark)} desktops):", ""]
    lines += [f"- D{d:02d}: " + ", ".join(f"step {s} -> {shot}" for s, shot in
                                          sorted(v.items(), key=lambda kv: int(kv[0])))
              for d, v in dark] or ["- none"]
    return lines


def row(rec: dict) -> str:
    d, di, sheet, c = rec["d"], rec["di"], rec["sheet"], rec["counts"]
    if sheet is not None:
        sheet_cell = (f"{sheet.n_rows} / {list(sheet.dupli_rows) or '-'} / "
                      f"{sheet.n_rows - len(sheet.dupli_rows)}")
    else:
        sheet_cell = "no sheet"
    counts = " / ".join("-" if c.get(k) is None else str(c[k])
                        for k in ("Camera_1", "Camera_2", "rs", "scan"))
    if di is not None:
        gallery = " / ".join(str(sum(1 for k in di.frames if k.view == v))
                             for v in COMPONENT_VIEWS)
    else:
        gallery = "-"
    notes = []
    if rec["anom"]:
        notes.append("survey: " + "; ".join(rec["anom"]))
    if rec["fixes"]:
        notes.append("fixes: " + "; ".join(rec["fixes"]))
    if di is not None:
        shown = [i for i in di.issues if not i.startswith(("rs time mismatch", "cmp_"))]
        times = sum(1 for i in di.issues if i.startswith("rs time mismatch"))
        if times:
            shown.append(f"{times} RealSense frame(s) more than 30 s from their OAK capture")
        if shown:
            notes.append("index: " + "; ".join(shown))
        missing = sorted(k.step for k in di.missing if k.view == "scan")
        if missing:
            notes.append(f"scan missing steps {missing}")
    notes += [f"error: {e}" for e in rec["errors"]]
    sheet_notes = []
    if sheet is not None:
        for r in A.rescan_rows(sheet.notes):
            sheet_notes.append(f"{r}: {sheet.notes[r - 1]}")
        for r in sheet.dupli_rows:
            sheet_notes.append(f"{r} (dupli): {sheet.names[r - 1]}")
    esc = lambda s: s.replace("|", "/").replace("\n", " ")  # noqa: E731
    return (f"| {d} | {sheet_cell} | {counts} | {gallery} | {esc(verdict_cell(rec['oak']))} | "
            f"{esc(verdict_cell(rec['rs']))} | {pictures_cell(rec)} | "
            f"{esc('<br>'.join(notes)) or '-'} | {esc('; '.join(sheet_notes)) or '-'} |")


def pictures_cell(rec: dict) -> str:
    pics = rec.get("pics")
    if not pics:
        return "-"
    kinds = {}
    for f in pics["flags"]:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    parts = [f"({k}) {n}" for k, n in sorted(kinds.items())] or ["no contradiction"]
    if pics.get("first_last") is not None:
        parts.append(f"first->last {pics['first_last']:.0%}")
    if pics["dark"]:
        parts.append(f"P_0 replaced: {len(pics['dark'])}")
    return "; ".join(parts)


def look(rec: dict) -> list[str]:
    a = rec["oak"]
    out = [f"### D{rec['d']:02d}", ""]
    if a is None:
        return out + ["- no scanner analysis: " + "; ".join(rec["errors"]), ""]
    j = a.judgement
    out.append(f"- verdict: **{j.verdict}** -- {j.note}")
    if j.runner_up is not None:
        out.append(f"- answer '{j.answer.hyp.label}' {j.answer.score:.3f}; runner-up "
                   f"'{j.runner_up.hyp.label}' {j.runner_up.score:.3f} (margin {j.margin:+.3f}, "
                   f"bootstrap {j.confidence:.2f})")
    if a.dupli_rows and j.proposal is not None:
        out.append(f"- dupli rows {a.dupli_rows}: skipped {j.proposal.skipped_dupli or 'none'}, "
                   f"captured {j.proposal.captured_dupli or 'none'} -> "
                   f"duplicate_folders {j.proposal.duplicate_folders} "
                   f"({'decided' if j.decided else 'not decided'})")
    if rec.get("renders"):
        out.append("- frames: " + ", ".join(f"`{p}`" for p in rec["renders"]))
    if rec["d"] in EYES:
        out.append(f"- **by eye:** {EYES[rec['d']]}")
    else:
        out.append(f"- look for: {what_to_look_for(a)}")
    out.append("")
    return out


def what_to_look_for(a: A.Analysis) -> str:
    j = a.judgement
    other = j.answer if j.answer is not j.declared else (j.runner_up or j.answer)
    if other is j.declared:
        return "nothing competes with the declared mapping; the flag is the low score"
    i = A.decisive_pair(a, other.hyp.phi)
    n0, n1 = a.folder_names[i], a.folder_names[i + 1]
    dd = (a.declared[i], a.declared[i + 1])
    oo = (other.hyp.phi[i], other.hyp.phi[i + 1])
    return (f"what changes between scanner {n0} and {n1} -- the declared mapping says it is "
            f"OAK rows {dd[0]}->{dd[1]}, '{short(other.hyp)}' says rows {oo[0]}->{oo[1]}; "
            f"find the part that goes in the scanner pair and in which OAK pair it goes")


METHOD = [
    "- **Steps.** A sheet row is a step unless it is a `dupli` row (a second photograph of "
    "the state before it); the collapsed steps are numbered on without them (spec 2.2, "
    "v1.25). Folder counts are the step folders on the drive (Camera_1/Camera_2/RealSense "
    "`NNN`, scanner `RGB<n>1`).",
    "- **Change magnitudes.** Every frame is cut to the chassis ROI (the database's pose-segment "
    "ROI, else the box `tda.core.cache.suggest_roi_over` proposes, else the whole frame), turned "
    "grey, downscaled to at most 256 px and normalised by its median and MAD. The change "
    "between two frames is their mean absolute difference.",
    "- **Hypotheses.** The scanner folders are mapped onto the OAK rows as the index declares "
    "(`as declared`), and by one or two edits of that: the scanner skipped row k, folder n "
    "re-shoots the previous state, or both (a skip compensated by an extra shot -- no count "
    "symptom). The score of a mapping is the Spearman rank correlation of the scanner's "
    "folder-to-folder changes with the OAK changes between the rows it pairs. Mappings that "
    "show the same states (a `dupli` row shows the row before it) are one hypothesis.",
    "- **Verdict.** Thousands of mappings are tried, so one always edges the declared mapping "
    "out by noise: on the desktops whose declaration was checked by eye the best single edit "
    "still scored up to +0.09 and the best skip + re-shot pair up to +0.21. So every edit "
    "costs 0.11 of score, and a mapping replaces the declared one only when it beats it by "
    "more than its edits cost, by at least 0.05, and in at least 95 % of 200 paired "
    "bootstrap resamples of the folder pairs. `aligned` otherwise (**FLAG** when such a "
    "mapping exists but wins too few resamples); `skip` / `dup` / `skip+dup` = the kind of "
    "the mapping that won (`skip rows a-b`: the positions as plausible as the best, i.e. not "
    "beaten in 90 % of the resamples); `unknown` = even the best correlates below 0.30. The "
    "cell's `(vs ...)` is the best rival and its score relative to the declared mapping.",
    "- **Scanner pictures vs the sheet** (`scripts/step_changes.py`, section 2c). For every "
    "step of the collapsed sheet, the scanner frame of step s-1 and of step s (on the steps the "
    "U25 index gives them; each burst stands for the shot the local cache shows, so an "
    "unexposed, blown-out or outlier `P_0` is replaced -- section 2d) are cut to the chassis "
    "ROI, downscaled to 256 px, the later one matched to the earlier one's 10th and 90th "
    "percentiles, and a pixel counts as changed when the 5x5-blurred difference exceeds 25 "
    "grey levels (3x3 opening removes specks). No registration: the scanner is fixed. Over the "
    "2,732 steps the changed fraction has medians of 0.4 % (unscrew), 1.3 % (open), 2.2 % "
    "(disconnect), 3.8 % (remove), 20 % (displace) and 86 % (reorient), and 95th percentiles of "
    "5-10 % for the quiet verbs, which sets the thresholds: (a) remove / displace of a part "
    "visible from above with < 0.5 %; (b) only unscrew / release (> 15 %), disconnect "
    "(> 25 %) or a latch opened (> 20 %); (c) first vs last < 15 %. RAM modules, CPUs and "
    "expansion cards are not held to (a): edge-on or a few pixels square at 256 px, their "
    "visible removals measure under 0.5 %. The measure compares grey levels, so a lamp or "
    "glare that lights one part of the chassis and not the rest is a change too: the "
    "**by eye** notes in 2c say which flags are that.",
    "- **RealSense column.** The same check against the RealSense frames where every row has "
    "one -- a second opinion only: its view of the chassis is smaller and the correlations are "
    "lower (a third of the desktops below 0.3), so a RealSense flag alone is not acted on.",
    "- **What the numbers cannot do.** A run of small changes (screws, connectors) gives near-"
    "equal scores to every skip position inside it, so a skip is localised to an interval, not "
    "a row; on D48 and D62 the best position was wrong while the sheet's own words (\"Dupli "
    "(scanner does not scan)\") and the frames agreed. Every declaration in "
    "`configs/index_fixes.yaml` was therefore settled by looking: the *scanner changes* crops "
    "show the two places consecutive scanner folders differ most -- a part gone, or only the "
    "light and the background (a re-shot).",
]


if __name__ == "__main__":
    raise SystemExit(main())
