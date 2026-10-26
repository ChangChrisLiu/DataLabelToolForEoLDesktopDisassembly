"""U30-QA, pass 1: every (scan frame, instance) of the verified desktops, as numbers.

Read-only on everything but its own output. It opens a *copy* of the database
(``--db``, made with the SQLite backup API and already run through ``cli check``)
and compiles every annotatable scanner frame again with
:class:`tda.core.truth.TruthService` -- only to get what the truth table does
not store: the amodal shape, the painted layer order and who claims the pixels a
part loses. The visible mask, its label and its occlusion ratio are the
**stored** rows (``compiled_mask``), i.e. what an export would ship.

Per (frame, instance) it writes one row: class, state, placement, keyframe in
force (id, source, anchor), amodal area / bbox, stored and recompiled visible
area, stored label and occlusion ratio, the frame override if any, the
occluders -- every other layer whose recompiled visible pixels fall inside this
part's amodal-minus-visible, with the pixel count -- the visible pixels outside
the scanner ROI, and the open conflict if any. The RLEs of the amodal and the
stored visible mask go next to the rows, for the contact sheets.

``python scripts/qa/u30_extract.py --db <copy> --out <dir> [--desktops 2-17,19-32]``
writes ``<out>/D<nn>.pkl``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tda import pipeline as P  # noqa: E402
from tda.core import masks as M  # noqa: E402
from tda.core.db import Db  # noqa: E402
from tda.core.model import FrameKey  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.core.truth_inputs import InputCache, annotatable_steps, state_of  # noqa: E402

VIEW = "scan"
DEFAULT_DESKTOPS = "2-17,19-32"


def _win(ci):
    w = ci.amodal_window
    if w is None:
        return None
    return int(w[0]), int(w[1]), int(w[2]), int(w[3])


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _outside(mask: np.ndarray, roi) -> int:
    if roi is None or mask is None:
        return 0
    x0, y0, x1, y1 = roi
    total = int(mask.sum())
    inside = int(mask[y0:y1, x0:x1].sum())
    return total - inside


def extract(db: Db, tax, truth: TruthService, desktop: int) -> dict:
    instances = db.instances(desktop)
    steps_meta = {s.step: {"type": s.step_type, "raw": s.raw_name, "dupli": s.dupli}
                  for s in db.steps(desktop)}
    actions = {}
    for a in db.actions(desktop):
        actions.setdefault(a.step, []).append({"target": a.target, "verb": a.verb,
                                               "result": a.result})
    frames = db.frames_for(desktop, VIEW)
    frame_rows = {int(r["step"]): r for r in frames}
    steps = annotatable_steps(db, desktop, VIEW, [int(r["step"]) for r in frames])
    seg_row = db.conn.execute(
        "SELECT roi_json FROM pose_segment WHERE desktop=? AND view=? ORDER BY seg",
        (desktop, VIEW)).fetchone()
    roi = json.loads(seg_row[0]) if seg_row and seg_row[0] else None
    open_conf = {}
    for r in db.conn.execute(
            "SELECT id, step, instance, sym_diff_px, created_at FROM conflict "
            "WHERE desktop=? AND view=? AND status='open'", (desktop, VIEW)):
        open_conf[(int(r[1]), r[2])] = {"id": int(r[0]), "sym_diff_px": r[3], "at": r[4]}
    kf_info = {}
    for r in db.conn.execute(
            "SELECT id, instance, anchor_step, source, placement, geom_type, edit_count "
            "FROM shape_keyframe WHERE desktop=? AND view=?", (desktop, VIEW)):
        kf_info[int(r[0])] = {"instance": r[1], "anchor": int(r[2]), "source": r[3],
                              "placement": r[4], "geom": r[5], "edits": int(r[6] or 0)}
    cache = InputCache()
    rows, rles, frames_out = [], {}, {}
    for step in steps:
        key = FrameKey(desktop, step, VIEW)
        inputs, cf = truth._compile(key, cache)
        state = state_of(db, tax, desktop, step, cache, view=VIEW)
        stored = {}
        for r in db.conn.execute(
                "SELECT instance, visible_rle_json, occlusion_ratio, visibility, placement, "
                "status, geom_type, box_json FROM compiled_mask "
                "WHERE desktop=? AND step=? AND view=?", (desktop, step, VIEW)):
            stored[r[0]] = {"rle": json.loads(r[1]) if r[1] else None, "occ": r[2],
                            "visibility": r[3], "placement": r[4], "status": r[5],
                            "geom": r[6], "box": json.loads(r[7]) if r[7] else None}
        fo = inputs.frame_overrides
        occl = inputs.occluders
        occl_dec = [(o.occluder_type, M.decode_rle(o.rle)) for o in occl if o.rle]
        frames_out[step] = {
            "painted": cf.painted, "pairs": cf.pairs, "problems": list(cf.problems),
            "review": (frame_rows.get(step) or {}).get("review_status"),
            "cache_path": ((frame_rows.get(step) or {}).get("aux") or {}).get("cache_path"),
            "stored_instances": sorted(stored), "compiled_instances": sorted(cf.instances),
        }
        # the painted position (bottom = 0) per placement group
        zpos = {}
        for group, order in cf.painted.items():
            for i, inst in enumerate(order):
                zpos[inst] = (group, i)
        dec_vis_stored = {}
        for inst, srow in stored.items():
            if srow["rle"]:
                dec_vis_stored[inst] = M.decode_rle(srow["rle"])
        names = sorted(set(cf.instances) | set(stored))
        for inst in names:
            ci = cf.instances.get(inst)
            srow = stored.get(inst)
            rec = instances.get(inst)
            st = state.get(inst)
            row = {
                "desktop": desktop, "step": step, "inst": inst,
                "cls": rec.cls if rec else None,
                "state": st.state if st else None,
                "placement_state": st.placement if st else None,
                "left_with": getattr(st, "left_with", None) if st else None,
                "in_compiled": ci is not None, "in_stored": srow is not None,
                "need": inputs.needs.get(inst),
                "kf_id": ci.keyframe_id if ci else None,
                "z": zpos.get(inst),
                "conflict": open_conf.get((step, inst)),
                "fo": None,
            }
            if ci is not None and row["kf_id"] in kf_info:
                k = kf_info[row["kf_id"]]
                row.update(kf_source=k["source"], kf_anchor=k["anchor"], kf_edits=k["edits"])
            if inst in fo:
                f = fo[inst]
                row["fo"] = {"visibility": f.visibility, "occluder": f.occluder,
                             "has_rle": f.visible_rle is not None}
            if srow is not None:
                row.update(visibility=srow["visibility"], occ_stored=srow["occ"],
                           status=srow["status"], placement=srow["placement"],
                           geom=srow["geom"])
                vs = dec_vis_stored.get(inst)
                row["vis_area"] = int(vs.sum()) if vs is not None else 0
                row["vis_bbox"] = _bbox(vs) if vs is not None and row["vis_area"] else None
                row["vis_outside_roi"] = _outside(vs, roi) if vs is not None else 0
            if ci is not None and ci.amodal is not None:
                am = ci.amodal
                win = _win(ci) or (0, 0, am.shape[1], am.shape[0])
                x0, y0, x1, y1 = win
                a_sub = am[y0:y1, x0:x1]
                row["amodal_area"] = int(a_sub.sum())
                row["amodal_bbox"] = _bbox(a_sub)
                if row["amodal_bbox"]:
                    bx = row["amodal_bbox"]
                    row["amodal_bbox"] = (bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0)
                row["amodal_outside_roi"] = _outside(am, roi)
                row["vis_area_fresh"] = 0 if ci.visible is None else int(
                    ci.visible[y0:y1, x0:x1].sum())
                row["visibility_fresh"] = ci.visibility
                row["occ_fresh"] = float(ci.occlusion_ratio)
                vs = dec_vis_stored.get(inst)
                v_sub = (vs[y0:y1, x0:x1] if vs is not None
                         else np.zeros_like(a_sub))
                hidden = a_sub & ~v_sub
                row["hidden_px"] = int(hidden.sum())
                # visible-stored pixels outside the amodal (an override, a stale row)
                row["vis_outside_amodal"] = int((v_sub & ~a_sub).sum()) if vs is not None else 0
                occ = {}
                if row["hidden_px"]:
                    for other, cj in cf.instances.items():
                        if other == inst or cj.visible is None:
                            continue
                        wj = cj.window or _win(cj)
                        if wj is not None and _inter(win, tuple(int(v) for v in wj)) is None:
                            continue
                        n = int((cj.visible[y0:y1, x0:x1] & hidden).sum())
                        if n:
                            occ[other] = n
                    for typ, om in occl_dec:
                        n = int((om[y0:y1, x0:x1] & hidden).sum())
                        if n:
                            occ[f"<{typ}>"] = n
                row["occluders"] = occ
                rles[(step, inst)] = {
                    "amodal": M.encode_rle(am, window=win),
                    "visible": srow["rle"] if srow is not None else None,
                }
            rows.append(row)
    return {"desktop": desktop, "roi": roi, "steps": steps, "steps_meta": steps_meta,
            "actions": actions, "frames": frames_out, "rows": rows, "rles": rles,
            "instances": {k: {"cls": r.cls, "parent": r.parent, "attached": r.attached,
                              "mounted_on": r.mounted_on, "fastens": r.fastens,
                              "socket_host": r.socket_host, "cable": r.cable,
                              "attrs": r.attrs}
                          for k, r in instances.items()},
            "brand": (db.get_desktop(desktop) or {}).get("brand"),
            "model": (db.get_desktop(desktop) or {}).get("model_family")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="a COPY of the database")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="D:/DataSet/cache", help="frame cache (read only)")
    ap.add_argument("--desktops", default=DEFAULT_DESKTOPS)
    args = ap.parse_args(argv)
    if os.path.normcase(os.path.abspath(args.db)) == os.path.normcase(
            os.path.abspath("D:/DataSet/annotations/tda.sqlite")):
        print("refused: give a copy of the database, not the live file")
        return 1
    os.makedirs(args.out, exist_ok=True)
    db = Db(args.db)
    tax = load_taxonomy()
    truth = TruthService(db, tax, cache_dir=args.cache)
    for desktop in sorted(P.parse_desktops(args.desktops)):
        t0 = time.time()
        res = extract(db, tax, truth, desktop)
        with open(os.path.join(args.out, f"D{desktop:02d}.pkl"), "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"D{desktop:02d}: {len(res['steps'])} frames, {len(res['rows'])} rows, "
              f"{time.time() - t0:.1f}s", flush=True)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
