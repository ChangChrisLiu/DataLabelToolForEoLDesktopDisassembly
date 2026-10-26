"""U30-QA, pass 3: contact sheets for the controller's eye.

Every desktop, every instance: the frame where the annotator first met it (the
last frame it is in the chassis) and the frames where its stored occlusion ratio
is highest (+ the first frame of its two strongest automatic flags). One tile =
the crop around the part's amodal box on the scanner frame the annotator saw
(the cache's chosen burst image); **solid** outline = the stored visible mask,
**thin dashed** outline = the amodal shape, thin red outlines = the occluders'
visible masks inside the crop; the caption names the occluders with the pixels
each one takes. A red caption bar = an automatic flag on that frame.

``--top`` draws the ranked suspicious items instead (two tiles each: the frame
and the frame it was compared with), ``--conflicts`` the open conflicts (frozen
row vs recompiled), into ``sheets/top_<n>.jpg`` / ``sheets/conflicts_<n>.jpg``.
Read-only; writes only under ``<root>/sheets``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import u30_common as U  # noqa: E402

TILE = 320
CAP = 84
COLS, ROWS = 4, 3
VIS_COLOR = (60, 255, 60)       # BGR green
AMODAL_COLOR = (0, 230, 255)    # yellow
OCC_COLOR = (40, 40, 255)       # red
OLD_COLOR = (255, 120, 0)       # blue (conflict: frozen row)


def _crop_box(bbox, shape, pad_frac=0.6, min_pad=28, min_side=90):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    side = max(w, h, min_side)
    pad = max(min_pad, int(pad_frac * side))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2 + pad
    X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
    X1, Y1 = int(min(shape[1], cx + half)), int(min(shape[0], cy + half))
    return X0, Y0, X1, Y1


def _contours(mask_crop):
    m = mask_crop.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cs


def _dashed(img, contours, color, scale, dash=4, thick=1):
    for c in contours:
        pts = (c[:, 0, :].astype(np.float32) * scale).astype(np.int32)
        for i in range(0, len(pts) - 1, dash * 2):
            seg = pts[i:i + dash + 1]
            if len(seg) >= 2:
                cv2.polylines(img, [seg.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)


def _solid(img, contours, color, scale, thick=2):
    scaled = [(c.astype(np.float32) * scale).astype(np.int32) for c in contours]
    cv2.drawContours(img, scaled, -1, color, thick, cv2.LINE_AA)


def _text(img, lines, y0, colors=None, scale=0.38):
    y = y0
    for i, line in enumerate(lines):
        col = (colors[i] if colors and i < len(colors) else (235, 235, 235))
        cv2.putText(img, line[:62], (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += 15


def tile(res, step, inst, lines, flagged=False, box=None, occluders=True,
         extra=None, header=None, own=True):
    """One TILE x (TILE + CAP) BGR tile; ``extra`` = [(mask, color, dashed)];
    ``own=False`` leaves the part's own stored masks off (``extra`` draws them)."""
    img = U.color(res, step)
    out = np.full((TILE + CAP, TILE, 3), 30, np.uint8)
    rles = (res["rles"].get((step, inst)) or {}) if own else {}
    amodal = U.decode(rles.get("amodal"))
    visible = U.decode(rles.get("visible"))
    if img is None:
        _text(out, ["(no image)"] + lines, 20)
        return out
    if box is None:
        ref = amodal if amodal is not None and amodal.any() else visible
        bb = U.bbox_of(ref) if ref is not None else None
        box = _crop_box(bb, img.shape) if bb else (0, 0, img.shape[1], img.shape[0])
    X0, Y0, X1, Y1 = box
    crop = img[Y0:Y1, X0:X1].copy()
    h, w = crop.shape[:2]
    scale = TILE / max(h, w)
    view = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if occluders:
        for (s, other), r2 in res["rles"].items():
            if s != step or other == inst or not r2.get("visible"):
                continue
            row = next((r for r in res["rows"] if r["step"] == step and r["inst"] == other), None)
            bb = (row or {}).get("vis_bbox")
            if bb and (bb[2] <= X0 or bb[0] >= X1 or bb[3] <= Y0 or bb[1] >= Y1):
                continue
            m = U.decode(r2["visible"])
            if m is None:
                continue
            _solid(view, _contours(m[Y0:Y1, X0:X1]), OCC_COLOR, scale, 1)
    for m, colr, dashed in (extra or []):
        if m is None:
            continue
        if dashed:
            _dashed(view, _contours(m[Y0:Y1, X0:X1]), colr, scale, thick=2)
        else:
            _solid(view, _contours(m[Y0:Y1, X0:X1]), colr, scale, 2)
    if amodal is not None:
        _dashed(view, _contours(amodal[Y0:Y1, X0:X1]), AMODAL_COLOR, scale)
    if visible is not None:
        _solid(view, _contours(visible[Y0:Y1, X0:X1]), VIS_COLOR, scale, 2)
    out[:view.shape[0], :view.shape[1]] = view
    if header:
        cv2.rectangle(out, (0, 0), (TILE, 16), (0, 0, 0), -1)
        cv2.putText(out, header[:60], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    1, cv2.LINE_AA)
    if flagged:
        cv2.rectangle(out, (0, TILE), (TILE, TILE + 3), (0, 0, 220), -1)
    _text(out, lines, TILE + 16)
    return out


def caption(res, step, inst, why="", flags=()):
    r = next((r for r in res["rows"] if r["step"] == step and r["inst"] == inst), {}) or {}
    occ = sorted((r.get("occluders") or {}).items(), key=lambda kv: -kv[1])
    occ_txt = ", ".join(f"{k}({v})" for k, v in occ[:3]) + (" ..." if len(occ) > 3 else "")
    fo = r.get("fo")
    lines = [f"D{res['desktop']:02d} s{step} {inst}",
             f"{r.get('visibility')} occ={r.get('occ_stored') or 0:.2f} vis={r.get('vis_area', 0)}"
             f"/{r.get('amodal_area', 0)}px st={r.get('state')}"
             + (f" FO:{fo.get('visibility')}/{fo.get('occluder')}" if fo else ""),
             f"by: {occ_txt or '-'}",
             f"{why}" + (f" | FLAG {','.join(flags)}" if flags else "")]
    return lines


def sheet(tiles, title, path):
    cols = COLS
    rows = max(1, (len(tiles) + cols - 1) // cols)
    H = 24 + rows * (TILE + CAP)
    W = cols * TILE + (cols - 1) * 4
    out = np.full((H, W, 3), 15, np.uint8)
    cv2.putText(out, title[:150], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = 24 + r * (TILE + CAP), c * (TILE + 4)
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def render_desktop(d: int, out_dir: str) -> list[str]:
    res = U.load(d)
    ana = json.load(open(os.path.join(U.QA_ROOT, "analysis", f"D{d:02d}.json"), encoding="utf-8"))
    items = ana["render"]
    tiles = []
    for it in items:
        lines = caption(res, it["step"], it["inst"], it["why"], it["flags"])
        tiles.append(tile(res, it["step"], it["inst"], lines, flagged=bool(it["flags"])))
    paths = []
    per = COLS * ROWS
    n = (len(tiles) + per - 1) // per
    for k in range(n):
        p = os.path.join(out_dir, f"D{d:02d}_{k + 1:02d}.jpg")
        chunk = items[k * per:(k + 1) * per]
        names = sorted({c["inst"] for c in chunk})
        sheet(tiles[k * per:(k + 1) * per],
              f"D{d:02d} {res['brand']} {res['model']} sheet {k + 1}/{n}: green = stored visible, "
              f"yellow dashed = amodal, red = occluders' visible", p)
        paths.append(p)
    return paths


def _payload_mask(payload):
    if not payload:
        return None
    if isinstance(payload, dict) and "counts" in payload:
        return U.decode(payload)
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, dict) and "counts" in v:
                return U.decode(v)
    return None


SAM_COLOR = (255, 0, 255)       # magenta


def _box_of(*masks, shape=(1600, 1600), pad_frac=0.6):
    acc = None
    for m in masks:
        if m is None or not m.any():
            continue
        acc = m.copy() if acc is None else (acc | m)
    bb = U.bbox_of(acc) if acc is not None else None
    return _crop_box(bb, shape, pad_frac=pad_frac) if bb else None


def _masks(res, step, inst):
    rl = res["rles"].get((step, inst)) or {}
    return U.decode(rl.get("amodal")), U.decode(rl.get("visible"))


def _last_present(res, inst, before):
    steps = sorted(s for (s, i) in res["rles"] if i == inst and s <= before)
    return steps[-1] if steps else None


def top_tiles(it: dict, analysis: dict) -> list:
    """Two tiles that show why one ranked item is suspicious."""
    d, inst, code, steps = it["desktop"], it["inst"], it["code"], it["steps"]
    res = U.load(d)
    head = f"#{it['rank']} {code} ({it['severity']})"
    detail = it["detail"]
    wrap = [detail[i:i + 60] for i in range(0, min(len(detail), 180), 60)]
    if code == "open_conflict":
        return []
    if code in ("frame_unchanged", "change_unexplained"):
        s0, s1 = steps[0], steps[-1]
        info = next((s for s in analysis["steps"] if s["from"] == s0 and s["to"] == s1), {})
        ms = [_masks(res, s0, i)[0] for i in info.get("leaving", [])]
        if code == "change_unexplained" and it.get("blob"):
            x0, y0, x1, y1 = it["blob"]
            box = _crop_box((x0, y0, x1, y1), (1600, 1600), pad_frac=0.3)
        else:
            box = _box_of(*ms, pad_frac=0.5) or tuple(res["roi"] or (0, 0, 1600, 1600))
        extra = [(m, AMODAL_COLOR, True) for m in ms]
        return [tile(res, s0, "__none__", [f"D{d:02d} step {s0} (before)"] + wrap, True, box,
                     False, extra, head),
                tile(res, s1, "__none__", [f"D{d:02d} step {s1} (after)",
                                           "dashed = the parts said to leave"], True, box,
                     False, extra, head)]
    if code in ("leave_no_change", "zorder_pixels"):
        s0, s1 = steps[0], steps[-1]
        am, vis = _masks(res, s0, inst)
        box = _box_of(am, pad_frac=0.8)
        return [tile(res, s0, inst, caption(res, s0, inst, "before")[:3] + wrap[:1], True, box,
                     True, None, head),
                tile(res, s1, "__none__", [f"D{d:02d} step {s1}: {inst} is gone per the log",
                                           "dashed = its amodal at the step before"] + wrap[1:],
                     True, box, False, [(am, AMODAL_COLOR, True)], head)]
    if code in ("jump", "amodal_jump"):
        s1, s2 = steps[0], steps[-1]
        a1, _ = _masks(res, s1, inst)
        a2, _ = _masks(res, s2, inst)
        box = _box_of(a1, a2, pad_frac=0.5)
        return [tile(res, s, inst, caption(res, s, inst)[:3] + wrap[:1], True, box, True, None,
                     head) for s in (s1, s2)]
    if code == "a_empty":
        k, ref = steps[0], it.get("ref")
        am, _ = _masks(res, k, inst)
        box = _box_of(am, pad_frac=1.0)
        out = [tile(res, k, inst, caption(res, k, inst)[:3] + wrap[:1], True, box, True, None,
                    head)]
        if ref:
            out.append(tile(res, ref, inst, caption(res, ref, inst, f"reference {ref}")[:3]
                            + wrap[1:2], True, box, True, None, head))
        else:
            out.append(tile(res, steps[-1], inst, caption(res, steps[-1], inst)[:3] + wrap[1:2],
                            True, box, True, None, head))
        return out
    if code in ("shape_predates_state", "cut_copy_in_force"):
        s = steps[-1]
        nxt = s + 1
        am, _ = _masks(res, s, inst)
        box = _box_of(am, _masks(res, nxt, inst)[0], pad_frac=0.8)
        return [tile(res, s, inst, caption(res, s, inst, f"cli check: {code}")[:4], True, box,
                     True, None, head),
                tile(res, nxt, inst, caption(res, nxt, inst, "the next frame")[:4], True, box,
                     True, None, head)]
    # b_floor, c_below, d_roi, e_dup, vis_outside_amodal and the rest
    s, s2 = steps[0], steps[-1]
    other = it.get("other")
    am, _ = _masks(res, s, inst)
    if other:
        ao, _ = _masks(res, s, other)
        box = _box_of(am, ao, pad_frac=0.4)
        return [tile(res, s, inst, caption(res, s, inst)[:3] + wrap[:1], True, box, True, None,
                     head),
                tile(res, s, other, caption(res, s, other)[:3] + wrap[1:2], True, box, True,
                     None, head)]
    box = _box_of(am, pad_frac=0.8)
    return [tile(res, s, inst, caption(res, s, inst)[:3] + wrap[:1], True, box, True, None, head),
            tile(res, s2, inst, caption(res, s2, inst)[:3] + wrap[1:2], True, box, True, None,
                 head)]


def render_top(out_dir: str, n: int = 30) -> list[str]:
    ranked = json.load(open(os.path.join(U.QA_ROOT, "ranked.json"), encoding="utf-8"))
    tiles = []
    for rank, it in enumerate(ranked["top"][:n], 1):
        it = dict(it, rank=rank)
        analysis = json.load(open(os.path.join(U.QA_ROOT, "analysis",
                                               f"D{it['desktop']:02d}.json"), encoding="utf-8"))
        got = top_tiles(it, analysis)
        if not got:
            continue
        tiles.extend(got)
    paths = []
    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        p = os.path.join(out_dir, f"top_{k // per + 1:02d}.jpg")
        sheet(tiles[k:k + per], "ranked suspicious items, two tiles each (see the report's list)", p)
        paths.append(p)
    return paths


def render_low(out_dir: str) -> list[str]:
    ranked = json.load(open(os.path.join(U.QA_ROOT, "ranked.json"), encoding="utf-8"))
    low = ranked["sam"]["low200"]
    sam_rows = {}
    tiles = []
    for n, r in enumerate(low, 1):
        d = r["desktop"]
        if d not in sam_rows:
            rows = json.load(open(os.path.join(U.QA_ROOT, "sam", f"D{d:02d}.json"),
                                  encoding="utf-8"))
            sam_rows[d] = {(x["step"], x["inst"]): x for x in rows}
        res = U.load(d)
        x = sam_rows[d][(r["step"], r["inst"])]
        sm = U.decode(x["tuned"]["rle"])
        am, vis = _masks(res, r["step"], r["inst"])
        box = _box_of(am, sm, pad_frac=0.4)
        lines = [f"#{n} D{d:02d} s{r['step']} {r['inst']}",
                 f"IoU(SAM,vis) tuned {r['tuned_vis']:.2f} stock {r['stock_vis']:.2f} "
                 f"amodal {r['tuned_amodal']:.2f}",
                 f"{r['visibility']} occ={r['occ'] or 0:.2f} vis={r['vis_area']} "
                 f"sam={r['sam_area']} kf={r['kf_source']}",
                 "magenta = SAM (shipped head), green = ours"]
        tiles.append(tile(res, r["step"], r["inst"], lines, flagged=r["best_vis"] < 0.3, box=box,
                          occluders=False, extra=[(sm, SAM_COLOR, False)]))
    paths = []
    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        p = os.path.join(out_dir, f"low_{k // per + 1:02d}.jpg")
        sheet(tiles[k:k + per], "lowest SAM agreement (triage, not truth): green = stored "
                                "visible, yellow dashed = amodal, magenta = SAM box prompt", p)
        paths.append(p)
    return paths


def render_conflicts(out_dir: str) -> list[str]:
    """Every open conflict of the copy: the frozen row (blue) vs the recompile (green
    dashed), on the frame and on the neighbouring frame for context."""
    import sqlite3

    db = sqlite3.connect(f"file:{os.path.join(U.QA_ROOT, 'live_copy.sqlite')}?mode=ro", uri=True)
    rows = db.execute("SELECT id, desktop, step, instance, old_rle_json, new_rle_json, "
                      "sym_diff_px FROM conflict WHERE status='open' AND view='scan' "
                      "ORDER BY desktop, instance, step").fetchall()
    db.close()
    # one desktop/instance run: its first, middle and last frame
    groups = {}
    for r in rows:
        groups.setdefault((r[1], r[3]), []).append(r)
    tiles = []
    for (d, inst), rs in groups.items():
        res = U.load(d)
        pick = [rs[0]] if len(rs) == 1 else [rs[0], rs[len(rs) // 2], rs[-1]]
        for cid, _d, step, _i, old, new, sym in pick:
            mo, mn = _payload_mask(json.loads(old) if old else None), \
                _payload_mask(json.loads(new) if new else None)
            ref = mo if mo is not None and mo.any() else mn
            img = U.color(res, step)
            bb = U.bbox_of(ref | (mn if mn is not None else ref)) if ref is not None else None
            box = _crop_box(bb, img.shape, pad_frac=2.5) if bb else None
            lines = [f"D{d:02d} s{step} {inst} conflict #{cid}",
                     f"blue = frozen ({0 if mo is None else int(mo.sum())} px)",
                     f"green dashed = recompiled ({0 if mn is None else int(mn.sum())} px)",
                     f"sym. diff {sym} px"]
            tiles.append(tile(res, step, inst, lines, flagged=True, box=box, occluders=True,
                              extra=[(mo, OLD_COLOR, False), (mn, VIS_COLOR, True)],
                              header="conflict: frozen vs recompiled", own=False))
            # the same box, nothing drawn, so the eye sees the picture itself
            tiles.append(tile(res, step, "__none__", [f"D{d:02d} s{step} picture only"],
                              box=box, occluders=False))
    paths = []
    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        p = os.path.join(out_dir, f"conflicts_{k // per + 1:02d}.jpg")
        sheet(tiles[k:k + per], "open conflicts: blue = frozen row, green dashed = recompile", p)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desktops", default=None)
    ap.add_argument("--conflicts", action="store_true")
    ap.add_argument("--top", action="store_true")
    ap.add_argument("--low", action="store_true", help="the 200 lowest SAM agreements")
    args = ap.parse_args(argv)
    out_dir = os.path.join(U.QA_ROOT, "sheets")
    os.makedirs(out_dir, exist_ok=True)
    if args.conflicts:
        print(render_conflicts(out_dir))
        return 0
    if args.top:
        print(render_top(out_dir))
        return 0
    if args.low:
        print(render_low(out_dir))
        return 0
    wanted = U.DESKTOPS if not args.desktops else [int(x) for x in args.desktops.split(",")]
    index = {}
    for d in wanted:
        paths = render_desktop(d, out_dir)
        index[d] = [os.path.basename(p) for p in paths]
        print(f"D{d:02d}: {len(paths)} sheets", flush=True)
    with open(os.path.join(out_dir, "index_desktops.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
