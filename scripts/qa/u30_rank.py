"""U30-QA, pass 5: one ranked list for the controller, and the per-desktop tables.

Reads ``<root>/analysis/D<nn>.json`` (``u30_analyse.py``), ``<root>/check/D<nn>.txt``
(``cli check`` on the copy), ``<root>/sam/D<nn>.json`` (``u30_sam.py``, when
there) and the open conflicts of ``<root>/live_copy.sqlite`` (read only); writes
``<root>/ranked.json``: ``{"top": [...], "desktops": {...}, "check": {...},
"conflicts": [...], "sam": {...}}``.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import u30_common as U  # noqa: E402

#: which flags count against a desktop for the data package, and how much
PACKAGE_WEIGHT = {"high": 1.0, "medium": 0.3, "low": 0.0, "eye": 0.0}


def parse_check(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    codes = Counter()
    items = defaultdict(list)
    for m in re.finditer(r"^  - ([a-z_]+):(\S+) \(step (\d+)\)", text, re.M):
        codes[m.group(1)] += 1
        items[m.group(1)].append((m.group(2), int(m.group(3))))
    for m in re.finditer(r"^  - (zorder_floor|missing_shape|empty_visible|[a-z_]+):(.*)$",
                         text, re.M):
        if "(step" not in m.group(2):
            codes[m.group(1)] += 1
            items[m.group(1)].append((m.group(2).strip(), None))
    hidden = re.search(r"(\d+) 个零件完全被挡住（正常）：(\d+) 个零件帧", text)
    problems = re.search(r"^problems: (\d+)", text, re.M)
    conflicts = re.search(r"^open conflicts: (\d+)", text, re.M)
    return {"codes": dict(codes), "items": {k: v for k, v in items.items()},
            "covered_parts": int(hidden.group(1)) if hidden else 0,
            "covered_part_frames": int(hidden.group(2)) if hidden else 0,
            "problems": int(problems.group(1)) if problems else None,
            "open_conflicts": int(conflicts.group(1)) if conflicts else None}


def open_conflicts() -> list[dict]:
    db = sqlite3.connect(f"file:{os.path.join(U.QA_ROOT, 'live_copy.sqlite')}?mode=ro", uri=True)
    out = []
    for r in db.execute("SELECT id, desktop, step, view, instance, sym_diff_px, created_at "
                        "FROM conflict WHERE status='open' ORDER BY desktop, step"):
        out.append({"id": r[0], "desktop": r[1], "step": r[2], "view": r[3], "inst": r[4],
                    "sym_diff_px": r[5], "created_at": r[6]})
    db.close()
    return out


def runs_of(pairs):
    """``[(inst, step)]`` -> ``{inst: [[a, b], ...]}`` runs of consecutive steps."""
    by = defaultdict(list)
    for inst, step in pairs:
        by[inst].append(step)
    out = {}
    for inst, steps in by.items():
        steps = sorted(set(steps))
        rr = [[steps[0], steps[0]]]
        for s in steps[1:]:
            if s == rr[-1][1] + 1:
                rr[-1][1] = s
            else:
                rr.append([s, s])
        out[inst] = rr
    return out


def sam_summary() -> dict:
    rows = []
    for d in U.DESKTOPS:
        p = os.path.join(U.QA_ROOT, "sam", f"D{d:02d}.json")
        if os.path.isfile(p):
            rows.extend(json.load(open(p, encoding="utf-8")))
    if not rows:
        return {}
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)

    def stats(rs, head, key):
        v = np.array([r[head][key] for r in rs], float)
        return {"n": int(v.size), "median": round(float(np.median(v)), 3),
                "p10": round(float(np.percentile(v, 10)), 3),
                "lt_0.5": int((v < 0.5).sum())}

    table = {cls: {"tuned_vis": stats(rs, "tuned", "iou_vis"),
                   "tuned_amodal": stats(rs, "tuned", "iou_amodal"),
                   "stock_vis": stats(rs, "stock", "iou_vis")}
             for cls, rs in sorted(by_cls.items())}
    by_desk = defaultdict(list)
    for r in rows:
        by_desk[r["desktop"]].append(r)
    desk = {d: {"tuned_vis": stats(rs, "tuned", "iou_vis"),
                "stock_vis": stats(rs, "stock", "iou_vis")} for d, rs in sorted(by_desk.items())}
    # the lowest agreement: the better of the two heads still disagrees
    for r in rows:
        r["best_vis"] = max(r["tuned"]["iou_vis"], r["stock"]["iou_vis"],
                            r["tuned"]["oracle_vis"])
    # at most two frames per (desktop, part): a shape wrong on 22 frames is one finding
    low, per = [], Counter()
    for r in sorted(rows, key=lambda r: (r["best_vis"], -(r["vis_area"] or 0))):
        if per[(r["desktop"], r["inst"])] >= 2:
            continue
        per[(r["desktop"], r["inst"])] += 1
        low.append(r)
        if len(low) >= 200:
            break
    n_low = {t: int(sum(1 for r in rows if r["best_vis"] < t)) for t in (0.1, 0.3, 0.5)}
    for r in low:
        am = max(r["tuned"]["iou_amodal"], r["stock"]["iou_amodal"])
        if am >= 0.5:
            r["kind"] = "SAM ~ amodal: the part looks unoccluded where we hide it"
        elif (r["vis_area"] or 0) < 0.25 * (r["amodal_area"] or 1):
            r["kind"] = "tiny visible in a big box: shape in pieces / mostly hidden"
        else:
            r["kind"] = "SAM agrees with neither"
    return {"n": len(rows), "n_best_below": n_low, "by_class": table, "by_desktop": desk,
            "overall": {"tuned_vis": stats(rows, "tuned", "iou_vis"),
                        "stock_vis": stats(rows, "stock", "iou_vis"),
                        "tuned_amodal": stats(rows, "tuned", "iou_amodal")},
            "low200": [{k: r[k] for k in ("desktop", "step", "inst", "cls", "visibility", "occ",
                                          "vis_area", "amodal_area", "kf_source", "best_vis",
                                          "kind")}
                       | {"tuned_vis": r["tuned"]["iou_vis"],
                          "tuned_amodal": r["tuned"]["iou_amodal"],
                          "stock_vis": r["stock"]["iou_vis"],
                          "sam_area": r["tuned"]["area"]} for r in low]}


def main() -> int:
    analysis = {d: json.load(open(os.path.join(U.QA_ROOT, "analysis", f"D{d:02d}.json"),
                                  encoding="utf-8")) for d in U.DESKTOPS}
    check = {d: parse_check(os.path.join(U.QA_ROOT, "check", f"D{d:02d}.txt"))
             for d in U.DESKTOPS}
    conflicts = open_conflicts()
    items = []
    for d, a in analysis.items():
        res = None
        for f in a["flags"]:
            f = dict(f, source="auto")
            if f["code"] == "d_roi" and f["inst"] != "-":
                # a part swung out of the chassis (displaced / an opened cover) leaves the
                # chassis ROI for real: information, not an error
                res = res or U.load(d)
                st = {r.get("state") for r in res["rows"]
                      if r["inst"] == f["inst"] and r["step"] in f["steps"]}
                if st & {"displaced", "open"}:
                    f["severity"] = "low"
                    f["detail"] += f" (state {sorted(st)}: the part is out of the chassis)"
            if f["code"] == "e_dup" and f.get("other"):
                # cpu_cooler.fan vs cpu_cooler.heatsink: one class, two parts, the fan
                # sits on the heatsink -- overlapping amodal shapes are physics
                kind = lambda k: k.rsplit(".", 1)[0]  # noqa: E731
                if kind(f["inst"]) != kind(f["other"]):
                    f["severity"] = "low"
                    f["detail"] += " (different kinds of one class stacked: expected)"
                else:
                    f["severity"] = "medium"
                    f["score"] = 4.0
                    f["detail"] += " (stacked twins, or one shape drawn twice: look)"
            items.append(f)
        for code, weight in (("shape_predates_state", 4.5), ("cut_copy_in_force", 3.5)):
            for inst, rr in runs_of(check[d]["items"].get(code, [])).items():
                for x, y in rr:
                    items.append({"desktop": d, "code": code, "inst": inst, "cls": None,
                                  "steps": list(range(x, y + 1)), "frames": f"{x}-{y}" if y > x
                                  else str(x), "severity": "high", "source": "cli check",
                                  "detail": f"cli check {code} on confirmed frame(s) {x}-{y}",
                                  "score": weight + min(2, (y - x + 1) / 10)})
    by_conf = defaultdict(list)
    for c in conflicts:
        by_conf[(c["desktop"], c["inst"])].append(c)
    for (d, inst), cs in by_conf.items():
        steps = sorted(c["step"] for c in cs)
        items.append({"desktop": d, "code": "open_conflict", "inst": inst, "cls": None,
                      "steps": steps, "frames": f"{steps[0]}-{steps[-1]}" if len(steps) > 1
                      else str(steps[0]), "severity": "high", "source": "conflict table",
                      "detail": f"{len(cs)} open conflict(s), sym. diff {cs[0]['sym_diff_px']} px, "
                                f"raised {cs[0]['created_at']}",
                      "score": 9.5})
    # the ranked list: one entry per (desktop, instance, code), at most 3 per (desktop, code)
    ranked, seen, per = [], set(), Counter()
    for it in sorted(items, key=lambda f: -f["score"]):
        key = (it["desktop"], it["inst"], it["code"])
        if key in seen or it["severity"] in ("eye", "low"):
            continue
        if per[(it["desktop"], it["code"])] >= 3:
            continue
        seen.add(key)
        per[(it["desktop"], it["code"])] += 1
        ranked.append(it)
    desktops = {}
    for d, a in analysis.items():
        flags = [f for f in items if f["desktop"] == d]
        cnt = Counter(f["code"] for f in flags)
        sev = Counter((f["code"], f["severity"]) for f in flags)
        burden = sum(PACKAGE_WEIGHT.get(f["severity"], 0) for f in flags
                     if f["code"] not in ("change_unexplained",))
        desktops[d] = {"summary": a["summary"], "codes": dict(cnt),
                       "by_severity": {f"{c}/{s}": n for (c, s), n in sev.items()},
                       "burden": round(burden, 1), "check": check[d]}
    out = {"top": ranked[:60], "desktops": desktops, "conflicts": conflicts,
           "sam": sam_summary(), "n_items": len(items)}
    with open(os.path.join(U.QA_ROOT, "ranked.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(U.QA_ROOT, "all_items.json"), "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=0, default=str)
    print(f"{len(items)} items, {len(ranked)} ranked; top 10:")
    for it in ranked[:10]:
        print(f"  D{it['desktop']:02d} {it['code']} {it['inst']} {it['frames']} {it['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
