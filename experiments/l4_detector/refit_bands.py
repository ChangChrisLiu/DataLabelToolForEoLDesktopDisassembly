"""Re-fit the scanner's area bands (``configs/prompt_box_bands.yaml``) on the verified frames.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.refit_bands [--apply]

``experiments/l1_localise/fit_bands.py``'s method, unchanged in what it
computes (its ``band_of``: the 5th-95th percentile over **parts**, >= 5 parts
from >= 2 desktops or unbanded; its ``render``), fed with the annotator's
verified data instead of the old ``ls:`` drafts:

* a part is ``(desktop, instance)`` of the verified scan frames (``gt.py``:
  in_chassis, visibility visible / visible_tiny / occluded_partial /
  too_small, a non-empty visible mask);
* its value is the median of its *visible-mask* area over the verified frames
  it has one on -- what the difference map's blob of a removed part measures,
  as the drafts' visible areas did; ``keyframes`` counts those frames;
* the class is ``instance.cls``.

Only the ``scan`` entry is replaced: oak1 / oak2 / rs have no verified frames
and keep the LS fit.  Without ``--apply`` the candidate goes to
``experiments_out/l4_detector/bands/`` with a diff against the shipped file;
``--apply`` writes ``configs/prompt_box_bands.yaml`` (decision_rule.txt: only
when tests stay green and no shared class's hi/lo grows beyond 2x).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402
from experiments.l4_detector import gt as G  # noqa: E402
from experiments.l1_localise import fit_bands as F  # noqa: E402

OUT = env.OUT / "bands"
CONFIG = env.REPO / "configs" / "prompt_box_bands.yaml"


def verified_parts(frames) -> dict[str, list[tuple[float, int, int]]]:
    """``{cls: [(median area, desktop, frames), ...]}`` -- fit_bands' part_values shape."""
    areas: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for f in frames:
        for b in f.boxes:
            areas[(b.cls, b.desktop, b.instance)].append(b.area)
    out: dict[str, list] = defaultdict(list)
    for (cls, desktop, _inst), vals in sorted(areas.items()):
        out[cls].append((float(np.median(vals)), desktop, len(vals)))
    return out


def fit_scan(frames) -> dict:
    parts = verified_parts(frames)
    classes, unbanded = {}, {}
    for cls in sorted(parts):
        band = F.band_of(parts[cls])
        if band is None:
            unbanded[cls] = {"parts": len(parts[cls]),
                             "desktops": len({d for _v, d, _n in parts[cls]})}
        else:
            classes[cls] = band
    return {"desktops": sorted({f.desktop for f in frames}), "classes": classes,
            "unbanded": unbanded}


def replay(frames, shipped_scan: dict, k: float = 3.0) -> list[dict]:
    """The U2h gate on the verified removal events, old bands vs new (L1's replay_m1_units idea).

    Per event: the difference map's rank-1 blob (``evaluate.m0app``: its pixel
    count ``a1`` and whether its point is on the part) judged against the band
    of the event's class with factor ``k``.  ``cut`` = wrong boxes withheld,
    ``withheld`` = good boxes withheld.  New bands leave-one-desktop-out (fit
    without the event's desktop) and in-sample; the shipped LS bands as they are
    (they were fitted on drafts that include D2 and D13).
    """
    from experiments.l4_detector.evaluate import M0_CACHE

    m0 = json.loads(M0_CACHE.read_text(encoding="utf-8"))
    con = G.connect()
    evs, _ = G.events(frames, con)
    con.close()
    loo = {d: fit_scan([f for f in frames if f.desktop != d])["classes"] for d in G.DESKTOPS}
    ins = fit_scan(frames)["classes"]
    arms = {"shipped (LS)": lambda ev: shipped_scan["classes"],
            "verified, leave-one-desktop-out": lambda ev: loo[ev.desktop],
            "verified, in-sample": lambda ev: ins}
    out = []
    for name, bands_of in arms.items():
        cut = wrong = held = good = 0
        for ev in evs:
            rec = m0.get(ev.key) or {}
            if rec.get("a1") is None:
                continue
            band = bands_of(ev).get(ev.cls)
            withheld = False
            if band is not None:
                a = float(rec["a1"])
                lo, hi = band["lo_px"], band["hi_px"]
                factor = lo / a if a < lo else (a / hi if a > hi else 1.0)
                withheld = factor > k
            if rec["in1"]:
                good += 1
                held += int(withheld)
            else:
                wrong += 1
                cut += int(withheld)
        out.append({"bands": name, "wrong_cut": f"{cut}/{wrong}",
                    "wrong_cut_pct": round(100 * cut / max(1, wrong), 1),
                    "good_withheld": f"{held}/{good}",
                    "good_withheld_pct": round(100 * held / max(1, good), 1)})
    return out


def diff_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for cls in sorted(set(old["classes"]) | set(new["classes"]) | set(old.get("unbanded") or {})
                      | set(new.get("unbanded") or {})):
        o, n = old["classes"].get(cls), new["classes"].get(cls)
        r = {"cls": cls,
             "old": f"{o['lo_px']:g}-{o['hi_px']:g} (med {o['median_px']:g}, {o['parts']} parts / "
                    f"{o['desktops']} desktops)" if o else
                    f"unbanded {(old.get('unbanded') or {}).get(cls)}",
             "new": f"{n['lo_px']:g}-{n['hi_px']:g} (med {n['median_px']:g}, {n['parts']} parts / "
                    f"{n['desktops']} desktops)" if n else
                    f"unbanded {(new.get('unbanded') or {}).get(cls)}"}
        if o and n:
            ro, rn = o["hi_px"] / o["lo_px"], n["hi_px"] / n["lo_px"]
            r["hi/lo old"] = round(ro, 2)
            r["hi/lo new"] = round(rn, 2)
            r["width x"] = round(rn / ro, 2)
            r["blows_up"] = int(rn > 2 * ro)
        rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    frames = G.load()
    shipped = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    new_scan = fit_scan(frames)
    rows = diff_rows(shipped["views"]["scan"], new_scan)
    bands = dict(shipped["views"])
    bands["scan"] = new_scan
    source = (f"scan: the {len(frames)} verified scan frames of D2/3/4/5/13 "
              "(.cache/tmp/l4/tda_l4.sqlite, 2026-10-16; per part the median visible-mask "
              "area over its verified frames, by experiments/l4_detector/refit_bands.py); "
              f"oak1/oak2/rs: unchanged, {shipped['fitted_from']}")
    text = F.render(bands, source)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompt_box_bands_verified.yaml").write_bytes(text.encode("utf-8"))
    (OUT / "scan_band_diff.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(json.dumps(r))
    blown = [r["cls"] for r in rows if r.get("blows_up")]
    print(f"[l4] classes whose hi/lo grew > 2x: {blown or 'none'}")
    rep = replay(frames, shipped["views"]["scan"])
    (OUT / "gate_replay.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for r in rep:
        print(json.dumps(r))
    if args.apply:
        if blown:
            raise SystemExit("not applied: the rule says no class's band may widen > 2x")
        CONFIG.write_bytes(text.encode("utf-8"))
        print(f"[l4] wrote {CONFIG}")
    else:
        print(f"[l4] candidate in {OUT / 'prompt_box_bands_verified.yaml'} (not applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
