"""U30-QA, pass 2: automatic flags over the extraction, with picture evidence.

Reads ``<root>/data/D<nn>.pkl`` (``u30_extract.py``) and the scanner frames the
annotator saw (the local cache, read only); writes ``<root>/analysis/D<nn>.json``
and ``<root>/analysis/flags.csv``. Nothing else is touched.

Flags (spec of the task U30-QA, section 2/4/5):

* ``a_empty`` -- a part present (in the chassis, with a shape) whose stored
  visible mask is empty or under 20 px, with no frame override. Grouped into runs
  (same top occluder). The picture test compares the first frame of the run with
  the nearest frame where the same shape version is <= 20 % occluded: if the
  pixels inside the amodal shape did not change, the part is probably in sight
  and the layer order is wrong (``pic_same`` >= 0.8).
* ``b_floor`` -- a chassis or motherboard claims >= 5 px of the part's amodal.
* ``c_below`` -- the occluder is physically below the part by the instance
  table: the part a screw fastens, the host a connector is plugged into, the
  board an attached child sits on, the CPU / socket lever under the cooler.
* ``d_roi`` -- stored visible pixels outside the scanner ROI (> 3 px).
* ``e_dup`` -- two instances of one class with amodal IoU > 0.5 on one frame.
* ``jump`` -- the visible area changes > 50 % between two neighbouring frames of
  one state while the occluder set (and the frame override) did not change;
  ``amodal_jump`` -- the amodal area changes > 50 % within one state.
* ``leave_no_change`` -- a part leaves between frames s-1 and s, but < 15 % of
  its visible pixels at s-1 changed in the picture.
* ``zorder_pixels`` -- a part leaves at s; an occluder that stays claimed >= 30 of
  its amodal pixels at s-1, and >= 50 % of those changed when it left (while the
  occluder's own pixels elsewhere did not): the part was on top there.
* ``change_unexplained`` -- the largest changed blob between s-1 and s that no
  leaving or re-shaped part's amodal (dilated 15 px) explains, >= 1500 px.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tda.core import masks as M  # noqa: E402
import u30_common as U  # noqa: E402

MIN_VIS = 20
OCC_MIN_PX = 5
OCC_MIN_FRAC = 0.02
#: grey std inside a shape below which the picture test has no evidence
LOW_TEXTURE = 12.0


def present(r: dict) -> bool:
    return (r.get("amodal_area") is not None and r.get("amodal_area", 0) > 0
            and (r.get("placement") or r.get("placement_state")) == "in_chassis")


def occ_set(r: dict) -> dict:
    """The occluders that matter: >= 5 px and >= 2 % of the amodal."""
    area = max(1, r.get("amodal_area") or 1)
    return {k: v for k, v in (r.get("occluders") or {}).items()
            if v >= max(OCC_MIN_PX, OCC_MIN_FRAC * area)}


def below_reason(inst: str, occ: str, tab: dict, state: str | None = None
                 ) -> tuple[str, str] | None:
    """``(severity, why)`` when ``occ`` is physically under ``inst``; else None.

    ``state`` is ``inst``'s state on the frame (a cover only hides what it
    covers while it is closed).
    """
    me, him = tab.get(inst) or {}, tab.get(occ) or {}
    mc, hc = me.get("cls"), him.get("cls")
    if hc in U.FLOOR or occ.startswith("<"):
        return None              # floor layers are flag (b); occluder masks are not parts
    of = (me.get("attrs") or {}).get("of")
    if mc == "cover" and of and state in (None, "closed", "installed"):
        role = (him.get("attrs") or {}).get("role")
        if of == "motherboard_screws" and hc == "screw" and role == "motherboard":
            return "high", f"{occ} is one of the screws under the closed screw cover {inst}"
        if of == hc:
            return "high", f"{inst} covers the {hc}: {occ} is under it"
    if me.get("fastens") == occ:
        return "high", f"{inst} fastens {occ}: the screw head is on top of it"
    if me.get("socket_host") == occ:
        return "high", f"{inst} is plugged into {occ}"
    if me.get("parent") == occ and me.get("attached"):
        return "low", f"{inst} is an attached child of {occ} (sits on it)"
    if mc == "cpu_cooler" and hc in ("cpu", "cpu_socket_lever"):
        return "high", f"the {hc} is under the cooler"
    if mc == "ram_module" and hc == "cpu_socket_lever":
        return "low", "socket lever over a RAM module"
    return None


def runs(steps_idx: dict, items: list[tuple[int, object]]):
    """Group ``(step, key)`` into runs of neighbouring annotatable frames with one key."""
    out = []
    for step, key in sorted(items, key=lambda t: (str(t[1]), steps_idx[t[0]])):
        if out and out[-1]["key"] == key and steps_idx[step] == steps_idx[out[-1]["steps"][-1]] + 1:
            out[-1]["steps"].append(step)
        else:
            out.append({"key": key, "steps": [step]})
    return out


def fmt_steps(steps: list[int]) -> str:
    if not steps:
        return ""
    if len(steps) == 1:
        return str(steps[0])
    return f"{steps[0]}-{steps[-1]}"


class Desk:
    def __init__(self, desktop: int):
        self.d = desktop
        self.res = U.load(desktop)
        self.roi = self.res["roi"]
        self.steps = list(self.res["steps"])
        self.idx = {s: i for i, s in enumerate(self.steps)}
        self.R = {(r["step"], r["inst"]): r for r in self.res["rows"]}
        self.tab = self.res["instances"]
        self.rles = self.res["rles"]
        self._cmap = {}
        self._dec = {}
        self.flags: list[dict] = []

    # -- masks / pictures ----------------------------------------------------
    def amodal(self, step, inst):
        key = ("a", step, inst)
        if key not in self._dec:
            self._dec[key] = U.decode((self.rles.get((step, inst)) or {}).get("amodal"))
        return self._dec[key]

    def visible(self, step, inst):
        key = ("v", step, inst)
        if key not in self._dec:
            self._dec[key] = U.decode((self.rles.get((step, inst)) or {}).get("visible"))
        return self._dec[key]

    def drop_masks(self, keep_steps=()):
        self._dec = {k: v for k, v in self._dec.items() if k[1] in keep_steps}

    def cmap(self, a: int, b: int):
        key = (a, b)
        if key not in self._cmap:
            ga, gb = U.gray(self.res, a), U.gray(self.res, b)
            self._cmap[key] = (None if ga is None or gb is None
                               else U.change_maps(ga, gb, self.roi))
            if len(self._cmap) > 6:
                self._cmap.pop(next(iter(self._cmap)))
        return self._cmap[key]

    def flag(self, code, inst, steps, severity, detail, score, **extra):
        r = self.R.get((steps[0], inst)) or {}
        self.flags.append({"desktop": self.d, "code": code, "inst": inst,
                           "cls": (self.tab.get(inst) or {}).get("cls") or r.get("cls"),
                           "steps": list(steps), "frames": fmt_steps(list(steps)),
                           "severity": severity, "detail": detail, "score": round(score, 2),
                           **extra})

    # -- per instance sequences ---------------------------------------------
    def present_steps(self, inst):
        return [s for s in self.steps if (s, inst) in self.R and present(self.R[(s, inst)])]

    def instances(self):
        names = sorted({r["inst"] for r in self.res["rows"] if present(r)})
        return names

    # -- (a) ------------------------------------------------------------------
    def check_a(self):
        items = []
        for (s, inst), r in self.R.items():
            if present(r) and (r.get("vis_area") or 0) < MIN_VIS and not r.get("fo"):
                occ = occ_set(r)
                top = max(occ, key=occ.get) if occ else "-"
                items.append((s, (inst, top)))
        for run in runs(self.idx, items):
            inst, top = run["key"]
            first = run["steps"][0]
            r = self.R[(first, inst)]
            ref, same = self.picture_same(inst, first)
            sev, score = "eye", 1.0
            why = (f"visible {r.get('vis_area', 0)} px of amodal {r['amodal_area']} px on "
                   f"{len(run['steps'])} frame(s); covered by {top}"
                   + (f" (+{', '.join(k for k in occ_set(r) if k != top)})"
                      if len(occ_set(r)) > 1 else ""))
            if ref is None:
                why += "; no frame of this shape version shows it <=20 % occluded"
            elif same is None:
                why += (f"; step {ref} shows it, but the picture has no evidence there "
                        f"(blown out, or flat in both frames)")
            else:
                why += f"; pixels inside its shape vs step {ref} (where it is visible): " \
                       f"{same:.0%} unchanged"
                if same >= 0.8:
                    sev, score = "high", 6.0 + min(4.0, r["amodal_area"] / 500.0)
                elif same >= 0.6:
                    sev, score = "medium", 3.0
            self.flag("a_empty", inst, run["steps"], sev, why, score,
                      occluder=top, ref=ref, pic_same=None if same is None else round(same, 3))

    def picture_same(self, inst, step):
        """``(ref step, fraction of the amodal unchanged vs ref)`` or ``(None, None)``."""
        r0 = self.R[(step, inst)]
        cands = []
        for s in self.present_steps(inst):
            r = self.R[(s, inst)]
            if s == step or r.get("kf_id") != r0.get("kf_id"):
                continue
            if (r.get("occ_stored") or 0) <= 0.2 and not r.get("fo"):
                cands.append((abs(self.idx[s] - self.idx[step]), 0 if s > step else 1, s))
        if not cands:
            return None, None
        ref = min(cands)[2]
        a = self.amodal(step, inst)
        ga, gb = U.gray(self.res, step), U.gray(self.res, ref)
        if a is None or ga is None or gb is None or not a.any():
            return None, None
        # a flat dark occluder over a flat dark part reads "unchanged": no evidence
        if float(ga[a].std()) < LOW_TEXTURE and float(gb[a].std()) < LOW_TEXTURE:
            return ref, None
        changed, valid = U.change_maps(ga, gb, self.roi)
        f = U.frac(changed, valid, a)
        return (ref, None) if f is None else (ref, 1.0 - f)

    # -- (b) (c) ------------------------------------------------------------------
    def check_bc(self):
        items_b, items_c = [], []
        for (s, inst), r in self.R.items():
            if not present(r):
                continue
            mcls = (self.tab.get(inst) or {}).get("cls") or r.get("cls")
            for occ, px in occ_set(r).items():
                ocls = (self.tab.get(occ) or {}).get("cls")
                # the board on the chassis is the one right order between the floors
                if ocls == "chassis" and mcls != "chassis" or \
                        ocls == "motherboard" and mcls not in U.FLOOR:
                    items_b.append((s, (inst, occ)))
            # a small part over a big one is below the 2 % of occ_set: (c) takes any
            # occluder that claims >= 20 px
            for occ, px in (r.get("occluders") or {}).items():
                if px < 20:
                    continue
                why = below_reason(inst, occ, self.tab, r.get("state"))
                if why:
                    items_c.append((s, (inst, occ, why)))
        for run in runs(self.idx, items_b):
            inst, occ = run["key"]
            r = self.R[(run["steps"][0], inst)]
            px = r["occluders"][occ]
            self.flag("b_floor", inst, run["steps"], "high",
                      f"{occ} ({(self.tab.get(occ) or {}).get('cls')}) claims {px} px of its "
                      f"amodal {r['amodal_area']} px", 5.0 + min(3, px / 200), occluder=occ)
        for run in runs(self.idx, items_c):
            inst, occ, (sev, why) = run["key"]
            r = self.R[(run["steps"][0], inst)]
            px = r["occluders"][occ]
            self.flag("c_below", inst, run["steps"], sev,
                      f"occluded by {occ} ({px} px of amodal {r['amodal_area']}): {why}",
                      (5.0 if sev == "high" else 1.5) + min(3, px / 100), occluder=occ)

    # -- (d) ------------------------------------------------------------------
    def check_d(self):
        if self.roi is None:
            self.flag("d_roi", "-", [self.steps[0]], "medium",
                      "this desktop has no scanner ROI stored (pose_segment.roi_json NULL)", 2.0)
            return
        items = []
        for (s, inst), r in self.R.items():
            if (r.get("vis_outside_roi") or 0) > 3:
                items.append((s, inst))
        out_am = [(s, inst) for (s, inst), r in self.R.items()
                  if present(r) and (r.get("vis_outside_amodal") or 0) > 20]
        for run in runs(self.idx, out_am):
            inst = run["key"]
            r = self.R[(run["steps"][0], inst)]
            px = max(self.R[(s, inst)]["vis_outside_amodal"] for s in run["steps"])
            self.flag("vis_outside_amodal", inst, run["steps"], "medium",
                      f"up to {px} stored visible px lie outside the amodal shape "
                      f"(frame override: {r.get('fo')}; visible {r.get('vis_area')} vs amodal "
                      f"{r.get('amodal_area')} px)", 2.0 + min(3, px / 200), px=px)
        for run in runs(self.idx, items):
            inst = run["key"]
            px = max(self.R[(s, inst)]["vis_outside_roi"] for s in run["steps"])
            self.flag("d_roi", inst, run["steps"], "medium" if px > 50 else "low",
                      f"up to {px} visible px outside the ROI {self.roi}",
                      1.0 + min(3, px / 200), px=px)

    # -- (e) ------------------------------------------------------------------
    def check_e(self):
        items = []
        for s in self.steps:
            by_cls = {}
            for (ss, inst), r in self.R.items():
                if ss == s and present(r):
                    by_cls.setdefault(r.get("cls"), []).append(inst)
            for cls, names in by_cls.items():
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        ra, rb = self.R[(s, a)], self.R[(s, b)]
                        ba, bb = ra.get("amodal_bbox"), rb.get("amodal_bbox")
                        if not ba or not bb or ba[0] >= bb[2] or bb[0] >= ba[2] \
                                or ba[1] >= bb[3] or bb[1] >= ba[3]:
                            continue
                        iou = M.rle_iou(self.rles[(s, a)]["amodal"], self.rles[(s, b)]["amodal"])
                        if iou > 0.5:
                            items.append((s, (a, b, round(iou, 2))))
        for run in runs(self.idx, [(s, (a, b)) for s, (a, b, _i) in items]):
            a, b = run["key"]
            ious = [i for s, (x, y, i) in items if (x, y) == (a, b)]
            self.flag("e_dup", a, run["steps"], "high",
                      f"{a} and {b} (same class) overlap with amodal IoU up to {max(ious)}",
                      8.0, other=b)

    # -- a shape in pieces far apart (two latches in one, a stray stroke) ----------
    def check_split(self):
        seen = {}
        for (s, inst), r in sorted(self.R.items()):
            if not present(r) or r.get("kf_id") is None:
                continue
            if (self.tab.get(inst) or {}).get("cls") in U.FLOOR:
                continue
            seen.setdefault((inst, r["kf_id"]), []).append(s)
        for (inst, kf), steps in seen.items():
            a = self.amodal(steps[0], inst)
            if a is None:
                continue
            bb = U.bbox_of(a)
            x0, y0, x1, y1 = bb
            sub = a[y0:y1, x0:x1].astype(np.uint8)
            n, _lab, stats, cents = cv2.connectedComponentsWithStats(sub, connectivity=8)
            if n <= 2:
                continue
            order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
            big, second = order[0], order[1]
            a1, a2 = int(stats[big, cv2.CC_STAT_AREA]), int(stats[second, cv2.CC_STAT_AREA])
            dist = float(np.hypot(*(cents[big] - cents[second])))
            if a2 >= 30 and a2 >= 0.1 * (a1 + a2) and dist > 2.5 * np.sqrt(a1):
                self.flag("split_shape", inst, steps, "high",
                          f"its amodal shape (keyframe {kf}, frames {fmt_steps(steps)}) is in "
                          f"{n - 1} pieces: {a1} px and {a2} px, {dist:.0f} px apart -- two "
                          f"parts in one shape, or a stray stroke", 6.0 + min(3, a2 / 300),
                          pieces=n - 1)
            self._dec.pop(("a", steps[0], inst), None)

    # -- (4) jumps ------------------------------------------------------------
    def check_jumps(self):
        for inst in self.instances():
            seq = self.present_steps(inst)
            for s1, s2 in zip(seq, seq[1:]):
                if self.idx[s2] != self.idx[s1] + 1:
                    continue
                r1, r2 = self.R[(s1, inst)], self.R[(s2, inst)]
                if (r1.get("state"), r1.get("placement")) != (r2.get("state"), r2.get("placement")):
                    continue
                a1, a2 = r1.get("vis_area") or 0, r2.get("vis_area") or 0
                m1, m2 = r1["amodal_area"], r2["amodal_area"]
                same_occ = set(occ_set(r1)) == set(occ_set(r2)) and \
                    (r1.get("fo") or None) == (r2.get("fo") or None)
                for o in occ_set(r1):
                    o1, o2 = self.R.get((s1, o)), self.R.get((s2, o))
                    if o.startswith("<") or not o1 or not o2:
                        continue
                    # an occluder that changed shape or state is a change of occluder
                    if o1.get("kf_id") != o2.get("kf_id") or o1.get("state") != o2.get("state"):
                        same_occ = False
                kf = (r1.get("kf_id"), r2.get("kf_id"))
                if max(a1, a2) >= 50 and abs(a1 - a2) > 0.5 * max(a1, a2) and same_occ:
                    self.flag("jump", inst, [s1, s2], "high",
                              f"visible {a1} -> {a2} px (amodal {m1} -> {m2}, keyframe "
                              f"{kf[0]} -> {kf[1]}), same state '{r1.get('state')}', same "
                              f"occluders {sorted(occ_set(r1)) or '-'}",
                              4.0 + min(4, abs(a1 - a2) / 1000))
                if max(m1, m2) >= 50 and abs(m1 - m2) > 0.5 * max(m1, m2):
                    self.flag("amodal_jump", inst, [s1, s2], "high",
                              f"amodal {m1} -> {m2} px within state '{r1.get('state')}' "
                              f"(keyframe {kf[0]} -> {kf[1]})", 4.0 + min(4, abs(m1 - m2) / 1000))

    # -- (5) log vs picture + z-order by pixels -----------------------------------
    def check_leaves(self):
        dil = np.ones((31, 31), np.uint8)
        summary = []
        for s0, s1 in zip(self.steps, self.steps[1:]):
            pres0 = {i for (s, i), r in self.R.items() if s == s0 and present(r)}
            pres1 = {i for (s, i), r in self.R.items() if s == s1 and present(r)}
            leaving = sorted(pres0 - pres1)
            appearing = sorted(pres1 - pres0)
            reshaped = sorted(i for i in pres0 & pres1
                              if self.R[(s0, i)].get("kf_id") != self.R[(s1, i)].get("kf_id")
                              or self.R[(s0, i)].get("state") != self.R[(s1, i)].get("state"))
            cv = self.cmap(s0, s1)
            if cv is None:
                continue
            c, valid = cv
            explained = np.zeros_like(c)
            for i in leaving + reshaped:
                a = self.amodal(s0, i)
                if a is not None:
                    explained |= a
            for i in reshaped + appearing:
                a = self.amodal(s1, i)
                if a is not None:
                    explained |= a
            explained = cv2.dilate(explained.astype(np.uint8), dil).astype(bool)
            if self.roi:
                x0, y0, x1, y1 = self.roi
                inroi = np.zeros_like(c)
                inroi[y0:y1, x0:x1] = True
            else:
                inroi = np.ones_like(c)
            un = (c & ~explained & inroi).astype(np.uint8)
            un = cv2.morphologyEx(un, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(un, connectivity=8)
            blob, blob_area = None, 0
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, area = (int(v) for v in stats[k])
                blob, blob_area = [x, y, x + w, y + h], area
            frac = U.frac(c, valid, inroi) or 0.0
            stype = (self.res["steps_meta"].get(s1) or {}).get("type")
            acts = [f"{a['verb']} {a['target']}" for a in self.res["actions"].get(s1, [])]
            summary.append({"from": s0, "to": s1, "changed": round(frac, 4), "leaving": leaving,
                            "reshaped": reshaped, "appearing": appearing,
                            "unexplained_blob": blob, "unexplained_px": blob_area,
                            "step_type": stype, "actions": acts})
            if blob_area >= 1500:
                sev = "medium" if blob_area >= 5000 else "low"
                self.flag("change_unexplained", "-", [s0, s1], sev,
                          f"step {s1} ({stype}; {'; '.join(acts) or 'no action'}): a "
                          f"{blob_area} px changed blob at {blob} that no leaving or re-shaped "
                          f"part explains (leaving: {', '.join(leaving) or 'none'})",
                          1.0 + min(4, blob_area / 4000), blob=blob)
            # a part that stays, unchanged, and is visible on both frames should look the
            # same there; when its visible pixels change, something covered or uncovered
            # it (or it moved) and the annotation did not follow
            for i in sorted((pres0 & pres1) - set(reshaped)):
                r0, r1 = self.R[(s0, i)], self.R[(s1, i)]
                if (r0.get("vis_area") or 0) < 50 or (r1.get("vis_area") or 0) < 50:
                    continue
                if r0.get("fo") or r1.get("fo"):
                    continue
                v0, v1 = self.visible(s0, i), self.visible(s1, i)
                if v0 is None or v1 is None:
                    continue
                both = v0 & v1
                if both.sum() < 50:
                    continue
                f = U.frac(c, valid, both)
                if f is None or f < 0.5:
                    continue
                if (self.tab.get(i) or {}).get("cls") in U.FLOOR:
                    continue      # the chassis / board show everything that moves on them
                px = int(both.sum())
                # whose shape lay over these pixels: a part that left or moved at s1
                movers = []
                for j in leaving + reshaped:
                    if j == i or (self.tab.get(j) or {}).get("cls") in U.FLOOR:
                        continue      # a floor's amodal covers everything: no evidence
                    aj = self.amodal(s0, j)
                    if aj is not None and (aj & both).sum() >= 0.3 * px:
                        movers.append(j)
                if movers:
                    self.flag("visible_changed", i, [s0, s1], "high",
                              f"visible on {s0} and {s1} with the same shape and state "
                              f"'{r0.get('state')}' ({px} px), yet {f:.0%} of those pixels "
                              f"changed when {', '.join(movers)} left / moved at {s1} -- and "
                              f"{'its' if len(movers) == 1 else 'their'} shape at {s0} covers "
                              f"them: {', '.join(movers)} is drawn BELOW {i} there but was on "
                              f"top in the picture", 5.0 + min(4, px * f / 200),
                              changed=round(f, 3), movers=movers)
                else:
                    self.flag("visible_changed", i, [s0, s1], "low",
                              f"visible on {s0} and {s1} with the same shape and state ({px} px), "
                              f"yet {f:.0%} of those pixels changed (step {s1}: "
                              f"{'; '.join(acts) or 'no action'}); no annotated part moved there "
                              f"-- a cable, a hand, or the part itself moved", 1.0 + min(2, px * f / 400),
                              changed=round(f, 3))
            if leaving:
                expected = sum(int(self.R[(s0, i)].get("vis_area") or 0) for i in leaving)
                changed_px = int((c & inroi).sum())
                if expected >= 150 and changed_px < 0.25 * expected and frac < 0.002:
                    self.flag("frame_unchanged", "-", [s0, s1], "high",
                              f"step {s1} ({stype}; {'; '.join(acts) or 'no action'}) says "
                              f"{', '.join(leaving)} left ({expected} visible px at {s0}), but "
                              f"frames {s0} and {s1} are the same picture ({changed_px} px, "
                              f"{frac:.2%} of the ROI changed)", 7.0 + min(3, expected / 1000),
                              expected_px=expected, changed_px=changed_px)
            for i in leaving:
                r = self.R[(s0, i)]
                v = self.visible(s0, i)
                a = self.amodal(s0, i)
                if v is not None and v.sum() >= 30:
                    fv = U.frac(c, valid, v)
                    fa = U.frac(c, valid, a) if a is not None else None
                    connector = (self.tab.get(i) or {}).get("cls") == "connector"
                    if fv is not None and fv < 0.15:
                        sev = "high" if v.sum() >= 300 and not connector else "medium"
                        self.flag("leave_no_change", i, [s0, s1], sev,
                                  f"leaves between {s0} and {s1} ({stype}; "
                                  f"{'; '.join(acts) or 'no action'}) but only {fv:.0%} of its "
                                  f"{int(v.sum())} visible px (amodal "
                                  f"{'-' if fa is None else f'{fa:.0%}'}) changed in the picture",
                                  (1.5 if connector else 3.0) + min(5, v.sum() / 400),
                                  changed_vis=round(fv, 3))
                hidden = (a & ~v) if (a is not None and v is not None) else None
                if hidden is None or not hidden.any():
                    continue
                for occ in occ_set(r):
                    if occ.startswith("<") or occ not in pres1:
                        continue
                    vo = self.visible(s0, occ)
                    if vo is None:
                        continue
                    h = hidden & vo
                    if h.sum() < 30:
                        continue
                    fh = U.frac(c, valid, h)
                    rest = vo & ~cv2.dilate(a.astype(np.uint8), dil).astype(bool)
                    base = (U.frac(c, valid, rest) if rest.any() else 0.0) or 0.0
                    if fh is not None and fh >= 0.5 and base < 0.2:
                        px = int(h.sum())
                        self.flag("zorder_pixels", i, [s0, s1], "high",
                                  f"{occ} (stays) claims {px} px of {i}'s amodal at {s0}; "
                                  f"{fh:.0%} of them changed when {i} left at {s1} "
                                  f"({occ}'s other pixels: {base:.0%}) -> {i} looks to be on "
                                  f"top there", 5.0 + min(5, px / 150), occluder=occ,
                                  changed=round(fh, 3))
            self.drop_masks(keep_steps=(s1,))
        return summary

    # -- what to draw -----------------------------------------------------------
    def render_list(self):
        flagged = {}
        for f in self.flags:
            if f["inst"] != "-":
                flagged.setdefault(f["inst"], []).append(f)
        out = []
        for inst in self.instances():
            seq = self.present_steps(inst)
            if not seq:
                continue
            picks = [(seq[-1], "first met")]
            occs = sorted(((self.R[(s, inst)].get("occ_stored") or 0, -self.idx[s], s)
                           for s in seq), reverse=True)
            n_occ = 0
            for occ, _neg, s in occs:
                if occ < 0.05 or n_occ >= 2:
                    break
                if any(p[0] == s for p in picks):
                    continue
                if n_occ == 1 and set(occ_set(self.R[(s, inst)])) == set(
                        occ_set(self.R[(picks[-1][0], inst)])):
                    continue
                picks.append((s, f"max occlusion {occ:.2f}" if n_occ == 0 else
                              f"occlusion {occ:.2f}"))
                n_occ += 1
            for f in sorted(flagged.get(inst, []), key=lambda f: -f["score"])[:2]:
                s = f["steps"][0] if f["code"] not in ("leave_no_change", "zorder_pixels") \
                    else f["steps"][0]
                if (s, inst) in self.R and present(self.R[(s, inst)]) and \
                        not any(p[0] == s for p in picks):
                    picks.append((s, f["code"]))
            for s, why in picks:
                out.append({"step": s, "inst": inst, "why": why,
                            "flags": sorted({f["code"] for f in flagged.get(inst, [])
                                             if s in f["steps"]})})
        return out

    def summary(self):
        rows = self.res["rows"]
        pres = [r for r in rows if present(r)]
        kf_src = {}
        for r in pres:
            kf_src[r.get("kf_source")] = kf_src.get(r.get("kf_source"), 0) + 1
        return {"desktop": self.d, "brand": self.res["brand"], "model": self.res["model"],
                "frames": len(self.steps), "instances": len(self.instances()),
                "part_frames": len(pres), "roi": self.roi,
                "verified": sum(1 for s in self.steps
                                if self.res["frames"][s]["review"] == "verified"),
                "open_conflicts": sorted({(r["step"], r["inst"]) for r in rows if r.get("conflict")}),
                "kf_sources": kf_src,
                "fo": sum(1 for r in rows if r.get("fo")),
                "occluded_full": sum(1 for r in pres if r.get("visibility") == "occluded_full"),
                "stored_fresh_diff": sum(1 for r in pres if abs((r.get("vis_area") or 0)
                                                                 - (r.get("vis_area_fresh") or 0)) > 20)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desktops", default=None, help="e.g. 2,3 (default: the 30)")
    args = ap.parse_args(argv)
    wanted = U.DESKTOPS if not args.desktops else [int(x) for x in args.desktops.split(",")]
    out_dir = os.path.join(U.QA_ROOT, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    all_flags = []
    for d in wanted:
        t0 = time.time()
        dk = Desk(d)
        dk.check_a()
        dk.check_bc()
        dk.check_d()
        dk.check_e()
        dk.check_split()
        dk.check_jumps()
        steps = dk.check_leaves()
        res = {"summary": dk.summary(), "flags": dk.flags, "steps": steps,
               "render": dk.render_list()}
        with open(os.path.join(out_dir, f"D{d:02d}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
        all_flags.extend(dk.flags)
        counts = {}
        for f in dk.flags:
            counts[f["code"]] = counts.get(f["code"], 0) + 1
        print(f"D{d:02d}: {len(dk.flags)} flags {counts} ({time.time() - t0:.0f}s)", flush=True)
    if not args.desktops:
        with open(os.path.join(out_dir, "flags.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["desktop", "code", "severity", "score", "inst", "cls", "frames", "detail"])
            for f in sorted(all_flags, key=lambda f: -f["score"]):
                w.writerow([f["desktop"], f["code"], f["severity"], f["score"], f["inst"],
                            f["cls"], f["frames"], f["detail"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
