"""Controller: U31-fix -- the bulk corrections U30-QA found on the 30 verified desktops.

What it does, per desktop of the scanner view (D2 is never touched: the user is checking it):

A. **Layering** (``set_zorder_move``, one ``set_zorder`` op with its inverse per gesture,
   exactly what ``Ctrl+Up`` / ``L`` writes).  The pairs come from the U30-QA machine lists
   (``all_items.json`` / ``reach.json``):

   * ``cage``         HP: ``drive_cage.01`` (D29/D31 also ``psu.01``) above every part QA saw
                      drawn over it before it moves away (``reach.json``);
   * ``screw_cover``  Dell (+ D24, D27): the motherboard screw cover above the screw it covers
                      (``c_below`` "one of the screws under the closed screw cover");
   * ``dell_shroud``  Dell: the CPU fan and its screws QA saw through the shroud's fan opening
                      (``a_empty`` high) above the shroud (``cooler_latch.01`` / ``cover.01``);
   * ``hp_shroud``    HP D23 / D27 / D28: the shroud above the heatsink (``c_below`` "covers the
                      cpu_cooler") and, on D23, above the two screws ``reach.json`` lists;
   * ``item29``       D3 / D8 / D16: the latch / lever / heatsink above the one screw QA saw
                      drawn over it (QA item 29).

   The order is per (view, pose segment), so a gesture reaches every frame of the segment.
   Every candidate is therefore compiled over the whole segment with the new order (nothing
   written) and compared part by part with the old one:

   * **raise** -- each mover still below the highest target goes directly above it (movers
     keep their relative order); **lower** -- each target still above the lowest mover goes
     directly under it; **raise_keep** -- raise, then the parts the picture says lie on top
     of a mover are lifted back above it;
   * a part outside the group that the raise would newly cover is looked at in the picture
     (U30-QA's change map between the last frame the cover is in place and the next; a pixel
     blown out in one of the two only counts as changed): >= 50 % changed -- the cover lay on
     it -- it joins the targets ("taken in", reported); otherwise it is kept above;
     ``EXTRA_MOVERS`` / ``EXTRA_TARGETS`` are parts added by eye where the map undersells it;
     ``--no-additions`` turns the taking-in off (then only QA's pairs change);
   * changes under the app's own conflict floor (<= 20 px, label unchanged) are ignored;
   * a change to a group member *after* QA's last frame (the cover has opened / moved / left
     by then) is a different physical situation: allowed only with the same picture evidence,
     otherwise that target is dropped ("out of reach of one order", reported);
   * the first candidate that changes nothing else wins (raise, raise_keep, lower); none --
     the group is refused.  A move that would leave a chassis / motherboard above anything it
     was not above before is refused too.

B. **HP cable clip** (D23-D26, D29-D33): is ``cable_clip.01`` still in its place after the
   shroud leaves at step 3?  Its frame-2 patch (its stored visible mask on the last frame
   before step 3) against every later frame of the segment: the masked NCC the controller
   asked for (best over +-6 px, >= 0.8 at <= 3 px) **or** at most 20 % of it changed in
   U30-QA's change map = in place -- the black clip on a black board is too flat for NCC
   alone (see :func:`measure_clip`).  In place on 3.. and gone from step k for good, where
   one part leaves: the Steps panel makes that part its parent (attached), else parent
   none / attached off (in the chassis to the end) -- U27 ``relation_fix`` ops either way;
   then its frame-2 shape is committed on the last frame it is in (``commit_edit``, scope
   keyframe: one ``commit_keyframe`` op) so those frames have its geometry.  D23 keeps the
   user's ruling (leaves with the shroud) whatever the numbers say; an ambiguous measure is
   left alone and reported.  Only a clip still parented to the shroud (``cover.*``) is
   ruled: once moved, a re-run measures it for the record and leaves it alone.

C. **Conflicts**: the whole scan view is refreshed and every conflict the changes opened is
   accepted (``accept_new``).  The conflicts open before the run are the user's: their
   (frame, part) keys are read from the database before anything changes, and every conflict
   on one of those keys stays open -- also the new id the app gives one it supersedes (today
   D22's 22 on ``connector.front_panel.04`` and D27's chassis).

D. **Before / after sheets** (``<out>/before_after/D<nn>.jpg``) of the worst six affected
   (frame, part) crops, and the clip evidence (``<out>/clip_D<nn>.jpg``).

Usage::

    python scripts/fix_u31_zorder.py [--db PATH] [--apply] [--desktops 3,22,...]
                                     [--qa DIR] [--out DIR] [--no-render] [--no-additions]

Without ``--apply`` it is a dry run: the whole thing is carried out on a backup-API copy of
``--db`` in ``--out`` (``dryrun_<stamp>.sqlite``), never on ``--db`` itself.  ``--db``
defaults to the LIVE database; ``--apply`` on it first takes the app's own safety backup
(``backup_dir`` of ``paths.yaml``) and refuses while the annotator app is running or holds
the lock.  Every run writes ``<out>/u31_<mode>_<stamp>.log`` (with the safety backup's
path) and ``u31_summary_*.json``.  Idempotent: a group already satisfied plans no move, a
clip whose parent is no longer the shroud is not ruled again, a shape already there is kept.

**If a ``--apply`` run fails part-way, restore the safety backup the log names; do not
re-run on top of it**: the conflicts the half-done run queued would then read as the user's
open questions and never be accepted.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda import pipeline as P  # noqa: E402
from tda.core import masks, rawroot  # noqa: E402
from tda.core.compiler import select_keyframe  # noqa: E402
from tda.core.model import FrameKey, ZOrderRec, machine_only  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402
from tda.core.truth import TruthService  # noqa: E402
from tda.core.truth_inputs import InputCache, gather, pose_segment_of, state_of  # noqa: E402
from tda.core.zorder_floor import FLOOR_CLASSES, floor_inversions  # noqa: E402

LIVE = r"D:\DataSet\annotations\tda.sqlite"
QA_DIR = r"D:\DataSet\.cache\tmp\u30qa"
OUT_DIR = r"D:\DataSet\.cache\tmp\u31"
VIEW = "scan"
ANNOTATOR = "controller"
MAIN = "main"

#: The user is checking D2 himself: never planned, never touched.
OFF_LIMITS = frozenset({2})
DELL = frozenset(range(2, 22))
HP = frozenset(range(22, 34))
#: Where the clip's ruling is measured; D23 keeps the user's ruling (leaves with the shroud).
CLIP_DESKTOPS = (23, 24, 25, 26, 29, 30, 31, 32, 33)
CLIP_KEEP_RULING = frozenset({23})
CLIP = "cable_clip.01"
CLIP_NCC = 0.8           # still there: masked NCC >= this ...
CLIP_SHIFT_OK = 3        # ... at a shift of at most this many pixels
CLIP_SEARCH = 6          # shifts tried: +-6 px in x and y
#: ... or at most this fraction of the clip's pixels changed (U30-QA's change map)
CLIP_CHANGE_MAX = 0.2
#: QA item 29: the single screws drawn over a latch / lever / heatsink.
ITEM29_DESKTOPS = frozenset({3, 8, 16})
SHROUD_CLASSES = frozenset({"cover", "cooler_latch"})
#: The clip is ruled only while it still hangs on the shroud (its parent ``cover.*``, the
#: state before U31): a clip some ruling already moved is never re-ruled by a re-run.
CLIP_SHROUD_PREFIX = "cover."

KINDS = ("cage", "screw_cover", "dell_shroud", "hp_shroud", "item29")
#: Looked at by the controller's worker (U31) and added to a QA group:
#: D21 frame 1 vs 5 -- the fourth heatsink screw is in plain sight through the shroud's
#: fan opening exactly like the other three; QA had no evidence either way (flat).
EXTRA_MOVERS = {(21, "dell_shroud"): ["screw.cpu_cooler.04"]}
#: ... and to a QA group's targets: parts under the cover on the QA frames by eye where
#: the change map undersells it.  D24 frame 8 vs 9: ram_latch.06 lies under the cage's
#: bright metal rail (the white latch only shows on 9); bright on bright reads 33 %.
#: D23 frames 1-2 vs 3 (U31 review B1): where connector.fan.01 is drawn the photo shows the
#: shroud's flat black plastic with its ridge running straight through the box; the 4-pin
#: header only appears on frame 3, once the shroud is out -- it lies under the shroud, and
#: the 8 % the change map reads is black on black, not evidence that it is on top.
EXTRA_TARGETS = {(24, "cage"): ["ram_latch.06"],
                 (23, "hp_shroud"): ["connector.fan.01"]}


# --------------------------------------------------------------------------- #
# the plan, from the QA lists
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Group:
    """``movers`` go above every one of ``targets`` on this desktop's scan view."""

    desktop: int
    kind: str
    movers: list[str]
    targets: list[str]
    why: list[str] = dataclasses.field(default_factory=list)
    #: the last frame on which QA saw the problem: the cover is still in place there,
    #: the next frame is where it opens / moves / leaves
    qa_last: int = 0


def _frames_of(text: str) -> list[int]:
    a, _, b = str(text).partition("-")
    return list(range(int(a), int(b or a) + 1))


def load_qa(qa_dir: str) -> tuple[list[dict], dict]:
    items = json.loads(Path(qa_dir, "all_items.json").read_text(encoding="utf-8"))
    reach = json.loads(Path(qa_dir, "reach.json").read_text(encoding="utf-8"))
    return items, {int(k): v for k, v in reach.items()}


def derive_groups(items: list[dict], reach: dict[int, list[dict]],
                  desktops: Optional[Iterable[int]] = None) -> list[Group]:
    """The layering groups the QA lists name, D2 left out."""
    wanted = None if desktops is None else set(desktops)
    groups: dict[tuple, Group] = {}

    def add(d: int, kind: str, mover: str, target: str, why: str, frames: list[int]) -> None:
        if d in OFF_LIMITS or (wanted is not None and d not in wanted):
            return
        g = groups.setdefault((d, kind, mover if kind != "dell_shroud" else target),
                              Group(d, kind, [], []))
        if mover not in g.movers:
            g.movers.append(mover)
        if target not in g.targets:
            g.targets.append(target)
        g.why.append(why)
        g.qa_last = max([g.qa_last, *frames])

    for d, rows in reach.items():
        for r in rows:
            mover, inst, frames = r["mover"], r["inst"], r["frames"]
            why = f"reach: {inst} drawn over {mover} on {frames[0]}-{frames[-1]} (moves at {r['at']})"
            if d in HP and mover in ("drive_cage.01", "psu.01"):
                add(d, "cage", mover, inst, why, frames)
            elif d in HP and mover.startswith("cover.") and d in (23, 27, 28):
                add(d, "hp_shroud", mover, inst, why, frames)
            elif d in ITEM29_DESKTOPS and not mover.startswith("cover."):
                add(d, "item29", mover, inst, why, frames)
    for it in items:
        d, code, sev, detail = it["desktop"], it["code"], it["severity"], it["detail"]
        occ = (it.get("occluder") or "").split(" ")[0]
        why = f"{code}: {it['inst']} {it['frames']} -- {detail[:90]}"
        frames = _frames_of(it["frames"])
        if code == "c_below" and "under the closed screw cover" in detail:
            add(d, "screw_cover", it["inst"], occ, why, frames)
        elif code == "c_below" and "covers the cpu_cooler" in detail and d in HP:
            add(d, "hp_shroud", it["inst"], occ, why, frames)
        elif (code == "a_empty" and sev == "high" and d in DELL
              and (it["inst"].startswith("screw.cpu_cooler.")
                   or it["inst"].startswith("cpu_cooler.fan"))
              and frames[0] == 1
              and (occ.startswith("cover.") or occ.startswith("cooler_latch."))):
            add(d, "dell_shroud", it["inst"], occ, why, frames)
    for g in groups.values():
        for extra in EXTRA_MOVERS.get((g.desktop, g.kind), ()):
            if extra not in g.movers:
                g.movers.append(extra)
                g.why.append(f"U31: {extra} added by eye (see EXTRA_MOVERS)")
    return sorted(groups.values(), key=lambda g: (g.desktop, KINDS.index(g.kind), g.movers))


# --------------------------------------------------------------------------- #
# orders
# --------------------------------------------------------------------------- #
Entry = tuple[str, str]


def move_above(order: list[Entry], instance: str, above_of: str) -> list[Entry]:
    """Exactly :func:`tda.ui.session_edit.set_zorder_move`'s rewrite of the order."""
    moved = [e for e in order if e[0] == instance] or [(instance, MAIN)]
    rest = [e for e in order if e[0] != instance]
    if not any(e[0] == above_of for e in rest):
        rest = rest + [(above_of, MAIN)]
    cut = max(i for i, e in enumerate(rest) if e[0] == above_of)
    return rest[: cut + 1] + moved + rest[cut + 1:]


def positions(order: list[Entry]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(order):
        out[e[0]].append(i)
    return out


def above(pos: dict[str, list[int]], a: str, b: str) -> Optional[bool]:
    """Is every layer of ``a`` above every layer of ``b``? ``None`` when they interleave."""
    if a not in pos or b not in pos:
        return None
    if min(pos[a]) > max(pos[b]):
        return True
    if max(pos[a]) < min(pos[b]):
        return False
    return None


def gestures(order: list[Entry], group: Group, strategy: str,
             keep: Iterable[str] = ()) -> list[tuple[str, str]]:
    """The ``(instance, above_of)`` moves that put the movers above the targets.

    ``raise_keep`` is ``raise`` followed by lifting ``keep`` -- parts the picture says
    lie on top of a mover -- back above the highest mover, in their own order.
    """
    moves: list[tuple[str, str]] = []
    cur = list(order)
    pos = positions(cur)
    present_t = [t for t in group.targets if t in pos]
    present_m = [m for m in group.movers if m in pos]
    if not present_t or not present_m:
        return moves
    if strategy in ("raise", "raise_keep"):
        top_t = max(present_t, key=lambda t: max(pos[t]))
        # highest mover first, each directly above the top target: relative order kept
        for m in sorted(present_m, key=lambda m: -max(pos[m])):
            if min(positions(cur)[m]) > max(positions(cur)[top_t]):
                continue
            moves.append((m, top_t))
            cur = move_above(cur, m, top_t)
        # a mover left above the top target but below a moved one keeps its place
        if strategy == "raise_keep":
            p = positions(cur)
            anchor = max(present_m, key=lambda m: max(p[m]))
            for x in sorted((k for k in keep if k in pos), key=lambda k: min(pos[k])):
                if min(positions(cur)[x]) > max(positions(cur)[anchor]):
                    anchor = x
                    continue
                moves.append((x, anchor))
                cur = move_above(cur, x, anchor)
                anchor = x
    elif strategy == "lower":
        low_m = min(present_m, key=lambda m: min(pos[m]))
        anchor = None
        for t in sorted(present_t, key=lambda t: min(pos[t])):
            p = positions(cur)
            if max(p[t]) < min(p[low_m]):
                continue
            if anchor is None:
                lm = min(positions(cur)[low_m])
                anchor = cur[lm - 1][0]
            moves.append((t, anchor))
            cur = move_above(cur, t, anchor)
            anchor = t
    else:
        raise ValueError(strategy)
    return moves


def apply_gestures(order: list[Entry], moves: list[tuple[str, str]]) -> list[Entry]:
    cur = list(order)
    for inst, anchor in moves:
        cur = move_above(cur, inst, anchor)
    return cur


def new_floor_inversions(old: list[Entry], new: list[Entry], cls_of: dict) -> dict:
    """Floor layers that sit above a part in ``new`` and did not in ``old``."""
    before = floor_inversions(old, cls_of)
    after = floor_inversions(new, cls_of)
    out = {}
    for floor, under in after.items():
        added = [u for u in under if u not in before.get(floor, [])]
        if added:
            out[floor] = added
    return out


# --------------------------------------------------------------------------- #
# compiling
# --------------------------------------------------------------------------- #
class Desk:
    """One desktop's scan view: frames, segment, order, compiled overlaps."""

    def __init__(self, db, tax, truth: TruthService, d: int) -> None:
        self.db, self.tax, self.truth, self.d = db, tax, truth, d
        self.cache = InputCache()
        self.steps = sorted(int(r["step"]) for r in db.frames_for(d, VIEW)
                            if not r.get("missing"))
        self.seg_of = {s: pose_segment_of(db, FrameKey(d, s, VIEW), self.cache)
                       for s in self.steps}
        self.cls_of = {k: r.cls for k, r in machine_only(db.instances(d)).items()}

    def segments(self) -> list[int]:
        return sorted(set(self.seg_of.values()))

    def order(self, seg: int) -> list[Entry]:
        return [tuple(e) for e in self.db.zorder(self.d, VIEW, seg).order]

    def verified(self) -> list[int]:
        return [int(r[0]) for r in self.db.conn.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            "ORDER BY step", (self.d, VIEW))]

    def inputs(self, step: int):
        return gather(self.db, self.tax, FrameKey(self.d, step, VIEW), self.cache,
                      self.truth.cache_dir)

    def compile(self, step: int, order: Optional[list[Entry]] = None):
        inputs = self.inputs(step)
        if order is not None:
            z = inputs.zorder
            inputs = dataclasses.replace(
                inputs, zorder=ZOrderRec(z.desktop, z.view, z.pose_segment,
                                         [tuple(e) for e in order], version=z.version))
        return self.truth._compile_inputs(FrameKey(self.d, step, VIEW), inputs)


# --------------------------------------------------------------------------- #
# what an order does to the pixels (compiled, nothing written)
# --------------------------------------------------------------------------- #
def simulate(desk: Desk, seg: int, old: list[Entry], new: list[Entry]) -> dict[str, dict]:
    """``inst -> {step: (px before, px after, lost, gained, label before, label after)}``.

    Every annotatable frame of the segment is compiled with both orders; only the
    (frame, instance) pairs whose visible mask or label differs are returned.
    """
    out: dict[str, dict] = defaultdict(dict)
    for step in desk.steps:
        if desk.seg_of[step] != seg:
            continue
        before = desk.compile(step, old)
        after = desk.compile(step, new)
        for k, b in before.instances.items():
            a = after.instances.get(k)
            if a is None or b.visible is None or a.visible is None:
                continue
            lost = int(np.count_nonzero(b.visible & ~a.visible))
            gained = int(np.count_nonzero(a.visible & ~b.visible))
            if lost or gained or b.visibility != a.visibility:
                out[k][step] = (int(b.visible.sum()), int(a.visible.sum()), lost, gained,
                                b.visibility, a.visibility)
    return dict(out)


def _qa_common():
    qa = str(ROOT / "scripts" / "qa")
    if qa not in sys.path:
        sys.path.insert(0, qa)
    import u30_common  # noqa: E402  (U30-QA's illumination-corrected change map)
    return u30_common


def evidence(desk: Desk, seg: int, old: list[Entry], new: list[Entry], inst: str,
             frames: dict[int, tuple]) -> dict:
    """Does the picture say the part that now covers ``inst`` really was on top of it?

    ``inst`` loses pixels on ``frames`` under the new order.  On the last of them
    (``f``) the pixels it loses are ``L``; on the next frame (``g``) whatever covered
    them has gone -- ``inst`` is visible there again under the new order, same shape.
    If the picture changed in ``L`` between ``f`` and ``g`` (U30-QA's illumination-
    corrected change map), the cover was there: ``ev`` is the changed fraction.
    """
    if any(v[3] for v in frames.values()):
        return {"ev": None, "why": "it gains pixels somewhere"}
    lost_on = sorted(s for s, v in frames.items() if v[2])
    if not lost_on:
        return {"ev": None, "why": "label change only"}
    f = lost_on[-1]
    later = [s for s in desk.steps if s > f and desk.seg_of[s] == seg]
    if not later:
        return {"ev": None, "why": f"nothing after frame {f}"}
    g = later[0]
    b, a = desk.compile(f, old).instances.get(inst), desk.compile(f, new).instances.get(inst)
    nxt = desk.compile(g, new).instances.get(inst)
    if b is None or a is None or nxt is None or nxt.visible is None:
        return {"ev": None, "why": f"not on frame {g}", "f": f, "g": g}
    if nxt.keyframe_id != a.keyframe_id:
        return {"ev": None, "why": f"another shape on frame {g}", "f": f, "g": g}
    lost = b.visible & ~a.visible
    back = int(np.count_nonzero(nxt.visible & lost))
    if back < 0.8 * int(lost.sum()):
        return {"ev": None, "why": f"still covered on frame {g} ({back}/{int(lost.sum())} px back)",
                "f": f, "g": g}
    q = _qa_common()
    ia = frame_image(desk.db, desk.d, f, desk.truth.cache_dir)
    ib = frame_image(desk.db, desk.d, g, desk.truth.cache_dir)
    if ia is None or ib is None:
        return {"ev": None, "why": "no cached picture", "f": f, "g": g}
    rows = desk.db.pose_segments(desk.d, VIEW)
    roi = next((r.get("roi") for r in rows if int(r.get("seg", 0)) == seg), None)
    changed, valid = q.change_maps(ia, ib, roi)
    # QA's map has no evidence where either frame is blown out; here a pixel blown out
    # in one frame only (a white drive label that went away) has changed, and only
    # one blown out in both is without evidence
    k5 = np.ones((5, 5), np.uint8)
    sa = cv2.dilate((ia >= q.SATURATED).astype(np.uint8), k5).astype(bool)
    sb = cv2.dilate((ib >= q.SATURATED).astype(np.uint8), k5).astype(bool)
    changed = changed | (sa ^ sb)
    valid = valid | (sa ^ sb)
    ev = q.frac(changed, valid, lost)
    return {"ev": None if ev is None else round(float(ev), 3), "f": f, "g": g,
            "px": int(lost.sum()), "why": "" if ev is not None else "blown out / no evidence"}


# --------------------------------------------------------------------------- #
# planning one desktop
# --------------------------------------------------------------------------- #
#: a part the order would newly cover is taken in when the picture changed on at
#: least this fraction of the pixels it loses, the moment the cover went away
EVIDENCE_MIN = 0.5
#: a part outside the group whose visible mask changes by at most this many pixels on
#: every frame (and keeps its label) is within the app's conflict floor
NEGLIGIBLE_PX = 20


def _felt(x: tuple) -> bool:
    """Does one (frame, part) change reach a frozen row? (pixels over the floor or label)"""
    return x[2] + x[3] > NEGLIGIBLE_PX or x[4] != x[5]


def _candidate(desk: Desk, seg: int, cur: list[Entry], g: Group, targets: list[str],
               strategy: str, keep: Iterable[str] = (), allowed: Iterable[str] = ()) -> dict:
    gg = dataclasses.replace(g, targets=list(targets))
    mv = gestures(cur, gg, strategy, keep)
    new = apply_gestures(cur, mv)
    sim = simulate(desk, seg, cur, new) if mv else {}
    movers, tset = set(g.movers), set(targets)
    others = {k: v for k, v in sim.items() if k not in movers and k not in tset}
    # under the app's own conflict floor (tda.core.masks.is_conflict: 20 px) on every
    # frame and no label change: a frozen row would not even disagree
    negligible = {k: v for k, v in others.items() if not any(_felt(x) for x in v.values())}
    collateral = {k: v for k, v in others.items() if k not in negligible}
    # a target may only lose pixels, a mover only gain them
    wrong = sorted(k for k, v in sim.items()
                   if (k in tset and any(x[3] > NEGLIGIBLE_PX for x in v.values()))
                   or (k in movers and any(x[2] > NEGLIGIBLE_PX for x in v.values())))
    # past QA's last frame the cover has opened / moved / gone: a change there is a
    # different physical situation the one order cannot also be right for
    beyond = {k: sorted(s for s, x in v.items() if s > g.qa_last and _felt(x))
              for k, v in sim.items() if (k in movers or k in tset) and k not in set(allowed)}
    beyond = {k: v for k, v in beyond.items() if v}
    return {"strategy": strategy, "moves": mv, "order": new, "sim": sim,
            "collateral": collateral, "negligible": negligible, "wrong": wrong,
            "beyond": beyond, "targets": list(targets), "keep": list(keep),
            "floors": new_floor_inversions(cur, new, desk.cls_of)}


STRATEGY_RANK = {"raise": 0, "raise_keep": 1, "lower": 2}


def plan_desktop(desk: Desk, groups: list[Group], log, verbose: bool = False,
                 additions: bool = True) -> dict:
    """Choose the gestures of every group on one desktop; nothing is written.

    Per group: **raise** and **lower** are compiled over the whole segment.  The one
    that changes no part outside (movers, targets) wins, raise first.  When both
    would, every part the raise newly covers is looked at in the picture
    (:func:`evidence`): one the cover demonstrably lay on is added to the targets
    and the two are tried again; a part with no such evidence is kept where it is,
    and if no gesture sequence can do that the group is refused.
    """
    d = desk.d
    segs = desk.segments()
    if len(segs) != 1:
        log(f"D{d:02d}: {len(segs)} pose segments on the scan view -- refused (planned for one)")
        return {"refused": "several pose segments", "moves": [], "groups": []}
    seg = segs[0]
    order0 = desk.order(seg)
    pos0 = positions(order0)
    cur = list(order0)
    moves: list[tuple[str, str, str]] = []
    out_groups = []
    for g in groups:
        missing = [x for x in g.movers + g.targets if x not in pos0]
        if missing:
            log(f"  D{d:02d} {g.kind}: not in the order {missing} -- skipped")
            out_groups.append({"group": g, "skipped": f"missing {missing}"})
            continue
        bad = [m for m in g.movers if desk.cls_of.get(m) in FLOOR_CLASSES]
        if bad:
            log(f"  D{d:02d} {g.kind}: refused, a floor layer would be raised: {bad}")
            out_groups.append({"group": g, "skipped": f"floor mover {bad}"})
            continue
        targets = list(g.targets)
        added: dict[str, dict] = {}
        rejected: dict[str, dict] = {}
        dropped: dict[str, list[int]] = {}
        beyond_ok: dict[str, dict] = {}
        for extra in EXTRA_TARGETS.get((d, g.kind), ()):
            if extra in pos0 and extra not in targets:
                targets.append(extra)
                added[extra] = {"ev": None, "why": "by eye (EXTRA_TARGETS)"}
        chosen = None
        why_not = ""
        for _round in range(12):
            if not targets:
                why_not = "no target left"
                break
            # first: every part the raise would newly cover is looked at in the picture
            raise_c = _candidate(desk, seg, cur, g, targets, "raise")
            unjudged = {k: v for k, v in raise_c["collateral"].items()
                        if k not in rejected and k not in added}
            if unjudged:
                for k, v in unjudged.items():
                    ev = (evidence(desk, seg, cur, raise_c["order"], k, v) if additions
                          else {"ev": None, "why": "additions off (--no-additions)"})
                    if ev.get("ev") is not None and ev["ev"] >= EVIDENCE_MIN:
                        added[k] = ev
                        targets.append(k)
                    else:
                        rejected[k] = ev
                continue
            strategies = [("raise", ()), ("lower", ())]
            if rejected:
                strategies.insert(1, ("raise_keep", sorted(rejected)))
            cands = [_candidate(desk, seg, cur, g, targets, s, keep, beyond_ok)
                     for s, keep in strategies]
            for c in cands:
                if verbose:
                    log(f"    [{g.kind} {c['strategy']}] moves {c['moves']} collateral "
                        f"{ {k: (len(v), max(x[2] + x[3] for x in v.values())) for k, v in c['collateral'].items()} }"
                        f" wrong {c['wrong']} beyond {c['beyond']} floors {c['floors']}")
            ok = [c for c in cands if not c["collateral"] and not c["wrong"]
                  and not c["floors"] and not c["beyond"]]
            if ok:
                # the same pixels either way: the gesture the QA named (raise) first
                chosen = min(ok, key=lambda c: (STRATEGY_RANK[c["strategy"]], len(c["moves"])))
                break
            # a target every sequence changes past QA's last frame is out of reach
            # of one order: dropped from the group (reported)
            moved = [c for c in cands if c["moves"]]
            out_of_reach = set(targets)
            for c in moved:
                out_of_reach &= set(c["beyond"])
            if moved and out_of_reach:
                for t in sorted(out_of_reach):
                    late = {s: moved[0]["sim"][t][s] for s in moved[0]["beyond"][t]}
                    ev = evidence(desk, seg, cur, moved[0]["order"], t, late)
                    if ev.get("ev") is not None and ev["ev"] >= EVIDENCE_MIN:
                        # the cover still lies on it there too: the picture says so
                        beyond_ok[t] = dict(ev, frames=sorted(late))
                        continue
                    dropped[t] = moved[0]["beyond"][t]
                    targets.remove(t)
                    added.pop(t, None)
                continue
            if any(m in c["beyond"] for c in moved for m in g.movers) and \
                    all(c["beyond"] for c in moved):
                why_not = f"the movers change past frame {g.qa_last}: {moved[0]['beyond']}"
                break
            why_not = "every gesture sequence also changes a part without picture evidence"
            break
        if chosen is not None and not set(g.targets) & set(chosen["targets"]):
            # QA's own pairs are all out of reach: the parts taken in on the way are not
            # a reason to touch the order on their own
            chosen, why_not = None, "every QA target dropped"
        if chosen is None:
            log(f"  D{d:02d} {g.kind}: REFUSED ({why_not}) -- kept: "
                + "; ".join(f"{k}: {v.get('why') or 'evidence %s' % v.get('ev')}"
                            f"{' (%s->%s)' % (v['f'], v['g']) if 'f' in v else ''}"
                            for k, v in sorted(rejected.items()))
                + (f"; out of reach of one order (changes past frame {g.qa_last}): "
                   + ", ".join(f"{k} {_span(v)}" for k, v in sorted(dropped.items()))
                   if dropped else ""))
            out_groups.append({"group": g, "skipped": why_not or "refused",
                               "rejected": rejected, "added": added, "dropped": dropped})
            continue
        chosen["added"], chosen["rejected"], chosen["dropped"] = added, rejected, dropped
        chosen["beyond_ok"] = beyond_ok
        cur = chosen["order"]
        moves += [(inst, anchor, g.kind) for inst, anchor in chosen["moves"]]
        out_groups.append({"group": g, "cand": chosen})
    return {"seg": seg, "order0": order0, "order1": cur, "moves": moves, "groups": out_groups}


def _span(steps) -> str:
    st = sorted(steps)
    if not st:
        return "-"
    return f"{st[0]}" if len(st) == 1 else f"{st[0]}-{st[-1]}"


def describe_plan(d: int, plan: dict, log) -> None:
    for item in plan["groups"]:
        g = item["group"]
        if "skipped" in item:
            log(f"  D{d:02d} {g.kind:11s} {'+'.join(g.movers)} above {', '.join(g.targets)}: "
                f"SKIPPED ({item['skipped']})")
            continue
        c = item["cand"]
        log(f"  D{d:02d} {g.kind:11s} {'+'.join(g.movers)} above {', '.join(c['targets'])}: "
            f"{c['strategy']} -- {len(c['moves'])} move(s): "
            f"{'; '.join(f'{a} directly above {b}' for a, b in c['moves']) or '(already so)'}")
        for k, ev in sorted(c.get("added", {}).items()):
            if ev.get("ev") is None:
                log(f"      + {k}: not in the QA list, taken in -- {ev.get('why')}")
                continue
            log(f"      + {k}: not in the QA list, taken in -- the picture changed on "
                f"{ev['ev']:.0%} of its {ev['px']} lost px when the cover went ({ev['f']}->{ev['g']})")
        for k, ev in sorted(c.get("beyond_ok", {}).items()):
            log(f"      > {k}: also covered on {_span(ev['frames'])} (after QA's frames) -- the "
                f"picture changed on {ev['ev']:.0%} of it when the cover went ({ev['f']}->{ev['g']})")
        for k, v in sorted(c.get("dropped", {}).items()):
            log(f"      - {k}: DROPPED -- any order that covers it on 1-{g.qa_last} also "
                f"covers it on {_span(v)}, after the cover moved (left for the user)")
        if c.get("keep"):
            log(f"      ^ kept above the movers (the picture says they lie on top): "
                f"{', '.join(c['keep'])}")
        for k, v in sorted(c.get("negligible", {}).items()):
            log(f"      ~ {k}: changes by at most "
                f"{max(x[2] + x[3] for x in v.values())} px on {len(v)} frame(s) {_span(v)} "
                f"(under the 20 px conflict floor)")
        for k, ev in sorted(c.get("rejected", {}).items()):
            log(f"      = {k}: kept where it was ({ev.get('why') or 'evidence %s' % ev.get('ev')})")
        for k, v in sorted(c["sim"].items()):
            lost = sum(x[2] for x in v.values())
            gained = sum(x[3] for x in v.values())
            labels = sorted({f"{x[4]}->{x[5]}" for x in v.values() if x[4] != x[5]})
            log(f"      {k}: frames {_span(v)} ({len(v)}), -{lost} / +{gained} px"
                f"{' labels ' + ', '.join(labels) if labels else ''}")


# --------------------------------------------------------------------------- #
# B. the HP cable clip: is it still there after the shroud leaves?
# --------------------------------------------------------------------------- #
def frame_image(db, d: int, step: int, cache_dir: str, gray: bool = True) -> Optional[np.ndarray]:
    """The cached scanner frame the annotator saw (read only)."""
    row = db.get_frame(FrameKey(d, step, VIEW)) or {}
    path = (row.get("aux") or {}).get("cache_path") or os.path.join(
        cache_dir, VIEW, f"D{d:02d}", f"s{step:03d}.png")
    if not os.path.isfile(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)


def stored_visible(db, d: int, step: int, inst: str) -> Optional[np.ndarray]:
    row = db.compiled(FrameKey(d, step, VIEW)).get(inst)
    if not row or not row.get("visible_rle"):
        return None
    m = masks.decode_rle(row["visible_rle"]).astype(bool)
    return m if m.any() else None


def masked_ncc(ref: np.ndarray, img: np.ndarray, mask: np.ndarray,
               search: int = CLIP_SEARCH) -> tuple[float, tuple[int, int], float]:
    """Best NCC of ``ref``'s pixels under ``mask`` against ``img`` shifted by <= ``search``.

    Returns ``(best ncc, (dx, dy) of the best, ncc at zero shift)``.
    """
    ys, xs = np.nonzero(mask)
    t_all = ref[ys, xs].astype(np.float64)
    h, w = img.shape
    best, at, zero = -1.0, (0, 0), -1.0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            if ok.mean() < 0.95:
                continue
            c = img[yy[ok], xx[ok]].astype(np.float64)
            t = t_all[ok] - t_all[ok].mean()
            c = c - c.mean()
            v = float((t * c).sum() / (np.sqrt((t * t).sum() * (c * c).sum()) + 1e-9))
            if dx == 0 and dy == 0:
                zero = v
            if v > best:
                best, at = v, (dx, dy)
    return best, at, zero


def leave_step(db, tax, d: int, inst: str, steps: list[int]) -> Optional[int]:
    """The first step at which ``inst`` is no longer in the chassis."""
    for st in steps:
        held = state_of(db, tax, d, st).get(inst)
        if held is None or held.state == "removed":
            return st
    return None


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    pad = 18 * len(lines) + 6
    out = np.full((img.shape[0] + pad, img.shape[1], 3), 24, np.uint8)
    out[:img.shape[0]] = img
    for i, text in enumerate(lines):
        cv2.putText(out, text, (4, img.shape[0] + 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return out


def _outline(img: np.ndarray, mask: Optional[np.ndarray], color, thick: int = 1) -> None:
    if mask is None or not mask.any():
        return
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, thick)


def _crop_box(mask: np.ndarray, hw: tuple[int, int], margin: int, side: int):
    ys, xs = np.nonzero(mask)
    cx, cy = (int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2
    half = max(side // 2, (int(xs.max()) - int(xs.min())) // 2 + margin,
               (int(ys.max()) - int(ys.min())) // 2 + margin)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(hw[1], cx + half), min(hw[0], cy + half)
    return x0, y0, x1, y1


def _grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1] + 4,
                                  cv2.BORDER_CONSTANT, value=(12, 12, 12))
               for t in tiles[i:i + cols]]
        while len(row) < cols:
            row.append(np.full((h, w + 4, 3), 12, np.uint8))
        rows.append(cv2.copyMakeBorder(np.hstack(row), 0, 4, 0, 0, cv2.BORDER_CONSTANT,
                                       value=(12, 12, 12)))
    return np.vstack(rows)


def render_clip(db, d: int, m: dict, cache_dir: str, path: str) -> None:
    """The clip's frame-2 mask over frames 2, 3.. (its NCC under each)."""
    if "rows" not in m or not m["rows"]:
        return
    mask = stored_visible(db, d, m["before"], CLIP)
    show = [m["before"]] + [r["step"] for r in m["rows"]][:6]
    for extra in (m.get("gone_from"), m["rows"][-1]["step"]):
        if extra is not None and extra not in show:
            show.append(extra)
    by = {r["step"]: r for r in m["rows"]}
    tiles = []
    for st in show:
        img = frame_image(db, d, st, cache_dir, gray=False)
        if img is None:
            continue
        x0, y0, x1, y1 = _crop_box(mask, img.shape[:2], 70, 240)
        crop = img[y0:y1, x0:x1].copy()
        sub = mask[y0:y1, x0:x1]
        _outline(crop, sub, (0, 255, 0) if st == m["before"] else (0, 220, 255), 1)
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
        r = by.get(st)
        text = [f"D{d:02d} s{st}" + (" (ref)" if st == m["before"] else "")]
        if r:
            chg = "-" if r["chg"] is None else f"{r['chg']:.0%}"
            text.append(f"changed {chg}  NCC {r['ncc']:.2f} {tuple(r['shift'])}  "
                        f"ctx {r['ctx']:.2f}  {'IN PLACE' if r['present'] else 'gone'}")
        tiles.append(_label(crop, text))
    if tiles:
        cv2.imwrite(path, _grid(tiles, 4))


def measure_clip(desk: "Desk", cache_dir: str) -> dict:
    """Is ``cable_clip.01`` still in its place on the frames after it leaves? (nothing written)

    Reference: its stored visible mask on the last frame before it leaves (frame 2,
    the shroud leaves at 3).  On every later frame of the segment:

    * ``ncc``  -- the masked NCC the controller asked for: the reference pixels under
      the mask against the frame, best over +-6 px (``shift``);
    * ``ctx``  -- the same over the mask grown 12 px, minus the pixels of the parts that
      leave with it (the shroud): the clip's edges carry the texture its flat black
      plastic does not have;
    * ``chg``  -- U30-QA's illumination-corrected change map (the one behind QA's
      "pixel change 0-13 %"): the fraction of the mask that changed.

    ``present`` = ``chg`` <= :data:`CLIP_CHANGE_MAX` or (``ncc`` >= 0.8 at <= 3 px).
    The flat black clip on a black board gives NCC 0.1-0.6 even where it plainly is
    still there (patch std 4-14 grey levels against ~3 of sensor noise), so NCC alone
    cannot decide; it is reported next to the change fraction.
    """
    db, tax, d = desk.db, desk.tax, desk.d
    out: dict = {"desktop": d}
    inst = db.instances(d)
    if CLIP not in inst:
        out["skip"] = "no cable_clip.01"
        return out
    rec = inst[CLIP]
    out["parent"], out["attached"] = rec.parent, rec.attached
    steps = desk.steps
    out["verified_frames"] = len(desk.verified())
    out["last_step"] = steps[-1] if steps else None
    left = leave_step(db, tax, d, CLIP, steps)
    out["leaves_at"] = left
    if not out["verified_frames"]:
        out["skip"] = "scan view not annotated (no verified frame)"
        return out
    if left is None:
        out["skip"] = "the clip never leaves"
        return out
    before = max(s for s in steps if s < left)
    mask = stored_visible(db, d, before, CLIP)
    if mask is None:
        out["skip"] = f"no stored visible mask of the clip on frame {before}"
        return out
    ref = frame_image(db, d, before, cache_dir)
    seg = desk.seg_of[before]
    held = state_of(db, tax, d, left)
    going = np.zeros_like(mask)
    for k, ci in desk.compile(before).instances.items():
        if k != CLIP and ci.amodal is not None and (
                k not in held or held[k].state == "removed"):
            going |= ci.amodal
    ctx = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~going
    rows_ = db.pose_segments(d, VIEW)
    roi = next((r.get("roi") for r in rows_ if int(r.get("seg", 0)) == seg), None)
    q = _qa_common()
    rows = []
    for st in steps:
        if st < left or desk.seg_of[st] != seg:
            continue
        img = frame_image(db, d, st, cache_dir)
        if img is None or ref is None:
            continue
        best, at, zero = masked_ncc(ref, img, mask)
        cbest, cat, _c0 = masked_ncc(ref, img, ctx)
        changed, valid = q.change_maps(ref, img, roi)
        chg = q.frac(changed, valid, mask)
        ncc_ok = best >= CLIP_NCC and max(abs(at[0]), abs(at[1])) <= CLIP_SHIFT_OK
        present = (chg is not None and chg <= CLIP_CHANGE_MAX) or ncc_ok
        rows.append({"step": st, "ncc": round(best, 3), "shift": list(at), "ncc0": round(zero, 3),
                     "ctx": round(cbest, 3), "ctx_shift": list(cat),
                     "chg": None if chg is None else round(float(chg), 3),
                     "present": bool(present)})
    out.update(before=before, mask_px=int(mask.sum()), rows=rows,
               patch_std=round(float(ref[mask].std()), 1) if ref is not None else None)
    out["ncc_3_8"] = [r["ncc"] for r in rows if r["step"] <= left + 5]
    out["chg_3_8"] = [r["chg"] for r in rows if r["step"] <= left + 5]
    # the first frame from which it is gone for good (three frames running)
    gone = None
    for i, r in enumerate(rows):
        if not r["present"] and all(not x["present"] for x in rows[i:i + 3]):
            gone = r["step"]
            break
    out["gone_from"] = gone
    run = [r for r in rows if gone is None or r["step"] < gone]
    out["present_until_gone"] = bool(run) and all(r["present"] for r in run)
    if gone is not None:
        prev = max(s for s in steps if s < gone)
        held0, held1 = state_of(db, tax, d, prev), state_of(db, tax, d, gone)
        out["leaving_there"] = sorted(
            k for k, v in held0.items()
            if v.state != "removed" and (k not in held1 or held1[k].state == "removed")
            and k != CLIP)
    return out


def clip_ruling(m: dict) -> dict:
    """What to do with one desktop's clip, from its measurement."""
    d = m["desktop"]
    if "skip" in m:
        return {"action": "none", "why": m["skip"]}
    gone, left = m.get("gone_from"), m.get("leaves_at")
    if d in CLIP_KEEP_RULING:
        return {"action": "none", "why": "the user's ruling: it leaves with the shroud "
                f"(measured: change {m['chg_3_8']}, NCC {m['ncc_3_8']}, gone for good from "
                f"{gone})"}
    if gone == left:
        early = [r["step"] for r in m.get("rows", []) if r["present"] and r["step"] <= left + 5]
        if early:
            return {"action": "none", "why": f"ambiguous: not in place on {left}, but in place "
                    f"again on {early} (change {m['chg_3_8']}) -- left as it is (parent "
                    f"{m.get('parent')}), for the user"}
        return {"action": "none", "why": f"gone at {left}: it does leave with {m.get('parent')} "
                "(nothing to change)"}
    if not m["present_until_gone"]:
        return {"action": "none", "why": "ambiguous: present on some frames after "
                f"{left - 1}, not on others (change {m['chg_3_8']}) -- left as it is"}
    if gone is None:
        return {"action": "unparent", "parent": None,
                "why": f"in its place on every frame {left}-{m['last_step']}"}
    leaving = m.get("leaving_there") or []
    carriers = [k for k in leaving if not k.split(".")[0] in ("screw", "connector")]
    if len(carriers) == 1:
        return {"action": "reparent", "parent": carriers[0],
                "why": f"in its place on {left}-{gone - 1}, gone from {gone}, where "
                       f"{carriers[0]} leaves"}
    return {"action": "none", "why": f"in its place on {left}-{gone - 1}, gone from {gone} but "
            f"no single part leaves there ({leaving}) -- left as it is"}


# --------------------------------------------------------------------------- #
# C. applying, through the app's own paths
# --------------------------------------------------------------------------- #
def _gesture_frame(desk: Desk, inst: str, anchor: str) -> tuple[int, Optional[set]]:
    """A frame of the segment where both are parts of the frame, and what it has."""
    for step in desk.steps:
        have = set(desk.compile(step).instances)
        if inst in have and anchor in have:
            return step, have
    return desk.steps[0], None


def apply_zorder(db, truth, desk: Desk, plan: dict, log) -> int:
    """Each planned gesture as ``Ctrl+Up`` / ``L`` writes it: one ``set_zorder`` op."""
    from tda.ui import session_edit as edit

    n = 0
    for inst, anchor, kind in plan["moves"]:
        step, known = _gesture_frame(desk, inst, anchor)
        edit.set_zorder_move(db, truth, FrameKey(desk.d, step, VIEW), inst, anchor, ANNOTATOR,
                             known=known)
        n += 1
    got = desk.order(plan["seg"])
    if got != plan["order1"]:
        raise RuntimeError(f"D{desk.d:02d}: the stored order is not the planned one")
    if n:
        log(f"  D{desk.d:02d}: {n} set_zorder op(s) written; order v"
            f"{db.zorder(desk.d, VIEW, plan['seg']).version}")
    return n


def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def steps_panel_relation(db, tax, d: int, key: str, parent: Optional[str], attached: bool,
                         cache_dir: str) -> str:
    """Parent / Attached of one part in Steps' instance table, then Apply (U27 relation_fix)."""
    from PySide6.QtCore import Qt
    from tda.ui.panels.steptable import StepTablePanel

    app = _qt_app()
    panel = StepTablePanel(db, d, taxonomy=tax, cache_dir=cache_dir, annotator=ANNOTATOR)
    try:
        model = panel.instances_model
        cols = [c.title for c in model.columns]
        row = model.keys.index(key)
        ok1 = model.setData(model.index(row, cols.index("Parent")), parent or "", Qt.EditRole)
        ok2 = model.setData(model.index(row, cols.index("Attached")),
                            Qt.Checked if attached else Qt.Unchecked, Qt.CheckStateRole)
        panel.apply()
        status = panel.status.text()
    finally:
        panel.deleteLater()
        app.processEvents()
    if not (ok1 and ok2) or not status.startswith("Saved"):
        raise RuntimeError(f"D{d:02d}: Steps refused {key} parent={parent}: {status}")
    return status


def clip_frames(db, tax, d: int, steps: list[int]) -> list[int]:
    out = []
    for st in steps:
        held = state_of(db, tax, d, st).get(CLIP)
        if held is not None and held.state != "removed" and held.placement == "in_chassis":
            out.append(st)
    return out


def fix_clip(db, tax, truth, desk: Desk, m: dict, ruling: dict, cache_dir: str, log) -> dict:
    """B: the clip's parent through Steps, then its frame-2 shape where it now stays.

    Only while the clip still hangs on the shroud (parent ``cover.*``): once a ruling --
    this script's or the user's -- has moved it, a re-run measures it again for the record
    and leaves it alone (``done["skipped"]``), so a re-measure that reads a later frame
    differently can never re-rule it.
    """
    from tda.ui import session_api as api
    from tda.ui import session_edit as edit
    from tda.ui.session_ops import REVERSE

    d = desk.d
    done: dict = {"relation": None, "shape": None}
    rec = db.instances(d).get(CLIP)
    if rec is None:
        done["skipped"] = f"no {CLIP} on this desktop"
        return done
    if not str(rec.parent or "").startswith(CLIP_SHROUD_PREFIX):
        done["skipped"] = (f"its parent is {rec.parent} (attached {bool(rec.attached)}), not "
                           f"the shroud: already ruled -- not re-ruled")
        log(f"  D{d:02d}: {CLIP} skipped: {done['skipped']}")
        return done
    if ruling["action"] not in ("reparent", "unparent"):
        return done
    parent = ruling.get("parent")
    attached = parent is not None
    if rec.parent != parent or bool(rec.attached) != attached:
        status = steps_panel_relation(db, tax, d, CLIP, parent, attached, cache_dir)
        done["relation"] = {"from": [rec.parent, rec.attached], "to": [parent, attached]}
        log(f"  D{d:02d}: {CLIP} parent {rec.parent} -> {parent}, attached {rec.attached} -> "
            f"{attached} (Steps: {status[:70]})")
    frames = clip_frames(db, tax, d, desk.steps)
    after = [s for s in frames if s > m["before"]]
    if not after:
        return done
    last = after[-1]
    seg = desk.seg_of[m["before"]]
    chain = [kf for kf in db.keyframes(d, VIEW, CLIP)
             if kf.pose_segment == seg and kf.placement == "in_chassis"]
    if select_keyframe(chain, last) is not None:
        log(f"  D{d:02d}: {CLIP} already has a shape on frame {last}")
        return done
    src = select_keyframe(chain, m["before"])
    mask = None
    for part in src.parts:
        if part.rle:
            pm = masks.decode_rle(part.rle).astype(bool)
            mask = pm if mask is None else (mask | pm)
    if mask is None or not mask.any():
        raise RuntimeError(f"D{d:02d}: the clip's frame-{m['before']} shape has no pixels")
    # The row says source 'manual' -- commit_edit stamps 'manual' or 'ls_adopted' and
    # nothing else, and writing the column past it would leave the op's redo payload
    # disagreeing with the row -- so where the pixels came from goes in the op's payload.
    r = edit.commit_edit(db, truth, FrameKey(d, last, VIEW), CLIP, mask, api.SCOPE_KEYFRAME,
                         REVERSE, ANNOTATOR,
                         extra={"u31": "controller U31-fix: the clip's own shape copied, "
                                       "unchanged, onto the frames it now stays on",
                                "copy_of_keyframe": int(src.id),
                                "copy_of_frame": int(m["before"])})
    done["shape"] = {"from_keyframe": src.id, "keyframe": r.get("keyframe_id"),
                     "anchor": r.get("anchor_step"), "frames": after, "px": int(mask.sum())}
    log(f"  D{d:02d}: {CLIP} shape of keyframe {src.id} (frame {m['before']}) committed on "
        f"frame {last}: keyframe {r.get('keyframe_id')} anchored at {r.get('anchor_step')}, "
        f"frames {after[0]}-{after[-1]}")
    return done


def clip_touches(desk: Desk, frames: list[int]) -> set[tuple[int, str]]:
    """(frame, part) whose amodal shape meets the clip's: where its pixels can move."""
    # the desk's input cache predates the relation fix and the new keyframe
    desk.cache = InputCache()
    out = set()
    for st in frames:
        fr = desk.compile(st)
        clip = fr.instances.get(CLIP)
        if clip is None or clip.amodal is None:
            continue
        out.add((st, CLIP))
        for k, ci in fr.instances.items():
            if k != CLIP and ci.amodal is not None and np.any(ci.amodal & clip.amodal):
                out.add((st, k))
    return out


def open_conflicts(db, d: int) -> dict[int, tuple[int, str]]:
    """The scan view's open conflicts: id -> (frame, part)."""
    return {int(c["id"]): (int(c["step"]), c["instance"])
            for c in db.conflicts(d, VIEW, open_only=True)}


def settle(db, truth, desk: Desk, before_open: dict[int, tuple[int, str]],
           predicted: set[tuple[int, str]], log) -> dict:
    """Refresh the scan view; accept every conflict the changes opened (accept_new).

    ``before_open`` is :func:`open_conflicts` read before the run touched the desktop.
    Those questions are the user's: every conflict on one of their (frame, part) keys is
    left open, whatever its id -- a change that moves such a row makes the app close the
    old one as superseded and open a new one (one open conflict per row, U7), and that new
    id must not be swallowed (U31 review N2).
    """
    from tda.core.truth import ACCEPT_NEW
    from tda.core.truth_resolve import StaleConflictError

    d = desk.d
    users = set(before_open.values())
    tot = truth.refresh_range(d, VIEW, desk.steps, ignore_digest=True)
    truth.run_pending_rechecks(d, VIEW)
    accepted, left_mine, unexpected = [], [], []
    for _round in range(3):
        new = [c for c in db.conflicts(d, VIEW, open_only=True)
               if int(c["id"]) not in before_open]
        todo = []
        for c in new:
            key = (int(c["step"]), c["instance"])
            if key in users:
                continue                        # the user's question, superseded: left open
            if key not in predicted:
                if c not in unexpected:
                    unexpected.append(c)
                continue
            todo.append(c)
        if not todo:
            break
        for c in todo:
            try:
                truth.resolve_conflict(int(c["id"]), ACCEPT_NEW, ANNOTATOR)
                accepted.append((int(c["step"]), c["instance"]))
            except StaleConflictError as exc:
                log(f"  D{d:02d}: conflict {c['id']} stale, re-queued: {exc}")
    still = open_conflicts(db, d)
    left_mine = sorted(k for i, k in still.items() if i not in before_open and k not in users)
    reopened = sorted(k for i, k in still.items() if i not in before_open and k in users)
    return {"accepted": accepted, "unexpected": [(int(c["step"]), c["instance"])
                                                 for c in unexpected],
            "left_mine": left_mine,
            "users_open": {"before": len(before_open),
                           "after": sum(1 for k in still.values() if k in users),
                           "reopened": reopened},
            "refresh": {k: tot.get(k) for k in ("updated", "conflicts", "standing", "closed",
                                                 "overtaken")}}


# --------------------------------------------------------------------------- #
# D. before / after sheets, from the stored rows of two databases
# --------------------------------------------------------------------------- #
def stored_rows(conn: sqlite3.Connection, d: int) -> dict[tuple[int, str], tuple]:
    out = {}
    for step, inst, rle, vis in conn.execute(
            "SELECT step, instance, visible_rle_json, visibility FROM compiled_mask "
            "WHERE desktop=? AND view=?", (d, VIEW)):
        out[(int(step), inst)] = (rle, vis)
    return out


def _decode(rle_json: Optional[str], hw) -> np.ndarray:
    if not rle_json:
        return np.zeros(hw, bool)
    return masks.decode_rle(json.loads(rle_json)).astype(bool)


def row_changes(before: sqlite3.Connection, after: sqlite3.Connection, d: int,
                hw=(1600, 1600)) -> list[dict]:
    """Every (frame, part) whose stored visible mask or label differs."""
    b, a = stored_rows(before, d), stored_rows(after, d)
    out = []
    for key in sorted(set(b) | set(a)):
        rb, ra = b.get(key, (None, None)), a.get(key, (None, None))
        if rb == ra:
            continue
        mb, ma = _decode(rb[0], hw), _decode(ra[0], hw)
        out.append({"step": key[0], "inst": key[1], "before": int(mb.sum()),
                    "after": int(ma.sum()), "xor": int(np.count_nonzero(mb ^ ma)),
                    "label_before": rb[1], "label_after": ra[1],
                    "new_row": key not in b, "row_gone": key not in a})
    return out


def render_before_after(db, before: sqlite3.Connection, after: sqlite3.Connection, d: int,
                        changes: list[dict], cache_dir: str, path: str, n: int = 6) -> None:
    """The worst ``n`` (frame, part): before | after, the stored visible mask outlined."""
    best: dict[str, dict] = {}
    for c in changes:
        if c["inst"] not in best or c["xor"] > best[c["inst"]]["xor"]:
            best[c["inst"]] = c
    worst = sorted(best.values(), key=lambda c: -c["xor"])[:n]
    if not worst:
        return
    rb, ra = stored_rows(before, d), stored_rows(after, d)
    tiles = []
    for c in worst:
        img = frame_image(db, d, c["step"], cache_dir, gray=False)
        if img is None:
            continue
        hw = img.shape[:2]
        mb = _decode(rb.get((c["step"], c["inst"]), (None,))[0], hw)
        ma = _decode(ra.get((c["step"], c["inst"]), (None,))[0], hw)
        both = mb | ma
        if not both.any():
            continue
        x0, y0, x1, y1 = _crop_box(both, hw, 40, 200)
        for tag, m_, lab, px in (("before", mb, c["label_before"], c["before"]),
                                 ("after", ma, c["label_after"], c["after"])):
            crop = img[y0:y1, x0:x1].copy()
            _outline(crop, m_[y0:y1, x0:x1], (0, 255, 0), 2)
            scale = max(1.0, 260.0 / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tiles.append(_label(crop, [f"D{d:02d} s{c['step']} {c['inst']} -- {tag}",
                                       f"{px} px  {lab}"]))
    if tiles:
        cv2.imwrite(path, _grid(tiles, 4))


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def snapshot(src: str, dst: str) -> None:
    """A copy through the SQLite backup API (the source opened read-only)."""
    if os.path.exists(dst):
        os.remove(dst)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    b = sqlite3.connect(dst)
    try:
        s.backup(b)
        ok = b.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        b.close()
        s.close()
    if ok != "ok":
        raise RuntimeError(f"snapshot {dst}: integrity {ok}")


def app_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "tda.cli" in cmd and " app" in cmd:
            return True
    return False


def _ops_since(db, d: int, after_id: int) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for (kind,) in db.conn.execute("SELECT kind FROM op_log WHERE id > ? AND desktop = ?",
                                   (after_id, d)):
        out[kind] += 1
    return dict(out)


def parse_desktops(text: Optional[str]) -> Optional[list[int]]:
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=LIVE)
    ap.add_argument("--apply", action="store_true", help="write (default: a dry run on a copy)")
    ap.add_argument("--desktops", default=None, help="e.g. 3,5,22-32 (D2 is always skipped)")
    ap.add_argument("--qa", default=QA_DIR, help="U30-QA's folder (all_items.json, reach.json)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--paths", default=str(ROOT / "configs" / "paths.yaml"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-additions", action="store_true",
                    help="change only the pairs the QA listed: a part the raise would also "
                         "cover is kept above it even where the picture says it is under it")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "before_after").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logf = open(out / f"u31_{'apply' if args.apply else 'dryrun'}_{stamp}.log", "w",
                encoding="utf-8")

    def log(text: str = "") -> None:
        print(text, flush=True)
        logf.write(text + "\n")
        logf.flush()

    src = os.path.abspath(args.db)
    is_live = os.path.normcase(src) == os.path.normcase(os.path.abspath(LIVE))
    if app_running():
        log("the annotator app is running: close it first -- nothing touched")
        return 2
    before_path = str(out / f"before_{stamp}.sqlite")
    snapshot(src, before_path)                       # the reference the sheets compare with
    if args.apply:
        target = src
    else:
        target = str(out / f"dryrun_{stamp}.sqlite")
        snapshot(src, target)
        log(f"DRY RUN: everything below is carried out on a copy, {target}")
    config = P.load_paths(args.paths)
    rawroot.configure(config)
    cache_dir = config.get("cache_dir")
    from tda.core.db_backup import acquire_lock_file, release_lock_file
    try:
        acquire_lock_file(target, ANNOTATOR)
    except RuntimeError as exc:
        log(f"locked: {exc} -- nothing touched")
        return 2
    # what a failure part-way through has to say (U31 review N3)
    run = {"backup": None, "writing": False, "desktop": None}
    try:
        db = P.open_db(dict(config, db_path=target), target)
        try:
            db.conn.execute("BEGIN IMMEDIATE")
            db.conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            log(f"DB busy ({exc}); is the annotator open? nothing touched")
            return 2
        if args.apply and is_live:
            from tda.cli_common import safety_backup
            said = io.StringIO()
            with contextlib.redirect_stdout(said):      # it prints the path; keep it in the log
                ok = safety_backup(config, db, "u31", "U31-fix")
            for line in said.getvalue().splitlines():
                log(line)
            if not ok:
                return 1
            run["backup"] = said.getvalue().strip().rsplit("-> ", 1)[-1].strip() or None
            log(f"SAFETY BACKUP: {run['backup']}")
            log("  on ANY error from here on: restore this backup over the live database and "
                "do not re-run on top of the half-done run")
        elif args.apply:
            log(f"no safety backup: {target} is not the live database")
        run["writing"] = True
        tax = load_taxonomy()
        truth = TruthService(db, tax, cache_dir=cache_dir)
        items, reach = load_qa(args.qa)
        wanted = parse_desktops(args.desktops)
        groups = derive_groups(items, reach, wanted)
        by_d: dict[int, list[Group]] = defaultdict(list)
        for g in groups:
            by_d[g.desktop].append(g)
        desks = sorted((set(by_d) | {d for d in CLIP_DESKTOPS
                                      if wanted is None or d in wanted}) - OFF_LIMITS)
        # The conflicts open before the run are the user's questions: their (frame, part)
        # keys, read here before anything changes, are left open whatever ids they carry
        # by the end (U31 review N2).
        users_open = {d: open_conflicts(db, d) for d in desks}
        summary: dict = {"target": target, "before": before_path, "apply": args.apply,
                         "stamp": stamp, "safety_backup": run["backup"],
                         "users_open_before": {d: len(v) for d, v in users_open.items() if v},
                         "desktops": {}}
        log("open conflicts before the run, left open: " + (", ".join(
            f"D{d:02d} {len(v)} ({', '.join(sorted({k for _s, k in v.values()}))})"
            for d, v in users_open.items() if v) or "none"))
        before_conn = sqlite3.connect(f"file:{before_path}?mode=ro", uri=True)
        for d in desks:
            run["desktop"] = d
            t0 = time.time()
            desk = Desk(db, tax, truth, d)
            if not desk.verified():
                log(f"D{d:02d}: no verified scan frame -- not touched")
                continue
            rec: dict = {"groups": [], "moves": [], "clip": None}
            ver0 = desk.verified()
            open0 = users_open[d]
            op0 = int(db.conn.execute("SELECT COALESCE(MAX(id), 0) FROM op_log").fetchone()[0])
            log(f"D{d:02d}: {len(desk.steps)} frames, {len(ver0)} verified, "
                f"{len(open0)} open conflict(s) of the user's")
            predicted: set[tuple[int, str]] = set()
            plan = None
            if by_d.get(d):
                plan = plan_desktop(desk, by_d[d], log, additions=not args.no_additions)
                describe_plan(d, plan, log)
                for item in plan.get("groups", []):
                    g = item["group"]
                    entry = {"kind": g.kind, "movers": g.movers, "targets": g.targets,
                             "why": g.why}
                    if "skipped" in item:
                        entry["skipped"] = item["skipped"]
                        entry["rejected"] = dict(item.get("rejected", {}))
                        entry["dropped"] = dict(item.get("dropped", {}))
                    else:
                        c = item["cand"]
                        entry.update(strategy=c["strategy"], moves=c["moves"],
                                     targets_final=c["targets"], added=c["added"],
                                     rejected=c["rejected"], dropped=c["dropped"],
                                     beyond_ok=c["beyond_ok"], keep=c["keep"],
                                     negligible={k: sorted(v) for k, v in c["negligible"].items()},
                                     changes={k: {str(s): list(x) for s, x in v.items()}
                                              for k, v in c["sim"].items()})
                        predicted |= {(s, k) for k, v in c["sim"].items() for s in v}
                    rec["groups"].append(entry)
                rec["moves"] = plan.get("moves", [])
                if plan.get("moves"):
                    apply_zorder(db, truth, desk, plan, log)
            if d in CLIP_DESKTOPS:
                m = measure_clip(desk, cache_dir)
                ruling = clip_ruling(m)
                rec["clip"] = {"measure": m, "ruling": ruling}
                if not args.no_render and "rows" in m:
                    render_clip(db, d, m, cache_dir, str(out / f"clip_D{d:02d}.jpg"))
                log(f"  D{d:02d} clip: {ruling['action']} -- {ruling['why']}")
                if m.get("rows"):
                    log("    " + " ".join(
                        f"{r['step']}:chg{'-' if r['chg'] is None else format(r['chg'], '.2f')}"
                        f"/ncc{r['ncc']:.2f}/ctx{r['ctx']:.2f}{'+' if r['present'] else '-'}"
                        for r in m["rows"][:10]))
                done = fix_clip(db, tax, truth, desk, m, ruling, cache_dir, log)
                rec["clip"]["done"] = done
                if done.get("shape"):
                    predicted |= clip_touches(desk, done["shape"]["frames"])
            st = settle(db, truth, desk, open0, predicted, log)
            ver1 = desk.verified()
            rec.update(settle=st, verified_before=len(ver0), verified_after=len(ver1),
                       demoted=sorted(set(ver0) - set(ver1)),
                       ops=_ops_since(db, d, op0),
                       open_before=len(open0),
                       open_after=len(db.conflicts(d, VIEW, open_only=True)))
            log(f"  D{d:02d}: ops {rec['ops']}; conflicts accepted {len(st['accepted'])}, "
                f"unexpected (left open) {st['unexpected']}, still open of mine "
                f"{st['left_mine']}; the user's {st['users_open']['before']} -> "
                f"{st['users_open']['after']} (superseded and reopened by the app: "
                f"{len(st['users_open']['reopened'])}); "
                f"open {rec['open_before']} -> {rec['open_after']}; "
                f"verified {len(ver0)} -> {len(ver1)}"
                f"{' demoted ' + str(rec['demoted']) if rec['demoted'] else ''} "
                f"({time.time() - t0:.0f} s)")
            if not args.no_render:
                after_conn = db.conn
                ch = row_changes(before_conn, after_conn, d)
                rec["row_changes"] = len(ch)
                render_before_after(db, before_conn, after_conn, d, ch, cache_dir,
                                    str(out / "before_after" / f"D{d:02d}.jpg"))
            summary["desktops"][d] = rec
        summary_path = out / f"u31_summary_{'apply' if args.apply else 'dryrun'}_{stamp}.json"
        if args.apply:
            db.log_op(0, "-", "note", {"note": "controller U31-fix: U30-QA layering groups "
                                       "(set_zorder), HP cable clip parent by picture "
                                       "(relation_fix + commit_keyframe), the conflicts they "
                                       f"opened accepted; summary {summary_path}"},
                      {}, ANNOTATOR)
        before_conn.close()
        db.close()
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, default=str)
        log(f"summary -> {summary_path}")
    except BaseException:
        log(traceback.format_exc())
        where = f" (on D{run['desktop']:02d})" if run["desktop"] is not None else ""
        if not run["writing"]:
            log(f"FAILED before anything was written{where}")
        elif args.apply and is_live:
            log(f"FAILED part-way through{where}: the live database is half done. Restore the "
                f"safety backup {run['backup']} over it; do NOT re-run on top -- the "
                f"conflicts already queued would be taken for the user's and never accepted")
        elif args.apply:
            log(f"FAILED part-way through{where}: {target} is half done; start again from a "
                f"fresh copy")
        else:
            log(f"FAILED part-way through{where}: only the dry-run copy {target} is spoiled")
        raise
    finally:
        release_lock_file(target)
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
