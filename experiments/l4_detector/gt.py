"""Ground truth from the annotator's verified scanner frames (the DB copy, read-only).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.gt

Population: every ``frame`` of view ``scan`` with ``review_status =
'verified'`` on desktops 2, 3, 4, 5 and 13 (230 frames).  Per frame:

* **boxes** -- the frame's ``compiled_mask`` rows (the compiled, verified
  per-frame masks) with ``placement = 'in_chassis'`` and ``visibility`` in
  :data:`KEEP_VIS`; the box is the bbox of the *visible* mask
  (``visible_rle_json``; ``box_json`` is empty for mask rows), exclusive
  x1/y1 as L2's drafts.  ``occluded_full`` / ``out_of_view`` rows and bench
  boxes are skipped.  The class is ``instance.cls``; the production head's 8
  classes are kept, every other class is counted (:func:`main`) but not a
  detector target.
* **ROI** -- the stored ROI of the pose segment that holds the step
  (``pose_segment.roi_json``), exactly what the app crops to
  (``tda.ui.app_roi.RoiMixin.roi``).
* **image** -- the file the app's canvas shows
  (``tda.ui.session_images.frame_candidates``: the local cache copy first,
  then the row's recorded cache copy, then the raw path through
  ``tda.core.rawroot``).

Removal events (the guess, L2's protocol with the DB's own actions): step
``k`` with an action whose verb is a removal (``remove``, ``disconnect``,
``unscrew``, ``displace`` -- B2's ``REMOVAL_VERBS``) on target ``t``, where
frames ``j = k - 1`` and ``k`` are both verified and ``t`` has a visible mask
on ``j``.  The part is the target's visible mask on ``j``; the parts already
drawn when the annotator reaches ``j`` in the reverse walk are the other
instances with a visible box on ``j``.  An event whose target is still
visible on ``k`` (a cable end that stays in view) is counted and dropped.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402

DESKTOPS = (2, 3, 4, 5, 13)
VIEW = "scan"
KEEP_VIS = ("visible", "visible_tiny", "occluded_partial", "too_small")
REMOVAL_VERBS = ("remove", "disconnect", "unscrew", "displace")
DET_CLASSES = ("screw", "connector", "ram_latch", "psu_latch",
               "cpu_socket_lever", "drive_latch", "ram_module", "cpu")
SMALL_CLASSES = DET_CLASSES[:6]
GT_JSON = env.OUT / "gt" / "gt_scan_verified.json"


@dataclass
class GtBox:
    desktop: int
    step: int
    instance: str
    cls: str                 # instance.cls (taxonomy)
    visibility: str
    area: int                # visible pixels
    box: tuple[int, int, int, int]

    @property
    def det(self) -> Optional[str]:
        return self.cls if self.cls in DET_CLASSES else None

    @property
    def width(self) -> float:
        return float(np.sqrt(max(1, self.area)))


@dataclass
class Frame:
    desktop: int
    step: int
    roi: tuple[int, int, int, int]
    image: Optional[str]
    boxes: list[GtBox] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)


@dataclass
class Event:
    desktop: int
    step: int                # k: the action's step; the guess is on j = k - 1
    verb: str
    target: str
    cls: str
    box: tuple[int, int, int, int]
    area: int
    roi: tuple[int, int, int, int]

    @property
    def j(self) -> int:
        return self.step - 1

    @property
    def width(self) -> float:
        return float(np.sqrt(max(1, self.area)))

    @property
    def key(self) -> str:
        return f"{self.desktop}:{VIEW}:{self.step}:{self.target}"


def connect() -> sqlite3.Connection:
    if "annotations" in Path(env.DB).resolve().parts:
        raise SystemExit("refusing to open the live database")
    con = sqlite3.connect(env.DB_URI, uri=True)
    con.row_factory = sqlite3.Row
    return con


_RAW = False


def _raw() -> None:
    global _RAW
    if _RAW:
        return
    import yaml

    from tda.core import rawroot

    paths = yaml.safe_load((env.REPO / "configs" / "paths.yaml").read_text("utf-8"))
    rawroot.configure(paths)            # a missing raw drive only matters if the cache misses
    _RAW = True


def image_path(con, desktop: int, step: int, view: str = VIEW) -> Optional[str]:
    """The file the app's canvas shows for this frame (``frame_candidates`` order)."""
    from tda.core.cache import cached_image_path
    from tda.core.model import FrameKey
    from tda.core.rawroot import resolve_raw

    _raw()
    r = con.execute("SELECT path, aux_json FROM frame WHERE desktop=? AND view=? AND step=?",
                    (desktop, view, step)).fetchone()
    if r is None:
        return None
    aux = json.loads(r["aux_json"]) if r["aux_json"] else {}
    cands = [cached_image_path(env.CACHE_DIR, FrameKey(desktop, step, view)),
             resolve_raw(aux.get("cache_path")), resolve_raw(r["path"])]
    for c in cands:
        if c and os.path.isfile(str(c)):
            return str(c)
    return None


def roi_of(con, desktop: int, step: int, view: str = VIEW) -> Optional[tuple]:
    r = con.execute(
        "SELECT roi_json FROM pose_segment WHERE desktop=? AND view=? AND start_step<=? "
        "AND end_step>=? ORDER BY seg LIMIT 1", (desktop, view, step, step)).fetchone()
    if r is None or not r["roi_json"]:
        return None
    return tuple(int(v) for v in json.loads(r["roi_json"]))


def _classes(con) -> dict:
    return {(int(r["desktop"]), str(r["key"])): str(r["cls"])
            for r in con.execute("SELECT desktop, key, cls FROM instance")}


def build(con=None) -> list[Frame]:
    """Every verified scan frame of :data:`DESKTOPS` with its boxes."""
    from tda.core import masks

    own = con is None
    con = con or connect()
    cls_of = _classes(con)
    frames: list[Frame] = []
    for d in DESKTOPS:
        steps = [int(r["step"]) for r in con.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            "ORDER BY step", (d, VIEW))]
        for s in steps:
            roi = roi_of(con, d, s)
            if roi is None:
                raise SystemExit(f"D{d} {VIEW} s{s}: no stored ROI")
            fr = Frame(desktop=d, step=s, roi=roi, image=image_path(con, d, s))
            for r in con.execute(
                    "SELECT instance, visible_rle_json, visibility, placement, status "
                    "FROM compiled_mask WHERE desktop=? AND view=? AND step=? ORDER BY instance",
                    (d, VIEW, s)):
                vis = str(r["visibility"] or "")
                if r["placement"] != "in_chassis":
                    fr.skipped["bench"] += 1
                    continue
                if vis not in KEEP_VIS:
                    fr.skipped[vis or "none"] += 1
                    continue
                if not r["visible_rle_json"]:
                    fr.skipped[f"{vis}:no_mask"] += 1
                    continue
                rle = json.loads(r["visible_rle_json"])
                area = int(masks.rle_area(rle))
                box = masks.rle_bbox(rle)
                if area <= 0 or box is None:
                    fr.skipped[f"{vis}:empty"] += 1
                    continue
                inst = str(r["instance"])
                cls = cls_of.get((d, inst), inst.split(".", 1)[0])
                fr.boxes.append(GtBox(desktop=d, step=s, instance=inst, cls=cls,
                                      visibility=vis, area=area,
                                      box=tuple(int(v) for v in box)))
            frames.append(fr)
    if own:
        con.close()
    return frames


def save(frames: list[Frame], path: Path = GT_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{
        "desktop": f.desktop, "step": f.step, "roi": list(f.roi), "image": f.image,
        "skipped": dict(f.skipped),
        "boxes": [{"instance": b.instance, "cls": b.cls, "visibility": b.visibility,
                   "area": b.area, "box": list(b.box)} for b in f.boxes]}
        for f in frames], indent=0), encoding="utf-8")


def load(path: Path = GT_JSON) -> list[Frame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in raw:
        fr = Frame(desktop=f["desktop"], step=f["step"], roi=tuple(f["roi"]),
                   image=f["image"], skipped=Counter(f["skipped"]))
        fr.boxes = [GtBox(desktop=f["desktop"], step=f["step"], instance=b["instance"],
                          cls=b["cls"], visibility=b["visibility"], area=b["area"],
                          box=tuple(b["box"])) for b in f["boxes"]]
        out.append(fr)
    return out


def index(frames: list[Frame]) -> dict[tuple[int, int], Frame]:
    return {(f.desktop, f.step): f for f in frames}


def visible_mask(con, desktop: int, step: int, instance: str) -> Optional[np.ndarray]:
    from tda.core import masks

    r = con.execute("SELECT visible_rle_json FROM compiled_mask WHERE desktop=? AND view=? "
                    "AND step=? AND instance=?", (desktop, VIEW, step, instance)).fetchone()
    if r is None or not r["visible_rle_json"]:
        return None
    return masks.decode_rle(json.loads(r["visible_rle_json"])).astype(bool)


def events(frames: list[Frame], con=None) -> tuple[list[Event], Counter]:
    """Removal events on verified frame pairs (j, k = j + 1); the second value says why others were dropped."""
    own = con is None
    con = con or connect()
    by = index(frames)
    cls_of = _classes(con)
    out: list[Event] = []
    why = Counter()
    for r in con.execute("SELECT desktop, step, idx, target, verb FROM action "
                         "WHERE desktop IN (%s) ORDER BY desktop, step, idx"
                         % ",".join(str(d) for d in DESKTOPS)):
        d, k, verb, target = int(r["desktop"]), int(r["step"]), str(r["verb"]), str(r["target"])
        if verb not in REMOVAL_VERBS:
            why[f"verb:{verb}"] += 1
            continue
        fj, fk = by.get((d, k - 1)), by.get((d, k))
        if fj is None or fk is None:
            why["frame_not_verified"] += 1
            continue
        on_j = [b for b in fj.boxes if b.instance == target]
        if not on_j:
            why["not_visible_on_j"] += 1
            continue
        if any(b.instance == target for b in fk.boxes):
            why["still_visible_on_k"] += 1
            continue
        b = on_j[0]
        out.append(Event(desktop=d, step=k, verb=verb, target=target,
                         cls=cls_of.get((d, target), b.cls), box=b.box, area=b.area,
                         roi=fj.roi))
    if own:
        con.close()
    return out, why


def drawn_on_j(frames_by, ev: Event, cls: str) -> list[tuple]:
    """Same-class boxes already drawn when the reverse walk reaches ``j``: every
    other instance of ``cls`` visible on ``j`` (the removed part is the one not drawn)."""
    fj = frames_by[(ev.desktop, ev.j)]
    return [b.box for b in fj.boxes if b.cls == cls and b.instance != ev.target]


def main() -> int:
    con = connect()
    frames = build(con)
    save(frames)
    evs, why = events(frames, con)
    con.close()
    n_img = sum(1 for f in frames if f.image)
    print(f"[l4] {len(frames)} verified scan frames ({n_img} with an image), "
          f"{sum(len(f.boxes) for f in frames)} boxes")
    cnt = Counter((f.desktop, b.cls) for f in frames for b in f.boxes)
    classes = sorted({c for _d, c in cnt}, key=lambda c: (c not in DET_CLASSES, c))
    print("class | " + " | ".join(f"D{d}" for d in DESKTOPS) + " | total")
    for c in classes:
        row = [cnt[(d, c)] for d in DESKTOPS]
        print(f"{c}{'' if c in DET_CLASSES else ' (other)'} | "
              + " | ".join(str(v) for v in row) + f" | {sum(row)}")
    sk = Counter()
    for f in frames:
        sk.update(f.skipped)
    print("skipped rows:", dict(sk))
    vis = Counter(b.visibility for f in frames for b in f.boxes)
    print("kept visibility:", dict(vis))
    print(f"events: {len(evs)}; by class {dict(Counter(e.cls for e in evs))}; dropped {dict(why)}")
    print(f"wrote {GT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
