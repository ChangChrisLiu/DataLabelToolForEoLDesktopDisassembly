"""Ground truth for L5: every verified scan mask of D2, D3, D4, D5, D13 (DB copy, read-only).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.gt

Population (L4's, :mod:`experiments.l4_detector.gt`): every ``frame`` of view
``scan`` with ``review_status = 'verified'`` on the five desktops; per frame the
``compiled_mask`` rows with ``placement = 'in_chassis'``, ``visibility`` in
:data:`KEEP_VIS` and a non-empty ``visible_rle_json``.  ``occluded_full`` (no
visible pixels), ``out_of_view`` and bench rows are skipped.

**Target = the visible mask.**  The app's editing layer is the instance's
*amodal* shape keyframe (``session_commits.begin_edit`` loads ``found.amodal``;
``shape_keyframe.amodal_complete``), and the compiler derives the visible mask
as amodal minus every part above it in the z-order (spec 3.3 step 5).  SAM only
ever sees pixels, so what it is asked to produce is the part's *visible* region
on the current image -- and in the reverse walk a newly drawn part goes on top
of the stack (spec 4.2, v1.16), so on the frame it is drawn the committed shape
*is* its visible region (except the two floor layers, chassis and motherboard,
whose amodal shape may continue under parts already drawn).  On the other
frames of its range the compiled visible mask is the shape minus whatever the
annotator later put on top.  Both are what the pixels show, so the visible mask
is the target on every frame.

Per mask the record keeps the class (``instance.cls``), visibility, occlusion
ratio, area, the bbox (exclusive x1/y1), the distance-transform click point,
whether this frame is the one the keyframe in force was committed on
(``drawn``: the op-log's ``commit_keyframe`` steps) and the bit-packed bbox crop
of the mask.  Written to ``OUT/gt/gt_l5.pkl``.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.l5_sam import env

DESKTOPS = (2, 3, 4, 5, 13)
VIEW = "scan"
KEEP_VIS = ("visible", "visible_tiny", "occluded_partial", "too_small")
GT_PKL = env.OUT / "gt" / "gt_l5.pkl"

#: Class groups of the decision rule (``decision_rule.txt``).
LATCHES = ("ram_latch", "psu_latch", "drive_latch")
CONNECTORS = ("connector",)
SIZE_EDGES = (400, 2_000, 20_000)
SIZE_NAMES = ("<400", "400-2k", "2k-20k", ">20k")


def size_bucket(area: int) -> str:
    for edge, name in zip(SIZE_EDGES, SIZE_NAMES):
        if area < edge:
            return name
    return SIZE_NAMES[-1]


@dataclass
class Rec:
    desktop: int
    step: int
    instance: str
    cls: str
    visibility: str
    occlusion: float
    area: int
    bbox: tuple[int, int, int, int]      # x0, y0, x1, y1 (exclusive)
    point: tuple[int, int]               # distance-transform argmax (x, y)
    drawn: bool                          # keyframe committed on this frame
    frame_hw: tuple[int, int]
    packed: np.ndarray                   # np.packbits of the bbox crop

    @property
    def key(self) -> str:
        return f"D{self.desktop}:{self.instance}"

    @property
    def frame(self) -> tuple[int, int]:
        return self.desktop, self.step

    def mask_crop(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        h, w = y1 - y0, x1 - x0
        return np.unpackbits(self.packed, count=h * w).astype(bool).reshape(h, w)

    def mask_in(self, win: tuple[int, int, int, int]) -> np.ndarray:
        """The mask cut to window ``(x0, y0, x1, y1)`` in frame coordinates."""
        wx0, wy0, wx1, wy1 = win
        out = np.zeros((wy1 - wy0, wx1 - wx0), dtype=bool)
        x0, y0, x1, y1 = self.bbox
        ix0, iy0 = max(x0, wx0), max(y0, wy0)
        ix1, iy1 = min(x1, wx1), min(y1, wy1)
        if ix1 <= ix0 or iy1 <= iy0:
            return out
        m = self.mask_crop()
        out[iy0 - wy0:iy1 - wy0, ix0 - wx0:ix1 - wx0] = m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        return out

    @property
    def group(self) -> list[str]:
        g = []
        if self.area < 400:
            g.append("small")
        if self.cls == "screw":
            g.append("screw")
        if self.cls in CONNECTORS:
            g.append("connector")
        if self.cls in LATCHES:
            g.append("latch")
        if self.area > 20_000:
            g.append("large")
        return g


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """The mask pixel farthest from the boundary (sam_compare's click)."""
    import cv2

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    idx = int(np.argmax(dist))
    y, x = divmod(idx, dist.shape[1])
    return x - 1, y - 1


_RAW = False


def image_path(con, desktop: int, step: int) -> Optional[str]:
    """The file the app's canvas shows (L4's ``frame_candidates`` order).

    The local cache copy first, then the row's recorded cache copy, then the raw
    path through :mod:`tda.core.rawroot` (read-only drive).  Copied from L4
    rather than imported: L4's env module would re-point every cache variable.
    """
    global _RAW
    import yaml

    from tda.core import rawroot
    from tda.core.cache import cached_image_path
    from tda.core.model import FrameKey

    if not _RAW:
        rawroot.configure(yaml.safe_load((env.REPO / "configs" / "paths.yaml").read_text("utf-8")))
        _RAW = True
    r = con.execute("SELECT path, aux_json FROM frame WHERE desktop=? AND view=? AND step=?",
                    (desktop, VIEW, step)).fetchone()
    if r is None:
        return None
    aux = json.loads(r["aux_json"]) if r["aux_json"] else {}
    cands = [cached_image_path(env.CACHE_DIR, FrameKey(desktop, step, VIEW)),
             rawroot.resolve_raw(aux.get("cache_path")), rawroot.resolve_raw(r["path"])]
    for c in cands:
        if c and os.path.isfile(str(c)):
            return str(c)
    return None


def _drawn_steps(con) -> dict[tuple[int, str], set[int]]:
    """``(desktop, instance) -> steps`` on which a keyframe was committed (op log)."""
    out: dict[tuple[int, str], set[int]] = {}
    for r in con.execute("SELECT payload_json FROM op_log WHERE kind='commit_keyframe' "
                         "AND view=?", (VIEW,)):
        try:
            p = json.loads(r["payload_json"] or "{}")
        except ValueError:
            continue
        if p.get("view") != VIEW or p.get("desktop") not in DESKTOPS:
            continue
        out.setdefault((int(p["desktop"]), str(p.get("instance"))), set()).add(int(p.get("step", -1)))
    return out


def build() -> tuple[list[Rec], dict[tuple[int, int], str], Counter]:
    from tda.core import masks

    con = env.connect()
    cls_of = {(int(r["desktop"]), str(r["key"])): str(r["cls"])
              for r in con.execute("SELECT desktop, key, cls FROM instance")}
    drawn = _drawn_steps(con)
    recs: list[Rec] = []
    images: dict[tuple[int, int], str] = {}
    skipped: Counter = Counter()
    for d in DESKTOPS:
        steps = [int(r["step"]) for r in con.execute(
            "SELECT step FROM frame WHERE desktop=? AND view=? AND review_status='verified' "
            "ORDER BY step", (d, VIEW))]
        for s in steps:
            img = image_path(con, d, s)
            if img is None:
                skipped["no_image_frame"] += 1
                continue
            images[(d, s)] = img
            for r in con.execute(
                    "SELECT instance, visible_rle_json, visibility, placement, occlusion_ratio "
                    "FROM compiled_mask WHERE desktop=? AND view=? AND step=? ORDER BY instance",
                    (d, VIEW, s)):
                vis = str(r["visibility"] or "")
                if r["placement"] != "in_chassis":
                    skipped["bench"] += 1
                    continue
                if vis not in KEEP_VIS:
                    skipped[vis or "none"] += 1
                    continue
                if not r["visible_rle_json"]:
                    skipped[f"{vis}:no_mask"] += 1
                    continue
                rle = json.loads(r["visible_rle_json"])
                m = masks.decode_rle(rle).astype(bool)
                ys, xs = np.nonzero(m)
                if ys.size == 0:
                    skipped[f"{vis}:empty"] += 1
                    continue
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                x0, y0, x1, y1 = bbox
                inst = str(r["instance"])
                recs.append(Rec(
                    desktop=d, step=s, instance=inst,
                    cls=cls_of.get((d, inst), inst.split(".", 1)[0]),
                    visibility=vis, occlusion=float(r["occlusion_ratio"] or 0.0),
                    area=int(ys.size), bbox=bbox, point=interior_point(m),
                    drawn=s in drawn.get((d, inst), set()),
                    frame_hw=(int(m.shape[0]), int(m.shape[1])),
                    packed=np.packbits(m[y0:y1, x0:x1])))
    con.close()
    return recs, images, skipped


def save(recs: list[Rec], images: dict, path: Path = GT_PKL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # plain dicts: a pickled Rec written by ``python -m ...gt`` would belong to __main__
    with open(path, "wb") as fh:
        pickle.dump({"recs": [vars(r) for r in recs], "images": images}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path = GT_PKL) -> tuple[list[Rec], dict[tuple[int, int], str]]:
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    return [Rec(**r) for r in blob["recs"]], blob["images"]


def load_frames(images: dict[tuple[int, int], str],
                keys: Optional[set] = None) -> dict[tuple[int, int], np.ndarray]:
    """Decode the frames (RGB uint8) into RAM; 1600^2 x 3 = 7.7 MB each."""
    import cv2

    out = {}
    for k, p in sorted(images.items()):
        if keys is not None and k not in keys:
            continue
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {p}")
        out[k] = np.ascontiguousarray(bgr[:, :, ::-1])
    return out


def main() -> int:
    recs, images, skipped = build()
    save(recs, images)
    print(f"[l5] {len(images)} verified frames with an image, {len(recs)} masks; skipped {dict(skipped)}")
    by_d = Counter(r.desktop for r in recs)
    print("masks per desktop:", dict(sorted(by_d.items())))
    print("instances per desktop:", dict(sorted(Counter(
        d for d, _i in {(r.desktop, r.instance) for r in recs}).items())))
    cnt = Counter((r.cls, r.desktop) for r in recs)
    classes = sorted({c for c, _d in cnt}, key=lambda c: -sum(cnt[(c, d)] for d in DESKTOPS))
    print("class | " + " | ".join(f"D{d}" for d in DESKTOPS) + " | total | instances")
    for c in classes:
        row = [cnt[(c, d)] for d in DESKTOPS]
        n_inst = len({(r.desktop, r.instance) for r in recs if r.cls == c})
        print(f"{c} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} | {n_inst}")
    print("size buckets:", dict(Counter(size_bucket(r.area) for r in recs)))
    print("groups:", dict(Counter(g for r in recs for g in r.group)))
    print("visibility:", dict(Counter(r.visibility for r in recs)))
    print("occluded (ratio > 0.02):", sum(r.occlusion > 0.02 for r in recs),
          "drawn-frame masks:", sum(r.drawn for r in recs))
    hw = Counter(r.frame_hw for r in recs)
    print("frame sizes:", dict(hw))
    print(f"wrote {GT_PKL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
