"""Where the part is in the scanner view: what the 「主视角」 panel shows (task U23).

The scanner view (``scan``) of a machine is annotated first; its OAK1, OAK2
and RealSense views come after, and on those the annotator kept hunting for
the part the task card names.  :class:`LocatorSource` answers one question for
the panel (:mod:`tda.ui.panels.locator`): *for this step and this part, which
scanner frame, what picture, and where on it?* -- as a :class:`LocatorScene`.

Costs, which are the point of most of this module:

* **the picture** is the scanner frame of the same step (the nearest earlier
  one when that step's scan frame is missing or a ``dupli`` step), read from
  the local cache at :data:`THUMB_SIDE` -- ``QImageReader`` is told the size,
  so a JPEG is decoded scaled by its own codec, the same way the timeline reads
  its rows -- and kept in memory, :data:`THUMB_CACHE` steps, least recently
  used first out.  Never a full-resolution frame kept per step.  (The offline
  thumbnail tier, :mod:`tda.core.cache_thumbs`, is 192 px and cut to the
  chassis: a screw is two pixels there.)
* **the outline** is read off the truth table: the stored ``compiled_mask`` row
  of that scanner frame and that part, one ``SELECT``, no recompilation.  Its
  RLE is decoded inside its own bounding box (:func:`tda.core.masks.
  decode_rle_shared`), traced at no more than :data:`OUTLINE_SIDE` pixels and
  remembered by the RLE's content, so a re-traced shape is a new entry rather
  than a stale one.  A frame with no stored row -- never visited, so never
  compiled -- is compiled **once** (:meth:`tda.core.truth.TruthService.
  compile`, nothing written) and every part's outline of it kept, until the
  scanner view's inputs change (:func:`scan_digest`).

The 「拆前 / 拆后」 pair of task U28 -- the scanner frames around the step a
part leaves at -- is built from the same pictures, outlines and frame choice
by :mod:`tda.ui.locator_pair` (:meth:`LocatorSource.pair`), and
:meth:`LocatorSource.own_shapes` answers whether a part is met for the first
time in a view.

Nothing here writes to the database.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QImageReader

from tda.core import masks
from tda.core.model import FrameKey, is_context_key, is_provisional
from tda.core.truth_inputs import state_of
from tda.ui.canvas.overlay import palette_color
from tda.ui.class_names import instance_label, state_zh
from tda.ui.session_images import ImageCache

__all__ = [
    "CAPTION", "EARLIER_DUPLI", "EARLIER_MISSING", "Geometry", "HEAD", "LocatorScene",
    "LocatorSource",
    "NO_SCAN_FRAME", "NO_TARGET", "NOT_ANNOTATED", "NOT_DRAWN", "NOT_HERE", "SCAN",
    "THUMB_CACHE", "THUMB_SIDE", "UNREADABLE", "caption_for", "geometry_from_box",
    "geometry_from_mask", "read_scaled", "scan_digest",
]

SCAN = "scan"
#: The long side the scanner frame is read at.  The panel is about 300 px wide
#: and a crop around a part is a fraction of the frame, so the picture is kept
#: larger than the panel; 1024 is a 1600 px scanner frame at 64 %.
THUMB_SIDE = 1024
#: Steps whose picture is kept (RGB888, at most ~3 MB each at 1024 px).
THUMB_CACHE = 20
#: Outlines remembered, by the content of the RLE they were traced from.
GEOMETRY_CACHE = 256
#: A part is traced at no more than this many pixels on its long side.
OUTLINE_SIDE = 256

# --- what the panel says ----------------------------------------------------
#: The caption: which scanner frame, which part, in which state.
HEAD = "主视角 第 {step} 帧{earlier}"
CAPTION = HEAD + " · {label}{state}"
#: ... when the step's own scanner frame could not be used.
EARLIER_MISSING = "（第 {asked} 帧扫描仪没有图）"
EARLIER_DUPLI = "（第 {asked} 帧是重复帧）"
#: No geometry for the part on that scanner frame, for each of the three reasons.
NOT_HERE = "扫描仪视角里这一帧没有它"
NOT_DRAWN = "扫描仪视角里还没画它"
NOT_ANNOTATED = "扫描仪视角还没标注，只显示画面"
#: Nothing to look for.
NO_TARGET = "选中一个零件（或任务卡上的当前行），这里标出它在扫描仪画面里的位置"
#: No scanner picture at or before the step.
NO_SCAN_FRAME = "扫描仪视角在第 {asked} 帧及之前都没有图"
#: The picture exists but cannot be read (the raw drive, a broken file).
UNREADABLE = "扫描仪第 {step} 帧的图读不到"

_DIGEST_SQL = (
    "SELECT "
    "(SELECT COUNT(*) || ':' || IFNULL(MAX(id), 0) || ':' || IFNULL(SUM(version), 0) "
    "  || ':' || IFNULL(SUM(edit_count), 0) FROM shape_keyframe WHERE desktop=? AND view=?), "
    "(SELECT COUNT(*) || ':' || IFNULL(SUM(LENGTH(visible_rle_json)), 0) || ':' "
    "  || IFNULL(GROUP_CONCAT(visibility), '') FROM frame_override WHERE desktop=? AND view=?), "
    "(SELECT COUNT(*) || ':' || IFNULL(MAX(id), 0) FROM state_event WHERE desktop=?), "
    "(SELECT COUNT(*) || ':' || IFNULL(MAX(rowid), 0) FROM instance WHERE desktop=?), "
    "(SELECT COUNT(*) || ':' || IFNULL(SUM(dupli), 0) FROM step WHERE desktop=?), "
    "(SELECT IFNULL(SUM(version), 0) || ':' || COUNT(*) FROM zorder WHERE desktop=? AND view=?), "
    "(SELECT COUNT(*) FROM pair_override WHERE desktop=? AND view=?), "
    "(SELECT COUNT(*) || ':' || IFNULL(SUM(missing), 0) FROM frame WHERE desktop=? AND view=?)"
)


def scan_digest(db, desktop: int) -> tuple:
    """A cheap fingerprint of everything a scanner frame is compiled from.

    One query, a handful of aggregates: the scanner view's keyframes, frame
    overrides, z-order and pair overrides, the machine's state events,
    instances and step table, and the scanner frame rows.  Whatever is cached
    from a compilation is valid exactly as long as this does not move.
    """
    row = db.conn.execute(_DIGEST_SQL, (desktop, SCAN, desktop, SCAN, desktop, desktop,
                                        desktop, desktop, SCAN, desktop, SCAN,
                                        desktop, SCAN)).fetchone()
    return tuple(row)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    """Where a part is on one scanner frame, in that frame's pixels.

    ``box`` is the union bounding box ``(x0, y0, x1, y1)`` (upper bounds
    exclusive) of every piece; ``outlines`` the traced outer boundaries, one
    ``(N, 2)`` array of ``x, y`` per piece; ``centre`` the point the dot goes on
    -- inside the part, the point furthest from its edge.
    """

    box: tuple[float, float, float, float]
    outlines: tuple = ()
    centre: tuple[float, float] = (0.0, 0.0)
    kind: str = "mask"


def geometry_from_box(box) -> Optional[Geometry]:
    """A bench box's rectangle as a :class:`Geometry`, ``None`` when degenerate."""
    if box is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        return None
    ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return Geometry((x0, y0, x1, y1), (ring,), ((x0 + x1) / 2.0, (y0 + y1) / 2.0), "box")


def geometry_from_mask(crop: np.ndarray, box) -> Optional[Geometry]:
    """Trace a mask given inside its bounding box ``box = (x0, y0, x1, y1)``.

    The crop is shrunk to at most :data:`OUTLINE_SIDE` on its long side first
    (the panel draws it a few hundred pixels across), and a part that thin
    survives the shrink: a pixel is set when any of the pixels it stands for is.
    """
    if crop is None or crop.size == 0 or box is None:
        return None
    x0, y0 = float(box[0]), float(box[1])
    h, w = crop.shape[:2]
    scale = min(1.0, OUTLINE_SIDE / float(max(h, w)))
    small_w, small_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    u8 = np.ascontiguousarray(crop, dtype=np.uint8)
    if scale < 1.0:
        u8 = cv2.resize(u8 * 255, (small_w, small_h), interpolation=cv2.INTER_AREA)
        binary = (u8 > 0).astype(np.uint8)
    else:
        binary = (u8 > 0).astype(np.uint8)
    if not binary.any():
        return None
    sx, sy = small_w / float(w), small_h / float(h)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(float)
        pts[:, 0] = x0 + (pts[:, 0] + 0.5) / sx
        pts[:, 1] = y0 + (pts[:, 1] + 0.5) / sy
        outlines.append(pts)
    # The dot: the point furthest from the edge, which is inside the part
    # whatever its shape (an L, a ring), on the thickest piece of a split one.
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    _min, _max, _minloc, (px, py) = cv2.minMaxLoc(dist)
    centre = (x0 + (px - 1 + 0.5) / sx, y0 + (py - 1 + 0.5) / sy)
    full = (x0, y0, x0 + float(w), y0 + float(h))
    return Geometry(full, tuple(outlines), centre, "mask")


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
@dataclass
class LocatorScene:
    """Everything the panel draws for one step and one part."""

    desktop: Optional[int] = None
    #: The step the annotator is on, in the OAK / RealSense view.
    asked_step: Optional[int] = None
    #: The scanner frame shown: ``asked_step``, or the nearest earlier one.
    scan_step: Optional[int] = None
    instance: Optional[str] = None
    #: The scanner frame at :data:`THUMB_SIDE`, ``None`` when there is none.
    image: Optional[QImage] = None
    #: The scanner frame's own size, ``(h, w)``: what the geometry is in.
    frame_hw: tuple[int, int] = (0, 0)
    geometry: Optional[Geometry] = None
    colour: tuple[int, int, int] = (255, 210, 0)
    #: The scanner pose segment's ROI, ``(x0, y0, x1, y1)``, when it has one.
    roi: Optional[tuple] = None
    label: str = ""
    state: str = ""
    caption: str = ""
    #: The sentence under the caption: why there is no outline, or ``""``.
    note: str = ""
    #: Which code path produced the geometry: ``row`` / ``compiled`` / ``""``.
    source: str = ""
    extras: dict = field(default_factory=dict)


def caption_for(scan_step: int, asked: int, why: str, label: str, state: str) -> str:
    """「主视角 第 k 帧 · <label>（<state>）」, with the earlier-frame clause when it applies."""
    earlier = ""
    if why:
        earlier = (EARLIER_DUPLI if why == "dupli" else EARLIER_MISSING).format(asked=int(asked))
    if not label:
        return HEAD.format(step=int(scan_step), earlier=earlier)
    return CAPTION.format(step=int(scan_step), earlier=earlier, label=label,
                          state=f"（{state}）" if state else "")


class LocatorSource:
    """The data side of the 「主视角」 panel; see the module doc for its costs."""

    def __init__(self, db, tax, truth, cache_dir: str, *,
                 thumb_side: int = THUMB_SIDE, thumb_cache: int = THUMB_CACHE) -> None:
        self.db = db
        self.tax = tax
        self.truth = truth
        self.cache_dir = str(cache_dir or "")
        self.thumb_side = int(thumb_side)
        self.thumb_cache = int(thumb_cache)
        self._paths = ImageCache(db, self.cache_dir)
        #: ``(desktop, step) -> (QImage | None, (h, w))``
        self._thumbs: "OrderedDict[tuple[int, int], tuple]" = OrderedDict()
        #: ``(counts, size) -> Geometry | None``, by content
        self._outlines: "OrderedDict[tuple, Optional[Geometry]]" = OrderedDict()
        #: Per desktop: ``(digest, facts)``; ``facts`` holds the scan frame
        #: list, the drawn set and the compiled fallbacks of that digest.
        self._facts: dict[int, tuple[tuple, dict]] = {}
        self._pinned: Optional[tuple[int, dict]] = None
        #: Counters the tests and the report read.  ``pairs`` / ``diffs`` (task
        #: U28): 「拆前 / 拆后」 scenes built, differences of two pictures taken.
        self.stats = {"thumb_reads": 0, "row_reads": 0, "compiles": 0, "traces": 0,
                      "pairs": 0, "diffs": 0}
        self._pair_maker = None

    # -- caches ---------------------------------------------------------------
    def forget(self) -> None:
        """Drop everything derived from the database (the pictures stay)."""
        self._facts.clear()

    def thumbs_cached(self) -> list[tuple[int, int]]:
        """``(desktop, step)`` of every picture held, least recent first."""
        return list(self._thumbs)

    def _desktop_facts(self, desktop: int) -> dict:
        """The per-desktop facts of the current scanner inputs (re-read when they move).

        One :func:`scan_digest` per :meth:`scene`: inside one, the facts it
        started from are pinned.
        """
        pinned = self._pinned
        if pinned is not None and pinned[0] == desktop:
            return pinned[1]
        digest = scan_digest(self.db, desktop)
        held = self._facts.get(desktop)
        if held is not None and held[0] == digest:
            return held[1]
        facts: dict[str, Any] = {"compiled": {}, "digest": digest}
        rows = {int(r["step"]): r for r in self.db.frames_for(desktop, SCAN)}
        dupli = {int(s.step) for s in self.db.steps(desktop)
                 if s.dupli or str(s.step_type) == "dupli"}
        facts["frames"] = rows
        facts["dupli"] = dupli
        facts["usable"] = sorted(
            step for step, row in rows.items()
            if not row.get("missing") and (row.get("path") or (row.get("aux") or {})
                                           .get("cache_path"))
            and step not in dupli)
        drawn = {str(r[0]) for r in self.db.conn.execute(
            "SELECT DISTINCT instance FROM shape_keyframe WHERE desktop=? AND view=? "
            "UNION SELECT DISTINCT instance FROM frame_override WHERE desktop=? AND view=? "
            "AND visible_rle_json IS NOT NULL", (desktop, SCAN, desktop, SCAN))}
        facts["drawn"] = drawn
        facts["annotated"] = bool(drawn)
        facts["instances"] = self.db.instances(desktop)
        self._facts[desktop] = (digest, facts)
        return facts

    # -- which scanner frame ----------------------------------------------------
    def scan_step_for(self, desktop: int, step: int) -> tuple[Optional[int], str]:
        """``(scanner step to show, why it is not ``step``)``.

        ``why`` is ``""`` for the step itself, ``"missing"`` or ``"dupli"`` for the
        nearest earlier frame; the step is ``None`` when there is no earlier one.
        """
        facts = self._desktop_facts(int(desktop))
        usable = facts["usable"]
        step = int(step)
        if step in usable:
            return step, ""
        earlier = [s for s in usable if s < step]
        why = "dupli" if step in facts["dupli"] else "missing"
        return (max(earlier) if earlier else None), why

    # -- the picture ------------------------------------------------------------
    def thumbnail(self, desktop: int, step: int) -> tuple[Optional[QImage], tuple[int, int]]:
        """The scanner frame at :attr:`thumb_side`, and its own size; cached."""
        slot = (int(desktop), int(step))
        hit = self._thumbs.get(slot)
        if hit is not None:
            self._thumbs.move_to_end(slot)
            return hit
        path = self._paths.image_path(FrameKey(int(desktop), int(step), SCAN))
        found = (None, (0, 0)) if path is None else read_scaled(path, self.thumb_side)
        self.stats["thumb_reads"] += 1
        self._thumbs[slot] = found
        while len(self._thumbs) > self.thumb_cache:
            self._thumbs.popitem(last=False)
        return found

    # -- the outline ------------------------------------------------------------
    def geometry(self, desktop: int, step: int,
                 instance: str) -> tuple[Optional[Geometry], str, str]:
        """``(geometry, why there is none, where it came from)`` of one part.

        The order the reasons are tried in: the scanner view not annotated at
        all; the stored row (with geometry: done); the part never drawn in the
        scanner view; a stored row without geometry; no stored row -- compile
        the frame once.
        """
        desktop, step = int(desktop), int(step)
        facts = self._desktop_facts(desktop)
        if not facts["annotated"]:
            return None, NOT_ANNOTATED, ""
        row = self.db.conn.execute(
            "SELECT visible_rle_json, geom_type, box_json FROM compiled_mask "
            "WHERE desktop=? AND step=? AND view=? AND instance=?",
            (desktop, step, SCAN, str(instance))).fetchone()
        self.stats["row_reads"] += 1
        if row is not None:
            found = self._from_row(row)
            if found is not None:
                return found, "", "row"
        if str(instance) not in facts["drawn"]:
            return None, NOT_DRAWN, ""
        if row is not None:
            return None, NOT_HERE, "row"
        compiled = facts["compiled"].get(step)
        if compiled is None:
            compiled = self._compile(desktop, step)
            facts["compiled"][step] = compiled
        found = compiled.get(str(instance))
        return found, ("" if found is not None else NOT_HERE), "compiled"

    def _from_row(self, row) -> Optional[Geometry]:
        if str(row["geom_type"] or "mask") == "box":
            box = json.loads(row["box_json"]) if row["box_json"] else None
            return geometry_from_box(box)
        text = row["visible_rle_json"]
        if not text:
            return None
        rle = json.loads(text)
        if not rle:
            return None
        key = (masks.rle_counts(rle), tuple(int(v) for v in rle.get("size") or ()))
        if key in self._outlines:
            self._outlines.move_to_end(key)
            return self._outlines[key]
        crop, box = masks.decode_rle_shared(rle)
        found = None if box == (0, 0, 0, 0) else geometry_from_mask(crop, box)
        self.stats["traces"] += 1
        self._outlines[key] = found
        while len(self._outlines) > GEOMETRY_CACHE:
            self._outlines.popitem(last=False)
        return found

    def _compile(self, desktop: int, step: int) -> dict[str, Optional[Geometry]]:
        """Every part's geometry on a scanner frame with no stored rows (nothing written)."""
        self.stats["compiles"] += 1
        try:
            frame = self.truth.compile(FrameKey(desktop, step, SCAN))
        except Exception:  # noqa: BLE001 - a locator is never worth a failure
            return {}
        out: dict[str, Optional[Geometry]] = {}
        for name, inst in frame.instances.items():
            found = None
            if inst.visible is not None and inst.box is not None:
                x0, y0, x1, y1 = (int(v) for v in inst.box)
                if x1 > x0 and y1 > y0:
                    found = geometry_from_mask(inst.visible[y0:y1, x0:x1], (x0, y0, x1, y1))
            elif inst.visible is None and inst.box is not None:
                found = geometry_from_box(inst.box)
            out[str(name)] = found
        return out

    # -- the part's name and state ------------------------------------------------
    def label_and_state(self, desktop: int, step: int, instance: str) -> tuple[str, str]:
        """``("导风罩 cover.01", "打开")`` -- the state at the scanner frame shown."""
        facts = self._desktop_facts(int(desktop))
        rec = facts["instances"].get(str(instance))
        label = instance_label(str(instance), getattr(rec, "cls", None),
                               getattr(rec, "attrs", None))
        state = ""
        try:
            held = state_of(self.db, self.tax, int(desktop), int(step)).get(str(instance))
        except Exception:  # noqa: BLE001 - a word in a caption is never worth a failure
            held = None
        if held is not None and getattr(held, "state", None):
            state = state_zh(str(held.state))
        return label, state

    def is_instance(self, desktop: int, instance: Optional[str]) -> bool:
        """Is ``instance`` a part of this machine (a card row may name a raw step)?"""
        if not instance:
            return False
        return str(instance) in self._desktop_facts(int(desktop))["instances"]

    def own_shapes(self, desktop: int, view: str, instance: Optional[str]) -> tuple[bool, bool]:
        """``(a part of this machine, it has shapes of its own in view)`` -- task U28.

        A shape of its own is a ``shape_keyframe`` row of that view (a cut copy
        made from another frame is one too) or a single frame's own pixels
        (``frame_override``): either is the annotator having drawn it there.
        One indexed query; a 框外零件 or a Label Studio draft key is not a part.
        """
        if not instance or is_context_key(instance) or is_provisional(instance):
            return False, False
        row = self.db.conn.execute(
            "SELECT EXISTS(SELECT 1 FROM instance WHERE desktop=? AND key=?), "
            "EXISTS(SELECT 1 FROM shape_keyframe WHERE desktop=? AND view=? AND instance=?), "
            "EXISTS(SELECT 1 FROM frame_override WHERE desktop=? AND view=? AND instance=? "
            "AND visible_rle_json IS NOT NULL)",
            (int(desktop), str(instance), int(desktop), str(view), str(instance),
             int(desktop), str(view), str(instance))).fetchone()
        return bool(row[0]), bool(row[1] or row[2])

    def pair(self, desktop: int, instance: str):
        """The part's 「拆前 / 拆后」 (:class:`tda.ui.locator_pair.PairScene`), task U28."""
        if self._pair_maker is None:
            from tda.ui.locator_pair import PairMaker

            self._pair_maker = PairMaker(self)
        desktop = int(desktop)
        self._pinned = (desktop, self._desktop_facts(desktop))
        try:
            return self._pair_maker.pair(desktop, instance)
        finally:
            self._pinned = None

    def roi(self, desktop: int, step: int) -> Optional[tuple]:
        """The scanner pose segment's ROI at ``step``, or ``None``."""
        row = self.db.pose_segment_for(FrameKey(int(desktop), int(step), SCAN)) or {}
        stored = row.get("roi")
        if not stored or len(stored) != 4:
            return None
        return tuple(float(v) for v in stored)

    # -- all of it ------------------------------------------------------------------
    def scene(self, desktop: int, step: int, instance: Optional[str]) -> LocatorScene:
        """What the panel shows for the annotator's ``step`` and ``instance``."""
        desktop, step = int(desktop), int(step)
        self._pinned = (desktop, self._desktop_facts(desktop))
        try:
            return self._scene(desktop, step, instance)
        finally:
            self._pinned = None

    def _scene(self, desktop: int, step: int, instance: Optional[str]) -> LocatorScene:
        scene = LocatorScene(desktop=desktop, asked_step=step)
        scan_step, why = self.scan_step_for(desktop, step)
        if scan_step is None:
            scene.caption = "主视角"
            scene.note = NO_SCAN_FRAME.format(asked=step)
            return scene
        scene.scan_step = scan_step
        scene.image, scene.frame_hw = self.thumbnail(desktop, scan_step)
        scene.roi = self.roi(desktop, scan_step)
        target = instance if self.is_instance(desktop, instance) else None
        scene.instance = target
        if target is None:
            scene.caption = caption_for(scan_step, step, why, "", "")
            scene.note = NO_TARGET
        else:
            scene.colour = tuple(int(c) for c in palette_color(target))
            scene.label, scene.state = self.label_and_state(desktop, scan_step, target)
            scene.caption = caption_for(scan_step, step, why, scene.label, scene.state)
            scene.geometry, scene.note, scene.source = self.geometry(desktop, scan_step,
                                                                     target)
        if scene.image is None:
            unreadable = UNREADABLE.format(step=scan_step)
            scene.note = f"{unreadable}；{scene.note}" if scene.note else unreadable
        if scene.frame_hw == (0, 0):
            # no picture: the size the frame row recorded, so an outline still
            # lands where it is on the frame
            row = self._desktop_facts(desktop)["frames"].get(scan_step) or {}
            hw = (row.get("aux") or {}).get("hw")
            if hw and len(hw) == 2:
                scene.frame_hw = (int(hw[0]), int(hw[1]))
        return scene


def read_scaled(path: str, side: int) -> tuple[Optional[QImage], tuple[int, int]]:
    """``(the picture at most side px on its long edge, its own (h, w))``.

    The size is asked for before decoding, so a JPEG is decoded scaled by its
    own codec; a PNG is decoded whole and scaled once.  RGB888 to keep three
    bytes a pixel, not four, in the cache.
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    hw = (size.height(), size.width()) if size.isValid() else (0, 0)
    if size.isValid() and max(size.width(), size.height()) > side:
        reader.setScaledSize(size.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        return None, hw
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    if hw == (0, 0):
        hw = (image.height(), image.width())
    return image, hw
