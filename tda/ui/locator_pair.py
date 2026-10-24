"""「拆前 / 拆后」: the scanner frames just before and just after a part leaves (task U28).

The annotation walk runs backwards, so the first frame a part is met on is the
one just before it was taken out -- and a log line like "CPU heatsink screw 3"
does not say *which* screw that is.  The 「主视角」 dock (task U23) therefore
has a second kind of scene: the two scanner frames around the step the part
leaves at, cropped to the same window around what changed, with what the log
says about that step underneath.  :class:`PairMaker` builds it as a
:class:`PairScene`; :mod:`tda.ui.panels.locator_pair` draws it and
:mod:`tda.ui.app_locator` decides when it is shown instead of the location.

* **Which step.**  The part's compiled state timeline (the derived events of
  the action table plus the hand-written ones, with the cascade -- a captive
  screw leaves with its heatsink, a clip with its shroud -- folded in by
  :func:`tda.core.states.state_at`): the first step its state becomes
  ``removed`` or ``displaced``; when it never leaves, the first step its state
  changes at all; when it never changes there is no pair
  (:data:`NO_CHANGE`).  :func:`leaving_steps` answers it for every part of a
  machine at once.
* **Which frames.**  The scanner frames of ``r - 1`` and ``r``, through the
  locator's own :meth:`~tda.ui.locator_source.LocatorSource.scan_step_for`
  (the nearest earlier frame when a step has none, said in the caption).
* **Which window.**  The part's own outline on the "before" frame when the
  scanner view has one (the stored row, or the frame compiled once -- the
  locator's :meth:`~tda.ui.locator_source.LocatorSource.geometry`), else the
  largest region that differs between the two pictures
  (:func:`change_window`) -- only inside one scanner pose segment: across a
  pose break the two frames are not comparable and are shown whole
  (:data:`POSE_BREAK`), as they are when nothing differs (:data:`NO_DIFF`).

Costs: the pictures are the locator's thumbnails (its LRU), the difference is
taken on them at :data:`DIFF_SIDE` px (a few ms) and remembered by
``(machine, the two frames, the ROI)``; the timeline is one fold per step that
has an event, remembered per machine until the scanner inputs or the action
table move; a scene is remembered by part and rebuilt only when they do.
Nothing here writes to the database.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage

from tda.core.model import FrameKey, is_context_key, is_provisional
from tda.core.states import REMOVED, state_at
from tda.core.truth_inputs import events_of, pose_segment_of
from tda.ui.canvas.overlay import palette_color
from tda.ui.class_names import class_zh, instance_label, state_zh
from tda.ui.locator_source import (
    EARLIER_DUPLI,
    EARLIER_MISSING,
    NO_SCAN_FRAME,
    SCAN,
    UNREADABLE,
    Geometry,
)

__all__ = [
    "AFTER", "BEFORE", "BEFORE_NONE", "DIFF_SIDE", "DISPLACED", "LEAVING_STATES", "LOG_LINE", "Leaving",
    "NO_BEFORE", "NO_CHANGE", "NO_DIFF", "PAD", "PAIR_HEAD", "PAIR_HEAD_FIRST", "POSE_BREAK",
    "PairMaker", "PairScene", "WITH_PARENT", "MIN_SIDE_FRAC", "change_window",
    "leaving_steps", "qimage_rgb",
]

DISPLACED = "displaced"
#: The states that take a part away: the first step one of them is reached is
#: the step the pair is about.
LEAVING_STATES = (REMOVED, DISPLACED)

# --- what the panel says ----------------------------------------------------
#: The panel's caption: the pair shown because the part is met for the first
#: time in this view, or because the annotator asked for it.
PAIR_HEAD_FIRST = "拆前 / 拆后 · 第一次标这个零件"
PAIR_HEAD = "拆前 / 拆后"
#: Above the two pictures (with U23's earlier-frame clause when it applies).
BEFORE = "拆前 第 {step} 帧{earlier}"
AFTER = "拆后 第 {step} 帧{earlier}"
#: ... when the part changes on the first step: there is no frame before it.
BEFORE_NONE = "拆前"
#: What the log says at the step: its raw name, and the verb.
LOG_LINE = "第 {step} 帧：{name}"
#: A part that went out inside another one (the cascade).
WITH_PARENT = "随 {parent} 一起拆走"
#: The notes under the pictures.
NO_CHANGE = "这个零件没有离开或改变状态的步骤"
POSE_BREAK = "两帧不在同一机位，无法对比"
NO_DIFF = "两帧看不出差别"
NO_BEFORE = "第 {step} 帧之前扫描仪没有图"

# --- the window ---------------------------------------------------------------
#: The window is padded this much (each side grows by a fifth of it) ...
PAD = 1.6
#: ... and the crop is never smaller than this share of the frame's long side.
MIN_SIDE_FRAC = 1.0 / 6.0
#: The difference is taken at no more than this many pixels on the long side.
DIFF_SIDE = 512
#: The blur before the difference (sensor noise, a hair of misregistration).
DIFF_BLUR = 5
DIFF_SIGMA = 1.5
#: A pixel differs when one channel moved by more than this (0-255).
DIFF_LEVEL = 30
#: The largest region must cover this share of the picture, or "no difference".
DIFF_FLOOR = 0.0002
DIFF_FLOOR_PX = 4
#: With no ROI, a margin this wide (share of the long side) is ignored: the
#: scanner's frame edge flickers with whatever is beside the table.
BORDER = 0.02
#: An ROI is widened by this share of the frame before it limits the difference.
ROI_PAD = 0.05
#: Remembered: scenes by part, differences by frame pair.
PAIR_CACHE = 64
DIFF_CACHE = 64


# --------------------------------------------------------------------------- #
# the step
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leaving:
    """The step a part leaves (or first changes) at, and how."""

    step: int
    #: ``removed`` / ``displaced``, or ``changed`` for a part that never leaves.
    kind: str
    old: str
    new: str
    #: The parent it went out inside of (the cascade), else ``None``.
    left_with: Optional[str] = None


def leaving_steps(instances: dict, events: Iterable, tax) -> dict[str, Leaving]:
    """``key -> Leaving`` for every part of a machine whose state ever changes.

    The state at each step that has an event is :func:`~tda.core.states.
    state_at`'s -- the folded log with the attached-child cascade closed over
    it, so a captive screw that rides out with its heatsink leaves at the
    heatsink's step, ``left_with`` the heatsink.  The first step a part reaches
    one of :data:`LEAVING_STATES` wins over an earlier change (a screw loosened
    at 4 and carried out at 8 leaves at 8); a part that never leaves has its
    first change of state.  Placement alone is not a change.
    """
    ordered = sorted(events, key=lambda e: e.step)
    steps = sorted({int(e.step) for e in ordered if int(e.step) >= 1})
    previous = state_at(instances, ordered, 0, tax)
    first: dict[str, Leaving] = {}
    left: dict[str, Leaving] = {}
    for step in steps:
        now = state_at(instances, ordered, step, tax)
        for key, inst in now.items():
            if key in left or key not in instances:
                continue
            before = previous.get(key)
            old = before.state if before is not None else ""
            if old == inst.state:
                continue
            if inst.state in LEAVING_STATES:
                left[key] = Leaving(step, inst.state, old, inst.state, inst.left_with)
            elif key not in first:
                first[key] = Leaving(step, "changed", old, inst.state, inst.left_with)
        previous = now
    out = dict(first)
    out.update(left)
    return out


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
def qimage_rgb(image: Optional[QImage]) -> Optional[np.ndarray]:
    """A view of an RGB888 :class:`QImage` as an ``(h, w, 3)`` array (no copy)."""
    if image is None or image.isNull():
        return None
    if image.format() != QImage.Format.Format_RGB888:
        image = image.convertToFormat(QImage.Format.Format_RGB888)
    h, w, line = image.height(), image.width(), image.bytesPerLine()
    flat = np.frombuffer(image.constBits(), np.uint8, image.sizeInBytes())
    return flat.reshape(h, line)[:, :w * 3].reshape(h, w, 3)


def change_window(before: np.ndarray, after: np.ndarray,
                  keep: Optional[tuple] = None) -> Optional[tuple[float, float, float, float]]:
    """The largest region that differs between two pictures, or ``None``.

    Both are shrunk to :data:`DIFF_SIDE` (never enlarged), blurred a little,
    and a pixel differs when a channel moved by more than :data:`DIFF_LEVEL`;
    specks are opened away, nearby pieces closed together, and the largest
    connected region wins -- unless it is smaller than :data:`DIFF_FLOOR` of
    the picture.  ``keep`` limits the search to ``(x0, y0, x1, y1)`` given as
    fractions of the picture (the scanner ROI); without it a :data:`BORDER`
    margin is left out.  The answer is in fractions too, so it lands on the
    frame at any resolution.
    """
    if before is None or after is None or before.size == 0 or after.size == 0:
        return None
    h, w = before.shape[:2]
    scale = min(1.0, DIFF_SIDE / float(max(h, w)))
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    a = cv2.resize(before, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else before
    b = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (DIFF_BLUR, DIFF_BLUR), DIFF_SIGMA)
    b = cv2.GaussianBlur(b, (DIFF_BLUR, DIFF_BLUR), DIFF_SIGMA)
    diff = cv2.absdiff(a, b)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    sh, sw = diff.shape
    region = np.zeros((sh, sw), dtype=bool)
    if keep is not None:
        x0, y0, x1, y1 = keep
        region[max(0, int(y0 * sh)):min(sh, int(np.ceil(y1 * sh))),
               max(0, int(x0 * sw)):min(sw, int(np.ceil(x1 * sw)))] = True
    else:
        margin = int(round(BORDER * max(sh, sw)))
        region[margin:sh - margin, margin:sw - margin] = True
    binary = ((diff > DIFF_LEVEL) & region).astype(np.uint8)
    if not binary.any():
        return None
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = (int(v) for v in stats[best])
    if area < max(DIFF_FLOOR_PX, DIFF_FLOOR * sh * sw):
        return None
    return (x / sw, y / sh, (x + bw) / sw, (y + bh) / sh)


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
@dataclass
class PairScene:
    """Everything the panel draws for one part's 「拆前 / 拆后」."""

    desktop: Optional[int] = None
    instance: Optional[str] = None
    label: str = ""
    colour: tuple[int, int, int] = (255, 210, 0)
    leaving: Optional[Leaving] = None
    #: ``(r - 1, r)``: the steps asked for.
    asked: tuple = (None, None)
    #: The scanner frames shown for them (an earlier one when a step has none).
    steps: tuple = (None, None)
    #: The two pictures at the locator's thumbnail size, ``None`` when unread.
    images: tuple = (None, None)
    #: Each frame's own size, ``(h, w)``: what the window is in.
    frame_hw: tuple = ((0, 0), (0, 0))
    captions: tuple = ("", "")
    #: The window, ``(x0, y0, x1, y1)`` in the before frame's pixels; ``None``
    #: means the two frames are shown whole.
    window: Optional[tuple] = None
    #: ``part`` (the scanner outline) / ``diff`` (what changed) / ``""``.
    window_source: str = ""
    #: The part's own outline on the before frame, when the window is it.
    geometry: Optional[Geometry] = None
    #: The part, with its change of state: 「理线夹 cable_clip.01（关上 → 打开）」.
    label_line: str = ""
    #: What the log says at ``r``: 「第 8 帧：Heatsink · remove」.
    log_line: str = ""
    #: 「随 散热片 一起拆走」 for a part that went out inside another one.
    with_line: str = ""
    #: The sentence under the pictures: why they are whole, or ``""``.
    note: str = ""
    roi: Optional[tuple] = None

    def lines(self) -> list[str]:
        """The text under the pictures, top to bottom, without the note."""
        return [line for line in (self.label_line, self.log_line, self.with_line) if line]


def _earlier(asked: Optional[int], shown: Optional[int], why: str) -> str:
    if asked is None or shown is None or shown == asked or not why:
        return ""
    return (EARLIER_DUPLI if why == "dupli" else EARLIER_MISSING).format(asked=int(asked))


class PairMaker:
    """The 「拆前 / 拆后」 half of :class:`~tda.ui.locator_source.LocatorSource`."""

    def __init__(self, source) -> None:
        self.source = source
        #: desktop -> (digest, {"leaving", "names", "verbs"})
        self._timelines: dict[int, tuple[tuple, dict]] = {}
        #: (desktop, instance) -> (digest, PairScene without its pictures)
        self._pairs: "OrderedDict[tuple, tuple]" = OrderedDict()
        #: (desktop, before step, after step, keep) -> window as fractions | None
        self._diffs: "OrderedDict[tuple, Optional[tuple]]" = OrderedDict()

    # -- what moves a scene ---------------------------------------------------
    def _digest(self, desktop: int, facts: dict) -> tuple:
        """The scanner inputs' digest plus what else a pair is made of.

        The action table (the timeline, the log's verbs) and the scanner's pose
        segments (which two frames may be compared, and the ROI the difference
        is looked for in).
        """
        row = self.source.db.conn.execute(
            "SELECT (SELECT COUNT(*) FROM action WHERE desktop=?), "
            "(SELECT IFNULL(GROUP_CONCAT(step || ':' || idx || ':' || target || ':' || verb "
            "  || ':' || result, '|'), '') FROM action WHERE desktop=?), "
            "(SELECT IFNULL(GROUP_CONCAT(seg || ':' || start_step || ':' || end_step || ':' "
            "  || IFNULL(roi_json, ''), '|'), '') FROM pose_segment WHERE desktop=? AND view=?), "
            "(SELECT IFNULL(SUM(IFNULL(pose_segment, 0) * step), 0) FROM frame "
            "  WHERE desktop=? AND view=?)",
            (int(desktop), int(desktop), int(desktop), SCAN, int(desktop), SCAN)).fetchone()
        text = f"{row[1]}#{row[2]}#{row[3]}"
        inputs = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        return (facts.get("digest"), int(row[0]), inputs)

    def timeline(self, desktop: int, digest: tuple) -> dict:
        """The machine's leaving steps, step names and verbs (re-read when ``digest`` moves)."""
        held = self._timelines.get(int(desktop))
        if held is not None and held[0] == digest:
            return held[1]
        db, tax = self.source.db, self.source.tax
        instances = db.instances(int(desktop))
        try:
            leaving = leaving_steps(instances, events_of(db, tax, int(desktop)), tax)
        except Exception:  # noqa: BLE001 - a side panel is never worth a failure
            leaving = {}
        names = {int(s.step): str(s.raw_name or "").strip() for s in db.steps(int(desktop))}
        verbs: dict[tuple[int, str], str] = {}
        for action in sorted(db.actions(int(desktop)), key=lambda a: (a.step, a.idx)):
            if action.result == "success":
                verbs.setdefault((int(action.step), str(action.target)), str(action.verb))
        found = {"leaving": leaving, "names": names, "verbs": verbs, "instances": instances}
        self._timelines[int(desktop)] = (digest, found)
        return found

    # -- the scene ------------------------------------------------------------
    def pair(self, desktop: int, instance: str) -> PairScene:
        """The 「拆前 / 拆后」 of one part, built once per digest; pictures attached fresh."""
        desktop = int(desktop)
        source = self.source
        facts = source._desktop_facts(desktop)
        digest = self._digest(desktop, facts)
        slot = (desktop, str(instance))
        held = self._pairs.get(slot)
        if held is not None and held[0] == digest:
            self._pairs.move_to_end(slot)
            scene = held[1]
        else:
            scene = self._build(desktop, str(instance), digest)
            source.stats["pairs"] += 1
            self._pairs[slot] = (digest, scene)
            while len(self._pairs) > PAIR_CACHE:
                self._pairs.popitem(last=False)
        return self._with_pictures(scene)

    def _with_pictures(self, scene: PairScene) -> PairScene:
        """A copy of ``scene`` holding the two pictures, from the thumbnail cache."""
        images, hws = [], []
        for step in scene.steps:
            if step is None:
                images.append(None)
                hws.append((0, 0))
                continue
            image, hw = self.source.thumbnail(scene.desktop, step)
            images.append(image)
            hws.append(hw if hw != (0, 0) else self._recorded_hw(scene.desktop, step))
        return replace(scene, images=tuple(images), frame_hw=tuple(hws))

    def _recorded_hw(self, desktop: int, step: int) -> tuple[int, int]:
        row = self.source._desktop_facts(desktop)["frames"].get(int(step)) or {}
        hw = (row.get("aux") or {}).get("hw")
        return (int(hw[0]), int(hw[1])) if hw and len(hw) == 2 else (0, 0)

    def _build(self, desktop: int, instance: str, digest: tuple) -> PairScene:
        source = self.source
        found = self.timeline(desktop, digest)
        rec = found["instances"].get(instance)
        scene = PairScene(desktop=desktop, instance=instance)
        scene.label = instance_label(instance, getattr(rec, "cls", None),
                                     getattr(rec, "attrs", None))
        scene.colour = tuple(int(c) for c in palette_color(instance))
        scene.label_line = scene.label
        leaving = found["leaving"].get(instance)
        if rec is None or is_context_key(instance) or is_provisional(instance) or leaving is None:
            scene.note = NO_CHANGE
            return scene
        scene.leaving = leaving
        scene.label_line = (f"{scene.label}（{state_zh(leaving.old)} → "
                            f"{state_zh(leaving.new)}）")
        r = int(leaving.step)
        scene.log_line = self._log_line(found, r, instance, leaving)
        if leaving.left_with:
            parent = found["instances"].get(leaving.left_with)
            scene.with_line = WITH_PARENT.format(parent=class_zh(
                getattr(parent, "cls", None), getattr(parent, "attrs", None),
                leaving.left_with))
        # the two scanner frames
        before, why_before = (source.scan_step_for(desktop, r - 1) if r - 1 >= 1
                              else (None, ""))
        after, why_after = source.scan_step_for(desktop, r)
        scene.asked = (r - 1, r)
        scene.steps = (before, after)
        scene.captions = (
            BEFORE.format(step=before, earlier=_earlier(r - 1, before, why_before))
            if before is not None else (BEFORE.format(step=r - 1, earlier="") if r > 1
                                        else BEFORE_NONE),
            AFTER.format(step=after, earlier=_earlier(r, after, why_after))
            if after is not None else AFTER.format(step=r, earlier=""),
        )
        notes = []
        if after is None:
            notes.append(NO_SCAN_FRAME.format(asked=r))
        elif before is None:
            notes.append(NO_BEFORE.format(step=r))
        if before is not None:
            scene.roi = source.roi(desktop, before)
        if before is not None and after is not None:
            note = self._window(scene, desktop, instance, before, after)
            if note:
                notes.append(note)
        scene.note = "；".join(notes)
        return scene

    def _log_line(self, found: dict, r: int, instance: str, leaving: Leaving) -> str:
        name = found["names"].get(r, "")
        line = LOG_LINE.format(step=r, name=name) if name else f"第 {r} 帧"
        verb = found["verbs"].get((r, instance))
        if verb is None and leaving.left_with:
            verb = found["verbs"].get((r, str(leaving.left_with)))
        return f"{line} · {verb}" if verb else line

    def _window(self, scene: PairScene, desktop: int, instance: str,
                before: int, after: int) -> str:
        """Fill the window in; the note when the frames are shown whole instead."""
        source = self.source
        db = source.db
        if (pose_segment_of(db, FrameKey(desktop, before, SCAN))
                != pose_segment_of(db, FrameKey(desktop, after, SCAN))):
            geometry, _why, _where = source.geometry(desktop, before, instance)
            scene.geometry = geometry
            return POSE_BREAK
        geometry, _why, _where = source.geometry(desktop, before, instance)
        if geometry is not None:
            scene.geometry = geometry
            scene.window = tuple(float(v) for v in geometry.box)
            scene.window_source = "part"
            return ""
        a, hw_a = source.thumbnail(desktop, before)
        b, hw_b = source.thumbnail(desktop, after)
        if a is None or b is None:
            unread = [s for s, img in ((before, a), (after, b)) if img is None]
            return "；".join(UNREADABLE.format(step=s) for s in unread)
        keep = self._keep(scene.roi, hw_a)
        slot = (desktop, before, after, keep)
        if slot in self._diffs:
            self._diffs.move_to_end(slot)
            frac = self._diffs[slot]
        else:
            frac = change_window(qimage_rgb(a), qimage_rgb(b), keep)
            source.stats["diffs"] += 1
            self._diffs[slot] = frac
            while len(self._diffs) > DIFF_CACHE:
                self._diffs.popitem(last=False)
        if frac is None:
            return NO_DIFF
        h, w = hw_a
        scene.window = (frac[0] * w, frac[1] * h, frac[2] * w, frac[3] * h)
        scene.window_source = "diff"
        return ""

    @staticmethod
    def _keep(roi: Optional[tuple], hw: tuple[int, int]) -> Optional[tuple]:
        """The scanner ROI as fractions of the frame, widened by :data:`ROI_PAD`."""
        h, w = hw
        if roi is None or not h or not w:
            return None
        x0, y0, x1, y1 = roi
        return (round(max(0.0, x0 / w - ROI_PAD), 4), round(max(0.0, y0 / h - ROI_PAD), 4),
                round(min(1.0, x1 / w + ROI_PAD), 4), round(min(1.0, y1 / h + ROI_PAD), 4))
