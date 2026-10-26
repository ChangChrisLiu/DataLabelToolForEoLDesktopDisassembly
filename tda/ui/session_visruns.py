"""「可见性审查」 over runs and across the views, in the session (task U36).

The user (2026-10-26): some 看不见 last many frames, and the same part may be
hidden in one view and seen in another -- the review has to take both into
account. On top of task U24's queue (:mod:`tda.ui.session_visreview`):

* the queue's lines are runs (「第 a–b 帧（n 帧）」, 「全程」 when it is every
  confirmed frame of the part), the part with the longest run first, and the
  annotator can cut a run where its reason changes:
  :meth:`VisRunsMixin.split_visibility_run` (「从这帧起另算」 -- the open frame
  starts a run of its own; kept for as long as the view is open, nothing is
  written);
* :meth:`VisRunsMixin.visibility_summary` -- per part, how long it is 看不见 in
  every view; :meth:`VisRunsMixin.visibility_detail` -- for one run, what
  every view says on the same steps and the reason they point at
  (:func:`~tda.core.vis_runs.suggest`, a default the annotator still answers);
  :meth:`VisRunsMixin.visibility_strip` -- the run's first, middle and last
  frames cut around the part's whole shape;
* :meth:`VisRunsMixin.sync_visibility` -- 「同步到其他视角」: the verdict a run
  was given, copied to the same steps of the other views where the part is
  hidden the same way, nobody has answered yet and nobody wrote anything by
  hand -- each view's own hand work is reviewed in that view
  (:func:`~tda.core.vis_runs.sync_plan`). One op, ``sync_visibility``, with
  its inverse: per view, the frames' stamps and overrides before and after;
  the U15 safeguards of a verdict hold there frame by frame -- a frame that
  already disagreed is not signed, and pixels a label reveals are never
  signed (no frame of another view is on screen).
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from tda.core.model import FrameKey
from tda.core.occluder import shape_applies
from tda.core.truth import VERIFIED
from tda.core.truth_inputs import InputCache
from tda.core.vis_review import VERDICT_COMPUTED, amodal_box
from tda.core.vis_runs import SHAPE_SKIP, cross_view, sample_steps, suggest, sync_plan
from tda.ui import visreview_text as T
from tda.ui.commands import Op
from tda.ui.session_api import SessionRefusal
from tda.ui.session_cable import CABLE_BY, _signed, apply_cable, disputed_before
from tda.ui.session_ops import _write_override, settle
from tda.ui.session_visreview import apply_vis_review

__all__ = ["OP_SYNC", "STRIP_SIZE", "VisRunsMixin", "apply_sync", "crop_strip"]

#: The op 「同步到其他视角」 logs.
OP_SYNC = "sync_visibility"
#: The frame strip's thumbnails fit in this many pixels, both ways.
STRIP_SIZE = 96
#: How many cut thumbnails the session keeps (three per line looked at).
_THUMBS_KEPT = 48


def apply_sync(db, truth, payload: dict, current: Optional[int] = None) -> dict:
    """Both directions of a ``sync_visibility`` op: each view's part, as a verdict's.

    ``payload["views"]`` holds ``{"view", "steps", "frames"}`` per view (and
    :data:`~tda.ui.session_cable.CABLE_BY` where the labels move), each frame
    an :func:`~tda.ui.session_visreview.apply_vis_review` frame. ``current``
    is not passed on: no frame of another view is the one on screen.
    """
    verdict = apply_cable(apply_vis_review)
    out = {"problems": {}, "frame": None, "rechecks": [], "moved": {}, "accepted": {},
           "withheld": {}, "conflicts": 0}
    for part in payload["views"]:
        sub = {"desktop": payload["desktop"], "view": part["view"],
               "instance": payload["instance"], "frames": part["frames"]}
        if part.get(CABLE_BY):
            sub[CABLE_BY] = part[CABLE_BY]
        stats = verdict(db, truth, sub, None)
        out["moved"][part["view"]] = list(stats.get("moved") or [])
        out["accepted"][part["view"]] = list(stats.get("accepted") or [])
        out["withheld"][part["view"]] = list(stats.get("withheld") or [])
        out["conflicts"] += int(stats.get("conflicts") or 0)
    return out


def crop_strip(image: Optional[np.ndarray], box, scale: float = 1.0,
               size: int = STRIP_SIZE) -> Optional[np.ndarray]:
    """One thumbnail: ``image`` cut around ``box`` (full-size coordinates), framed.

    ``scale`` is the image's size over the frame's (a reduced decode). The cut
    is the box with a margin of half its larger side (48 frame pixels at
    least: a 25 px latch needs what is around it to be judged), fitted into
    ``size`` x ``size``; the box is drawn in it, so a part with nothing
    visible still shows where it is. No box: the whole frame.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = (float(v) * scale for v in box)
        margin = max(48.0 * scale, 0.5 * max(x1 - x0, y1 - y0))
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1, cy1 = min(w, int(np.ceil(x1 + margin))), min(h, int(np.ceil(y1 + margin)))
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, w, h
    else:
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    cut = image[cy0:cy1, cx0:cx1]
    fit = min(size / max(1, cut.shape[1]), size / max(1, cut.shape[0]))
    out_w, out_h = max(1, int(round(cut.shape[1] * fit))), max(1, int(round(cut.shape[0] * fit)))
    small = cv2.resize(cut, (out_w, out_h),
                       interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_LINEAR)
    small = np.ascontiguousarray(small)
    if box is not None:
        bx0 = int((x0 - cx0) * fit)
        by0 = int((y0 - cy0) * fit)
        bx1 = int(np.ceil((x1 - cx0) * fit)) - 1
        by1 = int(np.ceil((y1 - cy0) * fit)) - 1
        cv2.rectangle(small, (max(0, bx0), max(0, by0)),
                      (min(out_w - 1, max(bx0, bx1)), min(out_h - 1, max(by0, by1))),
                      (255, 200, 0), 1)
    return small


class VisRunsMixin:
    """Runs, the other views, the split and the sync, for the session (U36)."""

    # -- reads -----------------------------------------------------------------
    def visibility_summary(self) -> list[dict]:
        """Per part of the queue, in its order: 看不见 frames / stretches in every view."""
        if not self.is_open or self.gallery:
            return []
        return self.review.visibility_summary()

    def visibility_detail(self, entry: Optional[dict]) -> dict:
        """What every view says about ``entry``'s part on its steps, and the suggestion.

        ``{"views": [...]`` (:func:`~tda.core.vis_runs.cross_view`)``,
        "suggestion": {...} | None}`` (:func:`~tda.core.vis_runs.suggest`).
        """
        if not self.is_open or self.gallery or not entry or not entry.get("steps"):
            return {"views": [], "suggestion": None}
        views = cross_view(entry, self.review.visibility_sides(), self.view,
                           self.review.names_in)
        return {"views": views, "suggestion": suggest(entry, views, self.view)}

    def visibility_strip(self, instance: str, steps: Iterable[int],
                         size: int = STRIP_SIZE) -> list[dict]:
        """``[{"step", "image"}]``: the run's first, middle and last frames, cut.

        Around the part's whole shape there (:func:`~tda.core.vis_review.amodal_box`);
        ``image`` is RGB, at most ``size`` px a side, or ``None`` when the
        frame has no image. Decoded frames are not kept (the canvas's cache is
        left alone); the cut ones are.
        """
        if not self.is_open:
            return []
        kept = self.__dict__.setdefault("_vis_thumbs", {})
        cache = InputCache()
        out = []
        for step in sample_steps(steps):
            key = FrameKey(self.desktop, int(step), self.view)
            row = self.db.compiled_label(key, str(instance)) or {}
            frame = self.db.get_frame(key) or {}
            hw = (frame.get("aux") or {}).get("hw")
            image, scale = None, 1.0
            if not hw:
                image, scale = self._strip_image(key, None)
                hw = None if image is None else image.shape[:2]
            if not hw:
                out.append({"step": int(step), "image": None})
                continue
            hw = tuple(int(v) for v in hw)
            box = amodal_box(self.db, key, str(instance), str(row.get("placement") or ""),
                             hw, cache)
            memo = (key.view, key.step, str(instance), box, int(size))
            if memo not in kept:
                if image is None:
                    image, scale = self._strip_image(key, hw)
                if image is None:
                    out.append({"step": int(step), "image": None})
                    continue
                if len(kept) >= _THUMBS_KEPT:
                    kept.pop(next(iter(kept)))
                kept[memo] = crop_strip(image, box, scale, size)
            out.append({"step": int(step), "image": kept[memo]})
        return out

    def _strip_image(self, key: FrameKey, hw) -> tuple[Optional[np.ndarray], float]:
        """The frame's pixels for a thumbnail, and their scale over the frame's size.

        The canvas's decoded frame when it has one; otherwise read here --
        halved for a large JPEG, which decodes in a quarter of the time.
        """
        held = self.images.peek(key)
        if held is not None:
            return held, 1.0
        path = self.images.image_path(key)
        if path is None:
            return None, 1.0
        large = hw is not None and max(int(v) for v in hw) > 2000
        flag = (cv2.IMREAD_REDUCED_COLOR_2
                if large and str(path).lower().endswith((".jpg", ".jpeg"))
                else cv2.IMREAD_COLOR)
        bgr = cv2.imread(str(path), flag)
        if bgr is None:
            return None, 1.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        scale = 1.0 if not hw else rgb.shape[1] / float(int(hw[1]))
        return rgb, scale

    # -- 从这帧起另算 ---------------------------------------------------------------
    def split_visibility_run(self, instance: str, step: int, steps: Iterable[int]) -> dict:
        """Cut the run ``steps`` of ``instance`` before ``step``: two lines from now on.

        Nothing is written: the cut is the annotator's, for as long as the view
        is open. On the run's first frame, a cut made there is taken back (the
        run joins the one before it, when they say the same thing). Refused
        when ``step`` is not in the run, and on its first frame with no cut.
        """
        if not self.is_open:
            raise SessionRefusal("no view is open")
        values = sorted({int(s) for s in steps})
        step = int(step)
        if step not in values:
            raise SessionRefusal(T.SPLIT_NOT_IN_RUN.format(step=step))
        key = (str(instance), step)
        if step == values[0]:
            if key not in self.review.vis_splits:
                raise SessionRefusal(T.SPLIT_AT_START.format(step=step))
            self.review.vis_splits.discard(key)
            return {"merged": True, "step": step}
        self.review.vis_splits.add(key)
        return {"merged": False, "step": step, "steps": [s for s in values if s >= step]}

    # -- 同步到其他视角 ---------------------------------------------------------------
    def sync_visibility_plan(self, instance: str, steps: Iterable[int]) -> dict:
        """What :meth:`sync_visibility` would write, without writing (:func:`sync_plan`)."""
        if not self.is_open or self.gallery:
            return {"source": {}, "views": {}, "count": 0, "skipped": {},
                    "refused": "no_verdict"}
        cache = InputCache()

        def shape_why(key: FrameKey, part: str) -> str:
            return shape_applies(self.db, self.tax, key, part, cache)

        def covered(view: str, step: int, part: str, row: dict) -> bool:
            # a part or any occluder layer (a hand, a tool, a cable layer) over
            # it there -- as :func:`suggest` reads them (round 2, B2)
            return bool(self.review.names_in(view, step, part, row))

        return sync_plan(self.db, self.desktop, self.view, str(instance), steps, shape_why,
                         covered)

    def _why_not_synced(self, key: FrameKey, instance: str) -> str:
        """Which :data:`~tda.core.vis_runs.SKIP_REASONS` a frame the write refused falls under.

        The plan has already sorted the frames; this is for one whose state
        moved between the plan and the write (round 2, N7).
        """
        if (self.db.get_frame(key) or {}).get("review_status") != VERIFIED:
            return "unconfirmed"
        if self.db.compiled_label(key, instance) is None:
            return "absent"
        flags = self.db.instance_flags(key, instance) or {}
        if flags.get("vis_reviewed"):
            return "verdict"
        if self.db.frame_overrides(key).get(instance) is not None:
            return "manual"
        return SHAPE_SKIP.get(shape_applies(self.db, self.tax, key, instance), "no_shape")

    def sync_visibility(self, instance: str, steps: Iterable[int]) -> dict:
        """「同步到其他视角」: the run's verdict onto the same steps of the other views.

        Only the frames :func:`~tda.core.vis_runs.sync_plan` lets through --
        hidden the same way there, no verdict yet, nothing a person wrote there
        (no override row: each view's hand work is reviewed in that view).
        **One op** (``sync_visibility``), one ``Ctrl+Z``; the other views'
        confirmed frames stay confirmed as a verdict leaves them (a frame
        already in dispute is not signed, revealed pixels never are).

        Returns ``{"op", "count", "written": {view: steps}, "skipped",
        "accepted", "withheld", "conflicts"}``; refused (nothing written) when
        the run has no verdict yet, a label that is not hidden, or nothing to
        copy to.
        """
        if not self.is_open:
            raise SessionRefusal("no view is open")
        if self.gallery:
            raise SessionRefusal(T.NOT_IN_GALLERY)
        instance = str(instance)
        plan = self.sync_visibility_plan(instance, steps)
        if plan["refused"]:
            raise SessionRefusal(T.SYNC_REFUSED[plan["refused"]])
        if not plan["count"]:
            raise SessionRefusal(T.sync_nothing(plan["skipped"]))
        key = self.current()
        cache = InputCache()
        parts, backs = [], []
        skipped = dict(plan["skipped"])
        for view, chosen in sorted(plan["views"].items()):
            shapes = self.db.keyframes(key.desktop, view, instance)
            frames, before = [], []
            for step, verdict in sorted(chosen.items()):
                here = FrameKey(key.desktop, int(step), view)
                label = None if verdict["verdict"] == VERDICT_COMPUTED else verdict["visibility"]
                try:
                    state, back = self._verdict_frame(here, instance, verdict["verdict"],
                                                      label, cache, shapes)
                except SessionRefusal:
                    why = self._why_not_synced(here, instance)
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                frames.append(state)
                before.append(back)
            if not frames:
                continue
            part = {"view": view, "steps": [f["step"] for f in frames], "frames": frames}
            back_part = {"view": view, "steps": list(part["steps"]), "frames": before}
            if any("override" in f for f in frames):
                part[CABLE_BY] = back_part[CABLE_BY] = self.annotator
            parts.append(part)
            backs.append(back_part)
        if not parts:
            raise SessionRefusal(T.sync_nothing(skipped))
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": instance, "steps": sorted(plan["source"])}
        payload = common | {"views": parts}
        inverse = common | {"views": backs}
        op = Op(OP_SYNC, payload, inverse)
        moved = {p["view"]: [f["step"] for f in p["frames"] if "override" in f] for p in parts}
        disputed = {view: disputed_before(self.db, self.truth, key.desktop, view, instance,
                                          steps_)
                    for view, steps_ in moved.items() if steps_}
        with self.db.transaction():
            for part in parts:
                for state in part["frames"]:
                    here = FrameKey(key.desktop, int(state["step"]), part["view"])
                    if "override" in state:
                        _write_override(self.db, here, instance, state["override"])
                    self.db.set_instance_flags(here, instance, state["flags"])
            self.db.log_op(key.desktop, key.view, OP_SYNC, payload, inverse, self.annotator)
        out = {"op": op, "count": sum(len(p["frames"]) for p in parts),
               "written": {p["view"]: list(p["steps"]) for p in parts},
               "skipped": skipped, "accepted": {}, "withheld": {}, "conflicts": 0}
        for view, steps_ in moved.items():
            if not steps_:
                continue
            stats = settle(self.db, self.truth, key.desktop, view, steps_, None)
            stats = _signed(self.db, self.truth, key.desktop, view, instance, steps_,
                            self.annotator, disputed.get(view, set()), stats, None)
            out["accepted"][view] = list(stats.get("accepted") or [])
            out["withheld"][view] = list(stats.get("withheld") or [])
            out["conflicts"] += int(stats.get("conflicts") or 0)
        self.undo_stack.push(op, apply=False)
        self.review.invalidate()
        self._refresh_dirty()
        self._announce()
        return out
