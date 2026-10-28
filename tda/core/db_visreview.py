"""The per-(frame, part) flags and the rows the visibility review reads (task U24).

``instance_frame_flags`` has been in the schema since version 1 -- one JSON
object per ``(desktop, step, view, instance)``, spec 3.1's ``InstanceFrameFlags``
-- and nothing wrote it.  Review's 「可见性审查」 keeps its stamp there
(:mod:`tda.core.vis_review`): the object gains ``vis_reviewed`` / ``by`` /
``at`` / ``verdict`` and what was reviewed, and any other key in it is left
alone.  No migration: the table and the column are as old as the file.

:meth:`VisReviewRowsMixin.vis_review_rows` is the one read behind the queue and
behind ``status``: every row of a confirmed frame whose label is not plain
``visible`` or that carries a label (or a reason) set by hand, with its hand
label, its reason and its flags, in one query -- no compilation.
"""
from __future__ import annotations

from typing import Optional

from tda.core import dbrows as R
from tda.core.model import CONTEXT_PREFIX, FrameKey, Visibility

__all__ = ["VisReviewRowsMixin"]

#: The candidate rows of one view: confirmed frames, the machine's parts only
#: (a 框外零件, ``ctx.*``, is not audited), and a label that is not plain
#: ``visible`` or one set by hand.  ``ORDER BY`` part, then step: the queue's order.
_ROWS = (
    "SELECT c.step AS step, c.instance AS instance, c.visibility AS visibility, "
    "c.placement AS placement, c.input_hash AS input_hash, c.geom_type AS geom_type, "
    "o.visibility AS hand_visibility, o.occluder AS occluder, "
    # the override keeps pixels of its own: a hand out_of_view is 画框外 only
    # without them (task U39 round 3, :func:`tda.core.vis_review.off_frame_row`)
    "o.visible_rle_json IS NOT NULL AS hand_pixels, "
    "g.difficulty_flags_json AS flags_json "
    "FROM compiled_mask c "
    "JOIN frame f ON f.desktop = c.desktop AND f.step = c.step AND f.view = c.view "
    "LEFT JOIN frame_override o ON o.desktop = c.desktop AND o.step = c.step "
    "AND o.view = c.view AND o.instance = c.instance "
    "LEFT JOIN instance_frame_flags g ON g.desktop = c.desktop AND g.step = c.step "
    "AND g.view = c.view AND g.instance = c.instance "
    "WHERE c.desktop = ? AND c.view = ? AND f.review_status = 'verified' "
    f"AND c.instance NOT LIKE '{CONTEXT_PREFIX}%' "
    f"AND (COALESCE(c.visibility, '') != '{Visibility.VISIBLE.value}' "
    "OR o.visibility IS NOT NULL OR o.occluder IS NOT NULL) "
    "ORDER BY c.instance, c.step"
)

#: Every row of one view's confirmed frames, ``visible`` ones included (task
#: U36): what the other views say about a part on the same steps, and on which
#: frames a part is there at all. No masks are read.
_ALL_ROWS = (
    "SELECT c.step AS step, c.instance AS instance, c.visibility AS visibility, "
    "c.placement AS placement, c.input_hash AS input_hash, "
    "o.visibility AS hand_visibility, o.occluder AS occluder, "
    # any override row -- a label, a reason, or a mask drawn by hand for this
    # frame only: what a person wrote there (U36 round 2, B1)
    "o.instance IS NOT NULL AS has_override, "
    "o.visible_rle_json IS NOT NULL AS hand_pixels, "
    "g.difficulty_flags_json AS flags_json "
    "FROM compiled_mask c "
    "JOIN frame f ON f.desktop = c.desktop AND f.step = c.step AND f.view = c.view "
    "LEFT JOIN frame_override o ON o.desktop = c.desktop AND o.step = c.step "
    "AND o.view = c.view AND o.instance = c.instance "
    "LEFT JOIN instance_frame_flags g ON g.desktop = c.desktop AND g.step = c.step "
    "AND g.view = c.view AND g.instance = c.instance "
    "WHERE c.desktop = ? AND c.view = ? AND f.review_status = 'verified' "
    f"AND c.instance NOT LIKE '{CONTEXT_PREFIX}%' "
    "ORDER BY c.instance, c.step"
)


class VisReviewRowsMixin:
    """``instance_frame_flags`` and the visibility review's rows, for ``Db``."""

    def instance_flags(self, key: FrameKey, instance: str) -> Optional[dict]:
        """The flags object of one ``(frame, part)``, or ``None`` when it has no row."""
        row = self.conn.execute(
            "SELECT difficulty_flags_json FROM instance_frame_flags "
            "WHERE desktop=? AND step=? AND view=? AND instance=?",
            (key.desktop, key.step, key.view, str(instance)),
        ).fetchone()
        if row is None:
            return None
        found = R.loads(row["difficulty_flags_json"])
        return dict(found) if isinstance(found, dict) else {}

    def instance_flags_at(self, key: FrameKey) -> dict[str, dict]:
        """``part -> flags object`` of every part of one frame that has a row."""
        rows = self.conn.execute(
            "SELECT instance, difficulty_flags_json FROM instance_frame_flags "
            "WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            found = R.loads(row["difficulty_flags_json"])
            out[str(row["instance"])] = dict(found) if isinstance(found, dict) else {}
        return out

    def set_instance_flags(self, key: FrameKey, instance: str, flags: Optional[dict]) -> None:
        """Make the ``(frame, part)`` flags exactly ``flags``; ``None`` removes the row.

        The whole object, not a merge: the caller read it, changed its keys and
        hands it back, and an undo hands back what was there before -- byte
        for byte, other keys included.
        """
        if flags is None:
            with self._tx():
                self.conn.execute(
                    "DELETE FROM instance_frame_flags "
                    "WHERE desktop=? AND step=? AND view=? AND instance=?",
                    (key.desktop, key.step, key.view, str(instance)))
            return
        self._upsert("instance_frame_flags", self._fk(key) | {"instance": str(instance)},
                     {"difficulty_flags_json": R.dumps(dict(flags))}, desktop=key.desktop)

    def compiled_label(self, key: FrameKey, instance: str) -> Optional[dict]:
        """``{"visibility", "status", "placement", "geom_type"}`` of one stored row, unread.

        The masks are not parsed: a verdict over a run of forty 12 MP frames
        asks this once per frame.
        """
        row = self.conn.execute(
            "SELECT visibility, status, placement, geom_type FROM compiled_mask "
            "WHERE desktop=? AND step=? AND view=? AND instance=?",
            (key.desktop, key.step, key.view, str(instance)),
        ).fetchone()
        return None if row is None else dict(row)

    def vis_review_rows(self, desktop: int, view: str) -> list[dict]:
        """Every row the visibility review is about in one view (see the module doc).

        ``{"step", "instance", "visibility", "placement", "input_hash",
        "geom_type", "hand_visibility", "occluder", "hand_pixels", "flags"}`` --
        ``flags`` the decoded object (``{}`` when the row has none),
        ``hand_visibility`` / ``occluder`` the frame override's columns (``None``
        without one), ``hand_pixels`` whether it keeps a mask of its own (U39).
        """
        return self._vis_rows(_ROWS, desktop, view)

    def vis_view_rows(self, desktop: int, view: str) -> list[dict]:
        """Every row of the view's confirmed frames, ``visible`` ones too (task U36).

        The same keys as :meth:`vis_review_rows` but ``geom_type``, plus
        ``has_override`` -- the frame has an override row for the part (a hand
        label, a reason, or only a single-frame mask drawn by hand): the other
        views' side of the visibility review, and the frames a part is on.
        """
        return self._vis_rows(_ALL_ROWS, desktop, view)

    def _vis_rows(self, sql: str, desktop: int, view: str) -> list[dict]:
        out = []
        for row in self.conn.execute(sql, (int(desktop), str(view))).fetchall():
            item = dict(row)
            flags = R.loads(item.pop("flags_json"))
            item["flags"] = dict(flags) if isinstance(flags, dict) else {}
            item["step"] = int(item["step"])
            for flag in ("has_override", "hand_pixels"):
                if flag in item:
                    item[flag] = bool(item[flag])
            out.append(item)
        return out

    def vis_presence(self, desktop: int, view: str) -> dict[str, list[int]]:
        """``part -> the confirmed frames it has a row on`` in one view (task U36).

        Where a part *is* in a view: a run of the visibility review that is all
        of it is a part never seen there.
        """
        out: dict[str, list[int]] = {}
        for row in self.conn.execute(
                "SELECT c.instance AS instance, c.step AS step FROM compiled_mask c "
                "JOIN frame f ON f.desktop = c.desktop AND f.step = c.step "
                "AND f.view = c.view WHERE c.desktop = ? AND c.view = ? "
                f"AND f.review_status = 'verified' AND c.instance NOT LIKE '{CONTEXT_PREFIX}%' "
                "ORDER BY c.instance, c.step", (int(desktop), str(view))).fetchall():
            out.setdefault(str(row["instance"]), []).append(int(row["step"]))
        return out

    def annotatable_frames(self, desktop: int, view: str) -> tuple[list[int], set[int]]:
        """``(order, frames)``: the view's annotatable steps, in one query (U36 round 2).

        ``frames`` is every frame of the view that has an image and whose step
        the step table does not skip (``ignore``) -- what
        :func:`tda.core.truth_inputs.annotatable_steps` keeps of any steps --
        and ``order`` those of them that are steps of the table, ascending
        (:func:`tda.core.vis_review.order_of`). ``status`` counts the runs of
        every view with it instead of a frame read per step.
        """
        from tda.core.model import step_is_annotatable

        order: list[int] = []
        frames: set[int] = set()
        for row in self.conn.execute(
                "SELECT f.step AS step, f.missing AS missing, s.step AS listed, "
                "s.step_type AS step_type FROM frame f LEFT JOIN step s "
                "ON s.desktop = f.desktop AND s.step = f.step "
                "WHERE f.desktop = ? AND f.view = ? ORDER BY f.step",
                (int(desktop), str(view))).fetchall():
            if row["missing"] or not step_is_annotatable(row["step_type"]):
                continue
            frames.add(int(row["step"]))
            if row["listed"] is not None:
                order.append(int(row["step"]))
        return order, frames

    def view_annotation(self, desktop: int, view: str) -> tuple[int, list[int]]:
        """``(keyframes, confirmed steps)`` of one view (task U36's 「—」 / 未确认).

        A view with neither is not annotated; with shapes but no confirmed
        frame it is drafts only (Label Studio imports, a view begun).
        """
        kfs = self.conn.execute(
            "SELECT COUNT(*) FROM shape_keyframe WHERE desktop = ? AND view = ?",
            (int(desktop), str(view))).fetchone()[0]
        steps = [int(r[0]) for r in self.conn.execute(
            "SELECT step FROM frame WHERE desktop = ? AND view = ? "
            "AND review_status = 'verified' ORDER BY step",
            (int(desktop), str(view))).fetchall()]
        return int(kfs), steps
