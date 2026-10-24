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
        "geom_type", "hand_visibility", "occluder", "flags"}`` -- ``flags`` the
        decoded object (``{}`` when the row has none), ``hand_visibility`` /
        ``occluder`` the frame override's columns (``None`` without one).
        """
        out = []
        for row in self.conn.execute(_ROWS, (int(desktop), str(view))).fetchall():
            item = dict(row)
            flags = R.loads(item.pop("flags_json"))
            item["flags"] = dict(flags) if isinstance(flags, dict) else {}
            item["step"] = int(item["step"])
            out.append(item)
        return out
