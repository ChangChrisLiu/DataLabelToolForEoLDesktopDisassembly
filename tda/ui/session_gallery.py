"""The session's half of two components-gallery gestures (task U45, spec 3.2b).

Every shot of the gallery shows a different subset of the parts, and some of
the parts on the table are named by no step at all (a loose cable). Two card
buttons make that quick; this module is what they call on the open shot:

* :meth:`GalleryPartsMixin.mark_rest_absent` -- 「其余的都看不见」: every card
  row of this shot that has no shape here and is not 看不见 yet becomes 看不见
  (an ``out_of_view`` label, no pixels: :func:`tda.core.compiler.marked_absent`)
  in **one** ``set_frame_overrides`` op -- one ``Ctrl+Z`` takes them all back.
  The drawn rows are not touched. (Draw what you see, press it, ``Space``.)
* :meth:`GalleryPartsMixin.add_gallery_parts` -- 「＋ 这张里还有…」: gallery-only
  parts (:func:`tda.core.model.is_gallery_only`) of a class and a count, staged
  through :meth:`tda.ui.steps_model.StepTableData.add_extras` -- the S1 rules
  for keys and discriminators -- with ``gallery_only: true``, and written at
  once (:func:`tda.core.extra.write_batch`: the rows and one ``op_log`` row with
  its inverse). One ``add_gallery_parts`` op on the undo stack: ``Ctrl+Z`` takes
  them back through :func:`tda.core.extra.undo_op` -- refused, with nothing
  written, once a shape of one exists -- and ``Ctrl+Y`` writes them again.

The batch is written with :func:`~tda.core.extra.write_batch` rather than
:meth:`~tda.ui.steps_model.StepTableData.save`: the parts carry no event and
no relation, so nothing of the step table, the derived log or the graph moves,
and only the gallery's own frozen shots gain a part (``save`` would queue every
confirmed frame of the desktop, sequence included, for a re-check).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from tda.core import extra as X
from tda.core.db import Db
from tda.core.model import COMPONENT_VIEWS, GALLERY_ONLY_ATTR, InstanceRec
from tda.core.truth import TruthService
from tda.ui import session_api as api
from tda.ui import session_edit as edit
from tda.ui.commands import Op
from tda.ui.gallery import ABSENT
from tda.ui.session_api import SessionRefusal
from tda.ui.session_ops import _write_override, frozen_among, override_state, settle

__all__ = ["ALL_SETTLED", "GalleryPartsMixin", "NOT_IN_GALLERY", "OP_KIND", "OP_RANGE",
           "apply_gallery_parts", "open_rows"]

#: The undo stack's kind of one 「＋ 这张里还有…」 (:data:`tda.ui.commands.KINDS`).
OP_KIND = "add_gallery_parts"
#: 「其余的都看不见」's op: the U15 range kind, one row per part of one shot.
OP_RANGE = "set_frame_overrides"
#: Both buttons outside the gallery (the window greys them out there).
NOT_IN_GALLERY = "只在零件合照里用（F6–F8 和工具栏的「零件合照·扫描仪」）/ components gallery only"
#: 「其余的都看不见」 on a shot whose every row is drawn or 看不见 already.
ALL_SETTLED = "这张的零件都已经画好或标了看不见，没有要标的 / every row of this shot is settled"


def open_rows(card: list[dict]) -> list[str]:
    """The parts of a gallery card still open here: no shape, not 看不见 (in card order)."""
    return [str(row.get("instance", "")) for row in card
            if row.get("gallery") and row.get("kind") == api.KIND_ADD_SHAPE
            and not row.get("done") and row.get("instance")]


def _gallery_shots(db: Db, desktop: int, view: str) -> list[int]:
    return sorted(int(row["step"]) for row in db.frames_for(desktop, view))


def apply_gallery_parts(db: Db, truth: TruthService, payload: dict,
                        current: Optional[int] = None) -> dict:
    """Both directions of an ``add_gallery_parts`` op: write the parts, or take them back.

    ``exists`` true writes ``records`` (refused when a key exists again);
    false undoes the ``op_log`` row the write left in the shared ``stash`` --
    :func:`tda.core.extra.undo_op`, which refuses while any of them has a
    shape (or anything else a human made) and writes nothing then. Every shot
    of the view is settled and the other gallery views' frozen shots queued:
    each of them gains (or loses) the parts alike.
    """
    desktop, view = int(payload["desktop"]), str(payload["view"])
    stash = payload.get("stash")
    stash = stash if isinstance(stash, dict) else {}
    if payload.get("exists"):
        records = [InstanceRec(**data) for data in payload.get("records") or []]
        stored = db.instances(desktop)
        clash = [rec.key for rec in records if rec.key in stored]
        if clash:
            raise SessionRefusal(f"{', '.join(clash)} 已经又有了，不能重做 / exists again")
        batch = X.ExtraBatch(desktop=desktop, records=records)
        stash["op_id"] = X.write_batch(db, batch, str(payload.get("annotator") or ""))
    else:
        op = db.get_op(int(stash.get("op_id") or 0)) if stash.get("op_id") else None
        if op is None:
            raise SessionRefusal("找不到这次添加的记录，不能撤销 / the addition's log row is gone")
        try:
            X.undo_op(db, op, str(payload.get("annotator") or ""), truth.tax)
        except X.ExtraError as refused:
            raise SessionRefusal(f"撤销不了：{refused}（先撤销或删掉画在它上面的形状）") from None
    for other in COMPONENT_VIEWS:
        if other != view:
            shots = _gallery_shots(db, desktop, other)
            truth.queue_rechecks(desktop, other, frozen_among(db, desktop, other, shots))
    return settle(db, truth, desktop, view, _gallery_shots(db, desktop, view), current)


class GalleryPartsMixin:
    """「其余的都看不见」 and 「＋ 这张里还有…」 for :class:`tda.ui.session.AnnotationSession`."""

    # -- 其余的都看不见 --------------------------------------------------------
    def rest_absent_targets(self) -> list[str]:
        """What 「其余的都看不见」 would mark on the open shot (nothing outside the gallery)."""
        if not self.is_open or not self.gallery:
            return []
        return open_rows(self.task_card())

    def mark_rest_absent(self) -> dict:
        """「其余的都看不见」: every open row of this shot is 看不见, in one op.

        Returns ``{"marked": [instances]}`` -- empty when every row is settled
        (nothing is written then). Refused outside the gallery and while one
        of them has pixels in the editing layer (an untouched layer of one is
        simply dropped, as ``` ` ``` drops it).
        """
        if not self.is_open or not self.gallery:
            raise SessionRefusal(NOT_IN_GALLERY)
        targets = self.rest_absent_targets()
        if not targets:
            return {"marked": []}
        editing = self.layer.instance if self.layer.active else None
        if editing in targets and self.layer.changed():
            raise SessionRefusal(f"{editing} 正在画、还没提交：先 Q 提交或 Esc 放弃")
        if editing in targets:
            self.clear_edit()
        key = self.current()
        current = self.db.frame_overrides(key)
        frames, before = [], []
        for instance in targets:
            previous = current.get(instance)
            frames.append({"step": key.step, "instance": instance, "exists": True,
                           "visible_rle": edit._mask_without_reason(previous),
                           "visibility": ABSENT, "occluder": None})
            before.append({"step": key.step, "instance": instance}
                          | override_state(previous))
        common = {"desktop": key.desktop, "view": key.view, "step": key.step,
                  "instance": targets[0], "instances": list(targets), "steps": [key.step]}
        payload = common | {"frames": frames}
        inverse = common | {"frames": before}
        with self.db.transaction():
            for state in frames:
                _write_override(self.db, key, state["instance"], state)
            self.db.log_op(key.desktop, key.view, OP_RANGE, payload, inverse,
                           self.annotator)
        result = settle(self.db, self.truth, key.desktop, key.view, [key.step], key.step)
        result["op"] = Op(OP_RANGE, payload, inverse)
        self._after_edit(result)
        return {"marked": list(targets)}

    # -- ＋ 这张里还有… ----------------------------------------------------------
    def gallery_parts_data(self):
        """The step table the dialog reads (keys, discriminators); refused outside the gallery."""
        from tda.ui.steps_model import StepTableData

        if not self.is_open or not self.gallery:
            raise SessionRefusal(NOT_IN_GALLERY)
        return StepTableData.load(self.db, int(self.desktop), self.tax, self.annotator)

    def add_gallery_parts(self, cls: str, count: int = 1,
                          attrs: Optional[dict] = None) -> list[str]:
        """Make ``count`` gallery-only parts of ``cls``; returns their keys.

        Staged through :meth:`~tda.ui.steps_model.StepTableData.add_extras`
        with ``gallery_only: true`` -- so the keys follow the class's ordinal
        run and a ``role`` / ``kind`` outside the taxonomy is refused -- then
        written at once, one op (``Ctrl+Z`` takes them back while nothing is
        drawn on them). Refused outside the gallery and with pixels of another
        part waiting in the editing layer.
        """
        from tda.ui.steps_model import EditError

        data = self.gallery_parts_data()
        if self.layer.active and self.layer.changed():
            raise SessionRefusal(f"{self.layer.instance} 正在画、还没提交：先 Q 或 Esc")
        flagged = {**dict(attrs or {}), GALLERY_ONLY_ATTR: True}
        try:
            keys = data.add_extras(cls, count, parent=None, attrs=flagged)
        except EditError as refused:
            raise SessionRefusal(str(refused)) from None
        records =[asdict(data.instances[key]) for key in keys]
        stash: dict = {}
        common = {"desktop": int(self.desktop), "view": str(self.view), "keys": list(keys),
                  "annotator": self.annotator, "stash": stash}
        payload = common | {"exists": True, "records": records}
        inverse = common | {"exists": False}
        result = apply_gallery_parts(self.db, self.truth, payload, self.current().step)
        result["op"] = Op(OP_KIND, payload, inverse)
        self._after_edit(result)
        return list(keys)
