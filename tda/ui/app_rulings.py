"""「谁压着谁」 in the main window (task W3c): the dock section's buttons, written through the session.

Mixed into :class:`tda.ui.app.MainWindow`.  The 「说明」 dock (U44) gains a
section above the occlusion list (:class:`~tda.ui.panels.notes_dock.RulingsSection`)
listing the overlapping pairs of the frame on screen that still want the
user's ruling (:func:`tda.ui.cover_rulings.pending_pairs`).  The pairs are the
ones the occlusion list is counted from -- :meth:`refresh_notes_order` counts
them once and hands them here (:meth:`RulingsWindowMixin.set_rulings_frame`) --
so this compiles nothing; it reads the desktop's instance records, its
relation rows and its rulings, all three cheap.

* 「挡着」 / 「不挡」 on a row, or on a batch line 「同类对本帧全部」: one
  ``cover_ruling`` op through :meth:`~tda.ui.session_rulings.RulingsMixin.set_cover_rulings`
  (view = the open view, step = the frame on screen); the status line says
  「D13 第 7 帧：optical_drive.01 压着 motherboard.01 → 挡着」 and the row goes.
* 「撤销」 in 「已裁定」: one op taking that ruling away; the pair is asked again.
* ``Ctrl+Z`` takes the last ruling back like any other edit (the session's
  undo stack), and names it (:meth:`RulingsWindowMixin.ruling_op_text`).

The section shows in Review only (the dock does) and not on the components
gallery's views; whether it is unfolded is ``QSettings notes/rulings_open``.
No key was added: buttons only.  Hovering a row does not light the two parts
up on the canvas -- the dock has no such hook (U44 had none).
"""
from __future__ import annotations

import time
from typing import Optional

from tda.core.db_rulings import OP_KIND as RULING_OP
from tda.core.graph import edges_from_db
from tda.core.model import is_component_view
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.cover_rulings import (
    RULING_TEXT,
    chassis_pairs,
    pending_pairs,
    status_text,
)

__all__ = ["RULINGS_SETTING", "RULING_CLEARED", "RULING_UNCHANGED", "RulingsWindowMixin"]

#: Is the section unfolded?  Remembered (task W3c).
RULINGS_SETTING = "notes/rulings_open"
RULING_CLEARED = "D{desktop:02d}：去掉了「{above} 压着 {below}」的裁定（原为「{word}」，这一对会重新列出）"
RULING_UNCHANGED = "这几对已经是这样裁定的：什么都没写"


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class RulingsWindowMixin:
    """The 「谁压着谁」 section of the 「说明」 dock."""

    def _init_rulings(self) -> None:
        """Wire the section (after the 「说明」 dock is built, :meth:`_init_notes`)."""
        section = self.notes_panel.rulings
        section.set_open(_as_bool(self.settings.value(RULINGS_SETTING), True))
        section.sigToggled.connect(self._on_rulings_toggled)
        section.sigRule.connect(self.on_rule_pairs)
        section.sigClear.connect(self.on_clear_ruling)
        changed = getattr(self.session, "sigRulingsChanged", None)
        if changed is not None:
            changed.connect(self._on_rulings_changed)
        #: ``(frame key, [(above, below, px)])`` of the frame the section is about.
        self._rulings_frame: Optional[tuple] = None
        #: Rebuilds done and the last one's time (tests, the report).
        self.rulings_builds = 0
        self.rulings_ms = 0.0

    def rulings_apply(self) -> bool:
        """Does the section show: Review, a desktop open, not a components-gallery view?"""
        return (self.mode == A.MODE_REVIEW and compat.is_open(self.session)
                and not is_component_view(str(getattr(self.session, "view", "") or "")))

    def set_rulings_frame(self, compiled, pairs, records=None) -> None:
        """The frame on screen and its overlapping pairs (``overlap_pairs``), from the order list."""
        if compiled is None:
            self._rulings_frame = None
        else:
            self._rulings_frame = (compiled.key, chassis_pairs(compiled, pairs or {}))
        self.refresh_rulings(records)

    @S.guard
    def refresh_rulings(self, records=None) -> None:
        """Rebuild the section from the frame's pairs and the desktop's edges and rulings."""
        panel = getattr(self, "notes_panel", None)
        if panel is None:
            return
        section = panel.rulings
        shown = self.rulings_apply()
        section.setVisible(shown)
        if not shown:
            return
        started = time.perf_counter()
        desktop = int(self.session.desktop)
        frame = self._rulings_frame
        triples = frame[1] if frame is not None and int(frame[0].desktop) == desktop else []
        db = self.session.db
        if records is None:
            records = db.instances(desktop)
        rulings = db.cover_rulings(desktop)
        candidates = pending_pairs(triples, records, edges_from_db(db, desktop), rulings)
        section.set_rulings(candidates, rulings)
        self.rulings_ms = (time.perf_counter() - started) * 1000.0
        self.rulings_builds += 1

    def _on_rulings_changed(self, *_args) -> None:
        self.refresh_rulings()

    def _on_rulings_toggled(self, opened: bool) -> None:
        self.settings.setValue(RULINGS_SETTING, bool(opened))

    @S.guard
    def on_rule_pairs(self, pairs: list, ruling: str, classes=None) -> None:
        """「挡着」 / 「不挡」: rule ``pairs`` (one op) and say it on the status line."""
        if not self.rulings_apply():
            return
        key = self.session.current()
        pairs = [(str(a), str(b)) for a, b in pairs]
        if not self.session.set_cover_rulings(pairs, ruling, step=int(key.step)):
            self.report(RULING_UNCHANGED)
            return
        self.logger.info("cover ruling D%s %s step %s: %d pair(s) %s", key.desktop, key.view,
                         key.step, len(pairs), ruling)
        self.report(status_text(key.desktop, key.step, pairs, ruling,
                                tuple(classes) if classes else None))

    @S.guard
    def on_clear_ruling(self, above: str, below: str) -> None:
        """「撤销」 on a ruled pair: take the ruling away (one op)."""
        before = self.session.cover_rulings().get((str(above), str(below)))
        if before is None or not self.session.clear_cover_ruling(str(above), str(below)):
            return
        self.report(RULING_CLEARED.format(desktop=int(self.session.desktop), above=above,
                                          below=below,
                                          word=RULING_TEXT.get(before["ruling"], "?")))

    def ruling_op_text(self, undo: bool) -> str:
        """「撤销：D13 的「谁压着谁」裁定（1 对）」 when ``Ctrl+Z`` / ``Ctrl+Y`` replays a ruling op."""
        stack = getattr(self.session, "undo_stack", None)
        if stack is None:
            return ""
        ops = stack.ops if undo else getattr(stack, "undone_ops", [])
        if not ops or ops[-1].kind != RULING_OP:
            return ""
        payload = ops[-1].payload
        n = len(payload.get("rows") or [])
        return (f"{'撤销' if undo else '重做'}：D{int(payload['desktop']):02d} 的「谁压着谁」裁定"
                f"（{n} 对） / {'undo' if undo else 'redo'}")
