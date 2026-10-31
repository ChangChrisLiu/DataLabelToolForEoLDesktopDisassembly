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
No key was added: buttons only.

**Task W3f.**  Resting the pointer on a row (open, folded or ruled) outlines
both parts on the canvas through the hook the task card's hover already uses,
:meth:`~tda.ui.canvas.view.ImageCanvas.set_hint_boxes` -- the part on top in
the difference-box orange (:data:`ABOVE_RGB`), the part underneath in the
draft blue (:data:`BELOW_RGB`), each box round the part's whole shape on the
frame on screen, with a chip naming it -- and the status line says which is
which (:data:`HOVER_TEXT`).  Leaving the row, a frame change and leaving
Review take the boxes away; selection, the editing layer and the database
are not touched.  The rows carry both parts' states at the frame
(:func:`tda.core.truth_inputs.state_of`) and the log's 日志 line, and the
pairs the graph orders indirectly or the log already decides are folded away
(:func:`tda.ui.cover_rulings.triage_pairs`; the log replay,
:func:`~tda.ui.cover_rulings.cover_log`, is kept per desktop until its actions
or hand-written events change).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from tda.core.db_rulings import OP_KIND as RULING_OP
from tda.core.graph import edges_from_db
from tda.core.model import is_component_view
from tda.core.states import initial_overrides
from tda.core.truth_inputs import state_of
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.canvas import overlay_style as OS
from tda.ui.class_names import state_zh
from tda.ui.cover_rulings import (
    RULING_TEXT,
    chassis_pairs,
    cover_log,
    status_text,
    triage_pairs,
)

__all__ = ["ABOVE_CHIP", "ABOVE_RGB", "BELOW_CHIP", "BELOW_RGB", "HOVER_TEXT", "NO_SHAPE",
           "RULINGS_SETTING", "RULING_CLEARED", "RULING_UNCHANGED", "RulingsWindowMixin"]

#: Is the section unfolded?  Remembered (task W3c).
RULINGS_SETTING = "notes/rulings_open"
RULING_CLEARED = "D{desktop:02d}：去掉了「{above} 压着 {below}」的裁定（原为「{word}」，这一对会重新列出）"
RULING_UNCHANGED = "这几对已经是这样裁定的：什么都没写"
#: The hover's two colours (task W3f): reserved overlay colours, so neither is
#: an instance colour; in Review neither the SAM box nor a draft hint is up.
ABOVE_RGB = OS.PROMPT_RGB
BELOW_RGB = OS.DRAFT_RGB
ABOVE_CHIP = "上：{key}"
BELOW_CHIP = "下：{key}"
HOVER_TEXT = "橙框 = 压在上面的 {above}{above_state}，蓝框 = 被压的 {below}{below_state}（只是显示，不改任何东西）"
NO_SHAPE = "（这一帧这个视角上没有它的形状）"


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
        section.sigHover.connect(self.on_rulings_hover)
        changed = getattr(self.session, "sigRulingsChanged", None)
        if changed is not None:
            changed.connect(self._on_rulings_changed)
        frame_changed = getattr(self.session, "sigFrameChanged", None)
        if frame_changed is not None:
            frame_changed.connect(self._clear_rulings_hover)
        #: ``(frame key, [(above, below, px)])`` of the frame the section is about.
        self._rulings_frame: Optional[tuple] = None
        #: ``(stamp, CoverLog)`` of the desktop on screen (task W3f).
        self._rulings_log: Optional[tuple] = None
        #: The pair outlined on the canvas, or ``None`` (task W3f).
        self.rulings_hover: Optional[tuple[str, str]] = None
        self._rulings_hover_text = ""
        #: ``{key: state}`` at the frame the section was last built for.
        self._rulings_states: dict = {}
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
            self._clear_rulings_hover()
            return
        started = time.perf_counter()
        desktop = int(self.session.desktop)
        frame = self._rulings_frame
        here = frame is not None and int(frame[0].desktop) == desktop
        triples = frame[1] if here else []
        db = self.session.db
        if records is None:
            records = db.instances(desktop)
        rulings = db.cover_rulings(desktop)
        step = int(frame[0].step) if here else int(self.session.current().step)
        try:
            states = {k: v.state for k, v in
                      state_of(db, self.session.tax, desktop, step).items()}
        except Exception:  # noqa: BLE001 - a state word is never worth the section
            states = {}
        self._rulings_states = states
        tri = triage_pairs(triples, records, edges_from_db(db, desktop), rulings,
                           log=self._desktop_cover_log(desktop, records), states=states)
        section.set_rulings(tri.asked, rulings, proven=tri.proven, indirect=tri.indirect)
        self.rulings_ms = (time.perf_counter() - started) * 1000.0
        self.rulings_builds += 1

    def _desktop_cover_log(self, desktop: int, records):
        """The desktop's log replay (:func:`cover_log`), rebuilt only when its inputs change."""
        db = self.session.db
        try:
            actions = db.actions(desktop)
            manual = [e for e in db.events(desktop) if not e.auto]
        except Exception:  # noqa: BLE001 - without the log nothing is proven; still asked
            return None
        stamp = (desktop, tuple((a.step, a.idx, a.target, a.verb, a.result) for a in actions),
                 tuple((e.step, e.target, e.attr, e.new) for e in manual),
                 tuple(sorted(records)))
        cached = self._rulings_log
        if cached is not None and cached[0] == stamp:
            return cached[1]
        log = cover_log(records, actions, self.session.tax, initial=initial_overrides(manual))
        self._rulings_log = (stamp, log)
        return log

    # ----------------------------------------------------------- the hover
    def _part_box(self, compiled, key: str) -> Optional[tuple]:
        """The box round ``key``'s whole shape on the frame on screen (its visible box without one)."""
        inst = (getattr(compiled, "instances", None) or {}).get(key)
        if inst is None:
            return None
        amodal = getattr(inst, "amodal", None)
        if amodal is not None:
            window = getattr(inst, "amodal_window", None)
            x0, y0 = 0, 0
            part = amodal
            if window is not None:
                x0, y0, x1, y1 = (int(v) for v in window)
                part = amodal[y0:y1, x0:x1]
            rows, cols = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
            if rows.size and cols.size:
                return (float(x0 + cols[0]), float(y0 + rows[0]),
                        float(x0 + cols[-1] + 1), float(y0 + rows[-1] + 1))
        box = getattr(inst, "box", None)
        return None if box is None else tuple(float(v) for v in box)

    @S.guard
    def on_rulings_hover(self, above: str, below: str) -> None:
        """The pointer rests on a pair's row: outline both parts; ``("", "")`` takes them away.

        Display only (task W3f): the canvas's hover outlines and the status
        line -- no selection, no layer, no op.
        """
        if not above or not below or not self.rulings_apply():
            self._clear_rulings_hover()
            return
        compiled = self._compiled_on_screen()
        hints, words = [], {}
        for key, rgb, chip, side in ((above, ABOVE_RGB, ABOVE_CHIP, "above"),
                                     (below, BELOW_RGB, BELOW_CHIP, "below")):
            box = None if compiled is None else self._part_box(compiled, key)
            if box is not None:
                hints.append((box, chip.format(key=key), rgb))
            state = self._rulings_states.get(key)
            words[side] = (f"（{state_zh(state)}）" if state else "") + ("" if box else NO_SHAPE)
        self.canvas.set_hint_boxes(hints)
        self.rulings_hover = (str(above), str(below))
        self._rulings_hover_text = HOVER_TEXT.format(above=above, below=below,
                                                     above_state=words["above"],
                                                     below_state=words["below"])
        self.report(self._rulings_hover_text)

    def _clear_rulings_hover(self, *_args) -> None:
        """Take the hover's outlines away (row left, frame changed, section hidden)."""
        if getattr(self, "rulings_hover", None) is None:
            return
        self.rulings_hover = None
        self.canvas.set_hint_boxes([])
        if self._rulings_hover_text and self.status_message() == self._rulings_hover_text:
            self.report("")
        self._rulings_hover_text = ""

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
