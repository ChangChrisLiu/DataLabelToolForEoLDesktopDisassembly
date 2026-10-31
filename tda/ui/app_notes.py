"""「这一帧的说明」 in the main window (task U35).

Mixed into :class:`tda.ui.app.MainWindow`.  The Review panel's note box
(:mod:`tda.ui.panels.frame_note`) reports; this writes through the session --
one ``set_frame_note`` op, ``Ctrl+Z`` takes it back -- and says what happened
on the status line:

* 「保存」 / 「清除」 (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteSave`,
  :attr:`~tda.ui.panels.review.ReviewPanel.sigNoteClear`); the keyboard goes
  back to the canvas afterwards, where the Review keys work -- and so does
  ``Tab`` in the box (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteTabOut`);
* **leaving the frame** with the box changed and not saved: every move that
  goes through :meth:`~tda.ui.app_edit.EditMixin.leave_frame`, a confirmation
  (which steps the frame on), leaving Review and closing the window save it
  first (:meth:`NotesWindowMixin.flush_frame_note`) and the status line says
  so; any other way the frame changes, the panel hands the text over with the
  frame it was about (:attr:`~tda.ui.panels.review.ReviewPanel.sigNoteLeft`)
  and it is saved there.

The line promises ``Ctrl+Z`` only while the op is still on the undo stack:
opening another view or machine (and the other moves that re-open the
session) clears the stack after the note was saved, and the line says that
instead (review round 2, N1).  The task card's 「说明：…」 line and the
timeline's ✎ follow the session's ``sigNoteChanged`` themselves.

Task U44 gives the box a dock of its own, 「说明」
(:mod:`tda.ui.panels.notes_dock`): under the Review dock, shown in Review only,
whether the annotator wants it remembered (:data:`NOTES_SETTING`); 「写说明」
on the Review panel (no key) puts it up and the keyboard in the box.  Above
the box, 「这一帧的遮挡顺序」 (:mod:`tda.ui.occlusion_order`) is rebuilt from the
compiled frame the canvas holds once the frame is repainted -- no compile of
its own, nothing written.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QTimer

from tda.core.frame_notes import OP_KIND
from tda.core.model import FrameKey, is_component_view
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.commands import Op
from tda.ui.occlusion_order import floor_parts, labeller, occlusion_order
from tda.ui.panels.notes_dock import DOCK_TITLE, NotesDock, NotesPanel

__all__ = ["NOTES_REVIEW_ONLY", "NOTES_SETTING", "NOTE_CLEARED", "NOTE_HOLD_MS", "NOTE_LEFT",
           "NOTE_LEFT_NO_UNDO", "NOTE_SAVED", "NOTE_SAVED_NO_UNDO", "NotesWindowMixin",
           "SavedNote"]

#: Does the annotator want the 「说明」 dock in Review (task U44)?  Remembered.
NOTES_SETTING = "notes/open"
#: Its menu entry ticked outside Review: it waits for Review.
NOTES_REVIEW_ONLY = "「说明」窗只在复查模式（Review）显示：切到 Review 就会出来"

NOTE_SAVED = "{frame}的说明已保存（Ctrl+Z 撤销）"
NOTE_CLEARED = "{frame}的说明已清除（Ctrl+Z 撤销）"
#: Leaving a frame with the box changed: saved, and said.
NOTE_LEFT = "{frame}的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）"
#: ... when the move opened another view (or machine) and the undo stack went
#: with it: ``Ctrl+Z`` is not promised (review round 2, N1).
NOTE_LEFT_NO_UNDO = ("{frame}的说明改过没保存：离开时已自动保存（换了视角或机器，撤销记录已清空；"
                     "要改回在复查里改）")
NOTE_SAVED_NO_UNDO = "{frame}的说明已保存（撤销记录已清空；要改回在复查里改）"
#: How long that line holds against the arrival's own lines.
NOTE_HOLD_MS = 3000


def _as_bool(value, default: bool = True) -> bool:
    """An INI value as a bool: ``QSettings`` hands booleans back as strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _frame(key: FrameKey) -> str:
    """「第 12 帧」, or 「cmp_oak1 第 1013 张」 for a components-gallery shot."""
    if is_component_view(key.view):
        return f"{key.view} 第 {key.step} 张"
    return f"第 {key.step} 帧"


@dataclass(frozen=True)
class SavedNote:
    """A note the window wrote: its frame, and the op it pushed (``None``: none)."""

    key: FrameKey
    op: Optional[Op] = None


class NotesWindowMixin:
    """The note box's buttons, and the note saved on the way out of a frame."""

    def _init_notes(self) -> None:
        self.review.sigNoteSave.connect(self.on_note_save)
        self.review.sigNoteClear.connect(self.on_note_clear)
        self.review.sigNoteLeft.connect(self.on_note_left)
        self.review.sigNoteTabOut.connect(self.leave_note_box)
        self._init_notes_dock()

    # ------------------------------------------------------- the 「说明」 dock
    def _init_notes_dock(self) -> None:
        """The 「说明」 dock (task U44): built before the saved layout is restored.

        Under the Review dock in the right column, floatable and resizable like
        the others; its place and size travel in the window state, whether the
        annotator wants it in ``notes/open``.  It holds the occlusion order of
        the frame and the Review panel's note box, moved here unchanged.
        """
        self.notes_panel = NotesPanel()
        self.notes_panel.set_note_box(self.review.detach_note_box())
        dock = NotesDock(f"{DOCK_TITLE} / Notes", self)
        dock.setObjectName("dock_notes")        # the key its place is saved under
        dock.setWidget(self.notes_panel)
        dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.splitDockWidget(self.review_dock, dock, Qt.Orientation.Vertical)
        dock.hide()
        self.notes_dock = dock
        self.notes_wanted = _as_bool(self.settings.value(NOTES_SETTING), True)
        #: The frame stamp the order list was built for; ``None``: build it.
        self._notes_order_for: Optional[tuple] = None
        #: Builds done and the last one's time (tests, the report).
        self.notes_order_builds = 0
        self.notes_order_ms = 0.0
        self._notes_timer = QTimer(self)
        self._notes_timer.setSingleShot(True)
        self._notes_timer.setInterval(0)
        self._notes_timer.timeout.connect(self.refresh_notes_order)
        dock.sigClosedByUser.connect(self._on_notes_closed)
        dock.toggleViewAction().triggered.connect(self._on_notes_menu)
        dock.visibilityChanged.connect(self._on_notes_visible)
        self.review.sigWriteNote.connect(self.act_write_note)
        self.notes_panel.order.sigPicked.connect(self.on_order_picked)
        # after the window's own repaint, which compiles (or finds) the frame
        self.session.sigFrameChanged.connect(self._schedule_notes_order)

    def notes_dock_applies(self) -> bool:
        """Is the 「说明」 dock up -- Review mode, and the annotator wants it?"""
        return self.mode == A.MODE_REVIEW and bool(self.notes_wanted) and not self.closed

    def follow_notes_dock(self) -> None:
        """Show the dock in Review (when wanted), hide it in the other modes."""
        dock = getattr(self, "notes_dock", None)
        if dock is None:
            return
        shown = self.notes_dock_applies()
        if dock.isHidden() == shown:
            dock.setVisible(shown)
        if shown:
            self._schedule_notes_order()

    def _want_notes(self, wanted: bool) -> None:
        if self.closed:          # a floating dock closed with the window is no wish
            return
        self.notes_wanted = bool(wanted)
        self.settings.setValue(NOTES_SETTING, bool(wanted))
        self.follow_notes_dock()

    @S.guard
    def _on_notes_closed(self) -> None:
        """The dock's close button: the annotator does not want it."""
        self._want_notes(False)

    @S.guard
    def _on_notes_menu(self, *_args) -> None:
        """The window's dock menu ticked or unticked it (it shows in Review only).

        The tick is read off the action: ``triggered(bool checked = false)``
        has a default, and through :func:`~tda.ui.app_support.guard`'s
        ``(*args, **kwargs)`` PySide picks the overload with no argument
        (round 3, F1) -- a required ``checked`` raised on every tick.
        """
        checked = self.notes_dock.toggleViewAction().isChecked()
        self._want_notes(bool(checked))
        if checked and self.mode != A.MODE_REVIEW:
            self.report(NOTES_REVIEW_ONLY)

    @S.guard
    def act_write_note(self) -> None:
        """「写说明」 (no key): the dock up, the keyboard in the note box."""
        if self.mode != A.MODE_REVIEW:
            return
        self._want_notes(True)
        dock = self.notes_dock
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()
        edit = self.review.note_box.edit
        if edit.isEnabled():
            edit.setFocus(Qt.FocusReason.OtherFocusReason)

    @S.guard
    def on_order_picked(self, instance: str) -> None:
        """A line of the order list was clicked: select its part, nothing more."""
        self.instances.select_instance(str(instance))

    def _on_notes_visible(self, visible: bool) -> None:
        if visible:
            self._schedule_notes_order()

    def _schedule_notes_order(self, *_args) -> None:
        if getattr(self, "notes_dock", None) is not None and not self.closed:
            self._notes_timer.start()

    def _compiled_on_screen(self):
        """The compiled frame the canvas paints (the session's cached one), or ``None``."""
        if not compat.is_open(self.session):
            return None
        try:
            return self.session.compiled()
        except Exception:  # noqa: BLE001 - no frame row, no image: nothing to list
            return None

    @S.guard
    def refresh_notes_order(self, force: bool = False) -> None:
        """Rebuild 「这一帧的遮挡顺序」 from the compiled frame the canvas holds.

        Only while the dock is up, and once per compilation: a commit
        re-announces the frame with a new one, which rebuilds it.  The compiled
        frame is the session's cached one -- the repaint of the frame has
        already asked for it, and this runs after that from the event loop --
        and nothing is written.
        """
        dock = getattr(self, "notes_dock", None)
        if dock is None or self.closed or dock.isHidden():
            return
        order = self.notes_panel.order
        started = time.perf_counter()
        compiled = self._compiled_on_screen()
        if compiled is None:
            order.set_lines([])
            self._notes_order_for = None
            return
        key = compiled.key
        stamp = (key, id(compiled), compiled.input_hash)
        if not force and stamp == self._notes_order_for:
            return
        try:
            drawn = {name for name, fo in (self.session.db.frame_overrides(key) or {}).items()
                     if fo.visible_rle is not None}
        except Exception:  # noqa: BLE001 - a missing table is "none drawn"
            drawn = set()
        try:
            records = self.session.db.instances(key.desktop)
        except Exception:  # noqa: BLE001 - names are a convenience; the keys still say it
            records = {}
        lines = occlusion_order(compiled, drawn, floor_parts(records, compiled.instances))
        order.set_lines(lines, labeller(records))
        self.notes_order_ms = (time.perf_counter() - started) * 1000.0
        self.notes_order_builds += 1
        self._notes_order_for = stamp

    def leave_note_box(self) -> None:
        """The keyboard out of the note box: to the canvas, or -- on a frame with no
        image, where the canvas is not shown -- to the timeline (round 3, R2-N1),
        so that the frame keys (``PgDn``) work again either way."""
        self.focus_canvas()
        box = getattr(getattr(self, "review", None), "note_box", None)
        if box is not None and box.edit.hasFocus():
            self.timeline.list_widget().setFocus(Qt.FocusReason.TabFocusReason)

    def _undo_ops(self) -> list:
        stack = getattr(self.session, "undo_stack", None)
        return list(getattr(stack, "ops", None) or [])

    def _write_note(self, key: Optional[FrameKey], text: str) -> Optional[SavedNote]:
        """One write through the session; what was saved, or ``None`` when nothing changed.

        Raises what the session raises: the caller decides whether a failed
        write stops it (a slot's guard reports it; closing the window does not stop).
        """
        writer = getattr(self.session, "set_frame_note", None)
        if key is None or not callable(writer) or not compat.is_open(self.session):
            return None
        before = self._undo_ops()
        written = writer(text, key=key)
        if written is None:
            return None
        self.logger.info("frame note %s D%s/%s step %s (%d chars)",
                         "set" if text else "cleared", key.desktop, key.view, key.step,
                         len(text or ""))
        after = self._undo_ops()
        top = after[-1] if after else None
        pushed = (top if top is not None and top.kind == OP_KIND
                  and not (before and before[-1] is top) else None)
        return SavedNote(key, pushed)

    def note_undoable(self, saved: Optional[SavedNote]) -> bool:
        """Is the op that wrote ``saved`` still on the undo stack (``Ctrl+Z`` reaches it)?"""
        return bool(saved is not None and saved.op is not None
                    and any(op is saved.op for op in self._undo_ops()))

    @S.guard
    def on_note_save(self, key: FrameKey, text: str) -> None:
        """「保存」: write what the box holds."""
        saved = self._write_note(key, text)
        if saved is not None:
            self.report(self.note_saved_text(saved, left=False))
        self.leave_note_box()

    @S.guard
    def on_note_clear(self, key: FrameKey) -> None:
        """「清除」: the frame has no note any more."""
        if self._write_note(key, "") is not None:
            self.report(NOTE_CLEARED.format(frame=_frame(key)))
        self.leave_note_box()

    @S.guard
    def on_note_left(self, key: FrameKey, text: str) -> None:
        """The frame changed under a box typed in and not saved: save it, say so."""
        saved = self._write_note(key, text)
        if saved is not None:
            self.say_note_later(self.note_saved_text(saved, left=True))

    def say_note_later(self, line: str) -> None:
        """Say ``line`` from the event loop, after the line the move itself ends with.

        An arrival clears the status line (and a confirmation that stepped on
        writes its own): said then, the note's line comes after it -- joined to
        what is there -- and holds 3 s against what comes next.  While a
        confirmation is running (:meth:`hold_note_line`) the line waits for it:
        the confirmation steps the frame again, which would clear it, and
        takes it into its own line instead (:meth:`take_note_line`).
        """
        if not line:
            return
        self._note_line_pending = line
        # the window as context: a window deleted meanwhile drops the call
        QTimer.singleShot(0, self, self._say_pending_note)

    def _say_pending_note(self) -> None:
        if self.closed or getattr(self, "_note_line_held", False):
            return
        line = self.take_note_line()
        if not line:
            return
        now = self.status_message()
        self.report(line if not now or line in now else f"{now}；{line}",
                    hold_ms=NOTE_HOLD_MS)

    def take_note_line(self) -> str:
        """The note line waiting to be said (``""``: none); it is not said again."""
        line = getattr(self, "_note_line_pending", "") or ""
        self._note_line_pending = ""
        return line

    def hold_note_line(self, held: bool) -> None:
        """While ``held``, a waiting note line is not said (a confirmation is running).

        Released, a line nobody took is said from the event loop as usual.
        """
        self._note_line_held = bool(held)
        if not held and getattr(self, "_note_line_pending", ""):
            QTimer.singleShot(0, self, self._say_pending_note)

    def flush_frame_note(self) -> Optional[SavedNote]:
        """Save what the note box holds for its frame if it was changed.

        ``None`` when there was nothing to save.  Called before every way out
        of a frame the window controls, so the op lands on the undo stack of
        the view it is about; the caller says it (:meth:`note_saved_text`)
        once it knows whether the frame was left -- and whether the stack still
        holds the op.
        """
        review = getattr(self, "review", None)
        box = getattr(review, "note_box", None)
        if box is None or not box.is_dirty():
            return None
        return self._write_note(box.key(), box.text())

    def note_text_typed(self) -> str:
        """What the note box holds right now (``""``: no box) -- for an error that loses it."""
        box = getattr(getattr(self, "review", None), "note_box", None)
        return box.text() if box is not None else ""

    def note_saved_text(self, saved: Optional[SavedNote], left: bool = True) -> str:
        """The status line for a note the window saved (``""``: none).

        ``left``: the frame (or Review) was left, so it was saved on the way
        out; otherwise the plain 「已保存」.  ``Ctrl+Z`` is promised only while
        the op is on the undo stack (:meth:`note_undoable`).
        """
        if saved is None:
            return ""
        undoable = self.note_undoable(saved)
        if left:
            line = NOTE_LEFT if undoable else NOTE_LEFT_NO_UNDO
        else:
            line = NOTE_SAVED if undoable else NOTE_SAVED_NO_UNDO
        return line.format(frame=_frame(saved.key))

    def note_op_text(self, undo: bool) -> str:
        """「撤销：第 12 帧的说明」 when the op ``Ctrl+Z`` / ``Ctrl+Y`` is about to replay is a note's."""
        stack = getattr(self.session, "undo_stack", None)
        if stack is None:
            return ""
        ops = stack.ops if undo else getattr(stack, "undone_ops", [])
        if not ops or ops[-1].kind != OP_KIND:
            return ""
        p = ops[-1].payload
        frame = _frame(FrameKey(int(p["desktop"]), int(p["step"]), str(p["view"])))
        return f"{'撤销' if undo else '重做'}：{frame}的说明 / {'undo' if undo else 'redo'}"
