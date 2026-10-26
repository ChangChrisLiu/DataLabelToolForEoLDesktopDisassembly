"""Review dock: the work queues of review mode (spec 4.4).

Conflicts, needs-review, missing shapes and unexplained differences each get a
tab whose label carries the queue's size, so the remaining work is visible
without opening anything.  Activating an entry asks for its frame; ``Enter``
confirms the frame that is open and ``R`` marks it for rework.

The fifth tab, 「可见性审查」 (task U24), lists the confirmed labels still to be
audited, one line per part and run of frames (按段) or, toggled, one per frame
(按帧); **selecting** a line asks for its frame and part
(:attr:`ReviewPanel.sigVisibilityPicked`) -- the keys then answer the whole
line -- and a line above the list says how far the view is (「还要审 n 段 / m
帧」), or 「可见性审查完成」.  The session lists that queue only while Review
shows it; a refresh without it leaves the tab as it was.

The panel decides nothing: the lists **report**
(:attr:`ReviewPanel.sigOpenStep`, :attr:`ReviewPanel.sigRework`) and the window
acts.  Opening another frame can be refused, and a resolution has three
possible outcomes; neither answer is the panel's to give.  The queues are then
re-read, never patched locally.

It has **no resolution buttons** of its own (task U2b round 1b): the "Keep old
K" / "Take new N" pair duplicated the tool palette's 保留旧的 / 采用新的, and two
places for one thing is the confusion the second trial reported.  The panel
says which conflict is selected (:meth:`ReviewPanel.selected_conflict`); the
palette's buttons and the keys resolve it through the window.

The two buttons it does have (task U32), 「重签这一帧」 and 「重签全部」, are
nowhere else and have no key: they only **report**
(:attr:`ReviewPanel.sigRefreeze`, ``"frame"`` or ``"all"``), and the window
plans, asks -- naming the frame count -- and writes.

Task U36 puts the runs and the other views on the 可见性审查 tab
(:mod:`tda.ui.panels.visruns`): a summary at the top -- per part, how long it
is 看不见 in every view; the lines with their frame count and 「全程」, the
part with the longest run first (the caption says so); and under the list,
for the selected line, its first / middle / last frames cut around the part,
what every view says on the same steps, the suggested reason, and two more
buttons without keys that only report: 「从这帧起另算」
(:attr:`ReviewPanel.sigSplitRun`) and 「同步到其他视角」
(:attr:`ReviewPanel.sigSyncViews`).
Below them, 「这一帧的说明」 (task U35, :mod:`tda.ui.panels.frame_note`): the
annotator's note on the frame on screen.  Its 「保存」 / 「清除」 report too
(:attr:`ReviewPanel.sigNoteSave`, :attr:`ReviewPanel.sigNoteClear`); when the
frame changes under a box that was typed in and not saved, the panel reports
the text with the frame it was about (:attr:`ReviewPanel.sigNoteLeft`) before
it shows the new frame's note -- the window saves it and says so.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tda.ui import session_api as api
from tda.ui import visreview_text as VT
from tda.ui.panels import session_is_open
from tda.ui.panels.visruns import RunDetail, RunSummary
from tda.ui.panels.frame_note import FrameNoteBox

__all__ = ["ReviewPanel", "QUEUE_TITLES", "REFREEZE_ALL", "REFREEZE_FRAME"]

STEP_ROLE = int(Qt.ItemDataRole.UserRole)
CID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
#: The whole entry of a 可见性审查 line (task U24): its part and its run.
ENTRY_ROLE = int(Qt.ItemDataRole.UserRole) + 2

#: Tab captions, in :data:`api.QUEUE_NAMES` order.
QUEUE_TITLES: dict[str, str] = {
    api.QUEUE_CONFLICTS: "Conflicts",
    api.QUEUE_NEEDS_REVIEW: "Needs review",
    api.QUEUE_MISSING_SHAPE: "Missing shape",
    api.QUEUE_UNEXPLAINED: "Unexplained",
    api.QUEUE_VISIBILITY: "可见性审查",
}

#: The keys of the 可见性审查 tab, under its progress line (the panel's tooltip).
VIS_KEYS = ("选中的一行整段生效：Space 确认（按计算） · 数字键 手动设 · "
            "` 被线挡住（Shift 部分） · Z 角度看不到（Shift 部分） · Ctrl+Z 撤销；"
            "只改一帧：切到「按帧」")
#: ... as the line under the progress says it (task U36: the dock is shared
#: with the summary and the selected line's detail; the whole text is its tooltip).
VIS_KEYS_SHORT = "整段：Space 对 · 数字 手动 · ` 线 · Z 角度 · Shift 部分 · Ctrl+Z 撤销"
#: The two listings of the 可见性审查 tab: a line per run, a line per frame.
BY_RUN, BY_FRAME = "按段", "按帧"
#: The 可见性审查 list keeps at least this many lines (U36 round 2, N2: the
#: summary and the detail share the dock with it).
LIST_MIN_ROWS = 5

#: 「重签」 (task U32): what :attr:`ReviewPanel.sigRefreeze` carries.
REFREEZE_FRAME, REFREEZE_ALL = "frame", "all"
REFREEZE_FRAME_TEXT, REFREEZE_ALL_TEXT = "重签这一帧", "重签全部"
REFREEZE_TIP = ("把已确认帧里和今天的编译只差容差以内的行（不算冲突、一直存着旧像素）改成今天的编译；"
                "帧仍是已确认，超出容差的行不动（它们是冲突）。先列出要改几帧再问。没有快捷键。")


def _entry_text(queue: str, entry: dict) -> str:
    """One line for a queue entry."""
    step = entry.get("step", "?")
    if queue == api.QUEUE_VISIBILITY:
        return VT.entry_text(entry)
    if queue == api.QUEUE_CONFLICTS:
        # A disagreement about a *label* moves no pixels: "visibility: visible
        # -> occluded_partial" is 0 differing pixels, and the row read "(Δ 0
        # px)" -- an entry somebody has to settle, described as nothing having
        # changed.  The queue says what disagrees; the pixel count is its
        # fallback, and is all an entry queued by an older session carries.
        what = entry.get("summary") or f"{entry.get('sym_diff_px', 0)} px differ"
        return f"Step {step} — {entry.get('instance', '')} ({what})"
    if queue == api.QUEUE_MISSING_SHAPE:
        return f"Step {step} — {entry.get('instance', '')}"
    return f"Step {step}"


class ReviewPanel(QWidget):
    """Queue tabs of review mode; the actions are the window's (keys, palette)."""

    #: The annotator marked a step for rework.
    sigRework = Signal(int)
    #: A queue entry was activated: the window should open that step -- through
    #: the gate.  The panel used to call ``session.goto`` itself, un-forced, and
    #: an uncommitted layer then made it raise out of a Qt slot.
    sigOpenStep = Signal(int)
    #: ``(step, part)``: the annotator selected a 可见性审查 line (task U24);
    #: the window opens that frame -- through the gate -- and selects the part.
    #: Never emitted by a refresh putting a selection back.
    sigVisibilityPicked = Signal(int, str)
    #: 「重签这一帧」 (``"frame"``) or 「重签全部」 (``"all"``) was clicked (task U32).
    sigRefreeze = Signal(str)
    #: 「从这帧起另算」 was clicked (task U36): the selected line, at the open frame.
    sigSplitRun = Signal()
    #: 「同步到其他视角」 was clicked (task U36): the sync the window offered.
    sigSyncViews = Signal()
    #: 「这一帧的说明」 (task U35): ``(frame key, text)`` -- 「保存」 was clicked.
    sigNoteSave = Signal(object, str)
    #: ... the frame key -- 「清除」 was clicked on a frame that has a note.
    sigNoteClear = Signal(object)
    #: ... ``(frame key, text)`` -- the frame changed under a box typed in and
    #: not saved; emitted before the new frame's note is shown.
    sigNoteLeft = Signal(object, str)
    #: ... ``Tab`` in the box: the keyboard goes back to the canvas.
    sigNoteTabOut = Signal()

    def __init__(self, session: Optional[api.SessionLike] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[api.SessionLike] = None
        self._problems: list[str] = []
        #: The queue list whose row was activated and not yet answered for.
        self._activated: Optional[QListWidget] = None
        #: Set while :meth:`refresh` rebuilds the lists: a selection it puts
        #: back is not the annotator picking a line.
        self._refreshing = False
        #: The 可见性审查 line picked, handed on from the event loop.
        self._picked: Optional[tuple[int, str]] = None
        #: The runs the last refresh read: what 按段 / 按帧 re-lists.
        self._vis_runs: list[dict] = []
        self._pick_timer = QTimer(self)
        self._pick_timer.setSingleShot(True)
        self._pick_timer.setInterval(0)
        self._pick_timer.timeout.connect(self._emit_picked)
        #: The line whose frame strip is shown, or asked for (task U36): the
        #: thumbnails are cut after the frame change a pick causes, not before.
        self._strip_for: Optional[tuple] = None
        #: The other views' suggestion for the selected line (U36), as last read.
        self._suggestion: Optional[dict] = None
        self._strip_timer = QTimer(self)
        self._strip_timer.setSingleShot(True)
        self._strip_timer.setInterval(30)
        self._strip_timer.timeout.connect(self._load_strip)

        self._tabs = QTabWidget()
        self._lists: dict[str, QListWidget] = {}
        for queue in api.QUEUE_NAMES:
            lw = QListWidget()
            lw.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            lw.itemActivated.connect(self._on_item_activated)
            self._lists[queue] = lw
            if queue == api.QUEUE_VISIBILITY:
                self._tabs.addTab(self._visibility_page(lw), QUEUE_TITLES[queue])
            else:
                self._tabs.addTab(lw, QUEUE_TITLES[queue])

        self._problems_label = QLabel("Problems")
        self._problems_list = QListWidget()
        self._problems_list.setMaximumHeight(90)
        self._problems_label.setVisible(False)
        self._problems_list.setVisible(False)

        # The keys live on the tool palette, in the tooltip and in the cheat
        # sheet, not in a 384 px label that decides how wide the dock has to be.
        self.setToolTip(
            "Enter: open the selected entry's frame and accept it\n"
            "R: rework it in Annotate mode\n"
            "K: keep the frozen shape    N: take the edit\n"
            "(左边工具面板上的「接受这一帧 / 返工 / 保留旧的 / 采用新的」是同样的键)\n"
            f"可见性审查：{VIS_KEYS}\n"
            "下方「重签这一帧 / 重签全部」：容差以内的旧行改成今天的编译（按钮，没有键）"
        )

        # 「重签」 (task U32): two buttons, no key, nothing else does it
        self.refreeze_frame_button = QToolButton()
        self.refreeze_all_button = QToolButton()
        resign = QHBoxLayout()
        resign.setContentsMargins(0, 0, 0, 0)
        for button, text, scope in ((self.refreeze_frame_button, REFREEZE_FRAME_TEXT,
                                     REFREEZE_FRAME),
                                    (self.refreeze_all_button, REFREEZE_ALL_TEXT,
                                     REFREEZE_ALL)):
            button.setText(text)
            button.setToolTip(REFREEZE_TIP)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, s=scope: self.sigRefreeze.emit(s))
            resign.addWidget(button)
        resign.addStretch(1)

        # 「这一帧的说明」 (task U35): the note on the frame on screen
        self.note_box = FrameNoteBox()
        self.note_box.sigSave.connect(self.sigNoteSave)
        self.note_box.sigClear.connect(self.sigNoteClear)
        self.note_box.sigTabOut.connect(self.sigNoteTabOut)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self._tabs, 1)
        layout.addLayout(resign)
        layout.addWidget(self.note_box)
        layout.addWidget(self._problems_label)
        layout.addWidget(self._problems_list)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if session is not None:
            self.set_session(session)

    # -- wiring -------------------------------------------------------------
    def set_session(self, session: Optional[api.SessionLike]) -> None:
        """Attach a session (or ``None``) and re-read the queues."""
        if self._session is not None:
            self._session.sigFrameChanged.disconnect(self._on_frame_changed)
            self._session.sigProblems.disconnect(self._on_problems)
            changed = getattr(self._session, "sigNoteChanged", None)
            if changed is not None:
                changed.disconnect(self._on_note_changed)
        self._session = session
        if session is not None:
            session.sigFrameChanged.connect(self._on_frame_changed)
            session.sigProblems.connect(self._on_problems)
            changed = getattr(session, "sigNoteChanged", None)
            if changed is not None:
                changed.connect(self._on_note_changed)
        self.refresh()
        self._follow_note()

    def tabs(self) -> QTabWidget:
        """The tab widget, for the main window's layout and for tests."""
        return self._tabs

    def list_for(self, queue: str) -> QListWidget:
        """The list of one queue of :data:`api.QUEUE_NAMES`."""
        return self._lists[queue]

    def queue_at(self, index: int) -> str:
        """Name of the queue shown in tab ``index``."""
        return api.QUEUE_NAMES[index]

    def current_queue(self) -> str:
        """Name of the queue currently on top."""
        return self.queue_at(max(0, self._tabs.currentIndex()))

    def _visibility_page(self, lw: QListWidget) -> QWidget:
        """The 可见性审查 tab: a progress line with the 按段 / 按帧 toggle, the
        keys, and the list (task U24)."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        self._vis_status = QLabel("")
        self._vis_status.setWordWrap(True)
        self._vis_by_run = QToolButton()
        self._vis_by_frame = QToolButton()
        self._vis_listing = QButtonGroup(page)
        for button, text, tip in (
                (self._vis_by_run, BY_RUN, "一行一段（同一零件、相邻帧、标签和原因都相同）：键作用于整段"),
                (self._vis_by_frame, BY_FRAME, "一行一帧：只改这一帧时用")):
            button.setText(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._vis_listing.addButton(button)
        self._vis_listing.setExclusive(True)
        self._vis_by_run.setChecked(True)
        self._vis_by_frame.toggled.connect(self._on_listing_toggled)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._vis_status, 1)
        head.addWidget(self._vis_by_run)
        head.addWidget(self._vis_by_frame)
        self._vis_keys = QLabel(VIS_KEYS_SHORT)
        self._vis_keys.setToolTip(VIS_KEYS)
        self._vis_keys.setWordWrap(True)
        self._vis_keys.setStyleSheet("color: gray; font-size: 8pt")
        # task U36: per part, how long it is 看不见 in every view, under a title
        # that says the order and folds the table away (round 2, N2)
        self.run_summary = RunSummary()
        self.run_summary.setVisible(False)
        self.run_summary.sigPartPicked.connect(self._on_summary_part)
        # ... and under the list, the selected line across the views; the two
        # halves share the dock's height through a splitter, the list keeps
        # LIST_MIN_ROWS lines whatever the detail wants, the detail scrolls
        self.run_detail = RunDetail()
        self.run_detail.sigSplit.connect(self.sigSplitRun)
        self.run_detail.sigSync.connect(self.sigSyncViews)
        lw.setMinimumHeight(LIST_MIN_ROWS * (lw.fontMetrics().height() + 2)
                            + 2 * lw.frameWidth() + 6)
        queue = QWidget()
        top = QVBoxLayout(queue)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(1)
        top.addWidget(self.run_summary)
        top.addWidget(lw, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.run_detail)
        scroll.setMinimumHeight(32)
        self._vis_split = QSplitter(Qt.Orientation.Vertical)
        self._vis_split.setChildrenCollapsible(False)
        self._vis_split.addWidget(queue)
        self._vis_split.addWidget(scroll)
        self._vis_split.setStretchFactor(0, 1)
        self._vis_split.setStretchFactor(1, 1)
        box.addLayout(head)
        box.addWidget(self._vis_keys)
        box.addWidget(self._vis_split, 1)
        lw.currentItemChanged.connect(self._on_visibility_current)
        return page

    def per_frame(self) -> bool:
        """Is the 可见性审查 list one line per frame (按帧) rather than per run?"""
        return self._vis_by_frame.isChecked()

    def set_per_frame(self, on: bool) -> None:
        """按帧 (``True``) or 按段: the same queue, the runs split or not."""
        (self._vis_by_frame if on else self._vis_by_run).setChecked(True)

    def _on_listing_toggled(self, _checked: bool) -> None:
        """Re-list what the last refresh read; nothing is asked of the session."""
        quiet, self._refreshing = self._refreshing, True
        try:
            self._fill_visibility(list(api.QUEUE_NAMES).index(api.QUEUE_VISIBILITY),
                                  self._vis_runs)
        finally:
            self._refreshing = quiet

    # -- content ------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read ``session.queues()`` into the lists.

        The 可见性审查 list is rebuilt only when the session lists that queue
        (Review shows it); its selection is put back by part and frame, not by
        row number -- a verdict takes rows out from above it.
        """
        queues = self._session.queues() if session_is_open(self._session) else {}
        self._refreshing = True
        try:
            for index, queue in enumerate(api.QUEUE_NAMES):
                if queue == api.QUEUE_VISIBILITY:
                    if queue in queues or not session_is_open(self._session):
                        self._fill_visibility(index, list(queues.get(queue, [])))
                    continue
                entries = list(queues.get(queue, []))
                lw = self._lists[queue]
                keep = lw.currentRow()
                lw.clear()
                for entry in entries:
                    item = QListWidgetItem(_entry_text(queue, entry))
                    item.setData(STEP_ROLE, int(entry.get("step", -1)))
                    if entry.get("id") is not None:
                        item.setData(CID_ROLE, int(entry["id"]))
                    lw.addItem(item)
                if 0 <= keep < lw.count():
                    lw.setCurrentRow(keep)
                self._tabs.setTabText(index, f"{QUEUE_TITLES[queue]} ({len(entries)})")
        finally:
            self._refreshing = False

    def _fill_visibility(self, index: int, entries: list[dict]) -> None:
        """The 可见性审查 list, its count (frames) and its progress line.

        ``entries`` are the session's runs; in the 按帧 listing each run is
        split into its frames -- the uncompressed queue.
        """
        self._vis_runs = list(entries)
        lw = self._lists[api.QUEUE_VISIBILITY]
        keep = self.selected_visibility()
        here = (int(self._session.current().step) if session_is_open(self._session)
                else None)
        lines = entries
        if self.per_frame():
            lines = [dict(entry, step=int(step), steps=[int(step)], whole=False, changes=[])
                     for entry in entries for step in entry.get("steps") or ()]
        lw.clear()
        rows = 0
        for entry in lines:
            text = _entry_text(api.QUEUE_VISIBILITY, entry)
            item = QListWidgetItem(text)
            item.setData(STEP_ROLE, int(entry.get("step", -1)))
            item.setData(ENTRY_ROLE, dict(entry))
            # the line is wider than the dock: the whole of it, and who hides it
            who = "、".join(entry.get("occluded_by") or ())
            tip = text + (f"\n挡住它的：{who}" if who else "")
            if entry.get("whole"):
                tip += f"\n{VT.WHOLE}：{VT.WHOLE_TIP}"
            if entry.get("changes"):
                tip += "\n" + VT.CHANGES.format(steps=VT.steps_text(entry["changes"]))
            item.setToolTip(tip)
            lw.addItem(item)
            rows += len(entry.get("steps") or [entry.get("step")])
        if keep is not None:
            # the same part, at the frame on screen if its run still has it,
            # else at the frames the old line covered
            wanted = [here] if here in (keep.get("steps") or ()) else list(keep.get("steps") or ())
            self.select_visibility(str(keep.get("instance")), wanted)
        self._tabs.setTabText(index, f"{QUEUE_TITLES[api.QUEUE_VISIBILITY]} ({rows})")
        self._vis_status.setText(self._visibility_line(len(entries), rows)
                                 if session_is_open(self._session) else "")
        summary = getattr(self._session, "visibility_summary", None)
        self.run_summary.set_rows(list(summary()) if callable(summary) and entries
                                  and session_is_open(self._session) else [])
        self._show_detail(self.selected_visibility())

    # -- the selected line across the views (task U36) ---------------------------
    def _show_detail(self, entry: Optional[dict]) -> None:
        """The detail under the list: what the views say, now; the strip, soon."""
        detail = getattr(self._session, "visibility_detail", None)
        found = (detail(entry) if callable(detail) and entry is not None
                 and session_is_open(self._session) else {})
        self._suggestion = (found or {}).get("suggestion")
        self.run_detail.show_entry(entry, found or {})
        wanted = (None if entry is None else
                  (str(entry.get("instance")), tuple(int(s) for s in entry.get("steps") or ())))
        if wanted != self._strip_for:
            self._strip_for = wanted
            self._strip_timer.start()

    def _load_strip(self) -> None:
        """Cut the selected line's thumbnails (after the frame change it caused)."""
        strip = getattr(self._session, "visibility_strip", None)
        if self._strip_for is None or not callable(strip) or not session_is_open(self._session):
            self.run_detail.set_strip([])
            return
        instance, steps = self._strip_for
        try:
            frames = list(strip(instance, list(steps)))
        except Exception:  # noqa: BLE001 - a thumbnail is never worth a failure
            frames = []
        self.run_detail.set_strip(frames)

    def load_strip_now(self) -> None:
        """Cut the thumbnails now rather than on the timer (the tests, a screenshot)."""
        self._strip_timer.stop()
        self._load_strip()

    def selected_suggestion(self) -> Optional[dict]:
        """The reason the other views point at for the selected line (or ``None``)."""
        return self._suggestion if self.selected_visibility() is not None else None

    def set_sync_offer(self, text: Optional[str]) -> None:
        """Offer 「同步到其他视角」 (what it would do), or take the offer back (``None``)."""
        self.run_detail.set_sync_offer(text)

    def _on_summary_part(self, instance: str) -> None:
        """A summary row was clicked: select that part's first line, as a click would."""
        lw = self._lists[api.QUEUE_VISIBILITY]
        for row in range(lw.count()):
            entry = lw.item(row).data(ENTRY_ROLE) or {}
            if entry.get("instance") == instance:
                lw.setCurrentRow(row)
                return

    def _visibility_line(self, runs: int, pending: int) -> str:
        """「还要审 n 段 / m 帧（已审 r / t 帧）」, or 「可见性审查完成」."""
        progress = getattr(self._session, "visibility_progress", None)
        reviewed, total = (0, pending)
        if callable(progress) and session_is_open(self._session):
            try:
                reviewed, total = (int(v) for v in progress())
            except Exception:  # noqa: BLE001 - a count is never worth a failure
                reviewed, total = 0, pending
        if pending:
            return f"还要审 {runs} 段 / {pending} 帧（已审 {reviewed} / {total} 帧）"
        if total:
            return f"{VT.DONE}（{total} 帧都审过了）"
        return f"{VT.DONE}{VT.NOTHING_TO_REVIEW}"

    def visibility_row(self) -> int:
        """The row of the selected 可见性审查 line, ``-1`` for none."""
        return self._lists[api.QUEUE_VISIBILITY].currentRow()

    def select_visibility_row(self, row: int) -> bool:
        """Select line ``row`` (the last one when the list is shorter); ``False`` when empty.

        After an answer the line that took the answered one's place is the next
        one. No :attr:`sigVisibilityPicked`: the window moves there itself.
        """
        lw = self._lists[api.QUEUE_VISIBILITY]
        if lw.count() == 0:
            return False
        quiet, self._refreshing = self._refreshing, True
        try:
            lw.setCurrentRow(min(max(int(row), 0), lw.count() - 1))
        finally:
            self._refreshing = quiet
        self._show_detail(self.selected_visibility())
        return True

    def visibility_status(self) -> str:
        """The 可见性审查 tab's progress line, as shown."""
        return self._vis_status.text()

    def problems(self) -> list[str]:
        """Problems of the last refused confirmation (empty after a good one)."""
        return list(self._problems)

    # -- actions ------------------------------------------------------------
    def selected_step(self) -> Optional[int]:
        """Step of the selected entry, falling back to the frame being shown."""
        item = self._lists[self.current_queue()].currentItem()
        if item is not None:
            return int(item.data(STEP_ROLE))
        if session_is_open(self._session):
            return int(self._session.current().step)
        return None

    def selected_visibility(self) -> Optional[dict]:
        """The selected 可见性审查 entry (``instance``, ``steps``, ...), or ``None``."""
        item = self._lists[api.QUEUE_VISIBILITY].currentItem()
        found = None if item is None else item.data(ENTRY_ROLE)
        return dict(found) if isinstance(found, dict) else None

    def select_visibility(self, instance: str, steps=()) -> bool:
        """Select the line of ``instance`` whose run holds one of ``steps``.

        Without such a line the nearest later line of the part, then its
        nearest earlier one, then the line that took its place; ``False``
        when the list is empty. No :attr:`sigVisibilityPicked`: this is not
        the annotator picking.
        """
        lw = self._lists[api.QUEUE_VISIBILITY]
        wanted = [int(s) for s in steps or ()]
        lines = [(row, lw.item(row).data(ENTRY_ROLE) or {}) for row in range(lw.count())]
        pick = None
        for row, entry in lines:
            if entry.get("instance") == instance and set(wanted) & set(entry.get("steps") or ()):
                pick = row
                break
        if pick is None and wanted:
            mine = [(row, entry) for row, entry in lines if entry.get("instance") == instance]
            later = [row for row, entry in mine if min(entry.get("steps") or [0]) > min(wanted)]
            earlier = [row for row, entry in mine if row not in later]
            pick = (later[0] if later else earlier[-1] if earlier else None)
        if pick is None and lines:
            after = [row for row, entry in lines if str(entry.get("instance")) > instance]
            pick = after[0] if after else lines[-1][0]
        quiet, self._refreshing = self._refreshing, True
        try:
            lw.setCurrentRow(-1 if pick is None else pick)
        finally:
            self._refreshing = quiet
        return pick is not None

    def show_visibility(self) -> None:
        """Put the 可见性审查 tab on top."""
        self._tabs.setCurrentIndex(list(api.QUEUE_NAMES).index(api.QUEUE_VISIBILITY))

    def selected_conflict(self) -> Optional[int]:
        """Id of the selected conflict, or ``None``."""
        item = self._lists[api.QUEUE_CONFLICTS].currentItem()
        cid = None if item is None else item.data(CID_ROLE)
        return None if cid is None else int(cid)

    def select_conflict(self, cid: int) -> bool:
        """Put the Conflicts tab on top with conflict ``cid`` selected.

        What a click on a conflict line of the task card asks for (U2e), so
        that ``K`` / ``N`` act on that one.  ``False`` when the queue does not
        list it (it was settled meanwhile).
        """
        lw = self._lists[api.QUEUE_CONFLICTS]
        self._tabs.setCurrentIndex(list(api.QUEUE_NAMES).index(api.QUEUE_CONFLICTS))
        for row in range(lw.count()):
            if lw.item(row).data(CID_ROLE) is not None and int(lw.item(row).data(CID_ROLE)) == int(cid):
                lw.setCurrentRow(row)
                return True
        return False

    def select_current_step(self) -> None:
        """Put the highlight back on the frame that is open, after a refusal.

        Qt selects the row before the click is delivered, so a refused move left
        the queue pointing at a step the canvas is not showing.  Only the list
        the annotator actually activated is touched: restoring all four cleared
        a conflict they had picked in another tab, and the ``K`` / ``N`` keys
        then had nothing to act on.
        """
        lw, self._activated = self._activated, None
        if lw is None or not session_is_open(self._session):
            return
        step = int(self._session.current().step)
        rows = [r for r in range(lw.count())
                if int(lw.item(r).data(STEP_ROLE)) == step]
        lw.setCurrentRow(rows[0] if rows else -1)

    def rework(self) -> None:
        """Mark the selected step for rework (``R``)."""
        step = self.selected_step()
        if step is not None:
            self.sigRework.emit(int(step))

    # -- keys ---------------------------------------------------------------
    # There is exactly one key map, and it is not here: the main window's
    # ``tda.ui.app_actions.ACTIONS`` table owns every binding, per mode, and
    # calls the plain methods above.  A second table in the panel is how ``Ctrl+K``
    # on this list came to raise out of ``keyPressEvent`` in Steps mode and how
    # ``Enter`` in Review committed with keyframe scope past ``act_commit``.


    # -- slots --------------------------------------------------------------
    def _on_item_activated(self, item: QListWidgetItem) -> None:
        step = item.data(STEP_ROLE)
        if step is None or int(step) < 0:
            return
        # Remember which list asked, so that a refusal puts back that one and
        # leaves the other three tabs' selections alone.
        self._activated = item.listWidget()
        self.sigOpenStep.emit(int(step))

    def _on_visibility_current(self, item: Optional[QListWidgetItem], _previous) -> None:
        """The annotator selected a 可见性审查 line: its frame and part, please.

        Asked from the event loop, not from inside this signal: the frame
        change it causes rebuilds this very list, and clearing a list while it
        is still delivering its own ``currentItemChanged`` deletes the item
        being delivered (an access violation, not an exception).
        """
        if self._refreshing:
            return
        if item is None:
            self._show_detail(None)
            return
        entry = item.data(ENTRY_ROLE) or {}
        steps = list(entry.get("steps") or ())
        if not steps or not entry.get("instance"):
            return
        self._show_detail(dict(entry))
        here = (int(self._session.current().step) if session_is_open(self._session)
                else None)
        self._picked = (here if here in steps else int(steps[0]), str(entry["instance"]))
        self._pick_timer.start()

    def _emit_picked(self) -> None:
        picked, self._picked = self._picked, None
        if picked is not None:
            self.sigVisibilityPicked.emit(*picked)

    def _on_frame_changed(self, _key: object) -> None:
        self._activated = None      # the move happened: there is nothing to put back
        self._follow_note()
        self.refresh()

    # -- 这一帧的说明 (task U35) ------------------------------------------------
    def _note_of(self, key) -> Optional[dict]:
        """The stored note of the frame ``key``, or ``None``."""
        finder = getattr(self._session, "frame_note", None)
        if key is None or not callable(finder):
            return None
        try:
            return finder(key=key)
        except Exception:  # noqa: BLE001 - a note is never worth a failure here
            return None

    def _follow_note(self) -> None:
        """Show the note of the frame on screen; hand on what was typed for the last one.

        A box typed in and not saved, about a frame that is no longer on
        screen, is reported first (:attr:`sigNoteLeft`) -- the window saves
        it synchronously -- and only then replaced.  On the same frame the
        typing is kept.
        """
        key = self._session.current() if session_is_open(self._session) else None
        box = self.note_box
        old = box.key()
        if key is not None and old == key:
            box.note_changed(self._note_of(key))
            return
        if old is not None and box.is_dirty():
            self.sigNoteLeft.emit(old, box.text())
        box.show_frame(key, self._note_of(key))

    def _on_note_changed(self, key: object) -> None:
        """The session wrote (or an undo put back) the note of ``key``."""
        if key is not None and key == self.note_box.key():
            self.note_box.note_changed(self._note_of(key))

    def _on_problems(self, problems: list) -> None:
        self._problems = [str(p) for p in problems]

    def _show_problems(self, problems: list[str]) -> None:
        self._problems_list.clear()
        for problem in problems:
            self._problems_list.addItem(problem)
        visible = bool(problems)
        self._problems_label.setVisible(visible)
        self._problems_list.setVisible(visible)
