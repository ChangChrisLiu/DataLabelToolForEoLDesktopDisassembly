"""The 「主视角」 dock in the main window (tasks U23, U28): when it shows, and what.

The panel itself (:mod:`tda.ui.panels.locator`) draws a scene and the data
comes from :class:`~tda.ui.locator_source.LocatorSource`; what only the window
can decide is here:

* **which scene.**  Two kinds.  The *location* (U23): where this step's part
  is on the scanner frame -- on an ``oak1`` / ``oak2`` / ``rs`` frame of the
  teardown.  The *pair* 「拆前 / 拆后」 (U28): the scanner frames just before
  and just after the part leaves, so the annotator can tell which screw "CPU
  heatsink screw 3" is.  The pair is up, on **any** view of the teardown the
  scanner's own included, when the part has no shape of its own in the view
  on screen yet -- the first time it is annotated there (a keyframe, a cut
  copy of one, or one frame's own pixels all count as a shape) -- or when the
  annotator asked for it with the toggle next to the panel's caption; the
  toggle also puts the location back.  Its choice is kept by ``(machine,
  part)`` -- it follows the part to the other views -- until the part on
  screen is another.  Otherwise the location is up on the OAK / RealSense
  views and the dock is down on the scanner view (it *is* the scanner view);
  while it is down there only because the part is drawn, the top bar's
  「主视角」 asks for that part's pair instead of closing anything.  Never on
  a components-gallery shot (a shot is not a step), never in Steps (no frame
  on screen).
* **whether the annotator wants it.**  The title bar's close button, the
  window's own dock menu and the 「主视角」 toggle on the top bar all set one
  flag, kept in the window's INI (``locator/open``, open by default) so a
  closed panel stays closed in the next session -- for both kinds of scene.
  Where it sits -- docked at the foot of the right column, floated, resized --
  is the dock layout the window already saves.
* **what for.**  The part being drawn; else the instance table's selection
  (a ``Removed`` row included); else the task card's current row, the first
  one not done.  A card row that names no part (a dupli frame's) shows the
  frame with nothing marked (the location), and no pair.
* **when.**  :meth:`follow_locator` runs at the end of every
  :meth:`~tda.ui.app_guide.GuideMixin.refresh_guidance` -- the one place a
  frame change, a selection, a card row or an edit all end up -- and does
  nothing unless the panel's ``(machine, view, step, part, edit epoch)``
  moved; the refresh then waits for the event loop, so a frame change --
  which passes through here while the card and the instance table are still
  being rebuilt -- is one refresh, for the part it ends on.  Whether the part
  has a shape in the view is one indexed lookup per part and edit epoch, and
  the dock is shown or hidden for it in that same settled refresh, not on the
  way there.  Nothing else is read while the dock is down.
* **detection chips** (round 3).  When the pair's window is a guess and the
  window's small-part detector knows the part's class, the vanished
  detections between the two frames are asked of
  :class:`~tda.ui.locator_dets.LocatorDetections` (the detector's disk cache,
  read only, else its model on a worker thread) and added as chips when they
  arrive -- unless the dock has moved on to other frames or another class by
  then.  Answers are kept by ``(machine, frames, class, crops)``.
"""
from __future__ import annotations

import weakref
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QToolButton

from tda.core.model import VIEWS, FrameKey, is_component_view
from tda.models.det_engine import PlanItem
from tda.models.detector import frame_crop, make_detector
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.class_names import class_zh
from tda.ui.locator_dets import DET_LABEL, LocatorDetections, ReadOnlyEngine
from tda.ui.locator_pair import PAIR_HEAD, PAIR_HEAD_FIRST, PairScene
from tda.ui.session_images import frame_candidates
from tda.ui.locator_source import SCAN, LocatorScene, LocatorSource
from tda.ui.panels.locator import TITLE, LocatorDock, LocatorPanel
from tda.ui.panels.locator_pair import PAIR_TIP

__all__ = ["KIND_PAIR", "KIND_SCENE", "LOCATOR_HEIGHT", "LOCATOR_MODES", "LOCATOR_VIEWS",
           "LocatorMixin", "NOT_ON_THIS_VIEW", "PAIR_VIEWS", "SETTING", "TOGGLE_TIP",
           "TOGGLE_TIP_DOWN", "TOP_NEEDS_PART", "locator_toggle_button"]

#: The views the location scene shows on: the sequence's cameras other than the scanner.
LOCATOR_VIEWS: tuple[str, ...] = tuple(v for v in VIEWS if v != SCAN)
#: ... and the 「拆前 / 拆后」 pair (task U28): every view of the teardown.
PAIR_VIEWS: tuple[str, ...] = tuple(VIEWS)
#: ... and the modes: a frame is on screen in both.
LOCATOR_MODES: tuple[str, ...] = (A.MODE_ANNOTATE, A.MODE_REVIEW)
#: The two kinds of scene.
KIND_SCENE = "scene"
KIND_PAIR = "pair"
#: The INI key of "the annotator wants the panel".
SETTING = "locator/open"
#: Asked for in the default layout, at the foot of the right column.
LOCATOR_HEIGHT = 260
TOGGLE_TIP = ("主视角：在 OAK / RealSense 的帧旁边显示扫描仪视角里这个零件在哪；第一次标某个零件时"
              "（任何视角）显示它被拆走前后的两帧（点一下开 / 关）"
              " / show where this part is in the scanner view, and the frames around its removal")
NOT_ON_THIS_VIEW = ("主视角只在拆解帧上显示（标注、复查模式）"
                    " / the scanner locator shows on frames of the teardown")
PAIR_TIP_NO_PART = "拆前/拆后：先选中一个零件 / select a part first"
#: The top bar's 「主视角」 on the scanner view while the part is drawn there (N2).
TOGGLE_TIP_DOWN = ("主视角：扫描仪视角里已经画过这个零件，小窗收着；点一下看它的拆前 / 拆后"
                   " / show this part's before / after frames")
#: ... pressed there with no part to show it for.
TOP_NEEDS_PART = "先选中一个零件，再按「主视角」看它的拆前 / 拆后 / select a part first"


def locator_toggle_button() -> QToolButton:
    """The top bar's 「主视角」 toggle; the window connects it."""
    button = QToolButton()
    button.setText(TITLE)
    button.setCheckable(True)
    button.setChecked(True)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setToolTip(TOGGLE_TIP)
    button.setObjectName("locator_toggle")
    return button


def _as_bool(value, default: bool = True) -> bool:
    """An INI value as a bool: ``QSettings`` hands booleans back as strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class LocatorMixin:
    """The 「主视角」 dock's place in :class:`tda.ui.app.MainWindow`."""

    def _init_locator(self) -> None:
        """Build the dock at the foot of the right column (before the layout is restored)."""
        self.locator = LocatorPanel()
        # The directory the session reads its own frames from: the same local
        # cache, the same fallbacks to the recorded paths.
        self.locator_source = LocatorSource(
            self.db, getattr(self.session, "tax", None), getattr(self.session, "truth", None),
            getattr(self.session, "cache_dir", None) or self.paths.get("cache_dir", ""))
        dock = LocatorDock(f"{TITLE} / Scanner", self)
        dock.setObjectName("dock_locator")     # the key its place is saved under
        dock.setWidget(self.locator)
        dock.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.locator_dock = dock
        dock.sigClosedByUser.connect(self._on_locator_closed)
        # The window's own dock menu (right-click a dock title or the top bar).
        dock.toggleViewAction().triggered.connect(self._on_locator_menu)
        self.locator.sigPairToggled.connect(self.act_locator_pair)
        self.locator_wanted = _as_bool(self.settings.value(SETTING), True)
        #: What the panel was last refreshed for; ``None`` forces the next one.
        self._locator_key: Optional[tuple] = None
        #: Refreshes actually done -- a scene read and drawn (tests, report).
        self.locator_refreshes = 0
        #: Has-a-shape lookups done (task U28): one per part and edit epoch.
        self.locator_checks = 0
        #: ``(desktop, view, part, epoch) -> (a part, has a shape here)``, one entry.
        self._locator_own: Optional[tuple] = None
        #: The toggle's choice: ``(desktop, part, pair wanted)``, for that part only.
        self._pair_choice: Optional[tuple] = None
        #: What the pair on screen was drawn for: ``(desktop, view, part, epoch, asked)``.
        self._locator_pair_shown: Optional[tuple] = None
        #: Round 3: the detection chips' worker, its answers by key, the keys
        #: asked and not answered, and counts for the tests and the report.
        self._locator_det: Optional[LocatorDetections] = None
        self._locator_det_answers: dict = {}
        self._locator_det_asked: set = set()
        self.locator_det_requests = 0
        self.locator_det_dropped = 0
        self._locator_timer = QTimer(self)
        self._locator_timer.setSingleShot(True)
        self._locator_timer.setInterval(0)
        self._locator_timer.timeout.connect(self._refresh_locator_now)
        self._locator_ready = True
        self._sync_locator_button(False)

    # ------------------------------------------------------------ the rules
    def locator_applies(self) -> bool:
        """Is a frame of the teardown on screen, in a mode the panel belongs to?"""
        if not compat.is_open(self.session):
            return False
        view = str(getattr(self.session, "view", ""))
        return (view in PAIR_VIEWS and not is_component_view(view)
                and self.mode in LOCATOR_MODES)

    def locator_target(self) -> Optional[str]:
        """The part the panel looks for: the edit's, the selection's, the card's."""
        editing = getattr(self.session, "editing_instance", None)
        if editing:
            return str(editing)
        selected = (self.instances.selected_instance()
                    or self.instances.selected_removed_instance())
        if selected:
            return str(selected)
        card = self.task_card.current_instance()
        return str(card) if card else None

    def locator_shown(self) -> bool:
        """Is the panel up (docked or floating), whatever the window's own state?"""
        return bool(getattr(self, "_locator_ready", False)) and not self.locator_dock.isHidden()

    def _locator_own_shapes(self, target: Optional[str], fresh: bool = False) -> tuple[bool, bool]:
        """``(a part, it has a shape in the view on screen)``, one lookup per epoch."""
        if not target:
            return False, False
        key = self.session.current()
        memo = (int(key.desktop), str(key.view), str(target),
                getattr(self.session, "_epoch", None))
        held = self._locator_own
        if not fresh and held is not None and held[0] == memo:
            return held[1]
        self.locator_checks += 1
        found = self.locator_source.own_shapes(memo[0], memo[1], memo[2])
        self._locator_own = (memo, found)
        return found

    def locator_first_time(self, target: Optional[str] = None, fresh: bool = False) -> bool:
        """Is the part met for the first time in this view (no shape of its own here)?"""
        target = self.locator_target() if target is None else target
        is_part, has_shape = self._locator_own_shapes(target, fresh)
        return is_part and not has_shape

    def locator_kind(self, fresh: bool = False) -> Optional[str]:
        """:data:`KIND_PAIR`, :data:`KIND_SCENE`, or ``None`` (the dock is down)."""
        if not self.locator_applies() or not self.locator_wanted:
            return None
        target = self.locator_target()
        if self._wants_pair(target, fresh):
            return KIND_PAIR
        return KIND_SCENE if str(self.session.view) in LOCATOR_VIEWS else None

    def _wants_pair(self, target: Optional[str], fresh: bool = False) -> bool:
        is_part, has_shape = self._locator_own_shapes(target, fresh)
        if not is_part:
            return False
        choice = self._pair_choice
        if choice is not None and choice[:2] == (int(self.session.current().desktop),
                                                 str(target)):
            return bool(choice[2])
        return not has_shape

    # ------------------------------------------------------------- follow
    def _locator_stamp(self) -> tuple:
        """``(machine, view, step, part, edit epoch)``: what a scene is valid for.

        The kind of scene is a function of these and of the annotator's
        choices, which settle at once (``force``) when they change.
        """
        key = self.session.current()
        return (int(key.desktop), str(key.view), int(key.step), self.locator_target(),
                getattr(self.session, "_epoch", None))

    def follow_locator(self, force: bool = False) -> None:
        """Show or hide the panel by the rules, and refresh it when its key moved.

        Nothing is read while it is down for the view, the mode or the
        annotator's wish -- closed, on a gallery shot, in Steps -- and those
        hide it at once.  Whether *the part* wants it up (a pair on the
        scanner view; the location or a pair elsewhere) is settled with the
        refresh, which waits for the event loop (:attr:`_locator_timer`): one
        frame change passes through here two or three times while the card
        and the instance table are rebuilt -- the old selection, no card row,
        the new card row -- and only the last of those is the part to look
        for, so the dock never blinks for the ones in between.  ``force``
        settles now.
        """
        if not getattr(self, "_locator_ready", False) or self.closed:
            return
        applies = self.locator_applies()
        self._sync_locator_button(applies)
        if not applies or not self.locator_wanted:
            self._hide_locator()
            return
        if force:
            self._locator_timer.stop()
            self._refresh_locator_now(fresh=True)
        elif self._locator_stamp() != self._locator_key:
            self._locator_timer.start()

    def _hide_locator(self) -> None:
        if not self.locator_dock.isHidden():
            self.locator_dock.setVisible(False)
        self._locator_key = None
        self._locator_timer.stop()

    @S.guard
    def _refresh_locator_now(self, fresh: bool = False) -> None:
        """The deferred half of :meth:`follow_locator`: show / hide and refresh, once."""
        if self.closed or not getattr(self, "_locator_ready", False):
            return
        if not self.locator_applies() or not self.locator_wanted:
            self._hide_locator()
            return
        target = self.locator_target()
        stamp = self._locator_stamp()
        choice = self._pair_choice
        if choice is not None and choice[:2] != (stamp[0], target):
            self._pair_choice = None           # the toggle answered for another part
        kind = self.locator_kind(fresh)
        self._locator_key = stamp
        self._tip_locator_button(kind is None)
        if kind is None:                       # the scanner view, the part drawn there
            if not self.locator_dock.isHidden():
                self.locator_dock.setVisible(False)
            self._sync_pair_button(None, target)
            return
        if self.locator_dock.isHidden():
            self.locator_dock.setVisible(True)
        if kind == KIND_PAIR:
            # the pair does not depend on the step: walking frames with the
            # same part, view, edit epoch and scanner pose segments (an ROI
            # stored is no edit, N13) leaves it as it is -- one small query
            shown = (stamp[0], stamp[1], stamp[3], stamp[4], self._pair_forced(stamp[3]),
                     self.locator_source.scan_poses(stamp[0]))
            if not (fresh or shown != self._locator_pair_shown
                    or not self.locator.showing_pair()):
                self._sync_pair_button(kind, target)
                return
            self.refresh_pair(stamp[0], stamp[3])
            self._locator_pair_shown = shown
        else:
            self.refresh_locator(stamp[0], stamp[2], stamp[3])
        self._sync_pair_button(kind, target)

    def refresh_locator(self, desktop: int, step: int, target: Optional[str]) -> None:
        """Read the scene and draw it; a failure is a sentence in the panel, not a crash."""
        self.locator_refreshes += 1
        try:
            scene = self.locator_source.scene(desktop, step, target)
        except Exception as exc:  # noqa: BLE001 - a side panel never stops the window
            self.logger.warning("locator: D%s step %s %s: %s", desktop, step, target, exc)
            scene = LocatorScene(desktop=desktop, asked_step=step, caption=TITLE,
                                 note=f"主视角读不出来：{exc}")
        self.locator.show_scene(scene)

    def refresh_pair(self, desktop: int, target: str) -> None:
        """Read the part's 「拆前 / 拆后」 and draw it (task U28); never a crash."""
        self.locator_refreshes += 1
        try:
            pair = self.locator_source.pair(desktop, target)
            pair = self._locator_detected(pair)
        except Exception as exc:  # noqa: BLE001 - a side panel never stops the window
            self.logger.warning("locator pair: D%s %s: %s", desktop, target, exc)
            pair = PairScene(desktop=desktop, instance=target,
                             note=f"拆前 / 拆后读不出来：{exc}")
        self.locator.show_pair(pair, PAIR_HEAD if self._pair_forced(target) else PAIR_HEAD_FIRST)

    # ------------------------------------------- vanished detections (round 3)
    def locator_detector(self) -> Optional[LocatorDetections]:
        """The worker for the pair's detection chips, built on first need; ``None`` when off.

        It goes with the window's small-part detector: same configuration,
        same model, same on-disk cache (read only), its own engine and thread
        -- so the reverse walk's background pass is never interrupted.  No
        detector in the window (switched off, failed) means no chips.
        """
        worker = getattr(self, "det_worker", None)
        config = getattr(self, "det_config", None)
        if worker is None or config is None or SCAN not in tuple(config.views):
            return None
        held = self._locator_det
        if held is not None and held.engine.config is config:
            return held
        if held is not None:
            # never a join on the GUI thread (N17): a job in hand finishes on
            # its own and its answer is not sent
            held.sigDone.disconnect(self._on_locator_dets)
            held.shutdown(wait=False)
            self._locator_det_asked.clear()
        base = worker.engine
        engine = ReadOnlyEngine(config, base.cache_root,
                                factory=None if base.factory is make_detector else base.factory,
                                identity=base.identity)
        me = weakref.ref(self)
        held = LocatorDetections(engine, config.conf, parent=self,
                                 after=getattr(self, "_sam_turn", None),
                                 stop_when=lambda: me() is None or bool(me().closed),
                                 ready=lambda: me() is None or me().window_detector_idle())
        held.sigDone.connect(self._on_locator_dets)
        self._locator_det = held
        self._locator_det_answers.clear()
        return held

    def window_detector_idle(self) -> bool:
        """Has the window's own detector nothing left to do? (U29, N18; any thread)

        The model loaded (or failed) and the pass over the view on screen run
        out -- or no detector at all. The dock's worker builds its own model
        only once this has been true (it asks until it is, then never again):
        the window's build and its first pass go first, and the frames that
        pass writes to the disk cache are read from there, not computed twice.
        """
        worker = getattr(self, "det_worker", None)
        try:
            return worker is None or not worker.pending()
        except Exception:  # noqa: BLE001 - a worker torn down is an idle one
            return True

    def _locator_det_key(self, pair: PairScene) -> Optional[tuple]:
        """What a pair's detection chips are asked by, or ``None`` when it takes none."""
        if (pair.leaving is None or not pair.compared or pair.window_source == "part"
                or None in tuple(pair.steps) or not pair.cls):
            return None
        config = getattr(self, "det_config", None)
        if config is None or pair.cls not in tuple(config.classes):
            return None
        before, after = (int(s) for s in pair.steps)
        crops = tuple(frame_crop(config, self.locator_source.roi(pair.desktop, s))
                      for s in (before, after))
        return (int(pair.desktop), before, after, pair.cls, crops)

    def _locator_detected(self, pair: PairScene) -> PairScene:
        """``pair`` with the chips of its vanished detections when they are on hand.

        Otherwise they are asked for -- off the GUI thread -- and this pair goes
        up as it is; :meth:`_on_locator_dets` adds them when they arrive.
        """
        key = self._locator_det_key(pair)
        if key is None:
            return pair
        found = self._locator_det_answers.get(key)
        if found is not None:
            return pair.with_detections(found, DET_LABEL.format(cls=class_zh(pair.cls)))
        detector = self.locator_detector()
        if detector is None or key in self._locator_det_asked:
            return pair
        items = tuple(self._locator_det_item(pair.desktop, step, crop)
                      for step, crop in zip(key[1:3], key[4]))
        self._locator_det_asked.add(key)
        self.locator_det_requests += 1
        displaced = detector.request((key, pair.instance), pair.desktop, items[0], items[1],
                                     pair.cls)
        if displaced is not None:
            # a queued request this one replaced gets no answer: forget it was
            # asked, so its pair asks again when it is back (B3)
            self._locator_det_asked.discard(displaced[0])
        return pair

    def _locator_det_item(self, desktop: int, step: int, crop) -> PlanItem:
        """A scanner frame as the detector engine wants it (where its pixels are)."""
        key = FrameKey(int(desktop), int(step), SCAN)
        cache_dir = getattr(self.session, "cache_dir", "") or self.paths.get("cache_dir", "")
        return PlanItem(step=int(step), crop=crop,
                        candidates=tuple(frame_candidates(cache_dir, key, self.db.get_frame(key))))

    @S.guard
    def _on_locator_dets(self, token: object, found: object) -> None:
        """An answer: kept, and drawn only if the pair it is for is still up."""
        key, instance = token  # type: ignore[misc]
        self._locator_det_asked.discard(key)
        if found is None:               # the detector failed: no chips, nothing said
            return
        self._locator_det_answers[key] = list(found)
        while len(self._locator_det_answers) > 64:
            self._locator_det_answers.pop(next(iter(self._locator_det_answers)))
        if self.closed or not self.locator_shown() or not self.locator.showing_pair():
            return
        current = self.locator.pair()
        # the dock moved on to a pair of other frames or another class: not its
        # answer (another of the screws that leave on the same step, with the
        # same two frames, has the very same vanished detections)
        if current is None or self._locator_det_key(current) != key:
            self.locator_det_dropped += 1
            return
        self.locator.show_pair(
            current.with_detections(found, DET_LABEL.format(cls=class_zh(current.cls))),
            self.locator.caption_text())

    def _pair_forced(self, target: Optional[str]) -> bool:
        """Is the pair up because the annotator asked, not because it is a first time?"""
        return (self._pair_choice is not None and bool(self._pair_choice[2])
                and not self.locator_first_time(target))

    def _sync_pair_button(self, kind: Optional[str], target: Optional[str]) -> None:
        """The panel's 「拆前/拆后」 is checked while the pair is up, live with a part."""
        is_part = self._locator_own_shapes(target)[0] if target else False
        self.locator.set_pair_button(kind == KIND_PAIR, is_part,
                                     PAIR_TIP if is_part else PAIR_TIP_NO_PART)

    # ---------------------------------------------------------- the wish
    def _sync_locator_button(self, applies: bool) -> None:
        """The toggle says what the annotator wants, and is live where it can show."""
        button = getattr(self, "locator_button", None)
        if button is None:
            return
        if button.isChecked() != bool(self.locator_wanted):
            blocked = button.blockSignals(True)
            button.setChecked(bool(self.locator_wanted))
            button.blockSignals(blocked)
        if button.isEnabled() != bool(applies):
            button.setEnabled(bool(applies))
            button.setToolTip(TOGGLE_TIP if applies else f"{TOGGLE_TIP}\n{NOT_ON_THIS_VIEW}")

    def _tip_locator_button(self, down_for_part: bool) -> None:
        """The top bar's tooltip says what a press does now (N2)."""
        button = getattr(self, "locator_button", None)
        if button is None or not button.isEnabled():
            return
        tip = TOGGLE_TIP_DOWN if down_for_part else TOGGLE_TIP
        if button.toolTip() != tip:
            button.setToolTip(tip)

    def _want_locator(self, wanted: bool) -> None:
        if self.closed:          # a floating dock closed with the window is no wish
            return
        self.locator_wanted = bool(wanted)
        self.settings.setValue(SETTING, bool(wanted))
        self.follow_locator(force=True)

    def locator_down_for_part(self) -> bool:
        """Is the dock down only because of the part (the scanner view, drawn there)?

        The annotator wants it and the frame is one it belongs next to; the
        part on screen just has nothing to show there.  In that state the top
        bar's 「主视角」 asks for the part's 「拆前 / 拆后」 instead of closing
        a dock that is not even up (round 2, N2).
        """
        return (bool(getattr(self, "_locator_ready", False)) and not self.closed
                and self.locator_applies() and bool(self.locator_wanted)
                and self.locator_dock.isHidden())

    @S.guard
    def act_toggle_locator(self, checked: bool) -> None:
        """The top bar's 「主视角」: open or close the panel, remembered.

        While the dock is down only because of the part (:meth:`locator_down_for_part`)
        a press shows that part's 「拆前 / 拆后」 and leaves the wish
        (``locator/open``) as it is.
        """
        if self.locator_down_for_part():
            self._sync_locator_button(True)          # the click unticked it: the wish stands
            target = self.locator_target()
            if not target or not self._locator_own_shapes(target)[0]:
                self.report(TOP_NEEDS_PART)
                return
            self._pair_choice = (int(self.session.current().desktop), str(target), True)
            self.follow_locator(force=True)
            return
        self._want_locator(bool(checked))
        if checked and not self.locator_applies():
            self.report(NOT_ON_THIS_VIEW)

    @S.guard
    def act_locator_pair(self, checked: bool) -> None:
        """The panel's 「拆前/拆后」: the pair for this part, or not (task U28).

        The choice is the annotator's for the part on screen and holds until
        the part changes; on the scanner view taking the pair away hides the
        dock, which has nothing else to show there.
        """
        if self.closed or not compat.is_open(self.session):
            return
        target = self.locator_target()
        if not target or not self._locator_own_shapes(target)[0]:
            self._sync_pair_button(None, target)
            return
        self._pair_choice = (int(self.session.current().desktop), str(target), bool(checked))
        self.follow_locator(force=True)

    @S.guard
    def _on_locator_closed(self) -> None:
        """The dock's close button: the annotator does not want it."""
        self._want_locator(False)

    @S.guard
    def _on_locator_menu(self, checked: bool) -> None:
        """The window's dock menu ticked or unticked it."""
        self._want_locator(bool(checked))
