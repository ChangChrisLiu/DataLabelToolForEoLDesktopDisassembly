"""The 「主视角」 dock in the main window (task U23): when it shows, and for what.

The panel itself (:mod:`tda.ui.panels.locator`) draws a scene and the data
comes from :class:`~tda.ui.locator_source.LocatorSource`; what only the window
can decide is here:

* **where it shows.**  On a frame of the teardown seen by another camera than
  the scanner -- ``oak1``, ``oak2``, ``rs`` -- in Annotate or Review.  Never on
  the scanner view itself (it *is* the scanner view), never on a
  components-gallery shot (a shot is not a step), never in Steps (no frame on
  screen).
* **whether the annotator wants it.**  The title bar's close button, the
  window's own dock menu and the 「主视角」 toggle on the top bar all set one
  flag, kept in the window's INI (``locator/open``, open by default) so a
  closed panel stays closed in the next session.  Where it sits -- docked at
  the foot of the right column, floated, resized -- is the dock layout the
  window already saves.
* **what for.**  The part being drawn; else the instance table's selection;
  else the task card's current row -- the order every key that acts on "the
  part" reads it in.
* **when.**  :meth:`follow_locator` runs at the end of every
  :meth:`~tda.ui.app_guide.GuideMixin.refresh_guidance` -- the one place a
  frame change, a selection, a card row or an edit all end up -- and does
  nothing unless the panel is on screen and its ``(machine, view, step, part,
  edit epoch)`` moved.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton

from tda.core.model import VIEWS
from tda.ui import app_actions as A
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.locator_source import SCAN, LocatorScene, LocatorSource
from tda.ui.panels.locator import TITLE, LocatorDock, LocatorPanel

__all__ = ["LOCATOR_HEIGHT", "LOCATOR_MODES", "LOCATOR_VIEWS", "LocatorMixin",
           "NOT_ON_THIS_VIEW", "SETTING", "TOGGLE_TIP", "locator_toggle_button"]

#: The views the panel shows on: the sequence's cameras other than the scanner.
LOCATOR_VIEWS: tuple[str, ...] = tuple(v for v in VIEWS if v != SCAN)
#: ... and the modes: a frame is on screen in both.
LOCATOR_MODES: tuple[str, ...] = (A.MODE_ANNOTATE, A.MODE_REVIEW)
#: The INI key of "the annotator wants the panel".
SETTING = "locator/open"
#: Asked for in the default layout, at the foot of the right column.
LOCATOR_HEIGHT = 260
TOGGLE_TIP = ("主视角：在 OAK / RealSense 的帧旁边显示扫描仪视角里这个零件在哪（点一下开 / 关）"
              " / show where this part is in the scanner view")
NOT_ON_THIS_VIEW = ("主视角只在 OAK / RealSense 的拆解帧上显示（标注、复查模式）"
                    " / the scanner locator shows on oak1, oak2 and rs frames")


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
        self.locator_wanted = _as_bool(self.settings.value(SETTING), True)
        #: What the panel was last refreshed for; ``None`` forces the next one.
        self._locator_key: Optional[tuple] = None
        #: Refreshes actually done -- a scene read and drawn (tests, report).
        self.locator_refreshes = 0
        self._locator_ready = True
        self._sync_locator_button(False)

    # ------------------------------------------------------------ the rules
    def locator_applies(self) -> bool:
        """Is this a frame the panel belongs next to (view, mode, an open session)?"""
        if not compat.is_open(self.session):
            return False
        return (str(getattr(self.session, "view", "")) in LOCATOR_VIEWS
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

    # ------------------------------------------------------------- follow
    def follow_locator(self, force: bool = False) -> None:
        """Show or hide the panel by the rules, and refresh it when its key moved.

        Nothing is read while it is hidden -- closed, on the scanner view, on a
        gallery shot, in Steps -- and nothing when its key has not moved.
        """
        if not getattr(self, "_locator_ready", False) or self.closed:
            return
        applies = self.locator_applies()
        show = applies and bool(self.locator_wanted)
        if self.locator_dock.isHidden() == show:
            self.locator_dock.setVisible(show)
        self._sync_locator_button(applies)
        if not show:
            self._locator_key = None
            return
        key = self.session.current()
        target = self.locator_target()
        stamp = (int(key.desktop), str(key.view), int(key.step), target,
                 getattr(self.session, "_epoch", None))
        if not force and stamp == self._locator_key:
            return
        self._locator_key = stamp
        self.refresh_locator(int(key.desktop), int(key.step), target)

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

    def _want_locator(self, wanted: bool) -> None:
        if self.closed:          # a floating dock closed with the window is no wish
            return
        self.locator_wanted = bool(wanted)
        self.settings.setValue(SETTING, bool(wanted))
        self.follow_locator(force=True)

    @S.guard
    def act_toggle_locator(self, checked: bool) -> None:
        """The top bar's 「主视角」: open or close the panel, remembered."""
        self._want_locator(bool(checked))
        if checked and not self.locator_applies():
            self.report(NOT_ON_THIS_VIEW)

    @S.guard
    def _on_locator_closed(self) -> None:
        """The dock's close button: the annotator does not want it."""
        self._want_locator(False)

    @S.guard
    def _on_locator_menu(self, checked: bool) -> None:
        """The window's dock menu ticked or unticked it."""
        self._want_locator(bool(checked))
