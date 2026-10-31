"""Stage S1 panel: "步骤与实例核对" -- review the imported log of one desktop.

Three tables behind a :class:`QTabWidget` (spec 4.1):

* **Steps** -- one row per action, grouped under its logical step, with the
  scanner thumbnails of frames k-1 and k, the parsed target, verb, tool,
  direction, result, failure reason, difficulty and the operator's notes;
* **Instances** -- the desktop's instance table with the relational attributes
  of spec 7.1 (parent/attached, mounted_on, fastens, socket_host, cable, screw
  head and captive flag, group order, removal direction);
* **Relations** -- stage S6: the constraint graph, with the spec 7.4 replay
  below it (:mod:`tda.ui.panels.relations`).

Below them sits the list of open questions, re-derived after every edit. The
context menus resolve them in place: split a compound row into N actions, add
or remove an action, retarget a step at a new instance, delete an instance
nothing points at any more, merge two instances that are one part
(「合并到…」, :mod:`tda.ui.panels.merge_into`, written through), say a drive
latch is the cage or the other way round (「其实是硬盘架…」 / 「其实是卡扣…」,
:mod:`tda.ui.panels.reclass_menu`, written through, task U21), say a part the
sheet logs is not in this machine at all (「这台没有这个零件…」 / 「其实有…」,
:mod:`tda.ui.panels.absent_menu`, written through, task U34). The
``＋ 添加零件 / Add parts`` button above the
Instances table stages parts the picture shows and the log never names
(:mod:`tda.ui.panels.add_parts`, :mod:`tda.core.extra`); its 「框外零件…」
hands over to the window for an object outside the machine (task U18), which
the Instances table then lists in a block of its own -- deletable with its
shapes, its name editable, named by no step, relation or question. ``Apply`` writes
everything back in one transaction and recompiles the automatic state-event log (emitting
:attr:`StepTablePanel.sigSaved`); ``Revert`` reloads from the database.

The models and delegates live in :mod:`tda.ui.panels.steptable_models` and all
editing logic in :mod:`tda.ui.steps_model`; this module is layout and commands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tda.core import context as C
from tda.core.db import Db
from tda.core.frame_notes import notes_by_step
from tda.core.model import GALLERY_ONLY_CLASSES
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.ui.panels.add_parts import BUTTON as ADD_PARTS_BUTTON
from tda.ui.panels.add_parts import HINT as ADD_PARTS_HINT
from tda.ui.panels.add_parts import TIP as ADD_PARTS_TIP
from tda.ui.panels.add_parts import AddPartsDialog
from tda.ui.panels.absent_menu import AbsentMenuMixin
from tda.ui.panels.merge_into import MergeIntoMixin
from tda.ui.panels.reclass_menu import ReclassMenuMixin
from tda.ui.panels.relations import RelationsTab
from tda.ui.panels.steptable_models import (
    BLANK,
    INSTANCE_COLUMNS,
    STEP_COLUMNS,
    THUMB_PX,
    THUMB_VIEW,
    Column,
    ComboDelegate,
    DifficultyDelegate,
    InstanceTableModel,
    StepTableModel,
)
from tda.ui.steps_model import (
    ANNOTATOR,
    DEFAULT_PARENT,
    EditError,
    StepTableData,
    captive_signature,
    context_signature,
    forget_deleted,
    stored_signature,
    table_signature,
    thumb_path,
)
from tda.ui.steps_values import DISCRIMINATORS

__all__ = [
    "BLANK",
    "INSTANCE_COLUMNS",
    "STEP_COLUMNS",
    "THUMB_PX",
    "THUMB_VIEW",
    "Column",
    "ComboDelegate",
    "DifficultyDelegate",
    "InstanceTableModel",
    "RelationsTab",
    "StepTableModel",
    "StepTablePanel",
    "thumb_path",
]

#: How many targets a compound row is split into unless the annotator says else.
DEFAULT_SPLIT = 2

#: What an "Open questions" row stores so a double-click can jump to an edge.
EDGE_ROLE = Qt.UserRole + 2

#: Deleting a 框外零件 with confirmed rows (U18 review, M2): the question ...
CONTEXT_DELETE_ASK = (
    "删除框外零件 {key}，连同它在所有视角的形状（Steps 里没有撤销）。\n"
    "它在 {n} 个已确认帧里有确认过的行（{where}）：这些帧会重新核对，各开一个冲突、"
    "回到需复核，要到 Review 里逐个「采用新的」。\n删除吗？")
#: ... and what the status line says afterwards.
CONTEXT_DELETED = "已删除框外零件 {key}（连同它的形状）{recheck} / Deleted {key} with its shapes."
CONTEXT_RECHECK = "；{n} 个已确认帧（{where}）会重新核对并开冲突，到 Review 里逐个「采用新的」"


def _frames_text(frames: dict) -> str:
    """``scan 第 10–12 帧、oak1 第 3 帧`` for the question and the status line."""
    parts = []
    for view, steps in sorted(frames.items()):
        steps = sorted(int(s) for s in steps)
        runs: list[list[int]] = []
        for step in steps:
            if runs and step == runs[-1][-1] + 1:
                runs[-1].append(step)
            else:
                runs.append([step])
        text = "、".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}–{r[-1]}" for r in runs)
        parts.append(f"{view} 第 {text} 帧")
    return "、".join(parts)


def _recheck_note(frames: dict) -> str:
    """The status line's tail about the confirmed frames a delete re-checks; ``""`` for none."""
    n = sum(len(s) for s in frames.values())
    return CONTEXT_RECHECK.format(n=n, where=_frames_text(frames)) if n else ""


def forget_context(signature: Optional[dict], key: str) -> dict:
    """A 框外零件 signature after a write-through delete of ``key``."""
    return {k: v for k, v in (signature or {}).items() if k != key}


def _edge_to_change(deadlock) -> object:
    """The edge in a deadlock a human can actually do something about.

    The manual one, if there is one: a rule edge is derived and can only be
    rejected, so the way out of a loop almost always runs through the edge
    somebody wrote. Falls back to the first edge of the loop.
    """
    for edge in deadlock.edges:
        if edge.source == "manual":
            return edge
    return deadlock.edges[0] if deadlock.edges else None


def _default_cache_dir() -> str:
    """``cache_dir`` from ``configs/paths.yaml``, falling back to the checkout."""
    from tda import pipeline as P

    try:
        return str(P.require(P.load_paths(P.DEFAULT_PATHS_PATH), "cache_dir"))
    except Exception:  # noqa: BLE001 - no config: stay inside the repository
        return str(Path(__file__).resolve().parents[3] / "cache")


class StepTablePanel(MergeIntoMixin, ReclassMenuMixin, AbsentMenuMixin, QWidget):
    """Stage S1: review one desktop's step table and instance table."""

    #: Emitted with the desktop id after ``Apply`` wrote the session back.
    sigSaved = Signal(int)
    #: Emitted after a staged part is added or unstaged (task U5b): that edit
    #: waits for ``Apply`` and is no ``dataChanged`` of either model, so the
    #: window's unsaved-edit gate listens here.
    sigEdited = Signal()
    #: Emitted by :meth:`revert`: nothing is staged any more, and the window's
    #: unsaved-edit flag has to hear it.
    sigReverted = Signal()
    #: 「框外零件…」 in the 「＋ 添加零件」 dialog (task U18): the window goes to
    #: Annotate and makes one there, where it is drawn at once.
    sigContextObject = Signal()
    #: A context object was deleted, written through (task U18): the window's
    #: compiled frames may show it still.
    sigContextChanged = Signal()

    def __init__(
        self,
        db: Db,
        desktop: int,
        taxonomy: Taxonomy | None = None,
        cache_dir: str | Path | None = None,
        parent: QWidget | None = None,
        annotator: str = ANNOTATOR,
        view_of: Optional[Callable[[], str]] = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.desktop = desktop
        #: The view the annotator has open: the add-parts dialog says how many
        #: of its confirmed frames the new parts will demote (task U7).
        self.view_of = view_of
        self.tax = taxonomy or load_taxonomy()
        self.annotator = str(annotator or ANNOTATOR)
        # No literal path here: the cache directory is configuration, and a
        # default baked into the panel would write to this machine's real cache
        # from any caller (or test) that forgot to pass one.
        self.cache_dir = Path(cache_dir if cache_dir else _default_cache_dir())
        self.data = StepTableData.load(db, desktop, self.tax, self.annotator)
        self._remember()
        #: The class the "＋ 添加零件" dialog opened on last time.
        self.last_extra_class: Optional[str] = None
        self.add_parts_dialog: Optional[AddPartsDialog] = None

        self.steps_model = StepTableModel(self.data, self.cache_dir, self)
        self.instances_model = InstanceTableModel(self.data, self)
        self.steps_view = self._table(self.steps_model, STEP_COLUMNS, THUMB_PX + 8)
        self.instances_view = self._table(self.instances_model, INSTANCE_COLUMNS)
        self.relations_tab = RelationsTab(self.data, self)
        self.issues = QListWidget(self)
        self.tabs = QTabWidget(self)
        self._build()
        self._connect()
        self._refresh_issues()

    # -- construction ------------------------------------------------------ #
    def _build(self) -> None:
        self.tabs.addTab(self.steps_view, "Steps")
        self.add_parts_button = QPushButton(ADD_PARTS_BUTTON, self)
        self.add_parts_button.setToolTip(ADD_PARTS_TIP)
        self.add_parts_button.clicked.connect(self.open_add_parts)
        self.instances_page = QWidget(self)
        page = QVBoxLayout(self.instances_page)
        page.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(self.add_parts_button)
        hint = QLabel(ADD_PARTS_HINT, self.instances_page)
        hint.setWordWrap(True)
        top.addWidget(hint, 1)
        page.addLayout(top)
        page.addWidget(self.instances_view)
        self.tabs.addTab(self.instances_page, "Instances")
        self.tabs.addTab(self.relations_tab, "Relations")
        self.apply_button = QPushButton("Apply", self)
        self.revert_button = QPushButton("Revert", self)
        self.status = QLabel(BLANK, self)
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.revert_button)
        buttons.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 3)
        layout.addWidget(QLabel("Open questions", self))
        layout.addWidget(self.issues, 1)
        layout.addLayout(buttons)

    def _connect(self) -> None:
        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        for model in (self.steps_model, self.instances_model):
            model.sigError.connect(self._show_error)
            # Any accepted edit can create or settle an open question.
            model.dataChanged.connect(self._on_data_changed)
            model.modelReset.connect(self._refresh_issues)
        for view, handler in (
            (self.steps_view, self._steps_context_menu),
            (self.instances_view, self._instances_context_menu),
        ):
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(handler)
        self.relations_tab.sigError.connect(self._show_error)
        self.relations_tab.sigChanged.connect(self._on_relations_changed)
        # The jump stays inside the panel: the Steps *table* moves, the session
        # does not, so the window's uncommitted-edit gate is not involved and
        # the Relations tab adds no second path around it.
        self.relations_tab.sigGoToStep.connect(self.show_step)
        self.issues.itemDoubleClicked.connect(self._on_issue_activated)

    def _table(
        self, model, columns: Sequence[Column], row_height: int = 0
    ) -> QTableView:
        view = QTableView(self)
        view.setModel(model)
        view.setSelectionBehavior(QAbstractItemView.SelectItems)
        view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        view.verticalHeader().setVisible(False)
        if row_height:
            view.verticalHeader().setDefaultSectionSize(row_height)
        for index, column in enumerate(columns):
            delegate = self._delegate(column, view)
            if delegate is not None:
                view.setItemDelegateForColumn(index, delegate)
        return view

    def _delegate(self, column: Column, parent: QWidget) -> Optional[QStyledItemDelegate]:
        if column.kind == "combo" and column.choices is not None:
            return ComboDelegate(column.choices(self.tax), parent)
        if column.kind == "spin":
            return DifficultyDelegate(parent)
        return None

    # -- loading ----------------------------------------------------------- #
    def set_desktop(self, desktop: int) -> None:
        """Load another desktop (or reload this one), dropping unsaved edits."""
        self.desktop = desktop
        self.data = StepTableData.load(self.db, desktop, self.tax, self.annotator)
        self._remember()
        self.steps_model.set_data(self.data)
        self.instances_model.set_data(self.data)
        self.relations_tab.set_data(self.data)
        self._refresh_issues()

    def refresh_notes(self) -> None:
        """Re-read 「这一帧的说明」 (task U35): the ✎ next to a step's thumbnail.

        Notes are written in Review, never here, so this is not an edit: no
        ``dataChanged`` (which would mark the table unsaved) -- the view just
        repaints from the data it asks for.
        """
        self.data.notes = notes_by_step(self.db, self.desktop)
        self.steps_view.viewport().update()

    # -- what moved since the last load / Apply (task U9) ------------------ #
    def _remember(self) -> None:
        """Snapshot the database and this panel's copy, which agree right now.

        Taken at every load and after every successful Apply. A correction
        made in Annotate mode (``Shift+T``) moves the database without the
        panel; the two snapshots tell that apart from the panel's own edits.
        """
        self.loaded_db = stored_signature(self.db, self.desktop)
        self.loaded_data = table_signature(self.data.steps, self.data.actions)
        # the screws' captive fields too (task U14): Shift+T on a screw writes
        # them from Annotate mode
        stored = self.db.instances(self.desktop)
        self.loaded_db_captive = captive_signature(stored)
        self.loaded_captive = captive_signature(self.data.instances)
        # the 框外零件 too (U18 review, B2): Annotate mode makes, undoes and
        # draws them without this panel
        self.loaded_db_context = context_signature(stored)
        self.loaded_context = context_signature(self.data.context)

    def context_moved(self) -> bool:
        """Have the stored 框外零件 changed since the last load / Apply / refresh?"""
        return (context_signature(self.db.instances(self.desktop))
                != getattr(self, "loaded_db_context", None))

    def database_moved(self) -> bool:
        """Has the stored step table (a screw's captive, a 框外零件) changed since the last load?"""
        stored = self.db.instances(self.desktop)
        return (stored_signature(self.db, self.desktop) != self.loaded_db
                or captive_signature(stored) != getattr(self, "loaded_db_captive", None)
                or context_signature(stored) != getattr(self, "loaded_db_context", None))

    def has_unapplied_edits(self) -> bool:
        """Does the panel hold steps, actions, captive flags, parts or edges not written yet?"""
        return (table_signature(self.data.steps, self.data.actions) != self.loaded_data
                or captive_signature(self.data.instances)
                != getattr(self, "loaded_captive", None)
                or bool(self.data.extras) or bool(self.data.relations.dirty)
                # a 框外零件 renamed and not applied (U18)
                or context_signature(self.data.context)
                != getattr(self, "loaded_context", None)
                or bool(self.data.relabelled))

    def refresh_context(self) -> None:
        """Re-read the 「框外零件」 block alone (U18 review, B2).

        What Annotate mode made, undid or renamed since the last load; the
        staged steps, actions and parts are kept, and so is a rename staged
        here for an object still there. The signatures move with it, so
        :meth:`database_moved` says nothing about it any more.
        """
        self.data.reload_context(self.db)
        stored = {key: rec for key, rec in self.db.instances(self.desktop).items()
                  if key in self.data.context}
        self.loaded_db_context = context_signature(stored)
        self.loaded_context = context_signature(stored)
        self.instances_model.refresh_structure()

    def show_step(self, step: int) -> None:
        """Select one logical step in the Steps table and bring it into view."""
        row = self.steps_model.first_row_of(step)
        if row < 0:
            return
        self.tabs.setCurrentWidget(self.steps_view)
        self.steps_view.selectRow(row)
        self.steps_view.scrollTo(self.steps_model.index(row, 0))

    def revert(self) -> None:
        """Throw away every unsaved edit and reload from the database."""
        self.set_desktop(self.desktop)
        self.status.setText("Reverted to the stored step table.")
        self.sigReverted.emit()

    def apply(self) -> None:
        """Write the session back, recompile the auto events and report.

        The save is one transaction, so a failure leaves the database as it
        was; the reason lands in the status line instead of escaping into Qt.
        """
        try:
            messages = self.data.save(self.db)
        except Exception as error:  # a failed save must not take the panel down
            self._show_error(f"could not save D{self.desktop:02d}: {error}")
            self.relations_tab.refresh()
            self._refresh_issues()
            # a refused Apply names an edge; put the annotator in front of it
            self.show_deadlock_edge()
            return
        self._remember()
        self.relations_tab.refresh()
        self._refresh_issues()
        # A step that became (or stopped being) a `reorient` moved a pose
        # boundary in every view, and every shape of those views is anchored to
        # one. The annotator has to be told on the Apply, not by a renumbered
        # segment turning up under them later.
        recut = ""
        if self.data.recut:
            cut = ", ".join(f"{view} {n}" for view, n in sorted(self.data.recut.items()))
            recut = f" Pose segments re-cut: {cut}."
        self.status.setText(
            f"Saved D{self.desktop:02d}: {len(self.data.issues)} open question(s), "
            f"{len(messages)} state warning(s).{recut}"
        )
        self.sigSaved.emit(self.desktop)

    # -- commands ---------------------------------------------------------- #
    # The four structural commands below wait for ``Apply`` like a cell edit,
    # but rebuild the models (no ``dataChanged``): they are ``staged`` so that
    # :attr:`sigEdited` tells the window, and leaving Steps mode asks first
    # (task U9 review -- an added or removed action used to vanish silently).
    def split_step(self, step: int, n: int) -> None:
        """Split a compound row into ``n`` actions/instances (spec 2.3)."""
        self._command(lambda: self.data.split_compound(step, n),
                      f"Split step {step} into {n} actions.", staged=True)

    def add_action(self, step: int) -> None:
        """Append one more action to a step."""
        self._command(lambda: self.data.add_action(step), f"Added an action to step {step}.",
                      staged=True)

    def remove_action(self, step: int, action_idx: int) -> None:
        """Drop one action of a step."""
        self._command(lambda: self.data.remove_action(step, action_idx),
                      f"Removed action {action_idx + 1} of step {step}.", staged=True)

    def retarget_new(self, step: int, cls: str, disc: str = BLANK, action_idx: int = 0) -> None:
        """Point a step at a brand-new instance of ``cls``.

        ``disc`` is the class's discriminator -- ``role`` for a screw, ``kind``
        for everything else that has one -- and decides which ordinal run the
        new key continues. The view-model checks it against the class's
        vocabulary, so a value from anywhere but the dialog is checked too.
        """
        attrs = {}
        if disc:
            name, _vocabulary = self.data.discriminator_of(cls)
            attrs[name or DISCRIMINATORS[0]] = disc
        self._command(
            lambda: self.data.change_target(step, action_idx=action_idx, cls=cls, attrs=attrs),
            f"Retargeted step {step}.", staged=True,
        )

    def delete_instance(self, key: str) -> None:
        """Delete an instance nothing references any more.

        A part added in S1 goes the same way -- only while it has no shape --
        and a staged one that was never applied is simply unstaged.
        """
        staged = self.data.staged_extra(key)
        outside = key in self.data.context        # a 框外零件 (task U18)
        confirmed = C.confirmed_frames(self.db, self.desktop, key) if outside else {}
        self._command(lambda: self.data.delete_instance(self.db, key),
                      f"Removed the staged {key}." if staged
                      else (CONTEXT_DELETED.format(key=key, recheck=_recheck_note(confirmed))
                            if outside else f"Deleted {key}."),
                      staged=staged)
        if outside and key not in self.data.context:
            # written through on both sides alike (as U14's captive snapshot)
            self.loaded_db_context = forget_context(self.loaded_db_context, key)
            self.loaded_context = forget_context(self.loaded_context, key)
            self.sigContextChanged.emit()
        if not staged and key not in self.data.instances:
            # written through: both sides lost it alike (U14's snapshot of the
            # screws' captive fields must not read that as a move or an edit)
            self.loaded_db_captive = forget_deleted(self.loaded_db_captive, key)
            self.loaded_captive = forget_deleted(self.loaded_captive, key)

    def prompt_delete_context(self, key: str) -> None:
        """Delete a 框外零件, asking first when confirmed rows of it exist (U18 review, M2).

        Its shapes go in every view and there is no undo in Steps mode; a
        frame somebody confirmed with it re-checks, opens a conflict about the
        row it no longer has and goes back to 需复核 -- the question says how
        many, in which views, and that Review takes each one 「采用新的」.
        """
        confirmed = C.confirmed_frames(self.db, self.desktop, key)
        if confirmed and not self.ask_delete_context(key, confirmed):
            self.status.setText(f"没有删除 / {key} was not deleted.")
            return
        self.delete_instance(key)

    def ask_delete_context(self, key: str, confirmed: dict) -> bool:
        """The question of :meth:`prompt_delete_context` (a method, so a test answers it)."""
        answer = QMessageBox.question(
            self, "删除框外零件 / Delete", CONTEXT_DELETE_ASK.format(
                key=key, where=_frames_text(confirmed),
                n=sum(len(s) for s in confirmed.values())),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def add_parts(self, cls: str, count: int = 1, parent=DEFAULT_PARENT,
                  state: Optional[str] = None, note: str = "",
                  attrs: Optional[dict] = None,
                  removed_at: Optional[int] = None) -> list[str]:
        """Stage ``count`` parts of ``cls`` the log never names (task U5b).

        Written by ``Apply`` with one ``op_log`` row, dropped by ``Revert``.
        ``removed_at`` is 「拆下于第 k 步」 (task U10). Returns the new keys;
        empty when the addition was refused (the reason is in the status line).
        """
        made: list[str] = []

        def run() -> None:
            made.extend(self.data.add_extras(cls, count, parent, state, note, attrs,
                                             removed_at=removed_at))

        self._command(run, "", staged=True)
        if made:
            self.last_extra_class = cls
            host = self.data.instances[made[0]].parent
            where = f" on {host}" if host else ""
            self.status.setText(
                f"已添加 {len(made)} 个（未写入，按 Apply 保存）/ staged {len(made)} "
                f"added part(s){where}: {', '.join(made)} - press Apply to write them.")
        return made

    def open_add_parts(self) -> AddPartsDialog:
        """Open the "＋ 添加零件" dialog; accepting it stages the parts.

        Opened with ``open()`` rather than ``exec()``, so it is window-modal
        without blocking -- and returned, so a caller (or a test) can fill it in.
        """
        dialog = AddPartsDialog(self.data, self, cls=self.last_extra_class,
                                demotion=self._demotion(), elsewhere=self._elsewhere(),
                                context=self.sigContextObject.emit)
        dialog.accepted.connect(lambda: self.add_parts(**dialog.values()))
        dialog.finished.connect(dialog.deleteLater)
        self.add_parts_dialog = dialog
        dialog.open()
        return dialog

    def _demotion(self) -> Optional[Callable[[dict], list[int]]]:
        """"Which confirmed frames of the open view would these parts demote?" (U7)."""
        view = str(self.view_of() or "") if self.view_of is not None else ""
        if not view:
            return None

        def frames(values: dict) -> list[int]:
            return self.data.frames_demoted_by_extras(
                self.db, view, values["cls"], values["count"], values.get("parent"),
                values.get("state"), values.get("attrs"), values.get("removed_at"))

        return frames

    def _elsewhere(self) -> Optional[Callable[[dict], dict]]:
        """The confirmed frames of every *other* view the parts would demote (U10).

        Both families: the teardown's views and the components gallery's (a
        part taken out at 「拆下于第 k 步」 joins every confirmed shot). Only
        views that hold a confirmed frame are asked; with no open view the
        dialog says nothing at all, as :meth:`_demotion` does.
        """
        from tda.core.model import ALL_VIEWS

        here = str(self.view_of() or "") if self.view_of is not None else ""
        if not here:
            return None

        def frames(values: dict) -> dict:
            confirmed = {v for v, _s in self.db.verified_frames(int(self.data.desktop))}
            return {view: self.data.frames_demoted_by_extras(
                        self.db, view, values["cls"], values["count"], values.get("parent"),
                        values.get("state"), values.get("attrs"), values.get("removed_at"))
                    for view in ALL_VIEWS if view != here and view in confirmed}

        return frames

    def _command(self, run, done: str, staged: bool = False) -> None:
        """Run one structural command, then rebuild both tables and the issues.

        A refused command shows its reason; anything unexpected shows its type
        as well, and neither takes the panel down. ``staged`` says the command
        waits for ``Apply``, which is what :attr:`sigEdited` tells the window.
        """
        try:
            run()
        except EditError as error:
            self._show_error(str(error))
            return
        except Exception as error:
            self._show_error(f"{type(error).__name__}: {error}")
            return
        self.steps_model.refresh_structure()
        self.instances_model.refresh_structure()
        self._refresh_issues()
        self.status.setText(done)
        if staged:
            self.sigEdited.emit()

    # -- context menus ----------------------------------------------------- #
    def steps_menu(self, view_row: int) -> QMenu:
        """The steps-table context menu for one view row (not shown yet)."""
        menu = QMenu(self.steps_view)
        step = self.steps_model.step_at(view_row)
        if step is None:
            return menu
        action_idx = self.steps_model.action_at(view_row)
        row = self.data.row(step)

        menu.addAction("Add action", lambda: self.add_action(step))
        remove = menu.addAction(
            "Remove action", lambda: self.remove_action(step, action_idx)
        )
        remove.setEnabled(bool(row.actions))
        split = menu.addAction("Split compound into N...", lambda: self._prompt_split(step))
        split.setEnabled(len(row.actions) == 1)
        menu.addAction(
            "Retarget to new instance...", lambda: self._prompt_retarget(step, action_idx)
        )
        return menu

    def instances_menu(self, view_row: int) -> QMenu:
        """The instances-table context menu for one view row (not shown yet)."""
        menu = QMenu(self.instances_view)
        key = self.instances_model.key_at(view_row)
        if key is not None and key in self.data.context:
            # a 框外零件 (task U18): deleted with its shapes; never merged
            menu.addAction(f"删除框外零件 {key}（连同形状）/ Delete {key}",
                           lambda: self.prompt_delete_context(key))
        elif key is not None:
            menu.addAction(f"Delete {key}", lambda: self.delete_instance(key))
            self.add_absent_action(menu, key)       # 「这台没有这个零件…」 (task U34)
            self.add_merge_action(menu, key)        # 「合并到…」 (task U12)
            self.add_reclass_action(menu, key)      # 「其实是硬盘架…」 (task U21)
        return menu

    def _steps_context_menu(self, pos) -> None:
        menu = self.steps_menu(self.steps_view.indexAt(pos).row())
        if not menu.isEmpty():
            menu.exec(self.steps_view.viewport().mapToGlobal(pos))

    def _instances_context_menu(self, pos) -> None:
        menu = self.instances_menu(self.instances_view.indexAt(pos).row())
        if not menu.isEmpty():
            menu.exec(self.instances_view.viewport().mapToGlobal(pos))

    def _prompt_split(self, step: int) -> None:
        count, ok = QInputDialog.getInt(
            self, "Split compound row", f"How many targets does step {step} name?",
            DEFAULT_SPLIT, 2, 99,
        )
        if ok:
            self.split_step(step, count)

    def _prompt_retarget(self, step: int, action_idx: int) -> None:
        """Ask for the class, then for its discriminator from a closed list.

        A class that carries no ``role``/``kind`` -- a motherboard, a CPU --
        skips the second question instead of inviting free text that would
        silently start a parallel ordinal run.
        """
        cls, ok = QInputDialog.getItem(
            # not a gallery-only class (cable, task U45): no step names one
            self, "New instance", "Taxonomy class:",
            sorted(c for c in self.tax.classes if c not in GALLERY_ONLY_CLASSES), 0, False
        )
        if not ok or not cls:
            return
        name, vocabulary = self.data.discriminator_of(cls)
        disc = BLANK
        if vocabulary:
            disc, ok = QInputDialog.getItem(
                self, "New instance", f"{name.capitalize()}:", vocabulary, 0, False
            )
            if not ok:
                return
        self.retarget_new(step, cls, disc, action_idx)

    # -- feedback ---------------------------------------------------------- #
    def _on_data_changed(self, *_args) -> None:
        self._refresh_issues()
        # A step edit can create or settle a spec 7.4 violation (a result that
        # becomes `failed` is the whole point of the Relations tab), so the
        # replay is re-run with the issues.
        self.relations_tab.refresh()

    def _on_relations_changed(self) -> None:
        staged = self.data.relations.dirty
        self.status.setText("Constraint edit staged; press Apply to write it."
                            if staged else "No constraint edit is staged.")

    def _refresh_issues(self) -> None:
        """Every open question, deadlocks first.

        A deadlock is the one thing in the Relations tab that stops ``Apply``
        from writing anything at all, so it is also said here, where the
        annotator sees it from the Steps and Instances tabs too.
        """
        self.issues.clear()
        for deadlock in self.data.relations.cycles():
            item = QListWidgetItem(
                f"约束死锁 / deadlock (Apply refuses): {deadlock.label()}")
            edge = _edge_to_change(deadlock)
            if edge is not None:
                # double-clicking jumps to the edge a human can actually change
                item.setData(EDGE_ROLE, [edge.target, edge.type, edge.blocker])
            self.issues.addItem(item)
        for end in self.data.relations.dead_ends():
            # not a refusal: an edge nothing can clear may want a missing part
            # created rather than the edge changed -- but it is never silent
            item = QListWidgetItem(end.label())
            item.setData(EDGE_ROLE, [end.edge.target, end.edge.type, end.edge.blocker])
            self.issues.addItem(item)
        self.issues.addItems(self.data.issues)
        self.issues.addItems(f"state event: {m}" for m in self.data.messages)

    def _on_issue_activated(self, item: QListWidgetItem) -> None:
        """A deadlock line points at an edge: show it in the Relations tab."""
        edge = item.data(EDGE_ROLE)
        if not edge:
            return
        self.tabs.setCurrentWidget(self.relations_tab)
        self.relations_tab.show_edge(*edge)

    def show_deadlock_edge(self) -> None:
        """Select the edge of the first deadlock, the one a human can change."""
        for deadlock in self.data.relations.cycles():
            edge = _edge_to_change(deadlock)
            if edge is not None:
                self.tabs.setCurrentWidget(self.relations_tab)
                self.relations_tab.show_edge(edge.target, edge.type, edge.blocker)
                return

    def _show_error(self, message: str) -> None:
        self.status.setText(f"Rejected: {message}")
