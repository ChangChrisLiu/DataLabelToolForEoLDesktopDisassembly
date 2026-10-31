"""The "＋ 添加零件 / Add parts" dialog of stage S1 (task U5b).

A part the picture shows but the log never names -- the clips of D13's two
empty RAM slots -- has no instance, so the detector learns it as background and
no state question can be asked about it. This dialog stages such parts on the
:class:`~tda.ui.steps_model.StepTableData` the panel edits; ``Apply`` writes
them and ``Revert`` drops them, like every other S1 edit.

It asks for five things, each pre-filled from the taxonomy so the usual answer
is one click:

* the class, by its Chinese name and key (:mod:`tda.ui.class_names`), plus the
  class's ``role`` / ``kind`` when it has one -- that is part of the key;
* how many (1-16);
* the part it rides on: the unique instance of the class's ``host_class`` when
  there is exactly one (a ``ram_latch`` -> ``motherboard.01``), else a pick;
* the state the picture shows it in (default: the class default);
* an optional note.

Below them, the keys it will create -- ``ram_latch.05 … ram_latch.08`` -- so
nothing about the numbering is a surprise, and how many **confirmed frames**
of the open view the new parts will appear in (task U7): each of those frames
gains an instance and goes back to 需复核.  On D13/scan four clips demoted the
25 frames 17-41 and nothing said so beforehand.  Not a confirmation -- adding
them is right, the frames just need a Space each once the clips are drawn --
only the sentence.  The dialog only *reads* the view model; :meth:`values` is
what the panel hands to :meth:`~tda.ui.steps_model.StepTableData.add_extras`.

Something in the picture that is **not part of the machine** at all is no
extra: the button 「框外零件…」 (task U18, shown when the panel wires it) closes
the dialog and hands over to the window, which makes a context object in
Annotate mode (:mod:`tda.ui.panels.context_object`). Nothing about adding parts
changes.

A part that lies on the table in the **components gallery** and that no step
names at all -- a loose cable, a screw the sheet never counted -- is added from
the shot itself (task U45): :class:`GalleryPartsDialog`, the task card's
「＋ 这张里还有…」, asks only for the class (with its ``role`` / ``kind``) and
how many; the parts are ``gallery_only`` and the teardown never asks for them.
The class ``loose_cable`` is offered there only.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tda.core import extra as X
from tda.core.logs import CHASSIS_KEY
from tda.core.model import GALLERY_ONLY_CLASSES
from tda.ui.class_names import class_zh, state_zh
from tda.ui.gallery import ADD_PARTS_BUTTON, ADD_PARTS_INTRO, ADD_PARTS_TIP
from tda.ui.guide import steps_text
from tda.ui.steps_model import StepTableData

__all__ = ["AddPartsDialog", "BUTTON", "CONTEXT_BUTTON", "DEMOTION", "GALLERY_BUTTON",
           "GALLERY_INTRO", "GALLERY_TIP", "GALLERY_TITLE", "GalleryPartsDialog", "HINT",
           "NO_PARENT", "TIP", "TITLE", "class_choices", "demotion_text",
           "gallery_class_choices", "gallery_preview_text", "preview_text"]

TITLE = "＋ 添加零件 / Add parts"
#: The button above the Instances table that opens the dialog.
BUTTON = TITLE
#: The line next to it.
HINT = "画面里有、日志里没有的零件（例如空内存槽的卡扣）在这里加 / parts the log never names"
#: The button's tooltip.
TIP = ("新建日志里没有的实例：选类别、数量、装在哪个零件上、画面里的状态。"
       "按 Apply 写入，Revert 丢弃；没画过形状的可以在表里右键删除。")
#: The parent combo's first entry: the part rides on nothing.
NO_PARENT = "（无 / none）"
#: What the dialog is for, in the words of the annotator's question.
INTRO = ("画面里有、日志里没有的零件（例如空内存槽上的卡扣）：每一个都要标，"
         "不然检测器会把它当背景。它没有动作，一直是下面选的状态，"
         "直到它所在的零件被拆走。")
#: How many keys the preview spells out before it abbreviates.
PREVIEW_SHOWN = 6
#: What adding them does to the confirmed frames of the open view (task U7).
DEMOTION = ("{who}出现在 {n} 个已确认帧（{steps}）里；添加后这些帧会回到“需复核”，"
            "补画后逐帧按 Space 即可。")
#: The same for the other views (task U10: the components gallery is a family
#: of its own, and a part taken out joins every confirmed shot of it).
DEMOTION_ELSEWHERE = "另外：{views} 的已确认帧也会回到“需复核”。"
#: 「拆下于第 k 步」's blank value (task U10).
NEVER_REMOVED = "（不拆 / 跟着装它的零件走）"
#: The removal field's tooltip.
REMOVED_TIP = ("这个零件在第几步被拆下来（之后在零件合照里列出、序列里不再要它）；"
               "装在别的零件上、跟着它一起离开的，留空 / the step it was taken out at")
#: 「框外零件…」 (task U18): the thing is not in the machine at all.
CONTEXT_BUTTON = "框外零件…"
CONTEXT_TIP = ("不在机器里的东西（旁边放着的另一个零件、无关的物件）：到标注模式起个名字单独画，"
               "不进步骤表 / something outside the machine")


def elsewhere_text(others: dict) -> str:
    """The sentence for the confirmed frames of the *other* views; ``""`` for none.

    ``others`` is ``{view: [steps]}``; a gallery view is named as the window
    names it (零件合照·OAK1) and counted in shots.
    """
    from tda.core.model import ALL_VIEWS
    from tda.ui.gallery import is_component_view, view_label

    order = {view: i for i, view in enumerate(ALL_VIEWS)}
    parts = []
    for view, steps in sorted(others.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])):
        if not steps:
            continue
        unit = "张" if is_component_view(view) else "帧"
        parts.append(f"{view_label(view)} {len(steps)} {unit}")
    return DEMOTION_ELSEWHERE.format(views="、".join(parts)) if parts else ""


def class_choices(data: StepTableData) -> list[tuple[str, str]]:
    """``(label, class)`` for every class an extra may have, taxonomy order.

    The chassis is unique per desktop and is left out, and so is a class only
    a gallery-only part can have (``loose_cable``, task U45): it is added from the
    shot, with 「＋ 这张里还有…」 (:class:`GalleryPartsDialog`).
    """
    return [(f"{class_zh(cls)}  {cls}", cls) for cls in data.tax.classes
            if cls != CHASSIS_KEY and cls not in GALLERY_ONLY_CLASSES]


def preview_text(keys: list[str], parent: Optional[str]) -> str:
    """The line under the form: the keys the batch will create, and its host."""
    if not keys:
        return "（这个类别不能添加 / this class cannot be added）"
    shown = keys if len(keys) <= PREVIEW_SHOWN else [*keys[:PREVIEW_SHOWN - 1], "…", keys[-1]]
    text = f"将新建 / will create: {', '.join(shown)}"
    if parent:
        text += f"\n跟 {parent} 一起离开机箱 / leaves the chassis with {parent}"
    else:
        text += "\n不挂在任何零件上：一直留在机箱里 / rides on nothing: stays in the chassis"
    return text


def demotion_text(steps: list[int], count: int) -> str:
    """The sentence for the confirmed frames a batch demotes; ``""`` for none."""
    if not steps:
        return ""
    who = "这个零件" if int(count) <= 1 else f"这 {int(count)} 个零件"
    return DEMOTION.format(who=who, n=len(steps), steps=steps_text(steps))


class AddPartsDialog(QDialog):
    """Pick a class, a count, a parent, a starting state and a note.

    ``demotion`` answers "which confirmed frames would these parts demote?"
    for the form's :meth:`values` (the panel's
    :meth:`~tda.ui.steps_model.StepTableData.frames_demoted_by_extras` on the
    open view); without it the dialog says nothing about them.
    """

    def __init__(self, data: StepTableData, parent: Optional[QWidget] = None,
                 cls: Optional[str] = None,
                 demotion: Optional[Callable[[dict], list[int]]] = None,
                 elsewhere: Optional[Callable[[dict], dict]] = None,
                 context: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self.data = data
        self._demotion = demotion
        #: ``{view: [steps]}`` the parts would demote in the other views (U10).
        self._elsewhere = elsewhere
        #: 「框外零件…」 (task U18): not a part of the machine at all -- the
        #: dialog closes and this is called; ``None`` shows no button.
        self._context = context
        self.setWindowTitle(TITLE)
        self.class_combo = QComboBox(self)
        for label, name in class_choices(data):
            self.class_combo.addItem(label, name)
        self.disc_label = QLabel("", self)
        self.disc_combo = QComboBox(self)
        self.count_spin = QSpinBox(self)
        self.count_spin.setRange(X.MIN_COUNT, X.MAX_COUNT)
        self.count_spin.setValue(1)
        self.parent_combo = QComboBox(self)
        self.state_combo = QComboBox(self)
        #: 「拆下于第 k 步」 (task U10); 0 is "never" (its special text).
        self.removed_spin = QSpinBox(self)
        self.removed_spin.setRange(0, max(int(data.last_step()), 0))
        self.removed_spin.setSpecialValueText(NEVER_REMOVED)
        self.removed_spin.setToolTip(REMOVED_TIP)
        self.note_edit = QLineEdit(self)
        self.note_edit.setPlaceholderText("可选 / optional，例如：空槽的卡扣")
        self.preview = QLabel("", self)
        self.preview.setWordWrap(True)
        #: The confirmed frames the parts will demote (U7); hidden when none.
        self.warning = QLabel("", self)
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #b35c00;")
        self.warning.setVisible(False)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("添加 / Add")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.context_button = None
        if context is not None:
            self.context_button = self.buttons.addButton(
                CONTEXT_BUTTON, QDialogButtonBox.ButtonRole.ActionRole)
            self.context_button.setToolTip(CONTEXT_TIP)
            self.context_button.clicked.connect(self._on_context)
        self._build()
        if cls:
            index = self.class_combo.findData(cls)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        self.class_combo.currentIndexChanged.connect(self._on_class)
        self.disc_combo.currentIndexChanged.connect(self._refresh_preview)
        self.count_spin.valueChanged.connect(self._refresh_preview)
        self.parent_combo.currentIndexChanged.connect(self._on_parent)
        self.state_combo.currentIndexChanged.connect(self._refresh_preview)
        self.removed_spin.valueChanged.connect(self._refresh_preview)
        self._on_class()

    def _build(self) -> None:
        intro = QLabel(INTRO, self)
        intro.setWordWrap(True)
        form = QFormLayout()
        form.addRow("类别 / class", self.class_combo)
        form.addRow(self.disc_label, self.disc_combo)
        form.addRow("数量 / count", self.count_spin)
        form.addRow("装在 / rides on", self.parent_combo)
        form.addRow("画面里的状态 / state", self.state_combo)
        form.addRow("拆下于第 __ 步 / taken out at", self.removed_spin)
        form.addRow("备注 / note", self.note_edit)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.preview)
        layout.addWidget(self.warning)
        layout.addWidget(self.buttons)

    def _on_context(self) -> None:
        """Not a part of the machine: close, and make a 框外零件 instead (U18)."""
        self.reject()
        if self._context is not None:
            self._context()

    # -- reading ------------------------------------------------------------ #
    @property
    def cls(self) -> str:
        return str(self.class_combo.currentData() or "")

    def _attrs(self) -> dict:
        name, vocabulary = self.data.discriminator_of(self.cls)
        value = self.disc_combo.currentText() if vocabulary else ""
        return {name: value} if name and value else {}

    def _parent(self) -> Optional[str]:
        value = self.parent_combo.currentData()
        return str(value) if value else None

    def keys(self) -> list[str]:
        """The keys the dialog would create as it stands."""
        return self.data.preview_extras(self.cls, self.count_spin.value(), self._attrs())

    def values(self) -> dict:
        """The keyword arguments of :meth:`StepTableData.add_extras`."""
        return {
            "cls": self.cls,
            "count": int(self.count_spin.value()),
            "parent": self._parent(),
            "state": str(self.state_combo.currentData() or "") or None,
            "note": self.note_edit.text().strip(),
            "attrs": self._attrs(),
            "removed_at": self._removed_at(),
        }

    def _removed_at(self) -> Optional[int]:
        """「拆下于第 k 步」, or ``None`` for never / with its parent."""
        value = int(self.removed_spin.value())
        return value if value > 0 and self.removed_spin.isEnabled() else None

    def _default_removal(self) -> None:
        """Pre-fill 「拆下于」: the last step for a part riding on nothing, else blank."""
        defaults = (self.data.extra_defaults(self.cls) if self.cls else
                    {"removable": False, "last_step": 0})
        removable = bool(defaults.get("removable"))
        self.removed_spin.blockSignals(True)
        self.removed_spin.setEnabled(removable)
        self.removed_spin.setValue(int(defaults.get("last_step") or 0)
                                   if removable and self._parent() is None else 0)
        self.removed_spin.blockSignals(False)

    def _on_parent(self, *_args) -> None:
        self._default_removal()
        self._refresh_preview()

    # -- keeping the form consistent --------------------------------------- #
    def _on_class(self, *_args) -> None:
        cls = self.cls
        name, vocabulary = self.data.discriminator_of(cls)
        self.disc_combo.blockSignals(True)
        self.disc_combo.clear()
        self.disc_combo.addItems(vocabulary)
        self.disc_combo.blockSignals(False)
        self.disc_label.setText(f"{name} / 细分" if vocabulary else "")
        self.disc_combo.setVisible(bool(vocabulary))
        self.disc_label.setVisible(bool(vocabulary))
        defaults = self.data.extra_defaults(cls) if cls else {
            "parent": None, "parents": [], "state": "", "states": []}
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem(NO_PARENT, "")
        for key in defaults["parents"]:
            rec = self.data.instances.get(key)
            label = f"{class_zh(rec.cls if rec else None, rec.attrs if rec else None, key)}  {key}"
            self.parent_combo.addItem(label, key)
        index = self.parent_combo.findData(defaults["parent"] or "")
        self.parent_combo.setCurrentIndex(max(index, 0))
        self.parent_combo.blockSignals(False)
        self.state_combo.clear()
        for state in defaults["states"]:
            self.state_combo.addItem(f"{state_zh(state)}  {state}", state)
        index = self.state_combo.findData(defaults["state"])
        self.state_combo.setCurrentIndex(max(index, 0))
        self._default_removal()
        self._refresh_preview()

    def demoted_frames(self) -> list[int]:
        """The confirmed frames of the open view the form's parts would demote."""
        if self._demotion is None or not self.keys():
            return []
        try:
            return sorted(int(s) for s in self._demotion(self.values()))
        except Exception:  # noqa: BLE001 - a warning is never worth a failure
            return []

    def demoted_elsewhere(self) -> dict:
        """``{view: [steps]}`` the parts would demote in the other views (U10)."""
        if self._elsewhere is None or not self.keys():
            return {}
        try:
            return {str(v): sorted(int(s) for s in steps)
                    for v, steps in (self._elsewhere(self.values()) or {}).items() if steps}
        except Exception:  # noqa: BLE001 - a warning is never worth a failure
            return {}

    def _refresh_preview(self, *_args) -> None:
        keys = self.keys()
        removed = self._removed_at()
        if keys and removed is not None:
            # taken out on its own: it does not leave with what it rides on
            text = preview_text(keys, None).splitlines()[0]
            text += (f"\n第 {removed} 步拆下：之前的帧要画它，之后列在零件合照里 / "
                     f"taken out at step {removed}")
        else:
            text = preview_text(keys, self._parent())
        self.preview.setText(text)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(keys))
        text = demotion_text(self.demoted_frames(), self.count_spin.value())
        elsewhere = elsewhere_text(self.demoted_elsewhere())
        text = "".join(t for t in (text, elsewhere) if t)
        self.warning.setText(text)
        self.warning.setVisible(bool(text))


# --------------------------------------------------------------------------- #
# 「＋ 这张里还有…」: a gallery-only part, from the shot (task U45)
# --------------------------------------------------------------------------- #
#: The task card's button on a components-gallery shot (:mod:`tda.ui.gallery`
#: says it, for the card and the guide), and the dialog's title.
GALLERY_BUTTON = ADD_PARTS_BUTTON
GALLERY_TITLE = f"{ADD_PARTS_BUTTON} / more parts on this shot"
GALLERY_TIP = ADD_PARTS_TIP
#: The one line the dialog says (the controller's brief, U45 §2.3).
GALLERY_INTRO = ADD_PARTS_INTRO
#: The discriminator combo's first entry: no ``kind`` / ``role`` (review M1).
NO_DISCRIMINATOR = "（不细分）"


def gallery_class_choices(data: StepTableData) -> list[tuple[str, str]]:
    """``(label, class)`` a gallery-only part may have: every class but a few.

    ``loose_cable`` (task U45) among them -- it is the class this dialog exists for;
    the screws and connectors the gallery leaves out of a log's parts are
    offered too, because an annotator who adds one is saying it can be told
    apart on the table. Left out: the chassis, and the classes moulded into
    their host (``host_class``: a RAM clip, the socket lever) -- they never lie
    on the table on their own.
    """
    return [(f"{class_zh(cls)}  {cls}", cls) for cls in data.tax.classes
            if cls != CHASSIS_KEY and not data.tax.host_class(cls)]


def gallery_preview_text(keys: list[str]) -> str:
    """The line under the form: the keys the parts will get."""
    if not keys:
        return "（这个类别不能添加 / this class cannot be added）"
    shown = keys if len(keys) <= PREVIEW_SHOWN else [*keys[:PREVIEW_SHOWN - 1], "…", keys[-1]]
    return (f"将新建 / will create: {', '.join(shown)}\n"
            f"每张合照都会列出它们：不在的那张标「看不见」")


class GalleryPartsDialog(QDialog):
    """A class and a count for gallery-only parts (task U45): no parent, state or step.

    The class picker and the count are :class:`AddPartsDialog`'s (and its
    ``role`` / ``kind`` combo, which is part of the key: ``loose_cable.sata.01``,
    ``screw.motherboard.09``). It only reads ``data``; :meth:`values` is what
    the window hands to the session, which stages them through
    :meth:`~tda.ui.steps_model.StepTableData.add_extras` with
    ``gallery_only: true`` and saves at once.
    """

    def __init__(self, data: StepTableData, parent: Optional[QWidget] = None,
                 cls: Optional[str] = None) -> None:
        super().__init__(parent)
        self.data = data
        self.setWindowTitle(GALLERY_TITLE)
        self.class_combo = QComboBox(self)
        for label, name in gallery_class_choices(data):
            self.class_combo.addItem(label, name)
        self.disc_label = QLabel("", self)
        self.disc_combo = QComboBox(self)
        self.count_spin = QSpinBox(self)
        self.count_spin.setRange(X.MIN_COUNT, X.MAX_COUNT)
        self.count_spin.setValue(1)
        self.preview = QLabel("", self)
        self.preview.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("添加 / Add")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        intro = QLabel(GALLERY_INTRO, self)
        intro.setWordWrap(True)
        form = QFormLayout()
        form.addRow("类别 / class", self.class_combo)
        form.addRow(self.disc_label, self.disc_combo)
        form.addRow("数量 / count", self.count_spin)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.preview)
        layout.addWidget(self.buttons)
        index = self.class_combo.findData(cls or "loose_cable")
        if index >= 0:
            self.class_combo.setCurrentIndex(index)
        self.class_combo.currentIndexChanged.connect(self._on_class)
        self.disc_combo.currentIndexChanged.connect(self._refresh_preview)
        self.count_spin.valueChanged.connect(self._refresh_preview)
        self._on_class()

    @property
    def cls(self) -> str:
        return str(self.class_combo.currentData() or "")

    def _attrs(self) -> dict:
        name, vocabulary = self.data.discriminator_of(self.cls)
        value = str(self.disc_combo.currentData() or "") if vocabulary else ""
        return {name: value} if name and value else {}

    def keys(self) -> list[str]:
        """The keys the dialog would create as it stands."""
        return self.data.preview_extras(self.cls, self.count_spin.value(), self._attrs())

    def values(self) -> dict:
        """``cls``, ``count`` and ``attrs`` -- the gallery-only flag is the session's to add."""
        return {"cls": self.cls, "count": int(self.count_spin.value()),
                "attrs": self._attrs()}

    def _on_class(self, *_args) -> None:
        name, vocabulary = self.data.discriminator_of(self.cls)
        self.disc_combo.blockSignals(True)
        self.disc_combo.clear()
        if vocabulary:
            # 「（不细分）」 first (review M1): the discriminator is optional and
            # only part of the key when given -- loose_cable.01 needs no kind
            self.disc_combo.addItem(NO_DISCRIMINATOR, "")
        for value in vocabulary:
            self.disc_combo.addItem(value, value)
        # a cable is left undivided unless the annotator says which; the other
        # classes start at their first value, as the S1 dialog does
        if vocabulary and self.cls not in GALLERY_ONLY_CLASSES:
            self.disc_combo.setCurrentIndex(1)
        self.disc_combo.blockSignals(False)
        self.disc_label.setText(f"{name} / 细分" if vocabulary else "")
        self.disc_combo.setVisible(bool(vocabulary))
        self.disc_label.setVisible(bool(vocabulary))
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        keys = self.keys()
        self.preview.setText(gallery_preview_text(keys))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(keys))
