"""The 「框外零件…」 dialog (task U18, spec 3.2c): what the thing outside the machine is.

Three fields: the class -- every taxonomy class by its Chinese name and key,
then 「其他」 for something no class describes -- a name (required for 「其他」,
optional otherwise) and a note. The keys it will get are not asked for: a
context object is ``ctx.<class>.NN``, the next ordinal of its class among the
context objects. The dialog only reads; :meth:`ContextObjectDialog.values` is
what the window hands to
:meth:`~tda.ui.session_context.ContextMixin.add_context_object`.

Opened from the canvas' right-click menu and from the 「＋ 添加零件」 dialog of
Steps mode (:mod:`tda.ui.panels.add_parts`), which switches to Annotate first:
a context object is drawn the moment it exists.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from tda.core.context import class_choices
from tda.core.model import CONTEXT_OTHER
from tda.core.taxonomy import Taxonomy
from tda.ui import context_text as T
from tda.ui.class_names import class_zh

__all__ = ["ContextObjectDialog", "choice_labels"]


def choice_labels(tax: Taxonomy) -> list[tuple[str, str]]:
    """``(label, class)`` of every class a context object may have, ``other`` last."""
    out = []
    for cls in class_choices(tax):
        if cls == CONTEXT_OTHER:
            out.append((f"{T.OTHER_ZH}  {cls}", cls))
        else:
            out.append((f"{class_zh(cls)}  {cls}", cls))
    return out


class ContextObjectDialog(QDialog):
    """Pick a class (or 「其他」), a name and a note for one 框外零件."""

    def __init__(self, tax: Taxonomy, parent: Optional[QWidget] = None,
                 cls: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(T.DIALOG_TITLE)
        self.class_combo = QComboBox(self)
        for label, name in choice_labels(tax):
            self.class_combo.addItem(label, name)
        if cls:
            index = self.class_combo.findData(cls)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        self.label_edit = QLineEdit(self)
        self.label_edit.setToolTip(T.LABEL_TIP)
        self.note_edit = QLineEdit(self)
        self.note_edit.setPlaceholderText("可选 / optional")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("新建并画它 / Draw it")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        intro = QLabel(T.INTRO, self)
        intro.setWordWrap(True)
        form = QFormLayout()
        form.addRow("类别 / class", self.class_combo)
        form.addRow("名字 / name", self.label_edit)
        form.addRow("备注 / note", self.note_edit)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self.class_combo.currentIndexChanged.connect(self._sync)
        self.label_edit.textChanged.connect(self._sync)
        self._sync()

    @property
    def cls(self) -> str:
        return str(self.class_combo.currentData() or "")

    def values(self) -> dict:
        """The keyword arguments of ``add_context_object``."""
        return {"cls": self.cls, "label": self.label_edit.text().strip(),
                "note": self.note_edit.text().strip()}

    def _sync(self, *_args) -> None:
        """「其他」 needs a name: OK waits for one."""
        other = self.cls == CONTEXT_OTHER
        self.label_edit.setPlaceholderText("必填：它是什么 / required" if other
                                           else "可选 / optional")
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(bool(self.cls) and (not other or bool(self.label_edit.text().strip())))
