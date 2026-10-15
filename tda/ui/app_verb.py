"""``Shift+T`` and the instance list's verb button: opened, or taken out (task U9).

On D2/scan the last frame's card asked for six parts -- two covers and four
latches -- that were lying in the components pile: the sheet said "Screw
Cover", "Power Module Blocker", and the importer drafted ``open`` for all of
them. Only the pictures say which, so the annotator says it: select the part
(or click its card row) and press ``Shift+T`` -- the action that opened it
becomes a ``remove`` (:meth:`tda.ui.session_verb.VerbMixin.correct_verb`) and the
part leaves every frame from that step on. A part in the ``Removed`` rows goes
the other way: 「其实只是打开了」.

It is a step-table edit made from Annotate mode, so the Steps panel, if it was
built, is re-read the next time Steps mode is shown when the database moved
under it and it has no edits of its own (:meth:`VerbMixin.refresh_stale_steps`)
-- its own copy of the actions would otherwise write the old verb back on the
next Apply; with edits of its own it is left alone and the annotator is told.
"""
from __future__ import annotations

from typing import Optional

from tda.ui import app_actions as A
from tda.ui import app_support as S
from tda.ui import session_api as api
from tda.ui.gallery import NO_VERB

__all__ = ["NO_VERB_TARGET", "STEPS_MOVED", "VerbMixin", "verb_note"]

#: ``Shift+T`` with nothing selected and nothing being drawn.
NO_VERB_TARGET = ("Shift+T 改的是实例表里选中的零件（拆下的零件勾上 Removed 才列出）："
                  "先在右下实例表点一行 / select a part in the instance table first")
#: Prefix of a refusal.
REFUSED = "改不了："
#: Steps mode shown again with edits of its own while the database moved (U9).
STEPS_MOVED = ("Steps 表里有没 Apply 的修改，数据库却在这期间变了（标注模式里按过 Shift+T）："
               "现在 Apply 会把动词按这张表写回去（盖掉那次修改）；Revert 丢掉没 Apply 的修改、"
               "读入新的 / unapplied edits here, and the database changed underneath")


def verb_note(label: str, result: dict) -> str:
    """The status line after a correction, in the annotator's words.

    ``盖板 cover.01 在第 38 步是拆下的（不是打开）：从第 38 步起它不在机箱里了；
    这些帧上它的形状不再使用（11 帧）`` -- and the mirror for 「其实只是打开了」,
    with the frames it now has to be drawn on.
    """
    fix = result["fix"]
    if fix.new == "remove":
        text = (f"{label} 在第 {fix.step} 步是拆下的（不是打开）：从第 {fix.step} 步起"
                f"它不在机箱里了")
        lost = result.get("lost") or []
        if lost:
            text += f"；这些帧上它的形状不再使用（{len(lost)} 帧）"
    else:
        text = (f"{label} 在第 {fix.step} 步只是打开了（不是拆下）：从第 {fix.step} 步起"
                f"它还在机箱里")
        todo = result.get("todo") or []
        if todo:
            text += f"，要画出来（{len(todo)} 帧）"
    return text + "（Ctrl+Z 撤销）"


class VerbMixin:
    """``Shift+T`` / the verb button, for :class:`tda.ui.app.MainWindow`."""

    def _init_verb(self) -> None:
        self.instances.sigCorrectVerb.connect(self.on_verb_picked)

    def _verb_target(self) -> Optional[str]:
        """The part ``Shift+T`` is about: the selected row (removed ones too), else the edit's."""
        return (self.instances.selected_instance()
                or self.instances.selected_removed_instance()
                or getattr(self.session, "editing_instance", None))

    @S.guard
    def act_correct_verb(self) -> None:
        """``Shift+T``: the selected open part was taken out (a removed one: only opened)."""
        if self._gallery_verb_refused():
            return
        instance = self._verb_target()
        if not instance:
            self._refuse_state(NO_VERB_TARGET)
            return
        if not callable(getattr(self.session, "correct_verb", None)):
            self._refuse_state(f"{REFUSED}这个会话不支持 / not supported by this session")
            return
        self.correct_verb_here(instance)

    @S.guard
    def on_verb_picked(self, instance: str, verb: str) -> None:
        """The instance list's button: the same gesture as ``Shift+T``, Annotate mode only."""
        if self.mode != A.MODE_ANNOTATE:
            self._refuse_state(f"{REFUSED}只在标注模式可用 / Annotate mode only")
            return
        if self._gallery_verb_refused():
            return
        self.correct_verb_here(instance, verb)

    def verb_button_refusal(self, dirty: bool) -> str:
        """Why 「其实拆下」 (``Shift+T``) cannot run now, ``""`` if it may (task U13).

        The slot's gates in its order and words; ``dirty`` is the guide's cached
        "the editing layer has uncommitted pixels". The row's ``verb_fix`` --
        the list's own offer -- answers for a part it offers; for any other the
        core's plan is asked once (:meth:`StateMixin._why_once`).
        """
        in_gallery = getattr(self, "in_gallery", None)
        if in_gallery is not None and in_gallery():
            return NO_VERB
        instance = self._verb_target()
        if not instance:
            return NO_VERB_TARGET
        if dirty and getattr(self.session, "editing_instance", None) == instance:
            return api.VERB_NEEDS_CLEAN_LAYER.format(instance=instance)
        row = self.instances.row_of(instance)
        if row is not None and row.get("verb_fix"):
            return ""
        gate = getattr(self.session, "verb_gate", None)
        return self._why_once("verb", instance, gate) if callable(gate) else ""

    def _gallery_verb_refused(self) -> bool:
        """A components-gallery shot is open: its number is a shot, not a step (U10).

        The step table is never changed from there -- the verb would flip at
        the step that happens to carry the shot's number. The key and the
        button both stop here; :meth:`tda.ui.session_verb.VerbMixin.verb_offers`
        offers nothing on a shot, so the button never shows a verb there.
        """
        in_gallery = getattr(self, "in_gallery", None)
        if in_gallery is None or not in_gallery():
            return False
        self._refuse_state(f"{REFUSED}{NO_VERB}")
        return True

    def correct_verb_here(self, instance: str, to: Optional[str] = None) -> Optional[dict]:
        """Flip the verb, repaint what reads it, and say what changed."""
        key = self.session.current()
        label = self._label_of(instance)[0] or instance   # before: its card row may go
        try:
            result = self.session.correct_verb(instance, to)
        except ValueError as refused:            # SessionRefusal is one
            self.logger.info("correct_verb refused %s at step %s: %s", instance, key.step,
                             refused)
            self.instances.refresh()
            self._refuse_state(f"{REFUSED}{refused}")
            return None
        fix = result["fix"]
        self.logger.info("correct_verb %s step %s: %s -> %s (asked at %s, %d frames lost)",
                         instance, fix.step, fix.old, fix.new, key.step,
                         len(result.get("lost") or []))
        self._sync_editing_layer()          # an untouched edit of the part was dropped
        self.task_card.refresh()
        self.timeline.refresh_statuses()
        self.forget_card_hints()
        note = verb_note(label, result)
        self.update_status()
        self.report(note, hold_ms=6000)
        self.flash_visibility_note(note)
        return result

    def refresh_stale_steps(self) -> None:
        """Re-read the Steps panel when the database moved under it (task U9).

        Only when the stored table changed since the panel's own load or Apply
        (:meth:`StepTablePanel.database_moved`) -- a plain Steps → Annotate →
        Steps round trip reloads nothing. And only when the panel has no edits
        of its own since then: those are never dropped here. With both, the
        annotator is told, and Apply / Revert / the leave-Steps question decide
        -- not re-reading would let the next Apply write the old verb back, and
        re-reading would lose the staged edits without a word.
        """
        panel = getattr(self, "_steps_panel", None)
        if panel is None or not panel.database_moved():
            return
        if getattr(self, "_steps_dirty", False) or panel.has_unapplied_edits():
            panel.status.setText(STEPS_MOVED)
            self.report(STEPS_MOVED, hold_ms=8000)
            return
        panel.set_desktop(int(panel.desktop))
