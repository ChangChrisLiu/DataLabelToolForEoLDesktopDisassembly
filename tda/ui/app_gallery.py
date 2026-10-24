"""The window's side of the components gallery (零件合照, task U10, spec 3.2b).

A gallery shot is an ordinary frame, so almost everything the window does on
one is what it does on any other.  Three things are its own and live here:

* **「看不见」**: a part the gallery asks for that is not in this shot (the
  layout was re-arranged between shots) is marked with an ``out_of_view``
  label and no pixels -- which settles its card row and its
  ``missing_shape`` (:func:`tda.core.compiler.marked_absent`).  ``I`` or the
  card row's right-click menu set it on the card's part; again takes it back.
  On a sequence frame the same key is 「被线挡住」 (task U15,
  :mod:`tda.ui.app_cable`).
* **what the gallery refuses**: the sequence's keys that have nothing to act
  on here (``T``, ``Ctrl+Enter``, the difference map, ``Tab``, the bench box,
  the ROI, a pose break), each with its reason (:mod:`tda.ui.gallery`).
  ``Enter`` -- and ``Ctrl+K``, its alias since task U16 -- is a plain commit:
  a shot is a segment of its own, with no frames behind it to keep.
* **the title**: 「零件合照 OAK1 第 k/N 张」 in the window's title bar.
"""
from __future__ import annotations

from typing import Optional

from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.class_names import instance_label
from tda.ui.gallery import (
    ABSENT,
    gallery_refusal,
    gallery_title,
    is_component_view,
)

__all__ = ["GalleryMixin", "NOT_A_GALLERY", "PICK_A_PART"]

#: 看不见 (`````` since task U22) outside the gallery.
NOT_A_GALLERY = ("「看不见」只用在零件合照里（F6–F8）；拆解序列里的可见性在复查的「可见性审查」里设 / "
                 "\"not in this shot\" is for the components gallery")
#: 看不见 with nothing to act on.
PICK_A_PART = "先在任务卡上点一条零件，再按 ` 键（看不见）/ pick a part on the card first"


class GalleryMixin:
    """The components gallery's gestures and refusals in the main window."""

    def _init_gallery(self) -> None:
        """The card's right-click 「看不见」 goes where ``I`` goes."""
        self.task_card.sigMarkAbsent.connect(self.act_mark_absent)

    # ------------------------------------------------------------- state
    def in_gallery(self) -> bool:
        """Is a components-gallery shot open?"""
        return bool(compat.is_open(self.session)
                    and is_component_view(getattr(self.session, "view", "")))

    def gallery_key_refusal(self, name: str) -> str:
        """Why action ``name`` does nothing on the open gallery shot, or ``""``."""
        if not self.in_gallery():
            return ""
        return gallery_refusal(str(self.session.view), name)

    def landing_step(self, view: str, step: Optional[int]) -> Optional[int]:
        """The step a switch to ``view`` should land on, from ``step`` now.

        Within one family (the four sequence views, or the three gallery ones)
        the number is the same moment and is kept. Across them it means
        nothing, so the frame this machine was last left on in the *other*
        family is remembered per desktop and returned instead -- ``None`` the
        first time, which opens the view on its own start (shot 1 in the
        gallery, the last step in the sequence).
        """
        memory = self.__dict__.setdefault("_family_step", {})
        if not compat.is_open(self.session):
            return step
        here = str(self.session.view)
        desktop = int(self.session.desktop)
        memory[(desktop, is_component_view(here))] = step
        if is_component_view(view) == is_component_view(here):
            return step
        return memory.get((desktop, is_component_view(view)))

    def frame_word(self, step: int) -> str:
        """``step 12`` on the sequence, ``第 2 张`` on a gallery shot (by position)."""
        if not self.in_gallery():
            return f"step {step}"
        shots = sorted(self.session.available_steps())
        index = shots.index(step) + 1 if step in shots else step
        return f"第 {index} 张"

    def gallery_title_text(self) -> str:
        """``零件合照 OAK1 第 2/5 张`` for the title bar, ``""`` outside the gallery."""
        if not self.in_gallery():
            return ""
        return gallery_title(str(self.session.view), int(self.session.current().step),
                             list(self.session.available_steps()))

    # ------------------------------------------------------------ 看不见
    def _absent_target(self, instance: Optional[str]) -> Optional[str]:
        """The part ``I`` acts on: the one asked for, edited, on the card, selected."""
        for found in (instance, getattr(self.session, "editing_instance", None),
                      self.task_card.current_instance(),
                      self.instances.selected_instance()):
            if found:
                return str(found)
        return None

    def _absent_check(self, instance: Optional[str],
                      dirty: Optional[bool] = None) -> tuple[Optional[str], Optional[dict], str]:
        """``(part, its card row, why not)`` of 「看不见」 -- ``why`` is ``""`` when it may.

        One set of gates for ``I`` and its palette button (task U13). ``dirty``:
        the editing layer has uncommitted pixels, when the caller knows it
        already (the palette, from the guide's cached facts); ``None`` asks.
        """
        if not self.in_gallery():
            return None, None, NOT_A_GALLERY
        target = self._absent_target(instance)
        row = next((r for r in self.task_card.rows()
                    if str(r.get("instance", "")) == target), None)
        if target is None or row is None or not row.get("gallery"):
            return target, None, PICK_A_PART
        label = instance_label(target, row.get("cls"), row.get("attrs"))
        if row.get("done") and not row.get("absent"):
            return target, row, (f"{label} 在这张里已经画好了：画过的零件就是在这张里 / "
                                 f"it has a shape on this shot")
        if getattr(self.session, "editing_instance", None) == target and (
                self.has_uncommitted_edit() if dirty is None else dirty):
            return target, row, f"{label} 正在画：先 Esc 放弃，再按 ` 键 / drop the edit first"
        return target, row, ""

    def absent_refusal(self, instance: Optional[str] = None,
                       dirty: Optional[bool] = None) -> str:
        """Why 「看不见」 (``I``) would do nothing now, ``""`` if it may (its button, U13).

        On a sequence frame ``I`` is 「被线挡住」 (task U15), with its own gates;
        on a 框外零件 (task U18) it is 「不在这张里」 in both families.
        """
        target = self.context_target(instance) if compat.is_open(self.session) else None
        if target is not None:
            return self.context_absent_refusal(target, dirty)
        if compat.is_open(self.session) and not self.in_gallery():
            return self.cable_refusal(instance, dirty)
        return self._absent_check(instance, dirty)[2]

    @S.guard
    def act_mark_absent(self, instance: Optional[str] = None) -> None:
        """「看不见」: ``instance`` is not in this gallery shot -- or, again, it is.

        Refused on a part the card does not ask for, on a part that already
        has a shape here (a drawn part *is* in the shot), and while pixels of
        it are being drawn. On a sequence frame the key is 「被线挡住」 instead
        (task U15, :meth:`tda.ui.app_cable.CableMixin.act_cable`). On a 框外零件
        (task U18) it is 「不在这张里」 wherever it is pressed.
        """
        target = self.context_target(instance) if compat.is_open(self.session) else None
        if target is not None:
            self.act_context_absent(target)
            return
        if compat.is_open(self.session) and not self.in_gallery():
            self.act_cable(instance)
            return
        target, row, why = self._absent_check(instance)
        if why:
            self.report(why)
            return
        label = instance_label(target, row.get("cls"), row.get("attrs"))
        editing = getattr(self.session, "editing_instance", None)
        if editing == target:
            self.session.clear_edit()       # an untouched layer: nothing to lose
            self.set_sam_instance(None)
            self._sync_editing_layer()
        back = bool(row.get("absent"))
        if back:
            self.session.clear_visibility(target)
        else:
            self.session.set_visibility(target, ABSENT)
        self.refresh_overlay()
        self.task_card.refresh()
        self.instances.refresh()
        self.timeline.refresh_statuses()
        self.update_status()
        self.report(f"{label}：" + ("回到待画（Ctrl+Z 撤销）" if back else
                                   "看不见，不在这张里（再按 ` 键或 Ctrl+Z 撤销）"))
