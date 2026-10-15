"""The components gallery (零件合照) in the annotator's words (task U10, spec 3.2b).

After each teardown the operators laid every removed part out on the table and
photographed the layout several times with both OAK cameras and the RealSense.
Those shots are three views of their own (:data:`tda.core.model.COMPONENT_VIEWS`):
a shot is an ordinary frame, so drawing, SAM, the compiler, Review and the
exports work on it unchanged. What is different is said here, once, for every
place that says it -- the window title, the task card, the guide, the status
bar -- and so is what the gallery refuses:

* there is no state on a shot (every part on the table is ``removed``), so
  ``T`` and the ✂ rows of the sequence do not exist;
* there is no neighbour to compare with (the layout was re-arranged between
  shots), so no difference map, no prompt box from it and no ``Tab`` flash;
* the whole image is the working area: no chassis ROI, no bench box, no pose
  breaks.

Qt-free: the window, the session and the panels all read it.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import COMPONENT_CAMERA, is_component_view

__all__ = [
    "ABSENT", "ABSENT_CHIP", "ABSENT_KEY", "GALLERY_REFUSALS", "GALLERY_SENTENCE",
    "NO_STATE", "NO_VERB", "camera_label", "gallery_header", "gallery_refusal", "gallery_title",
    "is_component_view", "view_label",
]

#: The camera of each gallery view, as the annotator says it.
_CAMERA_ZH = {"oak1": "OAK1", "oak2": "OAK2", "rs": "RealSense"}

#: The visibility label that says "this part is not in this shot" (看不见).
ABSENT = "out_of_view"
#: The key that sets it on the card's part (:data:`tda.ui.app_actions.ACTIONS`).
ABSENT_KEY = "I"
#: The chip of a card row the annotator marked 看不见.
ABSENT_CHIP = "看不见"
#: What a gallery row asks for.
GALLERY_SENTENCE = (f"这张里看得到就画出它（整个零件）；不在这张里：点「看不见」"
                    f"（右键这一条，或按 {ABSENT_KEY}）")
#: Why a gallery shot has no state to set (``T``, the state combo).
NO_STATE = "零件合照里没有状态：桌上的零件都是拆下来的，只画它本身"
#: Why a gallery shot does not change the step table (``Shift+T``, the 「其实…」 button):
#: a shot's number is not a step, so the verb would change at the wrong step.
NO_VERB = ("零件合照不改步骤表：「其实拆下来了 / 其实只是打开了」到拆解序列"
           "（OAK1/OAK2/RS）里那一步的帧上用（Shift+T）")

#: What the gallery refuses, per action name of :data:`tda.ui.app_actions.ACTIONS`.
GALLERY_REFUSALS: dict[str, str] = {
    "cycle_state": NO_STATE,
    "state_range": NO_STATE,                       # Ctrl+T (U11)
    "confirm_cut_copy": ("零件合照没有切开的形状副本：每张是独立的一段，"
                         "画好就 Enter"),            # Ctrl+Shift+K (U11 round 2)
    "correct_verb": NO_VERB,
    "commit_split": "零件合照每张是独立的，没有「从这帧起新版本」：用 Enter 提交",
    "toggle_heat": "零件合照没有差异图：每张照片之间零件被重新摆过",
    "cycle_prompt_box": "零件合照没有差异图给的提示框：用 S 点选或 X 框选",
    "flash_compare": "零件合照没有可对照的上一帧：每张照片之间零件被重新摆过",
    "flash_other": "零件合照没有可对照的上一帧：每张照片之间零件被重新摆过",
    "tool_bench_box": "零件合照没有台面框：零件都画掩码",
    "edit_roi": "零件合照的工作范围就是整张图，没有机箱 ROI",
    "split_pose": "零件合照每张自成一段，不用断开位姿段",
}


def camera_label(view: str) -> str:
    """``OAK1`` / ``OAK2`` / ``RealSense`` for a gallery view, else the view."""
    return _CAMERA_ZH.get(COMPONENT_CAMERA.get(str(view), ""), str(view))


def view_label(view: str) -> str:
    """The view as the status bar and the buttons say it: ``零件合照·OAK1``."""
    if is_component_view(view):
        return f"零件合照·{camera_label(view)}"
    return str(view)


def gallery_title(view: str, shot: Optional[int], shots: list[int]) -> str:
    """``零件合照 OAK1 第 2/5 张``."""
    total = len(shots) if shots else (shot or 0)
    index = (sorted(shots).index(shot) + 1) if shots and shot in shots else (shot or 0)
    return f"零件合照 {camera_label(view)} 第 {index}/{total} 张"


def gallery_header(view: str, shot: Optional[int], shots: list[int]) -> str:
    """The card's and the window's sentence for one shot (spec 3.2b)."""
    return (f"{gallery_title(view, shot, shots)}：把这张里看得到的零件都画出来；"
            f"不在这张里的点「看不见」")


def gallery_refusal(view: str, action: str) -> str:
    """Why ``action`` does nothing on a gallery shot, or ``""``."""
    if not is_component_view(view):
        return ""
    return GALLERY_REFUSALS.get(str(action), "")
