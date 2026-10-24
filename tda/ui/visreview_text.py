"""What the visibility review says on screen (task U24), without Qt.

The queue's lines, the reasons in Chinese, the refusals and the status-line
notes: one place, so the review panel, the window, the session and the tests
read the same words.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core.model import OCCLUDER_CABLE, OCCLUDER_OTHER, OCCLUDER_VIEWPOINT
from tda.core.vis_review import LAYER_PREFIX
from tda.ui.class_names import class_zh, visibility_zh

__all__ = [
    "DONE", "NOT_CONFIRMED", "NOT_IN_GALLERY", "NOTHING_TO_REVIEW", "PICK_A_ROW",
    "REASON_ZH", "VERDICT_ZH", "VIEWPOINT_BENCH", "VIEWPOINT_NEEDS_SHAPE",
    "VIEWPOINT_REMOVED", "entry_text", "occluded_by_text", "steps_text", "verdict_note",
]

#: The reason a hand label gives, as the queue says it (原因: …).
REASON_ZH: dict[str, str] = {
    OCCLUDER_CABLE: "线缆",
    OCCLUDER_VIEWPOINT: "角度",
    OCCLUDER_OTHER: "没标注的东西",
}
#: An occluder layer among the parts that hide one (spec 3.1 ``OccluderMask``).
LAYER_ZH: dict[str, str] = {"hand": "手", "arm": "手臂", "body": "身体", "tool": "工具",
                            "cable": "线（遮挡层）", "other": "遮挡层"}
#: What each verdict did, for the status line.
VERDICT_ZH: dict[str, str] = {
    "computed": "确认（按计算）",
    "label": "手动设为",
    "cable": "被线挡住",
    "viewpoint": "角度看不到",
}

#: The panel's line when nothing is left to review in this view.
DONE = "可见性审查完成"
#: ... and when this view never had anything to review.
NOTHING_TO_REVIEW = "（这个视角没有要审的）"
#: A verdict key with no row selected in the 可见性审查 tab.
PICK_A_ROW = ("先在右侧 Review 的「可见性审查」里选一条 / "
              "pick a row in the 可见性审查 tab first")
#: A verdict on a frame nobody confirmed.
NOT_CONFIRMED = "第 {step} 帧还没确认：可见性审查只看确认过的帧"
#: The components gallery has no visibility review (看不见 is its own label).
NOT_IN_GALLERY = "零件合照没有可见性审查（不在这张里用 ` 键「看不见」）"
#: 角度看不到 where it would describe nothing (the U15 refusals, its own words).
VIEWPOINT_NEEDS_SHAPE = "这一帧它没有形状：角度看不到要有形状（先在标注模式画出它）"
VIEWPOINT_REMOVED = "这一帧不要它的形状（已拆下，或这个状态不画）：没有可标的"
VIEWPOINT_BENCH = "它放在台面上（台面框），不在原位：角度看不到只标原位的零件"


def steps_text(steps: Iterable[int]) -> str:
    """``[1, 2, ..., 13]`` -> ``1–13``; one step -> ``12``."""
    values = sorted({int(s) for s in steps})
    if not values:
        return ""
    return str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}"


def occluded_by_text(names: Iterable[str]) -> str:
    """``被硬盘支架挡住`` / ``被硬盘支架、导风罩挡住`` / ``被手挡住``; ``""`` for nobody.

    Parts are named by their class (two screws are 「螺丝」 once); more than
    three names end in 「等 n 个」.
    """
    said: list[str] = []
    for name in names:
        text = (LAYER_ZH.get(name[len(LAYER_PREFIX):], name[len(LAYER_PREFIX):])
                if str(name).startswith(LAYER_PREFIX) else class_zh(None, None, str(name)))
        if text not in said:
            said.append(text)
    if not said:
        return ""
    if len(said) > 3:
        return f"被{'、'.join(said[:3])}等 {len(said)} 个挡住"
    return f"被{'、'.join(said)}挡住"


def entry_text(entry: dict) -> str:
    """One queue line: 「ram_latch.03 第 1–13 帧 · occluded_full（被硬盘支架挡住，自动）」."""
    steps = entry.get("steps") or [entry.get("step", "?")]
    why: list[str] = []
    who = occluded_by_text(entry.get("occluded_by") or ())
    if who:
        why.append(who)
    reason = entry.get("occluder")
    if reason:
        why.append(f"原因: {REASON_ZH.get(str(reason), str(reason))}")
    why.append("手动" if entry.get("manual") else "自动")
    return (f"{entry.get('instance', '')} 第 {steps_text(steps)} 帧 · "
            f"{entry.get('visibility', '')}（{'，'.join(why)}）")


def verdict_note(label: str, verdict: str, visibility: Optional[str], steps) -> str:
    """The status line after a verdict: what was said, about which frames."""
    where = f"第 {steps_text(steps)} 帧"
    if verdict in ("computed", "label"):
        what = f"{VERDICT_ZH[verdict]} {visibility_zh(visibility or '')}"
    else:
        part = "完全" if visibility == "occluded_full" else "部分"
        what = f"{VERDICT_ZH.get(verdict, verdict)}（{part}）"
    return f"可见性审查：{label} {where} → {what}，记为审过（Ctrl+Z 撤销）"
