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
    "CHANGES", "DONE", "MERGED", "NOT_CONFIRMED", "NOT_IN_GALLERY", "NOTHING_TO_REVIEW",
    "ORDER_CAPTION", "PICK_A_ROW", "REASON_ZH", "SPLIT_AT_START", "SPLIT_BUTTON",
    "SPLIT_DONE", "SPLIT_NOT_IN_RUN", "SYNC_BUTTON", "SYNC_OFFER", "SYNC_REFUSED",
    "VERDICT_ZH", "VIEWPOINT_BENCH", "VIEWPOINT_NEEDS_SHAPE", "VIEWPOINT_REMOVED",
    "VIEW_COLUMNS", "WHOLE", "entry_text", "occluded_by_text", "skipped_text",
    "steps_text", "suggestion_text", "summary_cell", "sync_note", "sync_nothing",
    "verdict_note", "view_cells",
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
    """One queue line: 「ram_latch.03 第 1–13 帧（13 帧） · occluded_full（被硬盘支架挡住，自动） · 全程」.

    Task U36: a run says how many frames it is (a gap -- a frame with no
    image -- shows as fewer than the span), and 「全程」 when it is every
    confirmed frame the part has in this view.
    """
    steps = entry.get("steps") or [entry.get("step", "?")]
    why: list[str] = []
    who = occluded_by_text(entry.get("occluded_by") or ())
    if who:
        why.append(who)
    reason = entry.get("occluder")
    if reason:
        why.append(f"原因: {REASON_ZH.get(str(reason), str(reason))}")
    why.append("手动" if entry.get("manual") else "自动")
    count = f"（{len(steps)} 帧）" if len(steps) > 1 else ""
    whole = f" · {WHOLE}" if entry.get("whole") else ""
    return (f"{entry.get('instance', '')} 第 {steps_text(steps)} 帧{count} · "
            f"{entry.get('visibility', '')}（{'，'.join(why)}）{whole}")


# --------------------------------------------------------------------------- #
# task U36: runs, the other views, the split, the sync
# --------------------------------------------------------------------------- #
#: A run that is every confirmed frame the part has in this view.
WHOLE = "全程"
WHOLE_TIP = "这一段是它在这个视角（已确认帧里）出现的全部帧：这个视角从来看不到它"
#: The caption over the summary and the list: what the numbers are, and the order.
ORDER_CAPTION = "上表：各视角看不见的帧数 / 段数（— 没标注）；零件按最长的一段排，最长的在前"
#: The other-views table's columns.
VIEW_COLUMNS = ("视角", "看得见", "看不见", "原因")
#: Where the parts hiding a run change along it.
CHANGES = "挡住它的零件在第 {steps} 帧变了：要分开答，停在那一帧点「从这帧起另算」"
#: 「从这帧起另算」, and its refusals.
SPLIT_BUTTON = "从这帧起另算"
SPLIT_TIP = ("这一段从屏幕上这一帧起分成两段，之后的按键只答后一段（理由中途变了时用）；"
             "在后一段的第一帧再点一次 = 并回去。只在这次打开时有效，不写库，没有快捷键")
SPLIT_NOT_IN_RUN = "屏幕上的第 {step} 帧不在选中的这一段里：先跳到要分开的那一帧"
SPLIT_AT_START = "第 {step} 帧已经是这一段的第一帧：停在要分开的那一帧再点"
SPLIT_DONE = "从第 {step} 帧起另算：{label} 第 {steps} 帧现在是单独一段（只在这次打开时有效）"
MERGED = "第 {step} 帧并回前一段"
#: 「同步到其他视角」, and what it says.
SYNC_BUTTON = "同步到其他视角"
SYNC_TIP = ("把刚答的这一段的判定（标签 + 原因）写到其他视角同样的帧上：只写那边标签一样（同样看不见）、"
            "还没审的帧，不覆盖任何判定；那边有零件挡着的不写「角度 / 线缆」。一次 Ctrl+Z 撤销。没有快捷键")
SYNC_IDLE = "先答一段；答完如果其他视角同样的帧也看不见、还没审，这个按钮可以把判定同步过去"
SYNC_REFUSED = {
    "no_verdict": "这一段还没有判定：先按键回答它，再同步到其他视角",
    "not_hidden": "同步只用于看不见的标签（完全 / 部分遮挡、画面外）",
}
SKIP_ZH = {"verdict": "已有判定", "different": "那边的标签不一样", "covered": "那边有零件挡着它",
           "unconfirmed": "那边没确认", "absent": "那边没有它", "no_shape": "那边没形状"}
SYNC_OFFER = "；其他视角还有 {n} 帧同样看不见、还没审：可以点「同步到其他视角」"


def skipped_text(skipped: dict) -> str:
    """「已有判定 3 帧、那边没确认 40 帧」 -- the frames a sync left, by reason."""
    return "、".join(f"{SKIP_ZH.get(k, k)} {n} 帧" for k, n in skipped.items() if n)


def sync_nothing(skipped: dict) -> str:
    """The refusal when no frame of another view may take the verdict."""
    why = skipped_text(skipped)
    return "其他视角没有可以同步的帧" + (f"（{why}）" if why else "（其他视角没有确认过的帧）")


def sync_note(label: str, result: dict) -> str:
    """The status line after a sync: where, how many, and what was left alone."""
    where = "、".join(f"{view} 第 {steps_text(steps)} 帧"
                     for view, steps in sorted((result.get("written") or {}).items()))
    note = (f"已同步到其他视角：{label} {where}，共 {result.get('count', 0)} 帧，"
            f"记为审过（Ctrl+Z 撤销）")
    if result.get("replaced"):
        note += f"；其中 {result['replaced']} 帧原来手动标的原因换成了这个判定"
    left = skipped_text(result.get("skipped") or {})
    if left:
        note += f"；没动：{left}"
    withheld = {v: s for v, s in (result.get("withheld") or {}).items() if s}
    if withheld:
        note += "；" + "、".join(f"{v} 第 {steps_text(s)} 帧露出了确认时没有的像素，没代签"
                                for v, s in sorted(withheld.items()))
    elif result.get("conflicts"):
        note += "；有的帧原来就和确认时不一样：冲突留在那个视角的 Conflicts 里"
    return note


def summary_cell(value, state: str) -> str:
    """One cell of the summary: 「40/2」 (frames / stretches), 「·」, 「—」 or 「未确认」."""
    if value is None:
        return "未确认" if state == "unconfirmed" else "—"
    frames, stretches = int(value[0]), int(value[1])
    return "·" if frames == 0 else f"{frames}/{stretches}"


def view_cells(row: dict) -> tuple[str, str, str, str]:
    """One line of the other-views table: view, 看得见, 看不见, 原因."""
    name = f"{row['view']}（这里）" if row.get("here") else str(row["view"])
    if row.get("state") == "none":
        return name, "—", "—", "未标注"
    if row.get("state") == "unconfirmed":
        return name, "—", "—", "只有草稿，没确认"
    why: list[str] = []
    who = occluded_by_text(row.get("by") or ())
    if who:
        why.append(who)
    for reason, count in sorted((row.get("reasons") or {}).items()):
        if reason == "computed":
            if not who:
                why.append(f"自动 {count}")
            continue
        text = "手动" if reason == "manual" else REASON_ZH.get(reason, reason)
        why.append(f"{text} {count}")
    if row.get("absent"):
        why.append(f"没有它 {row['absent']}")
    if row.get("unconfirmed"):
        why.append(f"没确认 {row['unconfirmed']}")
    return name, str(row.get("seen", 0)), str(row.get("hidden", 0)), "，".join(why) or "·"


def suggestion_text(suggestion: Optional[dict], entry: dict) -> str:
    """The suggested verdict, and why the other views point at it; ``""`` for none."""
    if not suggestion:
        return ""
    key = suggestion.get("key", "")
    if suggestion.get("kind") == "viewpoint":
        part = "完全" if suggestion.get("label") == "occluded_full" else "部分"
        return (f"建议：角度看不到（{part}，{key}）——{'、'.join(suggestion.get('views') or ())} "
                f"同样的帧看得见，这里没有挡住它的零件")
    too = ""
    if suggestion.get("scan_too"):
        too = ("，扫描仪视角也看不见" if suggestion.get("label") == "occluded_full"
               else "，扫描仪视角也被挡着")
    if suggestion.get("verdict") == "cable":
        return f"建议：被线挡住（{key}）——原因: 线缆{too}"
    who = occluded_by_text(entry.get("occluded_by") or ()) or "被挡住"
    return f"建议：按计算确认（{key}）——{who}{too}"


def verdict_note(label: str, verdict: str, visibility: Optional[str], steps) -> str:
    """The status line after a verdict: what was said, about which frames."""
    where = f"第 {steps_text(steps)} 帧"
    if verdict in ("computed", "label"):
        what = f"{VERDICT_ZH[verdict]} {visibility_zh(visibility or '')}"
    else:
        part = "完全" if visibility == "occluded_full" else "部分"
        what = f"{VERDICT_ZH.get(verdict, verdict)}（{part}）"
    return f"可见性审查：{label} {where} → {what}，记为审过（Ctrl+Z 撤销）"
