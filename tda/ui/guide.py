"""What the annotator should do next, as data: the "现在做什么 / What now" plan.

The second trial (task U2b) ended with an annotator who did not know what the
task card's other four items were, and a double-click that looked like it did
nothing -- it had started an edit, and one word in the status bar said so.
The tool's gestures were all there; nothing on the main window said *which*
one was next.

This module is that sentence and nothing else.  :func:`plan_for` reads a small
snapshot of the window (:class:`GuideFacts`) and answers with a
:class:`GuidePlan`: the phase, the one "现在：…" line that names the exact key
or button, the numbered list of the frame's five steps with the current one
marked, and the palette action worth highlighting.  No Qt, no session: the
window gathers the facts from state it already holds, so asking costs a few
dozen attribute reads, and the plan is a frozen value the panel compares with
the previous one before it repaints anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from tda.ui import session_api as api

__all__ = [
    "COMMIT_KEY", "CardItem", "DONE", "GuideFacts", "GuidePlan", "NOW", "TODO", "WARN",
    "PHASES", "plan_for", "steps_text",
]

#: Marker states of one line of the numbered list.
DONE, NOW, TODO, WARN = "done", "now", "todo", "warn"

PHASE_STEPS = "steps"
PHASE_REVIEW = "review"
PHASE_CLOSED = "closed"
PHASE_RAW_MISSING = "raw_missing"
PHASE_NO_IMAGE = "no_image"
PHASE_FLASH = "flash"
#: A ``P`` polygon or a ``Y`` circle half drawn (task U5a): ``Enter`` and
#: ``Esc`` are the shape's until it is filled or dropped.
PHASE_SHAPE = "shape"
PHASE_GHOST = "ghost"
PHASE_WARNING = "warning"
PHASE_SCOPE = "scope"
PHASE_ROI = "roi"
PHASE_BENCH = "bench"
PHASE_DRAW_EMPTY = "draw_empty"
PHASE_DRAW_PIXELS = "draw_pixels"
#: A stored shape loaded and not touched: ``Enter`` has nothing to commit.
PHASE_LOADED = "loaded"
PHASE_PICK = "pick"
#: Every card row is done and the card's pane still lists problems that make
#: ``Space`` refuse (U2d): the guide must not send anybody to that key.
PHASE_BLOCKED = "blocked"
PHASE_CONFIRM = "confirm"
PHASE_CONFIRMED = "confirmed"
PHASES: tuple[str, ...] = (
    PHASE_STEPS, PHASE_REVIEW, PHASE_CLOSED, PHASE_RAW_MISSING, PHASE_NO_IMAGE,
    PHASE_FLASH, PHASE_SHAPE, PHASE_GHOST, PHASE_WARNING, PHASE_SCOPE, PHASE_ROI,
    PHASE_BENCH,
    PHASE_DRAW_EMPTY, PHASE_DRAW_PIXELS, PHASE_LOADED, PHASE_PICK, PHASE_BLOCKED,
    PHASE_CONFIRM, PHASE_CONFIRMED,
)

#: The key of each commit action, for the area warning's "press it again":
#: since task U16 one scope can be either key's (``keyframe`` is the commit's
#: where the version starts and 改全部's everywhere), so the warning remembers
#: the action that raised it.  The commit's own key otherwise.  The left-hand
#: keys of task U22 (``tests/test_u22_keys.py`` pins them to the key map;
#: this module stays Qt-free, so it cannot read the map itself).
COMMIT_KEY = "Q"
_WARNING_KEYS: dict[str, tuple[str, str]] = {
    "commit_all": ("Ctrl+Q", "commit_all"),
    "commit_override": ("Alt+Q", "commit_override"),
}
#: ... and for a warning that does not say: the scope's key, as before U16.
_SCOPE_KEYS: dict[str, tuple[str, str]] = {
    api.SCOPE_FRAME_OVERRIDE: ("Alt+Q", "commit_override"),
}

#: Card kinds that are work to *do* on the canvas.  ``state_only`` and
#: ``confirm`` items are never "the next part to click".
_WORK_KINDS = (api.KIND_ADD_SHAPE, api.KIND_SPLIT_KEYFRAME, api.KIND_ADD_BENCH_BOX)


@dataclass(frozen=True)
class CardItem:
    """One task-card row as the guide needs it."""

    instance: str
    kind: str
    done: bool
    #: What the annotator reads: ``导风罩 cover.01``.
    label: str = ""


@dataclass(frozen=True)
class GuideFacts:
    """Everything the plan depends on; all of it already held by the window."""

    mode: str = "annotate"
    is_open: bool = True
    has_image: bool = True
    #: Why no image could be read at all (the raw drive is missing), or ``""``.
    raw_missing: str = ""
    view: str = "scan"
    step: Optional[int] = None
    #: The frame the card is diffed against; ``None`` on the start frame.
    neighbour: Optional[int] = None
    #: The start frame is the *first* step (browsing forward): the machine
    #: whole, not taken apart (U2b round 2, I3).
    forward_start: bool = False
    frame_confirmed: bool = False
    roi_stored: bool = False
    roi_editing: bool = False
    roi_unanswered: bool = False
    #: Label of the instance being edited, or ``None``.
    editing: Optional[str] = None
    #: The task-card kind of the row being edited (``""`` when it is not on
    #: the card): a ✂ row is a new version from this frame on.
    editing_kind: str = ""
    layer_pixels: bool = False
    layer_dirty: bool = False
    warning: bool = False
    #: The scope the area warning was raised for (its second press repeats it).
    warning_scope: str = ""
    #: The action whose key raised it (``commit_all`` …), ``""`` for Enter's.
    warning_action: str = ""
    #: The scope the non-modal bar is suggesting, or ``""``.
    scope: str = ""
    #: That scope in the annotator's words (``它在 机箱 chassis 上面（层级）``).
    scope_text: str = ""
    #: ``L`` was pressed once: the bar is asking whether the layering change's
    #: reach is meant (task U6).
    layering_asked: bool = False
    #: ``Ctrl+Enter`` was pressed once: the bar is asking about the confirmed
    #: frames 改全部 reaches (task U16).
    all_asked: bool = False
    ghost: bool = False
    #: Label of the part the bench box is armed for, or ``None``.
    bench: Optional[str] = None
    flashing: bool = False
    sam_ready: bool = True
    #: SAM's load failed (task U6): why, in a few words; ``""`` while it is
    #: loading or loaded.  The draw step then names the tools that work.
    sam_failed: str = ""
    items: tuple[CardItem, ...] = ()
    #: How many problems in the task card's pane stop ``Space``: the codes
    #: ``confirm_frame`` refuses on (``tda.core.truth_verify.is_blocking``),
    #: never the notes it accepts.  The window reads it off the card (U2d).
    blockers: int = 0
    #: When no blocking line in the pane can be clicked, the first one's own
    #: instruction ("输入刚变了：再按一次 Space"); ``""`` when one can (U2e).
    blocker_hint: str = ""
    #: The filled shape half drawn (task U5a): ``"polygon"``, ``"circle"`` or
    #: ``""``, and how many vertices the polygon has so far.
    shape: str = ""
    shape_vertices: int = 0
    #: Open conflicts of the view, as the Review queue lists them (task U7):
    #: more than one is when ``Shift+N`` is worth naming.
    open_conflicts: int = 0
    #: The confirmed frames a re-trace of the shape being edited reaches
    #: (``AnnotationSession.retrace_reach``), computed once when the edit
    #: starts; empty for a new shape (task U7).
    retrace_frozen: tuple[int, ...] = ()
    #: A components-gallery shot (task U10): its title, ``零件合照 OAK1 第 2/5
    #: 张``; ``""`` on the teardown sequence.
    gallery: str = ""
    #: Review's 「可见性审查」 tab is on top (task U24), and how many of its
    #: rows are still to review (``-1``: not known).
    vis_review: bool = False
    vis_pending: int = -1


@dataclass(frozen=True)
class GuidePlan:
    """The answer: what to say, and which palette button deserves a border."""

    phase: str
    title: str
    #: The one sentence naming the exact next key or button.
    now: str
    #: ``(marker state, text)`` per line of the numbered list; empty outside
    #: the per-frame flow (Review, Steps, no image).
    steps: tuple[tuple[str, str], ...] = ()
    #: :data:`tda.ui.app_actions.ACTIONS` name of the suggested button, or ``""``.
    action: str = ""
    extra: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# the five steps of a frame
# --------------------------------------------------------------------------- #
ROI_NOW = "回答机箱范围（ROI）：拖紫红框的白色小方块调整，Q 保存 / Esc 先跳过"
#: A stored ROI opened to check it: Esc keeps it, it skips nothing (round 2).
ROI_EDIT_NOW = "机箱范围（ROI）：拖白色小方块调整，Q 保存 / Esc 不改"
ROI_DONE = "机箱范围（ROI）✓"
#: A components-gallery shot's first line (task U10): no ROI question at all.
ROI_GALLERY = "零件合照：整张图都是工作范围 ✓"
ROI_NONE = "机箱范围（ROI）：这一段不用 ✓"
ROI_WARN = ("机箱范围（ROI）还没确认：画布下方点「确认建议框」，或 Shift+R 重画"
            "（不挡你继续）")
PICK = "在右边任务卡上单击一个零件"
DRAW = ("画：S 在零件上点一下 / X 拖框；W 画笔补、E 橡皮修边；D 连点填一块；"
        "R 圆形：拖一下画一颗螺丝")
#: Step ③ when SAM's load failed (task U6): S and X do nothing, so say what does.
DRAW_NO_SAM = "SAM 没加载：用 R 圆形 / W 画笔 / D 多边形"
#: The filled shapes half drawn (task U5a): the "now" line and step ③.
POLYGON_NOW = ("现在：多边形已点 {n} 个点 — 沿边接着点；Q / 双击 / 点回第一个点 填上，"
               "Backspace 删一个点，Esc 取消")
POLYGON_DRAW = "多边形 {n} 个点：Q / 双击 / 点回起点 填上"
CIRCLE_NOW = "现在：拖到零件边缘松手，圆就填进去；Esc 取消"
CIRCLE_DRAW = "圆形：拖到边缘松手"
DRAW_BENCH = "在台面上拖一个框框住它（台面框 V），松手就存好"
#: Step ④ (task U16): the commit is from this frame on; the frames walked keep
#: theirs.  ``Q`` since task U22 -- and the one line that still names Enter,
#: so an annotator whose fingers know it is not told it stopped working.
COMMIT = "Q（Enter）提交（从这帧起，走过的帧不变；只这一帧用 Alt+Q）"
COMMIT_SPLIT = "Q 提交（从这帧起新版本）"
#: Step ④ while a layering suggestion is on the bar (task U6).
LAYERING_COMMIT = "层级建议：Q 只改形状 / L 改层级（先看影响哪些帧，再按 L）"
#: Step ④ while 改全部 asks (task U16).
ALL_COMMIT = "改全部：再按 Ctrl+Q 确定 / Esc 取消"
CONFIRM = "全部画完：Space 确认整帧，自动退到上一帧"
#: Steps ④ and ⑤ on a components-gallery shot (task U10): no ✂, and Space
#: walks forwards.
COMMIT_GALLERY = "Q 提交；不在这张里的零件：任务卡上右键「看不见」或按 ` 键"
CONFIRM_GALLERY = "每个零件都画了或「看不见」：Space 确认这张，自动到下一张"
#: The same line while the card's pane lists what stops Space (U2d).
CONFIRM_BLOCKED = "先处理任务卡下面的 {n} 个问题（它们挡住 Space）"
NOTHING_TO_DRAW = "这一帧不用画"
#: Re-tracing a confirmed shape (task U7): every confirmed frame the keyframe
#: reaches comes back as a conflict, all answered by accepting the new value.
RETRACE_NOTE = "重画会影响 {n} 个已确认帧，提交后用 Shift+N 一次接受"
#: Review mode with several conflicts: when the bulk answer is the right one.
REVIEW_BULK = "这些冲突是你刚重画的形状造成的 → Shift+N 一次接受（共 {n} 个）"
#: Review's 「可见性审查」 tab (task U24): what the keys say about the selected row.
VIS_REVIEW = ("可见性审查：选一条（跳到那一帧、选中零件）→ 对就 Space（整段 Shift+Space）；"
              "不对：数字键手动设、` 键被线挡住、Z 角度看不到（加 Shift 是部分）；Ctrl+Z 撤销")
#: ... and with nothing left to review in this view.
VIS_REVIEW_DONE = "可见性审查完成：这个视角没有要审的了（换视角 F1–F4 继续）"


def steps_text(steps) -> str:
    """``[1, 2, ..., 40]`` -> ``1–40``; ``[39, 40]`` -> ``39、40``; runs of three or more joined."""
    values = sorted({int(s) for s in steps or ()})
    out: list[str] = []
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[end + 1] == values[end] + 1:
            end += 1
        if end - index >= 2:
            out.append(f"{values[index]}–{values[end]}")
        else:
            out.extend(str(v) for v in values[index:end + 1])
        index = end + 1
    return "、".join(out)


def _label(item: CardItem) -> str:
    return item.label or item.instance


_BILINGUAL_SPLIT = re.compile(r"\s*/ (?=[A-Za-z])")
_CJK = re.compile(r"[一-鿿]")


def _chinese_half(text: str) -> str:
    """``"中文 / English — path"`` -> ``"中文"``; anything else unchanged."""
    head = _BILINGUAL_SPLIT.split(text, 1)[0].strip()
    return head if _CJK.search(head) else text


def _title(facts: GuideFacts) -> str:
    step = facts.step
    if step is None:
        return "现在做什么"
    if facts.gallery:
        return facts.gallery          # 零件合照 OAK1 第 k/N 张 (task U10)
    if facts.neighbour is None:
        look = "还没开始拆的样子" if facts.forward_start else "已经拆完的样子"
        return f"第 {step} 帧（起点：{look}）"
    return f"第 {step} 帧（对照第 {facts.neighbour} 帧）"


def _roi_line(facts: GuideFacts) -> tuple[str, str]:
    if facts.gallery:
        return (DONE, ROI_GALLERY)
    if facts.roi_editing:
        return (NOW, ROI_EDIT_NOW if facts.roi_stored else ROI_NOW)
    if facts.roi_stored:
        return (DONE, ROI_DONE)
    if facts.roi_unanswered:
        return (WARN, ROI_WARN)
    return (DONE, ROI_NONE)


def _open_work(facts: GuideFacts) -> list[CardItem]:
    return [item for item in facts.items if not item.done and item.kind in _WORK_KINDS]


def _steps(facts: GuideFacts, at: int, *, pick_text: str = "", draw_text: str = "",
           commit_text: str = "", nothing: bool = False) -> tuple:
    """The five lines, with ``at`` the one being done now (``-1``: none).

    The last one says Space only while nothing in the card's pane stops it,
    in every phase: "全部画完：Space 确认" over a pane of blockers was a promise
    the confirmation would break (U2d round 2).
    """
    texts = [
        None,
        pick_text or PICK,
        draw_text or (DRAW_NO_SAM if facts.sam_failed else DRAW),
        commit_text or (COMMIT_GALLERY if facts.gallery else COMMIT),
        CONFIRM_BLOCKED.format(n=facts.blockers) if facts.blockers else (
            CONFIRM_GALLERY if facts.gallery else CONFIRM),
    ]
    if nothing:
        texts[1] = texts[2] = texts[3] = NOTHING_TO_DRAW
    out = [_roi_line(facts)]
    for index in range(1, 5):
        if at < 0 or index < at:
            state = DONE
        elif index == at:
            state = NOW
        else:
            state = TODO
        out.append((state, texts[index]))
    return tuple(out)


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def plan_for(facts: GuideFacts) -> GuidePlan:
    """The phase the annotator is in and the next thing to press."""
    if facts.mode == "steps":
        return GuidePlan(PHASE_STEPS, "Steps 步骤核对",
                         "Steps：核对步骤表和实例表，改完点 Apply；然后点顶部 "
                         "Annotate 回到画布")
    if not facts.is_open:
        return GuidePlan(PHASE_CLOSED, "现在做什么",
                         "这台机器的这个视角没有帧：在顶部换一台机器或换视角"
                         "（F1–F4，零件合照 F6–F8）")
    if not facts.has_image and facts.raw_missing:
        # The resolver's own sentence already says what to plug in and that
        # F5 looks again; saying it twice was noise (round 2).  It is
        # bilingual ("中文 / English — path"): the guide speaks Chinese, and the
        # raw-drive bar above the canvas keeps the whole sentence and the path.
        reason = _chinese_half(facts.raw_missing)
        advice = "" if "F5" in reason else " — 接上数据盘后按 F5"
        return GuidePlan(PHASE_RAW_MISSING, _title(facts),
                         f"读不到原始图像：{reason}{advice}")
    if not facts.has_image:
        return GuidePlan(PHASE_NO_IMAGE, _title(facts),
                         f"这一帧在 {facts.view} 视角没有图像 — 按 PgDn / PgUp "
                         f"翻到有图像的帧")
    if facts.mode == "review":
        if facts.vis_review:
            return GuidePlan(PHASE_REVIEW, "复查 Review",
                             VIS_REVIEW_DONE if facts.vis_pending == 0 else VIS_REVIEW,
                             action="" if facts.vis_pending == 0 else "vis_review_computed")
        now = ("复查模式，画布只读：在右侧队列选一条 → Enter 接受这一帧，"
               "R 回标注模式返工；冲突：K 保留旧的，N 采用新的")
        if facts.open_conflicts > 1:
            # a re-trace of a confirmed shape: one answer for all of them (U7)
            return GuidePlan(PHASE_REVIEW, "复查 Review",
                             f"{now}；{REVIEW_BULK.format(n=facts.open_conflicts)}",
                             action="review_accept_all_new")
        return GuidePlan(PHASE_REVIEW, "复查 Review", now)
    if facts.flashing:
        return GuidePlan(PHASE_FLASH, _title(facts),
                         "正在对照另一帧（画布上不是这一帧）— 松开 Tab 回来")
    return _annotate_plan(facts)


def _annotate_plan(facts: GuideFacts) -> GuidePlan:
    title = _title(facts)
    editing = facts.editing
    if facts.shape and editing:
        # First, as it is first in the Enter / Esc chains: the gesture in the
        # annotator's hand owns both keys until it is filled or dropped (U5a).
        picked = f"已选中：{editing}"
        if facts.shape == "polygon":
            n = int(facts.shape_vertices)
            return GuidePlan(PHASE_SHAPE, title, POLYGON_NOW.format(n=n),
                             _steps(facts, 2, pick_text=picked,
                                    draw_text=POLYGON_DRAW.format(n=n)), "")
        return GuidePlan(PHASE_SHAPE, title, CIRCLE_NOW,
                         _steps(facts, 2, pick_text=picked, draw_text=CIRCLE_DRAW), "")
    if facts.ghost:
        return GuidePlan(PHASE_GHOST, title,
                         f"现在：淡蓝色是旧草稿的预览 — {COMMIT_KEY} 采纳进编辑层再修，"
                         "Esc 取消，Shift+A 换下一个",
                         _steps(facts, 2, pick_text=f"已选中：{editing}" if editing else "",
                                draw_text=f"旧草稿预览：{COMMIT_KEY} 采纳 / Esc 取消"),
                         "commit")
    if facts.warning:
        key, action = _WARNING_KEYS.get(
            facts.warning_action,
            _SCOPE_KEYS.get(facts.warning_scope, (COMMIT_KEY, "commit")))
        return GuidePlan(PHASE_WARNING, title,
                         f"现在：面积和这类零件差得多 — 确认没画错就再按一次 {key}，"
                         f"否则 Esc 回去改",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text=f"面积提示：再按一次 {key} 仍然提交 / Esc 回去改"),
                         action)
    if facts.all_asked:
        # Task U16: 改全部 asks before it reaches the confirmed frames behind.
        return GuidePlan(PHASE_SCOPE, title,
                         "现在：提示条写着改全部会影响哪些帧（已确认的会重新检查）— "
                         "确定就再按 Ctrl+Q，不改按 Esc",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text=ALL_COMMIT),
                         "commit_all")
    if facts.scope and facts.layering_asked:
        return GuidePlan(PHASE_SCOPE, title,
                         "现在：提示条写着层级改动会影响哪些帧（已确认的会重新检查）— "
                         "确定就再按 L，不改层级按 Esc",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text=LAYERING_COMMIT),
                         "")
    if facts.scope:
        said = facts.scope_text or facts.scope
        # Task U6: the commit writes the shape; the layering is L's and asks first.
        tail = ("" if facts.gallery else
                "；Alt+Q 只改这一帧，Ctrl+Q 改全部")
        return GuidePlan(PHASE_SCOPE, title,
                         f"现在：看画布下方的提示条 — {COMMIT_KEY} 只改形状；要改层级（{said}）按 L"
                         f"{tail}",
                         _steps(facts, 3, pick_text=f"已选中：{editing}" if editing else "",
                                commit_text=LAYERING_COMMIT),
                         "commit")
    if facts.roi_editing:
        # Opening a stored ROI to check it is not a proposal to put off: Esc
        # keeps what is stored (U2b round 2 -- D13/scan already has one).
        now = (f"现在：拖边 / 角调整已存的机箱范围（框里按住整体挪），按 {COMMIT_KEY} 保存；"
               "不改按 Esc（保持原来的）" if facts.roi_stored else
               f"现在：调好紫红框（拖白色小方块），按 {COMMIT_KEY} 保存；暂时不想管就按 Esc")
        return GuidePlan(PHASE_ROI, title, now, _steps(facts, 0), "commit")
    if facts.bench:
        return GuidePlan(PHASE_BENCH, title,
                         f"现在：在台面上拖一个框框住「{facts.bench}」— 松手就存好",
                         _steps(facts, 2, pick_text=f"已选中：{facts.bench}",
                                draw_text=DRAW_BENCH, commit_text="台面框拖完自动保存"),
                         "tool_bench_box")
    if editing:
        picked = f"已选中：{editing}"
        split = facts.editing_kind == api.KIND_SPLIT_KEYFRAME
        if facts.layer_dirty and split:
            # a ✂ row is a new version from this frame on -- what Enter writes
            # there since task U16 (Ctrl+K before it: U2b round 2, I1)
            return GuidePlan(PHASE_DRAW_PIXELS, title,
                             f"现在：「{editing}」的形状从这一帧起变了 — 画好了按 {COMMIT_KEY}"
                             f"（从这帧起新版本）提交，走过的帧不变",
                             _steps(facts, 3, pick_text=picked, commit_text=COMMIT_SPLIT),
                             "commit")
        # re-tracing a confirmed shape puts its confirmed frames in conflict:
        # said before the commit, with the one key that answers them all (U7).
        # The frames are Enter's reach (U16), so a ✂ row counts its own.
        retrace = ("" if not facts.retrace_frozen else
                   "；" + RETRACE_NOTE.format(n=len(facts.retrace_frozen)))
        if facts.layer_dirty:
            return GuidePlan(PHASE_DRAW_PIXELS, title,
                             f"现在：「{editing}」的形状对了就按 {COMMIT_KEY} 提交；边缘不对用 "
                             f"W 补 / E 擦，漏掉的大块用 D 连点补，不想要就 Esc{retrace}",
                             _steps(facts, 3, pick_text=picked), "commit")
        if facts.layer_pixels:
            # A stored shape, loaded and untouched: Enter has nothing to write
            # and says so; the next thing is to change it or leave it.
            nxt = (f"画好了按 {COMMIT_KEY}（从这帧起新版本）" if split
                   else f"改完 {COMMIT_KEY}")
            # An open ✂ row loads the *old* version: it needs drawing, so no
            # button is the next one to press -- not Esc (round 3, item 6).
            open_split = split and not any(
                item.done for item in facts.items
                if editing in (item.label, item.instance)
                and item.kind == api.KIND_SPLIT_KEYFRAME)
            return GuidePlan(PHASE_LOADED, title,
                             f"现在：这是已存的形状：要改就画，{nxt}；不改按 Esc"
                             f"（或直接点下一条）{retrace}",
                             _steps(facts, 2, pick_text=picked,
                                    draw_text=f"这是已存的形状：要改就画（{nxt}），"
                                              f"不改按 Esc"),
                             "" if open_split else "clear_edit")
        if facts.sam_ready:
            now = (f"现在：按 S，在「{editing}」上点一下（或按 X 拖一个框），"
                   f"SAM 会给出形状")
            action = "tool_sam_point"
        elif facts.sam_failed:
            now = f"现在：{DRAW_NO_SAM} 画出「{editing}」（F5 重试加载 SAM）"
            action = "tool_circle"
        else:
            now = f"现在：SAM 还没准备好 — 按 W 用画笔涂出「{editing}」"
            action = "tool_brush"
        return GuidePlan(PHASE_DRAW_EMPTY, title, now,
                         _steps(facts, 2, pick_text=picked), action)
    work = _open_work(facts)
    if work:
        first = _label(work[0])
        return GuidePlan(PHASE_PICK, title,
                         f"现在：在右边任务卡上单击「{first}」开始画（还剩 {len(work)} 个）",
                         _steps(facts, 1, pick_text=f"{PICK}：{first}"), "")
    nothing = not any(item.kind in _WORK_KINDS for item in facts.items)
    if facts.blockers:
        # The rows are done, and Space would still say no: the pane under the
        # card is the list, and no button is the next one (U2d).  "单击一条"
        # only while a click on one goes somewhere; otherwise the line's own
        # instruction (U2e).
        now = (f"现在：{facts.blocker_hint}" if facts.blocker_hint else
               f"现在：任务卡做完了，但下面还有 {facts.blockers} 个问题挡住 "
               f"Space：单击一条去处理")
        return GuidePlan(PHASE_BLOCKED, title, now, _steps(facts, 4, nothing=nothing), "")
    if facts.frame_confirmed:
        # the components gallery walks forwards (task U10)
        move = "按 PgUp 去下一张继续" if facts.gallery else "按 PgDn 去上一帧继续"
        return GuidePlan(PHASE_CONFIRMED, title,
                         f"这一{'张' if facts.gallery else '帧'}已经确认 ✓ — {move}；"
                         f"要改哪个零件就在任务卡上单击它",
                         _steps(facts, -1, nothing=nothing), "")
    return GuidePlan(PHASE_CONFIRM, title,
                     ("现在：任务卡都做完了 — 按 Space 确认这张，自动到下一张"
                      if facts.gallery else
                      "现在：任务卡都做完了 — 按 Space 确认整帧，自动退到上一帧"),
                     _steps(facts, 4, nothing=nothing), "confirm")
