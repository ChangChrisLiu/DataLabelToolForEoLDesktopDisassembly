"""The one keyboard map of the annotator (spec 4.6), and what is generated from it.

:data:`ACTIONS` is the single source of truth for three things that used to
drift apart: what a key does, what the in-app cheat sheet (``?`` / ``F12``)
shows, and the ``快捷键`` table of ``docs/annotation_guide.md``.  Adding a
binding here adds it everywhere; there is nowhere else to add one.

**The 1-4 vs 1-7 collision.**  The plan gave the four views the number keys and
the seven visibility values the same ones.  The ruling implemented here (and in
the brief) is: the views are ``F1``-``F4``, and the visibility values are
``Ctrl+1``-``Ctrl+7`` (task U22, below) -- and since task U24 they are Review
mode's only (the 可见性审查 queue), bare ``1``-``7`` with ``Ctrl`` as an alias.

**The visibility review (task U24).**  Annotate mode has no label keys any
more: the compiler derives every label, and the grave key still says 被线挡住
while drawing.  Review's fifth queue lists the labels to audit; ``Space``
confirms one as computed (``Shift+Space`` the whole run), a digit sets one by
hand, `` ` `` / ``Shift+``` is 被线挡住 完全 / 部分 and ``Z`` / ``Shift+Z`` is
角度看不到 -- the camera cannot see the part, nothing covers it.  ``Ctrl+Z`` /
``Ctrl+Y`` work in Review for them.

**The left hand (task U22).**  The annotator's right hand is on the mouse or
the pen all day, so the keys pressed hundreds of times a day sit under the left
one: ``Q`` commits (``Alt+Q`` this frame only, ``Ctrl+Q`` 改全部), the grave key
(the backtick left of ``1``, Qt's ``Key_QuoteLeft``) is 看不见 / 被线挡住 (Ctrl
plus it over a span), ``W`` the brush, ``D`` the polygon, ``R`` the circle, ``V`` the bench box, ``G`` the
difference map and ``1`` / ``2`` the brush size.  ``Space``, ``E`` and ``Tab``
were already there.  The old keys that were not taken by something else stay
as aliases -- ``Enter``, ``Ctrl+K``, ``B``, ``[`` and ``]`` -- and ``keys[0]``
is always the left-hand one: it is what the palette, the status line and the
guide tell the annotator to press.  What the new keys displaced moved under a
modifier: visibility ``Ctrl+1``-``7``, its cycle ``Ctrl+V``, the outline
``Shift+Q``, the pixel grid ``Shift+G``.  Review mode keeps ``R`` for rework:
the modes are separate, as they were when ``R`` was the bench box.

**Modes.**  Every action declares the modes it is live in, so the step table
keeps ``Enter`` (and every letter) for its cell editors and the review panel
keeps ``R`` for "rework" while annotate mode uses it for the circle.  A key
with no action in the current mode is never swallowed -- it goes on to the
focused widget.

**Text fields.**  :func:`blocks_shortcuts` is consulted before any lookup, so a
``W`` typed into a note or a combo box is a letter, not a brush, and a ``1`` in
the brush-size box is a digit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QKeySequenceEdit,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

from tda.core.model import Visibility
from tda.ui import session_api as api

__all__ = [
    "ACTIONS",
    "GROUPS",
    "MENU_ENTRIES",
    "MODES",
    "MODE_ANNOTATE",
    "MODE_REVIEW",
    "MODE_STEPS",
    "PALETTE_ACTIONS",
    "PALETTE_OTHER",
    "PALETTE_REVIEW",
    "PALETTE_STATE",
    "PALETTE_TOOLS",
    "Action",
    "MenuEntry",
    "action_for",
    "action_named",
    "blocks_shortcuts",
    "cheat_sheet_html",
    "combos_of",
    "key_caption",
    "mode_cheat_html",
    "mode_cheat_rows",
    "navigates_a_list",
    "short_parts",
    "shortcut_markdown",
    "tool_key",
]

MODE_STEPS = "steps"
MODE_ANNOTATE = "annotate"
MODE_REVIEW = "review"
MODES: tuple[str, ...] = (MODE_STEPS, MODE_ANNOTATE, MODE_REVIEW)

#: Modes an action is live in.  ``modes[0]`` is the mode the action belongs to
#: first, which is what the cheat sheet and the tests use.
_ANN = (MODE_ANNOTATE,)
_REV = (MODE_REVIEW,)
_ANN_REV = (MODE_ANNOTATE, MODE_REVIEW)
_ALL = (MODE_ANNOTATE, MODE_REVIEW, MODE_STEPS)

#: Section keys and their headings in the guide and the cheat sheet.
GROUPS: tuple[tuple[str, str], ...] = (
    ("view", "视图与模式"),
    ("nav", "帧导航"),
    ("tool", "工具"),
    ("edit", "编辑与提交"),
    ("vis_review", "可见性审查（Review 模式）"),
    ("display", "显示"),
    ("session", "会话"),
)

#: The two halves of 被线挡住 / 角度看不到 (:mod:`tda.core.occluder`'s FULL / PARTIAL).
_FULL = Visibility.OCCLUDED_FULL.value
_PARTIAL = Visibility.OCCLUDED_PARTIAL.value


@dataclass(frozen=True)
class Action:
    """One binding: the keys, the window slot they call, and how to name it.

    Attributes:
        name: unique identifier, used by the tests and the cheat sheet anchors.
        keys: portable ``QKeySequence`` strings; all of them trigger the action
            (``Enter``/``Return`` are two spellings of one key).
        slot: method on the main window, called with ``args``.
        args: positional arguments handed to ``slot``.
        group: one of :data:`GROUPS`.
        label / label_zh: English and Simplified Chinese descriptions.
        modes: modes the binding is live in.
        hold: the action is held rather than pressed -- the slot is called with
            ``True`` on press and ``False`` on release (``Tab``, flash compare).
        repeat: holding the key fires the action again and again.  It is the
            difference between "bigger brush" (hold it until the brush is the
            size you want) and "commit" (once, however long the finger stays
            down).  Auto-repeat used to be swallowed for *every* binding, which
            made ``]``, ``.`` and ``PgDn`` do nothing at all when held.
        short: the caption of the action's button on the tool palette (task
            U2b), ``名字（注释）`` -- the part before ``（`` is the name, the rest
            a small note under it.  Empty for an action with no button; the
            button's key is always read from :attr:`keys`, so the palette can
            never show a key the keyboard does not have.
    """

    name: str
    keys: tuple[str, ...]
    slot: str
    label: str
    label_zh: str
    group: str = "edit"
    args: tuple = field(default_factory=tuple)
    modes: tuple[str, ...] = _ANN
    hold: bool = False
    repeat: bool = False
    short: str = ""


def _view(view: str, key: str, zh: str) -> Action:
    return Action(f"view_{view}", (key,), "act_set_view", f"View: {view}", zh,
                  "view", (view,), _ANN_REV)


def _visibility(index: int) -> Action:
    # Ctrl+digit since task U22 (the bare 1 and 2 are the brush size there).
    # Task U24: the labels are not set while annotating any more -- only in
    # Review's 可见性审查, where nothing else wants the digits: the bare digit
    # is the key, Ctrl+digit its alias.
    value = api.VISIBILITY_VALUES[index]
    return Action(f"visibility_{index + 1}", (f"{index + 1}", f"Ctrl+{index + 1}"),
                  "act_set_visibility", f"Visibility review: set {value} by hand",
                  f"可见性审查：手动设为 {value}（这一帧，记为审过）", "vis_review",
                  (value,), _REV)


def _tool(name: str, keys, label: str, zh: str, short: str) -> Action:
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    return Action(f"tool_{name}", keys, "act_tool", label, zh, "tool", (name,), _ANN,
                  short=short)


ACTIONS: tuple[Action, ...] = (
    # -- views and modes ----------------------------------------------------
    _view("scan", "F1", "切换到扫描仪视图"),
    _view("oak1", "F2", "切换到 OAK 正面视图"),
    _view("oak2", "F3", "切换到 OAK 侧面视图"),
    _view("rs", "F4", "切换到 RealSense 视图"),
    # Task U10: the components gallery (零件合照), one view per camera.
    _view("cmp_oak1", "F6", "切换到零件合照 OAK1（拆下来的零件摆在桌上拍的照片）"),
    _view("cmp_oak2", "F7", "切换到零件合照 OAK2"),
    _view("cmp_rs", "F8", "切换到零件合照 RealSense"),
    # -- frame navigation ---------------------------------------------------
    Action("step_back", ("PgDown",), "act_step", "Previous step (k-1)",
           "上一步 k-1（倒序标注的“前进”）", "nav", (-1,), _ANN_REV, repeat=True),
    Action("step_forward", ("PgUp",), "act_step", "Next step (k+1)",
           "下一步 k+1", "nav", (+1,), _ANN_REV, repeat=True),
    Action("step_first", ("Home",), "act_step_edge", "First step",
           "跳到第一步", "nav", ("first",), _ANN_REV),
    Action("step_last", ("End",), "act_step_edge", "Last step",
           "跳到最后一步", "nav", ("last",), _ANN_REV),
    Action("flash_compare", ("Tab",), "act_flash_compare",
           "Hold: flash the frame the task card is written against",
           "按住：闪回任务卡对照的那一帧（你来的方向）", "nav", (True,), _ANN, hold=True,
           short="对比上一帧"),
    Action("flash_other", ("Shift+Tab",), "act_flash_other",
           "Hold: flash the frame on the other side",
           "按住：闪回另一侧的那一帧", "nav", (True,), _ANN, hold=True),
    # -- tools --------------------------------------------------------------
    # Task U22: W under the left hand; B, its key before, stays an alias.
    _tool("brush", ("W", "B"), "Brush", "画笔（加像素）", "画笔"),
    _tool("eraser", "E", "Eraser", "橡皮擦（减像素）", "橡皮擦"),
    # Task U5a: two ways to add a whole region at once.  Task U22 moved them
    # from P and Y under the left hand: D for the polygon (the difference map
    # went to G) and R for the round one (the bench box went to V; Review
    # mode's R, rework, is another mode).
    _tool("polygon", "D", "Polygon fill",
          "多边形（连点围一块：Q / 双击 / 点回起点填上）", "多边形"),
    _tool("circle", "R", "Circle fill",
          "圆形（圆心按下拖到边缘，松手填上；单击 = 笔刷大小的圆）", "圆形"),
    Action("polygon_backspace", ("Backspace",), "act_polygon_backspace",
           "Polygon: remove the last vertex", "多边形：删掉最后一个点", "tool", (), _ANN),
    _tool("sam_point", "S", "SAM point prompt", "SAM 点选（在零件上点一下）", "SAM 点选"),
    _tool("sam_box", "X", "SAM box prompt", "SAM 框选（拖一个框）", "SAM 框选"),
    _tool("occluder", "O", "Occluder brush", "遮挡层画笔（手/工具）", "遮挡（手/工具）"),
    # Not the ROI: the trial annotator pressed ``R`` expecting the chassis
    # rectangle (task U2b).  The label says what the box is *for*.  ``V``
    # since task U22 (R is the circle now).
    _tool("bench_box", "V", "Bench box", "台面框（已拆下、放在台面上的零件）",
          "台面框（已拆下、放在台面上的零件）"),
    # Task U22: 1 / 2 under the left hand; the brackets stay aliases.
    Action("radius_down", ("1", "["), "act_radius", "Smaller brush",
           "笔刷变小", "tool", (-1,), _ANN, repeat=True),
    Action("radius_up", ("2", "]"), "act_radius", "Bigger brush",
           "笔刷变大", "tool", (+1,), _ANN, repeat=True),
    Action("cycle_candidate", ("C",), "act_cycle_candidate", "Next SAM candidate",
           "切换 SAM 候选掩码", "tool", (), _ANN, short="换候选"),
    Action("cycle_prompt_box", ("Shift+C",), "act_cycle_prompt_box",
           "Next box prompt (the detector's other screws, then the difference map's)",
           "切换 SAM 提示框（检测器找到的其它螺丝在前，差异图的在后；第 1 个永远是默认那块）",
           "tool", (), _ANN),
    Action("adopt_draft", ("Shift+A",), "act_adopt_draft", "Adopt a Label Studio draft",
           "采纳 Label Studio 旧草稿（再按一次换下一个）", "tool", (), _ANN),
    Action("fill_holes", ("Shift+F",), "act_fill_holes", "Fill holes",
           "填补掩码内部空洞", "tool", (), _ANN),
    Action("despeckle", ("Shift+D",), "act_despeckle", "Remove specks",
           "删除小碎块（默认 <16 像素）", "tool", (), _ANN),
    # -- editing ------------------------------------------------------------
    # The commit keys answer a half-drawn polygon / circle first (task U5a).
    # Task U16: the commit is "from this frame on" -- the frames the walk has
    # passed keep their shape -- and Ctrl+K, which used to be that key, is now
    # its alias.  Task U22: Q under the left hand; Enter (and Ctrl+K) stay
    # aliases, and Alt / Ctrl say the other two scopes on either key.
    Action("commit", ("Q", "Return", "Enter", "Ctrl+K"), "act_commit",
           "Commit edit: from this frame on (the frames already walked keep theirs)",
           "提交编辑：从这帧起生效，走过的帧不变（这帧就是这版的起点时重画这一版；"
           "有层级建议时只改形状，层级用 L；多边形没画完时先填上它）", "edit", (),
           _ANN, short="提交"),
    Action("commit_override", ("Alt+Q", "Alt+Return", "Alt+Enter"), "act_commit_override",
           "Commit for this frame only", "只对当前帧生效（帧覆盖）", "edit", (), _ANN,
           short="只改这一帧"),
    Action("commit_all", ("Ctrl+Q", "Ctrl+Return", "Ctrl+Enter"), "act_commit_all",
           "Commit everywhere the version applies, walked frames included "
           "(asks first when it reaches confirmed ones)",
           "改全部：这一版形状用到的每一帧都改，走过的、已确认的也改（会碰到已确认帧时先说"
           "哪些帧，再按一次确定，Esc 取消）", "edit", (), _ANN, short="改全部"),
    # Task U6: the layering answer is not Enter's any more.  ``L`` for 层 /
    # layer -- free in every mode (no plain L, no Shift/Ctrl+L anywhere).
    # Task U11 round 2: the cut copy on this frame is its shape after all (a
    # state change that does not move the outline).  Ctrl+Shift+K is free.
    Action("confirm_cut_copy", ("Ctrl+Shift+K",), "act_confirm_cut_copy",
           "This frame's shape is the cut copy on it (no pixels; the split row is done)",
           "这帧就是这个形状：切开时留下的形状副本就是这一帧的样子（不画像素，✂ 完成）",
           "edit", (), _ANN, short="形状没变"),
    Action("commit_layering", ("L",), "act_commit_layering",
           "Change the layering the scope bar suggests (asks, then a second L)",
           "改层级：回答画布下方的层级建议（先显示影响哪些帧，再按一次 L 确定；Q 只改形状）",
           "edit", (), _ANN),
    Action("clear_edit", ("Esc",), "act_clear_edit", "Discard the edit",
           "放弃当前编辑（多边形 / 圆形没画完时只取消它）", "edit", (), _ANN, short="放弃编辑"),
    Action("confirm", ("Space",), "act_confirm", "Confirm the frame",
           "确认当前帧并后退一帧", "edit", (), _ANN, short="确认整帧"),
    # Review mode: the canvas is read-only, so its two keys are the queue's.
    # Task U24: Space went to the 可见性审查 queue (vis_review_computed), which
    # still accepts the frame on the other tabs.
    Action("review_accept", ("Return", "Enter"), "act_confirm",
           "Accept the frame the queue points at", "接受队列选中的这一帧",
           "edit", (), (MODE_REVIEW,), short="接受这一帧"),
    Action("review_rework", ("R",), "act_rework_selected",
           "Rework: open it in Annotate mode", "返工：在标注模式下打开这一帧",
           "edit", (), (MODE_REVIEW,), short="返工"),
    Action("review_keep_old", ("K",), "act_resolve", "Conflict: keep the frozen shape",
           "冲突：保留已冻结的形状", "edit", (api.RESOLVE_KEEP_OLD,), (MODE_REVIEW,),
           short="保留旧的（冲突）"),
    Action("review_accept_new", ("N",), "act_resolve", "Conflict: take the edit",
           "冲突：接受新的编辑", "edit", (api.RESOLVE_ACCEPT_NEW,), (MODE_REVIEW,),
           short="采用新的（冲突）"),
    # Task U7: a deliberate re-trace of a confirmed shape puts every confirmed
    # frame it reaches in conflict, all with the same answer.
    Action("review_accept_all_new", ("Shift+N",), "act_accept_all_new",
           "Conflicts: take every new value (pick the parts, then confirm)",
           "接受全部新值…（冲突是你刚重画的形状造成的：按零件勾选，一次接受）",
           "edit", (), (MODE_REVIEW,), short="接受全部新值…"),
    # Task U24: the visibility review's verdicts are ops of the same history,
    # so Ctrl+Z / Ctrl+Y work in Review too.
    Action("undo", ("Ctrl+Z",), "act_undo", "Undo", "撤销", "edit", (), _ANN_REV,
           short="撤销"),
    Action("redo", ("Ctrl+Y",), "act_redo", "Redo", "重做", "edit", (), _ANN_REV,
           short="重做"),
    Action("toggle_hidden", ("H",), "act_toggle_hidden", "Hide/show instance",
           "隐藏或显示选中实例", "edit", (), _ANN),
    # Task U10: ``I`` for invisible / 看不见 -- free in every mode. Task U15:
    # on a sequence frame the part is there but under a cable (not labelled).
    # Task U18: on a 框外零件 (an object outside the machine) it is 「不在这张里」.
    # Task U22: the grave key ` (Qt's Key_QuoteLeft, left of 1) -- I was a
    # right-hand key; free in every mode, Ctrl+` too.
    Action("mark_absent", ("`",), "act_mark_absent",
           "Hidden: on the sequence, the selected part is under a cable (full -> partial "
           "-> off); in the components gallery, the card's part is not in this shot; "
           "an object outside the machine is not in this frame",
           "看不见：拆解序列里 = 选中零件被线挡住（完全→部分→撤回）；"
           "零件合照里 = 卡上这个零件不在这张里（再按撤回）；"
           "框外零件 = 不在这张里（再按撤回）", "edit", (), _ANN,
           short="看不见"),
    # Task U15: the same over a span, as Ctrl+T is T's (Ctrl+` since U22).
    Action("cable_range", ("Ctrl+`",), "act_cable_range",
           "Selected part under a cable from this frame to a given step (one op); "
           "an object outside the machine: not in those frames",
           "选中零件从这一帧到第 N 步都标被线挡住（完全 / 部分 / 撤回；一次撤销）；"
           "框外零件 = 这一段都标不在 / 撤回",
           "edit", (), _ANN),
    # -- 可见性审查 (task U24): Review mode only -------------------------------
    # While annotating nobody sets a visibility label any more (the compiler
    # derives it; ` still says 被线挡住 while drawing). At the end, Review's
    # fifth queue lists every confirmed (frame, part) whose label is not plain
    # visible, and these keys answer the selected row -- each answer is one op
    # (Ctrl+Z) and stamps the row reviewed.
    Action("vis_review_computed", ("Space",), "act_vis_review",
           "Visibility review: the label is right as it is (stamp this frame); "
           "on the other queues Space accepts the frame",
           "可见性审查：这一帧这个零件按计算的对，记为审过（其他队列里 = 接受这一帧）",
           "vis_review", ("computed",), _REV, short="确认可见性"),
    Action("vis_review_run", ("Shift+Space",), "act_vis_review_run",
           "Visibility review: the whole selected run is right as it is",
           "可见性审查：选中这一条的整段帧都按计算的对，一次记为审过", "vis_review", (),
           _REV, short="确认整段"),
    *[_visibility(i) for i in range(7)],
    Action("cycle_visibility", ("Ctrl+V",), "act_cycle_visibility",
           "Visibility review: the next value, by hand",
           "可见性审查：手动换成下一个取值（这一帧，记为审过）", "vis_review", (), _REV),
    Action("vis_review_cable", ("`",), "act_vis_review",
           "Visibility review: under a cable, completely",
           "可见性审查：被线挡住（完全）——零件在原位，被线盖住", "vis_review",
           ("cable", _FULL), _REV, short="被线挡住"),
    Action("vis_review_cable_partial", ("Shift+`",), "act_vis_review",
           "Visibility review: under a cable, partly",
           "可见性审查：被线挡住（部分）", "vis_review", ("cable", _PARTIAL), _REV),
    Action("vis_review_viewpoint", ("Z",), "act_vis_review",
           "Visibility review: not visible from this viewpoint (nothing covers it)",
           "可见性审查：角度看不到（完全）——这个相机从它的位置看不到，不是被挡住",
           "vis_review", ("viewpoint", _FULL), _REV, short="角度看不到"),
    Action("vis_review_viewpoint_partial", ("Shift+Z",), "act_vis_review",
           "Visibility review: partly not visible from this viewpoint",
           "可见性审查：角度看不到（部分）", "vis_review", ("viewpoint", _PARTIAL), _REV),
    # Task U8: the state the picture shows when the log's is different (a clip
    # shut again).  ``T`` for 状态 / sTate: free in every mode.
    Action("cycle_state", ("T",), "act_cycle_state",
           "Selected part's state at this frame: the next one (log correction)",
           "选中零件这一帧的状态换成下一个（如 打开→关上；下一帧起自动恢复；"
           "之后按任务卡 ✂ 条画形状、Q 提交）", "edit", (), _ANN, short="换状态"),
    # Task U9: the open cover / latch the card asks for is on the pile.  Next to
    # T because it is the other half of "the log is wrong about this part";
    # ``Shift+T`` is free in every mode.
    # Task U14: on a screw the log unscrews, the same key flips its captive flag.
    # Task U21: on a drive cage / its latch it also offers the other class.
    Action("correct_verb", ("Shift+T",), "act_correct_verb",
           "Selected cover/latch: taken out, not opened (the log's open -> remove; "
           "a removed one: back to open); a screw: captive or not; a drive cage/latch: "
           "the other one",
           "选中的盖板/卡扣其实拆下来了：日志里那一步的「打开」改成「拆下」"
           "（Removed 里的零件反过来：其实只是打开了）；螺丝：其实拧下来了 / 其实留在零件里；"
           "硬盘架 / 卡扣：其实是硬盘架 / 其实是卡扣",
           "edit", (), _ANN, short="其实拆下"),
    # Task U11: the same over a span, "from here to step N" -- D2/scan walked
    # 45 -> 37 with T at every frame.  Ctrl+T is free in every mode.
    Action("state_range", ("Ctrl+T",), "act_state_range",
           "Selected part's state from this frame to a given step (one op)",
           "选中零件从这一帧到第 N 步都设成同一状态（一次撤销；切开的版本跟着边界走）",
           "edit", (), _ANN, short="一段状态"),
    Action("zorder_up", ("Ctrl+Up",), "act_move_instance", "One layer up",
           "选中实例上移一层", "edit", (-1,), _ANN, repeat=True),
    Action("zorder_down", ("Ctrl+Down",), "act_move_instance", "One layer down",
           "选中实例下移一层", "edit", (+1,), _ANN, repeat=True),
    Action("edit_roi", ("Shift+R",), "act_edit_roi", "Edit the ROI",
           "机箱范围 ROI：编辑（已存的框直接拖边/角或整体移动，Esc 不改）",
           "edit", (), _ANN, short="机箱范围 ROI"),
    Action("split_pose", ("Ctrl+Shift+B",), "act_split_pose",
           "Split pose segment at this frame",
           "从这一帧起断开位姿段（本视角）", "edit", (), _ANN),
    # -- display ------------------------------------------------------------
    Action("toggle_overlays", ("A",), "act_toggle_overlays", "All overlays on/off",
           "开关全部标注图层", "display", (), _ANN_REV),
    Action("toggle_outline", ("Shift+Q",), "act_toggle_outline", "Outline / filled",
           "轮廓线与半透明填充切换", "display", (), _ANN_REV),
    Action("opacity_down", (",",), "act_opacity", "Less opaque",
           "图层更透明", "display", (-1,), _ANN_REV, repeat=True),
    Action("opacity_up", (".",), "act_opacity", "More opaque",
           "图层更不透明", "display", (+1,), _ANN_REV, repeat=True),
    Action("toggle_heat", ("G",), "act_toggle_heat", "Difference heat map",
           "开关帧间差异热力图", "display", (), _ANN_REV, short="差异图"),
    Action("fit_roi", ("F",), "act_fit_roi", "Fit the ROI",
           "缩放到 ROI", "display", (), _ANN_REV),
    Action("fit_image", ("Shift+0",), "act_fit_image", "Fit the whole frame",
           "缩放到整幅图像", "display", (), _ANN_REV),
    Action("toggle_grid", ("Shift+G",), "act_toggle_grid", "Pixel grid",
           "开关像素网格", "display", (), _ANN_REV),
    # -- session ------------------------------------------------------------
    Action("save", ("Ctrl+S",), "act_save", "Save", "保存（清除未保存标记）",
           "session", (), _ALL),
    Action("refresh_all", ("F5",), "act_refresh_all",
           "Recompile this frame; retry loading SAM when it failed",
           "重新编译当前帧（整视图请用命令行 cli check）；SAM 加载失败时重试加载", "session",
           (), _ALL),
    Action("cheat_sheet", ("?", "F12"), "act_cheat_sheet", "Shortcut cheat sheet",
           "快捷键速查表", "session", (), _ALL),
)


@dataclass(frozen=True)
class MenuEntry:
    """A gesture with no key: an entry of a right-click menu (task U18).

    Listed in the guide's table and the cheat sheet with the key bindings --
    the table is where the annotator looks up "how do I ..." -- but never in
    the key index: the right button is not a key.
    """

    name: str
    #: How it is reached, as the table's first column says it.
    where: str
    slot: str
    label: str
    label_zh: str
    group: str = "edit"
    modes: tuple[str, ...] = _ANN


#: The right-click entries, in table order.
MENU_ENTRIES: tuple[MenuEntry, ...] = (
    MenuEntry("context_object", "右键画布 → 框外零件…", "act_context_object",
              "Name and draw something outside the machine, apart from the annotation",
              "框外零件：机器外、画面里的东西（旁边的零件、无关物件），选类别或「其他」写名字，"
              "单独画；不进步骤表和任务卡、不挡 Space；某帧里没有它就选中按 ` 键（Ctrl+` 一段）"),
)


# --------------------------------------------------------------------------- #
# key lookup
# --------------------------------------------------------------------------- #
def _as_int(value) -> int:
    """``int`` of a Qt enum, a flag combination or a plain number.

    PySide6 hands a *combination* of ``KeyboardModifier`` flags back as an
    object ``int()`` refuses, while a single flag converts fine -- so both
    spellings are tried rather than assuming either.
    """
    try:
        return int(value)
    except TypeError:
        return int(getattr(value, "value", 0))


#: Modifiers that never take part in a binding: the numeric keypad reports its
#: own flag, and the group switch belongs to the keyboard layout.
_IGNORED_MODIFIERS = (_as_int(Qt.KeyboardModifier.KeypadModifier)
                      | _as_int(Qt.KeyboardModifier.GroupSwitchModifier))
_SHIFT = _as_int(Qt.KeyboardModifier.ShiftModifier)
#: Keys a US layout produces with Shift where the binding names the unshifted
#: character, so both spellings have to resolve to the same action. ``~`` is
#: ``Shift+``` (task U24: 被线挡住（部分） in the visibility review).
_SHIFTED = {_as_int(Qt.Key.Key_ParenRight): _as_int(Qt.Key.Key_0),
            _as_int(Qt.Key.Key_AsciiTilde): _as_int(Qt.Key.Key_QuoteLeft)}


def _combined(spec: str) -> Optional[int]:
    """The ``Qt.Key | modifiers`` integer of a portable key string."""
    seq = QKeySequence.fromString(spec)
    if seq.count() < 1:
        return None
    return int(seq[0].toCombined())


def combos_of(action: Action) -> tuple[int, ...]:
    """Every key combination that triggers ``action``."""
    return tuple(c for c in (_combined(spec) for spec in action.keys) if c is not None)


def _build_index() -> dict[str, dict[int, Action]]:
    index: dict[str, dict[int, Action]] = {mode: {} for mode in MODES}
    for action in ACTIONS:
        for combo in combos_of(action):
            for mode in action.modes:
                index[mode][combo] = action
    return index


_INDEX = _build_index()


def _candidates(key, modifiers) -> list[int]:
    """The combination the event means, plus the layout aliases of the brief."""
    code = _as_int(key)
    mods = _as_int(modifiers) & ~_IGNORED_MODIFIERS
    out = [code | mods]
    if mods & _SHIFT:
        alias = _SHIFTED.get(code)
        if alias is not None:
            out.append(alias | mods)
        # "?" is Shift+/ on a US layout but the binding is written "?"
        if 0x21 <= code <= 0x3F and not (0x30 <= code <= 0x39):
            out.append(code | (mods & ~_SHIFT))
    return out


#: The palette's groups of buttons (task U2b), top to bottom.  Names, not
#: copies: every caption, key and tooltip is read off :data:`ACTIONS`.
PALETTE_TOOLS: tuple[str, ...] = (
    "tool_brush", "tool_eraser", "tool_polygon", "tool_circle",
    "tool_sam_point", "tool_sam_box", "tool_occluder", "tool_bench_box", "edit_roi",
)
#: 「改全部」 took the slot of 「从这帧起新版本」 (task U16), whose key became
#: Enter's alias: the strip keeps its height (U13 fitted it to a 1080p window).
PALETTE_ACTIONS: tuple[str, ...] = (
    "commit", "commit_override", "commit_all", "confirm", "undo", "redo",
    "toggle_heat",
)
#: The 「状态」 group (task U13): the keys that say the log is wrong about a part
#: -- its state here, over a span, its verb -- and the two that settle a frame
#: without drawing (a cut copy that is right as it is, a part not in a gallery
#: shot). Annotate mode only, like the keys.
PALETTE_STATE: tuple[str, ...] = (
    "cycle_state", "state_range", "correct_verb", "confirm_cut_copy", "mark_absent",
)
#: The 「其他」 group at the foot of the strip (U13 follow-up): the action
#: buttons clicked least -- their keys are Tab (held), C and Esc -- so that the
#: 「状态」 group above them fits a maximized 1920x1080 window (a client area of
#: about 1920x1000) without scrolling.
PALETTE_OTHER: tuple[str, ...] = ("flash_compare", "cycle_candidate", "clear_edit")
#: What the strip keeps in Review mode (U2b round 1): the canvas is read-only,
#: so no tool and no brush -- only the difference map and the queue's keys.
#: Task U24: the 可见性审查 queue's four most used answers too (Shift 部分 has
#: keys only); undo / redo follow from their modes.
PALETTE_REVIEW: tuple[str, ...] = (
    "review_accept", "review_rework", "review_keep_old", "review_accept_new",
    "review_accept_all_new", "vis_review_computed", "vis_review_run",
    "vis_review_cable", "vis_review_viewpoint", "toggle_heat",
)


def action_named(name: str) -> Action:
    """The one action called ``name``; ``KeyError`` when there is none."""
    for action in ACTIONS:
        if action.name == name:
            return action
    raise KeyError(name)


def short_parts(action: Action) -> tuple[str, str]:
    """``(name, note)`` of a button caption: ``台面框（已拆下…）`` -> both halves."""
    text = action.short or action.label_zh
    head, sep, tail = text.partition("（")
    if not sep:
        return (text, "")
    return (head.strip(), tail.rstrip("）").strip())


def key_caption(action: Action) -> str:
    """The key a button shows: the primary one, and ``按住`` for a held key.

    The primary key is ``keys[0]`` -- the left-hand one since task U22 (``Q``
    on 提交, not the Enter alias).  Where that is spelled ``Return`` (Review's
    accept) the button says Enter, which is what the keyboard says.
    """
    keys = action.keys
    pick = keys[0] if keys else ""
    if "Return" in pick:
        pick = next((k for k in keys if "Enter" in k), pick)
    return f"按住 {pick}" if action.hold else pick


def _compact_keys(action: Action) -> str:
    return " / ".join(k for k in action.keys
                      if k not in ("Return", "Alt+Return", "Ctrl+Return"))


def mode_cheat_rows(mode: str) -> list[tuple[str, str]]:
    """``(keys, what)`` for every binding live in ``mode``, in table order.

    The guide panel's collapsible cheat sheet: only what works *here*, and the
    seven visibility keys as one row -- they are one idea with seven values
    (Review's 可见性审查 since task U24).
    """
    rows: list[tuple[str, str]] = []
    seen_visibility = False
    for _group, _title, actions in _by_group():
        for action in actions:
            if mode not in action.modes:
                continue
            if action.name.startswith("visibility_"):
                if not seen_visibility:
                    rows.append(("1–7", "可见性审查：手动设标签（这一帧，记为审过）"))
                    seen_visibility = True
                continue
            rows.append((_compact_keys(action), action.label_zh + _how_text(action)))
    return rows


def mode_cheat_html(mode: str) -> str:
    """:func:`mode_cheat_rows` as a compact two-column HTML table."""
    cells = [f"<td><b>{keys}</b></td><td>{what}</td>" for keys, what in mode_cheat_rows(mode)]
    half = (len(cells) + 1) // 2
    rows = []
    for index in range(half):
        right = cells[index + half] if index + half < len(cells) else "<td></td><td></td>"
        rows.append(f"<tr>{cells[index]}<td width='10'></td>{right}</tr>")
    return ("<table cellspacing='0' cellpadding='1' style='font-size:8pt'>"
            + "".join(rows) + "</table>")


def tool_key(name: str) -> str:
    """The key that arms tool ``name``, or ``""`` for one with no binding.

    The status bar shows the key next to the tool it arms, and it reads it from
    here rather than from a second table: :data:`ACTIONS` is the only key map,
    and a tool whose key moved must not keep the old one on screen.
    """
    for action in ACTIONS:
        if action.name == f"tool_{name}" and action.keys:
            return action.keys[0]
    return ""


def action_for(key, modifiers, mode: str = MODE_ANNOTATE) -> Optional[Action]:
    """The action bound to ``key`` + ``modifiers`` in ``mode``, or ``None``."""
    table = _INDEX.get(mode)
    if table is None:
        return None
    for combo in _candidates(key, modifiers):
        found = table.get(combo)
        if found is not None:
            return found
    return None


#: Widget classes that own every keystroke while they have the focus.
_TEXT_WIDGETS = (
    QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox,
    QKeySequenceEdit,
)


def blocks_shortcuts(widget: Optional[QWidget]) -> bool:
    """``True`` when ``widget`` is a text entry that must keep its keystrokes.

    An item view's open editor is a real ``QLineEdit`` child, so a cell being
    edited in the step table is covered by the same check.
    """
    return widget is not None and isinstance(widget, _TEXT_WIDGETS)


#: Keys a focused list or table must keep for moving its own selection.  Page
#: up/down are deliberately **not** here: they are frame navigation everywhere
#: in this window, and a list that scrolled instead would be a second, silent
#: meaning for the annotator's most used pair of keys.
_LIST_KEYS = frozenset({
    int(Qt.Key.Key_Up), int(Qt.Key.Key_Down),
    int(Qt.Key.Key_Left), int(Qt.Key.Key_Right),
})


def navigates_a_list(widget: Optional[QWidget], key) -> bool:
    """Is this an arrow key inside a list or table, i.e. its own navigation?"""
    if widget is None or _as_int(key) not in _LIST_KEYS:
        return False
    return isinstance(widget, QAbstractItemView)


# --------------------------------------------------------------------------- #
# generated documentation
# --------------------------------------------------------------------------- #
def _keys_text(action: Action) -> str:
    return " / ".join(action.keys)


def _code_span(text: str) -> str:
    """``text`` as one Markdown code span, even when it holds a backtick.

    The 看不见 key is the grave key itself (task U22): a span that contains a
    backtick is fenced with two and padded, as CommonMark asks.
    """
    if "`" not in text:
        return f"`{text}`"
    return f"`` {text} ``"


def _md_text(text: str) -> str:
    """Prose in a Markdown table cell: a backtick (the 看不见 key) stays a backtick.

    Two of them in one cell would otherwise open and close a code span.
    """
    return text.replace("`", "\\`")


#: What the generated tables add after a binding that answers auto-repeat.
REPEAT_NOTE = "（可长按）"
HOLD_NOTE = "（按住）"


def _how_text(action: Action) -> str:
    """"Hold it down" is part of what a key does, so the tables say so."""
    if action.repeat:
        return REPEAT_NOTE
    return HOLD_NOTE if action.hold else ""


def _by_group() -> list[tuple[str, str, list[Action]]]:
    return [
        (key, title, [a for a in ACTIONS if a.group == key])
        for key, title in GROUPS
    ]


def _menu_of(group: str) -> list[MenuEntry]:
    return [entry for entry in MENU_ENTRIES if entry.group == group]


def shortcut_markdown() -> str:
    """The ``快捷键`` table of ``docs/annotation_guide.md``, one row per binding.

    A right-click entry (:data:`MENU_ENTRIES`) is a row of its group too, after
    the keys.
    """
    lines = ["| 快捷键 | 作用 | 分组 |", "|---|---|---|"]
    for key, title, actions in _by_group():
        for action in actions:
            lines.append(f"| {_code_span(_keys_text(action))} | {_md_text(action.label_zh)}"
                         f"{_how_text(action)} | {title} |")
        for entry in _menu_of(key):
            lines.append(f"| {_code_span(entry.where)} | {_md_text(entry.label_zh)} "
                         f"| {title} |")
    return "\n".join(lines)


def cheat_sheet_html() -> str:
    """The same table as HTML, for the in-app cheat sheet (``?`` / ``F12``)."""
    parts = ["<h3>快捷键 / Shortcuts</h3>"]
    for key, title, actions in _by_group():
        parts.append(f"<h4>{title}</h4><table cellspacing='4'>")
        for action in actions:
            parts.append(
                f"<tr><td><b>{_keys_text(action)}</b></td>"
                f"<td>{action.label_zh}{_how_text(action)}</td>"
                f"<td><i>{action.label}</i></td></tr>"
            )
        for entry in _menu_of(key):
            parts.append(f"<tr><td><b>{entry.where}</b></td><td>{entry.label_zh}</td>"
                         f"<td><i>{entry.label}</i></td></tr>")
        parts.append("</table>")
    return "".join(parts)


#: Markers delimiting the generated block inside ``docs/annotation_guide.md``.
GUIDE_BEGIN = "<!-- shortcuts:begin (generated from tda/ui/app_actions.py) -->"
GUIDE_END = "<!-- shortcuts:end -->"


def write_guide_table(path: str) -> bool:
    """Replace the generated block of the annotation guide; ``True`` when it changed.

    Run as ``python -m tda.ui.app_actions`` after touching :data:`ACTIONS`, so
    the guide can never disagree with the keyboard (a test asserts they match).
    """
    from pathlib import Path

    target = Path(path)
    body = target.read_text(encoding="utf-8")
    head, _, rest = body.partition(GUIDE_BEGIN)
    _, _, tail = rest.partition(GUIDE_END)
    if not rest or not tail:
        raise ValueError(f"{target} has no {GUIDE_BEGIN} ... {GUIDE_END} block")
    rebuilt = f"{head}{GUIDE_BEGIN}\n\n{shortcut_markdown()}\n\n{GUIDE_END}{tail}"
    if rebuilt == body:
        return False
    target.write_text(rebuilt, encoding="utf-8")
    return True


def _main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - a tool
    import argparse
    from pathlib import Path

    default = Path(__file__).resolve().parents[2] / "docs" / "annotation_guide.md"
    ap = argparse.ArgumentParser(description="Regenerate the guide's shortcut table.")
    ap.add_argument("--guide", default=str(default))
    args = ap.parse_args(argv)
    changed = write_guide_table(args.guide)
    print(f"{'rewrote' if changed else 'already up to date:'} {args.guide}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(_main())
