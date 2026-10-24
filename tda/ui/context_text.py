"""What the window says about a 框外零件 (task U18, spec 3.2c), in one place.

Qt-free and import-light: the session's refusals, the panels' captions and the
dialog read their words from here, so the sentence a key says and the one its
button's tooltip says cannot drift apart.
"""
from __future__ import annotations

__all__ = [
    "ABSENT_BACK", "ABSENT_DONE", "ABSENT_RANGE_DONE", "ABSENT_RANGE_OFF", "BLOCK_TITLE",
    "BUTTON", "BUTTON_TIP", "CREATED", "DIALOG_TITLE", "INTRO", "LABEL_TIP", "MENU_TEXT",
    "NO_ADOPT", "NO_PROMPT_BOX", "NO_SHAPE_HERE", "NO_STATE", "NO_VERB", "NOT_IN_TABLE",
    "OTHER_ZH", "RANGE_TITLE", "TAG_ABSENT", "TAG_ABSENT_TIP",
]

#: The entry of the canvas right-click menu and of the 「添加零件」 panel.
MENU_TEXT = "框外零件…"
BUTTON = "框外零件… / Outside the machine"
BUTTON_TIP = ("画面里、机器外的东西（旁边放着的另一个零件、无关的物件）：起个名字单独画，"
              "不进步骤表、任务卡、状态和关系，不挡 Space")
#: The dialog.
DIALOG_TITLE = "框外零件 / Object outside the machine"
INTRO = ("机器外面、画面里看得到的东西：选类别（都不是就选「其他」并写名字），确定后直接画它，"
         "Q 提交（往后的帧带着它）。它不进步骤表、任务卡、状态和关系，缺了也不挡 Space；"
         "哪一帧里没有它就选中按 ` 键（不在这张里）。")
OTHER_ZH = "其他（写名字）"
LABEL_TIP = "名字：「其他」必填，别的类别可以不写 / name (required for other)"
#: After it was made.
CREATED = "{label}：已新建（框外，单独画）——现在画它，Q 提交；Ctrl+Z 撤销新建"
#: The instance tables' block, and their 「不在」 tag.
BLOCK_TITLE = "── 框外零件 / outside the machine ──"
TAG_ABSENT = "不在"
TAG_ABSENT_TIP = ("这一帧里没有它（框外零件，` 键标的）；在的话选中再按 ` 键撤回，Ctrl+Z 撤销 / "
                  "not in this frame")
NOT_IN_TABLE = "{instance} 是框外零件：这里没有它（右键画布「框外零件…」新建）"
#: `````` / ``Ctrl+``` on one (``I`` / ``Ctrl+I`` before task U22).
ABSENT_DONE = "{label}：不在这张里（再按 ` 键撤回，Ctrl+Z 撤销）"
ABSENT_BACK = "{label}：撤回「不在」，这张里有它（Ctrl+Z 撤销）"
NO_SHAPE_HERE = ("{label} 在这一帧没有形状，本来就不在；在的话直接画它、Q 提交")
RANGE_TITLE = "一段帧里没有它 / not in these frames"
ABSENT_RANGE_DONE = "{label} {where}都标为不在（Ctrl+Z 一次撤销）"
ABSENT_RANGE_OFF = "{label} {where}撤回「不在」（Ctrl+Z 一次撤销）"
#: What the machine's gestures say on one.
NO_STATE = "{instance} 是框外零件：不在机器里，没有状态可改 / no state outside the machine"
NO_VERB = "{instance} 是框外零件：不在步骤表里，没有动作可改 / not in the step table"
NO_ADOPT = "框外零件不采纳 Label Studio 草稿：直接画它 / draw it, no draft"
NO_PROMPT_BOX = "框外零件不用提示框（那是给机器里的零件猜的）：直接点它 / click it, no box"
