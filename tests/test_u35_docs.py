"""Task U35: the docs say what 「这一帧的说明」 is -- the data model row, the Review
paragraph, v1.34 on top of §14, sentences in the quickstart's Review section.

Round 2 (N8): the guide gets one sentence at the end of §6, paid for by
tightening lines elsewhere -- its prose stays within 7700 characters and the
key table is untouched. Round 3: 「不写注册表和 C 盘」 stays (the user's
promise); the sentence is the shorter one.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
GUIDE = ROOT / "docs" / "annotation_guide.md"


def _body() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_data_model_has_the_frame_note_row():
    table = _body().split("### 3.1 实体", 1)[1].split("### 3.2", 1)[0]
    [row] = [line for line in table.splitlines() if line.startswith("| `FrameNote` |")]
    for needed in ("(视角, 帧)", "「这一帧的说明」", "可写零件键", "每帧至多一条",
                   "`frame_note(desktop, view, step, text, author, updated_at)`", "库结构 v6",
                   "不是编译器的输入", "`load-index` 随照片移动", "`import-logs --force` 不碰",
                   "v1.34，U35"):
        assert needed in row, needed
    assert table.index("| `InstanceFrameFlags` |") < table.index("| `FrameNote` |")


def test_review_says_where_the_box_is_and_what_it_does():
    review = _body().split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    head = "**这一帧的说明**（v1.34，U35）"
    para = review.split(head, 1)[1].split("\n", 1)[0]
    for needed in ("复查面板队列下方「这一帧的说明」框", "可以写零件键", "「保存」写入、「清除」删掉",
                   "「未保存」", "**改过没保存就离开这一帧也会保存**", "`PgDn` / `PgUp`",
                   "`Enter`（确认后跳帧）", "离开复查模式", "关窗口",
                   "「第 k 帧的说明改过没保存：离开时已自动保存（Ctrl+Z 撤销）」",
                   "`set_frame_note` op", "逐字节", "`Ctrl+Z` / `Ctrl+Y`", "「撤销：第 k 帧的说明」",
                   "全空 = 没有说明", "存同样的文字什么都不写", "`Q` / `W` / `S` 是字母",
                   "也不提示「快捷键没生效」", "`TYPING_PROPERTY`", "**没有新快捷键**",
                   "「说明：<第一行>…」", "复查模式不显示", "缩略图旁带 ✎", "「After (k)」",
                   "不是编译器的输入", "`python -m tda.cli notes --desktop N [--view V] "
                   "[--format md] [--out FILE]`", "UTF-8",
                   "「这一帧的说明 / frame notes: n (scan a, oak1 b)」", "§3.1 `FrameNote`",
                   # round 2
                   "**不承诺 `Ctrl+Z`**",
                   "「…离开时已自动保存（换了视角或机器，撤销记录已清空；要改回在复查里改）」",
                   # round 3
                   "没有图像的帧上画布不显示，就交给时间轴，`PgDn` 照样能走",
                   "库旁可能留下空的 `-wal` / `-shm`，库文件不变",
                   "从事件循环里说", "走不动的帧上确认——写「已保存」",
                   "关窗口时写不进去，文字连同错误进日志，窗口照关",
                   "「快捷键没生效：焦点在说明框里，按 Tab 回到画布」", "字母数字键不提示",
                   "框有焦点时边框高亮", "`Tab` / `Shift+Tab` 回到画布", "**只读打开**",
                   "从不升库结构", "「没有说明（库结构 v5，应用升级后第一次打开会建表）」",
                   "`--out FILE` 或先 `chcp 65001`"):
        assert needed in para, needed
    assert review.index("**可见性审查**（v1.24") < review.index(head)


def test_v1_34_is_on_top_of_14():
    changes = _body().split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # U33's v1.32 and U34's v1.33 land at the same time; the controller orders them
    # U33 / U34 / U36 landed v1.32 / v1.33 / v1.35 the same day; the controller ordered them
    [entry] = [e for e in entries[:4] if e.startswith("- **v1.34")]
    assert entry.startswith("- **v1.34（2026-10-26）**：**这一帧的说明**（U35，§3.1、§4.4）")
    for needed in ("在 review 的时候我可以加入对这一帧说明的 notes",
                   "`frame_note(desktop, view, step, text, author, updated_at)`",
                   "**库结构 v6**", "拒绝打开 v6 文件", "`tda/core/db_notes.py`",
                   "`tda/core/frame_notes.py`", "`set_frame_note` op", "`TYPING_PROPERTY`",
                   "没有新键", "`cli notes --desktop N [--view V] [--format md] [--out FILE]`",
                   "`cli status` 每台计数", "`_KEYED_INPUTS`", "保留主照片的说明",
                   "`[load-index：以下是原第 k 步的说明（作者，时间）]`",
                   "n frame note(s) landed on a step that has one", "整台拒绝",
                   "`import-logs --force` 不碰说明", "`D<nn>/notes.json`", "不含作者",
                   "`frame_notes` 计数",
                   # round 2
                   "评审第二轮", "状态栏不再承诺 `Ctrl+Z`", "`cli notes` 只读打开",
                   "`chcp 65001`", "主照片的说明在前，不管旧步号大小", "手册 §6 一句"):
        assert needed in entry, needed
    assert changes.index("- **v1.34") < changes.index("- **v1.31")


def test_the_quickstart_says_it_in_two_sentences_of_the_review_section():
    quick = QUICK.read_text(encoding="utf-8")
    review = quick.split("## 4. 走完之后：Review", 1)[1].split("### 冲突从哪来", 1)[0]
    item = review.split("\n4. ", 1)[1].replace("\n   ", "")
    for needed in ("「这一帧的说明」", "「保存」", "改了没保存就翻帧也会自动存", "`Ctrl+Z` 能撤",
                   "框里打字时快捷键不生效", "按 `Tab` 回到画面", "时间轴上带 ✎", "「说明：…」",
                   "`python -m tda.cli notes --desktop 13`",
                   # round 2, N5: PowerShell
                   "在 PowerShell 里", "`--out notes.md`", "`chcp 65001`", "否则中文会乱码"):
        assert needed in item, needed
    assert item.count("。") == 3


def test_the_guide_has_one_sentence_in_6_and_its_prose_stays_within_its_limit():
    body = GUIDE.read_text(encoding="utf-8")
    section = body.split("## 6. 复查队列", 1)[1].split("\n---", 1)[0]
    last = [line for line in section.splitlines() if line.strip()][-1]
    assert last == "队列下「这一帧的说明」：翻帧自动存，`cli notes` 读。"         # round 3
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    # a promise the user cares about: it stays (round 3, R2-N2)
    assert "不写注册表和 C 盘" in body
    assert "面向每天标注 6 小时的人" not in body and "复查不扫全视图" not in body
