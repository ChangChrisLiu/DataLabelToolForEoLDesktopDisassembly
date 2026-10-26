"""Task U22: the keys pressed hundreds of times a day are under the left hand.

The annotator's right hand is on the mouse or the pen all day.  The map
(user, 2026-10-23, revised twice the same day): Q commits (Alt+Q / Ctrl+Q the
other two scopes), the grave key ` is 看不见 / 被线挡住 (Ctrl+` over a span), W
the brush, D the polygon, R the circle, V the bench box, G the difference map,
1 / 2 the brush size.  What those keys did moves under a modifier: visibility
Ctrl+1..7, its cycle Ctrl+V, the outline Shift+Q, the pixel grid Shift+G.  The
old keys that nothing else took stay aliases: Enter (and Alt+/Ctrl+Enter),
Ctrl+K, B, [ and ].

Pinned here: the map itself; that the aliases still reach their slots and the
retired keys (I, Ctrl+I, P, Y, bare digits 3-7) reach nothing; that no two
actions share a combination in any mode; that a text field keeps every new
letter and digit; that the grave key arrives the way Qt reports it; that the
palette's captions did not grow; and that no sentence on screen still names a
key the keyboard moved.
"""
from __future__ import annotations

import ast
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFontMetrics, QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QSpinBox

from app_scene import StubSamQueue, close_window, make_paths, make_session
from test_app_actions import key_event
from tda.ui import app_actions as A
from tda.ui import app_commit
from tda.ui import gallery as GA
from tda.ui import guide as G
from tda.ui.app import MainWindow
from tda.ui.panels.palette import PALETTE_WIDTH

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "annotation_guide.md"
QUICK = ROOT / "docs" / "trial_quickstart.md"
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"

#: The ruled map: action -> every key, the primary (left-hand) one first.
MAP: dict[str, tuple[str, ...]] = {
    "commit": ("Q", "Return", "Enter", "Ctrl+K"),
    "commit_override": ("Alt+Q", "Alt+Return", "Alt+Enter"),
    "commit_all": ("Ctrl+Q", "Ctrl+Return", "Ctrl+Enter"),
    "mark_absent": ("`",),
    "cable_range": ("Ctrl+`",),
    "tool_brush": ("W", "B"),
    "tool_polygon": ("D",),
    "tool_circle": ("R",),
    "tool_bench_box": ("V",),
    "toggle_heat": ("G",),
    "toggle_grid": ("Shift+G",),
    "radius_down": ("1", "["),
    "radius_up": ("2", "]"),
    "cycle_visibility": ("Ctrl+V",),
    "toggle_outline": ("Shift+Q",),
    # task U24: Review's 可见性审查 only, where the bare digit is free -- the
    # U22 key stays the alias
    **{f"visibility_{n}": (f"{n}", f"Ctrl+{n}") for n in range(1, 8)},
    # unchanged, and the reason the left hand was free for the rest
    "confirm": ("Space",),
    "tool_eraser": ("E",),
    "flash_compare": ("Tab",),
    "review_rework": ("R",),
    # task U24: Review's Space is the 可见性审查 verdict (which still accepts
    # the frame on the other tabs)
    "review_accept": ("Return", "Enter"),
}
#: Keys the map took away with nothing to replace them in Annotate mode.
RETIRED = ("I", "Ctrl+I", "P", "Y", "3", "4", "5", "6", "7")
#: The new keys a text field must keep for itself.
TYPED = ("Q", "W", "D", "R", "V", "G", "1", "2")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    yield win
    close_window(win)


def _grave(modifiers=Qt.KeyboardModifier.NoModifier, kind=QEvent.Type.KeyPress) -> QKeyEvent:
    """The grave key as Windows hands it to Qt on a US layout: Key_QuoteLeft, text "`"."""
    return QKeyEvent(kind, int(Qt.Key.Key_QuoteLeft), modifiers, "`")


# --------------------------------------------------------------------------- #
# the map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(MAP))
def test_the_map_is_the_ruled_one(name):
    assert A.action_named(name).keys == MAP[name]


def test_no_two_actions_share_a_combination_in_any_mode():
    for mode in A.MODES:
        seen: dict[int, str] = {}
        for action in A.ACTIONS:
            if mode not in action.modes:
                continue
            for combo in A.combos_of(action):
                assert combo not in seen, (mode, action.name, seen.get(combo))
                seen[combo] = action.name
        # every spelling parses to one combination: nothing silently dropped
    for action in A.ACTIONS:
        assert len(A.combos_of(action)) == len(action.keys), action.name


def test_r_is_the_circle_in_annotate_and_rework_in_review():
    """Mode-separate, as it was when R was the bench box."""
    r = Qt.Key.Key_R
    none = Qt.KeyboardModifier.NoModifier
    assert A.action_for(r, none, A.MODE_ANNOTATE).name == "tool_circle"
    assert A.action_for(r, none, A.MODE_REVIEW).name == "review_rework"
    assert A.action_for(r, none, A.MODE_STEPS) is None


@pytest.mark.parametrize("spec", ("Q", "W", "D", "V", "`", "1", "2", "Ctrl+Q", "Ctrl+1",
                                  "Ctrl+V", "Shift+Q", "Shift+G"))
def test_steps_mode_leaves_the_new_keys_to_its_widgets(spec):
    combination = key_event(spec)
    assert A.action_for(combination.key(), combination.modifiers(), A.MODE_STEPS) is None


def test_review_mode_has_the_difference_map_on_g_and_nothing_on_q_or_d():
    def named(spec):
        event = key_event(spec)
        found = A.action_for(event.key(), event.modifiers(), A.MODE_REVIEW)
        return found.name if found else None

    assert named("G") == "toggle_heat"
    assert named("Shift+G") == "toggle_grid"
    assert named("Shift+Q") == "toggle_outline"
    assert (named("Q"), named("D"), named("W")) == (None, None, None)
    # task U24: ` answers the 可见性审查 queue in Review (被线挡住)
    assert named("`") == "vis_review_cable"


# --------------------------------------------------------------------------- #
# pressing them on a real window
# --------------------------------------------------------------------------- #
ALIASES = [("commit", "Return"), ("commit", "Enter"), ("commit", "Ctrl+K"),
           ("commit_override", "Alt+Return"), ("commit_override", "Alt+Enter"),
           ("commit_all", "Ctrl+Return"), ("commit_all", "Ctrl+Enter"),
           ("tool_brush", "B"), ("radius_down", "["), ("radius_up", "]")]


@pytest.mark.parametrize("name,spec", ALIASES, ids=[f"{n}-{s}" for n, s in ALIASES])
def test_the_old_keys_still_work(window, name, spec):
    action = A.action_named(name)
    calls: list[tuple] = []
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(A.MODE_ANNOTATE)
    assert window.handle_key(key_event(spec)) is True
    assert calls == [tuple(action.args)]


@pytest.mark.parametrize("spec", RETIRED)
def test_the_retired_keys_reach_nothing(window, spec):
    window.set_mode(A.MODE_ANNOTATE)
    event = key_event(spec)
    assert A.action_for(event.key(), event.modifiers(), A.MODE_ANNOTATE) is None
    assert window.handle_key(event) is False


def test_the_grave_key_marks_absent_and_ctrl_grave_the_span(window):
    calls: list[str] = []
    window.act_mark_absent = lambda: calls.append("absent")
    window.act_cable_range = lambda: calls.append("range")
    window.set_mode(A.MODE_ANNOTATE)
    assert window.handle_key(_grave()) is True
    assert window.handle_key(_grave(Qt.KeyboardModifier.ControlModifier)) is True
    # Shift+` is "~" on the layout: not bound, not swallowed
    tilde = QKeyEvent(QEvent.Type.KeyPress, int(Qt.Key.Key_AsciiTilde),
                      Qt.KeyboardModifier.ShiftModifier, "~")
    assert window.handle_key(tilde) is False
    assert calls == ["absent", "range"]


def test_the_canvas_takes_no_input_method(window):
    """With no input method on the focused widget Qt switches the IME off for it
    (Windows: the window's input context is dissociated), so a Chinese IME left
    in Chinese mode cannot turn ` into 「·」 or Q into a pinyin composition while
    the canvas has the keys.  Offscreen this pins the attribute only; the
    behaviour of a real IME was not tested."""
    window.show()
    QApplication.processEvents()
    attr = Qt.WidgetAttribute.WA_InputMethodEnabled
    assert not window.canvas.testAttribute(attr)
    assert not window.canvas.viewport().testAttribute(attr)


@pytest.mark.parametrize("widget_cls", (QLineEdit, QPlainTextEdit))
@pytest.mark.parametrize("spec", TYPED + ("`",))
def test_a_text_field_keeps_the_new_letters_and_digits(window, widget_cls, spec):
    calls: list[tuple] = []
    action = None
    for candidate in A.ACTIONS:
        if spec in candidate.keys and A.MODE_ANNOTATE in candidate.modes:
            action = candidate
    assert action is not None
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(A.MODE_ANNOTATE)
    editor = widget_cls(window)
    editor.show()
    editor.setFocus()
    QApplication.processEvents()
    event = _grave() if spec == "`" else key_event(spec)
    assert window.handle_key(event) is False
    assert calls == []


@pytest.mark.parametrize("spec", ("1", "2"))
def test_the_brush_size_box_keeps_the_digits(window, spec):
    """1 and 2 are the brush keys now, and the brush-size box is where a number
    is typed: while it has the focus the digit is the box's."""
    calls: list[tuple] = []
    window.act_radius = lambda *args: calls.append(args)
    window.set_mode(A.MODE_ANNOTATE)
    window.show()
    spin = window.palette.radius_spin
    assert isinstance(spin, QSpinBox)
    spin.setFocus()
    QApplication.processEvents()
    assert window.handle_key(key_event(spec)) is False
    assert calls == []


# --------------------------------------------------------------------------- #
# what the screen says
# --------------------------------------------------------------------------- #
def test_the_palette_captions_are_the_left_hand_keys():
    captions = {name: A.key_caption(A.action_named(name)) for name in
                ("commit", "commit_override", "commit_all", "mark_absent", "tool_brush",
                 "tool_polygon", "tool_circle", "tool_bench_box", "toggle_heat",
                 "review_accept", "flash_compare")}
    assert captions == {"commit": "Q", "commit_override": "Alt+Q", "commit_all": "Ctrl+Q",
                        "mark_absent": "`", "tool_brush": "W", "tool_polygon": "D",
                        "tool_circle": "R", "tool_bench_box": "V", "toggle_heat": "G",
                        # Review's accept is spelled Return first: the button says Enter
                        "review_accept": "Enter", "flash_compare": "按住 Tab"}


#: The captions before U22 (934c6e5), for the width comparison.
OLD_CAPTIONS = {
    "tool_brush": "B", "tool_polygon": "P", "tool_circle": "Y", "tool_bench_box": "R",
    "commit": "Enter", "commit_override": "Alt+Enter", "commit_all": "Ctrl+Enter",
    "toggle_heat": "D", "mark_absent": "I",
}


def test_no_palette_caption_grew_and_the_strip_keeps_its_width(window):
    """U13 fitted the strip to a 1080p window: a wider key badge would wrap a
    button onto a second line and push the 「状态」 group under the fold."""
    window.show()
    QApplication.processEvents()
    assert window.palette.width() == PALETTE_WIDTH
    width = PALETTE_WIDTH - 12
    for name, old in OLD_CAPTIONS.items():
        button = window.palette.button(name)
        assert button.key == A.key_caption(A.action_named(name))
        small = QFontMetrics(button._fonts()[1])
        assert small.horizontalAdvance(button.key) <= small.horizontalAdvance(old), name
        now = button._layout(width)
        new_key = button.key
        button.key = old
        before = button._layout(width)
        button.key = new_key
        assert now["key_w"] <= before["key_w"] and now["height"] <= before["height"], name
        assert button.sizeHint().width() <= PALETTE_WIDTH - 12


def test_the_pure_modules_name_the_mapped_keys():
    """guide.py and gallery.py are Qt-free and cannot read the map: their literal
    keys are pinned to it here."""
    assert G.COMMIT_KEY == A.action_named("commit").keys[0]
    assert {name: key for name, (key, _) in G._WARNING_KEYS.items()} == {
        name: A.action_named(name).keys[0] for name in G._WARNING_KEYS}
    assert GA.ABSENT_KEY == A.action_named("mark_absent").keys[0]
    assert app_commit.WARNING_KEYS == {
        name: A.action_named(name).keys[0]
        for name in ("commit", "commit_all", "commit_override")}
    assert A.tool_key("polygon") == "D" and A.tool_key("circle") == "R"
    assert A.tool_key("brush") == "W" and A.tool_key("bench_box") == "V"
    assert G.COMMIT.startswith("Q（Enter）提交")      # the one line naming the alias


def test_the_mode_cheat_sheet_says_ctrl_digits():
    rows = dict(A.mode_cheat_rows(A.MODE_ANNOTATE))
    # task U24: no label key while annotating any more; Review's row is 1–7
    assert "Ctrl+1–7" not in rows and "1–7" not in rows
    assert rows["Q / Enter / Ctrl+K"].startswith("提交编辑")
    assert rows["1 / ["].startswith("笔刷变小")
    review = dict(A.mode_cheat_rows(A.MODE_REVIEW))
    assert review["1–7"].startswith("可见性审查")


# A sentence on screen that names a key the keyboard moved.  Scanned in every
# string literal of the window's modules (docstrings and comments are not on
# screen); each pattern is a retired spelling in a key's place.
STALE = [
    (re.compile(r"(?<![A-Za-z])Enter(?![A-Za-z])"), "Enter as the commit key"),
    (re.compile(r"Ctrl\+I\b"), "Ctrl+I"),
    (re.compile(r"(?<![A-Za-z0-9_+.\-/])[IPYB](?![A-Za-z0-9_])"), "I / P / Y / B as a key"),
    (re.compile(r"(?<!Ctrl\+)1–7"), "bare 1–7"),
    (re.compile(r"按 [1-7](?!\d)"), "a bare digit"),
    (re.compile(r"台面框 R|台面框（R）|差异图 D"), "the bench box's or the heat map's old key"),
]
#: Literals that name Enter where Enter is meant: the commit line's alias,
#: a text box's own Enter, Review's accept, and notes nobody sees.
ALLOWED = {
    ("guide.py", "Q（Enter）提交"),
    ("guide.py", "→ Enter 接受这一帧"),
    ("palette.py", "（Enter 确定，Esc 取消）"),
    ("review.py", "Enter: open the selected entry"),
    ("app_compat.py", "Enter re-traces"),
    ("app_compat.py", "Ctrl+Enter never folds"),
}


def _literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            yield node.lineno, node.value


def test_no_sentence_on_screen_names_a_moved_key():
    files = sorted((ROOT / "tda" / "ui").rglob("*.py")) + [ROOT / "tda" / "cli_app.py"]
    stale = []
    for path in files:
        if path.name == "app_actions.py":
            continue                    # the map itself: the aliases are its keys
        for line, text in _literals(path):
            # on-screen sentences carry Chinese or a key's English instruction
            if not re.search(r"[一-鿿]|\b(?:press|to save|saves|again|adopt|"
                             r"commit|there|at the)\b", text):
                continue
            if any(path.name == name and part in text for name, part in ALLOWED):
                continue
            for pattern, why in STALE:
                if pattern.search(text):
                    stale.append(f"{path.relative_to(ROOT)}:{line}: {why}: {text[:70]!r}")
    assert not stale, "\n".join(stale)


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #
def test_the_guide_says_once_at_the_top_of_section_4_that_enter_still_works():
    body = GUIDE.read_text(encoding="utf-8")
    assert A.shortcut_markdown() in body
    section = body.split("## 4. ", 1)[1].split("\n## ", 1)[0]
    first = next(line for line in section.splitlines()[1:] if line.strip())
    assert first == "左手 `Q`/`Alt+Q`/`Ctrl+Q` 提交；`Enter` 系列也行。"
    prose = [line for line in body.splitlines() if not line.startswith("| `")]
    assert sum(len(line) for line in prose) <= 7700
    # nowhere else in the prose is Enter the commit key; Review keeps its own
    enters = [line for line in prose if "Enter" in line]
    assert enters == [line for line in enters if line.startswith(("左手 ", "| **Review"))]
    for gone in ("`I`", "`Ctrl+I`", "`P`", "`Y`", "`B`", "`[`", "`1`–`7`"):
        assert gone not in "\n".join(prose), gone
    # the generated table keeps the grave key in a code span of its own
    assert "| `` ` `` | 看不见" in body and "| `` Ctrl+` `` |" in body


def test_the_quickstart_has_the_u22_note_first_in_section_6():
    body = QUICK.read_text(encoding="utf-8")
    section = body.split("## 6. ", 1)[1].split("\n## ", 1)[0]
    first = section.split("\n- ", 1)[1].split("\n- ", 1)[0]
    for needed in ("**常按的键搬到了左手**", "提交 `Q`", "`Alt+Q`", "`Ctrl+Q`", "画笔 `W`",
                   "多边形 `D`", "圆形 `R`", "台面框 `V`", "差异图 `G`", "`1` / `2`",
                   "`` ` ``", "`1` 左边", "`` Ctrl+` ``", "`Ctrl+1`–`7`", "`Ctrl+V`",
                   "`Shift+Q`", "`Shift+G`", "`Enter`", "`Ctrl+K`", "`B`", "`[` `]`",
                   "`I`、`P`、`Y` 不再有作用", "`R` 还是返工", "中文输入法", "「·」"):
        assert needed in first, needed
    # the walk-through uses the new keys
    walk = body.split("## 6. ", 1)[0]
    for gone in ("`I`", "`Ctrl+I`", "`P`", "`Y`", "`B` 画笔", "按 `D`（「差异图」）",
                 "Enter 提交", "改完 Enter"):
        assert gone not in walk, gone


def test_the_spec_has_v1_22_on_top_with_the_whole_table():
    body = SPEC.read_text(encoding="utf-8")
    changes = body.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    # on top when it landed; U23's v1.23, U24's v1.24, U25's v1.25 and U26's
    # v1.26, U27's v1.27 and U28's v1.28 and U29's v1.29 and L6's v1.30 have gone above it since
    # (and U32's v1.31)
    assert entries[0].startswith(("- **v1.22", "- **v1.23", "- **v1.24", "- **v1.25",
                                  "- **v1.26", "- **v1.27",
                                  "- **v1.28", "- **v1.29", "- **v1.30", "- **v1.31", "- **v1.32", "- **v1.33"))
    [first] = [e for e in entries[:12] if e.startswith("- **v1.22")]
    assert first.startswith("- **v1.22（2026-10-23）**：**左手键**（U22，§4.6）")
    for name, keys in MAP.items():
        if name in ("confirm", "tool_eraser", "flash_compare", "review_rework",
                    "review_accept") or name.startswith("visibility_"):
            continue
        assert f"`{name}`" in first, name
        primary = keys[0]
        assert (f"`{primary}`" in first or f"`` {primary} ``" in first), (name, primary)
    assert "`visibility_1..7` → `Ctrl+1`…`Ctrl+7`" in first
    assert changes.index("- **v1.22") < changes.index("- **v1.21")
