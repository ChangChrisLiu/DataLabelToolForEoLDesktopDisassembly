"""Task U44: the Review note in a dock of its own, with the frame's occlusion order above it.

* 「这一帧的遮挡顺序」 from a compiled frame: painting order top-down, only the
  pairs that overlap here with their pixel counts (largest first), a
  ``PairOverride`` pair marked 「（手动）」, a per-frame mask 「（这一帧单独画）」, a
  part nothing of which shows 「被完全盖住」, ``out_of_view`` parts left out, the
  header's counts, bench and chassis never paired;
* the 「说明」 dock: shown in Review, hidden in the other modes, closed by the
  annotator stays closed (``notes/open``), and the 「写说明」 button (no key)
  puts it up with the keyboard in the box; U35's box moved into it unchanged;
* the list in the window: built from the compiled frame the session holds --
  no compile, nothing written -- rebuilt after the frame changes; a click
  selects the part in the instance table; no key was added.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

import app_scene
from tda.core import masks
from tda.core.compiler import CompiledFrame, CompiledInstance
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_notes import NOTES_REVIEW_ONLY, NOTES_SETTING
from tda.ui.occlusion_order import (
    COVERED,
    EMPTY,
    FLOOR_NOTE,
    MANUAL,
    OWN_FRAME,
    floor_parts,
    header_text,
    labeller,
    line_text,
    occlusion_order,
    overlap_pairs,
)
from tda.ui.panels.notes_dock import DOCK_TITLE, INSTANCE_ROLE, ORDER_TITLE
from tda.ui.panels.review import NOTE_HINT, WRITE_NOTE_TEXT, ReviewPanel

DESKTOP, VIEW = app_scene.DESKTOP, app_scene.VIEW
HW = (20, 20)


# --------------------------------------------------------------------------- #
# the order, from a compiled frame
# --------------------------------------------------------------------------- #
def rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros(HW, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def inst(name: str, visible, amodal=None, placement: str = "in_chassis",
         visibility: str = "visible", window=None, amodal_window=None) -> CompiledInstance:
    return CompiledInstance(name, visible, None, amodal, 0.0, visibility, placement, 1,
                            window=window, amodal_window=amodal_window)


def frame() -> CompiledFrame:
    """Bottom-up: chassis, screw (under the cover), ram, ssd, cover; psu out of view;
    a fan on the bench over the chassis' pixels; cover over ssd by a pair override."""
    chassis_shape = rect(0, 20, 0, 20)
    cover = rect(2, 10, 2, 10)                 # 64 px
    ssd_shape = rect(6, 14, 6, 14)             # 64 px, 16 of them under the cover
    ssd = ssd_shape & ~cover                   # 48 px
    screw_shape = rect(3, 5, 3, 5)             # 4 px, all under the cover
    ram = rect(16, 19, 16, 19)                 # 9 px
    chassis = chassis_shape & ~(cover | ssd_shape | ram)
    fan = rect(0, 4, 14, 20)                   # on the bench: never paired with the chassis
    instances = {
        "chassis": inst("chassis", chassis, chassis_shape),
        "screw.01": inst("screw.01", np.zeros(HW, dtype=bool), screw_shape,
                         visibility="occluded_full", window=(3, 3, 5, 5)),
        # a loose window, and no amodal (a shape drawn for this frame only)
        "ram.01": inst("ram.01", ram, None, window=(10, 10, 20, 20)),
        "ssd.01": inst("ssd.01", ssd, ssd_shape),
        "cover.01": inst("cover.01", cover, cover, window=(2, 2, 10, 10),
                         amodal_window=(2, 2, 10, 10)),
        "psu.01": inst("psu.01", None, None, visibility="out_of_view"),
        "fan.01": inst("fan.01", fan, fan, placement="on_bench"),
    }
    return CompiledFrame(
        FrameKey(DESKTOP, 5, VIEW), instances, [], "hash",
        painted={"in_chassis": ["chassis", "screw.01", "ram.01", "ssd.01", "cover.01",
                                "psu.01"],
                 "on_bench": ["fan.01"]},
        pairs={"in_chassis": [("cover.01", "ssd.01")]},
    )


def plain(name: str) -> str:
    return name


def test_the_order_is_painting_order_top_down_with_what_each_part_takes():
    lines = occlusion_order(frame(), drawn={"ssd.01"})
    assert [line.instance for line in lines] == [
        "fan.01", "cover.01", "ssd.01", "ram.01", "screw.01", "chassis"]   # psu.01 left out
    by = {line.instance: line for line in lines}
    # largest first; the pair the override decided is marked
    assert [(t.below, t.pixels, t.manual) for t in by["cover.01"].takes] == [
        ("chassis", 64, False), ("ssd.01", 16, True), ("screw.01", 4, False)]
    assert [(t.below, t.pixels) for t in by["ssd.01"].takes] == [("chassis", 48)]
    assert by["ssd.01"].own_frame and not by["cover.01"].own_frame
    # a part with no amodal: its own pixels are its shape
    assert [(t.below, t.pixels) for t in by["ram.01"].takes] == [("chassis", 9)]
    assert by["screw.01"].covered and not by["screw.01"].takes
    assert by["fan.01"].takes == () and by["chassis"].takes == ()   # bench and chassis apart
    assert not any(line.covered for line in lines if line.instance != "screw.01")


def test_the_lines_say_it_in_plain_words():
    lines = occlusion_order(frame(), drawn={"ssd.01"})
    texts = [line_text(line, plain) for line in lines]
    assert texts[1] == f"cover.01　压着 chassis (64 px)、ssd.01 (16 px){MANUAL}、screw.01 (4 px)"
    assert texts[2] == f"ssd.01{OWN_FRAME}　压着 chassis (48 px)"
    assert texts[4] == f"screw.01　{COVERED}"
    assert texts[0] == "fan.01" and texts[-1] == "chassis"
    assert header_text(lines) == "这一帧 6 个零件，5 对重叠"
    assert header_text([]) == EMPTY == "这一帧没有编译出的零件"
    assert occlusion_order(None) == []


def test_only_overlapping_pairs_count_and_big_counts_are_spaced():
    found = overlap_pairs(frame())
    assert set(found) == {("cover.01", "chassis"), ("cover.01", "ssd.01"),
                          ("cover.01", "screw.01"), ("ssd.01", "chassis"),
                          ("ram.01", "chassis")}
    assert ("cover.01", "ram.01") not in found and ("fan.01", "chassis") not in found
    from tda.ui.occlusion_order import OrderLine, Taken
    assert line_text(OrderLine("a", (Taken("b", 1234567),)), plain) == "a　压着 b (1 234 567 px)"


def test_the_floor_layers_are_not_listed_as_pressed_on_unless_by_hand():
    """Round 2 (controller ruling): 「压着 机箱 / 主板」 on every line is noise."""
    from tda.core.model import InstanceRec

    compiled = frame()
    compiled.pairs["in_chassis"].append(("ram.01", "chassis"))     # a hand-set floor pair
    floor = floor_parts({"chassis": InstanceRec(key="chassis", desktop=DESKTOP, cls="chassis"),
                         "mb.x": InstanceRec(key="mb.x", desktop=DESKTOP, cls="motherboard")},
                        compiled.instances)
    assert floor == {"chassis", "mb.x"}
    assert floor_parts({}, ["motherboard.01", "cover.01"]) == {"motherboard.01"}
    lines = occlusion_order(compiled, drawn={"ssd.01"}, floor=floor)
    by = {line.instance: line for line in lines}
    assert [(t.below, t.pixels, t.manual) for t in by["cover.01"].takes] == [
        ("ssd.01", 16, True), ("screw.01", 4, False)]               # chassis (64 px) dropped
    assert by["ssd.01"].takes == ()                                   # its only pair was the floor
    assert [(t.below, t.pixels, t.manual) for t in by["ram.01"].takes] == [("chassis", 9, True)]
    assert "chassis" in by and by["chassis"].takes == ()             # keeps its own line
    assert line_text(by["ram.01"], plain) == f"ram.01　压着 chassis (9 px){MANUAL}"
    assert header_text(lines) == "这一帧 6 个零件，3 对重叠"        # the listed pairs only


# --------------------------------------------------------------------------- #
# round 3: the real compiler -- an inverting override, per-frame masks
# --------------------------------------------------------------------------- #
def _taken_from(lines, below: str) -> int:
    return sum(t.pixels for line in lines for t in line.takes if t.below == below)


def test_an_override_against_the_global_order_is_what_the_compiler_painted():
    from compiler_scenes import po, run, three_squares

    compiled = run(**three_squares(), overrides=[po("A", "C")])      # A over C, against A < B < C
    lines = occlusion_order(compiled)
    assert [line.instance for line in lines] == ["C", "B", "A"]      # the painting order stays
    by = {line.instance: line for line in lines}
    assert ("C", True) in {(t.below, t.manual) for t in by["A"].takes}
    assert "A" not in {t.below for t in by["C"].takes}               # C no longer takes from A
    assert all(not t.manual for t in by["C"].takes + by["B"].takes)
    # every pixel of a shape is either shown by it or listed as taken from it
    for name, inst in compiled.instances.items():
        assert _taken_from(lines, name) + int(inst.visible.sum()) == int(inst.amodal.sum()), name
    unordered = {frozenset(pair) for pair in overlap_pairs(compiled)}
    assert header_text(lines) == f"这一帧 3 个零件，{len(unordered)} 对重叠"


def test_a_per_frame_mask_shares_pixels_with_the_parts_over_it_and_neither_took_them():
    from compiler_scenes import KEY, rect, run, three_squares
    from tda.core.model import FrameOverride

    patch = rect(8, 8, 56, 56)                       # A drawn on this frame, reaching under B and C
    fo = {"A": FrameOverride(KEY, "A", visible_rle=masks.encode_rle(patch))}
    compiled = run(**three_squares(), frame_overrides=fo)
    shared = compiled.instances["A"].visible & compiled.instances["C"].visible
    assert shared.any()                              # the compiler leaves both showing them
    lines = occlusion_order(compiled, drawn={"A"})
    by = {line.instance: line for line in lines}
    pairs = {(line.instance, t.below) for line in lines for t in line.takes}
    assert not any((b, a) in pairs for a, b in pairs)                # never both ways
    assert ("A", "C") not in pairs and ("A", "B") not in pairs       # A is at the bottom
    assert by["A"].own_frame and line_text(by["A"], plain).startswith(f"A{OWN_FRAME}")
    # what C takes from A is only A's shape that A does not show any more
    a = compiled.instances["A"]
    for t in by["C"].takes:
        if t.below == "A":
            assert t.pixels == int((compiled.instances["C"].visible & a.amodal & ~a.visible).sum())


def test_a_part_with_only_a_per_frame_mask_is_listed_on_top():
    from compiler_scenes import KEY, rect, run, three_squares
    from tda.core.model import FrameOverride

    scene = three_squares()
    patch = rect(30, 30, 60, 60)
    compiled = run(**scene, needs={"A": "mask", "B": "mask", "C": "mask", "D": "mask"},
                   frame_overrides={"D": FrameOverride(KEY, "D",
                                                       visible_rle=masks.encode_rle(patch))})
    assert compiled.instances["D"].visible is not None
    assert "D" not in [n for names in compiled.painted.values() for n in names]
    lines = occlusion_order(compiled)
    assert lines[0].instance == "D" and lines[0].own_frame and not lines[0].covered
    assert header_text(lines).startswith("这一帧 4 个零件")


def test_names_are_the_annotators_name_and_the_key():
    from tda.core.model import InstanceRec

    rec = InstanceRec(key="chassis", desktop=DESKTOP, cls="chassis")
    name = labeller({"chassis": rec})
    text = line_text(occlusion_order(frame())[-1], name)
    assert text.endswith(" chassis") and text != "chassis"       # 「机箱 chassis」


# --------------------------------------------------------------------------- #
# the panel and the dock
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_review_panel_keeps_the_box_until_the_window_takes_it(qapp):
    panel = ReviewPanel()
    assert panel.note_box.parent() is panel and panel.note_row.isHidden()
    box = panel.detach_note_box()
    assert box is panel.note_box and not panel.note_row.isHidden()
    assert panel.note_hint.text() == NOTE_HINT == "说明在右侧「说明」窗"
    button = panel.write_note_button
    assert isinstance(button, QToolButton) and button.text() == WRITE_NOTE_TEXT == "写说明"
    assert panel.findChildren(QPushButton) == []
    asked = []
    panel.sigWriteNote.connect(lambda: asked.append(True))
    button.click()
    assert asked == [True]
    panel.deleteLater()


def seed_overlap(session) -> tuple[str, str]:
    """Two chassis parts of the frame on screen: a big one and one on top of it.

    Not the chassis or a motherboard: those are never listed as pressed on (round 2).
    """
    step = int(session.current().step)
    lower, upper = [key for key in app_scene.chassis_instances(session, step)
                    if key.split(".")[0] not in ("chassis", "motherboard")][:2]
    big = np.zeros(app_scene.HW, dtype=bool)
    big[4:40, 4:40] = True
    small = np.zeros(app_scene.HW, dtype=bool)
    small[10:20, 10:20] = True
    for key, mask in ((lower, big), (upper, small)):
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=app_scene.LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(lower, "main"), (upper, "main")]))
    session.refresh_all()
    return lower, upper


def open_window(tmp_path: Path, seed=None):
    session = app_scene.make_session(tmp_path)
    seeded = seed(session) if seed is not None else None
    win = MainWindow(session, app_scene.make_paths(tmp_path), "tester",
                     sam_queue=app_scene.StubSamQueue())
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:
        win.act_clear_edit()
    QApplication.processEvents()
    return win, seeded


@pytest.fixture
def window(qapp, tmp_path: Path):
    win, seeded = open_window(tmp_path, seed_overlap)
    win.seeded = seeded
    yield win
    app_scene.close_window(win)


def review(win) -> None:
    win.set_mode(A.MODE_REVIEW)
    QApplication.processEvents()
    QApplication.processEvents()


def test_the_dock_shows_in_review_only_and_holds_the_box(window):
    dock = window.notes_dock
    assert dock.objectName() == "dock_notes" and dock.windowTitle().startswith(DOCK_TITLE)
    assert dock.isHidden()                                   # Annotate
    review(window)
    assert not dock.isHidden() and dock.isAncestorOf(window.review.note_box)
    assert not window.review.note_row.isHidden()
    assert window.notes_panel.order.title.text() == ORDER_TITLE
    assert window.notes_panel.order.footer.text() == FLOOR_NOTE == "机箱与主板垫底，不列为被压的零件"
    assert not window.notes_panel.order.footer.isHidden()
    assert window.review.note_box.edit.maximumHeight() > window.review.note_box.EDIT_HEIGHT
    assert window.dockWidgetArea(dock) == window.dockWidgetArea(window.review_dock)
    window.set_mode(A.MODE_ANNOTATE)
    QApplication.processEvents()
    assert dock.isHidden()
    window.set_mode(A.MODE_STEPS)
    QApplication.processEvents()
    assert dock.isHidden()


def test_closed_stays_closed_and_write_note_brings_it_back_with_the_keyboard(window):
    review(window)
    dock = window.notes_dock
    dock.close()
    QApplication.processEvents()
    assert dock.isHidden() and not window.notes_wanted
    assert str(window.settings.value(NOTES_SETTING)).lower() == "false"
    window.set_mode(A.MODE_ANNOTATE)
    review(window)
    assert dock.isHidden()                                   # remembered
    window.review.write_note_button.click()
    QApplication.processEvents()
    assert not dock.isHidden() and window.notes_wanted
    assert str(window.settings.value(NOTES_SETTING)).lower() == "true"
    assert window.focusWidget() is window.review.note_box.edit


def test_a_new_window_remembers_the_dock_closed(qapp, tmp_path: Path):
    win, _ = open_window(tmp_path)
    review(win)
    win.notes_dock.close()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    app_scene.close_window(win)
    again, _ = open_window(tmp_path)
    try:
        review(again)
        assert again.notes_dock.isHidden()
        again.act_write_note()
        assert not again.notes_dock.isHidden()
    finally:
        app_scene.close_window(again)


def test_the_list_is_the_compiled_frame_on_screen_no_compile_no_write(window, monkeypatch):
    lower, upper = window.seeded
    review(window)
    order = window.notes_panel.order
    texts = order.texts()
    assert window.notes_order_builds >= 1
    assert [order.list.item(i).data(INSTANCE_ROLE) for i in range(order.list.count())][:2] \
        == [upper, lower]
    assert f"{lower} (100 px)" in texts[0] and "压着" in texts[0]
    assert order.header.text().startswith("这一帧 ") and "1 对重叠" in order.header.text()

    compiles = []
    real = window.session.truth.compile
    monkeypatch.setattr(window.session.truth, "compile",
                        lambda *a, **k: compiles.append(a) or real(*a, **k))
    writes = window.session.db.conn.total_changes
    window.refresh_notes_order(force=True)
    assert compiles == [] and window.session.db.conn.total_changes == writes
    assert window.notes_order_ms < 50.0
    assert order.texts() == texts

    # a click selects the part in the instance table, and nothing else happens
    mode, step = window.mode, window.session.current().step
    order.list.itemClicked.emit(order.list.item(1))
    assert window.instances.selected_instance() == lower
    assert (window.mode, window.session.current().step) == (mode, step)


def test_the_list_is_rebuilt_after_a_commit_on_the_frame(window):
    lower, upper = window.seeded
    review(window)
    order = window.notes_panel.order
    assert order.list.item(0).data(INSTANCE_ROLE) == upper
    builds = window.notes_order_builds
    window.session.set_zorder_move(lower, upper)          # lower now over upper
    QApplication.processEvents()
    QApplication.processEvents()
    assert window.notes_order_builds > builds
    assert order.list.item(0).data(INSTANCE_ROLE) == lower
    assert f"{upper} (100 px)" in order.texts()[0]


def test_the_dock_menu_ticks_and_unticks_it(window):
    """Round 3, F1: ``triggered`` reached the guarded slot with no argument and raised."""
    dock, action = window.notes_dock, window.notes_dock.toggleViewAction()
    review(window)
    assert not dock.isHidden() and action.isChecked()
    action.trigger()                                      # untick in Review
    QApplication.processEvents()
    assert dock.isHidden() and not window.notes_wanted
    assert str(window.settings.value(NOTES_SETTING)).lower() == "false"
    assert "TypeError" not in window.status_message()
    action.trigger()                                      # tick again
    QApplication.processEvents()
    assert not dock.isHidden() and window.notes_wanted
    assert str(window.settings.value(NOTES_SETTING)).lower() == "true"

    # ticked outside Review: wanted, said, and not shown in Annotate
    dock.close()
    QApplication.processEvents()
    window.set_mode(A.MODE_ANNOTATE)
    QApplication.processEvents()
    assert dock.isHidden() and not window.notes_wanted
    action.trigger()
    QApplication.processEvents()
    assert dock.isHidden() and window.notes_wanted
    assert window.status_message() == NOTES_REVIEW_ONLY
    assert str(window.settings.value(NOTES_SETTING)).lower() == "true"
    review(window)
    assert not dock.isHidden()


def test_the_list_follows_the_frame(window):
    review(window)
    builds = window.notes_order_builds
    step = int(window.session.current().step)
    window.timeline_goto(step - 1)
    QApplication.processEvents()
    QApplication.processEvents()
    assert int(window.session.current().step) == step - 1
    assert window.notes_order_builds > builds
    assert window.notes_panel.order.header.text() not in ("",)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    review_section = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    bullets = [line for line in review_section.splitlines() if line.startswith("- **")]
    heads = [b.split("**")[1] for b in bullets]
    assert heads.index("说明窗与遮挡顺序") == heads.index("这一帧的说明") + 1
    [bullet] = [b for b in bullets if b.startswith("- **说明窗与遮挡顺序**（v1.43，U44）")]
    for needed in ("「我还是想着我口述然后你来理解", "**说明仍是自由文字**", "「写说明」", "**没有快捷键**",
                   "「这一帧 n 个零件，m 对重叠」", "「这一帧没有编译出的零件」", "「（手动）」",
                   "「（这一帧单独画）」", "「被完全盖住」", "`out_of_view`", "**不另编译、不写库**",
                   "`notes/open`", "`cli notes` 不变", "**机箱和主板垫底**",
                   "「机箱与主板垫底，不列为被压的零件」"):
        assert needed in bullet, needed
    changes = spec.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    [entry] = [e for e in entries[:5] if e.startswith("- **v1.43")]
    assert entry.startswith("- **v1.43（2026-10-30）**：**复查的「说明」停靠窗与这一帧的遮挡顺序**（U44")
    quick = (ROOT / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    review_part = quick.split("## 4. 走完之后：Review", 1)[1].split("### 冲突从哪来", 1)[0]
    item = review_part.split("\n4. ", 1)[1].replace("\n   ", "")
    for needed in ("右侧「说明」窗", "「写说明」", "「这一帧的遮挡顺序」", "谁压着谁", "「（手动）」"):
        assert needed in item, needed
    guide = (ROOT / "docs" / "annotation_guide.md").read_text(encoding="utf-8")
    assert "「说明」窗：遮挡顺序与说明，翻帧自动存，`cli notes` 读。" in guide


def test_no_key_was_added():
    assert not any(action.slot == "act_write_note" for action in A.ACTIONS)
    assert not any("写说明" in (action.label_zh or "") for action in A.ACTIONS)
