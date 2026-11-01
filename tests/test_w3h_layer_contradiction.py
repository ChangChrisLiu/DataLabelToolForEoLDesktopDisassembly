"""Task W3h: 「日志里下面那件先动了」 -- the log's word is a question, not a proof; a covering
whose lower part the log moved first is 不挡 or a layer order painted the wrong way round,
and the user decides with the picture (「不挡」 / 「上下翻转」).

* the bucket is exactly ``log_proof``'s set (Qt-free): every candidate whose lower part
  moved while the upper one was still in the way, whatever the class pair -- a small
  part on top (connector, clip, latch) included, a screw / connector / cable underneath
  still not; a class pair the user ruled 挡 either way round carries the tag
  「用户已定：挡 → 层序可能画反」; above-first stays asked; judged after the chain and
  the class-level 不挡;
* the row text and tip, the status line;
* the dock's fold: 「不挡」 and 「上下翻转」 per row, the subtitle, ``sigFlip``;
* the flip on a tiny synthetic desktop with a confirmed frame (offscreen Qt): one
  ``flip_pair_layers`` op (``set_pair_override`` in the op log) with a note, the two
  parts' rows re-signed through ``accept_new``, the frame still verified, ``Ctrl+Z``
  takes the override away **and** puts the rows back (no conflict left); the part that
  gained pixels within the tolerance is re-signed by the refreeze (round 2, L1); a frame
  demoted before the flip stays demoted (M1); the refusals (another part's conflict,
  already flipped, by hand, no shared frame, outside Review);
* no new key; the docs (spec §4.4 / §7.3 / §8.1 / §14 v1.53, handoff, the discussion docs).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

import app_scene
from test_w3c_rulings_dock import box, open_window, review, settle
from tda.core import masks
from tda.core.graph import Edge
from tda.core.model import ActionRec, FrameKey, InstanceRec, PairOverride, ShapeKeyframe, \
    ShapePart, ZOrderRec
from tda.core.taxonomy import load_taxonomy
from tda.ui import app_actions as A
from tda.ui import cover_rulings as CR
from tda.ui.app_rulings import FLIP_OUTSIDE
from tda.ui.cover_rulings import (
    FLIP_TEXT,
    TAG_LAYER,
    TAG_OTHER,
    TAG_USER_BLOCKS,
    ClassRulings,
    class_blocking,
    cover_log,
    flip_text,
    log_proof,
    proven_text,
    proven_tip,
    triage_pairs,
)
from tda.ui.panels.notes_dock import PROVEN_NOTE, PROVEN_TITLE, RulingsSection
from tda.ui.session_api import SessionRefusal
from tda.ui.session_rulings import (
    FLIP_ALREADY,
    FLIP_BY_HAND,
    FLIP_CONFLICT,
    FLIP_NO_FRAMES,
    FLIP_OP,
)

D = 4
TAX = load_taxonomy()
ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
DISCUSSION = ROOT / "docs" / "superpowers" / "discussions" / "2026-10-30-vlm-paper-scope"
DESKTOP, VIEW, LAST = app_scene.DESKTOP, app_scene.VIEW, app_scene.LAST_STEP


def rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=D, cls=cls, attrs=attrs, **kw)


def act(step: int, target: str, verb: str, idx: int = 0, result: str = "success") -> ActionRec:
    return ActionRec(desktop=D, step=step, idx=idx, target=target, verb=verb, result=result)


SCENE = {r.key: r for r in (
    rec("cover.01", "cover", attrs={"of": "expansion_slot"}),
    rec("psu.01", "psu"),
    rec("motherboard.01", "motherboard"),
    rec("storage_drive.hdd.01", "storage_drive", attrs={"kind": "hdd"}),
    rec("drive_cage.01", "drive_cage"),
    rec("optical_drive.01", "optical_drive"),
    rec("cpu_cooler.fan.01", "cpu_cooler", attrs={"kind": "fan"}),
    rec("cooler_latch.01", "cooler_latch"),
    rec("connector.02", "connector"),
    rec("card_latch.01", "card_latch"),
)}
#: D04: the expansion-slot cover opens at 22 with the PSU in place; the PSU leaves at 39.
D04_LOG = [act(22, "cover.01", "open"), act(39, "psu.01", "remove")]
#: D22 before the fix: the Dell PSU module folds down (displace) at 12 with the HDD installed.
D22_LOG = [act(12, "psu.01", "displace"), act(20, "storage_drive.hdd.01", "remove"),
           act(31, "psu.01", "remove")]
PSU_COVER = ("psu.01", "cover.01", 1535)
HDD_PSU = ("storage_drive.hdd.01", "psu.01", 58_123)
PSU_HDD = ("psu.01", "storage_drive.hdd.01", 58_123)


def keys(cands) -> list[tuple[str, str]]:
    return [(c.above, c.below) for c in cands]


# --------------------------------------------------------------------------- #
# the bucket
# --------------------------------------------------------------------------- #
def test_the_shipped_rulings_name_the_pair_either_way_round():
    entry, reversed_ = class_blocking("psu", "cover")            # the picture: PSU over the cover
    assert entry[:2] == ("cover", "psu") and reversed_ is True    # E4 rules the other way round
    entry, reversed_ = class_blocking("psu", "storage_drive.hdd")
    assert entry[:2] == ("psu", "storage_drive.hdd") and reversed_ is False     # B8
    assert class_blocking("cpu_cooler.fan", "cooler_latch") == (None, False)
    assert class_blocking("psu", "motherboard") == (None, False)                 # A4 is 不挡
    assert class_blocking("psu", "cover", classes=ClassRulings()) == (None, False)


def test_the_bucket_is_the_log_proof_set_and_the_ruled_pairs_carry_the_tag():
    log = cover_log(SCENE, D04_LOG, TAX)
    tri = triage_pairs([PSU_COVER], SCENE, [], {}, log=log)
    assert keys(tri.proven) == [("psu.01", "cover.01")] and tri.asked == []
    [cand] = tri.proven
    assert cand.proof == log_proof(log, "psu.01", "cover.01")     # the same replay, no second one
    assert (cand.proof.step, cand.proof.verb) == (22, "open")
    assert cand.ruled == ("cover", "psu") and cand.ruled_reversed and "E4" in cand.ruled_note
    assert cand.layer_tag == TAG_LAYER == "用户已定：挡 → 层序可能画反"
    assert cand.tag == TAG_OTHER
    # D22 before the fix: the HDD painted over the PSU that folds down on it (B8 reversed)
    [cand] = triage_pairs([HDD_PSU], SCENE, [], {}, log=cover_log(SCENE, D22_LOG, TAX)).proven
    assert (cand.proof.step, cand.proof.verb) == (12, "displace")
    assert cand.ruled == ("psu", "storage_drive.hdd") and cand.ruled_reversed and cand.layer_tag
    # the picture's own orientation ruled 挡, with a log that removed the HDD first
    hdd_first = cover_log(SCENE, [act(5, "storage_drive.hdd.01", "remove"),
                                  act(9, "psu.01", "remove")], TAX)
    [cand] = triage_pairs([PSU_HDD], SCENE, [], {}, log=hdd_first).proven
    assert cand.ruled == ("psu", "storage_drive.hdd") and not cand.ruled_reversed
    assert cand.tag == TAG_USER_BLOCKS and cand.reason.startswith("B8") and cand.layer_tag
    # a pair the user did not rule by class is in the bucket too, without the tag
    latch = cover_log(SCENE, [act(3, "cooler_latch.01", "open"),
                              act(8, "cpu_cooler.fan.01", "remove")], TAX)
    [cand] = triage_pairs([("cpu_cooler.fan.01", "cooler_latch.01", 2000)], SCENE, [], {},
                          log=latch).proven
    assert cand.ruled == () and cand.layer_tag == "" and cand.proof.step == 3
    # without class rulings the tag goes, the bucket stays
    [plain] = triage_pairs([PSU_COVER], SCENE, [], {}, log=log, classes=ClassRulings()).proven
    assert plain.layer_tag == "" and plain.ruled == ()


def test_a_small_part_on_top_joins_the_bucket_but_is_never_asked():
    # the Dell cage swung aside (displace) with the drive still plugged in: the connector
    # painted over the cage went nowhere first; the latch opened before the cage moved
    log = cover_log(SCENE, [act(3, "card_latch.01", "open"), act(9, "drive_cage.01", "displace"),
                            act(20, "connector.02", "disconnect")], TAX)
    pairs = [("connector.02", "drive_cage.01", 1973), ("card_latch.01", "drive_cage.01", 1500)]
    tri = triage_pairs(pairs, SCENE, [], {}, log=log)
    assert keys(tri.proven) == [("connector.02", "drive_cage.01")]   # the latch opened first
    assert tri.asked == [] and tri.proven[0].ruled == ()
    # without a log the small part on top is dropped as before (W3c's 本版不做)
    assert triage_pairs(pairs, SCENE, [], {}).asked == []
    assert triage_pairs(pairs, SCENE, [], {}).proven == []
    # a connector underneath never takes part, whatever the log says
    under = cover_log(SCENE, [act(3, "connector.02", "disconnect"), act(9, "psu.01", "remove")], TAX)
    tri = triage_pairs([("psu.01", "connector.02", 1200)], SCENE, [], {}, log=under)
    assert tri.proven == [] and tri.asked == []


def test_a_ruled_pair_whose_upper_part_moved_first_is_asked_as_before():
    log = cover_log(SCENE, D22_LOG, TAX)
    tri = triage_pairs([PSU_HDD], SCENE, [], {}, log=log)          # D22 after the fix
    assert tri.proven == []
    assert [(c.above, c.below, c.tag) for c in tri.asked] == [
        ("psu.01", "storage_drive.hdd.01", TAG_USER_BLOCKS)]
    assert tri.asked[0].ruled == ("psu", "storage_drive.hdd") and tri.asked[0].layer_tag == ""
    # reversed ruling, upper part first: asked as 其他
    hdd_first = cover_log(SCENE, [act(5, "storage_drive.hdd.01", "remove"),
                                  act(9, "psu.01", "remove")], TAX)
    tri = triage_pairs([HDD_PSU], SCENE, [], {}, log=hdd_first)
    assert [(c.above, c.below, c.tag) for c in tri.asked] == [
        ("storage_drive.hdd.01", "psu.01", TAG_OTHER)]
    assert tri.proven == []


def test_judged_after_the_chain_and_the_class_level_not_blocks():
    log = cover_log(SCENE, D04_LOG, TAX)
    chain = [Edge(type="covered_by", target="motherboard.01", blocker="cover.01"),
             Edge(type="covered_by", target="cover.01", blocker="psu.01")]
    board_first = cover_log(SCENE, [act(3, "motherboard.01", "remove"),
                                    act(9, "psu.01", "remove")], TAX)
    tri = triage_pairs([("psu.01", "motherboard.01", 9000)], SCENE, chain, {}, log=board_first)
    assert keys(tri.indirect) == [("psu.01", "motherboard.01")] and tri.proven == []
    # psu -> motherboard is 不挡 by class (A4): folded there even with the board moved first
    tri = triage_pairs([("psu.01", "motherboard.01", 9000)], SCENE, [], {}, log=board_first)
    assert keys(tri.class_not_blocks) == [("psu.01", "motherboard.01")] and tri.proven == []
    # a ruled pair is in no list
    tri = triage_pairs([PSU_COVER], SCENE, [], {("psu.01", "cover.01"): {"ruling": "blocks"}},
                       log=log)
    assert tri.proven == [] and tri.asked == []


# --------------------------------------------------------------------------- #
# the texts
# --------------------------------------------------------------------------- #
def test_the_row_text_and_the_tip():
    log = cover_log(SCENE, D04_LOG, TAX)
    [cand] = triage_pairs([PSU_COVER], SCENE, [], {}, log=log,
                          states={"psu.01": "installed", "cover.01": "closed"}).proven
    assert proven_text(cand) == ("psu.01（装着）→ cover.01（关上） · 像素 1 535 · "
                                 "第 22 步先动了下面的 cover.01 · 用户已定：挡 → 层序可能画反")
    tip = proven_tip(cand)
    lines = tip.splitlines()
    assert lines[0].startswith("第 22 步打开 cover.01 时 psu.01 还是装着：要么 psu.01 没挡着")
    assert "「不挡」" in lines[0] and "「上下翻转」" in lines[0] and "Ctrl+Z" in lines[0]
    assert lines[1].startswith("你按类别裁定 cover → psu 挡（E4 挡")
    assert "cover.01 应画在 psu.01 之上，这一帧却画反了" in lines[1]
    assert lines[-1] == "日志：psu.01 第 39 步 拆下，cover.01 第 22 步 打开"
    # the picture's own orientation ruled 挡: the tip says the log disagrees with the ruling
    [cand] = triage_pairs([PSU_HDD], SCENE, [], {},
                          log=cover_log(SCENE, [act(5, "storage_drive.hdd.01", "remove"),
                                                act(9, "psu.01", "remove")], TAX)).proven
    assert proven_text(cand) == ("psu.01 → storage_drive.hdd.01 · 像素 58 123 · "
                                 "第 5 步先动了下面的 storage_drive.hdd.01 · 用户已定：挡 → 层序可能画反")
    assert "你按类别裁定 psu → storage_drive.hdd 挡（B8 挡" in proven_tip(cand)
    assert "日志却先动了下面那件" in proven_tip(cand)
    # no class ruling: no tag, no second line
    latch = cover_log(SCENE, [act(3, "cooler_latch.01", "release"),
                              act(8, "cpu_cooler.fan.01", "remove")], TAX)
    [cand] = triage_pairs([("cpu_cooler.fan.01", "cooler_latch.01", 2000)], SCENE, [], {},
                          log=latch).proven
    assert proven_text(cand) == ("cpu_cooler.fan.01 → cooler_latch.01 · 像素 2 000 · "
                                 "第 3 步先动了下面的 cooler_latch.01")
    assert len(proven_tip(cand).splitlines()) == 2 and "你按类别裁定" not in proven_tip(cand)
    # a candidate made without the log falls back to the plain row text
    assert proven_text(CR.Candidate("a", "b", "x", "y", 1500, TAG_OTHER, "", 0)) == \
        "a → b · 像素 1 500 · 其他"


def test_the_status_line():
    result = {"frames": list(range(1, 22)), "frozen": [3, 7, 21], "accepted": 6, "resigned": 1,
              "rewritten": 9}
    assert flip_text(4, 21, "psu.01", "cover.01", result) == (
        "D04 第 21 帧：cover.01 翻到 psu.01 之上 · 这一段 21 帧（已确认 3 帧）· 接受新值 6 条 · "
        "重签 1 帧 9 行")
    result["left_open"] = [(7, "motherboard.01")]
    result["verify_refused"] = [(3, "conflict 9 is open")]
    text = flip_text(4, 21, "psu.01", "cover.01", result)
    assert text.endswith("；另有 1 条别的零件的冲突留在复查队列（第 7 帧 motherboard.01）"
                         "；1 帧没能重签（第 3 帧 conflict 9 is open）")


# --------------------------------------------------------------------------- #
# the dock's fold
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_dock_fold_has_two_buttons_per_row(qapp):
    log = cover_log(SCENE, D04_LOG + [act(5, "drive_cage.01", "remove"),
                                      act(9, "optical_drive.01", "remove")], TAX)
    tri = triage_pairs([PSU_COVER, ("optical_drive.01", "drive_cage.01", 9000)], SCENE, [], {},
                       log=log)
    assert keys(tri.proven) == [("optical_drive.01", "drive_cage.01"), ("psu.01", "cover.01")]
    section = RulingsSection()
    heard, ruled = [], []
    section.sigFlip.connect(lambda a, b: heard.append((a, b)))
    section.sigRule.connect(lambda pairs, ruling, classes: ruled.append((pairs, ruling)))
    section.set_rulings(tri.asked, {}, proven=tri.proven)
    assert section.proven_toggle.text() == PROVEN_TITLE.format(k=2) == "日志里下面那件先动了（2）"
    assert not section.proven_toggle.isHidden() and not section.proven_toggle.isChecked()
    assert section.proven_note.text() == PROVEN_NOTE == "不挡，或者层序画反了 —— 请看图定"
    assert section.proven_texts()[1] == proven_text(tri.proven[1])
    buttons = section.proven.findChildren(QPushButton)
    assert [b.text() for b in buttons] == ["不挡", FLIP_TEXT, "不挡", FLIP_TEXT]   # never 「挡着」
    assert all(b.focusPolicy() == Qt.FocusPolicy.NoFocus for b in buttons)
    [(_c1, free1, flip1), (_c2, free2, flip2)] = section.proven_buttons
    flip2.click()
    assert heard == [("psu.01", "cover.01")]
    free1.click()
    assert ruled == [([("optical_drive.01", "drive_cage.01")], "not_blocks")]
    assert ("psu.01", "cover.01") in [pair for pair, _row in section.hover_rows]
    tip = section._proven_labels[1].toolTip()
    assert tip.startswith("psu → cover：") and "你按类别裁定 cover → psu 挡" in tip
    assert tip.count("日志：") == 1                           # the 日志 line once (round 2, S3)
    section.set_rulings([], {})
    assert section.proven_toggle.isHidden()


# --------------------------------------------------------------------------- #
# the flip on a tiny desktop with a confirmed frame
# --------------------------------------------------------------------------- #
#: The dock's own class rulings: the cover must open before the PSU comes out (E4),
#: so a PSU painted over an opened cover is a layer error.
FLIP_CLASSES = ClassRulings(blocks=(("cover", "psu", "测试：先开盖板再拆电源"),))
#: The small part the tolerance test paints over the board: 7 x 7 = 49 px, 1.2 % of it.
SMALL = box(10, 17, 10, 17)


@pytest.fixture
def flip_class_rulings(monkeypatch):
    """The window tests only: the Qt-free ones above read the shipped W3."""
    monkeypatch.setattr(CR, "class_rulings", lambda path=None: FLIP_CLASSES)


def seed_pile(session, top: dict) -> None:
    """``top`` painted in order over the full-frame board, every other chassis part of the
    last step a small cell underneath, so the frame can be confirmed and the overlaps are
    the pairs' alone; the last frame confirmed and on screen."""
    shapes = {"motherboard.01": box(0, 64, 0, 64), **top}
    others = [k for k in app_scene.chassis_instances(session, LAST) if k not in shapes]
    order = []
    for key, mask in [(k, app_scene.cell(i)) for i, k in enumerate(others)] + list(shapes.items()):
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=LAST, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(mask))]))
        order.append((key, "main"))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, order))
    session.sweeper.stop()
    session.sweeper_enabled = False                # every re-check runs here, in order
    session.refresh_all()
    session.goto(LAST, force=True)
    assert session.confirm_frame() is True         # the last frame is frozen ...
    session.goto(LAST, force=True)                 # ... and on screen


def seed_flip(session) -> None:
    """The PSU painted over the expansion-slot cover (D04 frame 21)."""
    seed_pile(session, {"cover.01": box(0, 30, 0, 40), "psu.01": box(0, 64, 0, 60)})


@pytest.fixture
def flip_window(qapp, tmp_path: Path, flip_class_rulings):
    win = open_window(tmp_path, seed=seed_flip)
    yield win
    app_scene.close_window(win)


def visible_px(db, key: FrameKey, instance: str) -> int:
    return masks.rle_area(db.compiled(key)[instance]["visible_rle"])


def flip_ops(win) -> list[dict]:
    return win.session.db.ops_of_kind(DESKTOP, "set_pair_override")


def test_the_flip_writes_the_override_and_resigns_the_two_parts(flip_window):
    win = flip_window
    db = win.session.db
    key = FrameKey(DESKTOP, LAST, VIEW)
    assert db.get_frame(key)["review_status"] == "verified"
    review(win)
    section = win.notes_panel.rulings
    assert keys(section.proven_candidates) == [("psu.01", "cover.01")]
    assert section.proven_texts() == ["psu.01（装着）→ cover.01（打开） · 像素 1 200 · "
                                      "第 2 步先动了下面的 cover.01 · 用户已定：挡 → 层序可能画反"]
    assert ("psu.01", "cover.01") not in keys(section.candidates)
    before = {k: visible_px(db, key, k) for k in ("psu.01", "cover.01")}
    [(cand, _free, flip)] = section.proven_buttons
    flip.click()
    settle()
    # one op-log row, with the note naming the pair and the frames; the undo stack's op
    # is the flip's own kind
    [op] = flip_ops(win)
    assert (op["payload"]["above"], op["payload"]["below"]) == ("cover.01", "psu.01")
    assert op["payload"]["flip"] is True and op["payload"]["steps"] == list(range(1, LAST + 1))
    assert op["payload"]["note"].startswith("日志里下面那件先动了（W3h）：按拆解顺序把 cover.01 翻到 psu.01 之上")
    assert win.session.undo_stack.ops[-1].kind == FLIP_OP == "flip_pair_layers"
    assert db.pair_overrides(DESKTOP, VIEW, 1) == [PairOverride(DESKTOP, VIEW, 1, "cover.01", "psu.01")]
    # the two parts' rows were re-signed through accept_new; the frame stays verified
    rows = db.compiled(key)
    assert {rows[k]["status"] for k in ("psu.01", "cover.01")} == {"verified"}
    assert {rows[k]["verified_by"] for k in ("psu.01", "cover.01")} == {"tester"}
    assert visible_px(db, key, "psu.01") == before["psu.01"] - 1200
    assert visible_px(db, key, "cover.01") == before["cover.01"] + 1200
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []
    closed = db.conflicts(DESKTOP, VIEW, open_only=False)
    assert sorted(c["instance"] for c in closed) == ["cover.01", "psu.01"]
    assert {c["resolution"] for c in closed} == {"accept_new"}
    assert db.get_frame(key)["review_status"] == "verified"
    assert win.status_message() == (f"D13 第 {LAST} 帧：cover.01 翻到 psu.01 之上 · 这一段 {LAST} 帧"
                                    "（已确认 1 帧）· 接受新值 2 条 · 重签 0 帧 0 行")
    # the section follows the new picture: the cover now lies over the PSU, an open row
    assert section.proven_candidates == [] and section.proven_toggle.isHidden()
    assert ("cover.01", "psu.01") in keys(section.candidates)
    # Ctrl+Z takes the override away AND puts the rows back, leaving no conflict (round 2, M2)
    win.act_undo()
    settle()
    assert db.pair_overrides(DESKTOP, VIEW, 1) == []
    assert len(flip_ops(win)) == 1                   # a replay logs no second override row
    assert {k: visible_px(db, key, k) for k in ("psu.01", "cover.01")} == before
    rows = db.compiled(key)
    assert {rows[k]["status"] for k in ("psu.01", "cover.01")} == {"verified"}
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []
    assert db.get_frame(key)["review_status"] == "verified"
    assert keys(section.proven_candidates) == [("psu.01", "cover.01")]
    # ... and redo flips again
    win.act_redo()
    settle()
    assert db.pair_overrides(DESKTOP, VIEW, 1) == [PairOverride(DESKTOP, VIEW, 1, "cover.01", "psu.01")]
    assert visible_px(db, key, "cover.01") == before["cover.01"] + 1200
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []


def test_the_flip_refuses_with_a_reason(flip_window):
    win = flip_window
    db = win.session.db
    key = FrameKey(DESKTOP, LAST, VIEW)
    # outside Review nothing is written
    win.on_flip_layers("psu.01", "cover.01")
    assert win.status_message() == FLIP_OUTSIDE and flip_ops(win) == []
    review(win)
    # another part's open conflict on a reached frame
    cid = db.add_conflict(key, "ram_module.01", None, None, 0)
    win.on_flip_layers("psu.01", "cover.01")
    assert win.status_message() == FLIP_CONFLICT.format(step=LAST, instance="ram_module.01", n=1)
    assert flip_ops(win) == [] and db.pair_overrides(DESKTOP, VIEW, 1) == []
    db.resolve_conflict(cid, "edited")
    # the segment already holds the override: refused (the row itself is gone from the
    # picture by then, so no 「已翻过」 is shown -- the guard is the data's)
    db.set_pair_override(PairOverride(DESKTOP, VIEW, 1, "cover.01", "psu.01"))
    win.on_flip_layers("psu.01", "cover.01")
    assert win.status_message() == FLIP_ALREADY.format(above="psu.01", below="cover.01")
    assert flip_ops(win) == []
    # the picture's own order set by hand: refused too
    db.delete_pair_override(PairOverride(DESKTOP, VIEW, 1, "cover.01", "psu.01"))
    db.set_pair_override(PairOverride(DESKTOP, VIEW, 1, "psu.01", "cover.01"))
    win.on_flip_layers("psu.01", "cover.01")
    assert win.status_message() == FLIP_BY_HAND.format(above="psu.01", below="cover.01")
    assert flip_ops(win) == []
    db.delete_pair_override(PairOverride(DESKTOP, VIEW, 1, "psu.01", "cover.01"))
    # a part this frame does not have is the layering gestures' own refusal ...
    with pytest.raises(SessionRefusal, match="no such instance in this frame"):
        win.session.flip_pair_layers("psu.01", "storage_drive.ssd.01")
    # ... and a pair that shares no frame where both are mask layers is this one's (S4)
    from tda.ui import session_rulings as SR
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SR.edit, "pair_steps", lambda *a, **k: [])
        with pytest.raises(SessionRefusal, match=FLIP_NO_FRAMES):
            win.session.flip_pair_layers("psu.01", "cover.01")
    assert flip_ops(win) == [] and db.pair_overrides(DESKTOP, VIEW, 1) == []


def test_the_part_that_gained_pixels_within_the_tolerance_is_resigned_too(qapp, tmp_path,
                                                                            flip_class_rulings):
    """Round 2, L1: ``cover.02`` (49 px) over the board; flipping moves 49 px between the
    two -- 1.2 % of the board, 9 px beyond the 2-px band of the small one: no conflict --
    which only the refreeze re-signs; the undo takes them back the same way."""
    session = app_scene.make_session(tmp_path)
    seed_pile(session, {"cover.02": SMALL})
    db, key = session.db, FrameKey(DESKTOP, LAST, VIEW)
    board, small = visible_px(db, key, "motherboard.01"), visible_px(db, key, "cover.02")
    assert small == 49 and db.compiled(key)["motherboard.01"]["verified_by"] == "tester"
    result = session.flip_pair_layers("cover.02", "motherboard.01")
    # 49 px either way is within the 2 % / 20 px tolerance band for both rows: no conflict
    # opens, and without the refreeze both would have kept their old masks (a hole)
    assert result["accepted"] == 0 and result["left_open"] == []
    assert result["rewritten"] == 2 and result["rewritten_frames"] == 1   # both, by refreeze
    assert result["resigned"] == 0 and result["verify_refused"] == []
    assert visible_px(db, key, "motherboard.01") == board + 49
    assert visible_px(db, key, "cover.02") == 0
    rows = db.compiled(key)
    assert {rows[k]["status"] for k in ("motherboard.01", "cover.02")} == {"verified"}
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []
    assert db.get_frame(key)["review_status"] == "verified"
    assert len(db.ops_of_kind(DESKTOP, "refreeze_frame")) == 1   # one refreeze_frame op
    # undo: the board loses the 49 px within the tolerance -> refreeze again; the small
    # part's conflict is accepted
    assert session.undo()
    assert visible_px(db, key, "motherboard.01") == board
    assert visible_px(db, key, "cover.02") == small
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []
    assert db.pair_overrides(DESKTOP, VIEW, 1) == []
    assert len(db.ops_of_kind(DESKTOP, "refreeze_frame")) == 2   # the undo's own
    session.close()


def test_a_frame_demoted_before_the_flip_stays_demoted(qapp, tmp_path, flip_class_rulings):
    """Round 2, M1: the flip is not the confirmation of a frame the annotator still owes."""
    session = app_scene.make_session(tmp_path)
    seed_flip(session)
    db, key = session.db, FrameKey(DESKTOP, LAST, VIEW)
    session.truth.demote_frame(key, "test: a part was added")
    assert db.get_frame(key)["review_status"] == "needs_review"
    signed_before = len(db.ops_of_kind(DESKTOP, "verify_frame"))
    result = session.flip_pair_layers("psu.01", "cover.01")
    assert result["frozen"] == [LAST] and result["accepted"] == 2      # its signed rows still count
    assert result["resigned"] == 0 and result["verify_refused"] == []
    assert db.get_frame(key)["review_status"] == "needs_review"        # not signed by the flip
    assert db.conflicts(DESKTOP, VIEW, open_only=True) == []
    assert len(db.ops_of_kind(DESKTOP, "verify_frame")) == signed_before   # nobody signed it
    session.close()


# --------------------------------------------------------------------------- #
# no key; the docs
# --------------------------------------------------------------------------- #
def test_no_key_was_added_and_the_docs_say_it():
    for action in A.ACTIONS:
        assert not any(word in (action.label_zh or "") for word in ("翻转", "层序", "先动了"))
        assert "flip" not in (action.slot or "")
    spec = SPEC.read_text(encoding="utf-8")
    review_section = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    [para] = [line for line in review_section.splitlines() if line.startswith("  **v1.51（W3f）")]
    for needed in ("**日志里下面那件先动了（k）**", "v1.53（W3h", "你的遮挡关系算错了", "「不挡」", "「上下翻转」",
                   "「用户已定：挡 → 层序可能画反」", "重签", "Ctrl+Z"):
        assert needed in para, needed
    assert "已翻过" not in para
    section = spec.split("### 7.3 生成", 1)[1].split("### 7.4", 1)[0]
    [para] = [line for line in section.splitlines()
              if line.startswith("**看图裁定：白名单 W3 与层序修正（v1.53，W3h）**")]
    for needed in ("`version: W3`", "你的遮挡关系算错了", "用我们的拆解的顺序作为主要参考", "B8", "E1",
                   "`pair_override`", "`fix_layer_pairs.py`", "D02", "「日志里下面那件先动了（k）」",
                   "「上下翻转」", "`log_proof`", "`session_rulings.flip_pair_layers`", "`accept_new`",
                   "`TruthService.refreeze`", "pixels:W3", "**241**"):
        assert needed in para, needed
    assert "已翻过" not in para
    cv = spec.split("### 8.1 CV 导出", 1)[1].split("### 8.2", 1)[0]
    assert "pixels:W3" in cv and "W3h" in cv
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:3] if e.startswith("- **v1.53（2026-10-31）**")]
    for needed in ("（W3h，", "W3", "日志里下面那件先动了", "上下翻转", "`pair_override`", "D02",
                   "**241**", "**22**", "refreeze"):
        assert needed in entry, needed
    assert "已翻过" not in entry
    assert spec.index("- **v1.53") < spec.index("- **v1.52") < spec.index("- **v1.51")
    handoff = (ROOT / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    export = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert "`pixels:W3`" in export and "W3h" in export
    assert "W3h" in handoff.split("## 11. ", 1)[1]
    questions = (DISCUSSION / "08-class-pair-questions.md").read_text(encoding="utf-8")
    assert "## 2026-10-31 深夜 · 看图裁定与层序修正" in questions
    for row in ("| A6 |", "| A8 |", "| B8 |", "| B10 |", "| C11 |", "| D3 |", "| E1 |", "| E5 |"):
        line = next(l for l in questions.splitlines() if l.startswith(row))
        assert line.rstrip().rstrip("|").rstrip().split("|")[-1].strip(), row   # 你的回答 filled
    minutes = (DISCUSSION / "05-minutes.md").read_text(encoding="utf-8")
    assert "### 7.7" in minutes and "W3h" in minutes.split("### 7.7", 1)[1]
    yaml_text = (ROOT / "configs" / "covers_whitelist.yaml").read_text(encoding="utf-8")
    assert "version: W4" in yaml_text        # W3 (the card rulings) + W4 (the shroud over RAM / heatsink, cards Q08-Q25)
