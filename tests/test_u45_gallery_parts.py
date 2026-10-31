"""Task U45: the components gallery -- 「其余的都看不见」, gallery-only parts, cables.

The user (2026-10-30): every all-components shot shows a different subset of
the parts, and there are cables on the table too. So:

* 「其余的都看不见」 on a shot marks every card row with no shape and no 看不见
  yet as 看不见, in one op (one ``Ctrl+Z``), and leaves the drawn rows alone;
* 「＋ 这张里还有…」 adds **gallery-only** parts (``attrs.gallery_only``): every
  shot lists them, whatever their class, and nothing of the teardown sequence
  -- geometry, the card, the compiler (so ``missing_shape`` and ``Space``),
  ``cli check``, the graph, the VLM context -- ever sees one;
* the taxonomy has a class ``cable`` (24th, last), made only as a gallery-only
  part; the virtual ``cable:<owner>`` node keeps its own states, and the COCO
  ids of the 23 classes and of ``other`` do not move (``cable`` is 25).

The scene is :mod:`gallery_scene`: the real D13 sheet, three shots per gallery
view.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from tda import pipeline as P
from tda.core.export.coco import category_ids, export_coco, load_ctx
from tda.core.extra import next_ordinal
from tda.core.model import (
    COMPONENT_VIEWS,
    GALLERY_ONLY_ATTR,
    FrameKey,
    is_extra,
    is_gallery_only,
    teardown_only,
)
from tda.core.states import AFTER_LAST_STEP, needs_geom, state_at
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of, gather, instances_of
from tda.ui import app_actions as A
from tda.ui import session_api as api
from tda.ui.gallery import ABSENT
from tda.ui.session_api import SessionRefusal
from tda.ui.session_tasks import gallery_card, task_card_for

import gallery_scene as GS

VIEW = GS.VIEW


def _card(session) -> dict:
    return {str(r["instance"]): r for r in session.task_card()}


def _draw(session, instance: str) -> None:
    session.begin_edit(instance)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    session.commit_edit(api.SCOPE_KEYFRAME)


# --------------------------------------------------------------------------- #
# 其余的都看不见
# --------------------------------------------------------------------------- #
def test_the_rest_absent_marks_only_the_open_rows_in_one_op_and_undo_restores(tmp_path):
    session = GS.make_session(tmp_path)
    parts = list(_card(session))
    assert len(parts) >= 3
    drawn, absent, *rest = parts
    _draw(session, drawn)
    session.set_visibility(absent, ABSENT)          # 看不见 already, by hand
    depth = len(session.undo_stack.ops)
    assert set(session.rest_absent_targets()) == set(rest)
    result = session.mark_rest_absent()
    assert set(result["marked"]) == set(rest)
    assert len(session.undo_stack.ops) == depth + 1          # one op for all of them
    card = _card(session)
    assert all(card[p]["done"] and card[p].get("absent") for p in rest)
    assert card[drawn]["done"] and not card[drawn].get("absent")       # untouched
    key = session.current()
    overrides = session.db.frame_overrides(key)
    assert drawn not in overrides
    assert all(overrides[p].visibility == ABSENT and overrides[p].visible_rle is None
               for p in rest)
    # nothing left: a second press writes nothing and says so (the window)
    assert session.mark_rest_absent() == {"marked": []}
    assert len(session.undo_stack.ops) == depth + 1
    # one Ctrl+Z takes all of them back; the hand-set one stays
    assert session.undo()
    card = _card(session)
    assert all(not card[p]["done"] for p in rest)
    assert card[absent].get("absent") and card[drawn]["done"]
    assert all(p not in session.db.frame_overrides(key) for p in rest)
    assert session.redo()
    assert all(_card(session)[p].get("absent") for p in rest)


def test_the_rest_absent_is_the_gallerys_only(tmp_path):
    session = GS.make_session(tmp_path)
    session.open(GS.DESKTOP, "scan")
    assert session.rest_absent_targets() == []
    with pytest.raises(SessionRefusal, match="只在零件合照里用"):
        session.mark_rest_absent()


def test_the_rest_absent_refuses_while_one_of_them_is_being_drawn(tmp_path):
    session = GS.make_session(tmp_path)
    part = next(iter(_card(session)))
    session.begin_edit(part)
    session.set_editing_mask(GS.rect(4, 4, 20, 20))
    with pytest.raises(SessionRefusal, match="正在画"):
        session.mark_rest_absent()
    session.clear_edit()
    session.begin_edit(part)                        # an untouched layer is dropped
    assert part in session.mark_rest_absent()["marked"]
    assert not session.layer.active


# --------------------------------------------------------------------------- #
# gallery-only parts
# --------------------------------------------------------------------------- #
def test_added_cables_are_listed_on_every_shot_of_every_gallery_view(tmp_path):
    session = GS.make_session(tmp_path)
    depth = len(session.undo_stack.ops)
    keys = session.add_gallery_parts("cable", 2, {"kind": "sata"})
    assert keys == ["cable.sata.01", "cable.sata.02"]
    assert len(session.undo_stack.ops) == depth + 1
    stored = session.db.instances(GS.DESKTOP)
    for key in keys:
        rec = stored[key]
        assert rec.cls == "cable" and rec.attrs[GALLERY_ONLY_ATTR] is True
        assert is_gallery_only(rec) and is_extra(rec)
        assert rec.parent is None and not rec.attached and rec.raw_names == []
    # no hand-written event: it has no state, no step
    assert not [e for e in session.db.events(GS.DESKTOP) if e.target in keys]
    for view in COMPONENT_VIEWS:
        for shot in range(1, GS.SHOTS + 1):
            rows = {r["instance"]: r for r in gallery_card(session.db, session.tax,
                                                           GS.DESKTOP, view, shot)}
            assert set(keys) <= set(rows), (view, shot)
            assert all(rows[k]["gallery"] and not rows[k]["done"] for k in keys)
    # 看不见 works on one like on any part
    session.goto(2)
    session.set_visibility(keys[0], ABSENT)
    assert _card(session)[keys[0]]["done"]


def test_an_added_screw_follows_the_logs_ordinal_run_whatever_the_gallery_excludes(tmp_path):
    session = GS.make_session(tmp_path)
    table = instances_of(session.db, GS.DESKTOP)
    first = next_ordinal(table, "screw", {"role": "motherboard"})
    assert first > 1                    # the sheet names motherboard screws
    [key] = session.add_gallery_parts("screw", 1, {"role": "motherboard"})
    assert key == f"screw.motherboard.{first:02d}"
    # a log's screw is not on the card (GALLERY_EXCLUDED_CLASSES); the added one is
    card = _card(session)
    assert key in card
    assert not [k for k in card if k.startswith("screw.") and k != key]


def test_the_cable_key_rule_and_its_refusals(tmp_path):
    session = GS.make_session(tmp_path)
    assert session.add_gallery_parts("cable", 1) == ["cable.01"]
    assert session.add_gallery_parts("cable", 1, {"kind": "power"}) == ["cable.power.01"]
    assert session.add_gallery_parts("cable", 1) == ["cable.02"]
    with pytest.raises(SessionRefusal, match="kind"):
        session.add_gallery_parts("cable", 1, {"kind": "hdmi"})
    with pytest.raises(SessionRefusal):
        session.add_gallery_parts("chassis", 1)
    # a cable is never a part of the sequence: Steps refuses one without the flag
    from tda.ui.steps_model import EditError, StepTableData

    data = StepTableData.load(session.db, GS.DESKTOP, session.tax, "tester")
    with pytest.raises(EditError, match="只在零件合照里加"):
        data.add_extras("cable", 1)
    from tda.ui.panels.add_parts import class_choices, gallery_class_choices

    assert "cable" not in [c for _l, c in class_choices(data)]
    assert "cable" in [c for _l, c in gallery_class_choices(data)]
    # and nowhere but the gallery
    session.open(GS.DESKTOP, "scan")
    with pytest.raises(SessionRefusal, match="只在零件合照里用"):
        session.add_gallery_parts("cable", 1)


def test_the_sequence_never_sees_a_gallery_only_part(tmp_path):
    session = GS.make_session(tmp_path)
    keys = session.add_gallery_parts("cable", 1, {"kind": "fan"})
    keys += session.add_gallery_parts("ram_latch", 1)
    db, tax = session.db, session.tax
    instances = instances_of(db, GS.DESKTOP)
    events = events_of(db, tax, GS.DESKTOP)
    for step in (1, 2, 20, GS.LAST_STEP, AFTER_LAST_STEP):
        state = state_at(instances, events, step, tax)
        assert not set(keys) & set(state), step          # in no snapshot at all
        assert not set(keys) & set(needs_geom(instances, state, tax))
    for step in (1, GS.LAST_STEP):
        # the task card, and what the compiler is asked for (missing_shape, Space)
        card = task_card_for(db, tax, GS.DESKTOP, "scan", step)
        assert not set(keys) & {str(r.get("instance")) for r in card}
        assert not set(keys) & set(gather(db, tax, FrameKey(GS.DESKTOP, step, "scan")).needs)
    # the gallery's compiler is asked for them
    assert set(keys) <= set(gather(db, tax, FrameKey(GS.DESKTOP, 1, VIEW)).needs)
    # the VLM export's context: neither counted nor named
    ctx = load_ctx(db, tax, GS.DESKTOP, "scan")
    assert not set(keys) & set(ctx.instances) and set(keys) <= set(ctx.gallery_only)
    assert not set(keys) & set(ctx.state_at(1))
    # the constraint graph's table
    assert not set(keys) & set(teardown_only(instances))
    from tda.core.graph_derive import settled_instances

    assert not set(keys) & set(settled_instances(instances))


def test_ctrl_z_takes_them_back_until_one_is_drawn(tmp_path):
    session = GS.make_session(tmp_path)
    keys = session.add_gallery_parts("cable", 2)
    assert session.undo()
    assert not set(keys) & set(session.db.instances(GS.DESKTOP))
    assert not set(keys) & set(_card(session))
    assert session.redo()
    assert set(keys) <= set(session.db.instances(GS.DESKTOP))
    added = session.undo_stack.ops[-1]
    assert added.kind == "add_gallery_parts"
    _draw(session, keys[0])
    # with a shape of one on a shot the addition cannot be taken back: refused,
    # nothing written (the commit would have to be undone first)
    from tda.ui.session_gallery import apply_gallery_parts

    with pytest.raises(SessionRefusal, match="撤销不了"):
        apply_gallery_parts(session.db, session.truth, added.inverse, session.current().step)
    assert set(keys) <= set(session.db.instances(GS.DESKTOP))


def test_steps_mode_deletes_one_like_any_extra(tmp_path):
    from tda.ui.steps_model import StepTableData

    session = GS.make_session(tmp_path)
    [key] = session.add_gallery_parts("cable", 1)
    data = StepTableData.load(session.db, GS.DESKTOP, session.tax, "tester")
    assert key in data.instances
    assert not [i for i in data.issues if key in i]      # no orphan question about it
    data.delete_instance(session.db, key)
    assert key not in session.db.instances(GS.DESKTOP)


# --------------------------------------------------------------------------- #
# status, check, COCO
# --------------------------------------------------------------------------- #
def test_status_counts_them_under_the_gallery_only(tmp_path):
    from tda.cli import format_desktop

    session = GS.make_session(tmp_path)
    keys = session.add_gallery_parts("cable", 2)
    detail = format_desktop(session.db, P.status_rows(session.db)[0])
    head, _sep, gallery = detail.partition("零件合照 / components gallery")
    assert f"gallery-only parts (仅合照): 2 - {', '.join(keys)}" in gallery
    assert not [line for line in head.splitlines() if "cable." in line]


def test_cli_check_names_them_in_the_gallery_and_not_in_the_sequence(tmp_path, capsys):
    from tda.cli import main
    from tda.core.db import Db
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession

    env = GS.make_env(tmp_path)
    db = Db(env["cfg"]["db_path"])
    tax = load_taxonomy()
    session = AnnotationSession(db, tax, TruthService(db, tax), env["cfg"]["cache_dir"],
                                "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, VIEW)
    [key] = session.add_gallery_parts("cable", 1)
    session.close(force=True)
    db.close()
    capsys.readouterr()
    main(["--paths", env["paths"], "check", "--desktop", str(GS.DESKTOP), "--view", VIEW])
    out = capsys.readouterr().out
    assert f"missing_shape:{key}" in out and "1 gallery-only part(s)" in out
    main(["--paths", env["paths"], "check", "--desktop", str(GS.DESKTOP), "--view", "scan"])
    out = capsys.readouterr().out
    assert key not in out and "gallery-only" not in out


def test_coco_writes_the_cable_category_and_the_flag_on_the_gallery_only(tmp_path):
    session = GS.make_session(tmp_path)
    [key] = session.add_gallery_parts("cable", 1, {"kind": "sata"})
    _draw(session, key)
    session.mark_rest_absent()
    assert session.confirm_frame()
    doc = export_coco(session.db, session.tax, [GS.DESKTOP], VIEW,
                      str(tmp_path / "coco_gallery.json"), only_verified=True)
    [ann] = [a for a in doc["annotations"] if a["attributes"]["instance_key"] == key]
    assert ann["category_id"] == category_ids(session.tax)["cable"] == 25
    assert ann["attributes"]["gallery_only"] is True
    assert (ann["attributes"]["state"], ann["attributes"]["placement"]) == (
        "removed", "on_bench")
    names = {c["name"]: c["id"] for c in doc["categories"]}
    assert names["cable"] == 25 and names["connector"] == 23 and len(names) == 24
    # the sequence's file: no cable category, no gallery_only key anywhere
    seq = export_coco(session.db, session.tax, [GS.DESKTOP], "scan",
                      str(tmp_path / "coco_scan.json"), only_verified=False)
    assert "cable" not in {c["name"] for c in seq["categories"]}
    assert len(seq["categories"]) == 23
    assert not [a for a in seq["annotations"] if "gallery_only" in a["attributes"]]
    assert key not in json.dumps(seq)


# --------------------------------------------------------------------------- #
# the taxonomy and the keys
# --------------------------------------------------------------------------- #
def test_the_taxonomy_has_24_classes_cable_last_and_the_virtual_node_is_unchanged():
    tax = load_taxonomy()
    assert len(tax.classes) == 24 and list(tax.classes)[-1] == "cable"
    cable = tax.classes["cable"]
    assert cable["group"] == "interface"
    assert cable["attrs"]["kind"] == ["sata", "power", "fan", "front_panel", "other"]
    assert cable["states"] == ["installed", "removed"]
    assert cable["default_state"] == "installed"
    assert cable["needs_mask"] == {"installed": True, "removed": True}
    assert tax.group_of("cable") == "interface"
    # the name-keyed state queries are the virtual node's, as before (U33)
    assert tax.states_of("cable") == ["routed", "released", "removed"]
    assert tax.default_state("cable") == "routed"
    assert tax.apply_verb("cable", {}, "release") == ("state", "released")
    # the 23 classes keep their ids; other keeps 24
    ids = category_ids(tax)
    assert [ids[c] for c in list(tax.classes)[:23]] == list(range(1, 24))
    from tda.core.export.coco import other_category

    assert other_category(tax)["id"] == 24 and ids["cable"] == 25


def test_the_class_has_a_chinese_name_and_an_english_line():
    import importlib.util

    from tda.ui.class_names import class_zh

    assert class_zh("cable") == "线缆" and class_zh("cable", {"kind": "sata"}) == "SATA 线"
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_scanpack_u45",
                                                  root / "scripts" / "export_scanpack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "cable" in module.CLASS_EN
    assert set(load_taxonomy().classes) <= set(module.CLASS_EN)


def test_the_docs_say_it():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    gallery = spec.split("### 3.2b 零件合照", 1)[1].split("### 3.2c", 1)[0]
    for needed in ("「其余的都看不见」", "**哪一步都没有的零件：仅合照（v1.44，U45）**",
                   "「＋ 这张里还有…」", "`gallery_only: true`", "`attributes.gallery_only = true`",
                   "「只在零件合照里出现，拆解序列里不会要它；"):
        assert needed in gallery, needed
    classes = spec.split("### 6.1 类别", 1)[1].split("### 6.2", 1)[0]
    assert "（24 类" in classes and "| `cable`（v1.44，U45） |" in classes
    assert "`cable` 是 25" in classes and "`other` 保持 24" in classes
    changes = spec.split("## 14. 修订记录", 1)[1]
    entries = [line for line in changes.splitlines() if line.startswith("- ")]
    assert any(e.startswith("- **v1.44（2026-10-30）**：**零件合照：一键「其余的都看不见」")
               for e in entries)
    guide = (root / "docs" / "annotation_guide.md").read_text(encoding="utf-8")
    assert "**「其余的都看不见」**" in guide and "**「＋ 这张里还有…」**" in guide
    quick = (root / "docs" / "trial_quickstart.md").read_text(encoding="utf-8")
    section = quick.split("## 4.5 零件合照", 1)[1].split("\n## ", 1)[0]
    assert "**「其余的都看不见」**" in section and "**「＋ 这张里还有…」**" in section
    assert "两个按钮都没有快捷键" in section


def test_no_key_reaches_either_button():
    slots = {action.slot for action in A.ACTIONS}
    assert "act_mark_rest_absent" not in slots and "act_add_gallery_parts" not in slots


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    from PySide6.QtWidgets import QApplication

    from app_scene import StubSamQueue, close_window, make_paths
    from tda.core.truth import TruthService
    from tda.ui.app import MainWindow
    from tda.ui.session import AnnotationSession

    cfg = make_paths(tmp_path)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db, tax, cache_dir = GS.make_db(tmp_path, db_path=cfg["db_path"],
                                    cache_dir=Path(cfg["cache_dir"]))
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), "tester")
    session.sweeper_enabled = False
    session.open(GS.DESKTOP, VIEW)
    win = MainWindow(session, cfg, "tester", sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    yield win
    close_window(win)


def test_the_buttons_are_on_a_shots_card_only(window):
    from PySide6.QtWidgets import QApplication

    card = window.task_card
    assert card.gallery_buttons_visible()
    assert card.rest_absent_button.text() == "其余的都看不见"
    assert card.add_parts_button.text() == "＋ 这张里还有…"
    window.act_set_view("scan")
    QApplication.processEvents()
    assert not card.gallery_buttons_visible()


def test_the_rest_absent_button_says_how_many_and_then_that_none_are_left(window):
    from PySide6.QtWidgets import QApplication

    reported: list[str] = []
    window.report = reported.append
    open_rows = [r for r in window.task_card.rows() if not r["done"]]
    window.task_card.rest_absent_button.click()
    QApplication.processEvents()
    assert reported[-1].startswith(f"这张：{len(open_rows)} 个零件标为看不见")
    assert all(r["done"] for r in window.task_card.rows())
    window.task_card.rest_absent_button.click()
    QApplication.processEvents()
    assert "没有要标的" in reported[-1]


def test_the_add_button_makes_them_and_the_card_lists_them(window):
    from PySide6.QtWidgets import QApplication

    reported: list[str] = []
    window.report = reported.append
    seen: list = []

    def answer(data):
        seen.append(data)
        return {"cls": "cable", "count": 1, "attrs": {"kind": "front_panel"}}

    window.ask_gallery_parts = answer
    window.task_card.add_parts_button.click()
    QApplication.processEvents()
    assert seen and "cable.front_panel.01" in reported[-1]
    rows = {r["instance"]: r for r in window.task_card.rows()}
    assert "cable.front_panel.01" in rows
    assert window.task_card.current_instance() == "cable.front_panel.01"
    # the instance table marks it 仅合照
    _draw(window.session, "cable.front_panel.01")
    window.instances.refresh()
    row = window.instances.row_of("cable.front_panel.01")
    assert row and row.get("gallery_only")


def test_the_dialog_offers_cable_first_with_its_kind_and_no_parent(window):
    from tda.ui.panels.add_parts import GALLERY_INTRO, GalleryPartsDialog

    data = window.session.gallery_parts_data()
    dialog = GalleryPartsDialog(data, window)
    assert dialog.cls == "cable" and dialog.disc_combo.isVisibleTo(dialog)
    assert dialog.keys() == ["cable.sata.01"]
    assert dialog.values() == {"cls": "cable", "count": 1, "attrs": {"kind": "sata"}}
    assert GALLERY_INTRO == ("只在零件合照里出现，拆解序列里不会要它；"
                             "日志里本来就有的零件请在步骤表里加（可填拆下于第几步）")
    assert not hasattr(dialog, "parent_combo") and not hasattr(dialog, "removed_spin")
    dialog.close()
