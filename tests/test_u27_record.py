"""Task U27: every relational field an Apply in Steps changes is one ``relation_fix`` op.

``StepTableData.save`` compares the instance table it writes with the stored
one and logs one row per changed field -- ``{desktop, key, raw_name, cls,
field, old, new, by, at}`` (plus the named part's raw name), the inverse the
same record swapped -- for ``parent``, ``attached``, ``mounted_on``,
``fastens``, ``socket_host``, ``cable``, ``slot_id``, ``group_id``,
``group_order``, ``removal_direction`` and a latch's ``of``. A field set back
to the importer's value is recorded too. The scene is the real D2 sheet.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from relation_scenes import D2, apply_steps, fixes, imported, reimport, replayed
from tda.core.model import FrameKey
from tda.core.relation_replay import (
    FIELDS,
    OP_RELATION_FIX,
    OP_VIEW,
    fix_rows,
    relation_changes,
)
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.ui import steps_merge
from tda.ui.session_captive import apply_captive_fix, commit_captive_fix
from tda.ui.session_verb import commit_verb_fix
from tda.ui.steps_model import StepTableData

KEY = FrameKey(D2, 48, "scan")
FAN = "screw.cpu_cooler.01"          # captive by the guess: parent = the fan, attached
BOARD = "screw.motherboard.01"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def db(tmp_path, tax):
    d = imported(tmp_path, tax, D2)
    yield d
    d.close()


#: One edit per field of the instance table, all different from the import.
EDITS = (
    ("psu_latch.01", "parent", "psu.01", None),
    ("psu_latch.01", "attached", True, False),
    ("storage_drive.ssd.01", "mounted_on", "drive_cage.01", None),
    (BOARD, "fastens", "chassis", "motherboard.01"),
    ("connector.03", "socket_host", "motherboard.01", "storage_drive.ssd.01"),
    ("connector.03", "cable", "cable:sata", "cable:storage_drive"),
    ("ram_module.01", "slot_id", "DIMM1", None),
    (BOARD, "group_id", "board", None),
    (BOARD, "group_order", "sequential", "unordered"),
    ("psu.01", "removal_direction", "+Z", None),
)


def test_every_field_of_the_instance_table_is_one_op(db, tax):
    apply_steps(db, tax, D2, *((key, name, new) for key, name, new, _old in EDITS))
    got = {(key, name): (old, new, by) for key, name, old, new, by in fixes(db, D2)}
    assert got == {(key, name): (old, new, "tester") for key, name, new, old in EDITS}
    ops = db.ops_of_kind(D2, OP_RELATION_FIX)
    assert {op["view"] for op in ops} == {OP_VIEW} and {op["annotator"] for op in ops} == \
        {"tester"}
    parent = next(op for op in ops if op["payload"]["field"] == "parent")
    p, inv = parent["payload"], parent["inverse"]
    assert (p["desktop"], p["key"], p["raw_name"], p["cls"]) == \
        (D2, "psu_latch.01", "Power Module Blocker", "psu_latch")
    assert (p["old"], p["new"], p["old_raw_name"], p["new_raw_name"]) == \
        (None, "psu.01", None, "Power Module")
    assert p["at"] and p["by"] == "tester"
    # the inverse is the same record, swapped
    assert (inv["old"], inv["new"], inv["old_raw_name"], inv["new_raw_name"]) == \
        ("psu.01", None, "Power Module", None)
    assert {k: v for k, v in inv.items() if "old" not in k and "new" not in k} == \
        {k: v for k, v in p.items() if "old" not in k and "new" not in k}
    # a field that names no part carries no raw names of a value
    slot = next(op for op in ops if op["payload"]["field"] == "slot_id")["payload"]
    assert "new_raw_name" not in slot


def test_an_of_changed_in_the_session_is_recorded_too(db, tax):
    """``attrs["of"]`` is no column of the table, but a save that changes it says so."""
    data = StepTableData.load(db, D2, tax, "tester")
    data.instances["ram_latch.01"].attrs["of"] = "ram_module.02"
    data.save(db)
    assert fixes(db, D2) == [("ram_latch.01", "of", "ram_module.01", "ram_module.02", "tester")]


def test_set_back_to_the_importers_value_is_recorded_and_replayed(db, tmp_path, tax):
    """The annotator's "no parent" must survive the heuristic that would fill it."""
    apply_steps(db, tax, D2, (FAN, "parent", ""), (FAN, "attached", False))
    assert fixes(db, D2) == [(FAN, "parent", "cpu_cooler.fan.01", None, "tester"),
                             (FAN, "attached", True, False, "tester")]
    run = reimport(db, tmp_path, tax, D2)
    fan = db.instances(D2)[FAN]
    assert (fan.parent, fan.attached) == (None, False)
    assert [(e["key"], e["field"], e["value"], e["sheet"]) for e in replayed(db, D2)["kept"]] \
        == [(FAN, "parent", None, "cpu_cooler.fan.01"), (FAN, "attached", False, True)]
    # the guess the fix overrode is no question for S1 any more
    assert f"{FAN}.parent = cpu_cooler.fan.01" not in run.fills
    assert "screw.cpu_cooler.02.parent = cpu_cooler.fan.01" in run.fills
    # ... and set back to the guess in a later Apply: the latest word is the guess
    apply_steps(db, tax, D2, (FAN, "parent", "cpu_cooler.fan.01"), (FAN, "attached", True))
    assert len(fixes(db, D2)) == 4
    reimport(db, tmp_path, tax, D2)
    fan = db.instances(D2)[FAN]
    assert (fan.parent, fan.attached) == ("cpu_cooler.fan.01", True)
    done = replayed(db, D2)
    assert done["kept"] == [] and [(e["field"], e["value"]) for e in done["agrees"]] == \
        [("parent", "cpu_cooler.fan.01"), ("attached", True)]


def test_an_apply_that_changes_no_field_records_nothing(db, tax):
    data = StepTableData.load(db, D2, tax, "tester")
    data.apply_edit(2, "difficulty", 3)                 # a step edit, no instance field
    data.apply_instance_edit("psu_latch.01", "parent", "psu.01")
    data.apply_instance_edit("psu_latch.01", "parent", "")   # and back before Apply
    data.save(db)
    assert fixes(db, D2) == []
    StepTableData.load(db, D2, tax).save(db)
    assert fixes(db, D2) == []


def test_a_staged_part_is_not_a_fix_until_it_is_stored(db, tax):
    data = StepTableData.load(db, D2, tax, "tester")
    [added] = data.add_extras("cable_clip", 1, parent="psu.01")
    data.save(db)
    assert fixes(db, D2) == []                          # its record is the add op's
    apply_steps(db, tax, D2, (added, "parent", "cpu_cooler.fan.01"))
    assert fixes(db, D2) == [(added, "parent", "psu.01", "cpu_cooler.fan.01", "tester")]


def test_apply_and_merge_records_the_staged_fields(db, tax):
    """「Apply 并合并」 saves the staged table first: its fields are fixes like any Apply."""
    data = StepTableData.load(db, D2, tax, "tester")
    data.apply_instance_edit("psu_latch.01", "parent", "psu.01")
    steps_merge.merge(data, db, "cover.02", "cover.01", with_staged=True)
    assert fixes(db, D2) == [("psu_latch.01", "parent", None, "psu.01", "tester")]


def test_a_verb_correction_writes_no_fix(db, tax):
    commit_verb_fix(db, TruthService(db, tax), KEY, "cover.01", "remove", "tester")
    assert fixes(db, D2) == []


def test_a_captive_flip_records_the_fields_it_moves_and_its_undo_moves_them_back(db, tax):
    """「其实留在零件里」 fills parent and ticks attached through the same save (U14)."""
    truth = TruthService(db, tax)
    result = commit_captive_fix(db, truth, KEY, BOARD, None, "tester")
    assert fixes(db, D2) == [(BOARD, "parent", None, "motherboard.01", "tester"),
                             (BOARD, "attached", False, True, "tester")]
    apply_captive_fix(db, truth, result["op"].inverse)         # Ctrl+Z
    assert fixes(db, D2)[2:] == [(BOARD, "parent", "motherboard.01", None, "tester"),
                                 (BOARD, "attached", True, False, "tester")]


def test_the_pure_diff_skips_drafts_and_context_objects_and_compares_normalised():
    from tda.core.model import CONTEXT_ATTR, InstanceRec

    def rec(key, cls="cover", **kw):
        return InstanceRec(key, 1, cls, **kw)

    before = {"cover.01": rec("cover.01", parent="", group_order=""),
              "ls:cover.9": rec("ls:cover.9"),
              "ctx.cover.01": rec("ctx.cover.01", attrs={CONTEXT_ATTR: True})}
    after = {"cover.01": rec("cover.01", parent=None, group_order="unordered", slot_id=" "),
             "ls:cover.9": rec("ls:cover.9", parent="psu.01"),
             "ctx.cover.01": rec("ctx.cover.01", attrs={CONTEXT_ATTR: True}, parent="psu.01")}
    assert relation_changes(before, after) == []        # blank is blank, however written
    after["cover.01"].attrs["of"] = "cpu_cooler"
    assert relation_changes(before, after) == [("cover.01", "of", None, "cpu_cooler")]
    assert FIELDS[-1] == "of"


def test_fix_rows_is_symmetric():
    from tda.core.model import InstanceRec

    clip = InstanceRec("cable_clip.01", 23, "cable_clip", raw_names=["remove the cable"])
    shroud = InstanceRec("cover.01", 23, "cover", raw_names=["CPU fan cover"])
    payload, inverse = fix_rows(23, clip, "parent", None, "cover.01", "chang", "t",
                                {"cover.01": shroud})
    assert payload["new_raw_name"] == "CPU fan cover" and payload["old_raw_name"] is None
    assert fix_rows(23, clip, "parent", "cover.01", None, "chang", "t",
                    {"cover.01": shroud}) == (inverse, payload)
