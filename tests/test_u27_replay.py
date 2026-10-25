"""Task U27: ``import-logs --force`` replays the relational fields set in Steps.

After the merges, reclasses, ``Shift+T`` verbs and captive flags are replayed
and the heuristic has filled the blanks, the latest ``relation_fix`` of every
``(key, field)`` goes over the importer's value -- the part found by key
(merges and later reclasses followed) or by raw name, the part it names found
the same way and still a valid reference -- and one ``replay_relations`` row
says what was kept and what no longer fits. The import scenes are the real D2
and D23 sheets; the rest is the pure function.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from relation_scenes import (
    CLIP,
    D2,
    D23,
    SHROUD,
    apply_steps,
    fixes,
    imported,
    reimport,
    replayed,
)
from tda.core.model import InstanceRec
from tda.core.relation_replay import OP_REPLAY_RELATIONS, replay_relations
from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


# --------------------------------------------------------------------------- #
# the pure replay
# --------------------------------------------------------------------------- #
def part(key: str, *names: str, **kw) -> InstanceRec:
    return InstanceRec(key, 1, kw.pop("cls", key.split(".")[0]), raw_names=list(names), **kw)


def fix(op_id: int, key: str, name: str, new, raw=None, new_raw=None, old=None) -> dict:
    payload = {"key": key, "field": name, "old": old, "new": new, "raw_name": raw,
               "by": "tester", "at": "t"}
    if new_raw is not None:
        payload["new_raw_name"] = new_raw
    return {"id": op_id, "payload": payload}


def table() -> dict[str, InstanceRec]:
    return {"cable_clip.01": part("cable_clip.01", "clip"),
            "cover.01": part("cover.01", "CPU fan cover"),
            "cover.02": part("cover.02", "Screw cover"),
            "screw.motherboard.01": part("screw.motherboard.01", "MB screw 1", cls="screw",
                                         fastens="motherboard.01"),
            "motherboard.01": part("motherboard.01", "Motherboard")}


def test_the_latest_fix_wins_whatever_order_the_rows_come_in():
    ops = [fix(1, "cable_clip.01", "parent", "cover.02", "clip", "Screw cover"),
           fix(4, "cable_clip.01", "parent", "cover.01", "clip", "CPU fan cover"),
           fix(2, "cable_clip.01", "attached", True, "clip")]
    out, report = replay_relations(ops, table())
    again, second = replay_relations(list(reversed(ops)), table())
    assert (out["cable_clip.01"].parent, out["cable_clip.01"].attached) == ("cover.01", True)
    assert out == again and report.payload() == second.payload()
    assert [(e["field"], e["ops"]) for e in report.kept] == [("parent", [1, 4]),
                                                            ("attached", [2])]
    assert report.inverse == [{"key": "cable_clip.01", "field": "parent", "value": None},
                              {"key": "cable_clip.01", "field": "attached", "value": False}]
    assert table()["cable_clip.01"].parent is None           # the input is left alone
    # replaying onto its own result changes nothing more: idempotent
    third, done = replay_relations(ops, out)
    assert third == out and done.kept == [] and len(done.agrees) == 2


def test_a_merged_value_follows_the_merge_and_a_merged_away_part_keeps_nothing():
    """合并到… re-points every pointer at the part merged away; its own fields go with it."""
    fresh = table()
    del fresh["cover.02"]                                     # merged into cover.01
    ops = [fix(1, "cover.02", "removal_direction", "+Z", "Screw cover"),
           fix(2, "cable_clip.01", "parent", "cover.02", "clip", "Screw cover")]
    out, report = replay_relations(ops, fresh, rename={"cover.02": "cover.01"})
    assert out["cable_clip.01"].parent == "cover.01"
    assert out["cover.01"].removal_direction is None          # cover.01 kept its own record
    assert [(e["key"], e["value"]) for e in report.kept] == [("cable_clip.01", "cover.01")]
    assert report.unmatched_lines() == [
        "relation fix not matched: cover.02.removal_direction = +Z (merged into cover.01, "
        "which kept its own fields)"]


def test_a_reclass_is_followed_only_by_the_fixes_made_before_it():
    fresh = {"drive_cage.01": part("drive_cage.01", "HDD Locker Shield"),
             "drive_latch.02": part("drive_latch.02", "a latch made later"),
             "drive_latch.01": part("drive_latch.01", "HDD Locker")}
    renames = [(5, "drive_latch.02", "drive_cage.01")]
    ops = [fix(3, "drive_latch.02", "removal_direction", "+Z", "HDD Locker Shield"),
           fix(9, "drive_latch.02", "removal_direction", "-X", "a latch made later"),
           # the reclass rule clears a bay part's parent naming the other bay class
           fix(4, "drive_latch.02", "parent", "drive_latch.01", "HDD Locker Shield",
               "HDD Locker")]
    out, report = replay_relations(ops, fresh, reclassed=renames)
    assert out["drive_cage.01"].removal_direction == "+Z"
    assert out["drive_latch.02"].removal_direction == "-X"
    assert out["drive_cage.01"].parent is None
    assert report.unmatched[0]["why"] == \
        "the reclass of drive_latch.02 to drive_cage.01 cleared it"


def test_a_renumbered_sheet_moves_the_fix_and_the_part_it_names_with_their_rows():
    fresh = {"screw.motherboard.04": part("screw.motherboard.04", "MB screw 5", cls="screw"),
             "screw.motherboard.05": part("screw.motherboard.05", "MB screw 6", cls="screw"),
             "cover.01": part("cover.01", "Screw cover"),
             "cable_clip.01": part("cable_clip.01", "clip")}
    ops = [fix(1, "screw.motherboard.05", "group_id", "left", "MB screw 5"),
           fix(2, "cable_clip.01", "parent", "cover.02", "clip", "Screw cover")]
    out, report = replay_relations(ops, fresh)
    assert out["screw.motherboard.04"].group_id == "left"
    assert out["screw.motherboard.05"].group_id is None
    assert out["cable_clip.01"].parent == "cover.01"
    line = report.kept_line()
    assert "cable_clip.01.parent = cover.01 (was cover.02, found by its name)" in line
    assert "screw.motherboard.04.group_id (was screw.motherboard.05, found by its name) = " \
           "left" in line


def test_a_renamed_row_keeps_its_key_but_rows_that_shifted_under_it_do_not():
    stored = table()
    renamed = table()
    renamed["cover.01"] = part("cover.01", "Fan cover")             # the row renamed
    ops = [fix(1, "cable_clip.01", "parent", "cover.01", "clip", "CPU fan cover")]
    out, report = replay_relations(ops, renamed, stored=stored)
    assert out["cable_clip.01"].parent == "cover.01" and report.unmatched == []
    shifted = table()
    shifted["cover.01"] = part("cover.01", "Screw cover")           # row 3 is no cover now
    del shifted["cover.02"]
    out, report = replay_relations(ops, shifted, stored=stored)
    assert out["cable_clip.01"].parent is None
    [miss] = report.unmatched
    assert miss["why"] == ('it names cover.01: the sheet\'s cover.01 is "Screw cover" now, '
                           'which was cover.02 before; nothing is named "CPU fan cover" '
                           'any more')


@pytest.mark.parametrize("op, why", [
    (fix(1, "psu.01", "parent", "cover.01", "Power"),
     'the sheet no longer has psu.01 nor a part named "Power"'),
    (fix(1, "cable_clip.01", "parent", "psu.01", "clip", "Power"),
     'it names psu.01: the sheet no longer has psu.01 nor a part named "Power"'),
    (fix(1, "cable_clip.01", "fastens", "cover.01", "clip", "CPU fan cover"),
     "only a screw carries fastens; the sheet makes cable_clip.01 a cable_clip"),
    (fix(1, "cover.01", "parent", "cover.01", "CPU fan cover", "CPU fan cover"),
     "cover.01 cannot name itself"),
    (fix(1, "motherboard.01", "parent", "screw.motherboard.01", "Motherboard", "MB screw 1"),
     "screw.motherboard.01 is already below motherboard.01; the parent would form a loop"),
    (fix(1, "cover.09", "group_id", "x", "clip"),
     'the sheet now makes "clip" a cable_clip (cable_clip.01), not a cover'),
])
def test_what_does_not_fit_is_reported_and_not_applied(op, why):
    fresh = table()
    fresh["screw.motherboard.01"].parent = "motherboard.01"
    out, report = replay_relations([op], fresh)
    [miss] = report.unmatched
    assert miss["why"] == why
    assert out == fresh and report.kept == [] and report.inverse == []
    assert report.unmatched_lines()[0].startswith("relation fix not matched: ")


def test_several_parts_with_the_name_is_no_match_by_name():
    fresh = {"screw.01": part("screw.01", "screw", cls="screw"),
             "screw.02": part("screw.02", "screw", cls="screw")}
    out, report = replay_relations([fix(1, "screw.07", "group_id", "g", "screw")], fresh)
    assert 'several parts of the sheet are named "screw": screw.01, screw.02' in \
        report.unmatched[0]["why"]
    # but the key itself still carries it: by key
    out, report = replay_relations([fix(1, "screw.02", "group_id", "g", "screw")], fresh)
    assert out["screw.02"].group_id == "g"


def test_a_part_the_import_keeps_as_it_is_is_not_rewritten_and_may_be_named():
    added = part("ram_latch.05", cls="ram_latch", attrs={"added_by": "tester"},
                 parent="motherboard.01")
    ops = [fix(1, "ram_latch.05", "parent", "motherboard.01"),
           fix(2, "cable_clip.01", "parent", "ram_latch.05", "clip")]
    out, report = replay_relations(ops, table(), kept={"ram_latch.05": added})
    assert "ram_latch.05" not in out
    assert out["cable_clip.01"].parent == "ram_latch.05"
    [kept_as_is] = report.agrees
    assert kept_as_is["kept_as_is"] == "added in S1"
    assert "ram_latch.05.parent = motherboard.01 (kept as it is: added in S1)" in \
        report.kept_line()


def test_of_is_a_reference_only_when_it_holds_a_key():
    fresh = table() | {"ram_latch.01": part("ram_latch.01", "RAM clip 1",
                                            attrs={"of": "ram_module.01"})}
    ops = [fix(1, "ram_latch.01", "of", "ram_module.02", "RAM clip 1"),
           fix(2, "cover.01", "of", "cpu_cooler", "CPU fan cover")]
    out, report = replay_relations(ops, fresh)
    assert out["cover.01"].attrs["of"] == "cpu_cooler"
    assert out["ram_latch.01"].attrs["of"] == "ram_module.01"         # no such module
    assert "it names ram_module.02" in report.unmatched[0]["why"]


def test_unfilled_drops_the_guesses_a_fix_overrode():
    fresh = table()
    out, report = replay_relations(
        [fix(1, "screw.motherboard.01", "fastens", "cover.02", "MB screw 1", "Screw cover")],
        fresh)
    fills = ["screw.motherboard.01.fastens = motherboard.01", "x.parent = y"]
    assert report.unfilled(fills) == ["x.parent = y"]


# --------------------------------------------------------------------------- #
# the import
# --------------------------------------------------------------------------- #
@pytest.fixture
def d23(tmp_path, tax):
    db = imported(tmp_path, tax, D23)
    apply_steps(db, tax, D23, (CLIP, "parent", SHROUD), (CLIP, "attached", True))
    yield db
    db.close()


def test_one_replay_row_per_import_and_its_lines(d23, tmp_path, tax):
    printed: list[str] = []
    run = reimport(d23, tmp_path, tax, D23, log=printed)
    clip = d23.instances(D23)[CLIP]
    assert (clip.parent, clip.attached) == (SHROUD, True)
    line = ("relation fixes kept: cable_clip.01.parent = cover.01 (sheet: none), "
            "cable_clip.01.attached = True (sheet: False)")
    assert line in run.corrections and any(text.endswith(line) for text in printed)
    [op] = d23.ops_of_kind(D23, OP_REPLAY_RELATIONS)
    assert op["inverse"] == {"instances": [
        {"key": CLIP, "field": "parent", "value": None},
        {"key": CLIP, "field": "attached", "value": False}]}
    reimport(d23, tmp_path, tax, D23)
    assert len(d23.ops_of_kind(D23, OP_REPLAY_RELATIONS)) == 2
    assert replayed(d23, D23) == op["payload"]


def test_no_fix_no_replay_row(tmp_path, tax):
    db = imported(tmp_path, tax, D23)
    run = reimport(db, tmp_path, tax, D23)
    assert db.ops_of_kind(D23, OP_REPLAY_RELATIONS) == []
    assert not any("relation fix" in line for line in run.corrections)
    db.close()


def test_a_shroud_the_sheet_no_longer_has_is_not_matched(d23, tmp_path, tax):
    """Row 3 read as the fan's locker: cover.01 is the screw cover now, not the shroud."""
    run = reimport(d23, tmp_path, tax, D23, edits=[("3,CPU fan cover", "3,CPU fan locker")])
    clip = d23.instances(D23)[CLIP]
    assert clip.parent is None                            # not the screw cover
    assert clip.attached is False                         # round 2, N5: nothing to ride on
    assert [(m["key"], m["field"]) for m in replayed(d23, D23)["unmatched"]] == [
        (CLIP, "parent"), (CLIP, "attached")]
    assert replayed(d23, D23)["kept"] == []
    assert ("relation fix not matched: cable_clip.01.parent = cover.01 (it names cover.01: "
            "the sheet's cover.01 is \"Screw cover\" now, which was cover.02 before; nothing "
            "is named \"CPU fan cover\" any more)") in run.corrections
    assert ("relation fix not matched: cable_clip.01.attached = True (not applied (no "
            "parent): the parent fix of this part was not matched)") in run.corrections


def test_renamed_rows_keep_their_fixes(d23, tmp_path, tax):
    reimport(d23, tmp_path, tax, D23, edits=[
        ("3,CPU fan cover", "3,Fan cover"),
        ("2,remove the cable from the case fan clip", "2,remove the cable from the fan clip")])
    clip = d23.instances(D23)[CLIP]
    assert (clip.parent, clip.attached) == (SHROUD, True)
    assert replayed(d23, D23)["unmatched"] == []


def test_a_renumbered_sheet_finds_the_screw_by_its_name(tmp_path, tax):
    db = imported(tmp_path, tax, D2)
    apply_steps(db, tax, D2, ("screw.motherboard.05", "group_id", "left-rear"))
    assert fixes(db, D2) == [("screw.motherboard.05", "group_id", None, "left-rear", "tester")]
    run = reimport(db, tmp_path, tax, D2,
                   edits=[("39,Motherboard screw 1,", "39,Optical drive screw 1,")])
    table = db.instances(D2)
    assert table["screw.motherboard.04"].raw_names == ["Motherboard screw 5"]
    assert table["screw.motherboard.04"].group_id == "left-rear"
    assert "screw.motherboard.05" not in table or table["screw.motherboard.05"].group_id is None
    assert any("screw.motherboard.04.group_id (was screw.motherboard.05, found by its name)"
               in line for line in run.corrections)
    db.close()
