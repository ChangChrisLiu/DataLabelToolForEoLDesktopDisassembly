"""Task U27 round 3: after a renumbering, the latest fix of a part wins -- whatever key it has.

The re-review's blocker (B2): a part edited before a renumbering has fixes
under its old key, edited after it under its new one; the replay grouped by
the recorded key and applied the groups in key order, so the older value could
win. Now every op is first resolved to the part it names in this import and
only the op with the highest id counts (the others are "superseded"); the
captive replay de-duplicates the same way. With the smaller rulings: R2-N1 (a
flip recorded without a raw name keeps its screw through a second renumbered
re-import), R2-N2 ("rebuilt as" on every import), R2-N4 (the ``Captive`` cell
fills ``parent`` / ``attached`` as ``Shift+T`` does). All on the real D2 / D5
sheets.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

from relation_scenes import D2, apply_steps, imported, reimport, replayed
from tda.core.captive_replay import OP_CAPTIVE_FIX, OP_REPLAY_CAPTIVE
from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.relation_replay import OP_RELATION_FIX
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import state_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui.session_captive import commit_captive_fix

KEY = FrameKey(D2, 48, "scan")
#: Row 18 "RAM clip 1" read as nothing: every later RAM clip moves one key down.
RAM_RENUMBER = [("18,RAM clip 1,", "18,Front fan,")]
#: Row 3 "CPU fan screw 1" read as an optical drive screw: the fan screws move down.
FAN_RENUMBER = [("3,CPU fan screw 1,", "3,Optical drive screw 9,")]
LAST = 48


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def d2(tmp_path, tax):
    db = imported(tmp_path, tax, D2)
    yield db
    db.close()


def latch(db: Db, key: str) -> tuple:
    rec = db.instances(D2)[key]
    return rec.raw_names[0], rec.removal_direction, rec.group_order


def fan_screws(db: Db, tax) -> dict[str, tuple]:
    """``raw name -> (captive, source, attached, [state per step])`` of the CPU fan screws."""
    table = db.instances(D2)
    states = [state_of(db, tax, D2, step) for step in range(1, LAST + 1)]
    return {rec.raw_names[0]: ((rec.attrs or {}).get("captive"),
                               (rec.attrs or {}).get("captive_source"), rec.attached,
                               [(s[key].state, s[key].placement) for s in states])
            for key, rec in table.items()
            if rec.cls == "screw" and rec.raw_names
            and rec.raw_names[0].startswith("CPU fan screw")}


def loosened_at_the_end(db: Db, tax) -> list[str]:
    held = state_of(db, tax, D2, LAST)
    return sorted(k for k, r in db.instances(D2).items()
                  if r.cls == "screw" and held[k].state == "loosened")


# --------------------------------------------------------------------------- #
# B2: the reviewer's p15
# --------------------------------------------------------------------------- #
def test_the_latest_fix_wins_after_a_renumbering(d2, tmp_path, tax):
    apply_steps(d2, tax, D2, ("ram_latch.03", "removal_direction", "+Z"),
                ("ram_latch.03", "group_order", "sequential"))
    reimport(d2, tmp_path, tax, D2, edits=RAM_RENUMBER)
    assert latch(d2, "ram_latch.02") == ("RAM clip 3", "+Z", "sequential")
    # the annotator changes it again, under its new key
    apply_steps(d2, tax, D2, ("ram_latch.02", "removal_direction", "+X"),
                ("ram_latch.02", "group_order", "unordered"))
    assert [(op["payload"]["key"], op["payload"]["field"])
            for op in d2.ops_of_kind(D2, OP_RELATION_FIX)] == [
        ("ram_latch.03", "group_order"), ("ram_latch.03", "removal_direction"),
        ("ram_latch.02", "group_order"), ("ram_latch.02", "removal_direction")]
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2, edits=RAM_RENUMBER)
        assert latch(d2, "ram_latch.02") == ("RAM clip 3", "+X", "unordered")
        done = replayed(d2, D2)
        assert [(e["key"], e["field"], e["value"]) for e in done["kept"]] == [
            ("ram_latch.02", "removal_direction", "+X")]
        assert [(e["key"], e["field"], e["value"]) for e in done["agrees"]] == [
            ("ram_latch.02", "group_order", "unordered")]
        assert done["unmatched"] == []
        assert sorted((e["key"], e["field"]) for e in done["superseded"]) == [
            ("ram_latch.03", "group_order"), ("ram_latch.03", "removal_direction")]
        assert ("relation fixes superseded: 2 (made under an earlier key of the same part, "
                "overridden by a later fix of that field)") in run.corrections
        assert not any("+Z" in line or "sequential" in line for line in run.corrections)
    # back to the original sheet: the same part, the same latest value
    reimport(d2, tmp_path, tax, D2)
    assert latch(d2, "ram_latch.03") == ("RAM clip 3", "+X", "unordered")


# --------------------------------------------------------------------------- #
# the captive replay de-duplicates too (R2-N3)
# --------------------------------------------------------------------------- #
def test_a_screw_flipped_under_two_keys_is_one_screw(d2, tmp_path, tax):
    truth = TruthService(d2, tax)
    commit_captive_fix(d2, truth, KEY, "screw.cpu_cooler.02", None, "tester")    # out
    reimport(d2, tmp_path, tax, D2, edits=FAN_RENUMBER)          # "screw 2" is .01 now
    commit_captive_fix(d2, TruthService(d2, tax), KEY, "screw.cpu_cooler.01", None,
                       "tester")                                                # back in
    before = fan_screws(d2, tax)
    assert before["CPU fan screw 2"][:3] == (True, "manual", True)
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2, edits=FAN_RENUMBER)
        assert fan_screws(d2, tax) == before
        [line] = [x for x in run.corrections if x.startswith("captive fixes (")]
        assert line.count("screw.cpu_cooler.01") == 1
        assert "captive fixes superseded: 1 (the same screw under an earlier key)" in \
            run.corrections
        assert loosened_at_the_end(d2, tax) == []


# --------------------------------------------------------------------------- #
# R2-N1: a flip recorded before round 2 (no raw name), two renumbered re-imports
# --------------------------------------------------------------------------- #
def _strip(db: Db, kind: str, *names: str) -> None:
    """Make the op rows of ``kind`` look as they did before the ``names`` were recorded."""
    with db.transaction():
        for row in db.conn.execute("SELECT id, payload_json FROM op_log WHERE desktop=? AND "
                                   "kind=?", (D2, kind)).fetchall():
            payload = json.loads(row["payload_json"])
            for name in names:
                payload.pop(name, None)
            db.conn.execute("UPDATE op_log SET payload_json=? WHERE id=?",
                            (json.dumps(payload), row["id"]))


def test_a_flip_without_a_raw_name_keeps_its_screw_through_two_renumbered_imports(
        d2, tmp_path, tax):
    commit_captive_fix(d2, TruthService(d2, tax), KEY, "screw.cpu_cooler.02", None, "tester")
    _strip(d2, OP_CAPTIVE_FIX, "raw_name")                      # as U14 wrote it
    _strip(d2, OP_RELATION_FIX, "captive")                      # as round 1 wrote it
    before = fan_screws(d2, tax)
    for n in range(2):
        reimport(d2, tmp_path, tax, D2, edits=FAN_RENUMBER)
        after = fan_screws(d2, tax)
        assert after["CPU fan screw 2"][0] is False, n           # the flip stays on its screw
        assert {k: v[3] for k, v in after.items() if k != "CPU fan screw 1"} == \
            {k: v[3] for k, v in before.items() if k != "CPU fan screw 1"}
        assert loosened_at_the_end(d2, tax) == []
    # the first import found the screw by its key and remembered the name
    first = d2.ops_of_kind(D2, OP_REPLAY_CAPTIVE)[0]["payload"]
    assert [e["name"] for e in first["kept"]] == ["CPU fan screw 2"]


# --------------------------------------------------------------------------- #
# R2-N2: "rebuilt as" on every import
# --------------------------------------------------------------------------- #
def test_rebuilt_as_is_said_on_every_import(tmp_path, tax):
    from relation_scenes import sheets
    edits = [("10,HDD shield locker", "10,Front fan")]
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path, 5, edits), tax, desktops={5})
    apply_steps(db, tax, 5, ("drive_latch.01", "parent", ""),
                ("drive_latch.01", "removal_direction", "+Z"))
    for _ in range(3):
        run = reimport(db, tmp_path, tax, 5, edits=edits)
        lines = [x for x in run.corrections if x.startswith("relation fix not matched")]
        assert len(lines) == 2 and all("(rebuilt as drive_latch.02: " in x for x in lines)
    db.close()


# --------------------------------------------------------------------------- #
# R2-N4: the Captive cell fills parent / attached as Shift+T does
# --------------------------------------------------------------------------- #
def test_ticking_captive_in_steps_rides_the_screw_out_with_its_part(d2, tmp_path, tax):
    board = "screw.motherboard.01"
    apply_steps(d2, tax, D2, (board, "captive", True))
    rec = d2.instances(D2)[board]
    assert (rec.attrs["captive"], rec.parent, rec.attached) == (True, "motherboard.01", True)
    states = [state_of(d2, tax, D2, s)[board].state for s in range(38, LAST + 1)]
    assert states[:2] == ["fastened", "loosened"] and "loosened" in states
    assert states[-1] == "removed"                       # out with the board at 45
    reimport(d2, tmp_path, tax, D2)
    assert [state_of(d2, tax, D2, s)[board].state for s in range(38, LAST + 1)] == states
    # and unticked: attached goes, parent stays (「其实拧下来了」)
    apply_steps(d2, tax, D2, (board, "captive", False))
    rec = d2.instances(D2)[board]
    assert (rec.attrs["captive"], rec.parent, rec.attached) == (False, "motherboard.01", False)
