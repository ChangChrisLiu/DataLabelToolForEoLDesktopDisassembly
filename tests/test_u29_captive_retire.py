"""Task U29 (D, U27 R3-N1): a flip out of captive retires the parent fix made while captive.

The reviewer's scene on the real D2 sheet: ``Shift+T`` takes a CPU fan screw
out (a plain screw: no parent after a re-import), ``Shift+T`` puts it back in
(its ``parent`` is filled from what it fastens -- one ``relation_fix`` row,
``captive: true``), ``Shift+T`` takes it out again (U14 keeps the ``parent``,
inert, so no new ``parent`` row). Every forced re-import after that logged

    relation fix not matched: screw.cpu_cooler.01.parent = cpu_cooler.fan.01
    (captive differs: ... is not captive now, the fix was set while it was captive)

-- the right result (the screw is plain, no parent) under a line that looked
like a fault, for ever. The flip now records that the field's fix made under
the other ``captive`` is retired (a ``relation_fix`` row with ``retires``),
and the replay lists it as retired, applies nothing and prints nothing. A flip
back in restates the field the screw kept, so the parent it rides out with is
replayed again.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from relation_scenes import D2, apply_steps, imported, reimport, replayed
from tda.core.model import FrameKey
from tda.core.relation_replay import OP_RELATION_FIX
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.ui.commands import Op
from tda.ui.session_captive import apply_captive_fix, commit_captive_fix

KEY = FrameKey(D2, 48, "scan")
SCREW = "screw.cpu_cooler.01"
FAN = "cpu_cooler.fan.01"


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture
def d2(tmp_path, tax):
    db = imported(tmp_path, tax, D2)
    yield db
    db.close()


def flip(db, tax) -> Op:
    return commit_captive_fix(db, TruthService(db, tax), KEY, SCREW, None, "tester")["op"]


def record(db) -> tuple:
    rec = db.instances(D2)[SCREW]
    return bool((rec.attrs or {}).get("captive")), rec.parent, bool(rec.attached)


def not_matched(run) -> list[str]:
    return [line for line in run.corrections if line.startswith("relation fix not matched")]


def parent_rows(db) -> list[dict]:
    return [op["payload"] for op in db.ops_of_kind(D2, OP_RELATION_FIX)
            if op["payload"]["key"] == SCREW and op["payload"]["field"] == "parent"]


def test_in_and_out_again_leaves_no_line_on_any_import(d2, tmp_path, tax):
    assert record(d2) == (True, FAN, True)                # D2's fan screws stay in the fan
    flip(d2, tax)                                         # 其实拧下来了
    reimport(d2, tmp_path, tax, D2)
    assert record(d2) == (False, None, False)             # a plain screw: no parent
    flip(d2, tax)                                         # 其实留在零件里: parent filled
    assert record(d2) == (True, FAN, True)
    assert [(p["old"], p["new"], p["captive"]) for p in parent_rows(d2)] == [(None, FAN, True)]
    flip(d2, tax)                                         # out again: parent kept, inert
    assert record(d2) == (False, FAN, False)
    retire = parent_rows(d2)[-1]
    assert retire["captive"] is False and retire["retires"] == [
        op["id"] for op in d2.ops_of_kind(D2, OP_RELATION_FIX)
        if op["payload"]["key"] == SCREW and op["payload"]["field"] == "parent"][:1]
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2)
        assert not_matched(run) == [], not_matched(run)
        assert record(d2) == (False, None, False)        # the result the reviewer called right
        done = replayed(d2, D2)
        assert [(e["key"], e["field"]) for e in done["retired"]] == [(SCREW, "parent")]
        assert not any(SCREW + ".parent" in line for line in run.corrections), run.corrections


def test_a_flip_back_in_restates_the_parent_it_kept(d2, tmp_path, tax):
    """Out, in, out, in: the last word is captive in the fan, and the replay says so."""
    flip(d2, tax)
    reimport(d2, tmp_path, tax, D2)
    flip(d2, tax)
    flip(d2, tax)
    flip(d2, tax)                                         # in again, parent unchanged
    assert record(d2) == (True, FAN, True)
    last = parent_rows(d2)[-1]
    assert (last["new"], last["captive"]) == (FAN, True) and "retires" not in last
    run = reimport(d2, tmp_path, tax, D2)
    assert not_matched(run) == []
    assert record(d2) == (True, FAN, True)


def test_the_undo_of_the_flip_out_takes_the_retirement_back(d2, tmp_path, tax):
    flip(d2, tax)
    reimport(d2, tmp_path, tax, D2)
    flip(d2, tax)
    out = flip(d2, tax)
    truth = TruthService(d2, tax)
    apply_captive_fix(d2, truth, out.inverse)            # Ctrl+Z: captive in the fan again
    assert record(d2) == (True, FAN, True)
    assert (parent_rows(d2)[-1]["captive"], parent_rows(d2)[-1]["new"]) == (True, FAN)
    run = reimport(d2, tmp_path, tax, D2)
    assert not_matched(run) == []
    assert record(d2) == (True, FAN, True)


def test_a_parent_set_by_hand_on_a_plain_screw_survives_a_flip_in_and_out(d2, tmp_path, tax):
    """Round 2, N1 (the reviewer's scene): only what was set while captive is retired.

    A plain screw given ``parent = motherboard.01`` by hand in Steps keeps it
    through a re-import (main's behaviour); ticking and unticking ``Captive``
    retires the restatement made while it was captive, not the hand-set fix,
    so the next re-imports still keep ``motherboard.01`` -- and say nothing.
    """
    board = "screw.motherboard.01"
    apply_steps(d2, tax, D2, (board, "captive", False))     # plain to begin with
    reimport(d2, tmp_path, tax, D2)
    rec = d2.instances(D2)[board]
    assert (rec.attrs.get("captive"), rec.parent, rec.attached) == (False, None, False)
    apply_steps(d2, tax, D2, (board, "parent", "motherboard.01"))
    reimport(d2, tmp_path, tax, D2)
    rec = d2.instances(D2)[board]
    assert (rec.attrs.get("captive"), rec.parent) == (False, "motherboard.01")
    apply_steps(d2, tax, D2, (board, "captive", True))      # in: parent kept, attached
    apply_steps(d2, tax, D2, (board, "captive", False))     # out again
    rows = [op["payload"] for op in d2.ops_of_kind(D2, OP_RELATION_FIX)
            if op["payload"]["key"] == board and op["payload"]["field"] == "parent"]
    hand = [r for r in rows if r.get("captive") is False and "retires" not in r][0]
    assert hand["new"] == "motherboard.01"
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2)
        assert not_matched(run) == [], not_matched(run)
        rec = d2.instances(D2)[board]
        assert (rec.attrs.get("captive"), rec.parent, rec.attached) == (
            False, "motherboard.01", False)
        done = replayed(d2, D2)
        kept = [e for e in done["kept"] + done["agrees"]
                if (e["key"], e["field"]) == (board, "parent")]
        assert kept and kept[0]["value"] == "motherboard.01" and kept[0]["after_retired"]


def test_a_steps_untick_retires_it_too(d2, tmp_path, tax):
    """The Captive cell (U27 round 2) is the same flip: unticked, parent kept."""
    board = "screw.motherboard.01"
    apply_steps(d2, tax, D2, (board, "captive", True))   # parent filled, captive true
    apply_steps(d2, tax, D2, (board, "captive", False))  # parent stays, attached goes
    for _ in range(2):
        run = reimport(d2, tmp_path, tax, D2)
        assert not_matched(run) == [], not_matched(run)
        rec = d2.instances(D2)[board]
        assert (rec.attrs.get("captive"), rec.attached) == (False, False)
