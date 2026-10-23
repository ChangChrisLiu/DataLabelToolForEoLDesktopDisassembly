"""Task U20, review round: a merge of the cage's latch survives, and Shift+T keeps the cage latched.

* M1 -- the implied latch merged in Steps into the sheet's latch is not made
  again by the next import (``cage_latches``'s ``merged``; the import side is in
  ``test_u20_import.py``, probe p3d);
* M2 -- ``Shift+T`` saying the cage's *adopted* latch (D5/D6/D7's "HDD shield
  locker") was taken out gives the cage its implied latch at once, in the same
  op: the status line says so, Ctrl+Z takes both back, redo brings both, the
  frames before the cage comes out are re-checked, and a forced re-import keeps
  exactly that latch (probe p8);
* N1 -- the ``Ctrl+T`` dialog proposes what the card row says.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tda.core.db import Db
from tda.core.implied import CAGE_LATCH_RULE, cage_fix, cage_latches, is_cage_latch
from tda.core.model import FrameKey
from tda.core.state_assert import assert_state
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import state_of
from tda.pipeline_logs import import_logs_into_db
from tda.ui import session_state as SS
from tda.ui.app_verb import cage_note, verb_note
from tda.ui.session_api import SessionRefusal
from tda.ui.session_verb import CAGE_PATCH, apply_correct_verb, commit_verb_fix
from test_u20_rule import CAGE, LATCH, LOCKER, d6_actions, d6_instances, with_plan

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D5 = 5
VIEW = "scan"
FROZEN = (1, 5, 13, 20)


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def sheets(tmp_path: Path) -> str:
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for suffix in ("", "_meta"):
        name = f"desktop_05{suffix}.csv"
        shutil.copy(FIXTURES / name, folder / name)
    return str(folder)


@pytest.fixture
def d5(tmp_path, tax):
    """D5's sheet imported as it is (the locker adopted), with scan frames, some confirmed."""
    db = Db(str(tmp_path / "tda.sqlite"))
    import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5})
    for step in range(1, 47):
        flags = {"review_status": "verified"} if step in FROZEN else None
        db.upsert_frame(FrameKey(D5, step, VIEW), f"F:/scan/{step}.png", {"hw": [32, 32]},
                        None, flags)
    assert db.instances(D5)[LOCKER].parent == CAGE
    yield db
    db.close()


def states(db, tax, key, steps):
    return [state_of(db, tax, D5, s).get(key).state if key in state_of(db, tax, D5, s)
            else None for s in steps]


def take_out(db, tax):
    return commit_verb_fix(db, TruthService(db, tax), FrameKey(D5, 12, VIEW), LOCKER,
                           "remove", "tester")


# --------------------------------------------------------------------------- #
# M2: Shift+T on the adopted latch
# --------------------------------------------------------------------------- #
def test_taking_the_adopted_locker_out_gives_the_cage_its_latch_at_once(d5, tax):
    result = take_out(d5, tax)
    assert result["cage"] == {"cage": CAGE, "added": [LATCH], "dropped": [], "kept": [],
                              "opens": 13}
    latch = d5.instances(D5)[LATCH]
    assert is_cage_latch(latch) and (latch.parent, latch.attached) == (CAGE, True)
    assert states(d5, tax, LATCH, (1, 12, 13, 14)) == ["closed", "closed", "open", "removed"]
    assert states(d5, tax, LOCKER, (9, 10)) == ["closed", "removed"]
    # one op, the latch in it both ways
    op = result["op"]
    assert [r["key"] for r in op.payload[CAGE_PATCH]["add"]] == [LATCH]
    assert op.payload[CAGE_PATCH]["drop"] == []
    assert [r["key"] for r in op.inverse[CAGE_PATCH]["drop"]] == [LATCH]
    assert [o["kind"] for o in d5.ops(D5, VIEW)][-1] == "correct_verb"
    # the frames before the cage comes out are settled too: frozen ones re-checked
    assert {1, 5, 13} <= set(d5.rechecks(D5, VIEW))
    note = verb_note("硬盘架卡扣 drive_latch.01", result)
    assert note.endswith("；机架 drive_cage.01 因此自动得到隐含卡扣 drive_latch.02（默认第 13 帧"
                         "打开）（Ctrl+Z 撤销）")


def test_the_card_gains_it_on_frame_13(d5, tax):
    from tda.ui.session_tasks import task_card_for
    take_out(d5, tax)
    rows = {r["instance"]: r for r in task_card_for(d5, tax, D5, VIEW, 13, 14)}
    assert rows[LATCH]["parent"] == CAGE and rows[LATCH]["transition"] == ["removed", "open"]


def test_undo_takes_both_back_and_redo_brings_both(d5, tax):
    truth = TruthService(d5, tax)
    op = take_out(d5, tax)["op"]
    record = d5.instances(D5)[LATCH]
    apply_correct_verb(d5, truth, op.inverse)                   # Ctrl+Z
    assert LATCH not in d5.instances(D5)
    assert not [e for e in d5.events(D5) if e.target == LATCH]
    assert [a.verb for a in d5.actions(D5) if a.target == LOCKER] == ["open"]
    assert states(d5, tax, LOCKER, (9, 10, 13, 14)) == ["closed", "open", "open", "removed"]
    apply_correct_verb(d5, truth, op.payload)                   # redo
    assert d5.instances(D5)[LATCH] == record
    assert states(d5, tax, LATCH, (12, 13, 14)) == ["closed", "open", "removed"]
    assert [a.verb for a in d5.actions(D5) if a.target == LOCKER] == ["remove"]


def test_undo_is_refused_when_the_latch_has_work_of_its_own(d5, tax):
    op = take_out(d5, tax)["op"]
    assert_state(d5, tax, D5, LATCH, 12, "open")                # a T nobody undid first
    with pytest.raises(SessionRefusal, match="手设过状态"):
        apply_correct_verb(d5, TruthService(d5, tax), op.inverse)
    assert LATCH in d5.instances(D5)
    assert [a.verb for a in d5.actions(D5) if a.target == LOCKER] == ["remove"]


def test_a_forced_re_import_afterwards_keeps_exactly_that_latch(d5, tmp_path, tax):
    take_out(d5, tax)
    record = d5.instances(D5)[LATCH]
    printed: list[str] = []
    run = import_logs_into_db(d5, sheets(tmp_path), tax, desktops={D5}, force=True,
                              force_verified=True, log=printed.append)
    assert [r.status for r in run.runs] == ["imported"]
    latches = sorted(k for k, r in d5.instances(D5).items() if r.cls == "drive_latch")
    assert latches == [LOCKER, LATCH]
    assert d5.instances(D5)[LATCH] == record
    assert run.runs[0].cage_latches == [
        "cage latch kept: drive_latch.02 (implied) is drive_cage.01's own latch"]
    assert not run.runs[0].implied and not run.runs[0].dropped


def test_flipped_back_the_implied_latch_goes_unless_it_has_work(d5, tax):
    truth = TruthService(d5, tax)
    take_out(d5, tax)
    back = commit_verb_fix(d5, truth, FrameKey(D5, 12, VIEW), LOCKER, "open", "tester")
    assert back["cage"]["dropped"] == [LATCH] and LATCH not in d5.instances(D5)
    assert "机架 drive_cage.01 的隐含卡扣 drive_latch.02 用不着了，已删掉" in verb_note("x", back)
    assert states(d5, tax, LOCKER, (10, 13, 14)) == ["open", "open", "removed"]
    # its undo brings the latch back
    apply_correct_verb(d5, truth, back["op"].inverse)
    assert LATCH in d5.instances(D5)
    # with a hand-set state on it, it stays
    assert_state(d5, tax, D5, LATCH, 12, "open")
    again = commit_verb_fix(d5, truth, FrameKey(D5, 12, VIEW), LOCKER, "open", "tester")
    assert again["cage"]["dropped"] == [] and again["cage"]["kept"] == [LATCH]
    assert LATCH in d5.instances(D5) and CAGE_PATCH not in again["op"].payload
    assert "画过了，留着" in cage_note(again["cage"])


def test_a_latch_the_import_did_not_adopt_is_hung_on_the_cage_when_only_opened(tmp_path, tax):
    """D6 after its forced re-import: the locker has no parent, the cage has .02.

    Task U21 changed this (it used to touch no cage): the locker is never taken
    off the cage, so 「其实只是打开了」 adopts it in the same op and the cage's
    untouched implied latch goes; Ctrl+Z takes both back.
    """
    db = Db(str(tmp_path / "tda.sqlite"))
    try:
        import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5})
        truth = TruthService(db, tax)
        commit_verb_fix(db, truth, FrameKey(D5, 12, VIEW), LOCKER, "remove", "tester")
        import_logs_into_db(db, sheets(tmp_path), tax, desktops={D5}, force=True)
        assert db.instances(D5)[LOCKER].parent is None and LATCH in db.instances(D5)
        back = commit_verb_fix(db, truth, FrameKey(D5, 12, VIEW), LOCKER, "open", "tester")
        assert back["cage"]["adopted"] == [LOCKER] and back["cage"]["dropped"] == [LATCH]
        rec = db.instances(D5)[LOCKER]
        assert (rec.parent, rec.attached) == (CAGE, True) and LATCH not in db.instances(D5)
        apply_correct_verb(db, truth, back["op"].inverse)         # Ctrl+Z
        rec = db.instances(D5)[LOCKER]
        assert (rec.parent, rec.attached) == (None, False) and LATCH in db.instances(D5)
    finally:
        db.close()


def test_cage_fix_only_answers_for_an_adopted_latch(tax):
    adopted = with_plan(d6_instances(), d6_actions("open"), tax)
    found = cage_fix(adopted, d6_actions("remove"), tax, LOCKER)
    assert found.cage == CAGE and [r.key for r in found.add] == [LATCH]
    assert (found.out_at, found.opens, found.drop) == (14, 13, [])
    plain = d6_instances()                                       # the locker has no parent
    assert cage_fix(plain, d6_actions("remove"), tax, LOCKER) is None
    assert cage_fix(adopted, d6_actions("open"), tax, LOCKER) is None     # nothing to do
    deleted = [{"id": 1, "payload": {"instance": LATCH, "rule": CAGE_LATCH_RULE,
                                     "cage": CAGE}}]
    assert cage_fix(adopted, d6_actions("remove"), tax, LOCKER, deleted) is None


# --------------------------------------------------------------------------- #
# M1: merged is merged
# --------------------------------------------------------------------------- #
def _merge_record(cage: str = CAGE) -> dict:
    return {"key": LATCH, "cls": "drive_latch", "parent": cage, "attached": True,
            "attrs": {"implied": True, "implied_rule": CAGE_LATCH_RULE, "of": "hdd"}}


def test_a_merged_implied_latch_is_not_made_again(tax):
    plan = cage_latches(d6_instances(), d6_actions(), tax, rename={LATCH: LOCKER},
                        merged={LATCH: _merge_record()})
    assert not plan.made
    assert plan.merged == [{"latch": LATCH, "into": LOCKER, "cage": CAGE}]
    assert plan.lines() == [
        "cage latch merged: drive_latch.02 → drive_latch.01 (merged in Steps; "
        "drive_latch.01 is drive_cage.01's latch now - not made again)"]


def test_a_merge_of_something_else_does_not_count(tax):
    other = _merge_record(cage="drive_cage.09")
    assert [r.key for r in cage_latches(d6_instances(), d6_actions(), tax,
                                        rename={LATCH: LOCKER},
                                        merged={LATCH: other}).made] == [LATCH]
    plain = dict(_merge_record(), attrs={"added_by": "x"})
    assert [r.key for r in cage_latches(d6_instances(), d6_actions(), tax,
                                        rename={LATCH: LOCKER},
                                        merged={LATCH: plain}).made] == [LATCH]
    # merged into a part the import no longer has
    assert cage_latches(d6_instances(), d6_actions(), tax, rename={LATCH: "drive_latch.07"},
                        merged={LATCH: _merge_record()}).created


# --------------------------------------------------------------------------- #
# N1: the Ctrl+T dialog says what the card says
# --------------------------------------------------------------------------- #
def test_the_ctrl_t_dialog_proposes_open_from_this_frame(d5, tax):
    take_out(d5, tax)
    for walk in ("backward", SS.FORWARD_WALK):
        for step in (13, 12, 9):
            fake = SimpleNamespace(db=d5, tax=tax, browsing=walk,
                                   current=lambda s=step: FrameKey(D5, s, VIEW),
                                   next_state=lambda inst: "closed")
            assert SS.StateMixin.state_range_default(fake, LATCH) == ("open", step), \
                (walk, step)
    # after it was said open from 9, frame 9 proposes the run's own state, back to it
    assert_state(d5, tax, D5, LATCH, 9, "open")
    fake = SimpleNamespace(db=d5, tax=tax, browsing="backward",
                           current=lambda: FrameKey(D5, 8, VIEW),
                           next_state=lambda inst: "open")
    assert SS.StateMixin.state_range_default(fake, LATCH)[0] == "open"
