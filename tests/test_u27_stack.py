"""Task U27: relation fixes stack with the U9 / U12 / U14 / U21 replays, which do not move.

The real D2 sheet under the name patterns the live D2 was imported with (the
cage read as a latch, :func:`test_u21_replay.old_patterns`) and the controller's
work on it -- the lockers ``Shift+T``-ed out and back, the shield reclassed as
the cage (U21), the fan latch taken out (U9), a fan screw flipped
「其实拧下来了」 (U14), ``cover.02`` merged into ``cover.01`` (U12) -- once
without and once with relational fields set in Steps around those edits. Both
are re-imported twice: the four older replays say exactly the same in both
scenes, and the instance tables differ only by the fixes.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
from pathlib import Path

import pytest
from test_u21_replay import CAGE, LOCKER, SHIELD, flip, live, old_patterns  # noqa: F401

from relation_scenes import D2, apply_steps, replayed
from tda.core.captive_replay import OP_REPLAY_CAPTIVE
from tda.core.db import Db
from tda.core.merge_instances import merge_instances
from tda.core.merge_replay import OP_REPLAY_MERGES
from tda.core.model import FrameKey
from tda.core.reclass import reclass_instance
from tda.core.reclass_replay import OP_REPLAY_RECLASS
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of
from tda.core.verb_replay import OP_REPLAY
from tda.pipeline_logs import import_logs_into_db

KEY = FrameKey(D2, 48, "scan")
FAN = "screw.cpu_cooler.01"
OLDER = (OP_REPLAY, OP_REPLAY_CAPTIVE, OP_REPLAY_MERGES, OP_REPLAY_RECLASS)
#: The fields the second scene sets, as they must stand after every re-import.
FIXED = {
    (CAGE, "removal_direction"): "+Z",             # set on drive_latch.02 before the reclass
    ("psu_latch.01", "mounted_on"): "cover.01",    # named cover.02 before the merge
    ("psu_latch.01", "parent"): "psu.01",
    ("psu_latch.01", "attached"): True,
    ("storage_drive.ssd.01", "mounted_on"): CAGE,
    (FAN, "group_id"): "fan",
}


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def scene(folder: Path, tax, fixes: bool) -> Db:
    folder.mkdir()
    db = live(folder, tax, D2, 17, 12)               # imported, lockers out by Shift+T (U9)
    truth = TruthService(db, tax)
    flip(db, tax, D2, 12, LOCKER, "open")
    if fixes:
        apply_steps(db, tax, D2, (SHIELD, "removal_direction", "+Z"))
    reclass_instance(db, tax, D2, SHIELD, annotator="controller")            # U21
    from tda.ui.session_captive import commit_captive_fix
    from tda.ui.session_verb import commit_verb_fix
    commit_verb_fix(db, truth, KEY, "cooler_latch.01", "remove", "tester")    # U9
    commit_captive_fix(db, truth, KEY, FAN, None, "tester")                  # U14
    if fixes:
        apply_steps(db, tax, D2, ("cover.02", "removal_direction", "+Y"),
                    ("psu_latch.01", "mounted_on", "cover.02"))
    merge_instances(db, tax, D2, "cover.02", "cover.01", "tester")           # U12
    if fixes:
        apply_steps(db, tax, D2, ("psu_latch.01", "parent", "psu.01"),
                    ("psu_latch.01", "attached", True),
                    ("storage_drive.ssd.01", "mounted_on", CAGE), (FAN, "group_id", "fan"))
    return db


def reimport(db: Db, folder: Path, tax):
    run = import_logs_into_db(db, str(folder / "drive"), tax, desktops={D2}, force=True,
                              force_verified=True)
    assert [r.status for r in run.runs] == ["imported"], run.runs[0].issues
    return run.runs[0]


def _no_ids(value):
    """A payload without its op ids: the fixes' own rows shift every later id."""
    if isinstance(value, dict):
        return {k: _no_ids(v) for k, v in value.items() if k != "ops"}
    if isinstance(value, list):
        return [_no_ids(v) for v in value]
    return value


def older(db: Db) -> dict:
    return {kind: _no_ids(db.ops_of_kind(D2, kind)[-1]["payload"]) for kind in OLDER}


def value(rec, name):
    return rec.attrs.get("of") if name == "of" else getattr(rec, name)


def test_the_older_replays_say_the_same_and_the_fixes_are_kept(tmp_path, tax, old_patterns):
    plain = scene(tmp_path / "plain", tax, fixes=False)
    fixed = scene(tmp_path / "fixed", tax, fixes=True)
    stored = fixed.instances(D2)
    assert {k: value(stored[k[0]], k[1]) for k in FIXED} == FIXED      # as the edits left it
    assert stored["cover.01"].removal_direction is None                # merge kept cover.01's
    for _ in range(2):
        a, b = reimport(plain, tmp_path / "plain", tax), reimport(fixed, tmp_path / "fixed", tax)
        assert older(plain) == older(fixed)
        assert [x for x in a.corrections if not x.startswith("relation fix")] == \
            [x for x in b.corrections if not x.startswith("relation fix")]
        expected = copy.deepcopy(plain.instances(D2))
        for (key, name), want in FIXED.items():
            setattr(expected[key], name, want)
        assert fixed.instances(D2) == expected
        assert fixed.actions(D2) == plain.actions(D2)
        assert not validate_events(fixed.instances(D2), events_of(fixed, tax, D2), tax)
    done = replayed(fixed, D2)
    assert {(e.get("parsed") or e["key"], e["field"]): e["value"] for e in done["kept"]} == FIXED
    assert [(e["key"], e["field"], e["why"]) for e in done["unmatched"]] == [
        ("cover.02", "removal_direction", "merged into cover.01, which kept its own fields")]
    # the plain scene's only fixes are the ones the captive flip moved (U14 writes
    # through the Steps save): they agree with the sheet once the flip is replayed
    assert replayed(plain, D2)["kept"] == [] and replayed(plain, D2)["unmatched"] == []
    plain.close()
    fixed.close()
