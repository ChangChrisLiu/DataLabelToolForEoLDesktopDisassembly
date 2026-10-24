"""Small things that would each cost somebody an afternoon.

A flag that silently does nothing, a warning that does not mention the edits it
is about to destroy, a destructive command that is the only one without a safety
copy, two refusals that say the same thing twice, and a message that offers no
way to act on itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from test_cli import d13_steps, env, open_db, run  # noqa: F401  (re-used fixtures)

from tda.cli import EXIT_ERROR, EXIT_OK
from tda.core.db import Db
from tda.core.model import FrameKey


def _imported(env: dict) -> None:
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK


def _freeze(db: Db, desktop: int, step: int, view: str = "scan") -> None:
    db.put_compiled(FrameKey(desktop, step, view), "chassis.01", None, 0.0, "visible",
                    "in_chassis", "verified", "h", verified_by="chang",
                    geom_type="box", box=(0, 0, 4, 4))


# --------------------------------------------------------------------------- #
# a flag that only means something with another one
# --------------------------------------------------------------------------- #
def test_force_verified_without_force_is_refused(env, capsys):
    _imported(env)
    capsys.readouterr()
    assert run(env, "import-logs", "--force-verified") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "--force" in out
    assert "Traceback" not in out


def test_force_verified_with_force_is_fine(env):
    _imported(env)
    assert run(env, "import-logs", "--desktops", "13", "--force",
               "--force-verified") == EXIT_OK


# --------------------------------------------------------------------------- #
# what --force really destroys
# --------------------------------------------------------------------------- #
def test_the_force_warning_says_which_relational_fields_are_replayed_and_what_is_lost(
        env, capsys):
    """Task U27: the fields set in Steps are replayed; a head, or a field never recorded, not."""
    _imported(env)
    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    out = capsys.readouterr().out
    assert "fastens" in out and "parent" in out and "socket_host" in out
    assert ("and the relational fields set in Steps' instance table (parent, attached, "
            "mounted_on, fastens, socket_host, cable, slot/group, removal direction) where "
            "the sheet still has the part and the part they name") in out
    assert ("and so is a screw head set by hand and any relational field written before it "
            "was recorded (run `python -m tda.cli adopt-relations --desktop 13` first to "
            "keep those)") in out
    assert "so is every relational field a human set" not in out


def test_a_relation_set_in_steps_survives_a_forced_reimport(env):
    """The same correction as below, made the Steps way (task U27): it is replayed."""
    from tda.core.taxonomy import load_taxonomy
    from tda.ui.steps_model import StepTableData

    _imported(env)
    db = open_db(env)
    try:
        data = StepTableData.load(db, 13, load_taxonomy())
        data.apply_instance_edit("screw.cpu_cooler.01", "fastens", "chassis")
        data.save(db)
    finally:
        db.close()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    db = open_db(env)
    try:
        assert db.instances(13)["screw.cpu_cooler.01"].fastens == "chassis"
    finally:
        db.close()


def test_a_hand_set_relation_really_is_lost_on_a_forced_reimport(env):
    """Written straight into the database: no ``relation_fix`` row, nothing to replay."""
    _imported(env)
    db = open_db(env)
    try:
        rec = db.instances(13)["screw.cpu_cooler.01"]
        rec.fastens = "chassis.01"  # an annotator's correction
        db.upsert_instance(rec)
    finally:
        db.close()

    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    db = open_db(env)
    try:
        assert db.instances(13)["screw.cpu_cooler.01"].fastens != "chassis.01"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# load-index is destructive too
# --------------------------------------------------------------------------- #
def test_load_index_backs_the_database_up_first(env, capsys):
    """It can drop a pose segment's corners, homography and ROI."""
    assert run(env, "load-index") == EXIT_OK
    backups = Path(env["cfg"]["backup_dir"])
    made = list(backups.glob("tda_*.sqlite"))
    assert len(made) == 1 and made[0].stat().st_size > 0


def test_a_failing_backup_stops_load_index(env, capsys, monkeypatch):
    monkeypatch.setattr(Db, "backup",
                        lambda self, dest, keep=None: (_ for _ in ()).throw(OSError("no room")))
    capsys.readouterr()
    assert run(env, "load-index") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "[load-index] backup failed: no room" in out
    db = open_db(env)
    try:
        assert db.frames_for(13, "oak1") == []  # nothing was written
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# one refusal, and one that can be acted on
# --------------------------------------------------------------------------- #
def test_the_prescan_refusal_names_the_desktops_and_the_way_round_it(env, capsys):
    _imported(env)
    db = open_db(env)
    try:
        _freeze(db, 13, 5)
    finally:
        db.close()
    capsys.readouterr()

    assert run(env, "infer-relations") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D13" in out
    # the exact argument that runs everything else, ready to paste
    assert "--desktops 1,63,77,78" in out
    # and it is said once, not twice
    assert out.count("carry verified frames") == 1


def test_the_refusal_says_so_even_when_nothing_else_could_run(env, capsys):
    _imported(env)
    db = open_db(env)
    try:
        _freeze(db, 13, 5)
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "infer-relations", "--desktops", "13") == EXIT_ERROR
    out = capsys.readouterr().out
    assert "D13" in out and "--force" in out
    assert "--desktops " not in out.split("--force")[1]  # nothing left to offer


# --------------------------------------------------------------------------- #
# what the dropped-note counter actually counts
# --------------------------------------------------------------------------- #
def test_the_dropped_note_line_counts_steps(env, capsys):
    import csv

    _imported(env)
    db = open_db(env)
    try:
        steps = db.steps(13)
        for step in steps[:3]:
            step.notes = "LS: step_name=x"
        db.replace_steps(13, steps, db.actions(13))
    finally:
        db.close()

    path = Path(env["drive"]) / "desktop_13.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    out_rows = [header, ["", "An extra row"] + [""] * (len(header) - 2)] + body
    for i, row in enumerate(out_rows[1:], start=1):
        row[0] = str(i)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(out_rows)

    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    assert "steps with LS notes could not be carried over" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the implied note reads like a sentence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,want", [(1, "1 instance,"), (3, "3 instances,")])
def test_the_implied_note_counts_in_english(n: int, want: str):
    from tda.core.implied import _note

    assert want in _note(["a"] * n)
