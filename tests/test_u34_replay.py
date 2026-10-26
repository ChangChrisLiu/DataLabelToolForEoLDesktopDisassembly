"""Task U34 (replay, check, exports): a part said absent from the start stays absent.

* ``import-logs --force`` keeps it: the two step-0 events are hand-written
  (``auto = 0``), the import replaces the automatic log only and derives it
  from :func:`~tda.core.states.initial_overrides`, so the sheet's own action
  on the part derives nothing again;
* ``cli check`` says nothing about it (no ``missing_shape``, no warning);
* the COCO export has no row for it and the export context holds it apart
  (``DesktopCtx.absent``), its actions with it; no VLM record names it.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from test_cli import LOG_FIXTURES, d13_steps, env, open_db, run  # noqa: F401

from absent_scene import DRIVE, add_drive
from tda.cli import EXIT_OK, main
from tda.core import absent as AB
from tda.core.db import Db
from tda.core.states import validate_events
from tda.core.taxonomy import load_taxonomy
from tda.core.truth_inputs import events_of, instances_of, state_of

#: D13's "Case cover for motherboard screws" (step 35, ``open``): the sheet
#: acts on it, so a forced re-import has something to re-derive.
COVER = "cover.02"


# --------------------------------------------------------------------------- #
# import-logs --force
# --------------------------------------------------------------------------- #
def test_a_forced_re_import_keeps_it(env, capsys):
    assert run(env, "load-index") == EXIT_OK
    assert run(env, "import-logs") == EXIT_OK
    tax = load_taxonomy()
    db = open_db(env)
    try:
        assert [a.verb for a in db.actions(13) if a.target == COVER] == ["open"]
        AB.mark_absent(db, tax, 13, COVER, "tester")
        assert AB.is_absent(db, 13, COVER)
    finally:
        db.close()
    capsys.readouterr()
    assert run(env, "import-logs", "--desktops", "13", "--force") == EXIT_OK
    db = open_db(env)
    try:
        assert AB.is_absent(db, 13, COVER)
        assert [(e.step, e.attr, e.new, e.evidence_view) for e in db.events(13)
                if e.target == COVER and not e.auto] == [
            (0, "state", "removed", AB.EVIDENCE), (0, "placement", "elsewhere", AB.EVIDENCE)]
        # the sheet's open at 35 is back in the step table and derives nothing
        assert [a.verb for a in db.actions(13) if a.target == COVER] == ["open"]
        assert [e for e in db.events(13) if e.target == COVER and e.auto] == []
        log = events_of(db, tax, 13)
        assert not [m for m in validate_events(instances_of(db, 13), log, tax) if COVER in m]
        assert state_of(db, tax, 13, 40)[COVER].state == "removed"
        # and the way back still works on the re-imported table
        AB.mark_present(db, tax, 13, COVER, "tester")
        assert state_of(db, tax, 13, 40)[COVER].state == "open"
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# cli check
# --------------------------------------------------------------------------- #
@pytest.fixture
def workspace(tmp_path: Path):
    from app_scene import make_db, write_paths_yaml

    db, paths, tax = make_db(tmp_path)
    add_drive(db, tax, 13)
    db.close()
    return {"paths": write_paths_yaml(tmp_path), "cfg": paths, "tax": tax}


def _check(workspace, capsys) -> str:
    main(["--paths", workspace["paths"], "check", "--desktop", "13", "--view", "scan",
          "--limit", "1000"])
    return capsys.readouterr().out


def test_cli_check_says_nothing_about_it(workspace, capsys):
    before = _check(workspace, capsys)
    assert f"missing_shape:{DRIVE}" in before
    db = Db(workspace["cfg"]["db_path"])
    try:
        AB.mark_absent(db, workspace["tax"], 13, DRIVE, "tester")
    finally:
        db.close()
    after = _check(workspace, capsys)
    import re

    named = re.compile(rf"(?<![\w.]){re.escape(DRIVE)}")    # not screw.optical_drive.01
    assert named.search(before) and not named.search(after)
    count = lambda out: int(out.split("problems: ")[1].split()[0])  # noqa: E731
    assert count(after) == count(before) - before.count(f"missing_shape:{DRIVE}")


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def test_the_coco_export_has_no_row_for_it(tmp_path: Path):
    from app_scene import make_session, seed_shapes
    from PySide6.QtWidgets import QApplication

    from tda.core.export.coco import export_coco, load_ctx

    QApplication.instance() or QApplication([])
    session = make_session(tmp_path)
    session.sweeper_enabled = False
    try:
        db, tax = session.db, session.tax
        add_drive(db, tax, 13)
        seed_shapes(session, 14, skip=(DRIVE,))
        AB.mark_absent(db, tax, 13, DRIVE, "tester")
        session.refresh_all()
        doc = export_coco(db, tax, [13], "scan", str(tmp_path / "c.json"),
                          only_verified=False)
        assert doc["annotations"], "the scene exports its drawn parts"
        keys = {a["attributes"]["instance_key"] for a in doc["annotations"]}
        assert DRIVE not in keys and "screw.optical_drive.03" in keys   # its rider stays
        ctx = load_ctx(db, tax, 13, "scan")
        assert DRIVE in ctx.absent and DRIVE not in ctx.instances
        assert all(a.target != DRIVE for a in ctx.actions)
    finally:
        session.close(force=True)


def test_no_vlm_record_names_it(tmp_path: Path):
    import vlm_scene as S

    from tda.core.export.vlm import export_vlm
    from tda.core.model import ActionRec, InstanceRec

    db = Db(str(tmp_path / "vlm.sqlite"))
    try:
        tax = S.build(db, views=(S.VIEW,))
        db.upsert_instance(InstanceRec(key=DRIVE, desktop=S.DESKTOP, cls="optical_drive",
                                       raw_names=["Push optical drive out"]))
        db.replace_steps(S.DESKTOP, db.steps(S.DESKTOP),
                         [*db.actions(S.DESKTOP),
                          ActionRec(S.DESKTOP, 4, 1, DRIVE, "displace", tool="hand")])
        AB.mark_absent(db, tax, S.DESKTOP, DRIVE, "tester")
        out = tmp_path / "v.jsonl"
        export_vlm(db, tax, [S.DESKTOP], S.VIEW, str(out))
        text = out.read_text(encoding="utf-8")
        assert text.strip() and DRIVE not in text and "optical" not in text
        records = [json.loads(line) for line in text.splitlines() if line]
        assert records
    finally:
        db.close()
