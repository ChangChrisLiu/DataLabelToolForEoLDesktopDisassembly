"""Shared scene for task U27: a real sheet imported, edits applied through Steps.

The sheets are the real exports under ``tests/fixtures/logs`` -- D2 (a Dell
OptiPlex 7020: captive fan screws, the one cover named twice, the drive cage)
and D23 (an HP Compaq 8100 Elite SFF: the cable clip on the CPU fan shroud,
``tests/fixtures/logs/desktop_23.csv``). An edit goes the way the Steps panel's
``Apply`` sends it: :meth:`StepTableData.load` -> ``apply_instance_edit`` ->
``save``.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

from tda.core.db import Db
from tda.core.relation_replay import OP_RELATION_FIX, OP_REPLAY_RELATIONS
from tda.pipeline_logs import import_logs_into_db
from tda.ui.steps_model import StepTableData

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
D2, D23 = 2, 23
CLIP, SHROUD = "cable_clip.01", "cover.01"          # D23: the clip rides out on the shroud

__all__ = ["CLIP", "D2", "D23", "SHROUD", "apply_steps", "fixes", "imported", "reimport",
           "replayed", "sheets"]


def sheets(tmp_path: Path, desktop: int, edits: Iterable[tuple[str, str]] = ()) -> str:
    """The desktop's sheet in a folder of its own, with ``(old, new)`` text edits."""
    folder = tmp_path / "drive"
    folder.mkdir(exist_ok=True)
    for suffix in ("", "_meta"):
        name = f"desktop_{desktop:02d}{suffix}.csv"
        shutil.copy(FIXTURES / name, folder / name)
    path = folder / f"desktop_{desktop:02d}.csv"
    text = path.read_text(encoding="utf-8-sig")
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return str(folder)


def imported(tmp_path: Path, tax, desktop: int, name: str = "tda.sqlite") -> Db:
    db = Db(str(tmp_path / name))
    run = import_logs_into_db(db, sheets(tmp_path, desktop), tax, desktops={desktop})
    assert [r.status for r in run.runs] == ["imported"]
    return db


def apply_steps(db: Db, tax, desktop: int, *edits: tuple[str, str, object],
                annotator: str = "tester") -> list[str]:
    """``(key, field, value)`` edits made in Steps' instance table, then ``Apply``."""
    data = StepTableData.load(db, desktop, tax, annotator)
    for key, name, value in edits:
        data.apply_instance_edit(key, name, value)
    return data.save(db)


def reimport(db: Db, tmp_path: Path, tax, desktop: int,
             edits: Iterable[tuple[str, str]] = (), log: Optional[list] = None):
    """``import-logs --force`` of one desktop; returns its :class:`DesktopRun`."""
    run = import_logs_into_db(db, sheets(tmp_path, desktop, edits), tax, desktops={desktop},
                              force=True, force_verified=True,
                              log=None if log is None else log.append)
    assert [r.status for r in run.runs] == ["imported"], run.runs[0].issues
    return run.runs[0]


def fixes(db: Db, desktop: int) -> list[tuple]:
    """``(key, field, old, new, by)`` of every relation_fix row, oldest first."""
    return [(op["payload"]["key"], op["payload"]["field"], op["payload"]["old"],
             op["payload"]["new"], op["payload"]["by"])
            for op in db.ops_of_kind(desktop, OP_RELATION_FIX)]


def replayed(db: Db, desktop: int) -> dict:
    """The payload of the last ``replay_relations`` row."""
    return db.ops_of_kind(desktop, OP_REPLAY_RELATIONS)[-1]["payload"]
