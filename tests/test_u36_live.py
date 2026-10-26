"""Task U36 on real data -- only with ``TDA_U36_LIVE`` naming a **copy** of the database.

Skipped otherwise (the suite never reads the live file). The copy is copied
again into the test's own folder first, so nothing here can write to it. On
D2's scanner view: the runs cover every row the review is about exactly once,
the queue lists the unreviewed ones longest first, and the other views of a
desktop with OAK drafts (D13) read 「未确认」.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

LIVE = os.environ.get("TDA_U36_LIVE", "")

pytestmark = pytest.mark.skipif(not LIVE or not Path(LIVE).is_file(),
                                reason="TDA_U36_LIVE names no database copy")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    from tda.core.db import Db

    path = tmp_path_factory.mktemp("u36_live") / "copy.sqlite"
    shutil.copyfile(LIVE, path)
    found = Db(str(path))
    yield found
    found.close()


def test_d2_runs_cover_the_rows_once_and_the_queue_is_longest_first(db):
    from tda.core import vis_review as VR
    from tda.core import vis_runs as RU

    runs = RU.view_runs(db, 2, "scan")
    pending, done = VR.review_rows(db, 2, "scan")
    rows = {(r["instance"], r["step"]) for r in pending + done}
    covered = [(run.instance, s) for run in runs for s in run.steps]
    assert len(covered) == len(set(covered)) and set(covered) == rows
    queue = VR.order_runs([run for run in runs if not run.reviewed])
    longest = {}
    for run in queue:
        longest.setdefault(run.instance, len(run.steps))
    values = list(longest.values())
    assert values == sorted(values, reverse=True)
    assert max(len(run.steps) for run in runs) >= 10       # the long 看不见 the user saw


def test_a_desktop_with_oak_drafts_reads_unconfirmed_there(db):
    from tda.core import vis_runs as RU

    states = {view: RU.read_side(db, 13, view).state for view in ("scan", "oak1", "oak2", "rs")}
    assert states["scan"] == RU.STATE_OK
    assert set(states.values()) - {RU.STATE_OK} <= {RU.STATE_UNCONFIRMED, RU.STATE_NONE}
