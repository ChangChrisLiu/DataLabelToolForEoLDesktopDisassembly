"""Task U36 on the command line: ``check --visibility`` lists the runs, ``status`` counts them.

``check`` stays quiet about the visibility review by default (U24 pinned
that); with ``--visibility`` it lists every run of every part -- first and
last frame, frames, label, reason, set by hand, verdict yes/no, 「全程」 -- so
the controller can audit; the exit code does not move. ``status`` says how
many runs are still to review per view.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib
import io

import pytest

import vis_review_scene as V
import vis_runs_scene as S
from session_scene import DESKTOP, VIEW
from tda.core import vis_review as VR
from tda.core.occluder import FULL


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    session = S.build(tmp_path_factory.mktemp("u36_cli"))
    session.goto(10, force=True)
    session.review_visibility(S.MARKED, VR.VERDICT_VIEWPOINT, FULL, steps=S.STEPS)
    session.goto(12, force=True)
    session.review_visibility(V.TINY, VR.VERDICT_COMPUTED)
    yield session
    session.close(force=True)


class Args:
    limit = 1000
    visibility = False


def check(scene, view: str = VIEW, visibility: bool = False) -> tuple[str, int]:
    from tda.cli_app import _check_one

    args = Args()
    args.visibility = visibility
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = _check_one(scene.db, scene.tax, args, DESKTOP, view)
    return out.getvalue(), code


def test_check_lists_the_runs_with_visibility_and_is_quiet_without(scene):
    quiet, code = check(scene)
    assert "visibility_run" not in quiet
    said, code_with = check(scene, visibility=True)
    assert code_with == code
    runs = [line.strip() for line in said.splitlines() if line.startswith("  visibility_run ")]
    assert runs == [
        "visibility_run connector.03 a=10 b=11 n=2 label=too_small occluder=- manual=no "
        "verdict=no whole=no",
        "visibility_run connector.03 a=12 b=12 n=1 label=too_small occluder=- manual=no "
        "verdict=yes whole=no",
        "visibility_run connector.03 a=13 b=14 n=2 label=too_small occluder=- manual=no "
        "verdict=no whole=no",
        "visibility_run connector.04 a=10 b=14 n=5 label=occluded_full occluder=viewpoint "
        "manual=yes verdict=yes whole=yes",
        "visibility_run connector.05 a=10 b=14 n=5 label=occluded_full occluder=- manual=yes "
        "verdict=no whole=yes",
        "visibility_run cpu.01 a=13 b=13 n=1 label=occluded_full occluder=cable manual=yes "
        "verdict=no whole=no",
        "visibility_run psu.01 a=12 b=12 n=1 label=out_of_view occluder=- manual=yes "
        "verdict=no whole=no",
        "visibility_run ram_latch.03 a=10 b=14 n=5 label=occluded_full occluder=- manual=no "
        "verdict=no whole=yes",
        "visibility_run ram_module.01 a=10 b=14 n=5 label=occluded_partial occluder=- "
        "manual=no verdict=no whole=yes",
    ]
    head = next(line for line in said.splitlines() if line.startswith("[check] visibility_runs"))
    assert head.startswith(f"[check] visibility_runs D{DESKTOP:02d} scan: 9 段 / 27 帧，待审 7 段 / 21 帧")
    # the last lines are what they were
    assert said.splitlines()[-1].startswith("stale_within_tolerance:")


def test_check_lists_the_other_view_too(scene):
    said, _code = check(scene, S.OTHER, visibility=True)
    assert ("  visibility_run connector.04 a=10 b=12 n=3 label=occluded_full occluder=cable "
            "manual=yes verdict=no whole=no") in said
    assert ("  visibility_run connector.04 a=13 b=13 n=1 label=occluded_full "
            "occluder=viewpoint manual=yes verdict=yes whole=no") in said


def test_the_flag_is_parsed():
    from tda.cli import build_parser

    args = build_parser().parse_args(["check", "--desktop", "13", "--visibility"])
    assert args.visibility is True
    assert build_parser().parse_args(["check", "--desktop", "13"]).visibility is False


def test_status_reads_each_view_once(scene, monkeypatch):
    """Round 2 (N4): one read of the rows and one of the frames per view -- the
    same counts as U24's progress and as the listing."""
    from tda import pipeline as P
    from tda.core import vis_runs as RU

    for view in (VIEW, S.OTHER):
        reviewed, total = VR.progress(scene.db, DESKTOP, view)
        left = sum(1 for run in RU.view_runs(scene.db, DESKTOP, view) if not run.reviewed)
        assert RU.review_counts(scene.db, DESKTOP, view) == (reviewed, total, left)
    calls = {"rows": 0, "frames": 0, "get_frame": 0}
    real_rows, real_frames, real_get = (scene.db.vis_review_rows, scene.db.annotatable_frames,
                                        scene.db.get_frame)

    def count(name, real):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(scene.db, "vis_review_rows", count("rows", real_rows))
    monkeypatch.setattr(scene.db, "annotatable_frames", count("frames", real_frames))
    monkeypatch.setattr(scene.db, "get_frame", count("get_frame", real_get))
    P.status_rows(scene.db, {DESKTOP})
    # two views have confirmed frames (scan, oak1): each read once, no frame read per step
    assert calls == {"rows": 2, "frames": 2, "get_frame": 0}


def test_status_counts_the_runs_still_to_review(scene):
    from tda import pipeline as P
    from tda.cli import format_desktop, format_status

    [row] = P.status_rows(scene.db, {DESKTOP})
    assert row["views"][VIEW]["vis_runs"] == 7
    assert row["views"][S.OTHER]["vis_runs"] == 2          # connector.04 10-12, ram_latch.03
    detail = format_desktop(scene.db, row)
    line = next(line for line in detail.splitlines() if line.startswith(f"  {VIEW} "))
    assert line.rstrip().endswith("（待审 7 段）")
    assert "scan 6/27（待审 7 段）" in format_status([row])
