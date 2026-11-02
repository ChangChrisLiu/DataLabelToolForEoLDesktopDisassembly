"""The clean-virtual-environment self-check of a VLM pack (W4-A; slow).

Builds a pack from the W3b scene, makes a fresh ``venv`` (no pip, the pure-Python
``yaml`` copied in: the offline stand-in for ``pip install PyYAML``) and runs
``tools/vlm_selfcheck.py`` in it through ``scripts/vlmpack_cleanvenv_check.py``.
Marked ``slow`` (a venv takes a few seconds). The same command, by hand::

    python scripts/vlmpack_cleanvenv_check.py --pack <pack> --venv D:/DataSet/.cache/tmp/w4a/venv --fresh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vlmpack_cleanvenv_check as C  # noqa: E402
from test_w4a_vlmpack import _build, _scene  # noqa: E402


@pytest.mark.slow
def test_the_pack_self_checks_in_a_fresh_venv(tmp_path, capsys):
    db_path, cache = _scene(tmp_path)
    _build(tmp_path, db_path, cache, "pack", images="none")
    venv = tmp_path / "venv"
    assert C.main(["--pack", str(tmp_path / "pack"), "--venv", str(venv),
                   "--out", str(tmp_path / "check")]) == 0
    out = capsys.readouterr().out
    assert "`import tda` fails in the venv" in out and "[cleanvenv] OK" in out
    report = json.loads((tmp_path / "check" / "cleanvenv_check.json").read_text(encoding="utf-8"))
    assert report["ok"] and Path(report["tda_module"]) == (tmp_path / "pack" / "tools" / "tda").resolve()
    assert {f["file"] for f in report["files"]} == {"scan.jsonl", "oak1.jsonl"}
    assert all(row["ok"] for f in report["files"] for row in f["rows"].values())
    # the venv never saw the repository: no tda, no yaml accelerator
    assert not list(venv.rglob("_yaml*"))
    assert not (C.site_packages(C.venv_python(venv)) / "tda").exists()
