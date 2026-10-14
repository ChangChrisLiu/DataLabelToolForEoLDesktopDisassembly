"""Task U8 round 3: the ✂ rows are decided once, without Qt.

:mod:`tda.core.split_rows` holds the card's split decisions so that ``cli
check`` and the commit rule can ask them; the task card calls the same
functions. These tests keep the two vocabularies equal and the core Qt-free.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tda.core import split_rows as SR
from tda.ui import session_api as api
from tda.ui import session_tasks as tasks

REPO = Path(__file__).resolve().parents[1]


def test_the_core_kinds_are_the_cards_kinds():
    assert SR.KIND_ADD_SHAPE == api.KIND_ADD_SHAPE
    assert SR.KIND_SPLIT_KEYFRAME == api.KIND_SPLIT_KEYFRAME
    assert SR.KIND_STATE_ONLY == api.KIND_STATE_ONLY
    assert tasks._kind_for is SR.kind_for and tasks._has_split is SR.has_split
    assert tasks.SPLIT_TRANSITIONS is SR.SPLIT_TRANSITIONS


def test_cli_check_reads_the_split_rows_without_qt():
    """``cli check`` imports this module; nothing it pulls in may be PySide6."""
    code = ("import sys; import tda.cli, tda.cli_app, tda.core.split_rows, tda.core.truth; "
            "print(sorted(m for m in sys.modules if m.startswith('PySide6')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out
