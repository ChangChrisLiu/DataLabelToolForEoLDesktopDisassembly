"""Task U25: ``scan-align`` scoring on synthetic change sequences (no images).

A synthetic desktop is a row of *states* -- one number per OAK capture, a
``dupli`` row repeating the state before it -- and the change between two
frames is the distance between their states plus a little noise. The scanner
photographs some of the states; ``judge`` has to say which.
"""
from __future__ import annotations

import numpy as np
import pytest

from tda.core import scan_align as A


def _states(n_rows: int, dupli: list[int], seed: int = 3) -> list[float]:
    rng = np.random.default_rng(seed)
    steps = rng.lognormal(mean=0.0, sigma=1.0, size=n_rows)
    states = [0.0]
    for r in range(2, n_rows + 1):
        states.append(states[-1] if r in dupli else states[-1] + float(steps[r - 1]))
    return states                       # states[r - 1] is row r


def _y_of(states: list[float], seed: int = 5):
    rng = np.random.default_rng(seed)
    noise = {}

    def y(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in noise:
            noise[key] = float(rng.normal(0, 0.02))
        return abs(states[b - 1] - states[a - 1]) + abs(noise[key])
    return y


def _x(states: list[float], rows: list[int], seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return [abs(states[rows[i + 1] - 1] - states[rows[i] - 1]) + abs(float(rng.normal(0, 0.02)))
            for i in range(len(rows) - 1)]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_ranks_average_ties():
    assert A.ranks([3, 1, 2, 2]).tolist() == [3.0, 0.0, 1.5, 1.5]


def test_spearman():
    assert A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert A.spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0
    assert A.spearman([1, 2], [2, 1]) == 0.0           # too short to say anything


def test_score_ignores_pairs_with_an_unknown_change():
    """A missing frame gives nan; its pairs drop out instead of poisoning the ranks."""
    y = lambda a, b: float("nan") if 3 in (a, b) else float(b * b - a * a)  # noqa: E731
    # pairs (1,2) (2,3) (3,4) (4,5) (5,6): the middle two are unknown
    assert A.score([1.0, 2.0, 3.0, 4.0, 5.0], y, (1, 2, 3, 4, 5, 6)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# hypotheses
# --------------------------------------------------------------------------- #
def test_hypotheses_edits_and_validity():
    hyps = A.hypotheses([1, 2, 3, 4], [1, 2, 3, 4], n_rows=5, pairs=False)
    by_label = {h.label: h.phi for h in hyps}
    assert by_label["as declared"] == (1, 2, 3, 4)
    assert by_label["scanner skipped row 3"] == (1, 2, 4, 5)
    assert by_label["folder 3 re-shoots the previous state"] == (1, 2, 2, 3)
    # skipping row 1 shifts everything, still inside 1..5
    assert by_label["scanner skipped row 1"] == (2, 3, 4, 5)
    assert len({h.phi for h in hyps}) == len(hyps)
    # with 5 folders on 5 rows no skip fits, only re-shots (and skip+re-shot pairs)
    full = A.hypotheses([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n_rows=5)
    assert not [h for h in full if h.kind == A.SKIP]
    assert any(h.kind == A.SKIP_DUP for h in full)


def test_signature_merges_a_dupli_row_with_the_row_it_repeats():
    canon = A.canonical_rows(6, [4])
    assert canon[1:] == [1, 2, 3, 3, 5, 6]
    assert A.signature((1, 2, 3, 5), canon) == A.signature((1, 2, 4, 5), canon)


# --------------------------------------------------------------------------- #
# the declaration a mapping needs
# --------------------------------------------------------------------------- #
def test_declaration_one_dupli_skipped_one_captured():
    """D18-like: 10 rows, dupli 4 and 8; the scanner skipped 4 and took 8."""
    rows = [1, 2, 3, 5, 6, 7, 8, 9, 10]            # the folder i photographs rows[i]
    d = A.declaration(rows, list(range(1, 10)), 10, [4, 8])
    assert d.duplicate_folders == [7]
    assert d.captured_dupli == {8: 7} and d.skipped_dupli == [4]
    assert d.missing_steps == [] and d.expressible


def test_declaration_both_skipped_is_an_empty_list():
    rows = [1, 2, 3, 5, 6, 7, 9, 10]
    d = A.declaration(rows, list(range(1, 9)), 10, [4, 8])
    assert d.duplicate_folders == [] and d.skipped_dupli == [4, 8] and d.expressible


def test_declaration_a_real_step_skipped_cannot_be_expressed():
    """The scanner skipped row 3 (not a dupli): the rule has no word for it."""
    rows = [1, 2, 4, 5, 6]
    d = A.declaration(rows, [1, 2, 3, 4, 5], 6, [])
    assert not d.expressible and "folder 3" in d.unexpressed[0]


def test_declaration_folders_named_by_row_with_gaps():
    """D1-like: folder numbers are rows, rows 2-3 never scanned, dupli 5 and 8 taken."""
    folders = [1, 4, 5, 6, 8, 9]
    d = A.declaration(folders, folders, 9, [5, 8])
    assert d.duplicate_folders == [5, 8]
    assert d.missing_steps == [2, 3]


# --------------------------------------------------------------------------- #
# judge on synthetic sequences
# --------------------------------------------------------------------------- #
def test_judge_finds_the_skipped_dupli_and_the_captured_one():
    n_rows, dupli = 30, [12, 22]
    states = _states(n_rows, dupli)
    photographed = [r for r in range(1, n_rows + 1) if r != 12]      # 29 folders
    folders = list(range(1, 30))
    declared = folders                                               # folder n = row n
    j = A.judge(_x(states, photographed), _y_of(states), declared, folders, n_rows, dupli,
                n_boot=200)
    assert j.verdict == A.SKIP and j.flagged
    assert j.declared.score < j.best.score - 0.2
    assert j.complete and j.complete[0].hyp.skip_row in (11, 12)     # 11 == 12 in pictures
    assert j.proposal.duplicate_folders == [21]
    assert j.proposal.skipped_dupli == [12] and j.proposal.captured_dupli == {22: 21}
    assert j.decided is True


def test_judge_keeps_an_aligned_desktop_aligned():
    n_rows = 25
    states = _states(n_rows, [], seed=11)
    folders = list(range(1, 26))
    j = A.judge(_x(states, folders), _y_of(states), folders, folders, n_rows, [],
                n_boot=200)
    assert j.verdict == "aligned" and j.answer is j.declared
    assert j.decided is None                     # no dupli question without dupli rows


def test_judge_with_a_dupli_the_scanner_also_photographed():
    """Count fits (folders = rows): declared, and the dupli folder is the duplicate."""
    n_rows, dupli = 20, [9]
    states = _states(n_rows, dupli, seed=13)
    folders = list(range(1, 21))
    j = A.judge(_x(states, folders), _y_of(states), folders, folders, n_rows, dupli,
                n_boot=200)
    assert j.verdict == "aligned"
    assert j.proposal.duplicate_folders == [9] and j.decided


def test_judge_says_unknown_when_nothing_correlates():
    """A reference view that never changes (a frozen camera) correlates with nothing."""
    rng = np.random.default_rng(0)
    x = list(rng.random(19))
    folders = list(range(1, 21))
    j = A.judge(x, lambda a, b: 0.0, folders, folders, 20, [], pairs=False, n_boot=50)
    assert j.verdict == "unknown" and j.flagged


def test_bootstrap_prefers_the_true_mapping():
    n_rows = 20
    states = _states(n_rows, [], seed=17)
    rows = [r for r in range(1, 21) if r != 8]
    x = _x(states, rows)
    truth = tuple(rows)
    wrong = tuple(range(1, 20))
    share = A.bootstrap(x, _y_of(states), truth, wrong, n=200)
    assert 0.9 <= share <= 1.0
