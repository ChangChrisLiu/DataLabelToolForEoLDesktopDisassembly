"""Do the scanner's folders line up with the OAK steps?  (task U25, spec 2.2)

The scanner numbers its own folders ``RGB<n>1`` and the OAK Camera_1 captures
are the steps, so "folder ``n`` is step ``n``" is an assumption -- and on eleven
desktops it is wrong from some point on: the scanner did not photograph the
sheet's ``dupli`` row (a capture that repeats the previous state), so every
later folder is one row behind. This module ranks, from the pictures alone, the
ways the scanner folders can map onto the rows; the declaration that goes into
``configs/index_fixes.yaml`` is then confirmed by looking at the frames
(:func:`render_changes`) and at what the sheet says about each ``dupli`` row.

**Change magnitudes.** Every frame is cut to the chassis ROI (the pose segment's
ROI when the database has one, else the box :func:`tda.core.cache.suggest_roi_over`
proposes, else the whole frame), turned grey, downscaled to at most
:data:`LONG_SIDE` px and normalised by its own median and MAD, so an exposure
change is not a change. ``change(a, b)`` is the mean absolute difference of two
such frames. ``x[i]`` is the change between scanner folders ``i`` and ``i+1``;
the OAK change between any two rows is computed on demand. A removed drive,
PSU, heatsink or cage is a peak in both sequences; a ``dupli`` pair is noise in
the OAK one.

**Hypotheses.** A hypothesis maps every scanner folder (in folder order) onto an
OAK row: ``phi[i]`` (1-based, uncollapsed rows -- the sheet's numbering). The
mapping the index uses today is ``declared``; the others are made from it by
one or two edits: the scanner **skipped** row ``k`` (the folders from ``k`` on
move one row later), folder ``n`` is a **re-shot** of the previous state (the
folders from ``n`` on move one row earlier), or both (a skipped step that an
extra shot compensates -- no count symptom). Its **score** is the Spearman rank
correlation of ``x`` with the OAK changes between the rows it pairs, so scale
does not matter, only which steps changed the scene most.

**Equivalent hypotheses.** A ``dupli`` row shows the state of the row before it,
so "skipped row 30" and "skipped row 29" are the same pictures when 30 is a
``dupli`` of 29. Hypotheses are compared by their **state signature** (each row
replaced by the row whose state it shows) and only the best of each signature is
listed.

**What it cannot do.** It ranks; it does not see. Between two big changes a
run of screws is a run of small changes, so *where* inside that run a skip
happened is decided by the ``dupli`` prior (the scanner skips the row that
repeated a state) and by the frames, not by the numbers (on D48 and D62 the
best position was wrong). A paired bootstrap over the folder pairs says how
often one hypothesis beats another; the ones the best does not beat in
:data:`PLAUSIBLE` of the resamples are listed as just as plausible, and every
edit costs :data:`EDIT_PENALTY` before it may replace the declared mapping.

Nothing here writes outside the cache directory it is given; the database is
opened read-only and the raw drive is only read.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "DECLARED", "SKIP", "DUP", "SKIP_DUP", "LONG_SIDE", "EDIT_PENALTY", "Analysis",
    "Declaration", "Hypothesis", "Judgement", "Ranked", "analyse", "canonical_rows",
    "change", "declaration", "describe", "edits", "hypotheses", "judge", "normalise",
    "ranks", "render_changes", "render_decisive", "render_strip", "signature", "spearman",
]

#: Longest side, in pixels, a frame is compared at.
LONG_SIDE = 256
#: To call the declared mapping wrong, an alternative must score this much more
#: ... and win this share of the bootstrap resamples; an aligned verdict is still
#: flagged when an alternative scores MIN_GAIN more without winning that often.
MIN_GAIN = 0.05
CONFIDENT = 0.95
#: What one edit of the declared mapping costs in score. Thousands of edits are
#: tried, and on desktops whose declaration was checked by eye the best of them
#: still beat it by up to 0.09 (one edit) and 0.21 (a skip plus a re-shot) from
#: noise alone (U25, the 66-desktop audit): misalignments are rare, so a mapping
#: has to beat the declared one by more than its edits cost.
EDIT_PENALTY = 0.11
#: Bootstrap resamples per comparison, and their seed (the answer is repeatable).
BOOTSTRAP = 400
SEED = 25
#: When even the best mapping correlates below this, the two sequences barely
#: agree anywhere: the verdict is "unknown", whatever ranks first.
POOR = 0.30

DECLARED, SKIP, DUP, SKIP_DUP = "declared", "skip", "dup", "skip+dup"


def edits(h: "Hypothesis") -> int:
    """How many edits of the declared mapping ``h`` is (0, 1 or 2)."""
    return {DECLARED: 0, SKIP: 1, DUP: 1, SKIP_DUP: 2}[h.kind]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks ``0..n-1`` of ``values``; tied values share their average rank."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    _values, first, counts = np.unique(v[order], return_index=True, return_counts=True)
    average = first + (counts - 1) / 2.0
    out = np.empty(len(v), dtype=float)
    out[order] = np.repeat(average, counts)
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``0.0`` when either side is constant or short."""
    if len(x) != len(y):
        raise ValueError(f"spearman() needs equal lengths, got {len(x)} and {len(y)}")
    if len(x) < 3:
        return 0.0
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# frames -> change magnitudes
# --------------------------------------------------------------------------- #
def normalise(img: np.ndarray) -> np.ndarray:
    """A small grey frame, lightly blurred, centred on its median, scaled by its MAD."""
    import cv2

    f = cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (3, 3), 0)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med)))
    return (f - med) / max(mad, 1.0)


def change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two :func:`normalise` d frames."""
    if a.shape != b.shape:
        import cv2

        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(b - a)))


# --------------------------------------------------------------------------- #
# hypotheses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hypothesis:
    """One mapping of the scanner folders onto OAK rows (see the module doc)."""

    kind: str
    phi: tuple[int, ...]
    skip_row: Optional[int] = None
    dup_folder: Optional[int] = None

    @property
    def label(self) -> str:
        if self.kind == DECLARED:
            return "as declared"
        parts = []
        if self.skip_row is not None:
            parts.append(f"scanner skipped row {self.skip_row}")
        if self.dup_folder is not None:
            parts.append(f"folder {self.dup_folder} re-shoots the previous state")
        return " + ".join(parts)


def _skip(phi: Sequence[int], k: int) -> tuple[int, ...]:
    """Folders mapped to row ``k`` or later move one row on."""
    return tuple(r + 1 if r >= k else r for r in phi)


def _dup(phi: Sequence[int], j: int) -> tuple[int, ...]:
    """Folder index ``j`` shows the state of folder ``j-1``; the later ones move back."""
    return tuple(phi[:j]) + tuple(phi[j - 1:len(phi) - 1])


def hypotheses(declared: Sequence[int], folders: Sequence[int], n_rows: int, *,
               pairs: bool = True) -> list[Hypothesis]:
    """The declared mapping, every single skip / re-shot, and (``pairs``) every
    skip compensated by a re-shot; mappings off the ends of ``1..n_rows`` and
    repeats of an earlier mapping are left out."""
    if len(declared) != len(folders):
        raise ValueError("one declared row per folder")
    base = tuple(int(r) for r in declared)
    out: dict[tuple[int, ...], Hypothesis] = {base: Hypothesis(DECLARED, base)}

    def add(h: Hypothesis) -> None:
        if h.phi not in out and all(1 <= r <= n_rows for r in h.phi):
            out[h.phi] = h

    top = max(base) if base else 0
    for k in range(1, top + 1):
        add(Hypothesis(SKIP, _skip(base, k), skip_row=k))
    for j in range(1, len(base)):
        add(Hypothesis(DUP, _dup(base, j), dup_folder=int(folders[j])))
    if pairs:
        for j in range(1, len(base)):
            dupped = _dup(base, j)
            for k in range(1, max(dupped) + 1):
                add(Hypothesis(SKIP_DUP, _skip(dupped, k), skip_row=k,
                               dup_folder=int(folders[j])))
    return list(out.values())


def canonical_rows(n_rows: int, dupli_rows: Iterable[int]) -> list[int]:
    """``canon[r]`` = the row whose state row ``r`` shows (index 0 unused).

    A ``dupli`` row repeats the state of the row before it, so it maps to that
    row's canonical row; every other row is its own.
    """
    dupli = {int(r) for r in dupli_rows}
    canon = list(range(n_rows + 1))
    for r in range(2, n_rows + 1):
        if r in dupli:
            canon[r] = canon[r - 1]
    return canon


def signature(phi: Sequence[int], canon: Sequence[int]) -> tuple[int, ...]:
    """The states a mapping shows, folder by folder (what the pictures can tell)."""
    return tuple(canon[r] for r in phi)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@dataclass
class Ranked:
    hyp: Hypothesis
    score: float
    signature: tuple[int, ...]


def _ys(phi: Sequence[int], y_of: Callable[[int, int], float]) -> list[float]:
    return [0.0 if phi[i] == phi[i + 1] else y_of(phi[i], phi[i + 1])
            for i in range(len(phi) - 1)]


def _usable(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """The pairs both of whose changes are known (a missing frame gives ``nan``)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep], ya[keep]


def score(x: Sequence[float], y_of: Callable[[int, int], float],
          phi: Sequence[int]) -> float:
    """Spearman of the scanner changes with the OAK changes ``phi`` pairs them with."""
    return spearman(*_usable(x, _ys(phi, y_of)))


def rank_all(x: Sequence[float], y_of: Callable[[int, int], float],
             hyps: Iterable[Hypothesis], canon: Sequence[int]) -> list[Ranked]:
    """Best hypothesis of every state signature, best first.

    The declared mapping always keeps its own entry (an equivalent mapping that
    scores a hair better is still "as declared"), and among equivalent edits the
    one that names a ``dupli`` row speaks for them ("skipped row 30", not "row 29",
    when 30 repeats 29) -- the pictures are the same, only the OAK capture
    noise differs.
    """
    def preference(r: Ranked) -> tuple:
        h = r.hyp
        names_dupli = h.skip_row is not None and canon[h.skip_row] != h.skip_row
        return (h.kind == DECLARED, names_dupli, r.score)

    best: dict[tuple[int, ...], Ranked] = {}
    for h in hyps:
        sig = signature(h.phi, canon)
        r = Ranked(h, score(x, y_of, h.phi), sig)
        held = best.get(sig)
        if held is None or preference(r) > preference(held):
            best[sig] = r
    return sorted(best.values(), key=lambda r: (-r.score, _kind_order(r.hyp)))


def _kind_order(h: Hypothesis) -> tuple:
    return ({DECLARED: 0, SKIP: 1, DUP: 2, SKIP_DUP: 3}[h.kind],
            h.skip_row or 0, h.dup_folder or 0)


def bootstrap(x: Sequence[float], y_of: Callable[[int, int], float],
              a: Sequence[int], b: Sequence[int], n: int = BOOTSTRAP,
              seed: int = SEED) -> float:
    """Share of pair resamples in which mapping ``a`` scores above ``b`` (ties: half)."""
    xa = np.asarray(x, dtype=float)
    ya, yb = np.asarray(_ys(a, y_of), dtype=float), np.asarray(_ys(b, y_of), dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(yb)
    xa, ya, yb = xa[keep], ya[keep], yb[keep]
    if len(xa) < 3:
        return 0.5
    rng = np.random.default_rng(seed)
    wins = 0.0
    for _ in range(n):
        idx = rng.integers(0, len(xa), len(xa))
        sa, sb = spearman(xa[idx], ya[idx]), spearman(xa[idx], yb[idx])
        wins += 1.0 if sa > sb else 0.5 if sa == sb else 0.0
    return wins / n


# --------------------------------------------------------------------------- #
# from a mapping to the index declaration
# --------------------------------------------------------------------------- #
@dataclass
class Declaration:
    """What :func:`declaration` makes of one mapping."""

    duplicate_folders: list[int] = field(default_factory=list)
    captured_dupli: dict[int, int] = field(default_factory=dict)   # dupli row -> folder
    skipped_dupli: list[int] = field(default_factory=list)
    #: duplicate folders that re-shoot a state no ``dupli`` row repeats
    reshot_folders: list[int] = field(default_factory=list)
    missing_steps: list[int] = field(default_factory=list)        # collapsed steps
    #: Real steps the scanner never photographed (``scan.<d>.skipped_steps``,
    #: U25 round 2): a folder shows a state later than the rule would put it on.
    skipped_steps: list[int] = field(default_factory=list)
    #: Folders neither rule can put on their step (a folder out of order).
    unexpressed: list[str] = field(default_factory=list)

    @property
    def expressible(self) -> bool:
        return not self.unexpressed


def collapsed_of(n_rows: int, dupli_rows: Iterable[int]) -> list[int]:
    """``cstep[r]`` = the collapsed step row ``r`` belongs to (index 0 unused)."""
    dupli = {int(r) for r in dupli_rows if 1 < int(r) <= n_rows}
    out, seen = [0] * (n_rows + 1), 0
    for r in range(1, n_rows + 1):
        if r in dupli:
            seen += 1
        out[r] = r - seen
    return out


def declaration(phi: Sequence[int], folders: Sequence[int], n_rows: int,
                dupli_rows: Iterable[int]) -> Declaration:
    """The ``scan.<d>.duplicate_folders`` that puts every folder where ``phi`` says.

    The index maps folder ``n`` to collapsed step ``n - #(duplicate folders <= n)``
    and makes a duplicate folder an extra shot of that step (spec 2.2, v1.25).
    Walking the folders in order, a folder is a duplicate exactly when counting
    it is what lands it on the collapsed step of the state ``phi`` gives it; a
    folder that neither choice lands is ``unexpressed``.
    """
    dupli = sorted({int(r) for r in dupli_rows if 1 < int(r) <= n_rows})
    canon = canonical_rows(n_rows, dupli)
    cstep = collapsed_of(n_rows, dupli)
    dupli_of_step: dict[int, list[int]] = {}
    for r in dupli:
        dupli_of_step.setdefault(cstep[r], []).append(r)
    out = Declaration()
    count = 0
    covered: set[int] = set()
    for n, r in zip(folders, phi):
        n, want = int(n), cstep[canon[r]]
        at = n - count + len(out.skipped_steps)   # where the rule puts folder n
        if at == want:
            pass                                  # the main shot of its step
        elif at - 1 == want:
            count += 1                            # an extra shot of its step
            out.duplicate_folders.append(n)
            rows = dupli_of_step.get(want) or []
            if rows:
                out.captured_dupli[rows.pop(0)] = n
            else:
                out.reshot_folders.append(n)
        elif want > at:
            # the scanner is ahead: the steps in between were never photographed
            out.skipped_steps.extend(range(at, want))
        else:
            out.unexpressed.append(
                f"folder {n} shows the state of row {r} (step {want}) but the rule "
                f"puts it on step {at}")
        covered.add(want)
    out.skipped_dupli = [r for r in dupli if r not in out.captured_dupli]
    n_steps = cstep[n_rows] if n_rows else 0
    out.missing_steps = [s for s in range(1, n_steps + 1) if s not in covered]
    return out


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #
#: A hypothesis the best one beats in fewer than this share of the bootstrap
#: resamples is not clearly worse: it is in the **plausible set**.
PLAUSIBLE = 0.90
#: How many of the next-best hypotheses are bootstrapped against the best.
PLAUSIBLE_POOL = 40


@dataclass
class Judgement:
    """Everything :func:`judge` found for one desktop and one reference view."""

    ranked: list[Ranked]
    declared: Ranked
    best: Ranked
    plausible: list[Ranked]           # the best first, then the ones not clearly worse
    answer: Ranked                    # the declared mapping when it is plausible, else best
    runner_up: Optional[Ranked]       # the best hypothesis that is a different answer
    margin: float                     # answer - runner_up
    confidence: float                 # bootstrap: answer beats runner_up
    gain: float                       # best - declared
    verdict: str                      # "aligned", "skip", "dup", "skip+dup", "unknown"
    flagged: bool
    note: str
    complete: list[Ranked] = field(default_factory=list)   # every state photographed
    complete_confidence: Optional[float] = None             # complete[0] beats complete[1]
    complete_consistent: Optional[bool] = None              # complete[0] is plausible
    proposal: Optional[Declaration] = None
    #: the ``dupli`` question answered: one complete mapping, plausible, and
    #: clearly ahead of any other complete one (``None``: no ``dupli`` rows)
    decided: Optional[bool] = None

    def positions(self) -> list[Ranked]:
        """The plausible hypotheses of the answer's kind (where the edit may be)."""
        return [r for r in self.plausible if r.hyp.kind == self.answer.hyp.kind]


def judge(x: Sequence[float], y_of: Callable[[int, int], float],
          declared: Sequence[int], folders: Sequence[int], n_rows: int,
          dupli_rows: Iterable[int] = (), *, pairs: bool = True,
          n_boot: int = BOOTSTRAP) -> Judgement:
    """Rank every hypothesis and say what the pictures support.

    The **plausible set** is the best hypothesis and every one it does not beat
    in :data:`PLAUSIBLE` of the bootstrap resamples (where an edit may be).
    ``verdict`` is ``aligned`` unless a hypothesis beats the declared mapping by
    more than its edits cost (:data:`EDIT_PENALTY` each), by :data:`MIN_GAIN`,
    and in :data:`CONFIDENT` of the resamples -- then it is that hypothesis'
    kind -- and ``unknown`` when even the best correlates below :data:`POOR`. An
    aligned verdict is still flagged when such a hypothesis wins too few
    resamples.

    ``complete`` lists the mappings that photograph every state (the ``dupli``
    question: which ``dupli`` rows did the scanner also take?); ``proposal`` is
    the declaration of the best of them, and ``decided`` says whether the
    pictures settle it: that mapping is plausible and clearly ahead of the next
    complete one. Where in a run of small changes a skip happened the numbers
    cannot say -- the plausible positions show it -- so a ``dupli`` row inside
    that run is the answer the declaration takes.
    """
    dupli = sorted({int(r) for r in dupli_rows})
    canon = canonical_rows(n_rows, dupli)
    hyps = hypotheses(declared, folders, n_rows, pairs=pairs)
    ranked = rank_all(x, y_of, hyps, canon)
    decl = next(r for r in ranked if r.hyp.kind == DECLARED)
    best = ranked[0]
    boot_cache: dict[tuple, float] = {}

    def beats(a: Ranked, b: Ranked) -> float:
        key = (a.signature, b.signature)
        if key not in boot_cache:
            boot_cache[key] = bootstrap(x, y_of, a.hyp.phi, b.hyp.phi, n_boot)
        return boot_cache[key]

    plausible = [best] + [r for r in ranked[1:PLAUSIBLE_POOL + 1] if beats(best, r) < PLAUSIBLE]
    gain = best.score - decl.score
    # thousands of edits are tried, so one of them always edges the declared
    # mapping out by noise in a run of small changes: to call the declared
    # mapping wrong, a mapping must beat it by more than its edits cost
    # (EDIT_PENALTY each), by MIN_GAIN, and in CONFIDENT of the resamples
    contender = max(ranked, key=lambda r: (r.score - EDIT_PENALTY * edits(r.hyp),
                                           -edits(r.hyp)))
    decl_ok = (contender is decl or contender.score - decl.score < MIN_GAIN
               or beats(contender, decl) < CONFIDENT)
    answer = decl if decl_ok else contender
    if answer is decl:
        runner = next((r for r in ranked if r is not decl), None)
    else:
        runner = next((r for r in ranked if r is not answer
                       and r.hyp.kind != answer.hyp.kind), None)
    margin = answer.score - runner.score if runner else float("inf")
    confidence = beats(answer, runner) if runner else 1.0

    states = set(canon[1:])
    # the dupli question is answered with the same prior: an edit of the
    # declared mapping has to be worth what it costs
    complete = sorted((r for r in ranked if set(r.signature) >= states),
                      key=lambda r: (-(r.score - EDIT_PENALTY * edits(r.hyp)), edits(r.hyp)))
    complete_conf = consistent = decided = None
    proposal = None
    if complete:
        c0 = complete[0]
        paid = lambda r: r.score - EDIT_PENALTY * edits(r.hyp)  # noqa: E731
        consistent = (c0 is best or c0 in plausible or paid(c0) >= paid(best)
                      or beats(best, c0) < PLAUSIBLE)
        if len(complete) > 1:
            complete_conf = beats(c0, complete[1])
        proposal = declaration(c0.hyp.phi, folders, n_rows, dupli)
    if dupli:
        ahead = True
        if len(complete) > 1:
            c0, c1 = complete[0], complete[1]
            worth = (c0.score - EDIT_PENALTY * edits(c0.hyp)) - (c1.score - EDIT_PENALTY * edits(c1.hyp))
            # clearly ahead of the next complete mapping: by MIN_GAIN after the
            # edits are paid for, and either with fewer edits or in most resamples
            ahead = worth >= MIN_GAIN and (edits(c1.hyp) > edits(c0.hyp)
                                           or (complete_conf or 0.0) >= PLAUSIBLE)
        decided = bool(complete) and bool(consistent) and proposal is not None \
            and proposal.expressible and ahead

    if best.score < POOR:
        verdict, flagged = "unknown", True
        note = (f"the best mapping correlates only {best.score:.2f}: the change "
                f"sequences barely agree; look at the frames")
    elif decl_ok:
        verdict = "aligned"
        # an edit that is worth its cost but wins too few resamples: say so
        flagged = contender is not decl and contender.score - decl.score >= MIN_GAIN
        if decl is best:
            note = "as declared"
        elif flagged:
            note = (f"'{contender.hyp.label}' scores {contender.score - decl.score:+.3f} -- "
                    f"more than its edits cost -- but wins only "
                    f"{beats(contender, decl):.2f} of the resamples")
        else:
            note = (f"the best edit, '{best.hyp.label}', scores {gain:+.3f}: within what "
                    f"{edits(best.hyp)} edit(s) cost ({EDIT_PENALTY:.2f} each)")
    else:
        verdict, flagged = answer.hyp.kind, True
        where = ", ".join(r.hyp.label for r in plausible[:6]
                          if r.hyp.kind == answer.hyp.kind and r is not answer)
        note = (f"'{answer.hyp.label}' beats the declared mapping by "
                f"{answer.score - decl.score:+.3f} (bootstrap {beats(answer, decl):.2f})"
                + (f"; as plausible: {where}" if where else ""))
    return Judgement(ranked, decl, best, plausible, answer, runner, margin, confidence,
                     gain, verdict, flagged, note, complete, complete_conf, consistent,
                     proposal, decided)


# --------------------------------------------------------------------------- #
# I/O: sequences from the index, frames from the raw drive
# --------------------------------------------------------------------------- #
@dataclass
class Sequences:
    """The pictures one desktop's alignment is judged on."""

    desktop: int
    n_rows: int
    row_paths: dict[str, list[str]]       # view -> image path per uncollapsed row
    folders: list[int]                    # scanner folder numbers, sorted
    folder_paths: list[str]
    declared: list[int]                   # row per folder, as the index maps it
    dupli_rows: list[int]
    rescan_rows: list[int] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)   # folders the index skipped


def rows_of_index(di, view: str) -> tuple[list[Optional[str]], dict[int, int],
                                          dict[int, list[int]]]:
    """``(path per row, main row of each step, dupli rows of each step)``.

    Works on a collapsed index (``di.rows`` / ``di.dupli_rows`` set, extra shots
    in ``aux["extra_shots"]``) and on one written before U25, where step = row.
    """
    from tda.core.index import collapse_rows

    dupli = list(getattr(di, "dupli_rows", None) or [])
    n_rows = int(getattr(di, "n_rows", 0) or 0) or di.n_steps + len(dupli)
    step_of_row, _extra = collapse_rows(n_rows, dupli)
    main_row: dict[int, int] = {}
    extras: dict[int, list[int]] = {}
    for r in range(1, n_rows + 1):
        if r in _extra:
            extras.setdefault(step_of_row[r], []).append(r)
        else:
            main_row.setdefault(step_of_row[r], r)
    paths: list[Optional[str]] = [None] * (n_rows + 1)
    for key, ff in di.frames.items():
        if key.view != view:
            continue
        rows = sorted(([main_row[key.step]] if key.step in main_row else [])
                      + (extras.get(key.step) or []))
        shots = [s for s in (ff.aux or {}).get("extra_shots") or [] if isinstance(s, dict)]
        named = {int(s["row"]) for s in shots if s.get("row") is not None}
        for s in shots:
            if s.get("row") is not None and 1 <= int(s["row"]) <= n_rows:
                paths[int(s["row"])] = s.get("path")
        # the step's own frame is the row no extra shot names (a dupli capture
        # that replaced a bad one is the frame, the earlier row the extra shot)
        free = [r for r in rows if r not in named]
        if free:
            paths[free[0]] = ff.path
            for r, s in zip(free[1:], [s for s in shots if s.get("row") is None]):
                paths[r] = s.get("path")
    return paths[1:], main_row, extras


def declared_rows(di, main_row: dict[int, int],
                  extras: dict[int, list[int]]) -> dict[str, int]:
    """Scanner folder name -> the row the index puts it on today.

    The folders of one step (its frame and its extra shots) take the step's
    rows in folder order -- the scanner photographs in time order, whichever
    folder the index made the frame -- and a folder beyond the step's rows (a
    re-shot no ``dupli`` row explains) shares the last one.
    """
    out: dict[str, int] = {}
    for key, ff in di.frames.items():
        if key.view != "scan":
            continue
        names = [ff.src_step_dir] + [s.get("src_step_dir") for s in
                                     (ff.aux or {}).get("extra_shots") or []
                                     if isinstance(s, dict) and s.get("src_step_dir")]
        rows = sorted(([main_row[key.step]] if key.step in main_row else [key.step])
                      + (extras.get(key.step) or []))
        by_number = sorted(names, key=lambda n: int(n[3:-1]) if n[3:-1].isdigit() else 0)
        for i, name in enumerate(by_number):
            out[name] = rows[min(i, len(rows) - 1)]
    return out


def scanner_folders(scanner_root: str, desktop: int,
                    issues: Optional[list[str]] = None) -> list[tuple[int, str, str]]:
    """``(number, folder name, main image)`` of every ``RGB<n>1`` folder, by number."""
    from tda.core.sources import BURST_RE, dirs, files, scanner_desktop_dir

    issues = issues if issues is not None else []
    base = scanner_desktop_dir(scanner_root, desktop, issues)
    if base is None:
        return []
    out = []
    for entry in dirs(base, issues):
        name = entry.name
        if not (name.startswith("RGB") and name.endswith("1") and name[3:-1].isdigit()):
            continue
        shots = sorted((int(m.group(1)), f.path) for f in files(entry.path, issues)
                       if (m := BURST_RE.match(f.name)))
        if shots:
            main = next((p for k, p in shots if k == 0), shots[0][1])
            out.append((int(name[3:-1]), name, main.replace("\\", "/")))
    return sorted(out)


def db_roi(db_path: Optional[str], desktop: int, view: str) -> Optional[list[int]]:
    """The union of a view's pose-segment ROIs, read with the database opened read-only."""
    if not db_path or not os.path.isfile(db_path):
        return None
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        try:
            rows = con.execute("SELECT roi_json FROM pose_segment WHERE desktop=? AND view=? "
                               "AND roi_json IS NOT NULL", (int(desktop), view)).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    boxes = [json.loads(r[0]) for r in rows if r[0]]
    boxes = [b for b in boxes if isinstance(b, list) and len(b) == 4]
    if not boxes:
        return None
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]


#: How much each view is shrunk while being decoded (OpenCV reduced decoding).
DECODE_REDUCE = {"scan": 2, "oak1": 4, "oak2": 4, "rs": 1}


def _read(path: str, reduce: int, colour: bool = False):
    import cv2

    flags = {(1, False): cv2.IMREAD_GRAYSCALE, (2, False): cv2.IMREAD_REDUCED_GRAYSCALE_2,
             (4, False): cv2.IMREAD_REDUCED_GRAYSCALE_4, (8, False): cv2.IMREAD_REDUCED_GRAYSCALE_8,
             (1, True): cv2.IMREAD_COLOR, (2, True): cv2.IMREAD_REDUCED_COLOR_2,
             (4, True): cv2.IMREAD_REDUCED_COLOR_4, (8, True): cv2.IMREAD_REDUCED_COLOR_8}
    img = cv2.imread(path, flags[(reduce, colour)])
    if img is None:
        raise OSError(f"cannot read {path}")
    return img


def auto_roi(paths: Sequence[str], view: str) -> Optional[list[int]]:
    """The chassis box :func:`tda.core.cache.suggest_roi_over` proposes, full-res px."""
    from tda.core.cache import suggest_roi_over

    picks = [p for p in (paths[0], paths[len(paths) // 2], paths[-1]) if p]
    if not picks:
        return None
    reduce = DECODE_REDUCE.get(view, 1)
    detector = "scan" if view == "scan" else "oak1"
    imgs = [_read(p, reduce, colour=True) for p in picks]
    box = suggest_roi_over(imgs, detector)
    h, w = imgs[0].shape[:2]
    if tuple(box) == (0, 0, w, h):
        return None
    return [int(v * reduce) for v in box]


def load_frames(paths: Sequence[Optional[str]], view: str, roi: Optional[Sequence[int]],
                cache_dir: Optional[str] = None, tag: str = "") -> list[Optional[np.ndarray]]:
    """Each path's frame cut to ``roi``, grey, at most :data:`LONG_SIDE` px (uint8).

    With ``cache_dir`` the result is kept in one ``.npz`` per (tag, view, ROI)
    and re-used while the paths are the same.
    """
    import cv2

    key = "_".join(str(int(v)) for v in roi) if roi else "whole"
    cache = (os.path.join(cache_dir, f"{tag}_{view}_{key}_{LONG_SIDE}.npz")
             if cache_dir else None)
    want = [p or "" for p in paths]
    if cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            if list(z["paths"]) == want:
                return [z[f"f{i}"] if want[i] else None for i in range(len(want))]
    reduce = DECODE_REDUCE.get(view, 1)
    out: list[Optional[np.ndarray]] = []
    for p in paths:
        if not p:
            out.append(None)
            continue
        img = _read(p, reduce)
        if roi:
            x0, y0, x1, y1 = (int(round(v / reduce)) for v in roi)
            img = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        h, w = img.shape[:2]
        s = LONG_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_AREA)
        out.append(np.ascontiguousarray(img, dtype=np.uint8))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        arrays = {f"f{i}": a for i, a in enumerate(out) if a is not None}
        np.savez_compressed(cache, paths=np.array(want), **arrays)
    return out


class ChangeTable:
    """``y_of(a, b)``: the change between rows ``a`` and ``b`` of one view, cached."""

    def __init__(self, frames: Sequence[Optional[np.ndarray]]) -> None:
        self.frames = [normalise(f) if f is not None else None for f in frames]
        self._cache: dict[tuple[int, int], float] = {}

    def __call__(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in self._cache:
            fa, fb = self.frames[key[0] - 1], self.frames[key[1] - 1]
            self._cache[key] = float("nan") if fa is None or fb is None else change(fa, fb)
        return self._cache[key]

    def complete(self) -> bool:
        return all(f is not None for f in self.frames)


# --------------------------------------------------------------------------- #
# one desktop, end to end
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    """One desktop's scanner judged against one reference view."""

    desktop: int
    view: str
    n_rows: int
    folders: list[int]
    folder_names: list[str]
    folder_paths: list[str]
    row_paths: list[Optional[str]]
    declared: list[int]
    dupli_rows: list[int]
    rescan_rows: list[int]
    x: list[float]
    table: ChangeTable
    judgement: Judgement
    roi: dict[str, tuple[Optional[list[int]], str]]
    notes: list[str] = field(default_factory=list)


#: Words in a sheet note that say a capture was redone or is missing.
RESCAN_WORDS = ("rescan", "re-scan", "re scan", "missing", "skip", "redo", "retake",
                "补拍", "重拍", "重扫", "漏")


def rescan_rows(notes: Sequence[str]) -> list[int]:
    """Sheet rows whose note mentions a rescan, a redo, a skip or a missing shot."""
    return [i for i, text in enumerate(notes, start=1)
            if any(w in (text or "").lower() for w in RESCAN_WORDS)]


def _roi(db_path: Optional[str], desktop: int, view: str,
         paths: Sequence[Optional[str]], auto: bool) -> tuple[Optional[list[int]], str]:
    box = db_roi(db_path, desktop, view)
    if box:
        return box, "db"
    if auto:
        try:
            found = auto_roi([p for p in paths if p], view)
        except (OSError, ValueError):
            found = None
        if found:
            return found, "auto"
    return None, "whole"


def analyse(desktop: int, di, *, scanner_root: str, sheet=None,
            db_path: Optional[str] = None, view: str = "oak1",
            cache_dir: Optional[str] = None, pairs: bool = True,
            n_boot: int = BOOTSTRAP, auto: bool = True) -> Analysis:
    """Judge one desktop's scanner folders against ``view`` (oak1, oak2 or rs).

    ``di`` is the desktop's :class:`~tda.core.index.DesktopIndex` (a collapsed
    index or one written before U25), ``sheet`` its
    :class:`~tda.core.logs.SheetRows` (the ``dupli`` rows when the index does not
    carry them, and the notes). Paths in the index are resolved through
    :func:`tda.core.rawroot.resolve_raw` when a raw root is configured.
    """
    from tda.core import rawroot

    def raw(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return rawroot.resolve_raw(p) or p

    notes: list[str] = []
    row_paths_rec, main_row, extras = rows_of_index(di, view)
    row_paths = [raw(p) for p in row_paths_rec]
    n_rows = len(row_paths)
    dupli = list(getattr(di, "dupli_rows", None) or [])
    if not dupli and sheet is not None:
        dupli = [r for r in sheet.dupli_rows if r <= n_rows]
    if sheet is not None and sheet.n_rows != n_rows:
        notes.append(f"the sheet has {sheet.n_rows} rows, the index {n_rows} "
                     f"{view} captures")
    rescans = rescan_rows(sheet.notes) if sheet is not None else []

    found = scanner_folders(scanner_root, desktop, notes)
    by_name = declared_rows(di, main_row, extras)
    folders = [n for n, _name, _p in found]
    names = [name for _n, name, _p in found]
    fpaths = [p for _n, _name, p in found]
    declared = [by_name.get(name) for name in names]
    if any(r is None for r in declared):
        # the index left folders out (more folders than steps, nothing declared):
        # judge the scanner's own numbering instead, folder n on row n
        left = [name for name, r in zip(names, declared) if r is None]
        notes.append(f"the index places no step for {left}; the declared mapping is "
                     f"taken as folder n = row n")
        declared = [min(int(n), n_rows) for n in folders]
    if len(folders) < 4:
        raise ValueError(f"D{desktop:02d}: only {len(folders)} scanner folders to judge")

    scan_roi = _roi(db_path, desktop, "scan", fpaths, auto)
    ref_roi = _roi(db_path, desktop, view, row_paths, auto)
    tag = f"D{desktop:02d}"
    scan_frames = load_frames(fpaths, "scan", scan_roi[0], cache_dir, tag)
    ref_frames = load_frames(row_paths, view, ref_roi[0], cache_dir, tag)
    scan_norm = [normalise(f) for f in scan_frames]
    x = [change(scan_norm[i], scan_norm[i + 1]) for i in range(len(scan_norm) - 1)]
    table = ChangeTable(ref_frames)
    if not table.complete():
        notes.append(f"{view} has no frame for rows "
                     f"{[i + 1 for i, f in enumerate(ref_frames) if f is None]}")
    j = judge(x, table, declared, folders, n_rows, dupli, pairs=pairs, n_boot=n_boot)
    return Analysis(desktop, view, n_rows, folders, names, fpaths, row_paths, declared,
                    dupli, rescans, x, table, j,
                    {"scan": scan_roi, view: ref_roi}, notes)


def _tags(h: Hypothesis, dupli: Sequence[int], rescans: Sequence[int]) -> str:
    tags = []
    if h.skip_row is not None:
        if h.skip_row in dupli:
            tags.append("dupli row")
        if h.skip_row in rescans:
            tags.append("'rescan' note")
    return ", ".join(tags)


def describe(a: Analysis, top: int = 12) -> str:
    """The ranked table, the verdict and the proposed declaration, as text."""
    j = a.judgement
    lines = [
        f"scan-align D{a.desktop:02d} vs {a.view}: {len(a.folders)} scanner folders, "
        f"{a.n_rows} {a.view} captures; sheet dupli rows {a.dupli_rows or 'none'}, "
        f"notes mentioning a rescan/redo/skip on rows {a.rescan_rows or 'none'}",
        "ROI " + "; ".join(f"{v}: {src} {box if box else ''}".rstrip()
                           for v, (box, src) in a.roi.items()),
    ]
    lines += [f"note: {n}" for n in a.notes]
    lines += ["", " rank  score   vs best  hypothesis"]
    for i, r in enumerate(j.ranked[:top], start=1):
        tag = _tags(r.hyp, a.dupli_rows, a.rescan_rows)
        lines.append(f" {i:4d}  {r.score:6.3f}  {r.score - j.best.score:+7.3f}  "
                     f"{r.hyp.label}{f'  [{tag}]' if tag else ''}")
    decl_rank = j.ranked.index(j.declared) + 1
    if decl_rank > top:
        lines.append(f" {decl_rank:4d}  {j.declared.score:6.3f}  "
                     f"{j.declared.score - j.best.score:+7.3f}  as declared")
    lines.append("")
    others = [r.hyp.label for r in j.plausible if r is not j.best][:8]
    if others:
        lines.append(f"as plausible as the best (bootstrap < {PLAUSIBLE:.2f}): "
                     + "; ".join(others))
    runner = j.runner_up
    lines.append(
        f"verdict: {j.verdict.upper()} -- {j.note}; answer '{j.answer.hyp.label}' "
        f"{j.answer.score:.3f}, runner-up '{runner.hyp.label if runner else '-'}' "
        f"{runner.score if runner else float('nan'):.3f} (margin {j.margin:+.3f}, "
        f"bootstrap {j.confidence:.2f}){'  ** FLAGGED **' if j.flagged else ''}")
    if a.dupli_rows:
        if j.complete:
            best = j.complete[0]
            p = j.proposal
            conf = (f"; next '{j.complete[1].hyp.label}' {j.complete[1].score:.3f}, "
                    f"bootstrap {j.complete_confidence:.2f}" if len(j.complete) > 1 else
                    "; no other mapping photographs every step")
            caught = ", ".join(f"row {r} as folder {n}"
                               for r, n in sorted(p.captured_dupli.items()))
            lines.append(
                f"dupli rows: skipped {p.skipped_dupli or 'none'}, captured "
                f"{caught or 'none'} -- '{best.hyp.label}' {best.score:.3f}{conf}; "
                f"{'plausible' if j.complete_consistent else 'NOT plausible'} against the "
                f"best mapping")
            lines.append(f"declaration: scan.{a.desktop}.duplicate_folders: "
                         f"{p.duplicate_folders}"
                         + (f", scan.{a.desktop}.skipped_steps: {p.skipped_steps}"
                            if p.skipped_steps else "")
                         + (f"   (cannot express: {'; '.join(p.unexpressed[:3])})"
                            if p.unexpressed else "")
                         + ("   -- decided" if j.decided else "   -- NOT decided: look "
                            "at the frames"))
            if p.missing_steps:
                lines.append(f"  steps without a scanner frame: {p.missing_steps}")
        else:
            lines.append("dupli rows: no mapping photographs every step "
                         f"({len(a.folders)} folders) -- NOT decided")
    return "\n".join(lines)


def render_strip(a: Analysis, folders: Sequence[int], rows: Sequence[int], out_path: str,
                 height: int = 210, per_line: int = 8) -> str:
    """Scanner folders (by number) above reference-view rows, ROI crops, labelled:
    the window a human reads to say which state each scanner folder shows."""
    import cv2

    scan_box, _ = a.roi.get("scan", (None, ""))
    ref_box, _ = a.roi.get(a.view, (None, ""))
    by_number = dict(zip(a.folders, a.folder_paths))

    def tile(path: Optional[str], view: str, box, label: str) -> np.ndarray:
        if path:
            reduce = DECODE_REDUCE.get(view, 1)
            img = _read(path, reduce, colour=True)
            if box:
                x0, y0, x1, y1 = (int(round(v / reduce)) for v in box)
                img = img[max(0, y0):y1, max(0, x0):x1]
            s = height / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), height),
                             interpolation=cv2.INTER_AREA)
        else:
            img = np.full((height, height, 3), 40, np.uint8)
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        return np.pad(img, ((0, 4), (0, 4), (0, 0)))

    tiles = ([tile(by_number.get(int(n)), "scan", scan_box, f"scan RGB{n}1") for n in folders],
             [tile(a.row_paths[r - 1] if 1 <= r <= len(a.row_paths) else None, a.view,
                   ref_box, f"{a.view} row {r}" + (" dupli" if r in a.dupli_rows else ""))
              for r in rows])
    lines = []
    for group in tiles:
        for i in range(0, len(group), per_line):
            lines.append(np.hstack(group[i:i + per_line]))
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1]), (0, 0))) for line in lines]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(lines))
    return out_path


def render_changes(pairs: Sequence[tuple[str, str, str, str]], view: str,
                   roi: Optional[Sequence[int]], out_path: str, size: int = 190) -> str:
    """For each ``(label_a, path_a, label_b, path_b)``: the two places the frames
    differ most (box-filtered absolute difference of the normalised grey
    images), cropped side by side, plus both whole frames -- what a human needs
    to say "a screw went" or "only a cable moved / the light changed"."""
    import cv2

    reduce = 1 if view == "scan" else 2
    lines = []
    for la, pa, lb, pb in pairs:
        ia, ib = _read(pa, reduce, colour=True), _read(pb, reduce, colour=True)
        if roi:
            x0, y0, x1, y1 = (int(v / reduce) for v in roi)
            ia, ib = ia[y0:y1, x0:x1], ib[y0:y1, x0:x1]
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=cv2.INTER_AREA)
        ga = normalise(cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY))
        gb = normalise(cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY))
        diff = cv2.boxFilter(np.abs(ga - gb), -1, (25, 25))
        tiles = []
        r = max(40, min(ia.shape[:2]) // 8)
        for _ in range(2):
            y, x = np.unravel_index(int(np.argmax(diff)), diff.shape)
            value = float(diff[y, x])
            win = (slice(max(0, y - r), y + r), slice(max(0, x - r), x + r))
            for img, lab in ((ia, la), (ib, lb)):
                crop = cv2.resize(img[win].copy(), (size, size), interpolation=cv2.INTER_AREA)
                cv2.putText(crop, f"{lab} {value:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(np.pad(crop, ((0, 3), (0, 3), (0, 0))))
            tiles.append(np.full((size + 3, 10, 3), 255, np.uint8))
            diff[max(0, y - 2 * r):y + 2 * r, max(0, x - 2 * r):x + 2 * r] = 0
        for img in (ia, ib):
            w = max(1, int(img.shape[1] * size / img.shape[0]))
            tiles.append(np.pad(cv2.resize(img, (w, size), interpolation=cv2.INTER_AREA),
                                ((0, 3), (0, 3), (0, 0))))
        lines.append(np.hstack(tiles))
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1]), (0, 0))) for line in lines]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(lines))
    return out_path


def disputed_pairs(a: Analysis, other: Sequence[int], limit: int = 8) -> list[int]:
    """Folder-pair indices where ``other`` and the declared mapping disagree,
    biggest scanner change first (all pairs when they agree)."""
    differ = [i for i in range(len(a.x)) if (a.declared[i], a.declared[i + 1])
              != (other[i], other[i + 1])] or list(range(len(a.x)))
    return sorted(differ, key=lambda i: -a.x[i])[:limit]


def decisive_pair(a: Analysis, other: Sequence[int]) -> int:
    """The folder pair (index ``i``: folders ``i``, ``i+1``) that best tells the
    declared mapping from ``other``: the largest scanner change among the pairs
    the two map differently, or the largest overall when they agree."""
    differ = [i for i in range(len(a.x)) if (a.declared[i], a.declared[i + 1])
              != (other[i], other[i + 1])]
    pool = differ or list(range(len(a.x)))
    return max(pool, key=lambda i: a.x[i])


def render_decisive(a: Analysis, out_path: str, phis: dict[str, Sequence[int]],
                    pair: Optional[int] = None, height: int = 300) -> str:
    """A montage: the two scanner folders of ``pair`` beside the reference-view
    rows each mapping in ``phis`` pairs them with (ROI crops, labelled)."""
    import cv2

    i = pair if pair is not None else decisive_pair(a, next(iter(phis.values())))
    scan_box, _ = a.roi.get("scan", (None, ""))
    ref_box, _ = a.roi.get(a.view, (None, ""))

    def tile(path: Optional[str], view: str, box, label: str) -> np.ndarray:
        if not path:
            img = np.zeros((height, height, 3), np.uint8)
        else:
            reduce = DECODE_REDUCE.get(view, 1)
            img = _read(path, reduce, colour=True)
            if box:
                x0, y0, x1, y1 = (int(round(v / reduce)) for v in box)
                img = img[max(0, y0):y1, max(0, x0):x1]
            s = height / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), height),
                             interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return img

    rows = [[tile(a.folder_paths[k], "scan", scan_box, f"scan {a.folder_names[k]}")
             for k in (i, i + 1)]]
    for name, phi in phis.items():
        rows.append([tile(a.row_paths[phi[k] - 1], a.view, ref_box,
                          f"{a.view} row {phi[k]} ({name})") for k in (i, i + 1)])
    strips = [np.hstack([np.pad(t, ((0, 0), (0, 8), (0, 0))) for t in r]) for r in rows]
    width = max(s.shape[1] for s in strips)
    lines = [np.pad(s, ((0, 8), (0, width - s.shape[1]), (0, 0))) for s in strips]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(lines))
    return out_path
