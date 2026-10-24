"""An empty visible mask: expected, or a problem? (task U26)

The compiler notes ``empty_visible:<instance>`` for every mask instance whose
visible mask came out empty with no frame override on it
(:mod:`tda.core.compiler`). The window shows it as a note -- 「这一帧里它被完全挡住了」
-- and ``cli check`` used to count every one as a problem: 100 to 300 rows per
fully confirmed desktop on the scanner (D3 232, D7 292, D11 265), RAM latches
under the drive cage, the socket lever under the cooler, screws under the cage.
Those parts are covered completely by labelled parts above them, which is what
layered amodal masks are for; the row's label is ``occluded_full``, computed.
Two hundred lines of that buried the few that mean something, and ``check``
exited 1 on every finished desktop.

The rule, one per row:

* **expected** (:data:`COVERED`) -- the visibility is ``occluded_full`` and the
  part's amodal shape has pixels on this frame: something covers all of it --
  labelled parts, a hand or tool occluder mask, or a label someone set;
* **expected** (:data:`OUT_OF_VIEW`) -- the part is labelled ``out_of_view``
  (「看不见」): nothing of it is in this picture, by the annotator's word;
* **a problem** (:data:`NO_SHAPE`) -- the amodal shape itself is empty here:
  nothing was drawn, or what was drawn lies wholly outside the frame after the
  pose transform. Then there is nothing to be covered, and the label the
  ladder falls back to (``too_small``) is not an explanation;
* **a problem** (:data:`UNEXPLAINED`) -- anything else: the label still claims
  something is seen (``visible``, ``occluded_partial``, ...) where nothing is.

The compiler only notes rows without a frame override, so its labels are
computed and the last case cannot arise from it today; it is in the rule so
that the rule is complete, not because ``check`` sees it.

``cli check`` prints the expected rows as one line, 「n 个零件完全被挡住（正常）」,
and neither lists nor counts them as problems; the window, ``status`` and the
compiler's own problem list are unchanged. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass

from tda.core.compiler import CompiledFrame, CompiledInstance
from tda.core.model import Visibility

__all__ = [
    "COVERED",
    "EMPTY_VISIBLE",
    "EXPECTED",
    "NO_SHAPE",
    "OUT_OF_VIEW",
    "UNEXPLAINED",
    "EmptyVisible",
    "classify",
    "empty_visible",
    "split_problems",
]

#: The compiler's problem code (``empty_visible:<instance>``).
EMPTY_VISIBLE = "empty_visible"
_PREFIX = EMPTY_VISIBLE + ":"

#: Covered completely by something: labelled parts, a hand / tool, a label.
COVERED = "covered"
#: Labelled 「看不见」: not in this picture at all.
OUT_OF_VIEW = Visibility.OUT_OF_VIEW.value
#: The part's amodal shape is empty on this frame.
NO_SHAPE = "no_shape"
#: The label says something is seen and nothing is.
UNEXPLAINED = "unexplained"
#: The reasons that make an empty visible mask what it should be.
EXPECTED = frozenset({COVERED, OUT_OF_VIEW})

_FULL = Visibility.OCCLUDED_FULL.value


@dataclass(frozen=True)
class EmptyVisible:
    """One ``empty_visible`` row of one frame, with the reason it is empty."""

    step: int
    instance: str
    reason: str
    visibility: str

    @property
    def expected(self) -> bool:
        """Is this what the frame should show (not a problem)?"""
        return self.reason in EXPECTED

    def problem(self) -> str:
        """The line ``check`` lists for a row that is a problem."""
        head = f"{EMPTY_VISIBLE}:{self.instance} (step {self.step})"
        if self.reason == NO_SHAPE:
            return f"{head}: 形状在这一帧是空的（没画，或整个在画面外）"
        return f"{head}: 什么都看不见，可见性却是 {self.visibility}"


def _has_pixels(found: CompiledInstance) -> bool:
    """Does the instance's amodal shape have a pixel on this frame?

    Asked inside ``amodal_window`` when there is one: the shape is known to be
    empty outside it, and a 12 MP ``any()`` per row is what this avoids.
    """
    amodal = found.amodal
    if amodal is None:
        return False
    window = found.amodal_window
    if window is None:
        return bool(amodal.any())
    x0, y0, x1, y1 = window
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(amodal[y0:y1, x0:x1].any())


def classify(found: CompiledInstance) -> str:
    """The reason one instance's visible mask is empty (see the module docstring)."""
    label = str(found.visibility)
    if label == OUT_OF_VIEW:
        return OUT_OF_VIEW
    if not _has_pixels(found):
        return NO_SHAPE
    if label == _FULL:
        return COVERED
    return UNEXPLAINED


def empty_visible(compiled: CompiledFrame, step: int | None = None) -> list[EmptyVisible]:
    """One entry per ``empty_visible`` problem of ``compiled``, in its order.

    ``step`` defaults to the frame's own. An instance the problem names but
    the compilation does not hold (it never happens; the compiler names only
    what it compiled) is a problem with no shape.
    """
    at = int(compiled.key.step if step is None else step)
    out: list[EmptyVisible] = []
    for text in compiled.problems:
        if not text.startswith(_PREFIX):
            continue
        instance = text[len(_PREFIX):]
        found = compiled.instances.get(instance)
        if found is None:
            out.append(EmptyVisible(at, instance, NO_SHAPE, ""))
            continue
        out.append(EmptyVisible(at, instance, classify(found), str(found.visibility)))
    return out


def split_problems(problems: list[str], rows: list[EmptyVisible]
                   ) -> tuple[list[str], list[EmptyVisible]]:
    """``(problems, expected rows)``: the expected ``empty_visible`` lines taken out.

    ``problems`` is a flat list as :meth:`tda.core.truth.TruthService.refresh_range`
    builds it, frame after frame, and ``rows`` the :func:`empty_visible` entries
    of the same frames in the same order, so the n-th ``empty_visible:`` line
    is the n-th row. An expected row's line is dropped, a problem's is replaced
    by :meth:`EmptyVisible.problem` (which says where and why); every other line
    is kept where it was. A line with no row to match -- a frame whose
    compilation was not handed back -- is kept as it is.
    """
    pending = list(rows)
    kept: list[str] = []
    expected: list[EmptyVisible] = []
    for text in problems:
        if not text.startswith(_PREFIX) or not pending:
            kept.append(text)
            continue
        row = pending[0]
        if row.instance != text[len(_PREFIX):]:
            kept.append(text)
            continue
        pending.pop(0)
        if row.expected:
            expected.append(row)
        else:
            kept.append(row.problem())
    return kept, expected
