"""「可见性审查」: the visibility labels, audited once at the end (task U24).

While annotating, nobody deals with visibility labels any more: the compiler
derives them (spec 3.3 step 7) and only the cable key (`` ` ``) is said while
drawing.  At the end, in Review mode, every ``(frame, part)`` of a **confirmed**
frame whose label is not plain ``visible`` -- a computed occlusion, ``too_small``
/ ``visible_tiny`` / ``out_of_view``, a 被线挡住 label, any label set by hand --
is shown once, and the annotator either confirms it as it is or says what it
really is.  One answer the old keys could not give: **角度看不到**, the camera
cannot see the part from where it stands (``occluder = "viewpoint"``), which is
a different reason from a cable in front of it.

This module is the rule, without Qt and without the session:

* the **stamp** -- what a verdict leaves in ``instance_frame_flags``
  (:func:`make_stamp`, :func:`with_stamp`) and when a stamp still covers a row
  (:func:`stamp_covers`): it records the label and the reason it was given for,
  so a row whose label changed afterwards (a conflict accepted, a re-trace)
  comes back to the queue instead of hiding behind an old answer;
* the **rows** -- :func:`review_rows` splits a view's candidates into the ones
  still to review and the ones done (:func:`progress` is ``status``'s count);
* the **occluders** -- :func:`occluders_on_frame` names the parts that hide a
  computed occlusion: a part of the same paint group (placement) whose
  confirmed visible pixels lie inside this part's shape took them, which is
  exactly what the compiler's layer order and pairwise overrides decided
  (plus the frame's hand / tool occluder layers).  Off the stored rows and the
  keyframe in force, in run-length space where the transform is the identity
  -- no compilation;
* the **runs** -- :func:`runs_of` compresses one part's rows over consecutive
  annotatable frames with the same label, reason, hand/auto and occluders into
  one queue entry (「ram_latch.03 第 1–13 帧 · occluded_full（被硬盘架挡住）」).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from tda.core import masks
from tda.core.compiler import (
    _empty_window,
    _intersect_window,
    _part_mask,
    _part_window,
    _wrong_size,
    select_keyframe,
)
from tda.core.model import FrameKey, Similarity, Visibility
from tda.core.truth_inputs import (
    InputCache,
    _keyframes_of,
    annotatable_steps,
    pose_segment_of,
    view_steps,
)

__all__ = [
    "OCCLUDED_LABELS",
    "STAMP_KEY",
    "STAMP_KEYS",
    "VERDICTS",
    "VERDICT_CABLE",
    "VERDICT_COMPUTED",
    "VERDICT_LABEL",
    "VERDICT_VIEWPOINT",
    "LAYER_PREFIX",
    "Run",
    "make_stamp",
    "occluders_on_frame",
    "order_of",
    "progress",
    "review_rows",
    "runs_of",
    "stamp_covers",
    "with_stamp",
]

#: ``Space`` / ``Shift+Space``: the label is right as the frame has it.
VERDICT_COMPUTED = "computed"
#: A digit (or ``Ctrl+V``): a bare label set by hand, no reason.
VERDICT_LABEL = "label"
#: `` ` `` / ``Shift+```: 被线挡住, 完全 / 部分 (task U15's label).
VERDICT_CABLE = "cable"
#: ``Z`` / ``Shift+Z``: 角度看不到 -- ``occluder = "viewpoint"``.
VERDICT_VIEWPOINT = "viewpoint"
VERDICTS: tuple[str, ...] = (VERDICT_COMPUTED, VERDICT_LABEL, VERDICT_CABLE,
                             VERDICT_VIEWPOINT)

#: The flag that says a row was reviewed; the other keys say by whom, when, how,
#: and **what** -- the label and the reason the answer was given for.
STAMP_KEY = "vis_reviewed"
STAMP_KEYS: tuple[str, ...] = (STAMP_KEY, "by", "at", "verdict", "visibility", "occluder")

#: The labels whose occluding parts are named (a part is hidden *by* something).
OCCLUDED_LABELS = frozenset({Visibility.OCCLUDED_PARTIAL.value,
                             Visibility.OCCLUDED_FULL.value})
#: How an occluder **layer** (hand, tool -- spec 3.1 ``OccluderMask``) is named
#: among the parts that hide one: ``@hand``.
LAYER_PREFIX = "@"


# --------------------------------------------------------------------------- #
# the stamp
# --------------------------------------------------------------------------- #
def make_stamp(annotator: str, verdict: str, visibility: Optional[str],
               occluder: Optional[str], at: Optional[str] = None) -> dict:
    """The keys a verdict writes into the row's flags object."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    return {
        STAMP_KEY: True,
        "by": str(annotator),
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "visibility": None if visibility is None else str(visibility),
        "occluder": None if not occluder else str(occluder),
    }


def with_stamp(flags: Optional[dict], stamp: Optional[dict]) -> Optional[dict]:
    """``flags`` with its stamp keys replaced by ``stamp``'s; ``None`` when nothing is left.

    Any other key of the object (spec 3.1's difficulty flags, one day) stays.
    """
    out = {k: v for k, v in (flags or {}).items() if k not in STAMP_KEYS}
    if stamp:
        out.update({k: stamp[k] for k in STAMP_KEYS if k in stamp})
    return out or None


def stamp_covers(flags: Optional[dict], visibility: Optional[str],
                 occluder: Optional[str]) -> bool:
    """Does the row's stamp answer for the label and the reason it has now?

    A stamp that does not record what it was given for (written by hand, or by
    something else) is taken at its word.
    """
    if not flags or flags.get(STAMP_KEY) is not True:
        return False
    if "visibility" in flags and flags.get("visibility") != visibility:
        return False
    if "occluder" in flags and (flags.get("occluder") or None) != (occluder or None):
        return False
    return True


# --------------------------------------------------------------------------- #
# the rows
# --------------------------------------------------------------------------- #
def review_rows(db, desktop: int, view: str) -> tuple[list[dict], list[dict]]:
    """``(to review, reviewed)``: the view's candidate rows, split by their stamp.

    Rows of a step the step table no longer asks to annotate (``ignore``) are
    left out, as the exports leave them out.
    """
    rows = db.vis_review_rows(desktop, view)
    if not rows:
        return [], []
    keep = set(annotatable_steps(db, desktop, view, {r["step"] for r in rows}))
    pending, done = [], []
    for row in rows:
        if row["step"] not in keep:
            continue
        covered = stamp_covers(row["flags"], row.get("visibility"), row.get("occluder"))
        (done if covered else pending).append(row)
    return pending, done


def progress(db, desktop: int, view: str) -> tuple[int, int]:
    """``(reviewed, total)`` of one view -- ``status``'s 「可见性审查 n/m」."""
    pending, done = review_rows(db, desktop, view)
    return len(done), len(pending) + len(done)


# --------------------------------------------------------------------------- #
# who hides it
# --------------------------------------------------------------------------- #
def _hits(part, rle: dict, box, transform: Similarity, hw: tuple[int, int]) -> bool:
    """Do ``part`` (in frame coordinates) and the visible ``rle`` share a pixel?"""
    window = _part_window(part, transform, hw)
    if _empty_window(window) or box is None or _intersect_window(window, box) is None:
        return False
    if transform.is_identity():
        return masks.rle_overlap(part.rle, rle) > 0     # run lengths: no decode
    mask = _part_mask(part, transform, hw, window)
    if mask is None:
        return False
    cropped, cbox = masks.decode_rle_shared(rle)
    inter = _intersect_window(window, cbox)
    if inter is None:
        return False
    mine = mask[inter[1] - window[1]:inter[3] - window[1],
                inter[0] - window[0]:inter[2] - window[0]]
    theirs = cropped[inter[1] - cbox[1]:inter[3] - cbox[1],
                     inter[0] - cbox[0]:inter[2] - cbox[0]]
    return bool((mine & theirs).any())


def occluders_on_frame(db, key: FrameKey, wanted: dict[str, str],
                       cache: Optional[InputCache] = None) -> dict[str, tuple[str, ...]]:
    """``part -> the parts (and @layers) that hide it`` on one confirmed frame.

    ``wanted`` is ``part -> its placement`` (the paint group). A part of the
    same group whose stored visible pixels lie inside this part's shape -- the
    keyframe in force here, through the frame's transform -- took those
    pixels from it; so did an occluder layer (``@hand``, ``@tool`` ...) that
    reaches into it. A part with no mask shape here names nobody.
    """
    if not wanted:
        return {}
    cache = cache if cache is not None else InputCache()
    rows = db.compiled(key)
    visible = {name: (str(row.get("placement") or ""), row["visible_rle"],
                      masks.rle_bbox(row["visible_rle"]))
               for name, row in rows.items() if row.get("visible_rle")}
    hw = next((tuple(int(v) for v in rle["size"]) for _p, rle, _b in visible.values()
               if rle.get("size")), None)
    out: dict[str, tuple[str, ...]] = {name: () for name in wanted}
    if hw is None:
        return out
    transform = db.transform(key)
    seg = pose_segment_of(db, key, cache)
    chains = _keyframes_of(db, key.desktop, key.view, cache)
    layers = [(layer.occluder_type, layer.rle, masks.rle_bbox(layer.rle))
              for layer in db.occluders(key) if layer.rle]
    for name, placement in wanted.items():
        chain = [kf for kf in chains.get(name, ())
                 if kf.pose_segment == seg and kf.placement == placement]
        kf = select_keyframe(chain, key.step)
        if kf is None or kf.geom_type != "mask":
            continue
        parts = [p for p in kf.parts if p.rle is not None and not _wrong_size(p.rle, hw)]
        found = [other for other, (where, rle, box) in sorted(visible.items())
                 if other != name and where == str(placement)
                 and any(_hits(p, rle, box, transform, hw) for p in parts)]
        found += [f"{LAYER_PREFIX}{kind}" for kind, rle, box in layers
                  if [int(v) for v in rle.get("size", ())] == list(hw)
                  and any(_hits(p, rle, box, transform, hw) for p in parts)]
        out[name] = tuple(found)
    return out


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """One queue entry: a part over consecutive frames that all say the same thing."""

    instance: str
    steps: list[int]
    visibility: str
    #: The label was set by hand (a frame override with a label).
    manual: bool
    #: The reason a hand label gives (``cable`` / ``viewpoint`` / ...), or ``None``.
    occluder: Optional[str]
    #: The parts (and ``@layers``) that hide a computed occlusion.
    occluded_by: tuple[str, ...] = ()
    placement: str = ""
    #: ``step -> input_hash`` of the rows, for whoever caches per row.
    hashes: dict = field(default_factory=dict, repr=False)

    def entry(self) -> dict:
        """The queue entry a panel lists (``step`` is the run's first frame)."""
        return {"step": self.steps[0], "steps": list(self.steps),
                "instance": self.instance, "visibility": self.visibility,
                "manual": self.manual, "occluder": self.occluder,
                "occluded_by": list(self.occluded_by), "placement": self.placement}


def _signature(row: dict, names: tuple[str, ...]) -> tuple:
    return (str(row.get("visibility") or ""), row.get("hand_visibility") is not None,
            row.get("occluder") or None, names)


def runs_of(rows: Iterable[dict], order: list[int],
            names: Optional[dict] = None) -> list[Run]:
    """Compress the rows to runs: one part, consecutive frames, one signature.

    ``order`` is the view's annotatable steps, ascending: two rows are
    consecutive when nothing of ``order`` lies between them -- a step with no
    image or an ``ignore`` step does not break a run, a frame that is not in the
    queue (not confirmed, or reviewed already) does. ``names`` is
    ``(step, part) -> occluders`` (missing: none). Parts in key order, each
    part's runs by step.
    """
    names = names or {}
    index = {int(step): i for i, step in enumerate(order)}
    by_part: dict[str, list[dict]] = {}
    for row in rows:
        by_part.setdefault(str(row["instance"]), []).append(row)
    out: list[Run] = []
    for part in sorted(by_part):
        current: Optional[Run] = None
        current_sig: Optional[tuple] = None
        last_at: Optional[int] = None
        for row in sorted(by_part[part], key=lambda r: int(r["step"])):
            step = int(row["step"])
            who = tuple(names.get((step, part), ()))
            sig = _signature(row, who)
            at = index.get(step)
            if (current is not None and at is not None and last_at is not None
                    and at == last_at + 1 and sig == current_sig):
                current.steps.append(step)
                current.hashes[step] = row.get("input_hash")
            else:
                current = Run(part, [step], sig[0], sig[1], sig[2], who,
                              str(row.get("placement") or ""),
                              {step: row.get("input_hash")})
                current_sig = sig
                out.append(current)
            last_at = at
    return out


def order_of(db, desktop: int, view: str) -> list[int]:
    """The view's annotatable steps, ascending: what "consecutive" is counted over."""
    return annotatable_steps(db, desktop, view, view_steps(db, desktop, view))
