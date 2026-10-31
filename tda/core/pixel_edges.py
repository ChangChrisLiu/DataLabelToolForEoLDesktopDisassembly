"""Freezing "who covers whom" into ``covered_by`` edges (task W3a, ``cli pixel-edges``).

A covering on a confirmed frame is *evidence* for a precedence edge, not the
edge (:mod:`tda.core.export.graph_covers`). Whether an upper part really holds
the lower one in place is a human call, and the user makes it **in Review, per
instance pair, with the picture in front of them**: the ``cover_ruling`` table
(task W3c) says ``blocks`` or ``not_blocks`` for ``(desktop, above, below)``.
This module turns those rulings into rows of the ``relation`` table:

* a pair ruled ``blocks`` that the confirmed frames show covering is written as
  ``covered_by(below <- above)``, ``necessity = required``, ``source =
  review:covers``, ``status = accepted``, the reason naming the ruling and the
  pixels, ``evidence_step`` = the first frame it covers on;
* a pair ruled ``blocks`` with **no** pixel overlap is listed ("ruled but no
  pixel evidence"), never written;
* a pair ruled ``not_blocks`` is never written and is listed under
  ``rejected_by_user``.

The class-pair whitelist (``configs/covers_whitelist.yaml``) is only a
suggestion layer: an **unruled** covering that matches one of its pairs and
passes its thresholds (``max_px >= max(min_px, min_frac * amodal(below))``) is
listed as a suggestion, and written -- with ``source = pixels:<version>`` -- only
when the caller explicitly allows unruled pairs.

Whatever the source, a pair is not written when an edge already says ``below``
waits for ``above`` (or a row of that triple exists in any status), when an
edge ties them the other way (a conflict: listed), when it would make the
recorded log break a constraint it kept before (listed -- those go back to the
user) or close a loop of actions. A whitelist pair is also refused when the
instance table already explains the covering structurally; the export-time
support table (``rule.supported_by``) is a stand-in for an empty ``mounted_on``
and does not count against an explicit word.

The rows are **imported** to :func:`tda.core.graph_derive.derive_edges`: kept
byte for byte by every later ``constraints`` run, never re-derived, so
``graph_version`` changes exactly once, when they appear. Everything here is
pure except :func:`write_edges` / :func:`undo_edges`, which the CLI calls
inside its guards.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

from tda.core.export.graph_covers import (
    BLOCKS,
    NOT_BLOCKS,
    SUPPORTED_BY_REASON,
    class_key,
    cover_rulings,
    scan_covers,
    structural_reasons,
)
from tda.core.graph import (
    Edge,
    constraint_edges,
    edges_from_db,
    edges_to_db,
    find_deadlocks,
    graph_version,
    validate_sequence,
)
from tda.core.graph_derive import settled_instances
from tda.core.graph_rules import ACCEPTED, active_edges
from tda.core.states import initial_overrides
from tda.core.taxonomy import Taxonomy

__all__ = [
    "ANNOTATOR", "COVERED_BY", "DEFAULT_WHITELIST", "OP_KIND", "OP_UNDO_KIND", "OP_VIEW",
    "OWNED_PREFIXES", "PIXELS_PREFIX", "REVIEW_SOURCE", "DesktopPlan", "Whitelist",
    "WhitelistError", "WhitelistPair", "edge_of", "evidence_from_db",
    "evidence_from_graph_dir", "evidence_json", "load_whitelist", "owned", "plan_desktop",
    "rulings_of", "split_suggestions", "undo_edges", "write_edges",
]

COVERED_BY = "covered_by"
#: ``relation.source`` of an edge the user ruled in Review.
REVIEW_SOURCE = "review:covers"
#: ``relation.source`` prefix of an edge the whitelist wrote (``pixels:W1``).
PIXELS_PREFIX = "pixels:"
#: Every source this command owns -- what ``--undo`` deletes.
OWNED_PREFIXES = (PIXELS_PREFIX, "review:")
OP_KIND = "pixel_edges"
OP_UNDO_KIND = "pixel_edges_undo"
#: ``op_log`` is scoped per ``(desktop, view)``; an edge belongs to no view.
OP_VIEW = "-"
ANNOTATOR = "cli:pixel-edges"
DEFAULT_WHITELIST = "configs/covers_whitelist.yaml"

#: ``below`` classes the loader refuses: a screw / connector / cable is decided
#: by its own rules (``fastened_by``, ``connected_to``) or is pixel noise.
REFUSED_BELOW = ("screw", "connector")
#: ``above`` refinements the loader refuses: small cards ride on the board.
REFUSED_ABOVE = ("expansion_card", "expansion_card.wlan", "expansion_card.other")


# --------------------------------------------------------------------------- #
# the whitelist
# --------------------------------------------------------------------------- #
class WhitelistError(ValueError):
    """The whitelist file is unreadable or names a pair the loader refuses."""


@dataclass(frozen=True)
class WhitelistPair:
    """One class pair: ``above`` / ``below`` are ``class`` or ``class.kind``."""

    above: str
    below: str
    min_px: int
    min_frac: float
    note: str = ""
    index: int = 0

    def matches(self, above_class: str, below_class: str) -> bool:
        return _spec_matches(self.above, above_class) and _spec_matches(self.below, below_class)

    def need(self, amodal: Optional[int]) -> int:
        """The pixels a covering must reach; ``min_px`` alone when the amodal area is unknown."""
        if not amodal:
            return int(self.min_px)
        # rounded first: 0.03 * 100000 is 3000.0000000000005 in floating point
        return max(int(self.min_px), math.ceil(round(self.min_frac * amodal, 6)))

    def label(self) -> str:
        return f"{self.above} -> {self.below}"


@dataclass(frozen=True)
class Whitelist:
    version: str
    path: str
    min_px: int
    min_frac: float
    pairs: tuple[WhitelistPair, ...]

    @property
    def source(self) -> str:
        return f"{PIXELS_PREFIX}{self.version}"

    def match(self, above_class: str, below_class: str) -> Optional[WhitelistPair]:
        return next((p for p in self.pairs if p.matches(above_class, below_class)), None)

    def as_json(self) -> dict:
        return {"version": self.version, "path": self.path,
                "defaults": {"min_px": self.min_px, "min_frac": self.min_frac},
                "pairs": [{"above": p.above, "below": p.below, "min_px": p.min_px,
                           "min_frac": p.min_frac, "note": p.note} for p in self.pairs]}


def _spec_matches(spec: str, key: str) -> bool:
    """``optical_drive`` matches ``optical_drive`` and ``optical_drive.<kind>``; a refined spec only itself."""
    if "." in spec:
        return spec == key
    return key == spec or key.split(".", 1)[0] == spec


def _check_spec(spec: str, tax: Taxonomy, side: str) -> Optional[str]:
    cls, _, kind = spec.partition(".")
    if cls not in tax.classes:
        return f"{side} {spec!r}: no such class"
    if kind:
        attrs = tax.classes[cls].get("attrs") or {}
        # the refinement class_key uses: the kind, or a screw's role
        known = [*(attrs.get("kind") or []), *(attrs.get("role") or [])]
        if kind not in known:
            return f"{side} {spec!r}: {cls} has no kind {kind!r}"
    return None


def _refusal(above: str, below: str) -> Optional[str]:
    below_cls = below.split(".", 1)[0]
    if above.split(".", 1)[0] == "cover" and below_cls == "motherboard":
        return "cover -> motherboard is decided elsewhere (the shroud question)"
    if above in REFUSED_ABOVE:
        return "a wlan / other expansion card rides on the board (name expansion_card.gpu)"
    if below_cls in REFUSED_BELOW or "cable" in below_cls:
        return "a screw / connector / cable below is decided by its own rules"
    return None


def load_whitelist(path: str | Path, tax: Taxonomy) -> Whitelist:
    """Read and check ``covers_whitelist.yaml``; :class:`WhitelistError` lists every problem."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WhitelistError(f"cannot read {path}: {' '.join(str(exc).split())}") from None
    if not isinstance(raw, dict):
        raise WhitelistError(f"{path}: not a mapping")
    version = str(raw.get("version") or "").strip()
    if not version or not version.startswith("W"):
        raise WhitelistError(f"{path}: `version` must be W<n> (e.g. W1), not {version!r}")
    defaults = raw.get("defaults") or {}
    min_px = int(defaults.get("min_px", 1000))
    min_frac = float(defaults.get("min_frac", 0.03))
    problems: list[str] = []
    pairs: list[WhitelistPair] = []
    for i, entry in enumerate(raw.get("pairs") or [], 1):
        if not isinstance(entry, dict) or not entry.get("above") or not entry.get("below"):
            problems.append(f"pair {i}: needs `above` and `below`")
            continue
        above, below = str(entry["above"]).strip(), str(entry["below"]).strip()
        bad = (_refusal(above, below) or _check_spec(above, tax, "above")
               or _check_spec(below, tax, "below"))
        if bad:
            problems.append(f"pair {i} ({above} -> {below}): {bad}")
            continue
        try:
            pair = WhitelistPair(above, below, int(entry.get("min_px", min_px)),
                                 float(entry.get("min_frac", min_frac)),
                                 str(entry.get("note") or ""), i)
        except (TypeError, ValueError) as exc:
            problems.append(f"pair {i} ({above} -> {below}): {exc}")
            continue
        if pair.min_px < 0 or not 0 <= pair.min_frac <= 1:
            problems.append(f"pair {i} ({above} -> {below}): thresholds out of range")
            continue
        pairs.append(pair)
    if problems:
        raise WhitelistError(f"{path}: refused -- " + "; ".join(problems))
    return Whitelist(version, str(path), min_px, min_frac, tuple(pairs))


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
def evidence_from_db(db, tax: Taxonomy, desktop: int, view: str) -> tuple[list[dict], int]:
    """``(covers, confirmed frames)`` of one desktop, measured as ``export-graph`` does."""
    from tda.core.export.coco import load_ctx

    ctx = load_ctx(db, tax, desktop, view)
    edges = active_edges(constraint_edges(edges_from_db(db, desktop)))
    scan = scan_covers(db, ctx, view, edges, {}, rulings={})
    return scan.covers, scan.confirmed_frames


def evidence_from_graph_dir(directory: str | Path, desktop: int) -> Optional[tuple[list[dict], int]]:
    """The same from an ``export-graph`` output's ``D<nn>.json``; ``None`` when it has none.

    A file written before task W3a has no ``below_amodal_px``: the pairs are
    then judged by ``min_px`` alone and flagged.
    """
    path = Path(directory) / f"D{int(desktop):02d}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("covers") or [], int(doc["meta"]["counts"].get("confirmed_frames") or 0)


# --------------------------------------------------------------------------- #
# judging one desktop
# --------------------------------------------------------------------------- #
@dataclass
class DesktopPlan:
    """What ``pixel-edges`` would write for one desktop, and everything it would not."""

    desktop: int
    confirmed_frames: int = 0
    review: list[dict] = field(default_factory=list)       # ruled blocks -> written
    whitelist: list[dict] = field(default_factory=list)    # unruled, written with --allow-unruled
    suggested: list[dict] = field(default_factory=list)    # unruled, whitelist only, not written
    rejected_by_user: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    rulings: int = 0

    def to_write(self, allow_unruled: bool) -> list[dict]:
        return self.review + (self.whitelist if allow_unruled else [])


def _frac(px: int, amodal: Optional[int]) -> Optional[float]:
    return round(px / amodal, 4) if amodal else None


def _pixels_text(item: dict, view: str) -> str:
    frac = item["frac"]
    share = f"{frac * 100:.1f}% of its amodal area" if frac is not None \
        else "its amodal area unknown"
    return (f"{item['above']} covers {item['below']} on {item['n_frames']} confirmed "
            f"{view} frames (max {item['px']} px, {share})")


def _item(desktop: int, cover: dict, instances: dict) -> dict:
    px = int(cover["max_px"])
    amodal = cover.get("below_amodal_px")
    return {
        "desktop": int(desktop), "above": cover["above"], "below": cover["below"],
        "above_class": cover.get("above_class") or class_key(instances.get(cover["above"])),
        "below_class": cover.get("below_class") or class_key(instances.get(cover["below"])),
        "px": px, "amodal_px": amodal, "frac": _frac(px, amodal),
        "amodal_unavailable": not amodal,
        "frames": list(cover.get("frames") or []), "n_frames": int(cover.get("n_frames") or 0),
        "first": cover.get("first"),
    }


def plan_desktop(desktop: int, covers: list[dict], confirmed_frames: int,
                 instances: dict, stored: list[Edge], actions: list, events: Iterable,
                 tax: Taxonomy, rulings: dict, whitelist: Optional[Whitelist],
                 view: str = "scan") -> DesktopPlan:
    """Judge every covering of one desktop against its rulings and the whitelist. Pure."""
    plan = DesktopPlan(desktop=int(desktop), confirmed_frames=confirmed_frames,
                       rulings=len(rulings))
    settled = settled_instances(instances)
    active = active_edges(constraint_edges(stored))
    rows = {(e.type, e.target, e.blocker): e for e in stored}
    listed = {(c["above"], c["below"]) for c in covers}
    seen: set[tuple[str, str]] = set()

    def skip(item: dict, why: str, **more) -> None:
        plan.skipped.append({**item, "why": why, **more})

    for cover in sorted(covers, key=lambda c: (c["above"], c["below"])):
        above, below = cover["above"], cover["below"]
        seen.add((above, below))
        ruling = rulings.get((above, below))
        item = _item(desktop, cover, instances)
        if ruling and ruling.get("ruling") == NOT_BLOCKS:
            plan.rejected_by_user.append({**item, "ruling": ruling})
            continue
        ruled = bool(ruling and ruling.get("ruling") == BLOCKS)
        pair = None if ruled or whitelist is None else whitelist.match(
            item["above_class"], item["below_class"])
        if not ruled and pair is None:
            continue
        item["ruling"] = ruling if ruled else None
        item["whitelist_pair"] = None if pair is None else pair.label()
        same = [e.label() for e in active if e.target == below and e.blocker == above]
        other = [e.label() for e in active if e.target == above and e.blocker == below]
        row = rows.get((COVERED_BY, below, above))
        if same or row is not None:
            skip(item, "already_an_edge", edges=same or [row.label()],
                 row_source=None if row is None else row.source,
                 row_status=None if row is None else row.status)
            continue
        if other:
            skip(item, "reverse_edge", edges=other)
            continue
        structure = [r for r in structural_reasons(settled, tax, above, below, listed)
                     if r != SUPPORTED_BY_REASON]
        item["structural"] = structure
        if not ruled:
            if structure:
                skip(item, "structural")
                continue
            need = pair.need(item["amodal_px"])
            item["need_px"] = need
            if item["px"] < need:
                skip(item, "below_threshold")
                continue
        if ruled:
            item["source"] = REVIEW_SOURCE
            item["reason"] = (f"{_pixels_text(item, view)}; ruled blocks in Review"
                              f" by {ruling.get('by') or '?'}"
                              f" ({str(ruling.get('ts') or '')[:10] or '?'},"
                              f" {ruling.get('view') or '?'} step {ruling.get('step')})")
            plan.review.append(item)
        else:
            item["source"] = whitelist.source
            item["reason"] = (f"{_pixels_text(item, view)}; whitelist {whitelist.version} "
                              f"pair {pair.label()}")
            plan.whitelist.append(item)

    for (above, below), ruling in sorted(rulings.items()):
        if (above, below) in seen:
            continue
        item = {"desktop": int(desktop), "above": above, "below": below,
                "above_class": class_key(instances.get(above)),
                "below_class": class_key(instances.get(below)), "ruling": ruling}
        if ruling.get("ruling") == NOT_BLOCKS:
            plan.rejected_by_user.append({**item, "px": None})
        else:
            skip(item, "ruled_no_pixel_evidence")

    _check_against_log(plan, settled, active, actions, events, tax)
    return plan


def edge_of(item: dict) -> Edge:
    return Edge(type=COVERED_BY, target=item["below"], blocker=item["above"],
                necessity="required", mode=None, reason=item["reason"],
                source=item["source"], evidence_step=item.get("first"), status=ACCEPTED)


def _check_against_log(plan: DesktopPlan, settled: dict, active: list[Edge], actions: list,
                       events: Iterable, tax: Taxonomy) -> None:
    """Move every edge the recorded log breaks, or that closes a loop, to ``skipped``.

    Judged one edge at a time against the stored graph, for every list (the
    suggestions too, so a reader sees which whitelist pairs the log refutes).
    """
    initial = initial_overrides(e for e in events if not e.auto)
    before = set(validate_sequence(settled, active, actions, tax, initial=initial))
    loops_before = bool(find_deadlocks(active, settled, tax, initial=initial))
    for name in ("review", "whitelist", "suggested"):
        kept = []
        for item in getattr(plan, name):
            edge = edge_of(item)
            lines = [line for line in validate_sequence(settled, [*active, edge], actions, tax,
                                                        initial=initial)
                     if line not in before]
            if lines:
                plan.skipped.append({**item, "why": "would_violate", "lines": lines})
                continue
            if not loops_before and find_deadlocks([*active, edge], settled, tax,
                                                   initial=initial):
                plan.skipped.append({**item, "why": "deadlock"})
                continue
            kept.append(item)
        setattr(plan, name, kept)


def split_suggestions(plan: DesktopPlan, allow_unruled: bool) -> None:
    """Without ``--allow-unruled`` the whitelist's pairs are suggestions, never rows."""
    if not allow_unruled:
        plan.suggested.extend(plan.whitelist)
        plan.whitelist = []


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_edges(db, desktop: int, items: list[dict]) -> dict:
    """Write the rows, stamp ``graph_version``, log one op. **Inside a transaction.**"""
    from tda import pipeline as P

    edges = [edge_of(item) for item in items]
    edges_to_db(db, desktop, edges)
    version = graph_version(db, desktop)
    P.merge_desktop_meta(db, desktop, {"graph_version": version})
    rows = [_edge_json(e) for e in edges]
    db.log_op(desktop, OP_VIEW, OP_KIND, {"rows": rows, "graph_version": version},
              {"delete": [[e.type, e.target, e.blocker] for e in edges]}, ANNOTATOR)
    return {"written": len(edges), "graph_version": version}


def owned(edge: Edge) -> bool:
    return str(edge.source or "").startswith(OWNED_PREFIXES)


def undo_edges(db, desktop: int) -> dict:
    """Delete every ``pixels:*`` / ``review:*`` row of one desktop, with an op. **Inside a transaction.**"""
    from tda import pipeline as P

    gone = [e for e in edges_from_db(db, desktop) if owned(e)]
    for edge in gone:
        db.delete_relation(desktop, edge.type, edge.target, edge.blocker)
    version = graph_version(db, desktop)
    if gone:
        P.merge_desktop_meta(db, desktop, {"graph_version": version})
        db.log_op(desktop, OP_VIEW, OP_UNDO_KIND,
                  {"deleted": [_edge_json(e) for e in gone], "graph_version": version},
                  {"restore": [_edge_json(e) for e in gone]}, ANNOTATOR)
    return {"deleted": len(gone), "graph_version": version}


def _edge_json(edge: Edge) -> dict:
    return {"type": edge.type, "target": edge.target, "blocker": edge.blocker,
            "necessity": edge.necessity, "mode": edge.mode, "reason": edge.reason,
            "source": edge.source, "evidence_step": edge.evidence_step,
            "status": edge.status}


# --------------------------------------------------------------------------- #
# the evidence file
# --------------------------------------------------------------------------- #
def evidence_json(plans: list[DesktopPlan], whitelist: Optional[Whitelist],
                  allow_unruled: bool, evidence_source: str, view: str,
                  rulings_table: bool, applied: bool) -> dict:
    """The evidence file that ships with the dataset: every edge, and every pair that is not one."""
    plans = sorted(plans, key=lambda p: p.desktop)

    def every(name: str) -> list[dict]:
        return [item for p in plans for item in getattr(p, name)]

    skipped = every("skipped")
    per_pair: dict[str, dict] = {}
    for name in ("review", "whitelist", "suggested"):
        for item in every(name):
            key = item.get("whitelist_pair") or "(ruled in Review)"
            per_pair.setdefault(key, {"review": 0, "whitelist": 0, "suggested": 0,
                                      "skipped": {}})[name] += 1
    for item in skipped:
        key = item.get("whitelist_pair") or ("(ruled in Review)" if item.get("ruling")
                                             else None)
        if key is None:
            continue
        slot = per_pair.setdefault(key, {"review": 0, "whitelist": 0, "suggested": 0,
                                         "skipped": {}})
        slot["skipped"][item["why"]] = slot["skipped"].get(item["why"], 0) + 1
    return {
        "generated_by": "python -m tda.cli pixel-edges",
        "evidence_source": evidence_source,
        "view": view,
        "applied": bool(applied),
        "rulings_table": bool(rulings_table),
        "allow_unruled": bool(allow_unruled),
        "whitelist": None if whitelist is None else whitelist.as_json(),
        "thresholds": None if whitelist is None else {
            "rule": "max_px >= max(min_px, min_frac * amodal_area(below) on the frame of "
                    "max_px); min_px alone when the amodal area is unknown",
            "min_px": whitelist.min_px, "min_frac": whitelist.min_frac},
        "edges": every("review"),
        "whitelist_edges": every("whitelist"),
        "suggested": every("suggested"),
        "rejected_by_user": every("rejected_by_user"),
        "skipped": skipped,
        "desktops": [{"desktop": p.desktop, "confirmed_frames": p.confirmed_frames,
                      "rulings": p.rulings, "edges": len(p.review),
                      "whitelist_edges": len(p.whitelist), "suggested": len(p.suggested),
                      "rejected_by_user": len(p.rejected_by_user),
                      "skipped": len(p.skipped)} for p in plans],
        "by_pair": dict(sorted(per_pair.items())),
        "counts": {"edges": len(every("review")), "whitelist_edges": len(every("whitelist")),
                   "suggested": len(every("suggested")),
                   "rejected_by_user": len(every("rejected_by_user")),
                   "skipped": len(skipped)},
    }


def rulings_of(db, desktop: int) -> dict:
    """:func:`tda.core.export.graph_covers.cover_rulings`, re-exported for the CLI."""
    return cover_rulings(db, desktop)
