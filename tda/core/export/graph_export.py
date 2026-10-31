"""One precedence-graph JSON per desktop -- ``python -m tda.cli export-graph`` (task U46).

Spec 8.1 promises 「每机约束图 JSON、步骤/动作表」 and nothing wrote it: the
``constraints`` command stores the edges and a Markdown report, the VLM export
consumes the graph internally. This module writes, per desktop, **one file a
reader can plan from**::

    D<nn>.json = {meta, instances, steps, edges, covers, legal_by_step, plans,
                  violations, deadlocks, dead_ends}

and beside them ``summary.json`` and ``README.md`` (:mod:`.graph_readme`).

* ``instances`` -- the machine's parts (no Label Studio draft, no 框外零件, no
  part the machine does not have, no gallery-only part) with their relational
  fields and their state timeline, initial state first;
* ``steps`` -- the step table and its actions, failed attempts included;
* ``edges`` -- every **active** hard-constraint edge (spec 7.2) as stored, and
  ``demonstrated_at``: the first step after which the log has the blocker in a
  state the edge accepts (``None`` when it already was at the start, or never);
* ``covers`` -- the pixel evidence of who covers whom on the view's confirmed
  frames (:mod:`.graph_covers`), each pair with the edges it agrees with; a pair
  nothing accounts for -- no edge either way, no relation in the instance
  table -- is a *candidate*;
* ``legal_by_step`` -- after every step (0 = the initial state) the strict and
  relaxed legal action sets (:func:`tda.core.graph.legal_actions`), the next
  demonstrated actions and whether they were legal;
* ``plans`` -- :func:`tda.core.graph_plan.plan_removal` from the initial state
  for the motherboard, the CPU and the PSU where the desktop has them, and for
  ``all`` (every part the log removes); plus the length of the ``all`` plan
  after every step, which is V10's progress label;
* ``violations`` -- :func:`tda.core.graph.validate_sequence`'s lines, the same
  replay ``constraints --validate`` prints, each split into its fields, and
  ``timeline_agrees``: whether the state timeline (hand-written events
  included, which the replay ignores) finds the same at that step;
* ``deadlocks`` / ``dead_ends`` -- :func:`~tda.core.graph.find_deadlocks` and
  :func:`~tda.core.graph.find_dead_ends`, as ``constraints`` reports them.

**Read-only.** The database is opened through a ``file:...?mode=ro`` URI
(:class:`ReadOnlyDb`) and never through :class:`~tda.core.db.Db`'s own
constructor, whose schema bootstrap writes; any statement that would write
fails. Nothing is compiled: the visible masks are the stored rows of confirmed
frames, the whole shapes the keyframes the compiler would select.

**Deterministic.** Everything is sorted, JSON is written with sorted keys and
``\\n`` line ends, and nothing depends on the wall clock, the machine or
``PYTHONHASHSEED``: two runs on one database write the same bytes.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import quote

from tda.core.db import SCHEMA_VERSION, Db
from tda.core.export.coco import DesktopCtx, load_ctx
from tda.core.export.graph_covers import CoverScan, class_key, scan_covers
from tda.core.export.release import model_names, public_attrs
from tda.core.graph import (
    GATED_VERBS,
    HARD_TYPES,
    REQUIRED_STATES,
    Edge,
    applicable_preconditions,
    constraint_edges,
    edges_from_db,
    find_dead_ends,
    find_deadlocks,
    graph_version,
    legal_actions,
    unmet,
    validate_sequence,
)
from tda.core.graph_plan import plan_removal
from tda.core.graph_rules import active_edges, blocker_state, verb_applies, verb_effect
from tda.core.implied import is_implied
from tda.core.model import (
    VIEWS,
    InstanceRec,
    is_extra,
    is_provisional,
    teardown_only,
)
from tda.core.states import (
    CABLE_CLASS,
    CABLE_PREFIX,
    IN_CHASSIS,
    INITIAL_STEP,
    REMOVED,
    FrameState,
    InstState,
    initial_overrides,
)
from tda.core.taxonomy import Taxonomy

__all__ = [
    "ALL_GOAL",
    "DEFAULT_VIEW",
    "EXPORT_GRAPH_VERSION",
    "FAILED_WITHOUT_UNMET",
    "PLAN_GOALS",
    "README_NAME",
    "ReadOnlyDb",
    "SUCCESS_VIOLATES",
    "SUMMARY_NAME",
    "build_desktop",
    "desktop_file_name",
    "dumps",
    "export_graph",
    "parse_violation",
    "summarize",
]

#: Bumped when a field of the per-desktop file changes meaning or goes away.
EXPORT_GRAPH_VERSION = 1
#: The view whose confirmed frames the pixel evidence is read off by default.
DEFAULT_VIEW = "scan"
#: The single-part goals planned from the initial state, where the desktop has them.
PLAN_GOALS = ("motherboard.01", "cpu.01", "psu.01")
#: The goal "every part the log removes".
ALL_GOAL = "all"
SUMMARY_NAME = "summary.json"
README_NAME = "README.md"
#: ``violations[].kind``: a successful action with an unmet required edge.
SUCCESS_VIOLATES = "success_violates"
#: ``violations[].kind``: a failed attempt nothing in the graph explains.
FAILED_WITHOUT_UNMET = "failed_without_unmet"
SUCCESS = "success"
#: ``legal_by_step[].next_illegal[].reason``: a precondition of spec 7.2 is unmet.
UNMET_EDGE = "unmet_edge"
#: ... the verb does not apply to the target in its state (a cover opened twice).
VERB_NOT_APPLICABLE = "verb_not_applicable"

_BREACH = re.compile(r"^step (-?\d+): (\S+) (\S+) violates ((\w+)\((\S+), (\S+)\)) "
                     r"\((\S+) is (.+)\)$")
_HINT = re.compile(r"^step (-?\d+): failed (\S+) (\S+) has no unmet constraint "
                   r"- missing edge\?$")


# --------------------------------------------------------------------------- #
# the database, read-only
# --------------------------------------------------------------------------- #
def read_only_uri(path: str) -> str:
    """``file:`` URI of ``path`` that sqlite may only read (as ``cli notes`` opens it)."""
    absolute = os.path.abspath(str(path)).replace("\\", "/")
    return f"file:{quote(absolute, safe='/:')}?mode=ro"


class ReadOnlyDb(Db):
    """:class:`~tda.core.db.Db`'s readers over a ``mode=ro`` connection (task U46).

    The constructor of :class:`~tda.core.db.Db` replays the DDL and stamps the
    schema version -- a write, on the live file. This one opens the file
    read-only and only *checks* the version: an older file lacks columns the
    readers select, a newer one has columns this build does not understand,
    and both are refused with one line. Every reader of ``Db`` works; every
    writer raises ``sqlite3.OperationalError`` (attempt to write a readonly
    database), which is the guarantee. Opening a WAL database this way may
    leave empty ``-wal`` / ``-shm`` files next to it; the database file itself
    is not changed.

    **One snapshot** (review round 1, N6): the connection opens a read
    transaction before its first ``SELECT`` and keeps it until
    :meth:`close`, so a whole export -- every table of every desktop -- reads
    the database as it was at that moment, even when the annotator's app is
    writing to the live file meanwhile (WAL readers see the snapshot they
    started with). A backup copy (``--db``) is still the recommended input.
    """

    def __init__(self, path: str):  # noqa: D107 - see the class docstring
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"{self.path} does not exist")
        self.conn = sqlite3.connect(read_only_uri(self.path), uri=True)
        self.conn.row_factory = sqlite3.Row
        self._lock_path = Path(self.path + ".lock")
        self._lock_annotator = None
        self._tx_depth = 0
        try:
            self.conn.execute("BEGIN")           # the snapshot starts at the next read
            found = self._stored_version()
        except sqlite3.Error as exc:
            self.conn.close()
            raise RuntimeError(f"{self.path} is not a TDA database: {exc}") from None
        if found != SCHEMA_VERSION:
            self.conn.close()
            raise RuntimeError(f"{self.path} has schema version {found}; this build reads "
                               f"version {SCHEMA_VERSION} (open it once with the app)")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def dumps(obj) -> str:
    """The one JSON spelling of every file: sorted keys, one-space indent, UTF-8 text."""
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def desktop_file_name(desktop: int) -> str:
    return f"D{int(desktop):02d}.json"


def _pair(verb: str, target: str) -> str:
    return f"{verb} {target}"


def _edge_order(edge: Edge) -> tuple:
    rank = HARD_TYPES.index(edge.type) if edge.type in HARD_TYPES else len(HARD_TYPES)
    return (rank, edge.target, edge.blocker)


def _copy(state: FrameState) -> FrameState:
    return {k: InstState(state=v.state, placement=v.placement) for k, v in state.items()}


def _apply(tax: Taxonomy, instances: dict[str, InstanceRec], sim: FrameState,
           verb: str, target: str) -> None:
    """One verb on a simulated state, with the spec-3.3 attached cascade (the planner's)."""
    rec = instances.get(target)
    if rec is not None:
        cls, attrs = rec.cls, rec.attrs
    elif target.startswith(CABLE_PREFIX):
        cls, attrs = CABLE_CLASS, {}
    else:
        return
    new = verb_effect(tax, cls, attrs, verb)
    if new is None:
        return
    inst = sim.get(target)
    if inst is None:
        sim[target] = InstState(state=new, placement=IN_CHASSIS)
    else:
        inst.state = new
    if new != REMOVED or cls == "connector":
        return
    for key, child in instances.items():
        if child.attached and child.parent == target and key in sim:
            sim[key].state = REMOVED


def _settled(db: Db, desktop: int) -> dict[str, InstanceRec]:
    """The instance table ``constraints --validate`` replays against.

    :func:`tda.core.model.teardown_only` (no context object, no gallery-only
    part, U45) without the Label Studio drafts -- the same call
    ``cli_graph._settled`` makes.
    """
    return {k: rec for k, rec in teardown_only(db.instances(desktop)).items()
            if not is_provisional(k)}


# --------------------------------------------------------------------------- #
# the parts of one desktop's file
# --------------------------------------------------------------------------- #
def _instances(ctx: DesktopCtx, timeline: list[int]) -> tuple[list[dict], dict]:
    """``instances[]`` and ``key -> first step it reads removed`` (``None``: never)."""
    out: list[dict] = []
    removed_at: dict[str, Optional[int]] = {}
    for key in sorted(ctx.instances):
        rec = ctx.instances[key]
        seen: list[dict] = []
        gone: Optional[int] = None
        for step in timeline:
            inst = ctx.state_at(step).get(key)
            if inst is None:
                continue
            point = {"step": step, "state": inst.state, "placement": inst.placement}
            if not seen or (seen[-1]["state"], seen[-1]["placement"]) != (
                    inst.state, inst.placement):
                seen.append(point)
            if gone is None and inst.state == REMOVED:
                gone = step
        removed_at[key] = gone
        out.append({
            "key": key, "class": rec.cls, "attrs": public_attrs(rec.attrs),
            "parent": rec.parent, "attached": bool(rec.attached),
            "mounted_on": rec.mounted_on, "fastens": rec.fastens,
            "socket_host": rec.socket_host, "cable": rec.cable,
            "implied": is_implied(rec), "extra": is_extra(rec),
            "timeline": seen,
        })
    return out, removed_at


def _steps(ctx: DesktopCtx, timeline: list[int]) -> list[dict]:
    out = []
    for step in timeline:
        if step == INITIAL_STEP and step not in ctx.steps:
            continue
        rec = ctx.steps.get(step)
        out.append({
            "step": step,
            "type": ctx.step_type(step),
            "raw_name": rec.raw_name if rec is not None else None,
            "actions": [{"idx": a.idx, "verb": a.verb, "target": a.target, "tool": a.tool,
                         "direction": a.direction, "result": a.result,
                         "failure_reason": a.failure_reason}
                        for a in ctx.actions_at(step, successful_only=False)],
        })
    return out


def _satisfied(edge: Edge, state: FrameState) -> bool:
    current = blocker_state(state, edge.blocker)
    return current is None or current == REMOVED or current in REQUIRED_STATES.get(
        edge.type, frozenset())


def _edges(ctx: DesktopCtx, edges: list[Edge], timeline: list[int]) -> list[dict]:
    out = []
    for edge in edges:
        shown: Optional[int] = None
        if not _satisfied(edge, ctx.state_at(timeline[0])):
            shown = next((s for s in timeline[1:] if _satisfied(edge, ctx.state_at(s))), None)
        out.append({
            "label": edge.label(), "type": edge.type, "target": edge.target,
            "blocker": edge.blocker, "necessity": edge.necessity, "mode": edge.mode,
            "reason": edge.reason or "", "source": edge.source, "status": edge.status,
            "evidence_step": edge.evidence_step, "demonstrated_at": shown,
        })
    return out


def _legal_sets(ctx: DesktopCtx, edges: list[Edge], state: FrameState
                ) -> tuple[list[str], list[str]]:
    strict = [_pair(v, t) for v, t in legal_actions(ctx.instances, edges, state, ctx.tax,
                                                    strict=True)]
    relaxed = [_pair(v, t) for v, t in legal_actions(ctx.instances, edges, state, ctx.tax,
                                                     strict=False)]
    return strict, relaxed


def _known_target(ctx: DesktopCtx, target: str) -> bool:
    """Is ``target`` something the graph can judge: an instance, or a ``cable:*`` node?"""
    return target in ctx.instances or target.startswith(CABLE_PREFIX)


def _why_illegal(ctx: DesktopCtx, sim: FrameState, verb: str, target: str) -> str:
    """:data:`VERB_NOT_APPLICABLE` when the verb cannot start from the target's state, else
    :data:`UNMET_EDGE` -- a preference (strict) or a necessity of spec 7.2 is in the way."""
    rec = ctx.instances.get(target)
    cls, attrs = (rec.cls, rec.attrs) if rec is not None else (CABLE_CLASS, {})
    current = blocker_state(sim, target) or ""
    return UNMET_EDGE if verb_applies(ctx.tax, cls, attrs, verb, current) \
        else VERB_NOT_APPLICABLE


def _judge(verdicts: list[bool], unknown: bool) -> Optional[bool]:
    """False if any judged action broke the set; ``None`` if nothing (or not all) could be judged."""
    if any(v is False for v in verdicts):
        return False
    if unknown or not verdicts:
        return None
    return True


def _legal_by_step(ctx: DesktopCtx, edges: list[Edge], timeline: list[int]) -> list[dict]:
    """After every step: what is legal, what was done next, and whether it was legal.

    A compound step's second action is judged in the state its first one left
    (the same replay :func:`~tda.core.graph.validate_sequence` does), so two
    actions of one step can depend on each other.

    Two kinds of next action are **not judged** (review round 1, S1), because
    the legal set cannot contain them whatever the graph says:

    * a verb no hard constraint gates (``reorient``, :data:`GATED_VERBS`):
      it has no state effect, so it is never in ``legal_actions``, and
      :func:`~tda.core.graph.validate_sequence` never finds it in breach -- it
      is skipped, and a step of nothing else is ``null``;
    * a target the graph does not know (``?``, ``connector.?``: a sheet that
      did not say which part): listed in ``next_unjudged`` and the step's
      verdict is ``null`` unless another of its actions is illegal.

    ``next_illegal`` says why each judged action is not in the strict set:
    :data:`UNMET_EDGE` or :data:`VERB_NOT_APPLICABLE` (a cover opened twice --
    a redundant log line, not a constraint broken).
    """
    out = []
    for i, step in enumerate(timeline):
        state = ctx.state_at(step)
        strict, relaxed = _legal_sets(ctx, edges, state)
        nxt = timeline[i + 1] if i + 1 < len(timeline) else None
        actions = ctx.actions_at(nxt, successful_only=False) if nxt is not None else []
        done = [a for a in actions if a.result == SUCCESS]
        verdict_strict: list[bool] = []
        verdict_relaxed: list[bool] = []
        unjudged: list[str] = []
        illegal: list[dict] = []
        sim = _copy(state)
        here_strict, here_relaxed = set(strict), set(relaxed)
        for j, action in enumerate(done):
            pair = _pair(action.verb, action.target)
            if action.verb not in GATED_VERBS:
                pass                                   # reorient: nothing gates it
            elif not _known_target(ctx, action.target):
                unjudged.append(pair)
            else:
                if j:
                    s, r = _legal_sets(ctx, edges, sim)
                    here_strict, here_relaxed = set(s), set(r)
                verdict_strict.append(pair in here_strict)
                verdict_relaxed.append(pair in here_relaxed)
                if pair not in here_strict:
                    illegal.append({"action": pair, "reason": _why_illegal(
                        ctx, sim, action.verb, action.target)})
            _apply(ctx.tax, ctx.instances, sim, action.verb, action.target)
        out.append({
            "step": step,
            "legal": strict, "n_legal": len(strict),
            "legal_relaxed": relaxed, "n_legal_relaxed": len(relaxed),
            "next_step": nxt,
            "next_demonstrated": [_pair(a.verb, a.target) for a in done],
            "next_failed": [_pair(a.verb, a.target) for a in actions
                            if a.result != SUCCESS],
            "next_unjudged": unjudged,
            "next_illegal": illegal,
            "next_is_legal": _judge(verdict_strict, bool(unjudged)),
            "next_is_legal_relaxed": _judge(verdict_relaxed, bool(unjudged)),
        })
    return out


def _log_goals(ctx: DesktopCtx) -> list[str]:
    """Every part the log removes, in the order it does: successful actions whose verb removes."""
    goals: list[str] = []
    for action in sorted(ctx.actions, key=lambda a: (a.step, a.idx)):
        rec = ctx.instances.get(action.target)
        if action.result != SUCCESS or rec is None or action.target in goals:
            continue
        if verb_effect(ctx.tax, rec.cls, rec.attrs, action.verb) == REMOVED:
            goals.append(action.target)
    return goals


def _plan_all(ctx: DesktopCtx, edges: list[Edge], state: FrameState,
              goals: list[str]) -> tuple[list[str], list[str], bool, list[str]]:
    """``(plan, unplanned goals, relaxed, dropped edge labels)`` for every goal still there.

    One goal after the other, in the log's own removal order, each planned
    from the state the previous plans left: a legal sequence that ends with
    every goal removed (a shortest one per goal, not proven shortest overall).
    """
    sim = _copy(state)
    plan: list[str] = []
    unplanned: list[str] = []
    relaxed = False
    dropped: set[str] = set()
    for goal in goals:
        inst = sim.get(goal)
        if inst is not None and inst.state == REMOVED:
            continue
        found = plan_removal(ctx.instances, edges, sim, goal, ctx.tax)
        if found is None:
            unplanned.append(goal)
            continue
        relaxed = relaxed or found.relaxed
        dropped.update(e.label() for e in found.dropped)
        for verb, target in found.actions:
            plan.append(_pair(verb, target))
            _apply(ctx.tax, ctx.instances, sim, verb, target)
    return plan, unplanned, relaxed, sorted(dropped)


def _demonstrated(ctx: DesktopCtx, until: Optional[int]) -> Optional[int]:
    """Successful actions the log took up to and including step ``until``."""
    if until is None:
        return None
    return sum(1 for a in ctx.actions if a.result == SUCCESS and a.step <= until)


def _plans(ctx: DesktopCtx, edges: list[Edge], timeline: list[int],
           removed_at: dict[str, Optional[int]]) -> dict:
    start = ctx.state_at(timeline[0])
    out: dict = {}
    for goal in PLAN_GOALS:
        if goal not in ctx.instances:
            continue
        found = plan_removal(ctx.instances, edges, start, goal, ctx.tax)
        plan = None if found is None else [_pair(v, t) for v, t in found.actions]
        out[goal] = {
            "plan": plan, "length": None if plan is None else len(plan),
            "relaxed": bool(found is not None and found.relaxed),
            "dropped": [] if found is None else sorted(e.label() for e in found.dropped),
            "removed_at": removed_at.get(goal),
            "demonstrated_length": _demonstrated(ctx, removed_at.get(goal)),
        }
    goals = _log_goals(ctx)
    plan, unplanned, relaxed, dropped = _plan_all(ctx, edges, start, goals)
    last = max((removed_at[g] for g in goals if removed_at.get(g) is not None), default=None)
    remaining = []
    for step in timeline:
        rest, missing, _, _ = _plan_all(ctx, edges, ctx.state_at(step), goals)
        remaining.append(None if missing else len(rest))
    out[ALL_GOAL] = {
        "goals": goals, "plan": plan, "length": None if unplanned else len(plan),
        "unplanned": unplanned, "relaxed": relaxed, "dropped": dropped,
        "removed_at": last, "demonstrated_length": _demonstrated(ctx, last),
        "remaining_steps_by_step": remaining,
    }
    return out


def parse_violation(line: str) -> dict:
    """One :func:`~tda.core.graph.validate_sequence` line as its fields.

    ``kind`` is :data:`SUCCESS_VIOLATES` (``edge``, ``blocker`` and the
    blocker's state filled in) or :data:`FAILED_WITHOUT_UNMET` (they are
    ``None``); a line of neither shape keeps ``kind: "other"`` and the text.
    """
    m = _BREACH.match(line)
    if m:
        try:
            state = ast.literal_eval(m.group(9))
        except (ValueError, SyntaxError):
            state = m.group(9)
        return {"step": int(m.group(1)), "verb": m.group(2), "target": m.group(3),
                "edge": m.group(4), "edge_type": m.group(5), "blocker": m.group(8),
                "blocker_state": state, "kind": SUCCESS_VIOLATES, "line": line}
    m = _HINT.match(line)
    if m:
        return {"step": int(m.group(1)), "verb": m.group(2), "target": m.group(3),
                "edge": None, "edge_type": None, "blocker": None, "blocker_state": None,
                "kind": FAILED_WITHOUT_UNMET, "line": line}
    return {"step": None, "verb": None, "target": None, "edge": None, "edge_type": None,
            "blocker": None, "blocker_state": None, "kind": "other", "line": line}


def _timeline_agrees(ctx: DesktopCtx, edges: list[Edge], violation: dict) -> Optional[bool]:
    """Does the state timeline -- hand-written events included -- say the same at that step?

    :func:`~tda.core.graph.validate_sequence` replays the actions from the
    initial state and ignores every hand-written event after it (review round
    1, S2; a later task may change that), while ``instances[].timeline`` and
    ``legal_by_step`` fold them in. This re-asks the violation's question in
    the timeline's state the moment before the action (the state after the
    previous step, then the same step's earlier actions): for a
    ``success_violates`` line, is the edge still unmet; for a
    ``failed_without_unmet`` line, is there still no unmet precondition.
    ``None`` for a line of another shape or an action the timeline lacks.
    """
    kind = violation["kind"]
    if kind not in (SUCCESS_VIOLATES, FAILED_WITHOUT_UNMET):
        return None
    step, verb, target = violation["step"], violation["verb"], violation["target"]
    wanted = SUCCESS if kind == SUCCESS_VIOLATES else None
    sim = _copy(ctx.state_at(step - 1))
    found = False
    for action in ctx.actions_at(step, successful_only=False):
        if action.verb == verb and action.target == target and (
                (action.result == SUCCESS) == (wanted == SUCCESS)):
            found = True
            break
        if action.result == SUCCESS:
            _apply(ctx.tax, ctx.instances, sim, action.verb, action.target)
    if not found:
        return None
    if kind == SUCCESS_VIOLATES:
        edge = next((e for e in edges if e.label() == violation["edge"]), None)
        return None if edge is None else bool(unmet([edge], sim, "required"))
    return not unmet(applicable_preconditions(edges, (verb, target)), sim, "required")


def build_desktop(db: Db, tax: Taxonomy, desktop: int, view: str = DEFAULT_VIEW) -> dict:
    """The whole ``D<nn>.json`` of one desktop, as a dict (nothing written)."""
    ctx = load_ctx(db, tax, desktop, view)   # no drafts, context or gallery-only parts
    edges = sorted(active_edges(constraint_edges(edges_from_db(db, desktop))),
                   key=_edge_order)
    timeline = sorted({INITIAL_STEP} | set(ctx.steps) | {a.step for a in ctx.actions})
    instances, removed_at = _instances(ctx, timeline)
    scan: CoverScan = scan_covers(db, ctx, view, edges, removed_at)

    settled = _settled(db, desktop)
    initial = initial_overrides(e for e in db.events(desktop) if not e.auto)
    lines = validate_sequence(settled, edges, db.actions(desktop), tax, initial=initial)
    violations = [parse_violation(line) for line in lines]
    for violation in violations:
        violation["timeline_agrees"] = _timeline_agrees(ctx, edges, violation)
    deadlocks = [d.label() for d in find_deadlocks(edges, settled, tax, initial=initial)]
    dead_ends = [d.label() for d in find_dead_ends(edges, settled, tax, initial=initial)]

    steps = _steps(ctx, timeline)
    meta_row = db.get_desktop(desktop) or {}
    by_type = Counter(e.type for e in edges)
    candidates = sum(1 for c in scan.covers if c["candidate"])
    no_edge = sum(1 for c in scan.covers if c["no_edge"])
    name, sheet, name_note = model_names(meta_row)
    meta = {
        "desktop": int(desktop),
        "brand": meta_row.get("brand"),
        # the sheet's text as written, the family as the VLM export quotes it,
        # and the real name the scanner package uses (tda.core.export.release)
        "model": sheet,
        "model_family": meta_row.get("model_family"),
        "model_name": name,
        "model_note": name_note,
        "chassis_type": meta_row.get("chassis_type"),
        "chassis_platform": meta_row.get("chassis_platform"),
        "graph_version": graph_version(db, desktop),
        "export_graph_version": EXPORT_GRAPH_VERSION,
        "view": view,
        "pixel_views": [view] if scan.confirmed_frames else [],
        "counts": {
            "instances": len(instances), "steps": len(steps),
            "actions": len(ctx.actions), "edges": len(edges),
            "edges_by_type": {t: by_type[t] for t in HARD_TYPES if by_type.get(t)},
            "confirmed_frames": scan.confirmed_frames,
            "frames_with_covers": scan.frames_with_covers,
            "covers": len(scan.covers), "no_edge": no_edge, "candidates": candidates,
            "violations": len(violations),
            "violations_timeline_disagrees": sum(
                1 for v in violations if v["timeline_agrees"] is False),
            "deadlocks": len(deadlocks), "dead_ends": len(dead_ends),
        },
    }
    return {
        "meta": meta,
        "instances": instances,
        "steps": steps,
        "edges": _edges(ctx, edges, timeline),
        "covers": scan.covers,
        "legal_by_step": _legal_by_step(ctx, edges, timeline),
        "plans": _plans(ctx, edges, timeline, removed_at),
        "violations": violations,
        "deadlocks": deadlocks,
        "dead_ends": dead_ends,
    }


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _desktop_row(doc: dict) -> dict:
    meta, counts = doc["meta"], doc["meta"]["counts"]
    judged = [e for e in doc["legal_by_step"] if e["next_is_legal"] is not None]
    legal = sum(1 for e in judged if e["next_is_legal"])
    return {
        "desktop": meta["desktop"], "graph_version": meta["graph_version"],
        "model_family": meta["model_family"],
        "edges": counts["edges"], "edges_by_type": counts["edges_by_type"],
        "confirmed_frames": counts["confirmed_frames"],
        "covers": counts["covers"], "no_edge": counts["no_edge"],
        "candidates": counts["candidates"],
        "violations": counts["violations"],
        "violations_timeline_disagrees": counts.get("violations_timeline_disagrees", 0),
        "deadlocks": counts["deadlocks"], "dead_ends": counts["dead_ends"],
        "next_is_legal": {"legal": legal, "judged": len(judged),
                          "ratio": _ratio(legal, len(judged))},
        "plans": {goal: {"length": p["length"], "demonstrated_length": p["demonstrated_length"]}
                  for goal, p in doc["plans"].items()},
    }


def summarize(docs: Iterable[dict], view: str, failed: Optional[dict] = None) -> dict:
    """``summary.json``: per-desktop counts, candidates grouped by class pair, all violations."""
    docs = sorted(docs, key=lambda d: d["meta"]["desktop"])
    rows = [_desktop_row(d) for d in docs]
    groups: dict[str, dict] = {}
    violations = []
    by_type: Counter = Counter()
    for doc in docs:
        desktop = doc["meta"]["desktop"]
        by_type.update(doc["meta"]["counts"]["edges_by_type"])
        for cover in doc["covers"]:
            if not cover["candidate"]:
                continue
            name = f"{cover['above_class']} → {cover['below_class']}"
            group = groups.setdefault(name, {
                "pair": name, "above_class": cover["above_class"],
                "below_class": cover["below_class"], "desktops": [], "pairs": [],
                "below_removed": 0, "above_removed_first": 0})
            if desktop not in group["desktops"]:
                group["desktops"].append(desktop)
            group["pairs"].append(f"D{desktop:02d} {cover['above']} → {cover['below']} "
                                  f"({cover['n_frames']} frames, max {cover['max_px']} px)")
            above, below = cover["above_removed_at"], cover["below_removed_at"]
            group["below_removed"] += int(below is not None)
            group["above_removed_first"] += int(below is not None and above is not None
                                                and above < below)
        for v in doc["violations"]:
            violations.append({"desktop": desktop, **v})
    for group in groups.values():
        group["n_desktops"] = len(group["desktops"])
        group["n_pairs"] = len(group["pairs"])
    ranked = sorted(groups.values(), key=lambda g: (-g["n_desktops"], -g["n_pairs"], g["pair"]))
    judged = sum(r["next_is_legal"]["judged"] for r in rows)
    legal = sum(r["next_is_legal"]["legal"] for r in rows)
    kinds = Counter(v["kind"] for v in violations)
    totals = {
        "desktops": len(rows),
        "edges": sum(r["edges"] for r in rows),
        "edges_by_type": {t: by_type[t] for t in HARD_TYPES if by_type.get(t)},
        "confirmed_frames": sum(r["confirmed_frames"] for r in rows),
        "covers": sum(r["covers"] for r in rows),
        "no_edge": sum(r["no_edge"] for r in rows),
        "candidates": sum(r["candidates"] for r in rows),
        "candidate_class_pairs": len(ranked),
        "violations": len(violations),
        "violations_by_kind": dict(sorted(kinds.items())),
        "violations_timeline_agrees": sum(1 for v in violations
                                          if v.get("timeline_agrees") is True),
        "violations_timeline_disagrees": sum(1 for v in violations
                                             if v.get("timeline_agrees") is False),
        "deadlocks": sum(r["deadlocks"] for r in rows),
        "dead_ends": sum(r["dead_ends"] for r in rows),
        "next_is_legal": {"legal": legal, "judged": judged, "ratio": _ratio(legal, judged)},
    }
    return {
        "export_graph_version": EXPORT_GRAPH_VERSION,
        "view": view,
        "desktops": [r["desktop"] for r in rows],
        "failed": {str(k): v for k, v in sorted((failed or {}).items())},
        "totals": totals,
        "per_desktop": rows,
        "candidates_by_class_pair": ranked,
        "violations": violations,
    }


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def export_graph(db: Db, tax: Taxonomy, desktops: Iterable[int], out_dir: str,
                 view: str = DEFAULT_VIEW,
                 log: Optional[Callable[[str], None]] = None) -> dict:
    """Write ``D<nn>.json`` per desktop, ``summary.json`` and ``README.md`` into ``out_dir``.

    A desktop that raises is reported, left out of the files and listed under
    ``summary["failed"]`` rather than ending the run. Returns the summary.
    Raises ``ValueError`` for a view that is not one of the teardown sequence's
    (:data:`tda.core.model.VIEWS`): a components-gallery shot is no step.
    """
    from tda.core.export.graph_readme import readme_text

    if view not in VIEWS:
        raise ValueError(f"--view must be one of {', '.join(VIEWS)} (the teardown "
                         f"sequence's views), not {view!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    failed: dict[int, str] = {}
    for desktop in sorted({int(d) for d in desktops}):
        try:
            doc = build_desktop(db, tax, desktop, view)
        except Exception as exc:  # one bad desktop must not end the run
            failed[desktop] = f"{type(exc).__name__}: {exc}"
            if log:
                log(f"[export-graph] D{desktop:02d}: 失败 / FAILED, {failed[desktop]}")
            continue
        _write(out / desktop_file_name(desktop), dumps(doc))
        docs.append(doc)
        if log:
            c = doc["meta"]["counts"]
            log(f"[export-graph] D{desktop:02d}: {c['edges']} edges, {c['covers']} covers "
                f"({c['candidates']} candidates) on {c['confirmed_frames']} confirmed frames, "
                f"{c['violations']} violations")
    summary = summarize(docs, view, failed)
    _write(out / SUMMARY_NAME, dumps(summary))
    _write(out / README_NAME, readme_text(summary))
    return summary
