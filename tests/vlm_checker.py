"""An independent re-derivation of every VLM answer (not a test module).

The point of this file is that it shares **no code** with the generator. It
reads the database and the taxonomy -- the data -- and re-implements the spec
from scratch: its own fold of the action log into a state, its own transcription
of the precondition table, its own notion of which verb may be applied to which
class in which state. Nothing from ``tda.core.graph*``, ``tda.core.states`` or
``tda.core.export.vlm*`` is imported, so a bug that lives in one of those
modules cannot hide by being used on both sides of the assertion.

:class:`Checker` dispatches on the record's ``answer_check`` block, which is the
contract: a question whose ``answer_check`` this file cannot execute is a
question the export may not emit, and :meth:`Checker.check` says so. Every
derivation is a real one -- V12's events come from a state diff and V15's
same-moment answer from comparing two folded states -- so corrupting a record
makes this file fail, which ``test_vlm_tasks.py`` asserts directly.

**The tables below are transcribed, and two of them are not the spec's literal
text.** :data:`REQUIRED_STATES` is spec 7.2 verbatim. :data:`GATES` and
:data:`BLOCKED_GATES` are spec 7.2 **as amended by the implementation**
(``tda.core.graph_rules``):

* the spec says every hard edge gates every verb, which across the 66 sheets
  produced 34 "displace psu.01 violates connected_to(...)" lines, every one of
  them describing correct work -- swinging a PSU aside is how you reach its
  plugs. The amendment distinguishes *moving the part* from *reaching it*;
* a ``blocked_by`` edge gates by its ``mode``: a cable under tension only stops
  the part leaving, something in the way stops it moving at all, and something
  you cannot reach past stops everything. A missing or unrecognised mode gates
  every verb, which is the loud way to be wrong.

:data:`INACTIVE_STATUSES` is the third transcription: a rejected edge and an
*orphaned decision* -- a human's yes or no about a rule the derivation no longer
produces -- gate nothing.

Transcribing rather than importing means a change to any of them has to be a
deliberate act in two files, which is what an independent checker is for.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

#: spec 7.2: the blocker must be in one of these states. ``covered_by`` takes
#: ``unplugged`` too (spec v1.55, task W3j): an unplugged connector has no
#: geometry (taxonomy ``needs_mask``, spec 6.2), so it covers nothing. Changed
#: here by hand, on purpose, in step with ``tda.core.graph_rules`` -- the table
#: is transcribed, not imported (see the module docstring).
REQUIRED_STATES = {
    "fastened_by": {"loosened", "removed"},
    "connected_to": {"unplugged", "removed"},
    "locked_by": {"open"},
    "covered_by": {"open", "removed", "displaced", "unplugged"},
    "blocked_by": {"removed", "displaced", "released"},
}

#: Every verb a hard constraint can stand in the way of (spec 7.2).
GATED_VERBS = {"remove", "displace", "open", "unscrew", "disconnect", "release"}

#: Which of those verbs each edge type actually gates (spec 7.2 as amended --
#: see the module docstring).
GATES = {
    "fastened_by": {"remove", "displace", "open"},
    "locked_by": {"remove", "displace", "open"},
    "covered_by": set(GATED_VERBS),
    "blocked_by": set(GATED_VERBS),
    "connected_to": {"remove"},
}

#: What each ``blocked_by`` mode stops. A missing or unknown mode gates
#: everything, because gating too little drops a constraint silently.
BLOCKED_GATES = {
    "cable_tension": {"remove"},
    "physical_path": {"remove", "displace"},
    "tool_access": set(GATED_VERBS),
    # a part mounted on the board (task W3d): as physical_path
    "mounted": {"remove", "displace"},
    # a load the part carries (task W3g): a drive in the cage; the cage may move
    "contained": {"remove"},
    # a part lying loose in the case (R16, the owner 2026-11-04): it comes out first --
    # every verb on anything outside its own way out waits for it
    "loose": set(GATED_VERBS),
}

#: ``blocked_by`` modes whose blocker must reach a narrower set of states
#: (task W3d): a part mounted on the board has to be removed, not swung aside;
#: a drive in the cage likewise (``contained``, task W3g); a loose part has to be out.
BLOCKED_REQUIRED = {"mounted": {"removed"}, "contained": {"removed"}, "loose": {"removed"}}

#: The log's vocabulary per ``(class, state)`` (the graph audit of 2026-11-04), transcribed
#: from the export's own table on purpose: V6's frontier offers nothing else.
LOG_VOCAB = {
    ("cable_clip", "closed"): {"release"}, ("card_latch", "closed"): {"open", "remove"},
    ("case_fan", "installed"): {"remove"}, ("connector", "plugged"): {"disconnect"},
    ("cooler_bracket", "installed"): {"remove"}, ("cooler_latch", "closed"): {"remove"},
    ("cover", "closed"): {"open", "remove"}, ("cover", "open"): {"remove"},
    ("cpu", "installed"): {"remove"}, ("cpu_cooler", "installed"): {"remove", "displace"},
    ("cpu_cooler", "displaced"): {"remove"}, ("cpu_socket_lever", "closed"): {"open"},
    ("drive_cage", "installed"): {"remove", "displace"}, ("drive_cage", "displaced"): {"remove"},
    ("drive_latch", "closed"): {"open"}, ("expansion_card", "installed"): {"remove"},
    ("misc_part", "installed"): {"remove"}, ("motherboard", "installed"): {"remove"},
    ("optical_drive", "installed"): {"remove"}, ("psu", "installed"): {"remove", "displace"},
    ("psu", "displaced"): {"remove"}, ("psu_latch", "closed"): {"remove"},
    ("ram_latch", "closed"): {"open"}, ("ram_module", "installed"): {"remove"},
    ("screw", "fastened"): {"unscrew"}, ("storage_drive", "installed"): {"remove", "displace"},
    ("storage_drive", "displaced"): {"remove"},
}
#: Classes whose every action is one frontier group.
ONE_GROUP = ("connector", "ram_latch", "ram_module")

#: A rejected edge, and a decision about a rule that is no longer derived, gate
#: nothing. Everything else -- proposed, accepted -- does.
INACTIVE_STATUSES = {"rejected", "accepted_orphan", "rejected_orphan"}

HARD_TYPES = tuple(REQUIRED_STATES)


def gates(edge_type: str, mode=None) -> set:
    """Which verbs one edge gates, by type and (for ``blocked_by``) by mode."""
    if edge_type == "blocked_by":
        return BLOCKED_GATES.get(mode or "", set(GATED_VERBS))
    return GATES.get(edge_type, set(GATED_VERBS))


def required(edge_type: str, mode=None) -> set:
    """The states one edge's blocker must be in, by type and (for ``blocked_by``) mode."""
    if edge_type == "blocked_by" and mode in BLOCKED_REQUIRED:
        return BLOCKED_REQUIRED[mode]
    return REQUIRED_STATES.get(edge_type, set())

#: Which states a verb may be applied *from* (spec 6.2/6.3).
VERB_FROM_STATES = {
    "unscrew": {"fastened"},
    "disconnect": {"plugged"},
    "open": {"closed"},
    "release": {"closed", "routed"},
    "displace": {"installed"},
    "remove": None,  # any state but `removed`, except a screw (below)
}
REMOVE_FROM_STATES = {"screw": {"loosened"}}

REMOVED = "removed"
IN_CHASSIS = "in_chassis"
ON_BENCH = "on_bench"
CABLE_PREFIX = "cable:"
LS_PREFIX = "ls:"
SKIP_STEP_TYPES = {"ignore"}
ANSWERABLE = {"visible", "occluded_partial", "visible_tiny"}
CHASSIS_CLASS = "chassis"
UNGRADED_TOOLS = {None, "", "unknown"}


class CheckFailure(AssertionError):
    """One record whose stated answer is not what the data says."""


def _fail(record: dict, message: str) -> None:
    named = (record.get("label") or {}).get("readable_id") or record.get("id")
    raise CheckFailure(f"{named}: {message}")


def _key(action) -> tuple:
    return (action.step, action.idx)


class Checker:
    """Re-derives the answers of one desktop from the database alone."""

    def __init__(self, db, tax, desktop: int):
        self.db = db
        self.tax = tax
        self.desktop = int(desktop)
        self.instances = {
            key: rec for key, rec in db.instances(desktop).items()
            if not key.startswith(LS_PREFIX)
        }
        self.actions = sorted(db.actions(desktop), key=_key)
        self.steps = {s.step: s for s in db.steps(desktop)}
        self.edges = [
            row for row in db.relations(desktop)
            if row["type"] in HARD_TYPES
            and row["status"] not in INACTIVE_STATUSES
            and not str(row["target"]).startswith(LS_PREFIX)
            and not str(row["blocker"]).startswith(LS_PREFIX)
        ]
        self.manual = [e for e in db.events(desktop) if not e.auto]
        self._rows: dict[tuple[str, int], dict] = {}
        self._states: dict[int, dict] = {}
        self.checked = 0
        self.by_type: dict[str, int] = {}

    # -- the data ---------------------------------------------------------- #
    def rows(self, view: str, step: int) -> dict[str, dict]:
        from tda.core.model import FrameKey

        hit = self._rows.get((view, step))
        if hit is None:
            key = FrameKey(self.desktop, step, view)
            hit = {k: dict(v) for k, v in self.db.compiled(key).items()}
            # what hides a part in its place (U15, 被线挡住) is the frame's override
            for inst, fo in self.db.frame_overrides(key).items():
                if fo.occluder and inst in hit:
                    hit[inst]["occluder"] = fo.occluder
            self._rows[(view, step)] = hit
        return hit

    def step_type(self, step: int) -> str:
        rec = self.steps.get(step)
        return "normal" if rec is None else rec.step_type

    def exportable(self, step: int) -> bool:
        return self.step_type(step) not in SKIP_STEP_TYPES

    def log_steps(self) -> list[int]:
        return [s for s in sorted(self.steps) if self.exportable(s)]

    # -- the state machine, folded by hand --------------------------------- #
    def _effect(self, target: str, verb: str) -> Optional[tuple[str, str]]:
        rec = self.instances.get(target)
        if rec is None:
            cls = "cable" if target.startswith(CABLE_PREFIX) else None
            if cls is None:
                return None
            return self.tax.apply_verb(cls, {}, verb)
        return self.tax.apply_verb(rec.cls, rec.attrs, verb)

    def state_at(self, step: int) -> dict[str, dict]:
        """``key -> {"state", "placement", "left_with"}`` after logical step ``step``."""
        if step in self._states:
            return self._states[step]
        frame: dict[str, dict] = {
            key: {"state": self.tax.default_state(rec.cls),
                  "placement": IN_CHASSIS, "left_with": None}
            for key, rec in self.instances.items()
        }
        removed_at: dict[str, int] = {}
        pending = sorted((e for e in self.manual if e.step <= step), key=lambda e: e.step)

        def correct(before: int) -> None:
            # a hand-written event corrects the derived log *of its step*: it
            # folds after that step's actions and before any later one -- a
            # re-close at 20 must not undo an open at 14 from step 21 on (U8)
            while pending and pending[0].step < before:
                event = pending.pop(0)
                slot = frame.get(event.target)
                if slot is not None and event.attr in ("state", "placement"):
                    slot[event.attr] = event.new

        for action in self.actions:
            if action.step > step or action.result != "success":
                continue
            correct(action.step)
            effect = self._effect(action.target, action.verb)
            if effect is None:
                continue
            attr, value = effect
            slot = frame.get(action.target)
            if slot is None:
                if not action.target.startswith(CABLE_PREFIX):
                    continue
                slot = frame.setdefault(
                    action.target,
                    {"state": self.tax.default_state("cable"),
                     "placement": IN_CHASSIS, "left_with": None},
                )
            if attr == "state" and value == REMOVED and slot["state"] != REMOVED:
                removed_at[action.target] = action.step
            slot[attr] = value
            if attr == "state" and value == REMOVED and slot["placement"] == IN_CHASSIS:
                slot["placement"] = ON_BENCH
        correct(step + 1)       # it never replaces the derived log, only corrects it
        self._cascade(frame, removed_at)
        self._states[step] = frame
        return frame

    def _cascade(self, frame: dict[str, dict], removed_at: dict[str, int]) -> None:
        """Spec 3.3: an ``attached`` child leaves inside its parent."""
        children: dict[str, list[str]] = {}
        for key, rec in self.instances.items():
            if rec.attached and rec.parent:
                children.setdefault(rec.parent, []).append(key)
        for parent in sorted(children):
            rec = self.instances.get(parent)
            slot = frame.get(parent)
            if rec is None or slot is None or slot["state"] != REMOVED:
                continue
            if rec.cls == "connector":  # removing one means the cable went out
                continue
            stack, seen = [parent], {parent}
            while stack:
                node = stack.pop()
                for child in sorted(children.get(node, ())):
                    if child in seen:
                        continue
                    seen.add(child)
                    own, theirs = removed_at.get(child), removed_at.get(node)
                    if own is not None and theirs is not None and own < theirs:
                        continue  # it had already left on its own
                    slot = frame[child]
                    slot["state"] = REMOVED
                    if slot["placement"] == IN_CHASSIS:
                        slot["placement"] = ON_BENCH
                    slot["left_with"] = node
                    stack.append(child)

    def bare_state(self, step: int) -> dict[str, str]:
        """The folded state as ``key -> state``, for comparing two moments."""
        return {k: v["state"] for k, v in self.state_at(step).items()}

    # -- the constraint graph ---------------------------------------------- #
    def _blocker_state(self, frame: dict[str, dict], blocker: str) -> Optional[str]:
        slot = frame.get(blocker)
        if slot is not None:
            return slot["state"]
        if blocker.startswith(CABLE_PREFIX):
            return "routed"
        return None

    def unmet(self, verb: str, target: str, frame: dict[str, dict]) -> list[dict]:
        """Every required edge of spec 7.2 this action would break.

        A blocker that is ``removed``, or that is not a node of this desktop at
        all, satisfies any edge: it is out of the way, whatever the edge asked
        for. A ``recommended`` edge is a preference and gates nothing.

        The record's own moment edges (:meth:`moment_edges`: what covers a part in this
        frame, the loose parts that come out first) and a constrained question's rule are
        in :attr:`extra` while that record is checked.
        """
        if verb not in GATED_VERBS:
            return []
        out = []
        for edge in [*self.edges, *getattr(self, "extra", [])]:
            if edge["target"] != target or edge["necessity"] != "required":
                continue
            if verb not in gates(edge["type"], edge["mode"]):
                continue
            current = self._blocker_state(frame, edge["blocker"])
            if current is None or current == REMOVED:
                continue
            if current not in required(edge["type"], edge["mode"]):
                out.append(dict(edge))
        return out

    def applies(self, key: str, verb: str, frame: dict[str, dict]) -> bool:
        slot = frame.get(key)
        if slot is None and key.startswith(CABLE_PREFIX):
            slot = {"state": self.tax.default_state("cable")}
        if slot is None or slot["state"] == REMOVED:
            return False
        effect = self._effect(key, verb)
        if effect is None or effect[0] != "state" or effect[1] == slot["state"]:
            return False
        rec = self.instances.get(key)
        cls = rec.cls if rec is not None else "cable"
        allowed = (REMOVE_FROM_STATES.get(cls) if verb == "remove"
                   else VERB_FROM_STATES.get(verb))
        return allowed is None or slot["state"] in allowed

    def _candidates(self, frame: dict[str, dict]) -> list[tuple[str, str]]:
        out = [(verb, key) for key in self.instances for verb in self.tax.verbs
               if self.applies(key, verb, frame)]
        cables = {e["blocker"] for e in self.edges
                  if str(e["blocker"]).startswith(CABLE_PREFIX)}
        cables |= {k for k in frame if k.startswith(CABLE_PREFIX)}
        for key in cables:
            current = self._blocker_state(frame, key) or "routed"
            if current in VERB_FROM_STATES["release"]:
                out.append(("release", key))
        return out

    def permitted(self, step: int) -> set[tuple[str, str]]:
        """What no recorded constraint forbids -- an upper bound, never a truth."""
        frame = self.state_at(step)
        return {(v, t) for v, t in self._candidates(frame)
                if not self.unmet(v, t, frame)}

    def blocked(self, step: int) -> set[tuple[str, str]]:
        """What the graph certainly forbids at this state."""
        frame = self.state_at(step)
        return {(v, t) for v, t in self._candidates(frame)
                if self.unmet(v, t, frame)}

    def inapplicable(self, step: int) -> set[tuple[str, str]]:
        """What the *state* rules out: the verb cannot start from where it is.

        Only verbs that belong to the class and that were applicable to this
        very instance at some earlier step -- "has this already happened?" is a
        question about the picture, "does this verb exist?" is not.
        """
        frame = self.state_at(step)
        earlier = [self.state_at(j) for j in [0, *self.log_steps()] if j < step]
        out: set[tuple[str, str]] = set()
        for key, rec in self.instances.items():
            if bool(rec.attrs.get("implied")) or key.startswith(LS_PREFIX):
                continue
            if key not in frame:
                continue
            for verb in self.tax.verbs:
                if self._verb_effect(key, verb) is None:
                    continue
                if self.applies(key, verb, frame):
                    continue
                if any(self.applies(key, verb, state) for state in earlier):
                    out.add((verb, key))
        return out

    def certainly_wrong(self, step: int) -> dict[tuple[str, str], str]:
        """``(verb, target) -> kind`` for everything that cannot be done now."""
        out = {pair: "state_inapplicable" for pair in self.inapplicable(step)}
        out.update({pair: "graph_blocked" for pair in self.blocked(step)})
        return out

    def successful_actions(self, step: int) -> list:
        return [a for a in self.actions
                if a.step == step and a.result == "success"
                and self.exportable(a.step)]

    def named(self, step: int) -> list:
        return [a for a in self.successful_actions(step)
                if _cls_of(self.instances, a.target) is not None]

    def changes_state(self, action) -> bool:
        """Does this action move the state machine? ``reorient`` does not."""
        effect = self._effect(action.target, action.verb)
        return effect is not None and effect[0] == "state"

    def planning_actions(self, step: int) -> list:
        return [a for a in self.named(step) if self.changes_state(a)]

    # -- part groups (redesign 2026-11-03), re-derived by hand --------------- #
    def _match_class(self, key: str) -> Optional[str]:
        rec = self.instances.get(key)
        if rec is None:
            return None
        if rec.cls == "screw" and rec.attrs.get("role"):
            return f"screw.{rec.attrs['role']}"
        if rec.cls in ("cpu_cooler", "connector") and rec.attrs.get("kind"):
            return f"{rec.cls}.{rec.attrs['kind']}"
        return rec.cls

    def _anchor(self, key: str) -> str:
        rec = self.instances[key]
        if rec.cls == "screw":
            return rec.fastens or "-"
        if rec.cls == "connector":
            if not rec.socket_host and not rec.cable:
                return "-"
            return f"{rec.socket_host or '-'}|{rec.cable or '-'}"
        of = rec.attrs.get("of")
        return of if isinstance(of, str) and of in self.instances else "-"

    @property
    def groups(self) -> dict[str, list[str]]:
        """``gid -> members``: same match class and anchor, at least two real parts."""
        if getattr(self, "_groups", None) is None:
            buckets: dict[str, list[str]] = {}
            for key in sorted(self.instances):
                rec = self.instances[key]
                if rec.attrs.get("implied") or rec.cls not in self.tax.classes:
                    continue
                buckets.setdefault(f"{self._match_class(key)}@{self._anchor(key)}",
                                   []).append(key)
            self._groups = {g: m for g, m in buckets.items() if len(m) >= 2}
        return self._groups

    def group_of(self, key: str) -> Optional[str]:
        return next((g for g, m in self.groups.items() if key in m), None)

    def captive(self, key: str) -> bool:
        rec = self.instances.get(key)
        return bool(rec is not None and rec.cls == "screw" and rec.attrs.get("captive"))

    @staticmethod
    def present(frame: dict, key: str) -> bool:
        slot = frame.get(key)
        return slot is not None and slot["state"] != REMOVED

    def todo(self, frame: dict, verb: str, members: Iterable[str]) -> list[str]:
        """Members still in the asked situation; captive screws while their part is in."""
        out = []
        for key in members:
            if not self.present(frame, key):
                continue
            if verb == "unscrew" and self.captive(key):
                held = self.instances[key].fastens
                if held is None or self.present(frame, held):
                    out.append(key)
                continue
            if self.applies(key, verb, frame):
                out.append(key)
        return out

    def expand(self, frame: dict, verb: str, target: str) -> list[str]:
        """A group action's members still to do, or the part itself."""
        if target in self.groups:
            return self.todo(frame, verb, self.groups[target])
        return [target]

    def as_group(self, frame: dict, verb: str, key: str) -> str:
        """``key``'s group id when two or more members are still to do, else ``key``."""
        gid = self.group_of(key)
        if gid is not None and len(self.todo(frame, verb, self.groups[gid])) >= 2:
            return gid
        return key

    def grouped(self, frame: dict, pairs, keep=()) -> list[tuple[str, str]]:
        out, seen = [], set()
        for verb, key in pairs:
            rec = self.instances.get(key)
            item = (verb, key if rec is not None and rec.cls in keep
                    else self.as_group(frame, verb, key))
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def with_captive(self, frame: dict, plan: list) -> list:
        pending = sorted({g for g, m in self.groups.items()
                          if all(self.captive(k) for k in m)
                          and self.todo(frame, "unscrew", m)})
        for gid in reversed(pending):
            held = self.instances[self.groups[gid][0]].fastens
            if ("unscrew", gid) not in plan and any(t == held for _, t in plan):
                plan = [("unscrew", gid), *plan]
        return plan

    def demonstrated(self, after: int) -> Optional[tuple[int, Any]]:
        for step in self.log_steps():
            if step > after and self.named(step):
                action = self.named(step)[0]
                return (step, action) if self.changes_state(action) else None
        return None

    # -- the moment's own rules (owner 2026-11-04), rebuilt from the record ------ #
    def loose_parts(self, frame: dict) -> list[str]:
        """Parts flagged ``attrs.loose`` (R16) still in the machine."""
        return sorted(k for k, rec in self.instances.items()
                      if rec.attrs.get("loose") and self.present(frame, k))

    def moment_edges(self, record: dict, spec: dict) -> list[dict]:
        """The record's ``answer_check.moment`` as edge rows: ``covered_by`` per occluder,
        ``blocked_by`` (mode ``loose``) from every node outside the loose scope to each loose
        part. The occlusion is the frame's geometry and is taken as given (it is checked
        against the masks by ``tests/test_graph_owner_rulings.py``); the loose parts are
        re-derived here and must agree."""
        moment = spec.get("moment")
        if moment is None:
            return []
        step = int(spec.get("state_after", spec.get("start_step", record["label"]["step"])))
        frame = self.state_at(step)
        loose = list(moment.get("loose") or [])
        if loose != self.loose_parts(frame):
            _fail(record, f"loose parts {loose} != {self.loose_parts(frame)}")
        out = [{"type": "covered_by", "target": x, "blocker": y, "necessity": "required",
                "mode": None, "source": "occlusion"}
               for x, ys in sorted((moment.get("occluded") or {}).items()) for y in ys]
        if loose:
            scope = set(moment.get("loose_scope") or [])
            nodes = {k for k, v in frame.items() if v["state"] != REMOVED}
            nodes |= {e["blocker"] for e in self.edges if str(e["blocker"]).startswith(CABLE_PREFIX)}
            for key in sorted(nodes - scope - set(loose)):
                out.extend({"type": "blocked_by", "target": key, "blocker": part,
                            "necessity": "required", "mode": "loose", "source": "loose"}
                           for part in loose)
        return out

    @staticmethod
    def constraint_edges(spec: dict) -> list[dict]:
        rule = spec.get("constraint") or {}
        return [{"type": e["type"], "target": e["target"], "blocker": e["blocker"],
                 "necessity": e.get("necessity", "required"), "mode": e.get("mode"),
                 "source": "constraint"} for e in rule.get("edges") or []]

    # -- dispatch ----------------------------------------------------------- #
    def check(self, record: dict) -> None:
        self._check_common(record)
        spec = record["label"].get("answer_check")
        if not isinstance(spec, dict) or "type" not in spec:
            _fail(record, "carries no answer_check block")
        handler = getattr(self, f"_check_{spec['type']}", None)
        if handler is None:
            _fail(record, f"answer_check type {spec['type']!r} has no checker")
        self.extra = self.moment_edges(record, spec)
        try:
            handler(record, spec)
        finally:
            self.extra = []
        text = (record["label"].get("rationale") or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            _fail(record, "a record without its reasoning in words (rationale.text)")
        if re.search(r"\b[a-z_]+\.\d{2}\b", text) or "violates" in text or "_by(" in text:
            _fail(record, f"rationale {text[:80]!r} names a key or the graph's words")
        self.checked += 1
        self.by_type[spec["type"]] = self.by_type.get(spec["type"], 0) + 1

    def check_all(self, records: Iterable[dict]) -> int:
        for record in records:
            self.check(record)
        return self.checked

    # -- the checks --------------------------------------------------------- #
    def _check_common(self, record: dict) -> None:
        if set(record) != {"id", "prompt", "label"}:
            _fail(record, f"a record is id/prompt/label, not {sorted(record)}")
        prompt, label = record["prompt"], record["label"]
        if set(prompt) - {"task", "images", "question", "options", "marks", "terms",
                          "state_terms", "image_legend", "hint_regions"}:
            _fail(record, f"the prompt side carries {sorted(prompt)}")
        if prompt.get("hint_regions") and (prompt["task"] != "V3"
                                           or label.get("v3_form") != "roi"):
            _fail(record, "a hint region outside V3's roi form")
        for hint in prompt.get("hint_regions") or []:
            if not 0 <= int(hint["image"]) < len(prompt["images"]):
                _fail(record, f"hint on image {hint['image']} of {len(prompt['images'])}")
        if prompt.get("marks") and prompt["task"] in ("V1", "V8"):
            _fail(record, "a mark on a task whose box is the answer")
        for mark in prompt.get("marks") or []:
            if not 0 <= int(mark["image"]) < len(prompt["images"]):
                _fail(record, f"mark on image {mark['image']} of {len(prompt['images'])}")
            if len(mark["bbox"]) != 4 or not re.fullmatch(r"[A-Z]", str(mark["tag"])):
                _fail(record, f"a malformed mark {mark}")
        if prompt.get("image_legend") is not None:
            if len(prompt["image_legend"]) != len(prompt["images"]):
                _fail(record, "the image legend and the images disagree")
            if label.get("views") != [e["view"] for e in prompt["image_legend"]]:
                _fail(record, "label.views is not the legend's views")
        for field in ("readable_id", "answer", "answer_check", "evidence",
                      "rationale", "desktop", "step", "view", "layer",
                      "template_id", "tier", "verified", "graph_version",
                      "model_family"):
            if field not in label:
                _fail(record, f"missing label field {field!r}")
        if not re.fullmatch(r"rec_[0-9a-f]{16}", str(record["id"])):
            _fail(record, f"the record id {record['id']!r} is not opaque")
        if label["desktop"] != self.desktop:
            _fail(record, "wrong desktop")
        if LS_PREFIX in repr(record):
            _fail(record, "a Label Studio draft reached the record")
        if not self.exportable(int(label["step"])):
            _fail(record, "emitted on a step nobody annotates")
        for image in prompt["images"]:
            if not re.fullmatch(r"img_[0-9a-f]{16}", str(image)):
                _fail(record, f"image {image!r} is not an opaque id")

    @staticmethod
    def _answer(record: dict) -> dict:
        return record["label"]["answer"]

    # V1, V8 -- a set of instances, each with a box
    def _check_boxes(self, record: dict, spec: dict) -> None:
        label = record["label"]
        rows = self.rows(label["view"], label["step"])
        answer = self._answer(record)
        items = answer[spec["field"]] if spec.get("field") else [answer]
        seen = set()
        for item in items:
            key = item[spec["id_field"]]
            seen.add(key)
            if key not in rows:
                _fail(record, f"{key} has no compiled row on this frame")
            if rows[key].get("visibility") not in ANSWERABLE:
                _fail(record, f"{key} cannot be pointed at in this view")
            want = _row_box(rows[key])
            if _iou(want, item[spec["box_field"]]) < 0.999:
                _fail(record, f"{key}: box {item[spec['box_field']]} != {want}")
        if spec.get("exhaustive"):
            dropped = set(spec.get("exclude_classes") or ())
            expect = {k for k, r in rows.items()
                      if r.get("visibility") in ANSWERABLE and _row_box(r) is not None
                      and k in self.instances
                      and self.instances[k].cls not in dropped
                      and self.instances[k].cls != "screw_hole"}
            if seen != expect:
                _fail(record, f"component set {sorted(seen)} != {sorted(expect)}")

    # V8 on a class with several instances: every visible member of that class
    def _check_components(self, record: dict, spec: dict) -> None:
        label = record["label"]
        rows = self.rows(label["view"], label["step"])
        mc = spec["class_filter"]
        seen = set()
        for item in self._answer(record)["components"]:
            key = item["instance"]
            seen.add(key)
            if self._match_class(key) != mc:
                _fail(record, f"{key} is not a {mc}")
            if key not in rows or rows[key].get("visibility") not in ANSWERABLE:
                _fail(record, f"{key} cannot be pointed at in this view")
            if _iou(_row_box(rows[key]), item["bbox"]) < 0.999:
                _fail(record, f"{key}: box {item['bbox']} is not its row's")
        expect = {k for k, r in rows.items() if r.get("visibility") in ANSWERABLE
                  and _row_box(r) is not None and self._match_class(k) == mc}
        if seen != expect:
            _fail(record, f"component set {sorted(seen)} != {sorted(expect)}")
        if sum(1 for k in self.instances if self._match_class(k) == mc) < 2:
            _fail(record, "the set form on a class with one part")

    # V2 states and counts, V3, V12, V14, V15
    def _check_exact(self, record: dict, spec: dict) -> None:
        answer = self._answer(record)
        for field, expected in self._expected(record, spec).items():
            if field not in spec.get("fields", [field]):
                continue
            if answer.get(field) != expected:
                _fail(record, f"{field}: {answer.get(field)!r} != {expected!r}")

    @staticmethod
    def _shown(state, presence: bool):
        """A screw's state as presence when the check asks in that space (W3b)."""
        if not presence or state is None:
            return state
        return {"fastened": "present", "loosened": "present"}.get(state, state)

    def _expected(self, record: dict, spec: dict) -> dict:
        kind = spec.get("derive")
        label = record["label"]
        step, view = int(label["step"]), label["view"]
        presence = spec.get("state_space") == "presence"
        if kind == "instance_state":
            keys = spec.get("members") or [spec["instance"]]
            if spec.get("members") and self.group_of(keys[0]) != spec["instance"]:
                _fail(record, "members of a group asked under another id")
            if "cannot_tell" not in (spec.get("allowed") or []):
                _fail(record, "a V2 state question without the cannot_tell answer")
            rows = self.rows(view, step)
            hidden = [k for k in keys if (rows.get(k) or {}).get("visibility")
                      in ("occluded_full", "too_small", "out_of_view")]
            if spec.get("hidden"):
                # V14 merged into V2 (owner 2026-11-04): a part in place this view cannot see
                if len(hidden) != len(keys):
                    _fail(record, "a cannot_tell question on a part this view can see")
                for key in keys:
                    if not self.present(self.state_at(step), key):
                        _fail(record, f"{key} is gone, not hidden")
                return {"state": "cannot_tell"}
            if hidden:
                _fail(record, f"a state asked of {hidden}, which this view cannot see")
            shown = set()
            for key in keys:
                if presence and self.instances[key].attrs.get("captive"):
                    _fail(record, "a captive screw has no state a picture shows (W3b)")
                shown.add(self._shown(self.state_at(step)[key]["state"], presence))
            if len(shown) != 1:
                _fail(record, f"a group asked as one whose members differ: {shown}")
            return {"state": shown.pop()}
        if kind == "screw_count":
            role, state = spec["role"], spec["state"]
            frame = self.state_at(step)
            n = sum(1 for key, rec in self.instances.items()
                    if rec.cls == "screw"
                    and str(rec.attrs.get("role") or "other") == role
                    and not (presence and rec.attrs.get("captive"))
                    and self._shown(frame[key]["state"], presence) == state)
            return {"count": n}
        if kind == "action_slots":
            actions = self.named(step)
            if spec.get("visible_only"):
                # W3b: unscrewing a captive screw is not a visible action
                actions = [a for a in actions if not (
                    a.verb == "unscrew"
                    and self.instances.get(a.target) is not None
                    and self.instances[a.target].attrs.get("captive"))]
            if not actions:
                _fail(record, "a change question on a step with no action")
            first = actions[0]
            out = {"verb": first.verb, "target_instance": first.target,
                   "target_class": _cls_of(self.instances, first.target)}
            if first.tool not in UNGRADED_TOOLS:
                out["tool"] = first.tool
            elif "tool" in spec.get("fields", ()):
                _fail(record, "an unknown tool must not be a graded field")
            return out
        if kind == "changed":
            old, new = self.state_at(step - 1), self.state_at(step)
            # fan screw holes are a detection category, never a part a question is about
            events = [{"target": key, "old": old[key]["state"], "new": new[key]["state"]}
                      for key in sorted(old.keys() & new.keys())
                      if old[key]["state"] != new[key]["state"]
                      and (self.instances.get(key) is None
                           or self.instances[key].cls != "screw_hole")]
            if spec.get("step_type") == "dupli" or (
                    self.steps.get(step) is not None and self.steps[step].dupli):
                if events:
                    _fail(record, "a dupli step whose state actually moved")
                return {"changed": False, "events": []}
            if spec.get("reason") == "captive_screw_loosened":
                # W3b: only captive screws moved, and a picture cannot show it
                if not events or any(
                        not (self.instances.get(e["target"]) is not None
                             and self.instances[e["target"]].attrs.get("captive"))
                        for e in events):
                    _fail(record, "a captive no-change pair where something else moved")
                return {"changed": False, "events": []}
            return {"changed": bool(events), "events": events}
        if kind == "answerable":
            answers = set()
            for key in spec.get("members") or [spec["instance"]]:
                row = self.rows(view, step).get(key)
                if row is None:
                    _fail(record, f"{key} has no compiled row here")
                ok = row.get("visibility") in ANSWERABLE
                answers.add((ok, self._shown(self.state_at(step)[key]["state"], presence)
                             if ok else None))
            if len(answers) != 1:
                _fail(record, f"a class asked as one whose members differ: {answers}")
            ok, state = answers.pop()
            return {"answerable": ok, "state": state}
        if kind == "cross_view":
            row = self.rows(spec["answer_view"], step).get(spec["instance"])
            if row is None or row.get("visibility") not in ANSWERABLE:
                _fail(record, "the answering view cannot see it either")
            blind = self.rows(spec["blind_view"], step).get(spec["instance"])
            if blind is not None and blind.get("visibility") in ANSWERABLE:
                _fail(record, "the blind view could answer after all")
            return {"state": self._shown(self.state_at(step)[spec["instance"]]["state"],
                                         presence),
                    "best_view": spec["answer_view"]}
        if kind == "same_moment":
            # derived, not copied: two folded states, compared
            other = int(spec["other_step"])
            return {"same_moment": self.bare_state(step) == self.bare_state(other)}
        _fail(record, f"unknown derivation {kind!r}")
        return {}

    # V4
    def _check_feasibility(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(spec.get("state_after", label["step"]))
        frame = self.state_at(step)
        verb, target = spec["verb"], spec["target"]
        answer = self._answer(record)
        source = spec.get("truth_source")
        if source == "state_inapplicable":
            # the verb cannot start from where the target is, so no edge is
            # being asked about -- whether one would *also* have blocked it is
            # beside the point and, on a desktop with a log gap, is noise
            self._check_inapplicable(record, spec, frame, step, verb, target)
            return
        members = self.expand(frame, verb, target)
        if target in self.groups:
            if sorted(spec.get("members") or []) != sorted(members) or len(members) < 2:
                _fail(record, f"group members {spec.get('members')} != {members}")
        bad = []
        shapes = set()
        for member in members:
            if not self.applies(member, verb, frame):
                _fail(record, f"{verb} {member} does not apply in this state at all")
            if self.captive(member):
                _fail(record, "an answer resting on a captive screw's tightness")
            mine = self.unmet(verb, member, frame)
            for edge in mine:
                if self.captive(edge["blocker"]):
                    _fail(record, "an answer resting on a captive screw's tightness")
            shapes.add(tuple(sorted({self.group_of(e["blocker"]) or e["blocker"]
                                     for e in mine})))
            bad.extend(mine)
        if len(shapes) > 1:
            _fail(record, f"a group asked as one whose members differ: {shapes}")
        # blockers are blocker groups (redesign 2026-11-03)
        blockers = sorted({self.group_of(e["blocker"]) or e["blocker"] for e in bad})
        if sorted(answer.get("blockers") or []) != blockers:
            _fail(record, f"blockers {answer.get('blockers')} != {blockers}")

        # the internal code is label side (answer_check.reason_code); answer.reason is the
        # reason in what the picture shows (owner 2026-11-04): never a code, never graph words,
        # and it speaks of every kind of blocker -- or ends on what can now be done
        want_reason = None if source == "demonstrated" else {
            "occlusion_blocked": "covered_now", "loose_first": "loose_part_first"}.get(
                source, "violates_edge")
        if spec.get("reason_code") != want_reason:
            _fail(record, f"reason_code {spec.get('reason_code')!r} != {want_reason!r}")
        self._check_reason_text(record, answer.get("reason"), verb, bad)
        if source == "demonstrated":
            # a "yes" is only ever the action the log went on to perform
            shown = self.demonstrated(step)
            if shown is None or shown[1].verb != verb or shown[1].target not in members:
                _fail(record, f"{verb} {target} is not what the log demonstrates here")
            if answer["feasible"] is not True:
                _fail(record, "a demonstrated action must be feasible")
            if bad:
                _fail(record, "the graph forbids what the log demonstrates")
        elif source in ("graph_blocked", "graph_blocked_pixels", "failed_attempt",
                        "occlusion_blocked", "loose_first"):
            if source == "occlusion_blocked" and any(e.get("source") != "occlusion" for e in bad):
                _fail(record, "an occlusion negative that something else blocks too")
            if source == "loose_first" and any(e.get("source") != "loose" for e in bad):
                _fail(record, "a loose-first negative that something else blocks too")
            if not bad:
                _fail(record, "a negative whose blocker set is empty")
            if answer["feasible"] is not False:
                _fail(record, "a blocked action must not be feasible")
            if source == "failed_attempt":
                attempts = [a for a in self.actions
                            if a.step == step + 1 and a.result != "success"
                            and a.verb == verb and a.target in members]
                if not attempts:
                    _fail(record, "no failed attempt at the next step says this")
        else:
            _fail(record, f"unknown truth_source {source!r}")

    #: "the CPU can be <taken out>" (transcribed from vlm_tasks.PARTICIPLES).
    PARTICIPLES = {"remove": "taken out", "displace": "moved aside", "open": "opened",
                   "unscrew": "unscrewed", "disconnect": "unplugged", "release": "released"}

    def _check_reason_text(self, record: dict, text, verb: str, bad: list) -> None:
        """A V4 reason a person reads (owner 2026-11-04): words, no code, no graph vocabulary, no
        instance key; a "no" names the class of every blocker, a "yes" says what can be done."""
        if not isinstance(text, str) or not text.strip():
            _fail(record, f"reason {text!r} is no text")
        low = text.lower()
        for word in ("violates", "_by", "edge", "recall", "precondition", "already_",
                     "not_present"):
            if word in low:
                _fail(record, f"reason {text!r} speaks the graph's words ({word})")
        if re.search(r"\b[a-z_]+\.\d{2}\b", text) or "@" in text:
            _fail(record, f"reason {text!r} names an instance key")
        if bad:
            for edge in bad:
                blocker = edge["blocker"]
                cls = ("cable" if blocker.startswith("cable:")
                       else self.instances[blocker].cls)
                if cls.replace("_", " ") not in low:
                    _fail(record, f"reason {text!r} does not speak of the {cls}")
        elif not low.endswith(f"can be {self.PARTICIPLES.get(verb, verb)}"):
            _fail(record, f"reason {text!r} does not say what can now be done")

    def _check_inapplicable(self, record: dict, spec: dict, frame: dict,
                            step: int, verb: str, target: str) -> None:
        """The third certainty: the state, not the graph, rules this action out.

        Three things have to hold, and the middle one is what makes it a
        *question*: the verb cannot be performed now, it could be performed at
        some earlier step on this very instance, and the reason is the one the
        state gives.
        """
        answer = self._answer(record)
        if answer["feasible"] is not False:
            _fail(record, "an inapplicable action must not be feasible")
        if answer.get("blockers"):
            _fail(record, "a state negative names no blocker")
        rec = self.instances.get(target)
        # `implied` is the flag `tda.core.implied` writes on an identity nobody
        # operated; transcribed, like everything else in this file
        if rec is None or bool(rec.attrs.get("implied")):
            _fail(record, f"{target} is not an instance a question may name")
        if self._verb_effect(target, verb) is None:
            _fail(record, f"{verb} does not belong to {target}'s class at all")
        if self.applies(target, verb, frame):
            _fail(record, f"{verb} {target} applies in this state after all")
        earlier = [j for j in [0, *self.log_steps()] if j < step
                   and self.applies(target, verb, self.state_at(j))]
        if not earlier:
            _fail(record, f"{verb} {target} was never possible, so it is no question")
        want = self._inapplicable_reason(frame, target)
        if answer.get("reason") != want:
            _fail(record, f"reason {answer.get('reason')!r} != {want!r}")

    def _verb_effect(self, target: str, verb: str):
        effect = self._effect(target, verb)
        return None if effect is None or effect[0] != "state" else effect[1]

    def _inapplicable_reason(self, frame: dict, key: str) -> str:
        """``not_present`` when it has gone, else ``already_<state>``."""
        slot = frame.get(key)
        if slot is None:
            return "not_present"
        if slot["state"] == REMOVED:
            return "not_present"
        parent = frame.get(slot.get("left_with") or "")
        if slot.get("left_with") and (parent is None
                                      or parent["state"] == REMOVED
                                      or parent["placement"] != IN_CHASSIS):
            return "not_present"
        return f"already_{slot['state']}"

    # V5
    def _check_bounded_set(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        shown = self.demonstrated(step)
        if shown is None:
            _fail(record, "a bounded set with nothing demonstrated after it")
        nxt, action = shown
        if int(spec["reference_step"]) != nxt:
            _fail(record, f"reference step {spec['reference_step']} != {nxt}")
        must = _pairs(answer["must_include"])
        must_not = _pairs(answer["must_not_include"])
        upper = _pairs(answer["permitted_upper_bound"])
        if must != {(action.verb, action.target)}:
            _fail(record, f"must_include {sorted(must)} is not the demonstrated action")
        if must_not != self.blocked(step):
            _fail(record, f"must_not_include differs: "
                          f"missing {sorted(self.blocked(step) - must_not)}, "
                          f"extra {sorted(must_not - self.blocked(step))}")
        if upper != self.permitted(step):
            _fail(record, f"upper bound differs: "
                          f"missing {sorted(self.permitted(step) - upper)}, "
                          f"extra {sorted(upper - self.permitted(step))}")
        if must & must_not:
            _fail(record, f"{sorted(must & must_not)} is both required and forbidden")
        if not must <= upper:
            _fail(record, "the demonstrated action is outside the upper bound")

    # V6 (the owner's 2026-11-04 rulings): every action possible now
    def _frontier_key(self, verb: str, key: str) -> str:
        rec = self.instances[key]
        if rec.cls in ONE_GROUP:
            return f"all:{rec.cls}"
        if rec.cls == "screw":
            return f"screws:{rec.fastens or rec.attrs.get('role') or key}"
        gid = self.group_of(key)
        return f"part:{gid}" if gid is not None else f"part:{key}"

    def _frontier(self, record: dict, step: int, extra: list) -> dict[str, set]:
        """``group key -> {(verb, target)}``, re-derived: legal under the graph, the record's
        moment edges and ``extra``; the log's vocabulary; a part a question may name, which
        the scanner frame can point at (or that only a cable hides)."""
        frame = self.state_at(step)
        rows = self.rows("scan", step)
        displaced = {a.target for a in self.actions if a.verb == "displace"}
        saved = self.extra
        self.extra = [*saved, *extra]
        try:
            legal = {(v, t) for v, t in self._candidates(frame)
                     if not t.startswith(CABLE_PREFIX) and not self.unmet(v, t, frame)}
        finally:
            self.extra = saved
        loose = self.loose_parts(frame)
        out: dict[str, set] = {}
        for verb, key in sorted(legal):
            rec = self.instances.get(key)
            if rec is None or rec.attrs.get("implied") or key.startswith(LS_PREFIX):
                continue
            if key in loose and verb != "remove":
                continue  # a loose part only comes out
            if verb not in LOG_VOCAB.get((rec.cls, frame[key]["state"]), set()):
                continue
            if verb == "displace" and key not in displaced:
                continue
            if rec.cls == "ram_latch":
                of = rec.attrs.get("of")
                if of and not self.present(frame, of):
                    continue
            row = rows.get(key) or {}
            seen = row.get("visibility") in ANSWERABLE and _row_box(row) is not None
            if not seen and row.get("occluder") != "cable":
                continue
            out.setdefault(self._frontier_key(verb, key), set()).add((verb, key))
        return out

    def _check_frontier(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        base = self._frontier(record, step, [])
        told = {g["key"]: {(a["verb"], a["target"]) for a in g["actions"]}
                for g in spec["groups"]}
        if told != base:
            _fail(record, f"the frontier differs: missing "
                          f"{sorted(set(base) - set(told))}, extra {sorted(set(told) - set(base))}")
        form = spec.get("form")
        if form == "mc":
            options = label.get("options") or []
            right = [o for o in options if o["kind"] == "frontier"]
            if len(right) != 1:
                _fail(record, "a multiple choice needs exactly one option inside the frontier")
            if {(a[0], a[1]) for a in right[0]["actions"]} != base.get(right[0]["group"]):
                _fail(record, "the frontier option is not one whole frontier group")
            frame = self.state_at(step)
            for o in options:
                if o["kind"] == "frontier":
                    continue
                if o["group"] in base:
                    _fail(record, f"a distractor {o['group']} inside the frontier")
                for verb, target in o["actions"]:
                    if not self.unmet(verb, target, frame):
                        _fail(record, f"distractor {verb} {target} is not blocked")
                    if not self.applies(target, verb, frame):
                        _fail(record, f"distractor {verb} {target} is done or meaningless")
            if len(options) - 1 < 2:
                _fail(record, "a multiple choice needs at least two blocked distractors")
            if (answer["verb"], answer["target"]) != (right[0]["verb"], right[0]["target"]):
                _fail(record, "the answer is not the frontier option")
            for shown in record["prompt"].get("options") or []:
                if "kind" in shown or "target" in shown:
                    _fail(record, "the prompt options give away which is which")
            return
        gold = base
        if form == "constrained":
            ruled = self._frontier(record, step, self.constraint_edges(spec))
            forbidden = sorted(set(base) - set(ruled))
            if not forbidden or sorted(spec.get("forbidden_groups") or []) != forbidden:
                _fail(record, f"forbidden groups {spec.get('forbidden_groups')} != {forbidden}")
            rule = (spec.get("constraint") or {}).get("rule") or ""
            if rule not in record["prompt"]["question"] or re.search(r"\d", rule):
                _fail(record, "the rule is not in the question in words")
            gold = ruled
        if sorted(spec["gold_groups"]) != sorted(gold):
            _fail(record, f"gold groups {spec['gold_groups']} != {sorted(gold)}")
        listed = [(a["verb"], a["target"]) for a in answer["actions"]]
        if sorted(self._frontier_key(v, t) for v, t in listed) != sorted(gold):
            _fail(record, "the answer does not list each gold group once")

    def _check_next_action(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        shown = self.demonstrated(step)
        if shown is None:
            _fail(record, "a next-step question with nothing demonstrated after it")
        nxt, action = shown
        if int(spec["reference_step"]) != nxt:
            _fail(record, f"reference step {spec['reference_step']} != {nxt}")
        frame = self.state_at(step)
        if answer["verb"] != action.verb or answer["target"] not in (
                action.target, self.group_of(action.target)):
            _fail(record, "the answer is not the action the log performed")
        if answer["target"] != action.target:
            # a group action: every member still to do is a permitted sibling
            todo = self.todo(frame, action.verb, self.groups[answer["target"]])
            if action.target not in todo or len(todo) < 2 or any(
                    (action.verb, m) not in self.permitted(step) for m in todo):
                _fail(record, "a group answer whose members are not all equivalent")
        # the reference is the answer (a group action where the answer is one); the
        # instance the log names is `demonstrated`
        if tuple(spec["reference"]) != (answer["verb"], answer["target"]):
            _fail(record, "the reference and the answer disagree")
        if tuple(spec.get("demonstrated") or ()) != (action.verb, action.target):
            _fail(record, "the demonstrated instance is not the log's")
        equivalent = _pairs(spec["equivalent_actions"])
        if (answer["verb"], answer["target"]) not in equivalent:
            _fail(record, "the reference is not among the equivalent actions")
        if answer.get("tool") in UNGRADED_TOOLS and "tool" in spec.get("fields", ()):
            _fail(record, "an unknown tool must not be a graded field")
        blocked, permitted = self.blocked(step), self.permitted(step)
        if _pairs(spec["blocked_actions"]) != blocked:
            _fail(record, "the materialised blocked set is not the graph's")
        if _pairs(spec["permitted_upper_bound"]) != permitted:
            _fail(record, "the materialised upper bound is not the graph's")
        options = label.get("options")
        if options is None:
            return
        kinds: dict[str, int] = {}
        for option in options:
            kind = option["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
            targets = self.expand(frame, option["verb"], option["target"])
            if option["target"] in self.groups and (
                    sorted(targets) != sorted(option.get("members") or []) or len(targets) < 2):
                _fail(record, f"option {option['target']} lists other members")
            for target in targets:
                pair = (option["verb"], target)
                if kind == "demonstrated" and (
                        pair[0] != action.verb or (pair[1] != action.target
                                                   and pair not in permitted)):
                    _fail(record, f"option {pair} is not the demonstrated one")
                if kind in ("graph_blocked", "graph_blocked_pixels") and pair not in blocked:
                    _fail(record, f"option {pair} is not blocked")
                if kind == "state_inapplicable" and self.applies(pair[1], pair[0], frame):
                    _fail(record, f"option {pair} applies after all")
                if kind == "permitted_unknown" and (pair not in permitted
                                                    or pair == (action.verb, action.target)):
                    _fail(record, f"option {pair} is not a permitted distractor")
            if kind == "demonstrated" and action.target not in targets:
                _fail(record, "the demonstrated option does not cover the demonstrated part")
        if kinds.get("demonstrated") != 1:
            _fail(record, "an options list needs exactly one demonstrated action")
        wrong = (kinds.get("graph_blocked", 0) + kinds.get("graph_blocked_pixels", 0)
                 + kinds.get("state_inapplicable", 0))
        if wrong < 2:
            _fail(record, "an options list needs at least two certainly-wrong actions")
        shown_options = record["prompt"].get("options") or []
        if len(shown_options) != len(options):
            _fail(record, "the prompt and the label list different options")
        for prompt_option, option in zip(shown_options, options):
            if prompt_option["verb"] != option["verb"]:
                _fail(record, "the prompt options are in another order")
            if "kind" in prompt_option or "target" in prompt_option:
                _fail(record, "the prompt options give away which is which")

    # V10
    def _check_history(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        log = self.log_steps()
        if any(_cls_of(self.instances, a.target) is None
               for s in log for a in self.successful_actions(s)):
            _fail(record, "an exhaustive history on a log with unresolved targets")
        happened = [a for s in log if s <= step for a in self.named(s)]
        frame = self.state_at(step)
        done, started = set(), set()
        for verb, target in {(a.verb, a.target) for a in happened}:
            gid = self.group_of(target)
            if gid is None:
                done.add((verb, target))
            elif verb == "unscrew" and self.captive(target):
                held = self.instances[target].fastens
                if held is None or not self.present(frame, held):
                    done.add((verb, gid))
            elif self.todo(frame, verb, self.groups[gid]):
                started.add((verb, gid))
            else:
                done.add((verb, gid))
        started -= done
        got = _pairs(answer["done"])
        if got != done:
            _fail(record, f"history differs: missing {sorted(done - got)}, "
                          f"extra {sorted(got - done)}")
        if _pairs(answer.get("started") or []) != started:
            _fail(record, f"started {answer.get('started')} != {sorted(started)}")
        total = sum(len(self.named(s)) for s in log)
        remaining = total - len(happened)
        if answer["remaining_actions"] != remaining:
            _fail(record, f"remaining {answer['remaining_actions']} != {remaining}")
        share = 0.0 if not total else len(happened) / total
        want = spec["bins"][min(int(share * 4), 3)]
        if answer["progress_bin"] != want:
            _fail(record, f"bin {answer['progress_bin']} != {want}")

    # V16
    def _check_plan(self, record: dict, spec: dict) -> None:
        label = record["label"]
        answer = self._answer(record)
        frame = {k: dict(v) for k, v in self.state_at(int(label["step"])).items()}
        first_bad, violated, reason = None, [], None
        for i, item in enumerate(spec["plan"]):
            verb, target = item["verb"], item["target"]
            if target in self.groups:
                # a group action: the verb on every member still to do (redesign)
                todo = self.todo(frame, verb, self.groups[target])
                bad = [e for m in todo for e in self.unmet(verb, m, frame)]
                if not todo or bad:
                    first_bad = i
                    violated = sorted({_edge_label(e) for e in bad})
                    reason = "violates_edge" if bad else "not_present"
                    break
                for m in todo:
                    if self.applies(m, verb, frame):
                        self._apply(frame, m, verb)
                continue
            bad = self.unmet(verb, target, frame)
            if bad or not self.applies(target, verb, frame):
                first_bad = i
                # every edge the step breaks: which one "stopped" it has no
                # single answer when a part is both screwed down and plugged in
                violated = sorted(_edge_label(e) for e in bad)
                reason = ("violates_edge" if bad
                          else self._inapplicable_reason(frame, target))
                break
            self._apply(frame, target, verb)
        valid = first_bad is None
        if bool(answer["valid"]) != valid:
            _fail(record, f"valid {answer['valid']} != {valid}")
        if answer.get("first_error_index") != first_bad:
            _fail(record, f"first error {answer.get('first_error_index')} != {first_bad}")
        if valid:
            # valid by the replay above, under the rules (owner 2026-11-04); the plan
            # itself is the log's continuation, told in group actions
            if spec.get("truth_source") != "simulated" or spec.get("validity") != "simulated":
                _fail(record, "a valid plan whose validity is not the replay's")
            wanted = [(i["verb"], i["target"]) for i in spec["plan"]]
            start = self.state_at(int(label["step"]))
            # the log read step by step until the grouped plan is long enough -- the export
            # tells no more of it than that (a captive group further on is not its business)
            logged: list = []
            for s in self.log_steps():
                if s <= int(label["step"]):
                    continue
                logged.extend((a.verb, a.target) for a in self.planning_actions(s))
                if len(self.grouped(start, logged)) >= len(wanted):
                    break
            told = self.with_captive(start, self.grouped(start, logged))
            if told[:len(wanted)] != wanted:
                _fail(record, "a valid plan that the log never carried out")
            if answer.get("violated_edges"):
                _fail(record, "a valid plan that names a violated edge")
            if answer.get("reason") is not None:
                _fail(record, "a valid plan that gives a reason for failing")
            return
        if sorted(answer.get("violated_edges") or []) != violated:
            _fail(record, f"violated edges {answer.get('violated_edges')} != {violated}")
        # the code is label side (answer_check.reason_code); answer.reason is the words a person
        # reads (owner 2026-11-04): about the first step that breaks, never a code
        if spec.get("reason_code") != reason:
            _fail(record, f"reason_code {spec.get('reason_code')!r} != {reason!r}")
        text = answer.get("reason")
        if not isinstance(text, str) or not text.startswith(f"step {first_bad + 1} (") \
                or "_" in text or "violates" in text:
            _fail(record, f"reason {text!r} is not the words of step {first_bad + 1}")
        if violated and answer.get("violated_edge") != violated[0]:
            _fail(record, "violated_edge is not the first of violated_edges")
        if not violated and answer.get("violated_edge") is not None:
            _fail(record, "an edge named where the state, not the graph, said no")
        family = spec.get("corruption")
        if family not in ("swap", "move", "substitute"):
            _fail(record, f"unknown corruption {family!r}")
        if family != "substitute" and not violated:
            # a rearrangement that violates nothing is a different plan of
            # unknown validity; only a substitute may fail on the state alone
            _fail(record, "a rearrangement called wrong without a violated edge")

    # V7 (W3b)
    def _check_goal_plan(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        goal = spec["goal"]
        frame = {k: dict(v) for k, v in self.state_at(step).items()}
        if frame[goal]["state"] == REMOVED:
            _fail(record, "a plan for a goal that is already out")
        # every part the export knows (fan screw holes are a detection category, not parts)
        if {k: v["state"] for k, v in frame.items()
            if k in spec["start_state"] or self.instances.get(k) is None
            or self.instances[k].cls != "screw_hole"} != {
                k: v for k, v in spec["start_state"].items() if k in frame}:
            _fail(record, "the start state is not the folded state of this step")
        # the rules' shortest plan and the answer (that plan in group actions) replay under
        # the graph, the moment's own rules and the question's rule without an unmet edge,
        # take the goal out and contain every required state change (owner 2026-11-04: the
        # required steps are the rules', never the demonstrated sequence's)
        self.extra = [*self.extra, *self.constraint_edges(spec)]
        for name in ("reference_shortest", "answer"):
            sim = {k: dict(v) for k, v in frame.items()}
            items = (spec[name] if name != "answer"
                     else record["label"]["answer"]["plan"])
            for item in items:
                verb, target = item["verb"], item["target"]
                for member in (self.todo(sim, verb, self.groups[target])
                               if target in self.groups else [target]):
                    if self.unmet(verb, member, sim) or (
                            not self.applies(member, verb, sim) and not self.captive(member)):
                        _fail(record, f"{name} breaks at {verb} {member}")
                    if self.applies(member, verb, sim):
                        self._apply(sim, member, verb)
            if sim[goal]["state"] != REMOVED:
                _fail(record, f"{name} does not reach the goal")
            for req in spec["required_actions"]:
                if req["target"] in self.groups:
                    # captive screws: one item, every member, while their part is in
                    if sorted(req.get("members") or []) != sorted(self.groups[req["target"]]):
                        _fail(record, f"group item {req['target']} lists other members")
                    targets = self.groups[req["target"]]
                else:
                    targets = [req["target"]]
                for target in targets:
                    got = sim.get(target, {}).get("state")
                    if got not in set(req["states"]) and got != REMOVED:
                        _fail(record, f"{name} misses required {target}")
            for item in spec["required_actions"]:
                if self.captive(item["target"]) and self.group_of(item["target"]) \
                        and self.present(frame, self.instances[item["target"]].fastens or ""):
                    _fail(record, "a captive screw required on its own while its part is in")
        # the demonstrated remainder is the log after this step, up to the goal: a
        # reference only
        logged = [(a.verb, a.target) for s in self.log_steps()
                  if step < s <= int(spec["goal_removed_at"])
                  for a in self.planning_actions(s)]
        if [(i["verb"], i["target"]) for i in spec["demonstrated_remainder"]] != logged:
            _fail(record, "the demonstrated remainder is not the logged suffix")
        start = self.state_at(step)
        keep = {"connector"} if (spec.get("constraint") or {}).get("kind") == "until_connectors" \
            else set()
        shortest = [(i["verb"], i["target"]) for i in spec["reference_shortest"]]
        told = self.with_captive(start, self.grouped(start, shortest, keep))
        if [(i["verb"], i["target"]) for i in record["label"]["answer"]["plan"]] != told:
            _fail(record, "V7's answer is not the rules' shortest plan in group actions")
        count = spec.get("count")
        if count:
            plugged = sorted(k for k, rec in self.instances.items() if rec.cls == "connector"
                             and not rec.attrs.get("implied")
                             and start.get(k, {}).get("state") == "plugged")
            if sorted(count["members"]) != plugged or int(count["present"]) != len(plugged):
                _fail(record, f"the plug count {count['present']} is not {len(plugged)}")
            rule = (spec.get("constraint") or {}).get("rule") or ""
            if re.search(r"\d", rule) or rule not in record["prompt"]["question"]:
                _fail(record, "the rule is not in the question in words")
        if spec.get("redundancy_penalty") is not False or not spec.get("any_legal_plan_scores"):
            _fail(record, "V7 must score any legal plan and not penalise redundant steps")
        if int(spec["plan_length"]) != len(spec["reference_shortest"]):
            _fail(record, "plan_length disagrees with the shortest plan")

    def _apply(self, frame: dict[str, dict], target: str, verb: str) -> None:
        effect = self._effect(target, verb)
        if effect is None or effect[0] != "state":
            return
        if target not in frame:  # a `cable:*` node no event has named yet
            frame[target] = {"state": self.tax.default_state("cable"),
                             "placement": IN_CHASSIS, "left_with": None}
        frame[target]["state"] = effect[1]
        if effect[1] != REMOVED:
            return
        rec = self.instances.get(target)
        if rec is not None and rec.cls == "connector":
            return
        for key, child in self.instances.items():
            if child.attached and child.parent == target:
                frame[key]["state"] = REMOVED


# --------------------------------------------------------------------------- #
# the prompt-side leak scan
# --------------------------------------------------------------------------- #
#: Words that only ever appear on the label side of a record.
LEAK_WORDS = ("feasible", "blocker", "must_include", "must_not_include",
              "permitted_upper_bound", "graph_version", "verified", "truth_source",
              "answer_check", "occluded_full", "out_of_view", "in_chassis",
              "on_bench", "graph_blocked", "demonstrated", LS_PREFIX)
#: A frame's position in the teardown, which several tasks ask the model to
#: infer, and the machine it belongs to, which tells it which teardown to recall.
STEP_PATTERNS = (re.compile(r"\bs\d{3}\b"), re.compile(r"\bstep\s+\d+", re.I),
                 re.compile(r"\bD\d+\b"),
                 re.compile(r"\b(scan|oak1|oak2|rs)\b", re.I))
#: An instance key -- `psu.01`, `screw.motherboard.03` -- rather than its wording.
INSTANCE_KEY = re.compile(r"\b[a-z_]+(?:\.[a-z_0-9]+)*\.\d{2,}\b")


def prompt_strings(record: dict) -> list[str]:
    """Every string a model would be shown -- and the id a harness would print.

    The record id is not on the prompt side, but every harness logs it, and
    ``V4-D07-scan-s001-demonstrated-remove-psu.01`` would hand over the desktop,
    the view, the step and the answer's own target. It is hashed for that
    reason, and it is scanned here for the same one.
    """
    out = [str(record.get("id") or ""), str(record["prompt"].get("question") or "")]
    out.extend(str(i) for i in record["prompt"].get("images") or [])
    for option in record["prompt"].get("options") or []:
        out.extend(str(v) for v in option.values())
    return out


def scan_prompt_leaks(records: Iterable[dict]) -> list[str]:
    """One line per prompt that gives away part of its own -- or a sibling's -- label.

    Three families, and every one of them was reachable before the prompt/label
    split: a path or an id that spells the step, the desktop or the view; a
    label-side word; and a raw instance key (which carries an ordinal, and an
    ordinal is most of the answer to "how many of these are there?").
    """
    problems: list[str] = []
    for record in records:
        answer = json_dumps(record["label"]["answer"])
        named = record["label"].get("readable_id") or record.get("id")
        for text in prompt_strings(record):
            for pattern in STEP_PATTERNS:
                if pattern.search(text):
                    problems.append(f"{named}: prompt names a step/desktop/view: "
                                    f"{text[:80]!r}")
            for word in LEAK_WORDS:
                if word in text:
                    problems.append(f"{named}: prompt uses the label word {word!r}")
            for hit in INSTANCE_KEY.findall(text):
                # unconditionally: an ordinal in a key bounds how many of that
                # class exist, which answers a V2 count on the same frame, and
                # a key names a part in a vocabulary no picture teaches
                problems.append(f"{named}: prompt names the instance key {hit!r}")
    return problems


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _pairs(items: Iterable[dict]) -> set[tuple[str, str]]:
    return {(i["verb"], i["target"]) for i in items}


def _cls_of(instances: dict, target: str) -> Optional[str]:
    if target.startswith(CABLE_PREFIX):
        return "cable"
    rec = instances.get(target)
    return None if rec is None else rec.cls


def _edge_label(edge: dict) -> str:
    return f"{edge['type']}({edge['target']}, {edge['blocker']})"


def _row_box(row: dict) -> Optional[list]:
    """``[x, y, w, h]`` of a compiled row, measured without the export's helpers."""
    if row.get("geom_type") == "box" and row.get("box"):
        x0, y0, x1, y1 = row["box"]
        return [x0, y0, x1 - x0, y1 - y0]
    rle = row.get("visible_rle")
    if not rle:
        return None
    import numpy as np
    from pycocotools import mask as mask_utils

    decoded = np.asarray(mask_utils.decode(rle), dtype=bool)
    ys, xs = np.nonzero(decoded)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _iou(a: Optional[Any], b: Optional[Any]) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, aw, ah = (float(v) for v in a)
    bx0, by0, bw, bh = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / (aw * ah + bw * bh - inter)
