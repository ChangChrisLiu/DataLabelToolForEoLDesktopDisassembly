"""Grading a model's answer against one exported VLM record (W3b).

The export (:mod:`tda.core.export.vlm`) ships every record with an
``answer_check`` block saying how the answer is re-derived. This module is the
other half: given the record, the label-side world of its desktop (the
``<name>.oracle.jsonl`` sidecar) and a model's raw text, it parses the text
and says how right it is. ``scripts/vlm_score.py`` loops it over a file and
aggregates; ``scripts/vlm_to_chat.py`` reads :data:`RESPONSE_FORMATS` to tell
the model what shape to answer in; ``scripts/vlm_references.py`` answers in
those shapes too.

Two parsers, both reported (spec 8.2, minutes 2026-10-31 §3):

* **strict** -- the whole reply, fences stripped, is one JSON object;
* **lenient** -- the first balanced ``{...}`` in the reply, Python-literal
  spellings tolerated, and failing that a per-task reading of the prose
  ("yes", "B", "3").

A reply neither parser can read scores 0 and is counted as a parse failure.

Grading follows the graph's three certainties (:mod:`tda.core.export.vlm_tasks`):
what the graph forbids is wrong, what the log demonstrates is right, and what
the graph merely permits is unknown. So V6's primary match is **group
equivalent** (the demonstrated action or a permitted sibling: same unordered
group, or same verb and match class), V6's ``blocked`` counts only frames
whose ``blocked_rate_defined`` is true, and V7 is graded on (a) no step
violating a required edge, (b) recall of the actions every legal plan must
contain and (c) the goal reached -- with **no penalty** for extra steps.

Part groups (the question redesign, spec 3.1 / 6): an oracle line may carry
``groups`` (``{gid: {members, match_class, anchor, label}}``, ``gid =
f"{match_class}@{anchor}"``), and a gold target may then be a group id. A
model's words are grounded to a group by class + kind (:meth:`Grounder.
ground_group`); a group action is its members' actions done together (V7),
a set of group actions is graded as a set (V4 blockers, V10 done / started),
and where two groups share a class the count of answers of that class is what
is compared. V3 accepts the part acted on alone (or a member of its group), never a
part attached to it; V8's ``components`` form is
V1's grade restricted to ``class_filter``. A world without groups grades
exactly as before.

Everything here is pure; the taxonomy is the only thing read from disk.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from tda.core.graph import Edge, applicable_preconditions, unmet
from tda.core.graph_rules import CABLE_PREFIX, verb_applies, verb_effect
from tda.core.model import InstanceRec
from tda.core.states import InstState
from tda.core.taxonomy import Taxonomy

__all__ = ["CAPABILITY_LAYERS", "RESPONSE_FORMATS", "VERBS", "Grade", "Grounder",
           "World", "format_instruction", "gold_reply", "answer_keys", "grade", "grounder",
           "parse_lenient", "parse_strict", "reference_option"]

#: The paper's capability layers (minutes 2026-10-31 §3; brief W3b §2.8). Not
#: spec 8.2's L1-L4 of ``label.layer``: these group the tasks the way the
#: results table does.
CAPABILITY_LAYERS = {
    "V1": "L1", "V2": "L1", "V8": "L1",
    "V4": "L2", "V5": "L2",
    "V6": "L3", "V7": "L3", "V16": "L3",
    "V3": "change", "V12": "change", "V10": "change",
    "V15": "cross",
}

#: The verbs a model may use (spec 6.3).
VERBS = ("unscrew", "disconnect", "open", "release", "remove", "displace", "reorient")

#: ``task -> the JSON a model is asked to answer with``. ``{allowed}`` is the
#: class's state vocabulary (V2 / V14), filled by :func:`format_instruction`.
RESPONSE_FORMATS = {
    "V1": 'Answer with JSON only: {"components": [{"class": "<part name>", '
          '"bbox": [x, y, width, height]}, ...]} in pixel coordinates of the image.',
    "V2.state": 'Answer with JSON only: {"state": "<one of: {allowed}>"} (cannot_tell: this '
                'image does not show it well enough to tell).',
    "V2.count": 'Answer with JSON only: {"count": <integer>}.',
    "V3": 'Answer with JSON only: {"verb": "<one of: ' + ", ".join(VERBS) + '>", '
          '"target": "<the part acted on>", "tool": "<tool, or null>"}.',
    # V3's free form (owner 2026-11-04): the action and where the machine changed
    "V3.region": 'Answer with JSON only: {"verb": "<one of: ' + ", ".join(VERBS) + '>", '
                 '"target": "<the part acted on>", "tool": "<tool, or null>", '
                 '"region": [x, y, width, height]} (region: where the machine changed, in '
                 'pixel coordinates of the top-down scanner image, {scanner}).',
    # V3's multi-step form (owner 2026-11-04): every action between the two moments
    "V3.span": 'Answer with JSON only: {"actions": [{"verb": "<one of: ' + ", ".join(VERBS)
               + '>", "target": "<part>"}, ...]} (every action carried out between the two '
               'moments).',
    "V4": 'Answer with JSON only: {"feasible": true or false, "blockers": '
          '["<part that has to be dealt with first>", ...]}.',
    "V5": 'Answer with JSON only: {"actions": [{"verb": "<verb>", "target": '
          '"<part>"}, ...]}.',
    # V6 (owner 2026-11-04): every action possible now, one entry per action; the optional
    # "why" is not graded
    "V6.open": 'Answer with JSON only: {"actions": [{"verb": "<one of: ' + ", ".join(VERBS)
               + '>", "target": "<part>"}, ...], "why": "<optional: one sentence>"}.',
    "V6.options": 'Answer with JSON only: {"choice": "<letter of the option>"}.',
    "V7": 'Answer with JSON only: {"plan": [{"verb": "<verb>", "target": "<part>"}, '
          '...]} in the order you would carry them out.',
    "V8": 'Answer with JSON only: {"bbox": [x, y, width, height]} in pixel '
          'coordinates of the image.',
    "V8.components": 'Answer with JSON only: {"components": [{"bbox": [x, y, width, '
                     'height]}, ...]}, one box per part asked for, in pixel coordinates '
                     'of the image.',
    "V10": 'Answer with JSON only: {"done": [{"verb": "<verb>", "target": "<part>"}, '
           '...], "started": [{"verb": "<verb>", "target": "<parts>"}, ...], '
           '"progress": "<one of: 0-25, 25-50, 50-75, 75-100>"} ("started": actions on '
           'a set of parts begun but not finished; progress: percent of the teardown '
           'finished).',
    "V12": 'Answer with JSON only: {"changed": true or false}.',

    "V15.cross": 'Answer with JSON only: {"state": "<state>", "best_view": '
                 '"<first or second>"}.',
    "V15.moment": 'Answer with JSON only: {"same_moment": true or false}.',
    "V16": 'Answer with JSON only: {"valid": true or false, "first_error_step": '
           '<number of the first step that cannot be carried out, or null>}.',
}

#: Option letters, in the order the options are listed.
LETTERS = "ABCDEFGHIJ"


def format_key(record: dict) -> str:
    """Which :data:`RESPONSE_FORMATS` entry a record is answered in."""
    task = record["prompt"]["task"]
    check = record["label"]["answer_check"]
    if task == "V2":
        return "V2.count" if check.get("derive") == "screw_count" else "V2.state"
    if task == "V3" and check.get("type") == "action_set":
        return "V3.span"
    if task == "V3" and check.get("region"):
        return "V3.region"
    if task == "V6":
        return "V6.options" if record["prompt"].get("options") else "V6.open"
    if task == "V15":
        return "V15.moment" if check.get("derive") == "same_moment" else "V15.cross"
    if task == "V8" and check.get("type") == "components":
        return "V8.components"
    return task


def format_instruction(record: dict) -> str:
    """The answer-format sentence a chat prompt appends to the question."""
    key = format_key(record)
    text = RESPONSE_FORMATS[key]
    if "{scanner}" in text:
        shown = [int(i) + 1 for i in record["label"]["answer_check"].get("region_images") or [0]]
        if len(shown) == 1:
            where = f"image {shown[0]}"
        else:
            where = "images " + ", ".join(str(i) for i in shown[:-1]) + f" and {shown[-1]}" \
                + (": the same camera" if len(shown) == 2 else "")
        text = text.replace("{scanner}", where)
    if "{allowed}" in text:
        allowed = record["label"]["answer_check"].get("allowed") or []
        if not allowed and record["label"]["answer_check"].get("state_space") == "presence":
            allowed = ["present", "removed"]
        if allowed:
            text = text.replace("{allowed}", ", ".join(allowed))
        else:  # a record that lists no state space (built before V14 wrote one)
            text = text.replace("<one of: {allowed}>", "<state>").replace("{allowed}", "state")
    return text


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)


def parse_strict(raw: Optional[str]) -> Optional[dict]:
    """The reply as one JSON object, a surrounding code fence allowed; else ``None``."""
    if raw is None:
        return None
    text = str(raw).strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _candidates(text: str) -> list[str]:
    """Every ``{...}`` substring whose braces balance, outermost first, in order.

    Only double quotes open a string here: an apostrophe in the prose ("It's
    open") is not a quote, and treating it as one swallowed every object after
    it. Braces inside double-quoted strings do not count.
    """
    out: list[str] = []
    stack: list[int] = []
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                out.append(text[start:i + 1])
    return out


def _as_object(chunk: str) -> Optional[dict]:
    """``chunk`` as a dict: JSON, else a Python literal (single quotes, True)."""
    try:
        value = json.loads(chunk)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value
    try:
        value = ast.literal_eval(re.sub(r"\btrue\b", "True",
                                        re.sub(r"\bfalse\b", "False",
                                               re.sub(r"\bnull\b", "None", chunk))))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        value = None
    return value if isinstance(value, dict) else None


def parse_lenient(raw: Optional[str], record: Optional[dict] = None) -> Optional[dict]:
    """A forgiving reading: the **last** well-formed object in the reply, else the prose.

    The last, because a model that thinks aloud quotes the format first
    (``{"changed": true or false}`` is no object) and answers at the end. The
    prose fallback reads only whole words of the answer vocabulary and never
    infers a negation (:func:`_yes_no`).
    """
    strict = parse_strict(raw)
    if strict is not None:
        return strict
    if raw is None:
        return None
    text = str(raw)
    for chunk in reversed(_candidates(text)):
        value = _as_object(chunk)
        if value is not None:
            return value
    if record is None:
        return None
    return _from_prose(text, record)


#: The answer vocabulary of a yes / no question, whole words only.
_YES = frozenset({"yes", "true"})
_NO = frozenset({"no", "false"})


def _yes_no(text: str) -> Optional[bool]:
    """``yes`` / ``true`` or ``no`` / ``false`` as whole words -- one polarity only.

    No negation is inferred: "it cannot be said that it did not change" is not
    read as anything, and a reply that says both "yes" and "no" is unreadable.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    yes, no = bool(words & _YES), bool(words & _NO)
    if yes == no:
        return None
    return yes


def _from_prose(text: str, record: dict) -> Optional[dict]:
    """The last resort of the lenient parser: one field read off the prose."""
    key = format_key(record)
    if key == "V6.options":
        n = len(record["label"].get("options") or [])
        letters = LETTERS[:n] or LETTERS
        # "option B", "(B)", "B." or a bare "B" -- never the pronoun in "I think"
        for pattern in (rf"(?:option|answer|choice)\s*(?:is\s*)?[:\-]?\s*\(?([{letters}])\b",
                        rf"\(([{letters}])\)", rf"^\s*([{letters}])\s*[\).:]?\s*$",
                        rf"^\s*([{letters}])[\).:]\s"):
            hit = re.search(pattern, text, re.I | re.M)
            if hit:
                return {"choice": hit.group(1).upper()}
        return None
    if key in ("V12", "V15.moment", "V4", "V16", "V14"):
        value = _yes_no(text)
        if value is None:
            return None
        field_name = {"V12": "changed", "V15.moment": "same_moment", "V4": "feasible",
                      "V16": "valid", "V14": "answerable"}[key]
        out: dict[str, Any] = {field_name: value}
        if key == "V16" and not value:
            hit = re.search(r"\b(?:step|action|item)?\s*#?(\d{1,2})\b", text)
            out["first_error_step"] = int(hit.group(1)) if hit else None
        return out
    if key == "V2.count":
        hit = re.search(r"\b(\d{1,3})\b", text)
        return {"count": int(hit.group(1))} if hit else None
    if key == "V2.state":
        allowed = record["label"]["answer_check"].get("allowed") or []
        low = text.lower()
        found = [s for s in allowed if re.search(rf"\b{re.escape(s)}\b", low)]
        return {"state": found[0]} if len(found) == 1 else None
    return None


# --------------------------------------------------------------------------- #
# the world a desktop's answers are graded against
# --------------------------------------------------------------------------- #
def _norm(text: Any) -> str:
    text = str(text or "").lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(the|a|an|its|of the)\b", " ", text)
    return " ".join(text.split())


def _singular(word: str) -> str:
    """One word of a part name in the singular: "latches" -> "latch", "screws" -> "screw".

    Applied to both sides of every comparison, so it only has to be
    consistent, not correct English ("chassis" and "glass" are left alone).
    """
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _gnorm(text: Any) -> str:
    """:func:`_norm`, every word in the singular: how a part name is compared."""
    return " ".join(_singular(w) for w in _norm(text).split())


def _class_words(label: str) -> str:
    """An instance label without its ordinal, lower case but for initialisms:
    "Motherboard screw 7" -> "motherboard screw", "CPU 1" -> "CPU"."""
    words = re.sub(r"\s+\d+$", "", str(label)).split(" ")
    if words and words[0][1:2].islower():
        words[0] = words[0][:1].lower() + words[0][1:]
    return " ".join(words)


def _plural(phrase: str) -> str:
    """``"RAM latch"`` -> ``"RAM latches"`` (``vlm_groups.plural``, which is not vendored)."""
    words = phrase.split(" ")
    last, low = words[-1], words[-1].lower()
    if last.isupper():
        words[-1] = last + "s"
    elif low.endswith(("s", "x", "ch", "sh")):
        words[-1] = last + "es"
    elif low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    else:
        words[-1] = last + "s"
    return " ".join(words)


@dataclass
class World:
    """One desktop's oracle line: instances, active edges, log (W3b sidecar).

    ``groups`` is the redesign's table of interchangeable parts (spec 3.1):
    ``{gid: {"members": [keys], "match_class", "anchor", "label"}}``, with
    ``gid = f"{match_class}@{anchor}"``; an old oracle line has none, and then
    every group lookup answers "none" and grading is what it always was. A
    group id stands wherever an instance key may (a gold target): it answers
    :meth:`cls_of` / :meth:`match_class` like its members, and
    :meth:`members` expands it.
    """

    desktop: int
    instances: dict[str, dict]
    edges: list[Edge]
    log: list[dict] = field(default_factory=list)
    groups: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        self._group_of: dict[str, str] = {}
        for gid, group in self.groups.items():
            for member in group.get("members") or []:
                self._group_of.setdefault(member, gid)
        for key, inst in self.instances.items():
            if inst.get("group"):
                self._group_of[key] = inst["group"]

    @classmethod
    def from_oracle(cls, line: dict) -> "World":
        edges = [Edge(type=e["type"], target=e["target"], blocker=e["blocker"],
                      necessity=e.get("necessity", "required"), mode=e.get("mode"),
                      source=e.get("source", "rule"), status=e.get("status", "proposed"))
                 for e in line.get("edges", [])]
        return cls(int(line["desktop"]), dict(line.get("instances", {})), edges,
                   list(line.get("log", [])), dict(line.get("groups") or {}))

    @property
    def grouped(self) -> bool:
        """Whether this world carries the redesign's groups (the new record contract)."""
        return bool(self.groups)

    def is_group(self, target: Optional[str]) -> bool:
        return target is not None and target in self.groups

    def group_of(self, key: str) -> Optional[str]:
        """The group id of an instance, or ``None`` (a single part, or an old world)."""
        return self._group_of.get(key)

    def members(self, target: str) -> list[str]:
        """A group id's member keys; an instance key is its own one member."""
        if target in self.groups:
            return list(self.groups[target].get("members") or [])
        return [target]

    def label_of(self, target: str) -> str:
        """How a person names a target: the group's plural phrase, the instance's label."""
        if target in self.groups:
            return self.groups[target].get("label") or target
        inst = self.instances.get(target)
        return (inst.get("label") or target) if inst else target

    def plain_name(self, target: Optional[str]) -> str:
        """How a person reads a target **without an ordinal** (owner 2026-11-04): a group by its
        label, a part alone of its class on the desktop by its class ("CPU socket lever"), a
        part with an unordered locator by it (``instances[*].name``, written by the export:
        "screw that fastens the PSU"), any other part as "one of the N <class>" -- never
        "CPU 1". An old oracle line without ``name`` gets the class / "one of" reading."""
        if target is None:
            return "-"
        if target in self.groups:
            return self.groups[target].get("label") or target
        if target.startswith(CABLE_PREFIX):
            owner = ".".join(p for p in target[len(CABLE_PREFIX):].split(".") if not p.isdigit())
            return f"{owner.replace('_', ' ')} cable".strip()
        inst = self.instances.get(target)
        if inst is None:
            return target
        if inst.get("name"):
            return str(inst["name"])
        base = _class_words(inst.get("label") or target)
        mclass = inst.get("match_class") or inst.get("class")
        peers = sum(1 for i in self.instances.values()
                    if (i.get("match_class") or i.get("class")) == mclass)
        return base if peers <= 1 else f"one of the {peers} {_plural(base)}"

    def rec(self, key: str) -> Optional[InstanceRec]:
        inst = self.instances.get(key)
        if inst is None:
            return None
        return InstanceRec(key=key, desktop=self.desktop, cls=inst["class"],
                           attrs=dict(inst.get("attrs") or {}), parent=inst.get("parent"),
                           attached=bool(inst.get("attached")))

    def cls_of(self, key: str) -> Optional[str]:
        if key.startswith(CABLE_PREFIX):
            return "cable"
        if key in self.groups:
            members = self.members(key)
            return self.cls_of(members[0]) if members else None
        inst = self.instances.get(key)
        return inst["class"] if inst else None

    def match_class(self, key: str) -> Optional[str]:
        if key.startswith(CABLE_PREFIX):
            return "cable"
        if key in self.groups:
            given = self.groups[key].get("match_class")
            if given:
                return given
            members = self.members(key)
            return self.match_class(members[0]) if members else None
        inst = self.instances.get(key)
        return inst.get("match_class") if inst else None


class Grounder:
    """Model wording -> an instance key or group id, or at least a match class.

    An exact instance label ("CPU cooler screw 3") or key names one instance;
    a class phrase ("a CPU cooler screw", "the cooler screws") names a match
    class, and a part or group of it is chosen only when the caller says which
    ones are eligible -- the group-equivalent reading of an answer that did not
    say which screw (:meth:`ground_in` for instances, :meth:`ground_group`
    for the redesign's groups). On a world with groups, words are compared in
    the singular, so a group's plural label and a member's singular name are
    the same class; a world without groups keeps the old wording rule
    (:func:`_norm`), so old records grade exactly as before.
    """

    def __init__(self, world: World):
        self.world = world
        self.norm = _gnorm if world.grouped else _norm
        self.by_label: dict[str, str] = {}
        self.phrases: list[tuple[str, str]] = []  # (phrase, match_class), longest first
        seen: set[tuple[str, str]] = set()

        def phrase_of(label: str, mclass: str) -> None:
            phrase = re.sub(r"\s+\d+$", "", label)
            item = (phrase, mclass)
            if phrase and item not in seen:
                seen.add(item)
                self.phrases.append(item)

        for key, inst in sorted(world.instances.items()):
            label = self.norm(inst.get("label") or key)
            self.by_label.setdefault(label, key)
            self.by_label.setdefault(self.norm(key), key)
            phrase_of(label, inst.get("match_class") or inst["class"])
        # the plain name a person reads (no ordinal: "fan connector of the fan CPU cooler",
        # "screw that fastens the PSU") names its part exactly where no other part has it
        names: dict[str, list[str]] = {}
        for key, inst in sorted(world.instances.items()):
            if inst.get("name"):
                names.setdefault(self.norm(inst["name"]), []).append(key)
        # a group's plural label names the group ("PSU cable connectors"), even where its
        # singular reads like a part of another kind ("PSU cable connector")
        labels: dict[str, list[str]] = {}
        for gid, group in sorted(world.groups.items()):
            if group.get("label"):
                labels.setdefault(self.norm(group["label"]), []).append(gid)
        self.group_by_label = {k: v[0] for k, v in labels.items() if len(v) == 1}
        bare = {phrase for phrase, _ in self.phrases}
        for name, keys in names.items():
            # a bare class phrase is a class, not a name ("fan connector")
            if len(keys) == 1 and not name.startswith("one ") and name not in self.by_label \
                    and name not in bare:
                self.by_label[name] = keys[0]
        for gid in sorted(world.groups):
            mclass = world.match_class(gid)
            if mclass and world.groups[gid].get("label"):
                phrase_of(self.norm(world.groups[gid]["label"]), mclass)
        self.phrases.sort(key=lambda p: (-len(p[0]), p[0]))

    def ground(self, text: Any) -> tuple[Optional[str], Optional[str]]:
        """``(instance key or None, match class or None)``; a raw group id is its own key."""
        if text is None:
            return None, None
        raw = str(text).strip()
        if raw in self.world.instances or raw.startswith(CABLE_PREFIX) or raw in self.world.groups:
            return raw, self.world.match_class(raw)
        norm = self.norm(raw)
        plain = _norm(raw)
        if plain.endswith("s") and plain != norm and norm in getattr(self, "group_by_label", {}):
            gid = self.group_by_label[norm]
            return gid, self.world.match_class(gid)
        if norm in self.by_label:
            key = self.by_label[norm]
            return key, self.world.match_class(key)
        # a class phrase: the label wording that shares the most words with the
        # answer, one of the two containing the other ("CPU cooler" is the
        # "heatsink fan CPU cooler", not the "CPU"); then the same head noun
        # (not the "CPU cooler screw"); then the fewest extra words
        ordered = re.sub(r"\b\d+\b", " ", norm).split()
        words = set(ordered)
        head = ordered[-1] if ordered else ""
        best, best_key = None, None
        for phrase, mclass in self.phrases:
            have = set(phrase.split())
            common = words & have
            if not common or not (words <= have or have <= words):
                continue
            rank = (len(common), phrase.split()[-1] == head, -len(have ^ words))
            if best_key is None or rank > best_key:
                best, best_key = mclass, rank
        return None, best

    def ground_in(self, text: Any, eligible: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
        """:meth:`ground`, resolving a class phrase to the first eligible member."""
        key, mclass = self.ground(text)
        if key is not None or mclass is None:
            return key, mclass
        for candidate in eligible:
            if self.world.match_class(candidate) == mclass:
                return candidate, mclass
        return None, mclass

    def resolve(self, text: Any, eligible: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
        """``(target, match class)`` -- :meth:`ground_group` with the class kept."""
        key, mclass = self.ground(text)
        if key is not None and self.world.is_group(key) and key not in eligible \
                and not any(m in eligible for m in self.world.members(key)):
            key = None  # a group's label read as its class where the group is not asked
        if key is not None and self.world.is_group(key) and key not in eligible:
            first = next(m for m in self.world.members(key) if m in eligible)
            return first, mclass
        if key is not None:
            if key in eligible:
                return key, mclass
            gid = self.world.group_of(key)
            if gid is not None and gid in eligible:
                # a member named exactly stands for its group where the group is asked
                return gid, mclass
            return key, mclass
        if mclass is None:
            return None, None
        return self.closest(text, [c for c in eligible
                                   if self.world.match_class(c) == mclass]), mclass

    def closest(self, text: Any, candidates: Sequence[str]) -> Optional[str]:
        """The candidate whose wording shares the most words with ``text``, the first on a tie.

        How a class phrase picks among the eligible targets of its class: "the PSU
        cable connectors" is the group anchored on the PSU cable, not the SSD's; words
        that tell no candidate apart ("the RAM latches" of two modules) leave the order
        of ``candidates`` -- the gold order -- to decide.
        """
        if not candidates:
            return None

        def words(t: Any) -> set:
            return set(re.sub(r"\b\d+\b", " ", self.norm(t)).split())

        said = words(text)
        # "one of the PSU cable connectors" names one member of that group (a plain name,
        # owner 2026-11-04): its parts by their group's words, the group itself left aside
        one = self.norm(text).startswith("one ")
        members = [c for c in candidates if not self.world.is_group(c)]
        if one and members:
            candidates = members

        def wording(c: str) -> str:
            gid = self.world.group_of(c) if one else None
            return self.world.label_of(gid) if gid else self.world.label_of(c)

        best, best_n = candidates[0], -1
        for candidate in candidates:
            n = len(said & words(wording(candidate)))
            if n > best_n:
                best, best_n = candidate, n
        return best

    def ground_group(self, text: Any, eligible: Sequence[str] = ()) -> Optional[str]:
        """Model wording -> a group id or an instance key (spec 3.1, 6.2).

        An exact key, label or group id names itself -- a member of a group
        that is among ``eligible`` stands for that group. A class phrase, in
        the singular or the plural, is the **first** eligible target (group id
        or key) of its match class: answers are grounded to the class + kind,
        never to an anchor, so where two groups share a class the caller
        hands the ones still unclaimed and the count of group actions per
        class is what gets compared. ``None`` when nothing eligible has the
        class (or the words name no class). On a world without groups this is
        exactly :meth:`ground_in`'s instance.
        """
        return self.resolve(text, eligible)[0]


_GROUNDERS: dict[int, tuple[World, Grounder]] = {}


def grounder(world: World) -> Grounder:
    """One :class:`Grounder` per world, built once (a scorer grades 100 k records)."""
    hit = _GROUNDERS.get(id(world))
    if hit is None or hit[0] is not world:
        hit = (world, Grounder(world))
        _GROUNDERS[id(world)] = hit
    return hit[1]


# --------------------------------------------------------------------------- #
# grades
# --------------------------------------------------------------------------- #
@dataclass
class Grade:
    """One record's grade: ``score`` in [0, 1] is the task's primary metric."""

    parsed: bool
    score: float
    metrics: dict = field(default_factory=dict)


def answer_keys(record: dict) -> dict:
    """The answer a grader reads: with instance keys and group ids
    (``answer_check.answer_keys``, the final round of 2026-11-04) where the record keeps
    them apart from ``label.answer`` -- which names every part plainly, no ordinal, for the
    people and the gold reply -- else ``label.answer`` itself (an older record)."""
    label = record["label"]
    return (label.get("answer_check") or {}).get("answer_keys") or label["answer"]


def grade(record: dict, raw: Optional[str], world: Optional[World] = None,
          tax: Optional[Taxonomy] = None, *, mode: str = "strict",
          transform: Optional[dict] = None) -> Grade:
    """Grade one reply. ``mode`` is ``strict`` or ``lenient``.

    ``world`` is needed by the tasks that name parts (V3 V4 V5 V6 V7 V10) and
    ``tax`` by V7's simulation; ``transform`` (``{"crop": [x0, y0],
    "scale": s}``) maps a box answered on a cropped / resized image back.
    """
    answer = (parse_strict(raw) if mode == "strict"
              else parse_lenient(raw, record))
    if answer is None:
        return Grade(False, 0.0, {"parse_failure": True})
    task = record["prompt"]["task"]
    handler = _GRADERS.get(task)
    if handler is None:
        return Grade(True, 0.0, {"ungraded": True})
    try:
        return handler(record, answer, world, tax, transform)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        # a reply of the right shape with the wrong types inside
        return Grade(True, 0.0, {"malformed": type(exc).__name__})


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
    return None


def _box(value: Any, transform: Optional[dict]) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    x, y, w, h = (float(v) for v in value)
    if transform:
        scale = float(transform.get("scale") or 1.0)
        x0, y0 = (transform.get("crop") or [0, 0])[:2]
        x, y, w, h = x / scale + x0, y / scale + y0, w / scale, h / scale
    return [x, y, w, h]


def _iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _f1(tp: int, n_pred: int, n_true: int) -> float:
    if n_pred == 0 and n_true == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return 2 * p * r / (p + r)


#: Longer names a class also goes by: the questions call an optical drive that never leaves its bracket
#: "Optical drive cage N", which contains "optical drive" -- the longest match must still be the cage.
CLASS_ALIASES = {"drive_cage": ("optical drive cage",)}


def _class_of_text(text: Any, tax: Optional[Taxonomy], classes: Sequence[str]) -> Optional[str]:
    norm = _norm(text)
    best, best_len = None, -1
    for cls in classes:
        phrase = _norm(cls)
        if tax is not None:
            from tda.core.export.vlm_tasks import class_label

            phrase = _norm(class_label(cls, tax))
        for p in (phrase, *(_norm(a) for a in CLASS_ALIASES.get(cls, ()))):
            if (norm == p or norm == _norm(cls) or re.search(rf"\b{re.escape(p)}\b", norm)) and len(p) > best_len:
                best, best_len = cls, len(p)
    return best


def _g_v1(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)["components"]
    classes = sorted({c["class"] for c in truth})
    preds = []
    for item in answer.get("components") or []:
        if not isinstance(item, dict):
            continue
        preds.append((_class_of_text(item.get("class"), tax, classes),
                      _box(item.get("bbox"), transform)))
    return _box_set_grade(truth, preds,
                          float(record["label"]["answer_check"].get("iou_threshold", 0.5)))


#: V2's count form (``derive == "screw_count"``, owner 2026-11-04): the score of an answer
#: ``|answer - gold|`` away from the gold count; any other distance scores 0. The exact match
#: is reported beside it (``exact``).
COUNT_SCORES = {0: 1.0, 1: 0.8}


def count_score(answered: Any, gold: int) -> tuple[float, Optional[int]]:
    """``(score, |answered - gold|)`` of a V2 count answer (:data:`COUNT_SCORES`); an answer
    that is no integer scores 0 at distance ``None``."""
    try:
        off = abs(int(answered) - int(gold))
    except (TypeError, ValueError):
        return 0.0, None
    return COUNT_SCORES.get(off, 0.0), off


def _g_v2(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)
    prior = record["label"].get("prior_is_correct")
    if "count" in truth:
        # exact 1.0, off by one 0.8, else 0 (owner 2026-11-04)
        score, off = count_score(answer.get("count"), truth["count"])
        exact = off == 0
        return Grade(True, score, {"correct": exact, "exact": exact, "off_by": off,
                                   "prior_is_correct": prior})
    ok = _norm(answer.get("state")) == _norm(truth["state"])
    # cannot_tell is graded like any other state (V14 merged into V2, owner 2026-11-04)
    return Grade(True, float(ok), {"correct": ok, "exact": ok, "prior_is_correct": prior,
                                   "cannot_tell": _norm(truth["state"]) == "cannot tell",
                                   "abstained": _norm(answer.get("state")) == "cannot tell"})


def _same_target(world: Optional[World], got: Optional[str], want: Optional[str]) -> bool:
    """``got`` names ``want``: the same key / group, or a member of the group asked."""
    if got is None or want is None:
        return False
    if got == want:
        return True
    if world is None:
        return False
    return world.group_of(got) == want if world.is_group(want) else world.group_of(want) == got


def _v3_slots(answer, slots, check, world, group) -> tuple[float, dict]:
    """One V3 slot set (verb, target class, target instance, tool) against the answer."""
    verb_ok = _norm(answer.get("verb")) == _norm(slots["verb"])
    target = slots.get("target_instance")
    # the part acted on is the answer and nothing else (owner 2026-11-03): a part
    # attached to it (a RAM latch on the motherboard, a screw on the cooler) is not,
    # whatever an older record's `accept_targets` lists beside the target; a group
    # target is named by its members' class, and a member of the group a target
    # belongs to (`target_group`) stands for it -- "RAM latch 2" for RAM latch 1
    accept = [t for t in (target, group) if t]
    if world is None:
        key, mclass = None, None
    elif check.get("accept_targets") or world.grouped:
        key, mclass = grounder(world).resolve(answer.get("target"), accept)
    else:
        key, mclass = grounder(world).ground(answer.get("target"))
    cls = world.cls_of(key) if (world and key) else (mclass.split(".")[0] if mclass else None)
    class_ok = cls == slots.get("target_class")
    inst_ok = _same_target(world, key, target) or (group is not None and key == group)
    metrics = {"verb": verb_ok, "target_class": class_ok, "target_instance": inst_ok}
    if slots.get("tool") is not None:
        metrics["tool"] = _norm(answer.get("tool")) == _norm(slots["tool"])
    return float(verb_ok and class_ok), metrics


def _g_v3(record, answer, world, tax, transform) -> Grade:
    if record["label"]["answer_check"].get("type") == "action_set":
        # the multi-step form: a set of actions, F1 over the same-kind groups
        return _g_v6_frontier(record, answer, world, tax, transform)
    return _g_v3_step(record, answer, world, tax, transform)


def _g_v3_step(record, answer, world, tax, transform) -> Grade:
    """Any action of the step, read as the first one is, scores (controller 2026-11-03).

    A compound step (``answer.actions``: a PSU displaced and a connector pulled in
    one step) was carried out in every one of its actions, so the answer is graded
    against each slot set and the best one counts; the top-level slots are the
    first action and grade alone where there is no list.
    """
    truth = answer_keys(record)
    check = record["label"].get("answer_check") or {}

    def group_of(target: Optional[str], given: Optional[str] = None) -> Optional[str]:
        if given:
            return given
        if world is None or not target or not world.grouped or world.is_group(target):
            return None
        return world.group_of(target)

    sets = [(0, truth, group_of(truth.get("target_instance"), check.get("target_group")))]
    for i, slots in enumerate(truth.get("actions") or []):
        if not isinstance(slots, dict) or "verb" not in slots:
            continue
        if i == 0 and all(slots.get(f) == truth.get(f) for f in
                          ("verb", "target_class", "target_instance", "tool")):
            continue  # the first action is the top-level slots
        sets.append((i, slots, group_of(slots.get("target_instance"))))
    best_rank, best, best_index = None, {}, 0
    for i, slots, group in sets:
        score, metrics = _v3_slots(answer, slots, check, world, group)
        rank = (score, metrics["target_instance"])
        if best_rank is None or rank > best_rank:  # the first one on a tie
            best_rank, best, best_index = rank, metrics, i
    if len(sets) > 1:
        best = {**best, "matched_action": best_index}
    if check.get("region"):
        # the free form (owner 2026-11-04): where the machine changed, graded beside the
        # action and never in its score -- IoU with the gold region, a hit at the threshold;
        # a box on a cropped / scaled picture is mapped back like V1 / V8's
        iou = _iou(_box(answer.get("region"), transform), check["region"])
        best = {**best, "region_iou": iou,
                "region_hit": iou >= float(check.get("region_iou_threshold", 0.5))}
    return Grade(True, best_rank[0], best)


def _g_v4(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)
    feasible = _as_bool(answer.get("feasible"))
    ok = feasible is not None and feasible == bool(truth["feasible"])
    metrics: dict = {"correct": ok, "truth_source": record["label"].get("truth_source")}
    if not truth["feasible"] and world is not None:
        metrics["blockers_f1"] = _blockers_f1(answer.get("blockers"), truth["blockers"], world)
    return Grade(True, float(ok), metrics)


def _blockers_f1(answered: Any, blockers: Sequence[str], world: World) -> float:
    """Set F1 of the answered blockers against the gold ones (keys or group ids).

    An old world (no groups) keeps the old reading: each answer is grounded on
    the whole gold list, and a class nothing in it has is dropped. With groups
    (spec 3.1, 6.2) a class phrase claims the first gold target of its class not
    yet claimed -- "the RAM latches" twice is the latches of two modules -- a
    member named exactly is its group, and a part that is no blocker, by class
    or by name, is a false positive; words that name no part are still dropped.
    """
    g = grounder(world)
    want = set(blockers)
    if not world.grouped:
        got = {g.ground_in(b, list(blockers))[0] for b in answered or []}
        got.discard(None)
        return _f1(len(got & want), len(got), len(want))
    left = list(blockers)
    classes = {world.match_class(b) for b in blockers}
    got: set = set()
    wrong = 0
    for text in answered or []:
        key, mclass = g.ground(text)
        if key is not None:
            # an exact name (or a member of a gold group) claims its target, so a
            # class phrase after it grounds to another one
            target = g.resolve(text, list(blockers))[0]
            if target in left:
                left.remove(target)
        else:
            target = g.ground_group(text, left)
            if target is None:
                # a class every gold target of which is claimed is a repeat;
                # any other class is a part that blocks nothing
                wrong += mclass is not None and mclass not in classes
                continue
            left.remove(target)
        got.add(target)
    return _f1(len(got & want), len(got) + wrong, len(want))


def _pairs_grounded(items: Any, world: Optional[World], eligible: Sequence[tuple[str, str]]
                    ) -> list[tuple[str, Optional[str], Optional[str]]]:
    """``[(verb, target, match_class)]`` for a list of ``{verb, target}`` answers.

    A class phrase is resolved to an eligible target of its class (instance
    key or group id), each eligible one used once and in the order given --
    "unscrew a CPU cooler screw" twice is two screws, "open the RAM latches"
    twice is the latches of two modules (spec 6.2: the count of group actions
    per class); among several of its class it takes the one its words fit best
    (:meth:`Grounder.closest`). A member named exactly stands for its group
    where the group is eligible, and on a world with groups an exact name
    spends its target too.
    """
    out = []
    g = grounder(world) if world else None
    left = list(eligible)
    spare = sorted(world.instances) if world else []

    def spend(target: str) -> None:
        for member in world.members(target):
            if member in spare:
                spare.remove(member)

    for item in items or []:
        if not isinstance(item, dict):
            continue
        verb = _norm(item.get("verb")).split(" ")[0] if item.get("verb") else ""
        if g is None:
            out.append((verb, None, None))
            continue
        mine = [t for v, t in left if v == verb]
        key, mclass = g.ground(item.get("target"))
        if key is not None:
            key = g.resolve(item.get("target"), mine)[0]
            if world.grouped and (verb, key) in left:
                # an exact name (or a member standing for its group) spends its
                # target, so a later class phrase grounds to another one; an old
                # world keeps the old reading (exact names are never spent)
                left.remove((verb, key))
        elif mclass is not None:
            key = g.ground_group(item.get("target"), mine)
            if key is not None:
                left.remove((verb, key))
        if key is None and mclass is not None:
            # a class nothing eligible has: the first part of that class (its
            # group, where it has one)
            for t in spare:
                if world.match_class(t) == mclass:
                    key = world.group_of(t) or t
                    break
        if key is not None:
            spend(key)
        out.append((verb, key, mclass))
    return out


def _g_v5(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)
    must = {(a["verb"], a["target"]) for a in truth["must_include"]}
    must_not = {(a["verb"], a["target"]) for a in truth["must_not_include"]}
    eligible = [(a["verb"], a["target"]) for a in truth["permitted_upper_bound"]]
    got = {(v, k) for v, k, _ in _pairs_grounded(answer.get("actions"), world,
                                                  eligible) if k}
    recall = len(got & must) / len(must) if must else 1.0
    blocked = len(got & must_not) / len(got) if got else 0.0
    return Grade(True, recall, {"demonstrated_recall": recall, "blocked_rate": blocked})


def _frontier_index(check: dict, world: World) -> tuple[dict, list[tuple[str, str]]]:
    """``(verb, target) -> group key`` over the base frontier and the blocked groups, and the
    eligible pairs a reply's wording is grounded on (part groups of a group's members
    included, so "unscrew the PSU screws" grounds to the group)."""
    index: dict[tuple[str, str], str] = {}
    for group in [*(check.get("groups") or []), *(check.get("ruled_groups") or []),
                  *(check.get("blocked_groups") or [])]:
        for a in group.get("actions") or []:
            index.setdefault((a["verb"], a["target"]), group["key"])
            gid = world.group_of(a["target"]) if world is not None else None
            if gid is not None:
                index.setdefault((a["verb"], gid), group["key"])
    return index, list(index)


def _g_v6_frontier(record, answer, world, tax, transform) -> Grade:
    """V6 since the owner's 2026-11-04 rulings: the gold is every action possible now, grouped.

    **mc**: the chosen option is right when it is the one inside the frontier. **open** and
    **constrained**: each answered action is grounded (class phrases to a target of their
    class, a member to its group) and mapped to its group; precision = groups hit /
    (groups hit + wrong answers), recall = groups hit / gold groups, F1 the score. An action
    on a blocked, already-done or absent part is a wrong answer; repeats of a group already
    hit cost nothing; ``why`` is not graded. **constrained** also reports whether an answer
    named an action the rule forbids (``constraint_violated``, ``violations``)."""
    label = record["label"]
    check = label["answer_check"]
    options = label.get("options")
    if options:
        choice = str(answer.get("choice") or "").strip().upper()[:1]
        if choice not in LETTERS[:len(options)]:
            return Grade(True, 0.0, {"invalid_choice": True, "blocked": None})
        option = options[LETTERS.index(choice)]
        right = option.get("kind") == "frontier"
        return Grade(True, float(right), {"correct": right, "blocked": not right,
                                          "group_match": right})
    gold = set(check.get("gold_groups") or [])
    forbidden = set(check.get("forbidden_groups") or [])
    index, eligible = _frontier_index(check, world)
    items = _pairs_grounded(answer.get("actions"), world, eligible) if world else []
    hit: set = set()
    wrong = violations = 0
    for verb, key, mclass in items:
        group = index.get((verb, key)) if key is not None else None
        if group is None and key is not None and world is not None and world.is_group(key):
            group = next((index[(verb, m)] for m in world.members(key)
                          if (verb, m) in index), None)
        if group in gold:
            hit.add(group)
            continue
        wrong += 1
        if group in forbidden:
            violations += 1
    n_pred = len(hit) + wrong
    precision = len(hit) / n_pred if n_pred else 0.0
    recall = len(hit) / len(gold) if gold else 1.0
    f1 = _f1(len(hit), n_pred, len(gold))
    metrics = {"precision": precision, "recall": recall, "f1": f1, "groups_hit": len(hit),
               "gold_groups": len(gold), "wrong_actions": wrong,
               "answered_actions": len(items)}
    if check.get("form") == "constrained":
        metrics.update(constraint_violated=violations > 0, violations=violations)
    return Grade(True, f1, metrics)


def _g_v6(record, answer, world, tax, transform) -> Grade:
    label = record["label"]
    check = label["answer_check"]
    if check.get("type") == "frontier":
        return _g_v6_frontier(record, answer, world, tax, transform)
    equivalent = {(a["verb"], a["target"]) for a in check.get("equivalent_actions")
                  or [{"verb": check["reference"][0], "target": check["reference"][1]}]}
    blocked = {(a["verb"], a["target"]) for a in check.get("blocked_actions") or []}
    reference = tuple(check["reference"])
    options = label.get("options")
    if options:
        choice = str(answer.get("choice") or "").strip().upper()[:1]
        if choice not in LETTERS[:len(options)]:
            return Grade(True, 0.0, {"invalid_choice": True, "blocked": None,
                                     "blocked_rate_defined": False})
        option = options[LETTERS.index(choice)]
        pair = (option["verb"], option["target"])
        exact = LETTERS.index(choice) == reference_option(label)
        # a group option (spec 3.1) is right when every member it lists is
        # equivalent -- the demonstrated option of a group reference always is
        group = pair in equivalent or exact or bool(option.get("members")) and all(
            (option["verb"], m) in equivalent for m in option["members"])
        hit_blocked = option["kind"] in ("graph_blocked", "graph_blocked_pixels")
        return Grade(True, float(group), {
            "group_match": group, "exact_match": exact, "blocked": hit_blocked,
            "certainly_wrong": option["kind"] in ("graph_blocked", "graph_blocked_pixels",
                                                  "state_inapplicable"),
            "blocked_rate_defined": bool(check.get("blocked_rate_defined"))})
    verb = _norm(answer.get("verb")).split(" ")[0] if answer.get("verb") else ""
    if world is not None and world.grouped:
        return _g_v6_groups(world, verb, answer.get("target"), reference, equivalent,
                            blocked, check)
    key, mclass = (grounder(world).ground(answer.get("target")) if world
                   else (None, None))
    want_verb, want_class = (check.get("match_class") or [reference[0], None])
    if key is not None:
        pair = (verb, key)
        group = pair in equivalent
        exact = pair == reference
        hit_blocked = pair in blocked
    else:
        group = verb == want_verb and mclass is not None and mclass == want_class
        exact = False
        hit_blocked = False
    return Grade(True, float(group), {
        "group_match": group, "exact_match": exact, "blocked": hit_blocked,
        "blocked_rate_defined": bool(check.get("blocked_rate_defined"))})


def reference_option(label: dict) -> Optional[int]:
    """The index of the V6 option that is the reference action, or ``None``.

    The option ``(verb, target)`` equal to ``answer_check.reference``; failing
    that (a record whose reference is the member the log names while the option
    lists its group), the option of that verb whose ``members`` hold it, then the
    option of kind ``demonstrated``.
    """
    options = label.get("options") or []
    ref = tuple(label["answer_check"].get("reference") or ())
    for i, o in enumerate(options):
        if (o.get("verb"), o.get("target")) == ref:
            return i
    for i, o in enumerate(options):
        if len(ref) == 2 and o.get("verb") == ref[0] and ref[1] in (o.get("members") or ()):
            return i
    for i, o in enumerate(options):
        if o.get("kind") in ("demonstrated", "frontier"):
            return i
    return None


def _in_actions(world: World, verb: str, target: Optional[str], actions: set) -> bool:
    """``(verb, target)`` is one of ``actions``, a group and its members standing for each other."""
    if target is None:
        return False
    if (verb, target) in actions:
        return True
    gid = world.group_of(target)
    if gid is not None and (verb, gid) in actions:
        return True
    return world.is_group(target) and any((verb, m) in actions for m in world.members(target))


def _still_to_do(world: World, verb: str, member: str, gid: Optional[str], equivalent: set,
                 check: dict) -> bool:
    """May ``member``, named exactly, stand for its group ``gid`` in a V6 answer?

    Only while it is still to do: its own action is equivalent, or it is among the
    record's ``part_group_members`` (the members still in the asked situation). A record
    that lists neither -- the group action alone, no member of it -- cannot tell, and
    lets every member stand for it.
    """
    if (verb, member) in equivalent:
        return True
    todo = check.get("part_group_members") or check.get("members")
    if todo:
        return member in todo
    members = world.members(gid) if gid is not None and world.is_group(gid) else []
    return not any((verb, m) in equivalent for m in members)


def _g_v6_groups(world: World, verb: str, text: Any, reference: tuple, equivalent: set,
                 blocked: set, check: dict) -> Grade:
    """V6's open form on a world with groups (spec 3.1, 6.1).

    The answer is grounded on the targets of the equivalent actions of its
    verb, then on the blocked ones -- so "remove the CPU" is the blocked action
    without an ordinal. A group and any of its remaining members are the same
    answer; a member named exactly stands for its group only while it is still to
    do -- its own ``(verb, member)`` is an equivalent action -- so "unscrew
    motherboard screw 1" when screw 1 is already out is no match (review of the
    fix wave). ``exact_match``: the reference itself, named exactly or (a group
    reference) by its class; on a group step any member still to do is exact.
    """
    g = grounder(world)
    eligible = [t for v, t in sorted(equivalent) if v == verb] + \
        [t for v, t in sorted(blocked) if v == verb]
    named, mclass = g.ground(text)
    target = g.resolve(text, eligible)[0]
    if (named is not None and not world.is_group(named) and target != named
            and not _still_to_do(world, verb, named, target, equivalent, check)):
        target = named  # a member already done does not stand for its group
    group = target is not None and (
        (verb, target) in equivalent
        or (world.is_group(target)
            and any((verb, m) in equivalent for m in world.members(target))))
    hit_blocked = not group and _in_actions(world, verb, target, blocked)
    if named is not None:
        exact = group and verb == reference[0] \
            and _same_target(world, target, reference[1]) \
            and (target == reference[1] or world.is_group(reference[1]))
    else:
        exact = (verb == reference[0] and world.is_group(reference[1])
                 and mclass is not None and mclass == world.match_class(reference[1]))
    return Grade(True, float(group), {
        "group_match": group, "exact_match": bool(exact), "blocked": hit_blocked,
        "blocked_rate_defined": bool(check.get("blocked_rate_defined"))})


def _v7_eligible(record: dict, world: World) -> list[tuple[str, str]]:
    """The ``(verb, target)`` pairs a V7 plan's wording is grounded on.

    The gold plan first (its targets are keys or group ids), then the graph's
    shortest plan and the demonstrated remainder. Those two are told member by
    member; on a world with groups a member's action is its group's action
    (spec 3.1: one group action replaces the N member actions in a plan), so a
    class phrase or an exact member name grounds to the group, never to one
    screw of it. Repeats are kept (a group action said twice grounds twice and
    is a no-op the second time). A world without groups grades as before.
    """
    check = record["label"]["answer_check"]
    told = [*(check.get("reference_shortest") or []), *(check.get("demonstrated_remainder") or [])]
    if not world.grouped:
        return [(i["verb"], i["target"]) for i in told]
    pairs = [(i["verb"], i["target"]) for i in answer_keys(record).get("plan") or []]
    for item in told:
        pairs.append((item["verb"], world.group_of(item["target"]) or item["target"]))
    return pairs


def moment_edges_of(check: dict, nodes) -> list[Edge]:
    """The moment's own rules of a record (``answer_check.moment``) as edges: what covers a
    part in this frame (``covered_by``) and the loose parts that come out first -- every
    node of ``nodes`` outside ``loose_scope`` waits for each loose part (``blocked_by``,
    mode ``loose``: every verb, only ``removed`` clears it). The twin of
    :func:`tda.core.graph.moment_edges`, rebuilt from the label alone."""
    moment = check.get("moment") or {}
    out = [Edge(type="covered_by", target=x, blocker=y, source="occlusion", status="accepted")
           for x, ys in sorted((moment.get("occluded") or {}).items()) for y in ys]
    loose = list(moment.get("loose") or [])
    if loose:
        scope = set(moment.get("loose_scope") or [])
        for key in sorted(set(nodes)):
            if key in scope or key in loose:
                continue
            out.extend(Edge(type="blocked_by", target=key, blocker=part, mode="loose",
                            source="loose", status="accepted") for part in loose)
    return out


def constraint_edges_of(check: dict) -> list[Edge]:
    """A constrained question's rule as edges (``answer_check.constraint.edges``)."""
    rule = check.get("constraint") or {}
    return [Edge(type=e["type"], target=e["target"], blocker=e["blocker"],
                 necessity=e.get("necessity", "required"), mode=e.get("mode"),
                 source=e.get("source", "constraint"), status="accepted")
            for e in rule.get("edges") or []]


def _g_v7(record, answer, world, tax, transform) -> Grade:
    check = record["label"]["answer_check"]
    if world is None or tax is None:
        raise ValueError("V7 needs the world and the taxonomy")
    sim: dict[str, InstState] = {k: InstState(state=v, placement="in_chassis")
                                 for k, v in check["start_state"].items()}
    # the rules a plan is replayed under (owner 2026-11-04): the stored graph, this frame's
    # occlusion, loose parts first -- and a constrained question's own rule
    present = [k for k, v in check["start_state"].items() if v != "removed"]
    rule_edges = constraint_edges_of(check)
    edges = [*world.edges, *moment_edges_of(check, present), *rule_edges]
    count = check.get("count") or None
    counted = set((count or {}).get("members") or [])
    disconnected: set = set()
    rule_broken = False
    steps = _pairs_grounded(answer.get("plan"), world, _v7_eligible(record, world))
    violations, first_violation, unresolved, noop = 0, None, 0, 0
    for index, (verb, target, _) in enumerate(steps):
        if target is None:
            unresolved += 1
            continue
        # a group action (spec 3.1) is its members' actions, done together:
        # one step, one violation at most, a no-op only if no member moves
        known = applied = violated = False
        for key in world.members(target):
            rec = world.rec(key)
            cls = rec.cls if rec is not None else ("cable" if key.startswith(CABLE_PREFIX)
                                                   else None)
            if cls is None:
                continue
            known = True
            attrs = rec.attrs if rec is not None else {}
            current = sim[key].state if key in sim else tax.default_state(cls)
            new = verb_effect(tax, cls, attrs, verb)
            if new is None or not verb_applies(tax, cls, attrs, verb, current):
                # no redundancy penalty: an action already done (or one that
                # changes nothing) is a no-op, not an error
                continue
            gating = applicable_preconditions(edges, (verb, key))
            broken = unmet(gating, sim, "required")
            if broken:
                violated = True
                rule_broken = rule_broken or any(e.source == "constraint" for e in broken)
            if count and verb == count.get("verb") and key in counted:
                disconnected.add(key)
            applied = True
            sim[key] = InstState(state=new, placement="in_chassis")
            if new == "removed" and cls != "connector":
                for child, inst in world.instances.items():
                    if inst.get("attached") and inst.get("parent") == key and child in sim:
                        sim[child] = InstState(state="removed", placement="in_chassis")
        if not known:
            unresolved += 1
        elif not applied:
            noop += 1
        elif violated:
            violations += 1
            if first_violation is None:
                first_violation = index

    def reached_one_of(target: str, states: set) -> bool:
        # a group target is reached when every member is
        members = world.members(target)
        for key in members:
            state = sim.get(key)
            got = state.state if state else None
            if not (got in states or got == "removed"):
                return False
        return bool(members)

    required = check["required_actions"]
    met = sum(reached_one_of(req["target"], set(req["states"])) for req in required)
    recall = met / len(required) if required else 1.0
    goal_state = check.get("goal_state", "removed")
    goal_members = world.members(check["goal"])
    reached = bool(goal_members) and all(
        sim.get(k) is not None and sim[k].state == goal_state for k in goal_members)
    success = reached and violations == 0 and recall == 1.0
    metrics = {
        "violation_free": violations == 0, "violations": violations,
        "first_violation": first_violation, "required_recall": recall,
        "goal_reached": reached, "unresolved_steps": unresolved, "noop_steps": noop,
        "plan_length": len(steps), "reference_length": check.get("plan_length"),
        "valid": reached and violations == 0}
    if check.get("form") == "constrained":
        metrics["constraint_kept"] = not rule_broken
    if count:
        # how many plugs the plan unplugs against how many are in (owner 2026-11-04)
        planned, present_n = len(disconnected), int(count.get("present") or 0)
        metrics.update(count_planned=planned, count_present=present_n,
                       count_exact=planned == present_n,
                       count_abs_error=abs(planned - present_n))
    return Grade(True, float(success), metrics)


def _box_set_grade(truth: Sequence[dict], preds: list[tuple[Optional[str], Optional[list]]],
                   thr: float) -> Grade:
    """V1's set F1: each prediction claims the best unclaimed truth box of its class."""
    used, tp = set(), 0
    for cls, box in preds:
        best, best_iou = None, thr
        for i, comp in enumerate(truth):
            if i in used or comp["class"] != cls:
                continue
            iou = _iou(box, comp["bbox"])
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used.add(best)
            tp += 1
    f1 = _f1(tp, len(preds), len(truth))
    return Grade(True, f1, {"f1": f1, "tp": tp, "pred": len(preds), "true": len(truth)})


def _g_v8_components(record, answer, world, tax, transform) -> Grade:
    """V8 on a class with several instances (spec 6.3): "localise every visible
    <class / group>", graded as V1 restricted to ``class_filter``.

    The question names the class, so every answered box is a box of that class
    (a class word in the answer is not read); the truth is the record's
    components of that match class (a component without one is taken as it).
    """
    check = record["label"]["answer_check"]
    truth = [{"class": "asked", "bbox": c["bbox"]} for c in _asked_components(record, world)]
    preds = [("asked", _box(item.get("bbox"), transform))
             for item in answer.get("components") or [] if isinstance(item, dict)]
    return _box_set_grade(truth, preds, float(check.get("iou_threshold", 0.5)))


def _asked_components(record: dict, world: Optional[World]) -> list[dict]:
    """A ``components`` V8's truth boxes of the class asked (``class_filter``)."""
    wanted = record["label"]["answer_check"].get("class_filter")
    out = []
    for comp in answer_keys(record)["components"]:
        mclass = comp.get("match_class")
        if mclass is None and world is not None and comp.get("instance"):
            mclass = world.match_class(comp["instance"])
        if (not wanted or (mclass == wanted if mclass is not None
                           else comp.get("class") in (wanted, wanted.split(".")[0]))):
            out.append(comp)
    return out


def _g_v8(record, answer, world, tax, transform) -> Grade:
    check = record["label"]["answer_check"]
    if check.get("type") == "components":
        return _g_v8_components(record, answer, world, tax, transform)
    truth = answer_keys(record)["bbox"]
    box = _box(answer.get("bbox"), transform)
    if check.get("metric") == "point_in_box":
        point = answer.get("point")
        if point is None and box is not None:
            point = [box[0] + box[2] / 2, box[1] + box[3] / 2]
        elif point is not None and transform:
            scale = float(transform.get("scale") or 1.0)
            x0, y0 = (transform.get("crop") or [0, 0])[:2]
            point = [float(point[0]) / scale + x0, float(point[1]) / scale + y0]
        hit = bool(point) and truth[0] <= float(point[0]) <= truth[0] + truth[2] \
            and truth[1] <= float(point[1]) <= truth[1] + truth[3]
        return Grade(True, float(hit), {"hit": hit})
    iou = _iou(box, truth)
    ok = iou >= float(check.get("iou_threshold", 0.5))
    return Grade(True, float(ok), {"iou": iou, "hit": ok})


def _g_v10(record, answer, world, tax, transform) -> Grade:
    """Set F1 over the done actions; a group partly done is ``started`` (spec 6.1).

    A group action is done only when every member is; the groups begun but not
    finished are their own set (``started_f1``, on records that carry
    ``answer.started``). Each list is grounded on its own set first, then on
    the other one -- a begun group named as done is a false positive there.
    """
    truth = answer_keys(record)
    # the gold lists in their own order: a class phrase grounds to the first
    # unclaimed target of its class, so the gold reply reads back exactly
    done = list(dict.fromkeys((a["verb"], a["target"]) for a in truth["done"]))
    started = list(dict.fromkeys((a["verb"], a["target"])
                                 for a in truth.get("started") or []))
    want, begun = set(done), set(started)
    got = {(v, k) for v, k, _ in _pairs_grounded(answer.get("done"), world,
                                                  done + started) if k}
    f1 = _f1(len(got & want), len(got), len(want))
    progress = answer.get("progress", answer.get("progress_bin"))
    bin_ok = _norm(progress).replace(" ", "-") == _norm(truth["progress_bin"]).replace(" ", "-")
    metrics = {"set_f1": f1, "bin_correct": bin_ok}
    if "started" in truth:
        said = {(v, k) for v, k, _ in _pairs_grounded(answer.get("started"), world,
                                                       started + done) if k}
        metrics["started_f1"] = _f1(len(said & begun), len(said), len(begun))
    return Grade(True, f1, metrics)


def _g_bool(field_name: str):
    def handler(record, answer, world, tax, transform) -> Grade:
        truth = bool(answer_keys(record)[field_name])
        got = _as_bool(answer.get(field_name))
        ok = got is not None and got == truth
        return Grade(True, float(ok), {"correct": ok})
    return handler



def _g_v15(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)
    if "same_moment" in truth:
        return _g_bool("same_moment")(record, answer, world, tax, transform)
    state_ok = _norm(answer.get("state")) == _norm(truth["state"])
    view_ok = _norm(answer.get("best_view")) in ("second", "2", "image 2",
                                                 _norm(truth["best_view"]))
    return Grade(True, float(state_ok and view_ok), {"state": state_ok, "best_view": view_ok})


def _g_v16(record, answer, world, tax, transform) -> Grade:
    truth = answer_keys(record)
    got = _as_bool(answer.get("valid"))
    ok = got is not None and got == bool(truth["valid"])
    located = None
    if not truth["valid"]:
        step = answer.get("first_error_step", answer.get("first_error_index"))
        try:
            index = int(step) - 1 if "first_error_step" in answer else int(step)
        except (TypeError, ValueError):
            index = None
        located = ok and index == truth["first_error_index"]
    return Grade(True, float(ok), {"correct": ok, "localised": located})


def gold_reply(record: dict, world: Optional[World] = None, *, plain: bool = False) -> str:
    """The reply a perfect model would give, in :data:`RESPONSE_FORMATS`.

    Instance keys become their wording when a ``world`` is given, the way a
    model would name them. A sanity check of the whole chain: every gold reply
    must score 1 (``tests/test_w3b_scripts.py``). ``plain`` names every part the way a
    person reads it, no ordinal (:meth:`World.plain_name`): the reasoning side product's
    answer, never graded.
    """
    label, task = record["label"], record["prompt"]["task"]
    answer = label["answer"]

    def name(key):
        if world is not None and (key in world.instances or world.is_group(key)
                                  or (plain and str(key).startswith(CABLE_PREFIX))):
            return world.plain_name(key) if plain else world.label_of(key)
        return key

    key = format_key(record)
    if key == "V1":
        out = {"components": [{"class": c["class"], "bbox": c["bbox"]}
                              for c in answer["components"]]}
    elif key.startswith("V2"):
        out = {"count": answer["count"]} if "count" in answer else {"state": answer["state"]}
    elif key == "V3.span":
        out = {"actions": [{"verb": a["verb"], "target": name(a["target"])}
                           for a in answer["actions"]]}
    elif task == "V3":
        out = {"verb": answer["verb"], "target": name(answer["target_instance"]),
               "tool": answer.get("tool")}
        if label["answer_check"].get("region"):
            out["region"] = label["answer_check"]["region"]
    elif task == "V4":
        out = {"feasible": answer["feasible"], "blockers": [name(b) for b in answer["blockers"]]}
    elif task == "V5":
        out = {"actions": [{"verb": a["verb"], "target": name(a["target"])}
                           for a in answer["must_include"]]}
    elif key == "V6.options":
        index = reference_option(label)
        if index is None:
            raise ValueError(f"{record.get('id')}: no option is the reference action")
        out = {"choice": LETTERS[index]}
    elif key == "V6.open" and "actions" in answer:
        out = {"actions": [{"verb": a["verb"], "target": name(a["target"])}
                           for a in answer["actions"]]}
    elif key == "V6.open":  # a record from before the final round: one next action
        out = {"verb": answer["verb"], "target": name(answer["target"])}
    elif task == "V7":
        out = {"plan": [{"verb": a["verb"], "target": name(a["target"])}
                        for a in answer["plan"]]}
    elif key == "V8.components":
        out = {"components": [{"bbox": c["bbox"]} for c in _asked_components(record, world)]}
    elif task == "V8":
        out = {"bbox": answer["bbox"]}
    elif task == "V10":
        out = {"done": [{"verb": a["verb"], "target": name(a["target"])} for a in answer["done"]],
               "progress": answer["progress_bin"]}
        if "started" in answer:
            out["started"] = [{"verb": a["verb"], "target": name(a["target"])}
                              for a in answer["started"]]
    elif task == "V12":
        out = {"changed": answer["changed"]}

    elif key == "V15.moment":
        out = {"same_moment": answer["same_moment"]}
    elif key == "V15.cross":
        out = {"state": answer["state"], "best_view": "second"}
    elif task == "V16":
        index = answer["first_error_index"]
        out = {"valid": answer["valid"],
               "first_error_step": None if index is None else index + 1}
    else:
        out = dict(answer)
    return json.dumps(out, ensure_ascii=False)


_GRADERS = {
    "V1": _g_v1, "V2": _g_v2, "V3": _g_v3, "V4": _g_v4, "V5": _g_v5, "V6": _g_v6,
    "V7": _g_v7, "V8": _g_v8, "V10": _g_v10, "V12": _g_bool("changed"),
    "V15": _g_v15, "V16": _g_v16,
}
