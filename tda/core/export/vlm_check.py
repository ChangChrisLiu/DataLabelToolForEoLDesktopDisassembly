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
           "World", "format_instruction", "gold_reply", "grade", "grounder",
           "parse_lenient", "parse_strict"]

#: The paper's capability layers (minutes 2026-10-31 §3; brief W3b §2.8). Not
#: spec 8.2's L1-L4 of ``label.layer``: these group the tasks the way the
#: results table does.
CAPABILITY_LAYERS = {
    "V1": "L1", "V2": "L1", "V8": "L1", "V14": "L1",
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
    "V2.state": 'Answer with JSON only: {"state": "<one of: {allowed}>"}.',
    "V2.count": 'Answer with JSON only: {"count": <integer>}.',
    "V3": 'Answer with JSON only: {"verb": "<one of: ' + ", ".join(VERBS) + '>", '
          '"target": "<the part acted on>", "tool": "<tool, or null>"}.',
    "V4": 'Answer with JSON only: {"feasible": true or false, "blockers": '
          '["<part that has to be dealt with first>", ...]}.',
    "V5": 'Answer with JSON only: {"actions": [{"verb": "<verb>", "target": '
          '"<part>"}, ...]}.',
    "V6.open": 'Answer with JSON only: {"verb": "<one of: ' + ", ".join(VERBS) + '>", '
               '"target": "<part>"}.',
    "V6.options": 'Answer with JSON only: {"choice": "<letter of the option>"}.',
    "V7": 'Answer with JSON only: {"plan": [{"verb": "<verb>", "target": "<part>"}, '
          '...]} in the order you would carry them out.',
    "V8": 'Answer with JSON only: {"bbox": [x, y, width, height]} in pixel '
          'coordinates of the image.',
    "V10": 'Answer with JSON only: {"done": [{"verb": "<verb>", "target": "<part>"}, '
           '...], "progress": "<one of: 0-25, 25-50, 50-75, 75-100>"} (percent of '
           'the teardown finished).',
    "V12": 'Answer with JSON only: {"changed": true or false}.',
    "V14": 'Answer with JSON only: {"answerable": true or false, "state": '
           '"<one of: {allowed}>, or null when it cannot be told"}.',
    "V15.cross": 'Answer with JSON only: {"state": "<state>", "best_view": '
                 '"<first or second>"}.',
    "V15.moment": 'Answer with JSON only: {"same_moment": true or false}.',
    "V16": 'Answer with JSON only: {"valid": true or false, "first_error_step": '
           '<number of the first step that cannot be carried out, or null>}.',
}

#: Option letters, in the order the options are listed.
LETTERS = "ABCDEFGHIJ"
#: Tasks whose ``answer_check`` carries the verb vocabulary in play.
_BOOL_TRUE = ("yes", "true", "same", "possible", "can", "valid", "changed")
_BOOL_FALSE = ("no", "false", "not", "cannot", "can't", "invalid", "different",
               "unchanged")


def format_key(record: dict) -> str:
    """Which :data:`RESPONSE_FORMATS` entry a record is answered in."""
    task = record["prompt"]["task"]
    check = record["label"]["answer_check"]
    if task == "V2":
        return "V2.count" if check.get("derive") == "screw_count" else "V2.state"
    if task == "V6":
        return "V6.options" if record["prompt"].get("options") else "V6.open"
    if task == "V15":
        return "V15.moment" if check.get("derive") == "same_moment" else "V15.cross"
    return task


def format_instruction(record: dict) -> str:
    """The answer-format sentence a chat prompt appends to the question."""
    key = format_key(record)
    text = RESPONSE_FORMATS[key]
    if "{allowed}" in text:
        allowed = record["label"]["answer_check"].get("allowed") or []
        if not allowed and record["label"]["answer_check"].get("state_space") == "presence":
            allowed = ["present", "removed"]
        text = text.replace("{allowed}", ", ".join(allowed) if allowed else "<state>")
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


def _balanced(text: str) -> list[str]:
    """Every top-level ``{...}`` substring, in order."""
    out, depth, start = [], 0, None
    in_str, esc, quote = False, False, ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
    return out


def parse_lenient(raw: Optional[str], record: Optional[dict] = None) -> Optional[dict]:
    """A forgiving reading: the first object found, else the prose, per task."""
    strict = parse_strict(raw)
    if strict is not None:
        return strict
    if raw is None:
        return None
    text = str(raw)
    for chunk in _balanced(text):
        for attempt in (chunk, chunk.replace("'", '"')):
            try:
                value = json.loads(attempt)
            except ValueError:
                value = None
            if isinstance(value, dict):
                return value
        try:
            value = ast.literal_eval(chunk.replace("true", "True")
                                     .replace("false", "False").replace("null", "None"))
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, dict):
            return value
    if record is None:
        return None
    return _from_prose(text, record)


def _yes_no(text: str) -> Optional[bool]:
    words = re.findall(r"[a-z']+", text.lower())
    for word in words[:12]:
        if word in _BOOL_FALSE:
            return False
        if word in _BOOL_TRUE:
            return True
    return None


def _from_prose(text: str, record: dict) -> Optional[dict]:
    """The last resort of the lenient parser: one field read off the prose."""
    key = format_key(record)
    if key == "V6.options":
        hit = re.search(r"\b([A-J])\b", text)
        return {"choice": hit.group(1)} if hit else None
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


@dataclass
class World:
    """One desktop's oracle line: instances, active edges, log (W3b sidecar)."""

    desktop: int
    instances: dict[str, dict]
    edges: list[Edge]
    log: list[dict] = field(default_factory=list)

    @classmethod
    def from_oracle(cls, line: dict) -> "World":
        edges = [Edge(type=e["type"], target=e["target"], blocker=e["blocker"],
                      necessity=e.get("necessity", "required"), mode=e.get("mode"),
                      source=e.get("source", "rule"), status=e.get("status", "proposed"))
                 for e in line.get("edges", [])]
        return cls(int(line["desktop"]), dict(line.get("instances", {})), edges,
                   list(line.get("log", [])))

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
        inst = self.instances.get(key)
        return inst["class"] if inst else None

    def match_class(self, key: str) -> Optional[str]:
        if key.startswith(CABLE_PREFIX):
            return "cable"
        inst = self.instances.get(key)
        return inst.get("match_class") if inst else None


class Grounder:
    """Model wording -> an instance key, or at least a match class.

    An exact instance label ("CPU cooler screw 3") or key names one instance;
    a class phrase ("a CPU cooler screw") names a match class, and an instance
    of it is chosen only when the caller says which ones are eligible -- that
    is the group-equivalent reading of an answer that did not say which screw.
    """

    def __init__(self, world: World):
        self.world = world
        self.by_label: dict[str, str] = {}
        self.phrases: list[tuple[str, str]] = []  # (phrase, match_class), longest first
        seen: set[tuple[str, str]] = set()
        for key, inst in sorted(world.instances.items()):
            label = _norm(inst.get("label") or key)
            self.by_label.setdefault(label, key)
            self.by_label.setdefault(_norm(key), key)
            phrase = re.sub(r"\s+\d+$", "", label)
            item = (phrase, inst.get("match_class") or inst["class"])
            if phrase and item not in seen:
                seen.add(item)
                self.phrases.append(item)
        self.phrases.sort(key=lambda p: (-len(p[0]), p[0]))

    def ground(self, text: Any) -> tuple[Optional[str], Optional[str]]:
        """``(instance key or None, match class or None)``."""
        if text is None:
            return None, None
        raw = str(text).strip()
        if raw in self.world.instances or raw.startswith(CABLE_PREFIX):
            return raw, self.world.match_class(raw)
        norm = _norm(raw)
        if norm in self.by_label:
            key = self.by_label[norm]
            return key, self.world.match_class(key)
        for phrase, mclass in self.phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", norm):
                # a phrase plus an ordinal nobody has is no instance at all
                return None, mclass
        return None, None

    def ground_in(self, text: Any, eligible: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
        """:meth:`ground`, resolving a class phrase to the first eligible member."""
        key, mclass = self.ground(text)
        if key is not None or mclass is None:
            return key, mclass
        for candidate in eligible:
            if self.world.match_class(candidate) == mclass:
                return candidate, mclass
        return None, mclass


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


def _class_of_text(text: Any, tax: Optional[Taxonomy], classes: Sequence[str]) -> Optional[str]:
    norm = _norm(text)
    best = None
    for cls in classes:
        phrase = _norm(cls)
        if tax is not None:
            from tda.core.export.vlm_tasks import class_label

            phrase = _norm(class_label(cls, tax))
        if norm == phrase or norm == _norm(cls) or re.search(rf"\b{re.escape(phrase)}\b", norm):
            if best is None or len(phrase) > len(_norm(best)):
                best = cls
    return best


def _g_v1(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]["components"]
    classes = sorted({c["class"] for c in truth})
    preds = []
    for item in answer.get("components") or []:
        if not isinstance(item, dict):
            continue
        preds.append((_class_of_text(item.get("class"), tax, classes),
                      _box(item.get("bbox"), transform)))
    used, tp = set(), 0
    thr = float(record["label"]["answer_check"].get("iou_threshold", 0.5))
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


def _g_v2(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    if "count" in truth:
        try:
            ok = int(answer.get("count")) == int(truth["count"])
        except (TypeError, ValueError):
            ok = False
    else:
        ok = _norm(answer.get("state")) == _norm(truth["state"])
    return Grade(True, float(ok), {"correct": ok,
                                   "prior_is_correct": record["label"].get("prior_is_correct")})


def _g_v3(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    verb_ok = _norm(answer.get("verb")) == _norm(truth["verb"])
    key, mclass = (grounder(world).ground(answer.get("target")) if world
                   else (None, None))
    cls = world.cls_of(key) if (world and key) else (mclass.split(".")[0] if mclass else None)
    class_ok = cls == truth.get("target_class")
    inst_ok = key == truth.get("target_instance")
    metrics = {"verb": verb_ok, "target_class": class_ok, "target_instance": inst_ok}
    if truth.get("tool") is not None:
        metrics["tool"] = _norm(answer.get("tool")) == _norm(truth["tool"])
    return Grade(True, float(verb_ok and class_ok), metrics)


def _g_v4(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    feasible = _as_bool(answer.get("feasible"))
    ok = feasible is not None and feasible == bool(truth["feasible"])
    metrics: dict = {"correct": ok, "truth_source": record["label"].get("truth_source")}
    if not truth["feasible"] and world is not None:
        g = grounder(world)
        got = {g.ground_in(b, truth["blockers"])[0] for b in answer.get("blockers") or []}
        got.discard(None)
        want = set(truth["blockers"])
        metrics["blockers_f1"] = _f1(len(got & want), len(got), len(want))
    return Grade(True, float(ok), metrics)


def _pairs_grounded(items: Any, world: Optional[World], eligible: Sequence[tuple[str, str]]
                    ) -> list[tuple[str, Optional[str], Optional[str]]]:
    """``[(verb, key, match_class)]`` for a list of ``{verb, target}`` answers.

    A class phrase is resolved to an eligible instance of its class, each
    eligible one used once -- "unscrew a CPU cooler screw" twice is two screws.
    """
    out = []
    g = grounder(world) if world else None
    left = list(eligible)
    spare = sorted(world.instances) if world else []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        verb = _norm(item.get("verb")).split(" ")[0] if item.get("verb") else ""
        if g is None:
            out.append((verb, None, None))
            continue
        key, mclass = g.ground(item.get("target"))
        if key is None and mclass is not None:
            for i, (v, t) in enumerate(left):
                if v == verb and world.match_class(t) == mclass:
                    key = t
                    del left[i]
                    break
        if key is None and mclass is not None:
            # a class nothing eligible has: the first part of that class
            for i, t in enumerate(spare):
                if world.match_class(t) == mclass:
                    key = t
                    del spare[i]
                    break
        if key is not None and key in spare:
            spare.remove(key)
        out.append((verb, key, mclass))
    return out


def _g_v5(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    must = {(a["verb"], a["target"]) for a in truth["must_include"]}
    must_not = {(a["verb"], a["target"]) for a in truth["must_not_include"]}
    eligible = [(a["verb"], a["target"]) for a in truth["permitted_upper_bound"]]
    got = {(v, k) for v, k, _ in _pairs_grounded(answer.get("actions"), world,
                                                  eligible) if k}
    recall = len(got & must) / len(must) if must else 1.0
    blocked = len(got & must_not) / len(got) if got else 0.0
    return Grade(True, recall, {"demonstrated_recall": recall, "blocked_rate": blocked})


def _g_v6(record, answer, world, tax, transform) -> Grade:
    label = record["label"]
    check = label["answer_check"]
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
        exact = pair == reference
        group = pair in equivalent
        hit_blocked = option["kind"] in ("graph_blocked", "graph_blocked_pixels")
        return Grade(True, float(group), {
            "group_match": group, "exact_match": exact, "blocked": hit_blocked,
            "certainly_wrong": option["kind"] in ("graph_blocked", "graph_blocked_pixels",
                                                  "state_inapplicable"),
            "blocked_rate_defined": bool(check.get("blocked_rate_defined"))})
    verb = _norm(answer.get("verb")).split(" ")[0] if answer.get("verb") else ""
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


def _g_v7(record, answer, world, tax, transform) -> Grade:
    check = record["label"]["answer_check"]
    if world is None or tax is None:
        raise ValueError("V7 needs the world and the taxonomy")
    sim: dict[str, InstState] = {k: InstState(state=v, placement="in_chassis")
                                 for k, v in check["start_state"].items()}
    eligible_pairs = [(i["verb"], i["target"]) for i in check["reference_plan"]] + \
        [(i["verb"], i["target"]) for i in check["demonstrated_remainder"]]
    steps = _pairs_grounded(answer.get("plan"), world, eligible_pairs)
    violations, first_violation, unresolved, noop = 0, None, 0, 0
    for index, (verb, key, _) in enumerate(steps):
        if key is None:
            unresolved += 1
            continue
        rec = world.rec(key)
        cls = rec.cls if rec is not None else ("cable" if key.startswith(CABLE_PREFIX)
                                               else None)
        if cls is None:
            unresolved += 1
            continue
        attrs = rec.attrs if rec is not None else {}
        current = sim[key].state if key in sim else tax.default_state(cls)
        new = verb_effect(tax, cls, attrs, verb)
        if new is None or not verb_applies(tax, cls, attrs, verb, current):
            # no redundancy penalty: an action already done (or one that
            # changes nothing) is a no-op, not an error
            noop += 1
            continue
        gating = applicable_preconditions(world.edges, (verb, key))
        if unmet(gating, sim, "required"):
            violations += 1
            if first_violation is None:
                first_violation = index
        sim[key] = InstState(state=new, placement="in_chassis")
        if new == "removed" and cls != "connector":
            for child, inst in world.instances.items():
                if inst.get("attached") and inst.get("parent") == key and child in sim:
                    sim[child] = InstState(state="removed", placement="in_chassis")
    required = check["required_actions"]
    met = 0
    for req in required:
        state = sim.get(req["target"])
        got = state.state if state else None
        if got in set(req["states"]) or got == "removed":
            met += 1
    recall = met / len(required) if required else 1.0
    goal = sim.get(check["goal"])
    reached = bool(goal is not None and goal.state == check.get("goal_state", "removed"))
    success = reached and violations == 0 and recall == 1.0
    return Grade(True, float(success), {
        "violation_free": violations == 0, "violations": violations,
        "first_violation": first_violation, "required_recall": recall,
        "goal_reached": reached, "unresolved_steps": unresolved, "noop_steps": noop,
        "plan_length": len(steps), "reference_length": check.get("plan_length")})


def _g_v8(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]["bbox"]
    check = record["label"]["answer_check"]
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
    truth = record["label"]["answer"]
    want = {(a["verb"], a["target"]) for a in truth["done"]}
    got = {(v, k) for v, k, _ in _pairs_grounded(answer.get("done"), world,
                                                  sorted(want)) if k}
    f1 = _f1(len(got & want), len(got), len(want))
    progress = answer.get("progress", answer.get("progress_bin"))
    bin_ok = _norm(progress).replace(" ", "-") == _norm(truth["progress_bin"]).replace(" ", "-")
    return Grade(True, f1, {"set_f1": f1, "bin_correct": bin_ok})


def _g_bool(field_name: str):
    def handler(record, answer, world, tax, transform) -> Grade:
        truth = bool(record["label"]["answer"][field_name])
        got = _as_bool(answer.get(field_name))
        ok = got is not None and got == truth
        return Grade(True, float(ok), {"correct": ok})
    return handler


def _g_v14(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    got = _as_bool(answer.get("answerable"))
    ok = got is not None and got == bool(truth["answerable"])
    state_ok = None
    if truth["answerable"]:
        state_ok = _norm(answer.get("state")) == _norm(truth["state"])
    return Grade(True, float(ok and state_ok is not False),
                 {"answerable_correct": ok, "state_correct": state_ok,
                  "abstained": got is False})


def _g_v15(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
    if "same_moment" in truth:
        return _g_bool("same_moment")(record, answer, world, tax, transform)
    state_ok = _norm(answer.get("state")) == _norm(truth["state"])
    view_ok = _norm(answer.get("best_view")) in ("second", "2", "image 2",
                                                 _norm(truth["best_view"]))
    return Grade(True, float(state_ok and view_ok), {"state": state_ok, "best_view": view_ok})


def _g_v16(record, answer, world, tax, transform) -> Grade:
    truth = record["label"]["answer"]
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


def gold_reply(record: dict, world: Optional[World] = None) -> str:
    """The reply a perfect model would give, in :data:`RESPONSE_FORMATS`.

    Instance keys become their wording when a ``world`` is given, the way a
    model would name them. A sanity check of the whole chain: every gold reply
    must score 1 (``tests/test_w3b_scripts.py``).
    """
    label, task = record["label"], record["prompt"]["task"]
    answer = label["answer"]

    def name(key):
        if world is not None and key in world.instances:
            return world.instances[key].get("label") or key
        return key

    key = format_key(record)
    if key == "V1":
        out = {"components": [{"class": c["class"], "bbox": c["bbox"]}
                              for c in answer["components"]]}
    elif key.startswith("V2"):
        out = {"count": answer["count"]} if "count" in answer else {"state": answer["state"]}
    elif task == "V3":
        out = {"verb": answer["verb"], "target": name(answer["target_instance"]),
               "tool": answer.get("tool")}
    elif task == "V4":
        out = {"feasible": answer["feasible"], "blockers": [name(b) for b in answer["blockers"]]}
    elif task == "V5":
        out = {"actions": [{"verb": a["verb"], "target": name(a["target"])}
                           for a in answer["must_include"]]}
    elif key == "V6.options":
        ref = tuple(label["answer_check"]["reference"])
        index = next(i for i, o in enumerate(label["options"])
                     if (o["verb"], o["target"]) == ref)
        out = {"choice": LETTERS[index]}
    elif key == "V6.open":
        out = {"verb": answer["verb"], "target": name(answer["target"])}
    elif task == "V7":
        out = {"plan": [{"verb": a["verb"], "target": name(a["target"])}
                        for a in answer["plan"]]}
    elif task == "V8":
        out = {"bbox": answer["bbox"]}
    elif task == "V10":
        out = {"done": [{"verb": a["verb"], "target": name(a["target"])} for a in answer["done"]],
               "progress": answer["progress_bin"]}
    elif task == "V12":
        out = {"changed": answer["changed"]}
    elif task == "V14":
        out = {"answerable": answer["answerable"], "state": answer["state"]}
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
    "V14": _g_v14, "V15": _g_v15, "V16": _g_v16,
}
