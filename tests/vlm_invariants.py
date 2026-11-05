"""The correctness invariants of a built VLM pack (owner 2026-11-04: "no wrong questions").

``python tests/vlm_invariants.py <pack dir>`` reads every ``data/vlm_<unit>.jsonl`` with its
oracle and asserts, record by record:

1. **demonstrated actions are feasible**: no V4 record says "no" to the action the log
   performed next from that moment;
2. **no feasible "yes" under a cover or a loose part**: a V4 "yes" names no part a part above
   covers (>= 20 %, the record's ``moment.occluded``) nor one a loose part holds back
   (outside ``moment.loose_scope``);
3. **the frontier is visible**: every target of a V6 group is visible or pointable in the
   scanner frame shown (the oracle's visibility), or listed as hidden by a cable;
4. **plans replay**: every V7 gold plan and every valid V16 plan replays under the stored
   edges, the moment's own rules and the question's rule (the scorer's own replay for V7,
   a replay of the listed plan for V16), and every invalid V16 plan breaks where it says;
5. **no V12 "changed" without a state event**: the events listed are the state changes
   between the two moments' state tables, and a "no change" has none a picture shows;
6. **words**: no question, option, answer string or reasoning names an instance key, an
   ordinal ("connector 3") or the graph's words.

Exit 0 only when nothing is violated; ``invariants.json`` beside the pack's data says what
was checked and every violation.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent))

import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import (  # noqa: E402
    answer_keys, constraint_edges_of, gold_reply, grade, moment_edges_of)
from tda.core.graph import applicable_preconditions, unmet  # noqa: E402
from tda.core.graph_rules import verb_applies, verb_effect  # noqa: E402
from tda.core.states import InstState  # noqa: E402
from tda.core.taxonomy import load_taxonomy  # noqa: E402

ANSWERABLE = {"visible", "occluded_partial", "visible_tiny"}
KEY = re.compile(r"\b[a-z_]+(?:\.[a-z_0-9]+)*\.\d{2}\b")
GRAPH_WORDS = re.compile(r"violates|_by\(|\bedge\b|recall_relation|check_precondition")
ORDINAL = re.compile(r"\b([A-Za-z]+) (\d+)\b(?! ?[A-Za-z%])")
#: Words a number may follow without being an ordinal of a part.
NOT_ORDINAL = {"step", "steps", "image", "images", "of", "the", "all", "and", "or", "is", "are",
               "only", "to", "than", "with", "at", "in", "by", "for", "every", "first", "last",
               "next", "from", "least", "most", "about", "after", "before", "those", "these",
               "were", "was", "be", "on", "show", "shows", "has", "have", "plugs", "count"}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def ordinals(text: str) -> list[str]:
    return [m.group(0) for m in ORDINAL.finditer(text)
            if m.group(1).lower() not in NOT_ORDINAL]


def replay(world, tax, start: dict, plan, edges) -> tuple[bool, int]:
    """``(clean, first bad index)`` of a plan replayed from ``start`` (``-1`` when clean)."""
    sim = {k: InstState(state=v, placement="in_chassis") for k, v in start.items()}
    for i, item in enumerate(plan):
        verb, target = item["verb"], item["target"]
        members = world.members(target)
        todo, captive_loose = [], 0
        for key in members:
            rec = world.rec(key)
            cls = rec.cls if rec is not None else ("cable" if key.startswith("cable:") else None)
            if cls is None:
                return False, i
            cur = sim[key].state if key in sim else tax.default_state(cls)
            if verb_applies(tax, cls, rec.attrs if rec else {}, verb, cur):
                todo.append((key, cls, rec))
            elif rec is not None and rec.cls == "screw" and (rec.attrs or {}).get("captive") \
                    and verb == "unscrew" and cur == "loosened":
                captive_loose += 1  # a captive screw already loose: still "to do" in words
        if not todo:
            if captive_loose:
                continue
            return False, i
        for key, cls, rec in todo:
            if unmet(applicable_preconditions(edges, (verb, key)), sim, "required"):
                return False, i
        for key, cls, rec in todo:
            new = verb_effect(tax, cls, rec.attrs if rec else {}, verb)
            sim[key] = InstState(state=new, placement="in_chassis")
            if new == "removed" and cls != "connector":
                for child, inst in world.instances.items():
                    if inst.get("attached") and inst.get("parent") == key and child in sim:
                        sim[child] = InstState(state="removed", placement="in_chassis")
    return True, -1


def demonstrated(world) -> dict[int, tuple[str, str]]:
    """``step -> (verb, target)``: the action the log takes next from that step's moment."""
    by_step = defaultdict(list)
    for a in world.log:
        by_step[int(a["step"])].append(a)
    steps = sorted(by_step)
    out = {}
    for s in [0, *steps]:
        nxt = next((t for t in steps if t > s), None)
        if nxt is None:
            continue
        first = sorted(by_step[nxt], key=lambda a: a["idx"])[0]
        if first.get("changes_state"):
            out[s] = (first["verb"], first["target"])
    return out


def check_unit(path: Path, tax) -> dict:
    worlds, frames = IO.load_oracle(path)
    demo = {d: demonstrated(w) for d, w in worlds.items()}
    seen = Counter()
    bad: dict[str, list[str]] = defaultdict(list)
    for r in IO.read_jsonl(path):
        task = r["prompt"]["task"]
        label = r["label"]
        check = label["answer_check"]
        keys = answer_keys(r)
        world = worlds.get(IO.desktop_of(r))
        rid = label["readable_id"]
        seen[task] += 1
        # 6. words
        texts = [r["prompt"]["question"], *strings(r["prompt"].get("options") or []),
                 *strings(label["answer"]), label.get("rationale", {}).get("text") or ""]
        for text in texts:
            if KEY.search(text):
                bad["key_in_text"].append(f"{rid}: {KEY.search(text).group(0)}")
            if GRAPH_WORDS.search(text):
                bad["graph_words"].append(f"{rid}: {GRAPH_WORDS.search(text).group(0)}")
            found = ordinals(text)
            if found:
                bad["ordinal"].append(f"{rid}: {found[0]!r}")
        for text in strings(label["answer"]):
            if ordinals(text) or KEY.search(text):
                bad["ordinal_in_answer"].append(f"{rid}: {text!r}")
        if task == "V4":
            members = check.get("members") or [check["target"]]
            step = int(check.get("state_after", label["step"]))
            nxt = demo.get(IO.desktop_of(r), {}).get(step)
            # 1. a demonstrated action is never a "no"
            if not keys["feasible"] and nxt and nxt[0] == check["verb"] and nxt[1] in members:
                bad["demonstrated_infeasible"].append(rid)
            # 2. no "yes" under a cover or a loose part
            if keys["feasible"]:
                moment = check.get("moment") or {}
                if any(moment.get("occluded", {}).get(m) for m in members):
                    bad["feasible_but_covered"].append(rid)
                if moment.get("loose"):
                    scope = set(moment.get("loose_scope") or [])
                    if any(m not in scope and m not in moment["loose"] for m in members):
                        bad["feasible_but_loose_first"].append(rid)
        if task == "V6" and check.get("type") == "frontier":
            # 3. the frontier is visible
            table = (frames.get(r["prompt"]["images"][0]) or {}).get("state") or {}
            for g in check["groups"] + (check.get("ruled_groups") or []):
                hidden = set(g.get("hidden_by_cable") or [])
                for a in g["actions"]:
                    vis = (table.get(a["target"]) or {}).get("visibility")
                    if vis not in ANSWERABLE and a["target"] not in hidden:
                        bad["frontier_not_visible"].append(f"{rid}: {a['target']} {vis}")
        if task == "V7":
            # 4. the gold plan replays (the scorer's own replay)
            g = grade(r, gold_reply(r, world), world, tax)
            if not (g.metrics.get("violation_free") and g.metrics.get("goal_reached")):
                bad["v7_gold_does_not_replay"].append(f"{rid}: {g.metrics}")
            nodes = [k for k, v in check["start_state"].items() if v != "removed"]
            ok, at = replay(world, tax, check["start_state"], keys["plan"],
                            [*world.edges, *moment_edges_of(check, nodes),
                             *constraint_edges_of(check)])
            if not ok:
                bad["v7_plan_breaks"].append(f"{rid}: step {at}")
        if task == "V16":
            table = (frames.get(r["prompt"]["images"][0]) or {}).get("state") or {}
            start = {k: v["state"] for k, v in table.items()}
            members = sorted({m for item in check["plan"] for m in world.members(item["target"])})
            ok, at = replay(world, tax, start, check["plan"],
                            [*world.edges, *moment_edges_of(check, members)])
            if keys["valid"] and not ok:
                bad["v16_valid_breaks"].append(f"{rid}: step {at}")
            if not keys["valid"] and (ok or at != keys["first_error_index"]):
                bad["v16_invalid_mislocated"].append(f"{rid}: replay {at} vs "
                                                     f"{keys['first_error_index']}")
        if task == "V12":
            # 5. no "changed" without a state event; the events are the tables' differences
            images = r["prompt"]["images"]
            steps = label.get("image_steps") or []
            first = int(check.get("from_step", label["step"] - 1))
            before = next((frames[i]["state"] for i, s in zip(images, steps) if s == first
                           and i in frames), None)
            after = next((frames[i]["state"] for i, s in zip(images, steps)
                          if s == label["step"] and i in frames), None)
            if keys["changed"] and not keys["events"]:
                bad["v12_changed_without_event"].append(rid)
            if before is not None and after is not None:
                for e in keys["events"]:
                    if (before.get(e["target"]) or {}).get("state") != e["old"] or \
                            (after.get(e["target"]) or {}).get("state") != e["new"]:
                        bad["v12_event_not_in_tables"].append(f"{rid}: {e['target']}")
    return {"records": dict(seen), "violations": {k: len(v) for k, v in bad.items()},
            "examples": {k: v[:10] for k, v in bad.items()}}


def main(argv=None) -> int:
    pack = Path((argv or sys.argv[1:])[0])
    tax = load_taxonomy()
    report = {}
    for path in sorted((pack / "data").glob("vlm_*.jsonl")):
        if path.name.count(".") > 1:
            continue
        report[path.stem] = check_unit(path, tax)
        print(path.stem, report[path.stem]["records"], "violations:",
              report[path.stem]["violations"] or 0)
    total = sum(n for u in report.values() for n in u["violations"].values())
    (pack.parent / "invariants.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("total violations", total)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
