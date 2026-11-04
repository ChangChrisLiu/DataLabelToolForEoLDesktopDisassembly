"""Score a model's answers against an export-vlm file (W3b).

``python scripts/vlm_score.py --truth <vlm_scan.jsonl> --pred <model.jsonl>
[--chat <chat.jsonl>] --out <dir> [--name <model>] [--seed 0] [--resamples 1000]``

``--pred`` is one JSON object per line, ``{"id", "raw_text"}``. Every record
of ``--truth`` is graded twice by :func:`tda.core.export.vlm_check.grade`:
**strict** (the reply is one JSON object) and **lenient** (the first object
in it, else the prose); a record with no prediction counts as a parse failure.
Both parse-failure rates are reported.

Aggregated per task, per capability layer (L1 V1 V2 V8 / L2 V4 / L3 V6 V7
V16 / change V3 V12 V10 / cross V15) and per question template, each with a
95 % interval from a **cluster bootstrap over desktops** (desktops resampled
with replacement, ``--resamples`` times, seeded). Beside the primary score:

* V2's headline row ``V2`` is every V2 question -- perceiving the part's
  current state (owner 2026-11-03; ``eval_v3`` draws V2 equally over classes and
  half ``prior_is_correct = false``); ``V2-prior-wrong`` (the questions the
  class alone does not answer) is the secondary row, a shortcut check;
* V6's headline (``headline.v6_open``, table row ``V6``) is the **open**
  question "list every action that can be carried out now": F1 over the frontier's
  groups (owner 2026-11-04), precision and recall beside it; the multiple choice
  (one option inside the frontier) is the secondary row ``V6-mc``
  (``headline.v6_mc``) and the constrained form (one rule in words) the row
  ``V6-constrained``, graded like the open form plus ``constraint_violation`` (the
  share of answers naming an action the rule forbids);
* V7 adds violation-free rate, required-action recall and goal-reached rate; the
  constrained form is the row ``V7-constrained``, with ``constraint_kept`` and the
  plug count (``count_exact``, ``count_abs_error``: how many plugs the plan unplugs
  against how many are in);
* V2's state questions answer ``cannot_tell`` on a part the picture does not show
  (V14 merged into V2): ``cannot_tell_accuracy`` is the row over those;
* V10 adds the progress-bin accuracy and, on records that carry
  ``answer.started`` (groups partly done, spec 6.1), ``started_f1`` -- the set
  of group actions begun but not finished, graded apart from the done set;
* action targets may be group ids (the redesign's ``groups`` in the oracle):
  :mod:`tda.core.export.vlm_check` grounds the model's words to them;
* records whose truth rests on a captive screw's tightness
  (``image_undecidable``) are also scored without;
* two reference rows stand beside every task (W4-C): **chance** -- the
  expected score of a uniform guess over the record's answer space, by the
  rule in :data:`CHANCE_RULES` (binary tasks 0.5, V6-mc 1/|options|, V2
  1/|allowed|, open tasks 0) -- and **frequency** -- the most frequent gold
  reply over the truth file's ``train`` desktops (``label.split``; for V6-mc
  the most frequent option letter), given as the answer to every record and
  graded like a model;
* the export's text-only shortcut bound per task (``<truth>.summary.json``) is
  a standing column, labelled what it is: an **in-sample upper bound**,
  measured on the full export by prompt-only classifiers.

Writes ``<out>/<name>.score.json`` and ``<out>/<name>.score.md``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import CAPABILITY_LAYERS, count_score, grade  # noqa: E402

LAYER_ORDER = ("L1", "L2", "L3", "change", "cross")
#: V6's secondary record type, the multiple choice (W3b ruling): ``V6`` in the
#: tables is the open form, the headline.
V6_MC = "V6-mc"
#: The V2 questions the class prior answers wrong (``prior_is_correct = false``);
#: the ``V2`` row, the headline, is every V2 question.
V2_PRIOR_WRONG = "V2-prior-wrong"
#: V3's second form (owner 2026-11-04): the same step with a yellow hint box drawn around where
#: the machine changed. ``V3`` in the tables is the free form, the headline.
V3_ROI = "V3-roi"
#: The constrained forms (owner 2026-11-04): one rule in words added to the question.
V6_CONSTRAINED = "V6-constrained"
V7_CONSTRAINED = "V7-constrained"
#: Secondary record types: the layers count each headline row once.
SECONDARY_ROWS = frozenset({"V6-mc", V3_ROI, V6_CONSTRAINED, V7_CONSTRAINED})


#: How the ``chance`` row is computed, per table row (printed under the table).
CHANCE_RULES = {
    "V1": "0: the set of (class, box) pairs is open; a guess has F1 0",
    "V2": "state form 1/|allowed| (the class's state vocabulary; presence = 2); "
          "count form (1 + 0.8 k)/(n+1), a uniform guess over 0..n (n the desktop's screws "
          "of that role) scored exact 1.0 / off by one 0.8, k = the gold's neighbours in "
          "0..n (the extra row exact: exact matches only)",
    V2_PRIOR_WRONG: "as V2, over the questions the class prior answers wrong",
    "V3": "0: verb and target are open and the score needs both (the region is the extra "
          "rows region_iou / region_hit)",
    V3_ROI: "0: verb and target are open and the score needs both (a hint box shows where)",
    "V4": "0.5: feasible yes / no (blockers are the extra row blockers_f1)",
    "V6": "0: the set of actions possible now is open (F1 over groups)",
    V6_MC: "1/|options| per record",
    V6_CONSTRAINED: "0: the set of actions possible under the rule is open",
    "V7": "0: the plan is open",
    V7_CONSTRAINED: "0: the plan is open",
    "V8": "0: a box with IoU >= 0.5 by chance (the components form: a box set, F1 0)",
    "V10": "0: the set F1 over the done actions is open; the progress bin alone is "
           "1/|bins| (the extra row bin_accuracy)",
    "V12": "0.5: changed yes / no",

    "V15": "0.5: same moment yes / no",
    "V16": "0.5: valid yes / no (the error step is the extra row localised)",
}


def task_key(record: dict) -> str:
    """The table row of a record: its task, with V6's two forms apart and V3's two forms apart."""
    task = record["prompt"]["task"]
    if task == "V6" and (record["label"].get("v6_form") == "mc"
                         or record["prompt"].get("options")):
        return V6_MC
    if task == "V6" and record["label"].get("v6_form") == "constrained":
        return V6_CONSTRAINED
    if task == "V7" and record["label"].get("v7_form") == "constrained":
        return V7_CONSTRAINED
    if task == "V3" and (record["label"].get("v3_form") == "roi"
                         or record["prompt"].get("hint_regions")):
        return V3_ROI
    return task


def _rows_of(task: str, label: dict) -> list[str]:
    """The table rows a record counts in: its task; V2 also ``V2-prior-wrong``.

    V2's headline row is every V2 question (owner 2026-11-03: V2 measures the
    perception of the current state); the questions the class prior gets wrong
    are the secondary row ``V2-prior-wrong`` (review M).
    """
    if task == "V2" and label.get("prior_is_correct") is False:
        return [task, V2_PRIOR_WRONG]
    return [task]


def _states_of(record: dict, world, tax) -> list[str]:
    """The answer space of a state question: ``allowed``, presence, or the class's."""
    check = record["label"]["answer_check"]
    allowed = check.get("allowed") or []
    if allowed:
        return list(allowed)
    if check.get("state_space") == "presence":
        return ["present", "removed"]
    inst = check.get("instance")
    cls = (world.cls_of(inst) if (world is not None and inst) else None) or check.get("class")
    if cls and tax is not None:
        try:
            return list(tax.states_of(cls))
        except KeyError:
            pass
    return []


def chance_of(record: dict, world, tax) -> Optional[float]:
    """The expected score of a uniform guess over the record's answer space
    (:data:`CHANCE_RULES`); ``None`` when the space cannot be sized."""
    task = record["prompt"]["task"]
    label = record["label"]
    answer = label["answer"]
    check = label["answer_check"]
    if task == "V6":
        options = record["prompt"].get("options") or label.get("options")
        return 1.0 / len(options) if options else 0.0
    if task in ("V1", "V3", "V7", "V8", "V10"):
        return 0.0
    if task == "V2":
        if "count" in answer:
            if world is None:
                return None
            role, captive = check.get("role"), check.get("captive")
            n = sum(1 for inst in world.instances.values()
                    if inst.get("class") == "screw"
                    and (inst.get("attrs") or {}).get("role") == role
                    and (captive is None
                         or bool((inst.get("attrs") or {}).get("captive")) == bool(captive)))
            # a uniform guess over 0..n, graded exact 1.0 / off by one 0.8 (owner 2026-11-04)
            gold = int(answer["count"])
            return sum(count_score(guess, gold)[0] for guess in range(n + 1)) / (n + 1)
        states = _states_of(record, world, tax)
        return 1.0 / len(states) if states else None
    if task in ("V4", "V12", "V16"):
        return 0.5
    if task == "V15":
        return 0.5 if "same_moment" in answer else 0.0
    return None


def frequency_rows(records: list[dict], worlds: dict, tax, *, resamples: int, seed: int
                   ) -> dict[str, dict]:
    """Per table row: the most frequent gold reply over the row's ``train``
    records, answered to every record of the row and graded strict.

    Fitted on ``label.split == "train"`` (in-sample there, out of sample on the
    test desktops); a row with no train record is fitted on all its records and
    says so (``fit``). Ties take the smallest reply string.
    """
    from tda.core.export.vlm_check import gold_reply

    by_row: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for key in _rows_of(task_key(record), record["label"]):
            by_row[key].append(record)
    out: dict[str, dict] = {}
    for key, rows in sorted(by_row.items()):
        train = [r for r in rows if r["label"].get("split") == "train"]
        pool, fit = (train, "train") if train else (rows, "all (no train record)")
        counts: Counter = Counter(gold_reply(r, worlds.get(IO.desktop_of(r))) for r in pool)
        best = min(counts, key=lambda s: (-counts[s], s))
        graded = [(IO.desktop_of(r), grade(r, best, worlds.get(IO.desktop_of(r)), tax,
                                           mode="strict").score) for r in rows]
        cell = _cell(graded, resamples, seed)
        cell.update({"answer": best, "fit": fit, "fit_records": len(pool),
                     "answer_share_in_fit": _r(counts[best] / len(pool))})
        out[key] = cell
    return out


def _task_order(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


def cluster_bootstrap(rows: list[tuple[int, float]], resamples: int, seed: int
                      ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """``(mean, lo, hi)``: 95 % percentile interval, desktops resampled whole.

    Records of one desktop are not independent (one machine, one operator, one
    lighting), so the unit of resampling is the desktop: each resample draws
    as many desktops as there are, with replacement, and takes the mean over
    all their records. Seeded, so the same inputs give the same interval.
    """
    if not rows:
        return None, None, None
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for desk, value in rows:
        sums[desk] += float(value)
        counts[desk] += 1
    desks = sorted(sums)
    point = sum(sums.values()) / sum(counts.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        total = n = 0.0
        for _ in desks:
            d = desks[rng.randrange(len(desks))]
            total += sums[d]
            n += counts[d]
        stats.append(total / n)
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return point, lo, hi


def _cell(rows, resamples, seed) -> dict:
    mean, lo, hi = cluster_bootstrap(rows, resamples, seed)
    return {"n": len(rows), "mean": _r(mean), "ci95": [_r(lo), _r(hi)],
            "desktops": len({d for d, _ in rows})}


def _r(x):
    return None if x is None else round(float(x), 4)


def score(truth_path, pred_path, chat_path=None, *, resamples: int = 1000,
          seed: int = 0, prior_path=None) -> dict:
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    records = list(IO.read_jsonl(truth_path))
    worlds, _ = IO.load_oracle(truth_path)
    summary = IO.load_summary(truth_path)
    preds: dict[str, Optional[str]] = {}
    for row in IO.read_jsonl(pred_path):
        preds[str(row["id"])] = row.get("raw_text")
    transforms: dict[str, list] = {}
    if chat_path:
        for row in IO.read_jsonl(chat_path):
            meta = row.get("meta") or {}
            if meta.get("transforms"):
                transforms[str(row["id"])] = meta["transforms"]
    out: dict = {"records": len(records), "predicted": sum(1 for r in records
                                                           if r["id"] in preds)}
    chance_rows: dict[str, list[float]] = defaultdict(list)
    chance_unsized: dict[str, int] = defaultdict(int)
    for mode in ("strict", "lenient"):
        per_task: dict[str, list] = defaultdict(list)
        per_layer: dict[str, list] = defaultdict(list)
        per_template: dict[str, list] = defaultdict(list)
        failures: dict[str, int] = defaultdict(int)
        failed_records = 0
        extras: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            label = record["label"]
            task = task_key(record)
            desk = int(label["desktop"])
            world = worlds.get(desk)
            trans = transforms.get(record["id"])
            g = grade(record, preds.get(record["id"]), world, tax, mode=mode,
                      transform=trans[0] if trans else None)
            row = (desk, g.score)
            rows_of = _rows_of(task, label)
            if mode == "strict":
                p = chance_of(record, world, tax)
                for key in rows_of:
                    if p is None:
                        chance_unsized[key] += 1
                    else:
                        chance_rows[key].append(p)
            if not g.parsed:
                failed_records += 1
                for key in rows_of:
                    failures[key] += 1
            for key in rows_of:
                per_task[key].append(row)
            if task not in SECONDARY_ROWS:
                # the multiple choice and V3's hinted form are secondary record types:
                # the layers count each headline row once
                per_layer[CAPABILITY_LAYERS.get(task, "other")].append(row)
            if task not in SECONDARY_ROWS:
                per_template[label.get("template_id", "?")].append(row)
            m = g.metrics
            if task == "V3" and label["answer_check"].get("region"):
                # where the machine changed (owner 2026-11-04): beside the action, not in it
                extras[task]["region_iou"].append((desk, float(m.get("region_iou") or 0.0)))
                extras[task]["region_hit"].append((desk, float(bool(m.get("region_hit")))))
            if not label.get("image_undecidable"):
                extras[task]["decidable"].append(row)
            if task == "V2":
                # the count form scores an off-by-one 0.8 (owner 2026-11-04): `exact` is
                # the exact matches alone, over every V2 question (and the prior-wrong row)
                for key in rows_of:
                    extras[key]["exact"].append((desk, float(bool(m.get("exact")))))
                if "count" in label["answer"]:
                    extras[task]["count_exact"].append((desk, float(bool(m.get("exact")))))
                if m.get("cannot_tell"):
                    extras[task]["cannot_tell_accuracy"].append((desk, float(bool(m.get("exact")))))
            if task in ("V6", V6_CONSTRAINED) and "f1" in m:
                extras[task]["precision"].append((desk, float(m["precision"])))
                extras[task]["recall"].append((desk, float(m["recall"])))
                if task == V6_CONSTRAINED:
                    extras[task]["constraint_violation"].append(
                        (desk, float(bool(m.get("constraint_violated")))))
            if task == V6_MC and g.parsed and m.get("blocked") is not None:
                extras[task]["blocked_rate"].append((desk, float(bool(m.get("blocked")))))
            if task in ("V7", V7_CONSTRAINED) and g.parsed and "violation_free" in m:
                extras[task]["violation_free"].append((desk, float(m["violation_free"])))
                extras[task]["required_recall"].append((desk, float(m["required_recall"])))
                extras[task]["goal_reached"].append((desk, float(m["goal_reached"])))
                if "constraint_kept" in m:
                    extras[task]["constraint_kept"].append((desk, float(m["constraint_kept"])))
                if "count_exact" in m:
                    extras[task]["count_exact"].append((desk, float(m["count_exact"])))
                    extras[task]["count_abs_error"].append((desk, float(m["count_abs_error"])))
            if task == "V16" and m.get("localised") is not None:
                extras[task]["localised"].append((desk, float(m["localised"])))
            if task == "V4" and "blockers_f1" in m:
                extras[task]["blockers_f1"].append((desk, float(m["blockers_f1"])))
            if task == "V10" and "bin_correct" in m:
                extras[task]["bin_accuracy"].append((desk, float(m["bin_correct"])))
            if task == "V10" and "started_f1" in m:
                extras[task]["started_f1"].append((desk, float(m["started_f1"])))
        out[mode] = {
            "parse_failure_rate": {t: _r(failures[t] / len(rows))
                                   for t, rows in sorted(per_task.items())},
            "parse_failure_rate_all": _r(failed_records / max(len(records), 1)),
            "by_task": {t: _cell(rows, resamples, seed) for t, rows in sorted(per_task.items())},
            "by_layer": {l: _cell(per_layer[l], resamples, seed)
                         for l in LAYER_ORDER if per_layer.get(l)},
            "by_template": {t: _cell(rows, resamples, seed)
                            for t, rows in sorted(per_template.items())},
            "extra": {t: {k: _cell(v, resamples, seed) for k, v in sorted(d.items())}
                      for t, d in sorted(extras.items())},
        }
    # the in-sample text-only bound (the export's prompt-only classifiers)
    out["shortcut_upper_bound"] = _shortcuts(summary)
    out["shortcut_upper_bound_note"] = ("in-sample upper bound: the best prompt-only "
                                        "classifier measured on the full export, not a model")
    # the two reference rows (W4-C): chance by rule, frequency fitted on train
    out["chance"] = {}
    for key in sorted(set(chance_rows) | set(chance_unsized)):
        values = chance_rows.get(key) or []
        cell = {"n": len(values), "mean": _r(sum(values) / len(values)) if values else None,
                "rule": CHANCE_RULES.get(key, "no rule: not a reference row")}
        if chance_unsized.get(key):
            cell["unsized"] = chance_unsized[key]
        if key == "V10":
            bins = next((r["label"]["answer_check"].get("bins") for r in records
                         if r["prompt"]["task"] == "V10"
                         and r["label"]["answer_check"].get("bins")), None)
            cell["extra"] = {"bin_accuracy": _r(1.0 / len(bins)) if bins else 0.25}
        out["chance"][key] = cell
    out["frequency"] = frequency_rows(records, worlds, tax, resamples=resamples, seed=seed)
    # V6's headline is the open form; the multiple choice rides beside it
    out["headline"] = {"v6_open": out["strict"]["by_task"].get("V6"),
                       "v6_mc": out["strict"]["by_task"].get(V6_MC),
                       "v6_constrained": out["strict"]["by_task"].get(V6_CONSTRAINED),
                       "v7": out["strict"]["by_task"].get("V7"),
                       "v7_constrained": out["strict"]["by_task"].get(V7_CONSTRAINED),
                       "v3": out["strict"]["by_task"].get("V3"),
                       "v3_roi": out["strict"]["by_task"].get(V3_ROI),
                       "v2": out["strict"]["by_task"].get("V2"),
                       "v2_prior_wrong": out["strict"]["by_task"].get(V2_PRIOR_WRONG)}
    if prior_path:
        report = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        out["protocol_prior"] = {t: row.get("accuracy")
                                 for t, row in (report.get("by_task") or {}).items()}
    return out


def _shortcuts(summary: dict) -> dict:
    """The export's text-only upper bound per task: the best of its classifiers."""
    table = {}
    for task, row in (summary.get("shortcuts") or {}).items():
        best = max((row.get(k) or 0.0) for k in ("verb", "class", "verb_class", "template"))
        # the export measures V2's shortcut on every V2 question: that is the
        # `V2` row (on `V2-prior-wrong` the class prior is 0 by design)
        table[task] = {"majority": row.get("majority"), "best_text_only": _r(best)}
    option = summary.get("v6_option_shortcut") or {}
    if option.get("records"):
        # the option-verb shortcut is the multiple choice's, not the open form's
        table.setdefault(V6_MC, {})["option_verb"] = option.get("shortcut")
        table[V6_MC]["random"] = option.get("random")
    return table


def markdown(result: dict, name: str) -> str:
    def fmt(cell):
        if not cell or cell.get("mean") is None:
            return "–"
        lo, hi = cell["ci95"]
        return f"{cell['mean']:.3f} [{lo:.3f}, {hi:.3f}]"

    s, l = result["strict"], result["lenient"]
    short = result.get("shortcut_upper_bound") or {}
    prior = result.get("protocol_prior") or {}
    chance = result.get("chance") or {}
    freq = result.get("frequency") or {}

    def pct(x):
        return "–" if x is None else f"{x:.1%}"

    lines = [f"# {name}", "",
             f"{result['predicted']} of {result['records']} records answered. "
             f"Parse failures: strict {s['parse_failure_rate_all']:.1%}, "
             f"lenient {l['parse_failure_rate_all']:.1%}. "
             "Intervals: 95 %, cluster bootstrap over desktops. "
             "V6 is the open question, graded group-equivalently (the headline); "
             f"{V6_MC} is the multiple choice (secondary). V3 is the action between two "
             f"moments, no hint (the headline; its region is an extra row); {V3_ROI} is the "
             "same step with a hint box drawn where it happened (secondary). V2 is every V2 question "
             f"(the headline); {V2_PRIOR_WRONG} is the questions the class prior "
             "answers wrong (secondary). Reference columns: *chance* is a uniform guess "
             "over the answer space (rules below); *frequency* is the most frequent "
             "gold reply over the file's train desktops given to every record "
             "(strict); *text-only upper bound (in-sample)* is the export's best "
             "prompt-only classifier measured on the full export -- a ceiling for "
             "template leakage, not a model.", "",
             "| task | n | strict | lenient | parse fail (strict / lenient) | chance | "
             "frequency | text-only upper bound (in-sample) | protocol prior |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in sorted(s["by_task"], key=_task_order):
        sc = short.get(task) or {}
        bound = sc.get("option_verb", sc.get("best_text_only"))
        lines.append(f"| {task} | {s['by_task'][task]['n']} | {fmt(s['by_task'][task])} | "
                     f"{fmt(l['by_task'][task])} | "
                     f"{s['parse_failure_rate'][task]:.1%} / "
                     f"{l['parse_failure_rate'][task]:.1%} | "
                     f"{pct((chance.get(task) or {}).get('mean'))} | "
                     f"{fmt(freq.get(task))} | "
                     f"{pct(bound)} | {pct(prior.get(task))} |")
    if chance:
        lines += ["", "| task | chance rule | frequency answer (fit) |", "|---|---|---|"]
        for task in sorted(set(chance) | set(freq), key=_task_order):
            f = freq.get(task) or {}
            answer = f.get("answer")
            shown = (answer if answer is None or len(answer) <= 60 else answer[:57] + "...")
            fit = (f"`{shown}` ({f.get('fit')}, {f.get('answer_share_in_fit', 0):.0%} of "
                   f"{f.get('fit_records')})" if answer is not None else "–")
            rule = (chance.get(task) or {}).get("rule", "–")
            lines.append(f"| {task} | {rule} | {fit} |")
    lines += ["", "| layer | n | strict | lenient |", "|---|---|---|---|"]
    for layer, cell in s["by_layer"].items():
        lines.append(f"| {layer} | {cell['n']} | {fmt(cell)} | {fmt(l['by_layer'][layer])} |")
    lines += ["", "| task | metric | strict | lenient |", "|---|---|---|---|"]
    for task, metrics in s["extra"].items():
        for key, cell in metrics.items():
            other = l["extra"].get(task, {}).get(key)
            lines.append(f"| {task} | {key} | {fmt(cell)} | {fmt(other)} |")
    lines += ["", "| template | n | strict |", "|---|---|---|"]
    for tid, cell in s["by_template"].items():
        lines.append(f"| {tid} | {cell['n']} | {fmt(cell)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--chat", default=None,
                    help="the chat JSONL the model was run on (box transforms of --roi)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--prior", default=None,
                    help="protocol_prior.report.json of scripts/vlm_references.py: "
                         "its accuracy is printed beside each task (the V6 "
                         "multiple-choice ceiling)")
    args = ap.parse_args(argv)
    name = args.name or Path(args.pred).stem
    result = score(args.truth, args.pred, args.chat, resamples=args.resamples,
                   seed=args.seed, prior_path=args.prior)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    (out / f"{name}.score.md").write_text(markdown(result, name), encoding="utf-8")
    print(f"[vlm-score] {name}: strict parse failures "
          f"{result['strict']['parse_failure_rate_all']:.1%}, lenient "
          f"{result['lenient']['parse_failure_rate_all']:.1%} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
