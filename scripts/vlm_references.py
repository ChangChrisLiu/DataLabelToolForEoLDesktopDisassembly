"""The three text references of the VLM evaluation (W3b; minutes 2026-10-31 §3).

``python scripts/vlm_references.py --export <vlm_scan.jsonl> --out <dir>``

Reads one export and its sidecars -- never the database -- and writes:

* ``blind.jsonl`` -- every record with its images taken away: question and
  options only. What a model scores here is template leakage (the export's
  ``shortcuts`` table is the in-sample upper bound of the same thing).
* ``protocol_prior.jsonl`` + ``protocol_prior.predictions.jsonl`` -- V6 / V7 /
  V16 answered from **nothing but the previous demonstrated action**, by a
  bigram over ``(verb, match class)`` fitted leave-one-desktop-out on the other
  desktops' logs (or on ``--train-desktops`` when a split exists). The
  predictions are graded with :mod:`tda.core.export.vlm_check`, and
  ``protocol_prior.report.json`` gives its accuracy by task -- the line a text
  model reading the log would start from. Its next-action accuracy over coarse
  ``(verb, class)`` is reported too, the number the 71.8 % probe measured.
* ``state_oracle.jsonl`` -- every record with each image replaced by the
  structured state table of that moment (instances with state, placement and,
  only where that view has confirmed rows, visibility): the "perfect
  perception" upper bound. Records whose truth rests on a captive screw's
  tightness keep ``label.image_undecidable = true`` so the oracle-vs-VLM gap
  can leave them out.

Each variant keeps the record ids and the label side (``answer_check``
included); only ``prompt`` changes, and ``label.reference`` says which variant
it is.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import LETTERS, World, grade  # noqa: E402

START = ("<start>", "<start>")
#: Tasks the protocol prior answers.
PRIOR_TASKS = ("V6", "V7", "V16")


# --------------------------------------------------------------------------- #
# (i) blind
# --------------------------------------------------------------------------- #
def blind(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["prompt"]["images"] = []
    out["label"]["reference"] = "blind"
    return out


# --------------------------------------------------------------------------- #
# (iii) state oracle
# --------------------------------------------------------------------------- #
def state_table_text(frame: Optional[dict], world: Optional[World]) -> str:
    """One line per instance: wording, state, placement[, visibility]."""
    if frame is None:
        return "(no state table)"
    lines = []
    for key, entry in sorted(frame["state"].items()):
        inst = world.instances.get(key) if world else None
        name = inst.get("label") if inst else key
        if key.startswith("cable:"):
            name = "cable of the " + key.split(":", 1)[1].replace("_", " ")
        bits = [entry.get("state") or "?", entry.get("placement") or "?"]
        if entry.get("visibility"):
            bits.append(entry["visibility"])
        lines.append(f"- {name}: " + ", ".join(bits))
    return "\n".join(lines)


def state_oracle(record: dict, worlds: dict[int, World], frames: dict[str, dict]) -> dict:
    out = copy.deepcopy(record)
    world = worlds.get(IO.desktop_of(record))
    tables = [state_table_text(frames.get(image), world)
              for image in record["prompt"]["images"]]
    out["prompt"]["images"] = []
    out["prompt"]["state_tables"] = tables
    out["label"]["reference"] = "state_oracle"
    out["label"]["image_undecidable"] = bool(record["label"].get("image_undecidable"))
    return out


# --------------------------------------------------------------------------- #
# (ii) protocol prior
# --------------------------------------------------------------------------- #
def _sequence(world: World) -> list[tuple[str, str]]:
    return [(a["verb"], a.get("match_class") or a.get("class") or "?")
            for a in world.log if a.get("changes_state")]


def _coarse(item: tuple[str, str]) -> tuple[str, str]:
    return item[0], item[1].split(".")[0]


class Bigram:
    """``P(next | previous)`` over ``(verb, match class)``, add-one smoothed."""

    def __init__(self, sequences: list[list[tuple[str, str]]], coarse: bool = False):
        self.coarse = coarse
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()
        runs: dict[tuple, list[int]] = defaultdict(list)
        for seq in sequences:
            seq = [_coarse(x) for x in seq] if coarse else seq
            prev = START
            per: Counter = Counter()
            for item in seq:
                self.counts[prev][item] += 1
                self.unigram[item] += 1
                per[item] += 1
                prev = item
            for item, n in per.items():
                runs[item].append(n)
        self.vocab = sorted(self.unigram)
        # how many times one desktop does one kind of action (four cooler screws)
        self.repeats = {k: sorted(v)[len(v) // 2] for k, v in runs.items()}

    def key(self, item: tuple[str, str]) -> tuple[str, str]:
        return _coarse(item) if self.coarse else item

    def prob(self, prev: tuple, item: tuple) -> float:
        prev, item = self.key(prev), self.key(item)
        row = self.counts.get(prev)
        total_uni = sum(self.unigram.values()) or 1
        uni = (self.unigram.get(item, 0) + 1) / (total_uni + len(self.vocab) + 1)
        if not row:
            return uni
        total = sum(row.values())
        return (row.get(item, 0) + uni * 5.0) / (total + 5.0)

    def best(self, prev: tuple, exclude: Optional[set] = None) -> Optional[tuple]:
        options = [v for v in self.vocab if not exclude or v not in exclude]
        if not options:
            return None
        return max(options, key=lambda v: (self.prob(prev, v), v))


def _prev(record: dict) -> tuple[str, str]:
    pa = record["label"]["answer_check"].get("previous_action")
    if not pa:
        return START
    return pa["verb"], pa.get("match_class") or pa.get("class") or "?"


def _phrase(world: World, match_class: str) -> str:
    """The wording of a match class: an instance label without its ordinal."""
    for key, inst in sorted(world.instances.items()):
        if (inst.get("match_class") or inst["class"]) == match_class:
            return re.sub(r"\s+\d+$", "", inst.get("label") or key)
    return match_class.replace("_", " ").replace(".", " ")


def prior_answer(record: dict, model: Bigram, world: World,
                 threshold: float) -> Optional[dict]:
    task = record["prompt"]["task"]
    prev = _prev(record)
    if task == "V6":
        options = record["label"].get("options")
        if options:
            scored = [(model.prob(prev, (o["verb"], world.match_class(o["target"]) or "?")),
                       -i) for i, o in enumerate(options)]
            best = max(range(len(options)), key=lambda i: scored[i])
            return {"choice": LETTERS[best]}
        guess = model.best(prev)
        if guess is None:
            return None
        return {"verb": guess[0], "target": _phrase(world, guess[1])}
    if task == "V7":
        goal_class = world.match_class(record["label"]["answer_check"]["goal"]) or "cpu"
        plan, done = [], Counter()
        current = prev
        for _ in range(20):
            spent = {k for k, n in done.items() if n >= model.repeats.get(k, 1)}
            nxt = model.best(current, spent)
            if nxt is None:
                break
            plan.append({"verb": nxt[0], "target": _phrase(world, nxt[1])})
            done[nxt] += 1
            current = nxt
            if nxt == ("remove", goal_class):
                break
        return {"plan": plan}
    if task == "V16":
        items = [(i["verb"], world.match_class(i["target"]) or "?")
                 for i in record["label"]["answer_check"]["plan"]]
        probs, current = [], prev
        for item in items:
            probs.append(model.prob(current, item))
            current = item
        low = min(range(len(probs)), key=lambda i: probs[i])
        if probs[low] < threshold:
            return {"valid": False, "first_error_step": low + 1}
        return {"valid": True, "first_error_step": None}
    return None


def _v16_min_prob(record: dict, model: Bigram, world: World) -> float:
    items = [(i["verb"], world.match_class(i["target"]) or "?")
             for i in record["label"]["answer_check"]["plan"]]
    current, low = _prev(record), 1.0
    for item in items:
        low = min(low, model.prob(current, item))
        current = item
    return low


def _fit_threshold(records: list[dict], model: Bigram, worlds: dict[int, World]) -> float:
    """The cut on the plan's least likely transition that best splits valid / not."""
    scored = [(_v16_min_prob(r, model, worlds[IO.desktop_of(r)]),
               bool(r["label"]["answer"]["valid"])) for r in records
              if IO.desktop_of(r) in worlds]
    if not scored:
        return 0.0
    cuts = sorted({s for s, _ in scored}) + [1.01]
    best, best_acc = 0.0, -1.0
    for cut in cuts:
        acc = sum((s >= cut) == valid for s, valid in scored) / len(scored)
        if acc > best_acc:
            best, best_acc = cut, acc
    return best


def protocol_prior(records: list[dict], worlds: dict[int, World], tax,
                   train: Optional[set[int]] = None) -> tuple[list[dict], list[dict], dict]:
    """``(variant records, predictions, report)``."""
    desks = sorted(worlds)
    variant, predictions = [], []
    v16 = [r for r in records if r["prompt"]["task"] == "V16"]
    models: dict[int, tuple[Bigram, Bigram, float]] = {}

    def fold(desktop: int) -> tuple[Bigram, Bigram, float]:
        if desktop not in models:
            pool = sorted(train) if train else [d for d in desks if d != desktop]
            pool = [d for d in pool if d != desktop and d in worlds]
            seqs = [_sequence(worlds[d]) for d in pool]
            fine, coarse = Bigram(seqs), Bigram(seqs, coarse=True)
            cut = _fit_threshold([r for r in v16 if IO.desktop_of(r) in set(pool)],
                                 fine, worlds)
            models[desktop] = (fine, coarse, cut)
        return models[desktop]

    by_task: dict[str, list[float]] = defaultdict(list)
    coarse_hits: list[float] = []
    for record in records:
        task = record["prompt"]["task"]
        if task not in PRIOR_TASKS:
            continue
        desktop = IO.desktop_of(record)
        world = worlds.get(desktop)
        if world is None:
            continue
        fine, coarse, cut = fold(desktop)
        answer = prior_answer(record, fine, world, cut)
        raw = json.dumps(answer, ensure_ascii=False) if answer is not None else ""
        predictions.append({"id": record["id"], "raw_text": raw})
        out = copy.deepcopy(record)
        out["prompt"] = {"task": task, "images": [], "question": record["prompt"]["question"],
                         "previous_action": record["label"]["answer_check"].get(
                             "previous_action")}
        if record["prompt"].get("options"):
            out["prompt"]["options"] = record["prompt"]["options"]
        out["label"]["reference"] = "protocol_prior"
        variant.append(out)
        mc = task == "V6" and bool(record["prompt"].get("options"))
        by_task["V6-mc" if mc else task].append(grade(record, raw, world, tax).score)
        if task == "V6" and not mc:
            ref = record["label"]["answer_check"]["reference"]
            want = (ref[0], (world.match_class(ref[1]) or "?").split(".")[0])
            coarse_hits.append(float(coarse.best(_prev(record)) == want))
    report = {
        "by_task": {t: {"records": len(v), "accuracy": round(sum(v) / len(v), 4)}
                    for t, v in sorted(by_task.items()) if v},
        "v6_next_verb_class_coarse": (round(sum(coarse_hits) / len(coarse_hits), 4)
                                      if coarse_hits else None),
        "fit": "train desktops" if train else "leave-one-desktop-out",
        "desktops": len(desks),
    }
    return variant, predictions, report


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--export", required=True, help="an export-vlm JSONL")
    ap.add_argument("--out", required=True, help="directory for the three variants")
    ap.add_argument("--train-desktops", default=None,
                    help="e.g. 1-40: fit the protocol prior on these (default: "
                         "leave one desktop out)")
    args = ap.parse_args(argv)
    from tda.core.taxonomy import load_taxonomy

    records = list(IO.read_jsonl(args.export))
    worlds, frames = IO.load_oracle(args.export)
    out = Path(args.out)
    n_blind = IO.write_jsonl(out / "blind.jsonl", (blind(r) for r in records))
    n_oracle = IO.write_jsonl(out / "state_oracle.jsonl",
                              (state_oracle(r, worlds, frames) for r in records))
    train = _parse_range(args.train_desktops) if args.train_desktops else None
    variant, predictions, report = protocol_prior(records, worlds, load_taxonomy(), train)
    IO.write_jsonl(out / "protocol_prior.jsonl", variant)
    IO.write_jsonl(out / "protocol_prior.predictions.jsonl", predictions)
    (out / "protocol_prior.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[vlm-references] blind {n_blind}, state oracle {n_oracle}, "
          f"protocol prior {len(variant)} -> {out}")
    for task, row in report["by_task"].items():
        print(f"[vlm-references] protocol prior {task}: {row['accuracy']:.1%} "
              f"({row['records']} records)")
    if report["v6_next_verb_class_coarse"] is not None:
        print(f"[vlm-references] protocol prior next (verb, class): "
              f"{report['v6_next_verb_class_coarse']:.1%}")
    return 0


def _parse_range(text: str) -> set[int]:
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
