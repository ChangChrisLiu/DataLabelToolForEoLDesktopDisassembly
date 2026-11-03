"""The reference lines of the VLM evaluation (W3b; minutes 2026-10-31 §3; W4-C).

``python scripts/vlm_references.py --export <vlm_scan.jsonl> --out <dir>
[--assets <dir> --image-root <images> [--masks <masks.jsonl> | --coco-root <scanpack>]]``

Reads one export and its sidecars -- never the database -- and writes:

* ``blind.jsonl`` -- every record with its images taken away: question and
  options only, the mark wording of the 2026-11-03 redesign removed ("the RAM
  module outlined in red" -> "the RAM module", :func:`unmark`); a question that
  names a part by its mark tag alone ("part A") is left out (``mark_tag``, also
  in the state oracle). What a model scores here is template leakage (the export's
  ``shortcuts`` table is the in-sample upper bound of the same thing).
* ``protocol_prior.jsonl`` + ``protocol_prior.predictions.jsonl`` -- V6 / V7 /
  V16 answered from **nothing but the previous demonstrated action**, by a
  bigram over ``(verb, match class)`` fitted leave-one-desktop-out on the other
  desktops' logs (or on ``--train-desktops`` when a split exists). The
  predictions are graded with :mod:`tda.core.export.vlm_check`, and
  ``protocol_prior.report.json`` gives its accuracy by task -- the line a text
  model reading the log would start from. Its next-action accuracy over coarse
  ``(verb, class)`` is reported too, the number the 71.8 % probe measured.
* ``state_oracle.jsonl`` -- every record **except V14** with its images replaced
  by the structured state table of each moment (instances with state, placement
  and, only where that view has confirmed rows, visibility): the "perfect
  perception" upper bound. One table per image in a camera unit, **one per
  moment** in the ``multi`` unit (its cameras of one step share the scanner's
  table; V3 / V12 have two, before and after -- :func:`moment_images`). A screw is shown as present / removed, never as
  tight or loose -- no picture shows that either. Records whose truth rests on
  a captive screw's tightness keep ``label.image_undecidable = true`` so the
  oracle-vs-VLM gap can leave them out. V14 asks whether *this view* can settle
  a part's state: the table's visibility column is that answer, so the oracle
  has no V14 items (W4-C) and the summary says how many were left out.
* ``mask_overlay.jsonl`` (+ ``.images_manifest.jsonl``, with ``--assets``) --
  every eligible record with its frames replaced by a copy on which the
  confirmed visible masks are painted in class colours (:mod:`vlm_overlay`):
  the "minimal set" perception reference. Eligible: every frame confirmed and
  carrying masks, task not V8 (the box is the answer); in the ``multi`` unit only
  the scanner frames are painted (and must be confirmed with masks), the side
  cameras' pictures stay as they are. Needs ``--image-root``
  and the masks -- ``<export>.masks.jsonl`` beside the export, ``--masks``, or
  ``--coco-root`` (the scanner pack's COCO). Without ``--assets`` the variant
  is not written and the summary says so.

The protocol prior refuses (exit 2, nothing written) when a desktop has no
other desktop's log to be fitted on.

Each variant keeps the record ids and the label side (``answer_check``
included); only ``prompt`` changes, and ``label.reference`` says which variant
it is. ``references.summary.json`` in ``--out`` counts every variant.
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
from tda.core.export.vlm_tasks import SCREW_PRESENCE  # noqa: E402

START = ("<start>", "<start>")


class PriorRefused(RuntimeError):
    """The protocol prior has no other desktop's data to be fitted on."""
#: Tasks the protocol prior answers.
PRIOR_TASKS = ("V6", "V7", "V16")


# --------------------------------------------------------------------------- #
# (i) blind
# --------------------------------------------------------------------------- #
#: The mark wording of the redesign of 2026-11-03 ("the RAM module outlined in red"):
#: meaningless without the picture, so the text-only variants drop it.
MARK_PHRASE = re.compile(r"\s*\(?\b(?:which (?:is|are) |that (?:is|are) )?outlined in red\b\)?")
#: A part named by its tag alone ("part A", "parts A and B"): no text can stand in for it.
MARK_TAG = re.compile(r"\bparts?\s+[A-Z]\b")
#: Why a text-only variant leaves a record out.
UNMARK_SKIP = "mark_tag"


def unmark(record: dict) -> Optional[dict]:
    """The record for a variant without pictures (blind, state oracle), or ``None``.

    The converter drops the images and the marks of such a variant, but the question
    would still say "the RAM module outlined in red": the phrase is removed ("the RAM
    module" -- the class phrase the mark qualified), in the question and the options,
    and ``prompt.marks`` / ``prompt.image_legend`` go. A question that names a part by
    its tag only ("part A") cannot stand without the mark and is left out
    (:data:`UNMARK_SKIP`). A record without marks is returned as a plain copy.
    """
    out = copy.deepcopy(record)
    prompt = out["prompt"]
    marked = bool(prompt.pop("marks", None))
    prompt.pop("image_legend", None)
    texts = [prompt.get("question") or ""] + [str(o.get("target_label") or "")
                                              for o in prompt.get("options") or []]
    if marked and any(MARK_TAG.search(t) for t in texts):
        return None
    prompt["question"] = MARK_PHRASE.sub("", prompt.get("question") or "")
    for option in prompt.get("options") or []:
        if option.get("target_label"):
            option["target_label"] = MARK_PHRASE.sub("", option["target_label"])
    return out


def blind(record: dict) -> Optional[dict]:
    """The record without its pictures (and without the mark wording, :func:`unmark`),
    or ``None`` when its question names a part by its tag only."""
    out = unmark(record)
    if out is None:
        return None
    out["prompt"]["images"] = []
    out["label"]["reference"] = "blind"
    return out


def blind_rows(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """``(blind records, left out per reason)``."""
    rows, skipped = [], Counter()
    for record in records:
        row = blind(record)
        if row is None:
            skipped[UNMARK_SKIP] += 1
        else:
            rows.append(row)
    return rows, dict(skipped)


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
        state = entry.get("state") or "?"
        if inst and inst.get("class") == "screw":
            # a picture shows a screw's presence, never its tightness: neither
            # does the "perfect perception" table (review M)
            state = SCREW_PRESENCE.get(state, state)
        bits = [state, entry.get("placement") or "?"]
        if entry.get("visibility"):
            bits.append(entry["visibility"])
        lines.append(f"- {name}: " + ", ".join(bits))
    return "\n".join(lines)


#: The export unit whose records show one moment from every camera (the redesign of
#: 2026-11-03); its state oracle gives one table per moment, not one per picture.
MULTI_VIEW = "multi"


def moment_images(record: dict, frames: dict[str, dict]) -> list[str]:
    """The image standing for each moment a record shows, in image order.

    A ``multi`` record shows each moment from several cameras (the scanner first), so
    its moments are its images grouped by step (``label.image_steps``, else the
    oracle's ``step`` of each image) and each is represented by its first image -- the
    scanner's, whose table alone carries confirmed visibility. Every other record keeps
    one table per image: V15 asks whether two pictures are the same moment, and merging
    their tables would answer it.
    """
    images = list(record["prompt"].get("images") or [])
    if record["label"].get("view") != MULTI_VIEW:
        return images
    steps = record["label"].get("image_steps")
    if not steps or len(steps) != len(images):
        steps = [(frames.get(image) or {}).get("step") for image in images]
    out: list[str] = []
    seen: set = set()
    for image, step in zip(images, steps):
        key = image if step is None else step
        if key in seen:
            continue
        seen.add(key)
        out.append(image)
    return out


def state_oracle(record: dict, worlds: dict[int, World], frames: dict[str, dict]
                 ) -> Optional[dict]:
    """The record with its pictures replaced by state tables (one per moment), the mark
    wording removed (:func:`unmark`); ``None`` when its question needs the mark."""
    out = unmark(record)
    if out is None:
        return None
    world = worlds.get(IO.desktop_of(record))
    tables = [state_table_text(frames.get(image), world)
              for image in moment_images(record, frames)]
    out["prompt"]["images"] = []
    out["prompt"]["state_tables"] = tables
    out["label"]["reference"] = "state_oracle"
    out["label"]["image_undecidable"] = bool(record["label"].get("image_undecidable"))
    return out


#: Tasks the state oracle does not answer: V14 asks whether the view can settle
#: a part's state, and the table's visibility column is exactly that (W4-C).
ORACLE_SKIPS = ("V14",)


def state_oracle_rows(records: list[dict], worlds: dict[int, World],
                      frames: dict[str, dict]) -> tuple[list[dict], dict[str, int]]:
    """``(oracle records, skipped per task)``: every record but V14's (and but those
    whose question names a part by its mark tag only, counted as ``mark_tag``)."""
    rows, skipped = [], Counter()
    for record in records:
        task = record["prompt"]["task"]
        if task in ORACLE_SKIPS:
            skipped[task] += 1
            continue
        row = state_oracle(record, worlds, frames)
        if row is None:
            skipped[UNMARK_SKIP] += 1
            continue
        rows.append(row)
    return rows, dict(skipped)


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
            if not any(seqs):
                # a prior fitted on nothing answers uniformly, and a number
                # from it would be reported as the protocol's (review M)
                raise PriorRefused(
                    f"protocol prior: no other desktop's log to fit D{desktop} on "
                    f"(training pool {pool or 'empty'}); export more desktops or "
                    f"pass --train-desktops with desktops other than D{desktop}")
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
    ap.add_argument("--assets", default=None,
                    help="also write the mask_overlay variant; the painted frames go "
                         "here as <view>/D<nn>/s<nn>.png")
    ap.add_argument("--image-root", default=None,
                    help="the frames the manifest names (needed with --assets)")
    ap.add_argument("--masks", default=None,
                    help="the masks sidecar (default: <export>.masks.jsonl beside the export)")
    ap.add_argument("--coco-root", default=None,
                    help="build the masks from a scanner pack's COCO when there is no sidecar")
    ap.add_argument("--force", action="store_true", help="repaint overlays that already exist")
    args = ap.parse_args(argv)
    from vlm_eval_set import refuse_old_eval

    refuse_old_eval(args.export, args.out, args.assets)
    from tda.core.taxonomy import load_taxonomy

    records = list(IO.read_jsonl(args.export))
    worlds, frames = IO.load_oracle(args.export)
    out = Path(args.out)
    train = _parse_range(args.train_desktops) if args.train_desktops else None
    try:
        variant, predictions, report = protocol_prior(records, worlds, load_taxonomy(),
                                                      train)
    except PriorRefused as exc:
        print(f"[vlm-references] refused: {exc}")
        return 2
    if args.assets and not args.image_root:
        print("[vlm-references] refused: --assets needs --image-root (the frames to paint)")
        return 2
    blind_records, blind_skipped = blind_rows(records)
    n_blind = IO.write_jsonl(out / "blind.jsonl", blind_records)
    oracle_rows, oracle_skipped = state_oracle_rows(records, worlds, frames)
    n_oracle = IO.write_jsonl(out / "state_oracle.jsonl", oracle_rows)
    IO.write_jsonl(out / "protocol_prior.jsonl", variant)
    IO.write_jsonl(out / "protocol_prior.predictions.jsonl", predictions)
    (out / "protocol_prior.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    summary = {
        "blind": {"records": n_blind, **({"skipped": blind_skipped} if blind_skipped else {})},
        "state_oracle": {"records": n_oracle, "skipped": oracle_skipped,
                         "skipped_why": "V14 asks whether the view can settle the state; "
                                        "the table's visibility column is that answer"
                                        + ("; mark_tag: the question names a part by its "
                                           "mark only, which no text can stand in for"
                                           if UNMARK_SKIP in oracle_skipped else "")},
        "protocol_prior": {"records": len(variant), "report": report},
    }
    skipped_text = ", ".join(f"{t} {n} skipped" for t, n in oracle_skipped.items()) or "nothing skipped"
    print(f"[vlm-references] blind {n_blind}, state oracle {n_oracle} ({skipped_text}: its "
          f"visibility column would be the answer), protocol prior {len(variant)} -> {out}")
    for task, row in report["by_task"].items():
        print(f"[vlm-references] protocol prior {task}: {row['accuracy']:.1%} "
              f"({row['records']} records)")
    if report["v6_next_verb_class_coarse"] is not None:
        print(f"[vlm-references] protocol prior next (verb, class): "
              f"{report['v6_next_verb_class_coarse']:.1%}")
    if args.assets:
        import vlm_overlay

        try:
            summary["mask_overlay"] = vlm_overlay.render_export(
                args.export, out / "mask_overlay.jsonl", args.image_root, args.assets,
                masks_path=args.masks, coco_root=args.coco_root, force=args.force,
                log=lambda line: print(line.replace("[vlm-overlay]", "[vlm-references]")))
        except vlm_overlay.OverlayError as exc:
            print(f"[vlm-references] refused: {exc}")
            return 2
    else:
        summary["mask_overlay"] = {"records": 0, "skipped_why": "not asked for: pass --assets "
                                                                 "and --image-root"}
        print("[vlm-references] mask overlay not written (pass --assets <dir> --image-root <images>)")
    (out / "references.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
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
