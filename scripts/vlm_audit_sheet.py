"""The fillable human-audit sheet of the evaluation pack (W4-E).

``python tools/vlm_audit_sheet.py --records eval_v1/audit/audit.jsonl
--export eval_v1/scan.jsonl eval_v1/oak1.jsonl eval_v1/oak2.jsonl eval_v1/rs.jsonl
--image-root images --out eval_v1/audit [--batch 50] [--name audit]``

``scripts/vlm_sample.py --purpose audit`` draws the 300 items and writes a
Markdown table; nothing collected the answers. This builds, from that sample
and the pack's own files, what a reviewer fills in:

* ``<name>.sheet.b01.html``, ``b02`` ... -- one **static HTML page per batch**
  of ``--batch`` items (plain HTML + inline JS + inline CSS; no server, no
  network, no CDN script). Every item shows the frame (both frames of a
  two-image question: V3, V12, V15) **as the model gets it** -- only its red
  dashed marks (and V3's yellow hint box), drawn the converter's way --, the
  question with its lettered options, the gold answer in words and as the raw
  JSON, its evidence (a separate reviewer layer, hidden until "show reviewer
  boxes / 显示审核框" is ticked; the model never sees it) and the rationale, then three radio buttons
  正确 / 错误 / 存疑 (correct / wrong / doubtful) and a free-text note. The
  page keeps what is typed in the browser's ``localStorage`` (a reload loses
  nothing) and its **export button downloads a CSV** of the answers; a
  reviewer who prefers a spreadsheet fills in
* ``<name>.sheet.csv`` -- the same rows as an empty CSV skeleton.

Image paths are **relative to the page** (``../../images/<view>/D<nn>/s<nn>.png``
for a page in ``eval_v1/audit/``), so the pack can be zipped and sent whole;
the page itself opens from the file system. Parts are named without an ordinal
(:func:`display_name`, owner 2026-11-04): "part A" for a part the question
outlines, a group by its label, a part alone of its class by its class, any other
"one of the N <class>" -- never ``screw.motherboard.07``, never "Motherboard screw 7". The sheet is deterministic: no clock, no machine path, so two pack
builds are byte-identical. ``scripts/vlm_audit_summary.py`` reads the filled
CSVs back.

The same CSS, image figure and CSV machinery serve the human-baseline form
(``scripts/vlm_human_sheet.py``), which imports this module.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from vlm_glossary import glossary_text, is_presence  # noqa: E402
import vlm_to_chat as CHAT  # noqa: E402
from vlm_to_chat import (VIEW_NAMES, image_legend, legend_line, marks_line,  # noqa: E402
                         shown_hints, shown_marks)
from tda.core.export.vlm_check import LETTERS, World, format_key, reference_option  # noqa: E402

__all__ = ["CSV_COLUMNS", "VERDICTS", "boxes_of", "build", "csv_skeleton", "frames_html",
           "display_name", "gold_lines", "item_html", "load_context", "main", "marks_of",
           "name_of", "page",
           "rationale_lines", "rel_image"]

#: Columns of the skeleton and of the page's export, in this order.
CSV_COLUMNS = ("id", "task", "readable_id", "desktop", "view", "step", "batch", "verdict",
               "note", "reviewer", "answered_at")
#: ``value -> (Chinese label, English gloss)`` of the three radio buttons.
VERDICTS = (("correct", "正确", "the answer is right for the picture(s)"),
            ("wrong", "错误", "the answer is wrong, or the question cannot be asked of this picture"),
            ("doubtful", "存疑", "cannot be decided from the picture(s), or only partly right"))
DEFAULT_BATCH = 50
_BIN_LABELS = {"0-25": "0-25 %", "25-50": "25-50 %", "50-75": "50-75 %", "75-100": "75-100 %"}


# --------------------------------------------------------------------------- #
# reading the pack
# --------------------------------------------------------------------------- #
def load_context(exports: Sequence) -> tuple[dict[str, dict], dict[int, World]]:
    """``(image id -> manifest row, desktop -> World)`` over several truth files.

    A desktop's world is the same in every view's oracle; the first one read
    is kept.
    """
    manifest: dict[str, dict] = {}
    worlds: dict[int, World] = {}
    for path in exports:
        manifest.update(IO.load_manifest(path))
        found, _ = IO.load_oracle(path)
        for desk, world in found.items():
            worlds.setdefault(desk, world)
    return manifest, worlds


def display_name(key: Optional[str], world: Optional[World], record: Optional[dict] = None,
                 *, short: bool = False) -> str:
    """How the sheet names a part a person reads -- **no ordinal** (owner 2026-11-04).

    A part a record outlines under a tag of its own is "part A"; a group is its group label
    (with its tag where it is outlined, "CPU cooler screws (parts B)"); any other part is the
    oracle's ordinal-free name (:meth:`World.plain_name`: its class when it is the only one,
    "one of the 4 RAM modules" when it is not). ``short`` (a box label: the box already says
    which one) is the class alone. Never an instance key, never "CPU 1".
    """
    if key is None:
        return "-"
    key = str(key)
    tags = ((record or {}).get("label") or {}).get("mark_names") or {}
    if world is not None and world.is_group(key):
        label = world.plain_name(key)
        members = set(world.members(key))
        tag = next((t for t, e in sorted(tags.items())
                    if len(e.get("parts") or []) > 1 and set(e["parts"]) <= members), None)
        return f"{label} (parts {tag})" if tag and not short else label
    if not short:
        tag = next((t for t, e in sorted(tags.items()) if (e.get("parts") or []) == [key]), None)
        if tag:
            return f"part {tag}"
    if world is None:
        # no oracle: the key's words, its ordinal dropped ("cpu_cooler.01" -> "cpu cooler")
        return " ".join(p.replace("_", " ") for p in key.replace(":", ".").split(".")
                        if p and not p.isdigit()) or key
    if short and key in world.instances:
        inst = world.instances[key]
        return re.sub(r"\s+\d+$", "", str(inst.get("label") or key)) if inst.get("label") \
            else world.plain_name(key)
    return world.plain_name(key)


def name_of(key: Optional[str], world: Optional[World]) -> str:
    """The oracle's label of an instance key (``Motherboard screw 7``), else the key -- the
    **grader's** wording (a reply naming it grounds back to the key), for the human-baseline
    self-test; nothing a reviewer reads uses it (:func:`display_name`)."""
    if key is None:
        return "-"
    if world is not None:
        inst = world.instances.get(str(key))
        if inst and inst.get("label"):
            return str(inst["label"])
        group = (getattr(world, "groups", None) or {}).get(str(key))  # a group id (class@anchor)
        if group and group.get("label"):
            return f"{group['label']} ({len(group.get('members') or [])})"
    return str(key)


def rel_image(image_root, rel_path: str, out_dir) -> str:
    """The ``src`` of a frame: its path relative to the page's directory, POSIX."""
    target = Path(image_root) / rel_path
    return Path(os.path.relpath(str(target), str(out_dir))).as_posix()


# --------------------------------------------------------------------------- #
# the gold answer in words
# --------------------------------------------------------------------------- #
def _action(item: dict, world, record: Optional[dict] = None) -> str:
    return f"{item.get('verb', '?')} {display_name(item.get('target'), world, record)}"


def _yes(value) -> str:
    return "yes" if value else "no"


def gold_lines(record: dict, world: Optional[World] = None) -> list[tuple[str, str]]:
    """``[(field, text)]`` -- the gold answer of one record, readable by a person."""
    label = record["label"]
    answer = label["answer"]
    check = label.get("answer_check") or {}
    key = format_key(record)

    def name(k):
        return display_name(k, world, record)
    task = record["prompt"]["task"]
    out: list[tuple[str, str]] = []
    if key == "V1":
        comps = answer.get("components") or []
        out.append(("visible parts", f"{len(comps)}"))
        for c in comps:
            out.append((c.get("class", "?"),
                        f"{name(c.get('instance'))} box {c.get('bbox')}"))
    elif key == "V2.count":
        out.append(("count", str(answer.get("count"))))
        out.append(("screws", f"{check.get('role', '?')} screws still {check.get('state', 'present')}"))
    elif key == "V2.state":
        out.append(("state", str(answer.get("state"))))
        if check.get("allowed"):
            out.append(("allowed", ", ".join(check["allowed"])))
    elif task == "V3" and check.get("type") != "action_set":
        out.append(("action", f"{answer.get('verb')} {name(answer.get('target_instance'))}"))
        out.append(("class", str(answer.get("target_class"))))
        out.append(("tool", str(answer.get("tool"))))
        if label.get("v3_form"):
            out.append(("form", "free (no hint; also asks the region)" if label["v3_form"] == "free"
                        else "roi (a yellow dashed hint box is drawn)"))
        if check.get("region"):
            out.append(("region (scanner)", str(check["region"])))
        if check.get("hint_region"):
            out.append(("hint box (scanner)", str(check["hint_region"])))
        # no "also accepted" line: the part acted on is the only answer (owner 2026-11-03),
        # whatever an older record's `accept_targets` lists beside it
    elif task == "V4":
        out.append(("feasible", _yes(answer.get("feasible"))))
        out.append(("action", f"{check.get('verb')} {name(check.get('target'))}"))
        if answer.get("blockers"):
            out.append(("blockers", "; ".join(name(b) for b in answer["blockers"])))
        if answer.get("reason"):
            out.append(("reason", str(answer["reason"])))
        out.append(("truth source", str(label.get("truth_source"))))
    elif key == "V6.options":
        options = label.get("options") or []
        index = reference_option(label)
        letter = LETTERS[index] if index is not None else "?"
        right = options[index] if index is not None else {}
        out.append(("choice", f"{letter}: {right.get('say_verb', right.get('verb', '?'))} "
                              f"{right.get('target_label', '?')}"))
        kinds = ", ".join(f"{LETTERS[i]} {o.get('kind', '?')}" for i, o in enumerate(options))
        if kinds:
            out.append(("option kinds", kinds))
    elif key in ("V6.open", "V3.span") and check.get("type") in ("frontier", "action_set"):
        # every action possible now, by group (owner 2026-11-04)
        if check.get("form") == "constrained":
            out.append(("rule", str((check.get("constraint") or {}).get("rule"))))
        groups = {g["key"]: g for g in [*(check.get("ruled_groups") or []),
                                        *(check.get("groups") or [])]}
        for i, key_ in enumerate(check.get("gold_groups") or [], start=1):
            g = groups.get(key_) or {}
            hidden = " (a cable hides some)" if g.get("hidden_by_cable") else ""
            out.append((f"possible {i}", f"{g.get('phrase', key_)}{hidden}"))
        for key_ in check.get("forbidden_groups") or []:
            out.append(("ruled out", str((groups.get(key_) or {}).get("phrase", key_))))
        blocked = [g.get("phrase", g.get("key")) for g in check.get("blocked_groups") or []]
        if blocked:
            out.append(("not yet", "; ".join(blocked[:6]) + (" ..." if len(blocked) > 6 else "")))
    elif key == "V6.open":
        out.append(("next action", _action(answer, world, record)))
        eq = check.get("equivalent_actions") or []
        if len(eq) > 1:
            out.append(("equivalent", "; ".join(_action(a, world, record) for a in eq)))
        if check.get("blocked_actions"):
            out.append(("blocked", "; ".join(_action(a, world, record) for a in check["blocked_actions"][:8])
                        + (" ..." if len(check["blocked_actions"]) > 8 else "")))
    elif task == "V7":
        plan = answer.get("plan") or []
        if check.get("constraint"):
            out.append(("rule", str(check["constraint"].get("rule"))))
        if check.get("count"):
            out.append(("connectors to unplug", str(check["count"].get("present"))))
        out.append(("plan", f"{len(plan)} steps to remove {name(answer.get('goal'))}"))
        for i, a in enumerate(plan, start=1):
            out.append((f"{i}", _action(a, world, record)))
        if check.get("required_actions"):
            out.append(("required", "; ".join(
                f"{name(r.get('target'))} -> {'/'.join(r.get('states') or [])}"
                for r in check["required_actions"])))
    elif task == "V8" and "components" in answer:  # every visible part of one class / group
        comps = answer.get("components") or []
        out.append(("visible parts", f"{len(comps)} ({check.get('class_filter', '?')})"))
        for c in comps:
            out.append((c.get("class", "?"),
                        f"{name(c.get('instance'))} box {c.get('bbox')}"))
    elif task == "V8":
        out.append(("part", name(answer.get("instance"))))
        out.append(("box", str(answer.get("bbox"))))
    elif task == "V10":
        done = answer.get("done") or []
        out.append(("progress", f"{_BIN_LABELS.get(str(answer.get('progress_bin')), answer.get('progress_bin'))} "
                                f"({check.get('done_actions', len(done))} of {check.get('total_actions', '?')} actions done)"))
        for i, a in enumerate(done, start=1):
            out.append((f"done {i}", _action(a, world, record)))
        for i, a in enumerate(answer.get("started") or [], start=1):
            out.append((f"started {i}", _action(a, world, record)))
    elif task == "V12":
        out.append(("changed", _yes(answer.get("changed"))))
        for ev in answer.get("events") or []:
            out.append(("event", f"{name(ev.get('target'))}: {ev.get('old')} -> {ev.get('new')}"))
        for ev in check.get("hidden_events") or []:
            out.append(("hidden (no visible change)",
                        f"{name(ev.get('target'))}: {ev.get('old')} -> {ev.get('new')}"))
        if label.get("negative"):
            out.append(("negative kind", str(label["negative"])))
    elif key == "V15.moment":
        out.append(("same moment", _yes(answer.get("same_moment"))))
        views = label.get("views") or []
        if views:
            out.append(("views", f"{views[0]} step {label.get('step')} vs {views[-1]} "
                                 f"step {check.get('other_step', '?')}"))
    elif key == "V15.cross":
        out.append(("state", str(answer.get("state"))))
        out.append(("best view", str(answer.get("best_view"))))
    elif task == "V16":
        out.append(("valid", _yes(answer.get("valid"))))
        index = answer.get("first_error_index")
        if index is not None:
            out.append(("first impossible step", f"{int(index) + 1}"))
        if answer.get("reason"):
            out.append(("reason", str(answer["reason"])))
        out.append(("truth source", str(label.get("truth_source"))))
    else:  # pragma: no cover - a task this sheet does not know
        out.append(("answer", json.dumps(answer, ensure_ascii=False)))
    if label.get("image_undecidable"):
        out.append(("note", "image_undecidable: the truth rests on a captive screw's tightness"))
    return out


def rationale_lines(record: dict, world: Optional[World] = None) -> list[str]:
    """The rationale, readable: ``rationale.text`` (the reason in what the picture shows, owner
    2026-11-04 -- no graph vocabulary) where the record has one; else the chain, one line per
    step, every part named without an ordinal."""
    rationale = record["label"].get("rationale") or {}
    if rationale.get("text"):
        return [str(rationale["text"])]
    steps = rationale.get("steps") or []
    out = []
    for s in steps:
        op = s.get("op", "?")
        bits = [op]
        if s.get("target"):
            bits.append(display_name(s["target"], world, record))
        if s.get("value") is not None:
            bits.append(f"= {s['value']}")
        if s.get("frames"):
            bits.append(f"frames {s['frames']}" + (f" ({s['view']})" if s.get("view") else ""))
        if s.get("required"):
            bits.append(f"requires {s['required']}, actual {s.get('actual')}, "
                        f"satisfied {s.get('satisfied')}")
        if s.get("reason"):
            bits.append(str(s["reason"]))
        ev = s.get("evidence") or {}
        if ev.get("bbox"):
            bits.append(f"[box {ev['bbox']}, {ev.get('visibility', '')}]")
        out.append(" ".join(str(b) for b in bits))
    return out


def boxes_of(record: dict, world: Optional[World] = None) -> list[dict]:
    """The reviewer's boxes: ``[{"label", "bbox", "kind"}]`` -- evidence, V1 parts, V8 answer.
    Labelled by class (:func:`display_name` ``short``): never an ordinal, and the model never
    sees them (the page draws them on their own layer, hidden by default)."""
    label = record["label"]
    out: list[dict] = []
    seen: set[tuple] = set()

    def name(k):
        return display_name(k, world, record, short=True)

    def add(text, bbox, kind):
        if not bbox or len(bbox) != 4:
            return
        key = tuple(float(v) for v in bbox)
        if key in seen:
            return
        seen.add(key)
        out.append({"label": text, "bbox": [float(v) for v in bbox], "kind": kind})

    if record["prompt"]["task"] == "V1":
        for c in label["answer"].get("components") or []:
            add(name(c.get("instance")), c.get("bbox"), "answer")
    if record["prompt"]["task"] == "V8":
        for c in label["answer"].get("components") or []:
            add(name(c.get("instance")), c.get("bbox"), "answer")
        add(name(label["answer"].get("instance")), label["answer"].get("bbox"), "answer")
    ev = label.get("evidence") or {}
    for key, bbox in (ev.get("bboxes") or {}).items():
        add(name(key), bbox, "evidence")
    return out


def marks_of(record: dict, index: int) -> list[dict]:
    """What the model sees drawn on image ``index``: the ``prompt.marks`` (``kind = "mark"``,
    the tag as label; red, dashed) and V3's ``prompt.hint_regions`` (``kind = "hint"``, no
    label; yellow, dashed). None on V1 / V8."""
    marks = [{"label": str(m.get("tag") or ""), "bbox": [float(v) for v in m["bbox"]],
              "kind": "mark"}
             for m in shown_marks(record)
             if int(m.get("image", 0)) == index and m.get("bbox") and len(m["bbox"]) == 4]
    hints = [{"label": "", "bbox": [float(v) for v in h["bbox"]], "kind": "hint"}
             for h in shown_hints(record)
             if int(h.get("image", 0)) == index and len(h["bbox"]) == 4]
    return marks + hints


def _image_views(record: dict, manifest: dict) -> list[Optional[str]]:
    images = record["prompt"].get("images") or []
    legend = image_legend(record)
    if len(legend) == len(images):
        return [e.get("view") for e in legend]
    return [(manifest.get(i) or {}).get("view") for i in images]


def frames_html(record: dict, manifest: dict, image_root, out_dir, *, evidence: list[dict],
                marks: bool = True, where: bool = True) -> str:
    """The record's images: one row, or a before row and an after row (V3 / V12 multi-view).

    **The model's view first** (owner 2026-11-04): each figure shows exactly what the model
    gets -- the picture and its own marks (red dashed, tagged; V3's yellow dashed hint), drawn
    the way the converter draws them. The ``evidence`` boxes -- every image of a single-view
    record, the scanner images of a multi-view one (the evidence is read off the scanner
    frame) -- go on a separate reviewer layer the page hides until asked. ``where=False``
    keeps the step and the desktop out of the captions (the human form).
    """
    label = record["label"]
    images = record["prompt"].get("images") or []
    legend = image_legend(record)
    views = _image_views(record, manifest)
    multi = bool(legend)
    figs: list[tuple[str, str]] = []
    for i, image_id in enumerate(images):
        entry = manifest.get(image_id) or {}
        moment = legend[i].get("moment") if multi else None
        place = f" (step {entry.get('step', '?')}, " \
                f"D{int(entry.get('desktop', label.get('desktop', 0))):02d})" if where else ""
        if multi:
            name = VIEW_NAMES.get(views[i], views[i] or "?")
            cap = f"image {i + 1} of {len(images)}: {name}" \
                  + (f", {moment}" if moment in ("before", "after") else "") + place
        elif not where:
            cap = f"image {i + 1} of {len(images)}" if len(images) > 1 else ""
        else:
            cap = (f"image {i + 1} of {len(images)}: " if len(images) > 1 else "") + \
                  f"{entry.get('view', label.get('view', '?'))} step {entry.get('step', '?')} " \
                  f"(D{int(entry.get('desktop', label.get('desktop', 0))):02d})"
        boxes = list(evidence) if (not multi or views[i] == "scan") else []
        if marks:
            boxes += marks_of(record, i)
        figs.append((moment or "", figure_html(image_id, manifest, image_root, out_dir, boxes, cap,
                                               index=i)))
    if not multi:
        return f'<div class="frames single">{"".join(f for _, f in figs)}</div><!--/frames-->'
    rows: list[tuple[str, list[str]]] = []
    for moment, fig in figs:
        if not rows or rows[-1][0] != moment:
            rows.append((moment, []))
        rows[-1][1].append(fig)
    out = []
    for moment, row in rows:
        head = f'<span class="rowlabel">{esc(moment)}</span>' if moment in ("before", "after") else ""
        cls = f"multi {moment}" if moment in ("before", "after") else "multi"
        out.append(f'<div class="frames {cls}">{head}{"".join(row)}</div><!--/frames-->')
    return "".join(out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def esc(text) -> str:
    return html.escape(str(text), quote=True)


CSS = """
:root { --ink: #1b1b1b; --muted: #666; --line: #d9d9d9; --paper: #fafafa; --card: #fff;
        --ok: #2e7d32; --bad: #c62828; --doubt: #ef6c00; --accent: #1565c0; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
       color: var(--ink); background: var(--paper); }
header.bar { position: sticky; top: 0; z-index: 5; background: var(--card);
             border-bottom: 1px solid var(--line); padding: 8px 16px; display: flex;
             gap: 14px; align-items: center; flex-wrap: wrap; }
header.bar h1 { font-size: 17px; margin: 0; }
header.bar .spacer { flex: 1; }
header.bar input[type=text] { padding: 4px 8px; border: 1px solid var(--line); border-radius: 4px; }
button { padding: 5px 12px; border: 1px solid var(--accent); background: var(--accent);
         color: #fff; border-radius: 4px; cursor: pointer; font: inherit; }
button.quiet { background: var(--card); color: var(--accent); }
main { max-width: 1400px; margin: 0 auto; padding: 12px 16px 80px; }
section.guide { background: var(--card); border: 1px solid var(--line); border-radius: 6px;
                padding: 10px 14px; margin-bottom: 14px; color: var(--muted); }
section.guide b { color: var(--ink); }
article.item { background: var(--card); border: 1px solid var(--line); border-radius: 6px;
               padding: 12px 16px; margin-bottom: 16px; }
article.item.done { border-color: #9ccc9c; }
article.item h2 { font-size: 16px; margin: 0 0 8px; display: flex; gap: 12px; flex-wrap: wrap;
                  align-items: baseline; }
article.item h2 .task { background: #e3f2fd; color: var(--accent); padding: 1px 8px;
                        border-radius: 10px; font-size: 13px; }
article.item h2 .meta { color: var(--muted); font-size: 13px; font-weight: normal; }
.frames { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
figure.frame { margin: 0; position: relative; display: inline-block; max-width: 100%;
               flex: 1 1 420px; }
figure.frame .imgbox { position: relative; display: block; border: 1px solid var(--line); }
figure.frame img { display: block; width: 100%; height: auto; background: #eee; }
figure.frame svg.boxes, figure.frame svg.review { position: absolute; left: 0; top: 0;
                         width: 100%; height: 100%; pointer-events: none; }
.frames.multi figure.frame { flex: 1 1 300px; }
.frames .rowlabel { flex: 0 0 100%; font-size: 13px; color: var(--muted); font-weight: bold; }
/* the model's layer (svg.boxes): the marks exactly as the converter draws them (mark-v2) */
svg.boxes line.mark { stroke: #ff1744; }
svg.boxes line.hint { stroke: #ffd600; }
svg.boxes rect.marktag { fill: #ff1744; stroke: none; }
svg.boxes text.marktag { fill: #fff; font-family: sans-serif; }
/* the reviewer's layer (svg.review): evidence and answer boxes, never shown to the model,
   hidden until "show reviewer boxes" is ticked */
body:not(.show-review) svg.review { display: none; }
svg.review rect { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
svg.review rect.answer { stroke: #00e676; }
svg.review rect.evidence { stroke: #00e5ff; }
svg.review text { fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 3px;
                  font-weight: bold; }
/* the human form's own boxes, drawn on the picture by the participant */
svg.boxes rect.human-box { fill: none; stroke: #00e676; stroke-width: 3;
                           vector-effect: non-scaling-stroke; }
svg.boxes text.human-box { fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 3px;
                           font-weight: bold; }
svg.boxes rect.draft { fill: none; stroke: #ff4081; stroke-width: 3; stroke-dasharray: 6 4;
                       vector-effect: non-scaling-stroke; }
.legend-line { font-size: 13px; color: var(--muted); flex-basis: 100%; }
.legend-line .red { color: #ff1744; font-weight: bold; }
.legend-line .yel { color: #b8860b; font-weight: bold; }
.legend-line .cyan { color: #00838f; font-weight: bold; }
.legend-line .green { color: #2e7d32; font-weight: bold; }
.prompt-line { color: var(--muted); margin: 4px 0; white-space: pre-wrap; }
figure.frame figcaption { font-size: 12px; color: var(--muted); margin-top: 2px; }
.question { font-size: 16px; margin: 6px 0; white-space: pre-wrap; }
ol.options { margin: 4px 0 8px 20px; padding: 0; }
ol.options li.gold { font-weight: bold; }
.gold { background: #fffde7; border-left: 4px solid #fbc02d; padding: 6px 10px; margin: 8px 0; }
.gold table, .fields table { border-collapse: collapse; }
.gold td { padding: 1px 10px 1px 0; vertical-align: top; }
.gold td.k { color: var(--muted); white-space: nowrap; }
details { margin: 4px 0; }
details summary { cursor: pointer; color: var(--accent); font-size: 13px; }
pre { font-size: 12px; white-space: pre-wrap; word-break: break-all; background: #f4f4f4;
      padding: 6px 8px; border-radius: 4px; max-height: 260px; overflow: auto; }
ul.rationale { margin: 4px 0 4px 20px; padding: 0; font-size: 13px; color: var(--muted); }
.verdict { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin-top: 10px;
           padding-top: 8px; border-top: 1px dashed var(--line); }
.verdict label { cursor: pointer; padding: 4px 10px; border: 1px solid var(--line);
                 border-radius: 16px; }
.verdict input:checked + span { font-weight: bold; }
.verdict label.correct:has(input:checked) { border-color: var(--ok); color: var(--ok); }
.verdict label.wrong:has(input:checked) { border-color: var(--bad); color: var(--bad); }
.verdict label.doubtful:has(input:checked) { border-color: var(--doubt); color: var(--doubt); }
textarea.note { width: 100%; min-height: 44px; margin-top: 8px; font: inherit; padding: 6px;
                border: 1px solid var(--line); border-radius: 4px; }
.fields { margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--line); }
.fields label.opt { display: inline-block; margin: 2px 14px 2px 0; cursor: pointer; }
.fields input[type=text], .fields input[type=number], .fields select, .fields textarea {
  font: inherit; padding: 4px 6px; border: 1px solid var(--line); border-radius: 4px; }
.fields textarea { width: 100%; min-height: 70px; }
.fields .row { margin: 6px 0; }
.fields .hint { color: var(--muted); font-size: 13px; }
ul.boxlist { margin: 4px 0; padding-left: 18px; font-size: 13px; }
ul.boxlist li { margin: 2px 0; }
.timer { color: var(--muted); font-size: 12px; margin-top: 4px; }
footer { color: var(--muted); font-size: 13px; text-align: center; padding: 20px; }
"""

#: Shared JS: CSV quoting, download, the reviewer name, box overlays on image load.
JS_COMMON = r"""
function q(s) { s = (s === undefined || s === null) ? '' : String(s);
  return '"' + s.replace(/"/g, '""') + '"'; }
function download(name, text) {
  const blob = new Blob(['﻿' + text], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = name; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); }
function store(key, obj) { try { localStorage.setItem(key, JSON.stringify(obj)); } catch (e) {} }
function load(key) { try { return JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { return {}; } }
const REVIEWER_KEY = 'tda-sheet-reviewer';
function reviewer() { return (document.getElementById('reviewer').value || '').trim(); }
function nowIso() { return new Date().toISOString(); }
function safeName(s) { return (s || 'anon').replace(/[^A-Za-z0-9_\-一-鿿]+/g, '_'); }
const SVGNS = 'http://www.w3.org/2000/svg';
function el(name, attrs, parent) {
  const n = document.createElementNS(SVGNS, name);
  Object.keys(attrs).forEach(k => n.setAttribute(k, attrs[k]));
  if (parent) parent.appendChild(n);
  return n;
}
/* the converter's mark-v2, transcribed (vlm_to_chat.draw_marks): a thin dashed outline just
   outside the box at MARK.opacity, each side starting with a dash at its corner; one filled
   label per tag outside the tag's top-most outline (label_place); a hint drawn inside its box */
function markWidth(W) { return Math.max(2, Math.round(2 * W / 1600)); }
function markDash(W) { return [Math.max(6, Math.round(14 * W / 1600)), Math.max(4, Math.round(8 * W / 1600))]; }
function labelFont(W) { return Math.max(12, Math.round(18 * W / 1600)); }
function labelPlace(o, tw, th, W, H) {
  const gap = 1, lx = Math.min(Math.max(0, o[0]), Math.max(0, W - tw));
  if (o[1] - gap - th >= 0) return [lx, o[1] - gap - th];
  if (o[3] + gap + th <= H - 1) return [lx, o[3] + gap];
  const ly = Math.min(Math.max(0, o[1]), Math.max(0, H - th));
  if (o[0] - gap - tw >= 0) return [o[0] - gap - tw, ly];
  if (o[2] + gap + tw <= W - 1) return [o[2] + gap, ly];
  return [0, 0];
}
function drawModel(svg, boxes, W, H) {
  const lw = markWidth(W), dash = markDash(W), tags = {}, order = [];
  const g = el('g', {opacity: MARK.opacity, class: 'gold-box'}, svg);
  boxes.forEach(b => {
    const hint = b.kind === 'hint', x = b.bbox[0], y = b.bbox[1], w = b.bbox[2], h = b.bbox[3];
    if (x + w <= 0 || y + h <= 0 || x >= W || y >= H) return;
    const o = hint ? [Math.max(0, Math.round(x)), Math.max(0, Math.round(y)),
                      Math.min(W - 1, Math.round(x + w) - 1), Math.min(H - 1, Math.round(y + h) - 1)]
                   : [Math.round(x) - lw, Math.round(y) - lw, Math.round(x + w) + lw - 1,
                      Math.round(y + h) + lw - 1];
    const half = lw / 2, cls = hint ? 'hint' : 'mark';
    [[o[0], o[1] + half, o[2] + 1, o[1] + half], [o[0], o[3] + 1 - half, o[2] + 1, o[3] + 1 - half],
     [o[0] + half, o[1], o[0] + half, o[3] + 1], [o[2] + 1 - half, o[1], o[2] + 1 - half, o[3] + 1]]
      .forEach(s => el('line', {x1: s[0], y1: s[1], x2: s[2], y2: s[3], class: cls,
                                'stroke-width': lw, 'stroke-dasharray': dash[0] + ' ' + dash[1],
                                'stroke-linecap': 'butt'}, g));
    if (!hint && b.label) {
      if (!tags[b.label]) { tags[b.label] = []; order.push(b.label); }
      tags[b.label].push(o);
    }
  });
  const fs = labelFont(W), pad = Math.max(2, lw);
  order.forEach(tag => {
    const o = tags[tag].reduce((a, c) => (c[1] < a[1] || (c[1] === a[1] && c[0] < a[0])) ? c : a);
    const tw = Math.ceil(fs * 0.62 * tag.length) + 2 * pad, th = Math.ceil(fs * 0.8) + 2 * pad;
    const p = labelPlace(o, tw, th, W, H);
    el('rect', {x: p[0], y: p[1], width: tw, height: th, class: 'gold-box marktag'}, svg);
    const t = el('text', {x: p[0] + pad, y: p[1] + th - pad, 'font-size': fs,
                          class: 'gold-box marktag'}, svg);
    t.textContent = tag;
  });
}
function drawReview(svg, boxes, W) {
  const fs = Math.max(12, Math.round(W / 70));
  boxes.forEach(b => {
    el('rect', {x: b.bbox[0], y: b.bbox[1], width: b.bbox[2], height: b.bbox[3],
                class: 'gold-box ' + b.kind}, svg);
    const t = el('text', {x: b.bbox[0] + 4, y: Math.max(fs, b.bbox[1] - 4), 'font-size': fs,
                          class: 'gold-box ' + b.kind}, svg);
    t.textContent = b.label;
  });
}
function drawBoxes(fig) {
  const img = fig.querySelector('img'), svg = fig.querySelector('svg.boxes'),
        review = fig.querySelector('svg.review');
  if (!img || !svg || !img.naturalWidth) return;
  const W = img.naturalWidth, H = img.naturalHeight;
  [svg, review].forEach(s => { if (!s) return;
    s.setAttribute('viewBox', '0 0 ' + W + ' ' + H); s.setAttribute('preserveAspectRatio', 'none');
    s.querySelectorAll('.gold-box').forEach(n => n.remove()); });
  let boxes = [];
  try { boxes = JSON.parse(img.dataset.boxes || '[]'); } catch (e) {}
  drawModel(svg, boxes.filter(b => b.kind === 'mark' || b.kind === 'hint'), W, H);
  if (review) drawReview(review, boxes.filter(b => b.kind !== 'mark' && b.kind !== 'hint'), W);
}
document.addEventListener('DOMContentLoaded', () => {
  const rv = document.getElementById('reviewer');
  if (rv) { rv.value = (load(REVIEWER_KEY).name || ''); rv.addEventListener('input',
    () => store(REVIEWER_KEY, {name: rv.value})); }
  document.querySelectorAll('figure.frame').forEach(fig => {
    const img = fig.querySelector('img');
    if (img.complete && img.naturalWidth) drawBoxes(fig);
    img.addEventListener('load', () => drawBoxes(fig));
    img.addEventListener('error', () => { const c = fig.querySelector('figcaption');
      if (c) c.textContent = 'image not found: ' + img.getAttribute('src') + ' -- ' + c.textContent; });
  });
  const tr = document.getElementById('toggle-review');
  if (tr) { tr.checked = false;
    tr.addEventListener('change', () => document.body.classList.toggle('show-review', tr.checked)); }
});
"""
#: The converter's mark style, handed to the page's script (the marks a reviewer sees are drawn
#: with the model's parameters, not a look-alike).
JS_COMMON = ("const MARK = " + json.dumps({"opacity": CHAT.MARK_OPACITY,
                                           "style": CHAT.MARK_STYLE_VERSION}) + ";\n"
             + JS_COMMON)

JS_AUDIT = r"""
const SHEET = document.body.dataset.sheet;
const KEY = 'tda-audit:' + SHEET;
let answers = load(KEY);
function items() { return Array.from(document.querySelectorAll('article.item')); }
function progress() {
  const all = items(), done = all.filter(a => answers[a.dataset.id] && answers[a.dataset.id].verdict);
  document.getElementById('progress').textContent = done.length + ' / ' + all.length;
  all.forEach(a => a.classList.toggle('done', !!(answers[a.dataset.id] && answers[a.dataset.id].verdict)));
}
function restore() {
  items().forEach(a => {
    const st = answers[a.dataset.id]; if (!st) return;
    if (st.verdict) { const r = a.querySelector('input[value="' + st.verdict + '"]'); if (r) r.checked = true; }
    if (st.note) a.querySelector('textarea.note').value = st.note;
  });
  progress();
}
function save(a) {
  const id = a.dataset.id, picked = a.querySelector('input[type=radio]:checked');
  answers[id] = {verdict: picked ? picked.value : '', note: a.querySelector('textarea.note').value,
                 at: nowIso()};
  store(KEY, answers); progress();
}
function exportCsv() {
  const cols = JSON.parse(document.body.dataset.columns);
  const rows = [cols.map(q).join(',')];
  items().forEach(a => {
    const st = answers[a.dataset.id] || {};
    const base = JSON.parse(a.dataset.row);
    base.verdict = st.verdict || ''; base.note = st.note || ''; base.reviewer = reviewer();
    base.answered_at = st.verdict ? (st.at || '') : '';
    rows.push(cols.map(c => q(base[c])).join(','));
  });
  download(SHEET + '.' + safeName(reviewer()) + '.csv', rows.join('\r\n') + '\r\n');
}
document.addEventListener('DOMContentLoaded', () => {
  restore();
  items().forEach(a => a.addEventListener('change', () => save(a)));
  items().forEach(a => a.querySelector('textarea.note').addEventListener('input', () => save(a)));
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('reset').addEventListener('click', () => {
    if (!confirm('Clear every answer of this page from this browser?')) return;
    answers = {}; store(KEY, answers);
    items().forEach(a => { a.querySelectorAll('input[type=radio]').forEach(r => r.checked = false);
      a.querySelector('textarea.note').value = ''; });
    progress();
  });
});
"""


def figure_html(image_id: str, manifest: dict, image_root, out_dir, boxes: list[dict],
                caption: str, *, index: int = 0) -> str:
    """One frame: the ``<img>`` with its boxes as a data attribute and two SVG overlays -- the
    model's layer (``svg.boxes``: the marks, as the converter draws them) and the reviewer's
    (``svg.review``: evidence and answer boxes, hidden until asked; owner 2026-11-04)."""
    entry = manifest.get(image_id) or {}
    rel = entry.get("path", image_id)
    src = rel_image(image_root, rel, out_dir)
    data = esc(json.dumps(boxes, ensure_ascii=False, separators=(",", ":")))
    # the overlay shares a box with the picture alone: spanning the caption too, it was stretched
    # vertically and every box was drawn too low (owner, 2026-11-03)
    return (f'<figure class="frame" data-index="{index}"><span class="imgbox"><img src="{esc(src)}" '
            f'alt="{esc(rel)}" loading="lazy" data-boxes="{data}"><svg class="boxes" viewBox="0 0 1 1">'
            f'</svg><svg class="review" viewBox="0 0 1 1"></svg></span>'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def question_html(record: dict, *, gold_letter: Optional[str] = None) -> str:
    """The question and, for a multiple choice, the lettered options (gold in bold if given)."""
    prompt = record["prompt"]
    parts = []
    if prompt.get("images"):  # the lines the converter puts before the question (chat-v2)
        for line in (legend_line(record), marks_line(record)):
            if line:
                parts.append(f'<p class="prompt-line">{esc(line)}</p>')
    parts.append(f'<p class="question">{esc(prompt["question"])}</p>')
    options = prompt.get("options") or []
    if options:
        items = []
        for i, o in enumerate(options):
            cls = ' class="gold"' if gold_letter == LETTERS[i] else ""
            items.append(f'<li{cls}>{LETTERS[i]}. {esc(o.get("verb", ""))} '
                         f'{esc(o.get("target_label", ""))}</li>')
        parts.append('<ol class="options" type="none">' + "".join(items) + "</ol>")
    glossary = glossary_text(prompt.get("terms"), prompt.get("state_terms"),
                             presence=is_presence(record))
    if glossary:
        parts.append(f'<details><summary>glossary the model reads</summary>'
                     f'<p class="prompt-line">{esc(glossary)}</p></details>')
    return "".join(parts)


def _gold_letter(record: dict) -> Optional[str]:
    label = record["label"]
    if format_key(record) != "V6.options":
        return None
    index = reference_option(label)
    return None if index is None else LETTERS[index]


def csv_row(record: dict, batch: int) -> dict:
    label = record["label"]
    return {"id": record["id"], "task": record["prompt"]["task"],
            "readable_id": label.get("readable_id", ""), "desktop": label.get("desktop", ""),
            "view": label.get("view", ""), "step": label.get("step", ""), "batch": batch,
            "verdict": "", "note": "", "reviewer": "", "answered_at": ""}


def item_html(index: int, record: dict, manifest: dict, world: Optional[World], image_root,
              out_dir, batch: int) -> str:
    """One audit item: frames, question, gold, evidence, rationale, verdict."""
    label = record["label"]
    task = record["prompt"]["task"]
    frames = frames_html(record, manifest, image_root, out_dir, evidence=boxes_of(record, world))
    gold = gold_lines(record, world)
    gold_rows = "".join(f'<tr><td class="k">{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in gold)
    raw = json.dumps(label["answer"], ensure_ascii=False, indent=1)
    rationale = rationale_lines(record, world)
    rationale_html = ("<ul class=\"rationale\">" + "".join(f"<li>{esc(l)}</li>" for l in rationale)
                      + "</ul>") if rationale else ""
    ev = (label.get("evidence") or {}).get("instances") or []
    ev_html = (f'<div class="hint">evidence parts: '
               f'{esc("; ".join(display_name(k, world, record) for k in ev))}</div>') if ev else ""
    flags = [f"template {label.get('template_id', '?')}"]
    if label.get("negative"):
        flags.append(f"negative: {label['negative']}")
    if label.get("truth_source"):
        flags.append(f"truth source: {label['truth_source']}")
    if label.get("view_confirmed") is False:
        flags.append("view not confirmed (log and state machine only)")
    if label.get("image_undecidable"):
        flags.append("image_undecidable")
    radios = "".join(
        f'<label class="{v}"><input type="radio" name="v_{esc(record["id"])}" value="{v}">'
        f'<span>{zh} {v}</span></label>' for v, zh, _ in VERDICTS)
    row = esc(json.dumps(csv_row(record, batch), ensure_ascii=False, separators=(",", ":")))
    return f"""
<article class="item" id="{esc(record["id"])}" data-id="{esc(record["id"])}" data-row="{row}">
<h2><span>#{index}</span><span class="task">{esc(task)}</span>
<span class="meta">{esc(label.get("readable_id", ""))} &middot; {esc(record["id"])} &middot; {esc("; ".join(flags))}</span></h2>
{frames}
{question_html(record, gold_letter=_gold_letter(record))}
<div class="gold"><b>Gold answer</b><table>{gold_rows}</table>
<details><summary>raw JSON</summary><pre>{esc(raw)}</pre></details></div>
{ev_html}{rationale_html}
<div class="verdict"><b>Verdict</b> {radios}</div>
<textarea class="note" placeholder="note (what is wrong, what you could not see, ...)"></textarea>
</article>"""


#: The header's switch for the reviewer layer (unticked: the page opens on the model's view) and
#: the one-line legend saying what is and is not shown to the model (owner 2026-11-04: the
#: reviewer boxes, cyan and yellow beside the red marks, were taken for marks the model sees).
REVIEW_TOGGLE = (
    '<label><input type="checkbox" id="toggle-review"> show reviewer boxes / 显示审核框</label>'
    '<span class="legend-line">The pictures show exactly what the model gets: '
    '<span class="red">red dashed outlines</span> (tagged A, B, ...) and, on V3\'s hinted form, '
    'a <span class="yel">yellow dashed box</span>. Reviewer boxes '
    '(<span class="cyan">cyan</span> = evidence, <span class="green">green</span> = answer) are '
    '<b>not shown to the model</b>. / 图上就是模型看到的：红色虚线框（字母标签）和 V3 提示题的黄色虚线框；'
    '审核框（青色＝证据，绿色＝答案）<b>模型看不到</b>。</span>')


def page(title: str, body: str, *, sheet: str, js: str, columns: Sequence[str],
         guide: str, extra_bar: str = "") -> str:
    """A complete static page: sticky bar (reviewer, progress, export), guide, items."""
    cols = esc(json.dumps(list(columns)))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body data-sheet="{esc(sheet)}" data-columns="{cols}">
<header class="bar"><h1>{esc(title)}</h1>
<label>reviewer / 审核人 <input type="text" id="reviewer" placeholder="your name"></label>
{REVIEW_TOGGLE}
{extra_bar}
<span class="spacer"></span>
<span>answered <b id="progress">0 / 0</b></span>
<button id="export">Export answers (CSV) / 导出答案</button>
<button id="reset" class="quiet">Reset page / 清空本页</button>
</header>
<main>
<section class="guide">{guide}</section>
{body}
</main>
<footer>Answers stay in this browser (localStorage) until exported; export when the page is done.
Static page: no server, no network.</footer>
<script>{JS_COMMON}</script>
<script>{js}</script>
</body></html>
"""


AUDIT_GUIDE = """
<b>What to do.</b> For every item look at the picture(s) -- they show exactly what the model is
shown: the red dashed outlines the question names (part A, B, ...) and, on V3's hinted form, a
yellow dashed box -- read the question and the gold answer, then pick one verdict and, for
anything but 正确, write why. Tick <b>show reviewer boxes</b> to see the answer's evidence
(cyan = the parts it rests on, green = the answer's parts); the model never sees those.<br>
<b>正确 correct</b> -- the answer is right for this picture and this question.
<b>错误 wrong</b> -- the answer is wrong, or the question cannot be asked of this picture (wrong
frame, part not there, mislabelled).
<b>存疑 doubtful</b> -- cannot be decided from the picture (hidden part, captive screw, blur), or
only partly right.<br>
Parts are named without ordinals: "part A" is the part outlined with tag A, a part alone of its
class is named by its class, "one of the 4 RAM modules" is one of several; a two-image question shows both frames
in the order asked. Your answers are kept in this browser; press <b>Export answers</b> when the
page is done and send the CSV. <code>tools/vlm_audit_summary.py</code> reads it.
"""


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #
def csv_skeleton(records: Iterable[dict], batch: int = DEFAULT_BATCH) -> str:
    """The empty CSV, one row per record, ``batch`` as the page would number it."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
    writer.writeheader()
    for i, r in enumerate(records):
        writer.writerow(csv_row(r, i // batch + 1))
    return buf.getvalue()


def batches(records: Sequence[dict], batch: int) -> list[list[dict]]:
    batch = max(1, int(batch))
    return [list(records[i:i + batch]) for i in range(0, len(records), batch)]


def build(records: Sequence[dict], exports: Sequence, image_root, out_dir, *,
          batch: int = DEFAULT_BATCH, name: str = "audit", title: Optional[str] = None) -> dict:
    """Write the pages and the skeleton under ``out_dir``; returns what was written."""
    records = list(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, worlds = load_context(exports)
    files: list[str] = []
    parts = batches(records, batch)
    for b, chunk in enumerate(parts, start=1):
        body = "".join(
            item_html((b - 1) * batch + i + 1, r, manifest, worlds.get(IO.desktop_of(r)),
                      image_root, out, b)
            for i, r in enumerate(chunk))
        sheet = f"{name}.sheet.b{b:02d}"
        head = f"{title or 'Human audit'} -- batch {b} of {len(parts)} ({len(chunk)} items)"
        text = page(head, body, sheet=sheet, js=JS_AUDIT, columns=CSV_COLUMNS, guide=AUDIT_GUIDE)
        (out / f"{sheet}.html").write_text(text, encoding="utf-8", newline="\n")
        files.append(f"{sheet}.html")
    skeleton = f"{name}.sheet.csv"
    (out / skeleton).write_text(csv_skeleton(records, batch), encoding="utf-8", newline="")
    files.append(skeleton)
    missing = sorted({img for r in records for img in r["prompt"].get("images") or []}
                     - set(manifest))
    return {"items": len(records), "batches": len(parts), "batch_size": batch, "files": files,
            "images_without_manifest": missing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--records", required=True, help="the audit sample (audit.jsonl)")
    ap.add_argument("--export", nargs="+", required=True,
                    help="the truth files whose manifests and oracles name the images and parts")
    ap.add_argument("--image-root", required=True, help="the pack's images/ directory")
    ap.add_argument("--out", required=True, help="where the pages go (eval_v1/audit)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="items per page")
    ap.add_argument("--name", default="audit")
    args = ap.parse_args(argv)
    records = list(IO.read_jsonl(args.records))
    result = build(records, args.export, args.image_root, args.out, batch=args.batch,
                   name=args.name)
    print(f"[vlm-audit-sheet] {result['items']} items in {result['batches']} pages -> {args.out}: "
          + ", ".join(result["files"]))
    if result["images_without_manifest"]:
        print(f"[vlm-audit-sheet] warning: {len(result['images_without_manifest'])} image ids "
              f"have no manifest row (their <img> points at the id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
