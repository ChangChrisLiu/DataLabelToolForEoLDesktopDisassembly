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
  two-image question: V3, V12, V15), the question with its lettered options,
  the gold answer in words and as the raw JSON, its evidence (boxes drawn over
  the frame, switchable) and the rationale chain, then three radio buttons
  正确 / 错误 / 存疑 (correct / wrong / doubtful) and a free-text note. The
  page keeps what is typed in the browser's ``localStorage`` (a reload loses
  nothing) and its **export button downloads a CSV** of the answers; a
  reviewer who prefers a spreadsheet fills in
* ``<name>.sheet.csv`` -- the same rows as an empty CSV skeleton.

Image paths are **relative to the page** (``../../images/<view>/D<nn>/s<nn>.png``
for a page in ``eval_v1/audit/``), so the pack can be zipped and sent whole;
the page itself opens from the file system. Instance keys are written the way
the oracle labels them ("Motherboard screw 7", never ``screw.motherboard.07``
alone). The sheet is deterministic: no clock, no machine path, so two pack
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
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import LETTERS, World, format_key  # noqa: E402

__all__ = ["CSV_COLUMNS", "VERDICTS", "boxes_of", "build", "csv_skeleton", "gold_lines",
           "item_html", "load_context", "main", "name_of", "page", "rationale_lines",
           "rel_image"]

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


def name_of(key: Optional[str], world: Optional[World]) -> str:
    """The oracle's label of an instance key (``Motherboard screw 7``), else the key."""
    if key is None:
        return "-"
    if world is not None:
        inst = world.instances.get(str(key))
        if inst and inst.get("label"):
            return str(inst["label"])
    return str(key)


def rel_image(image_root, rel_path: str, out_dir) -> str:
    """The ``src`` of a frame: its path relative to the page's directory, POSIX."""
    target = Path(image_root) / rel_path
    return Path(os.path.relpath(str(target), str(out_dir))).as_posix()


# --------------------------------------------------------------------------- #
# the gold answer in words
# --------------------------------------------------------------------------- #
def _action(item: dict, world) -> str:
    return f"{item.get('verb', '?')} {name_of(item.get('target'), world)}"


def _yes(value) -> str:
    return "yes" if value else "no"


def gold_lines(record: dict, world: Optional[World] = None) -> list[tuple[str, str]]:
    """``[(field, text)]`` -- the gold answer of one record, readable by a person."""
    label = record["label"]
    answer = label["answer"]
    check = label.get("answer_check") or {}
    key = format_key(record)
    task = record["prompt"]["task"]
    out: list[tuple[str, str]] = []
    if key == "V1":
        comps = answer.get("components") or []
        out.append(("visible parts", f"{len(comps)}"))
        for c in comps:
            out.append((c.get("class", "?"),
                        f"{name_of(c.get('instance'), world)} box {c.get('bbox')}"))
    elif key == "V2.count":
        out.append(("count", str(answer.get("count"))))
        out.append(("screws", f"{check.get('role', '?')} screws still {check.get('state', 'present')}"))
    elif key == "V2.state":
        out.append(("state", str(answer.get("state"))))
        if check.get("allowed"):
            out.append(("allowed", ", ".join(check["allowed"])))
    elif task == "V3":
        out.append(("action", f"{answer.get('verb')} {name_of(answer.get('target_instance'), world)}"))
        out.append(("class", str(answer.get("target_class"))))
        out.append(("tool", str(answer.get("tool"))))
    elif task == "V4":
        out.append(("feasible", _yes(answer.get("feasible"))))
        out.append(("action", f"{check.get('verb')} {name_of(check.get('target'), world)}"))
        if answer.get("blockers"):
            out.append(("blockers", "; ".join(name_of(b, world) for b in answer["blockers"])))
        if answer.get("reason"):
            out.append(("reason", str(answer["reason"])))
        out.append(("truth source", str(label.get("truth_source"))))
    elif key == "V6.options":
        ref = tuple(check.get("reference") or [])
        options = label.get("options") or []
        index = next((i for i, o in enumerate(options)
                      if (o.get("verb"), o.get("target")) == ref), None)
        letter = LETTERS[index] if index is not None else "?"
        out.append(("choice", f"{letter}: {_action(answer, world)}"))
        kinds = ", ".join(f"{LETTERS[i]} {o.get('kind', '?')}" for i, o in enumerate(options))
        if kinds:
            out.append(("option kinds", kinds))
    elif key == "V6.open":
        out.append(("next action", _action(answer, world)))
        eq = check.get("equivalent_actions") or []
        if len(eq) > 1:
            out.append(("equivalent", "; ".join(_action(a, world) for a in eq)))
        if check.get("blocked_actions"):
            out.append(("blocked", "; ".join(_action(a, world) for a in check["blocked_actions"][:8])
                        + (" ..." if len(check["blocked_actions"]) > 8 else "")))
    elif task == "V7":
        plan = answer.get("plan") or []
        out.append(("plan", f"{len(plan)} steps to remove {name_of(answer.get('goal'), world)}"))
        for i, a in enumerate(plan, start=1):
            out.append((f"{i}", _action(a, world)))
        if check.get("required_actions"):
            out.append(("required", "; ".join(
                f"{name_of(r.get('target'), world)} -> {'/'.join(r.get('states') or [])}"
                for r in check["required_actions"])))
    elif task == "V8":
        out.append(("part", name_of(answer.get("instance"), world)))
        out.append(("box", str(answer.get("bbox"))))
    elif task == "V10":
        done = answer.get("done") or []
        out.append(("progress", f"{_BIN_LABELS.get(str(answer.get('progress_bin')), answer.get('progress_bin'))} "
                                f"({check.get('done_actions', len(done))} of {check.get('total_actions', '?')} actions done)"))
        for i, a in enumerate(done, start=1):
            out.append((f"done {i}", _action(a, world)))
    elif task == "V12":
        out.append(("changed", _yes(answer.get("changed"))))
        for ev in answer.get("events") or []:
            out.append(("event", f"{name_of(ev.get('target'), world)}: {ev.get('old')} -> {ev.get('new')}"))
        for ev in check.get("hidden_events") or []:
            out.append(("hidden (no visible change)",
                        f"{name_of(ev.get('target'), world)}: {ev.get('old')} -> {ev.get('new')}"))
        if label.get("negative"):
            out.append(("negative kind", str(label["negative"])))
    elif task == "V14":
        out.append(("answerable", _yes(answer.get("answerable"))))
        out.append(("part", name_of(check.get("instance"), world)))
        if answer.get("answerable"):
            out.append(("state", str(answer.get("state"))))
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
    """The rationale chain, one line per step."""
    steps = ((record["label"].get("rationale") or {}).get("steps")) or []
    out = []
    for s in steps:
        op = s.get("op", "?")
        bits = [op]
        if s.get("target"):
            bits.append(name_of(s["target"], world))
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
    """The boxes to draw: ``[{"label", "bbox", "kind"}]`` -- evidence, V1 parts, V8 answer."""
    label = record["label"]
    out: list[dict] = []
    seen: set[tuple] = set()

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
            add(name_of(c.get("instance"), world), c.get("bbox"), "answer")
    if record["prompt"]["task"] == "V8":
        add(name_of(label["answer"].get("instance"), world), label["answer"].get("bbox"), "answer")
    ev = label.get("evidence") or {}
    for key, bbox in (ev.get("bboxes") or {}).items():
        add(name_of(key, world), bbox, "evidence")
    return out


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
figure.frame img { display: block; width: 100%; height: auto; border: 1px solid var(--line);
                   background: #eee; }
figure.frame svg.boxes { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
                         pointer-events: none; }
body.no-boxes svg.boxes rect, body.no-boxes svg.boxes text { display: none; }
svg.boxes rect { fill: none; stroke-width: 3; vector-effect: non-scaling-stroke; }
svg.boxes rect.answer { stroke: #ffeb3b; }
svg.boxes rect.evidence { stroke: #00e5ff; }
svg.boxes rect.draft { stroke: #ff4081; stroke-dasharray: 6 4; }
svg.boxes text { fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 3px;
                 font-weight: bold; }
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
function drawBoxes(fig) {
  const img = fig.querySelector('img'), svg = fig.querySelector('svg.boxes');
  if (!img || !svg || !img.naturalWidth) return;
  svg.setAttribute('viewBox', '0 0 ' + img.naturalWidth + ' ' + img.naturalHeight);
  svg.setAttribute('preserveAspectRatio', 'none');
  let boxes = [];
  try { boxes = JSON.parse(img.dataset.boxes || '[]'); } catch (e) {}
  const fs = Math.max(12, Math.round(img.naturalWidth / 70));
  svg.querySelectorAll('.gold-box').forEach(n => n.remove());
  boxes.forEach(b => {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', b.bbox[0]); r.setAttribute('y', b.bbox[1]);
    r.setAttribute('width', b.bbox[2]); r.setAttribute('height', b.bbox[3]);
    r.setAttribute('class', 'gold-box ' + b.kind); svg.appendChild(r);
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', b.bbox[0] + 4); t.setAttribute('y', Math.max(fs, b.bbox[1] - 4));
    t.setAttribute('font-size', fs); t.setAttribute('class', 'gold-box');
    t.textContent = b.label; svg.appendChild(t);
  });
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
  const tb = document.getElementById('toggle-boxes');
  if (tb) tb.addEventListener('change', () => document.body.classList.toggle('no-boxes', !tb.checked));
});
"""

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
    """One frame: the ``<img>`` with its boxes as a data attribute and an SVG overlay."""
    entry = manifest.get(image_id) or {}
    rel = entry.get("path", image_id)
    src = rel_image(image_root, rel, out_dir)
    data = esc(json.dumps(boxes, ensure_ascii=False, separators=(",", ":")))
    return (f'<figure class="frame" data-index="{index}"><img src="{esc(src)}" alt="{esc(rel)}" '
            f'loading="lazy" data-boxes="{data}"><svg class="boxes" viewBox="0 0 1 1"></svg>'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def question_html(record: dict, *, gold_letter: Optional[str] = None) -> str:
    """The question and, for a multiple choice, the lettered options (gold in bold if given)."""
    prompt = record["prompt"]
    parts = [f'<p class="question">{esc(prompt["question"])}</p>']
    options = prompt.get("options") or []
    if options:
        items = []
        for i, o in enumerate(options):
            cls = ' class="gold"' if gold_letter == LETTERS[i] else ""
            items.append(f'<li{cls}>{LETTERS[i]}. {esc(o.get("verb", ""))} '
                         f'{esc(o.get("target_label", ""))}</li>')
        parts.append('<ol class="options" type="none">' + "".join(items) + "</ol>")
    return "".join(parts)


def _gold_letter(record: dict) -> Optional[str]:
    label = record["label"]
    if format_key(record) != "V6.options":
        return None
    ref = tuple(label["answer_check"].get("reference") or [])
    for i, o in enumerate(label.get("options") or []):
        if (o.get("verb"), o.get("target")) == ref:
            return LETTERS[i]
    return None


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
    boxes = boxes_of(record, world)
    images = record["prompt"].get("images") or []
    figs = []
    for i, image_id in enumerate(images):
        entry = manifest.get(image_id) or {}
        cap = (f"image {i + 1} of {len(images)}: " if len(images) > 1 else "") + \
              f"{entry.get('view', label.get('view', '?'))} step {entry.get('step', '?')} " \
              f"(D{int(entry.get('desktop', label.get('desktop', 0))):02d})"
        figs.append(figure_html(image_id, manifest, image_root, out_dir, boxes, cap, index=i))
    gold = gold_lines(record, world)
    gold_rows = "".join(f'<tr><td class="k">{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in gold)
    raw = json.dumps(label["answer"], ensure_ascii=False, indent=1)
    rationale = rationale_lines(record, world)
    rationale_html = ("<ul class=\"rationale\">" + "".join(f"<li>{esc(l)}</li>" for l in rationale)
                      + "</ul>") if rationale else ""
    ev = (label.get("evidence") or {}).get("instances") or []
    ev_html = (f'<div class="hint">evidence parts: '
               f'{esc("; ".join(name_of(k, world) for k in ev))}</div>') if ev else ""
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
<div class="frames">{"".join(figs)}</div>
{question_html(record, gold_letter=_gold_letter(record))}
<div class="gold"><b>Gold answer</b><table>{gold_rows}</table>
<details><summary>raw JSON</summary><pre>{esc(raw)}</pre></details></div>
{ev_html}{rationale_html}
<div class="verdict"><b>Verdict</b> {radios}</div>
<textarea class="note" placeholder="note (what is wrong, what you could not see, ...)"></textarea>
</article>"""


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
<label><input type="checkbox" id="toggle-boxes" checked> boxes / 显示框</label>
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
<b>What to do.</b> For every item look at the picture(s), read the question and the gold answer
(the boxes drawn on the frame are its evidence: yellow = the answer's parts, cyan = the parts it
rests on), then pick one verdict and, for anything but 正确, write why.<br>
<b>正确 correct</b> -- the answer is right for this picture and this question.
<b>错误 wrong</b> -- the answer is wrong, or the question cannot be asked of this picture (wrong
frame, part not there, mislabelled).
<b>存疑 doubtful</b> -- cannot be decided from the picture (hidden part, captive screw, blur), or
only partly right.<br>
Instance names are the dataset's ("Motherboard screw 7"); a two-image question shows both frames
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
