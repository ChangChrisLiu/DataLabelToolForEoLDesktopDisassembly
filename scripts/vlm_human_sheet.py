"""The human-baseline form and its converter to model-style predictions (W4-E).

``python tools/vlm_human_sheet.py build --export eval_v1/scan.jsonl eval_v1/oak1.jsonl
eval_v1/oak2.jsonl eval_v1/rs.jsonl --image-root images --out runs/human
[--ids <file> | --size 300 --per-task 25 --seed 0] [--batch 50] [--name human]``

``python tools/vlm_human_sheet.py convert --csv runs/human/*.csv --out runs/human/scored
[--export eval_v1/scan.jsonl ...] [--name human]``

**build** picks the items and writes static pages **without any answer**: the
frame(s), the question, the lettered options, and a plain form per task
instead of the JSON a model types (radio buttons for yes / no and states, a
verb list and a target box, one line per action for plans, a box-drawing tool
on the picture for V1 / V8). Nothing of ``label`` is in the page but the
record id and the task (no evidence, no rationale, no readable id, no
desktop or step); the image path does carry the view and the step index
(``../../images/scan/D46/s021.png``), which a participant who reads the URL
could use on V10 -- the pages do not show it. Every page keeps the answers in
``localStorage``, records **per-item time stamps** (first time the item is
half visible on screen, last change) and exports a CSV
(``id, task, format_key, fields_json, started_at, answered_at, seconds,
reviewer``). The default subset is the stratified draw of ``vlm_sample.draw``
(``purpose="human"``: a different hash stream from the audit's; about
``--per-task`` per task, positives and negatives balanced, every desktop
represented) over the records of the truth files given -- the pack's
``eval_v1`` -- with ``--size`` items; ``--ids`` takes any list instead. V6
pairs that are drawn whole are shown open form first.

**convert** maps the exported fields to the model answer formats of
:data:`tda.core.export.vlm_check.RESPONSE_FORMATS` and writes
``<out>/<name>.pred.jsonl`` -- ``{"id", "raw_text", "meta": {seconds, ...}}``
per answered item -- so ``tools/vlm_score.py --truth <view>.jsonl --pred
<name>.pred.jsonl`` scores a person exactly like a model. With ``--export``
it also cuts each truth file to the answered ids (``<out>/<name>.<view>.jsonl``
with its three sidecars; an item the person skipped would otherwise count as
a parse failure). A spreadsheet user may put a finished JSON answer into an
``answer_json`` column instead of ``fields_json``; it is passed through.
:func:`gold_fields` is the inverse for the self-test: the form a perfect
participant would fill in, which must score 1.0.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_audit_sheet as S  # noqa: E402
import vlm_io as IO  # noqa: E402
from tda.core.export.vlm_check import (  # noqa: E402
    LETTERS, VERBS, World, format_instruction, format_key, reference_option)

__all__ = ["CSV_COLUMNS", "FIELDS", "PROGRESS_BINS", "allowed_states", "build", "convert",
           "draw_subset", "gold_fields", "item_html", "main", "to_answer"]

CSV_COLUMNS = ("id", "task", "format_key", "fields_json", "started_at", "answered_at", "seconds",
               "reviewer")
PROGRESS_BINS = ("0-25", "25-50", "50-75", "75-100")
CANNOT_TELL = "cannot_tell"
#: ``format key -> the form's field names`` (what ``fields_json`` may carry).
FIELDS = {
    "V1": ("boxes",), "V2.state": ("state",), "V2.count": ("count",),
    "V3": ("verb", "target", "tool"), "V4": ("feasible", "blockers"),
    # V3's free form also asks the model for a region; a person answers the action alone
    "V3.region": ("verb", "target", "tool"),
    "V6.open": ("verb", "target"), "V6.options": ("choice",), "V7": ("plan",),
    "V8": ("boxes",), "V10": ("done", "progress"), "V12": ("changed",), "V14": ("state",),
    "V15.moment": ("same_moment",), "V15.cross": ("state", "best_view"),
    "V16": ("valid", "first_error_step"),
}
DEFAULT_SIZE = 300
DEFAULT_PER_TASK = 25
_SPLIT = re.compile(r"[;\n,]+")
_NUMBERED = re.compile(r"^\s*(?:\d+[.):]|[-*])\s*")


# --------------------------------------------------------------------------- #
# the subset
# --------------------------------------------------------------------------- #
def draw_subset(records: Sequence[dict], *, size: int = DEFAULT_SIZE,
                per_task: int = DEFAULT_PER_TASK, seed: int = 0) -> list[dict]:
    """``vlm_sample.draw`` with its own hash stream; V6 pairs open form first."""
    import vlm_sample

    desks = {IO.desktop_of(r) for r in records}
    rows = vlm_sample.draw(list(records), size=size, per_task=per_task, seed=seed,
                           purpose="human", test_desktops=desks)
    return order_pairs(rows)


def order_pairs(rows: list[dict]) -> list[dict]:
    """Where both forms of a V6 pair are present, the open form comes first."""
    rows = list(rows)
    where: dict[str, dict[str, int]] = {}
    for i, r in enumerate(rows):
        if r["prompt"]["task"] == "V6" and r["label"].get("v6_pair"):
            where.setdefault(r["label"]["v6_pair"], {})[r["label"].get("v6_form", "")] = i
    for forms in where.values():
        if "open" in forms and "mc" in forms and forms["mc"] < forms["open"]:
            i, j = forms["mc"], forms["open"]
            rows[i], rows[j] = rows[j], rows[i]
    return rows


def allowed_states(record: dict) -> list[str]:
    """The state vocabulary a V2 / V14 question is answered in (what the model is told)."""
    check = record["label"].get("answer_check") or {}
    allowed = list(check.get("allowed") or [])
    if not allowed and check.get("state_space") == "presence":
        allowed = ["present", "removed"]
    return allowed


# --------------------------------------------------------------------------- #
# the form
# --------------------------------------------------------------------------- #
def _radio(rid: str, field: str, options: Sequence[tuple[str, str]]) -> str:
    name = S.esc(f"{rid}:{field}")
    return "".join(f'<label class="opt"><input type="radio" name="{name}" data-field="{field}" '
                   f'value="{S.esc(v)}"> {S.esc(t)}</label>' for v, t in options)


def _verb_select(field: str) -> str:
    opts = "".join(f'<option value="{v}">{v}</option>' for v in VERBS)
    return f'<select data-field="{field}"><option value="">verb ...</option>{opts}</select>'


def _text(field: str, placeholder: str, size: int = 34) -> str:
    return (f'<input type="text" data-field="{field}" size="{size}" '
            f'placeholder="{S.esc(placeholder)}" autocomplete="off">')


def _yes_no(rid: str, field: str) -> str:
    return _radio(rid, field, (("yes", "yes / 是"), ("no", "no / 否")))


def fields_html(record: dict) -> str:
    """The form of one record, by its answer format."""
    key = format_key(record)
    rid = record["id"]
    rows: list[str] = []
    if key in ("V1", "V8"):
        mode = "multi" if key == "V1" else "single"
        what = ("Drag a box on the picture for <b>every</b> visible part and pick its class "
                "below; drag again for the next part." if key == "V1" else
                "Drag <b>one</b> box on the picture around the part asked for; a new drag replaces it.")
        rows.append(f'<div class="row hint">{what}</div>'
                    f'<input type="hidden" data-field="boxes" data-boxmode="{mode}" value="[]">'
                    f'<ul class="boxlist"></ul>')
    elif key == "V2.state":
        allowed = allowed_states(record)
        rows.append('<div class="row">state: ' + (_radio(rid, "state", [(s, s) for s in allowed])
                                                  if allowed else _text("state", "state")) + "</div>")
    elif key == "V2.count":
        rows.append('<div class="row">count: <input type="number" min="0" step="1" '
                    'data-field="count" style="width:6em"></div>')
    elif key in ("V3", "V3.region"):
        rows.append(f'<div class="row">{_verb_select("verb")} '
                    f'{_text("target", "the part acted on, e.g. a motherboard screw")} '
                    f'tool: {_text("tool", "tool, or leave empty", 14)}</div>')
    elif key == "V4":
        rows.append(f'<div class="row">feasible now? {_yes_no(rid, "feasible")}</div>'
                    f'<div class="row">blockers (if not; separate with ;): '
                    f'{_text("blockers", "e.g. the CPU cooler; the fan connector", 50)}</div>')
    elif key == "V6.open":
        rows.append(f'<div class="row">{_verb_select("verb")} '
                    f'{_text("target", "the part, e.g. the PSU")}</div>')
    elif key == "V6.options":
        n = len(record["prompt"].get("options") or [])
        rows.append('<div class="row">choice: ' + _radio(rid, "choice", [(L, L) for L in LETTERS[:n]])
                    + "</div>")
    elif key == "V7":
        rows.append('<div class="row">the remaining plan, one action per line as "verb part" '
                    '(e.g. <code>unscrew the motherboard screws</code>), in order, ending with the '
                    'removal asked for:<br><textarea data-field="plan" rows="6"></textarea></div>')
    elif key == "V10":
        rows.append('<div class="row">actions already done, one per line as "verb part":<br>'
                    '<textarea data-field="done" rows="6"></textarea></div>'
                    '<div class="row">share of the teardown finished: '
                    + _radio(rid, "progress", [(b, b + " %") for b in PROGRESS_BINS]) + "</div>")
    elif key == "V12":
        rows.append(f'<div class="row">did the state change? {_yes_no(rid, "changed")}</div>')
    elif key == "V14":
        allowed = allowed_states(record)
        opts = [(s, s) for s in allowed] + [(CANNOT_TELL, "cannot tell from this view / 看不出")]
        rows.append('<div class="row">state: ' + _radio(rid, "state", opts) + "</div>")
    elif key == "V15.moment":
        rows.append(f'<div class="row">same moment? {_yes_no(rid, "same_moment")}</div>')
    elif key == "V15.cross":
        rows.append(f'<div class="row">state: {_text("state", "state", 16)} best view: '
                    + _radio(rid, "best_view", (("first", "first"), ("second", "second"))) + "</div>")
    elif key == "V16":
        rows.append(f'<div class="row">can the sequence be carried out? {_yes_no(rid, "valid")}</div>'
                    '<div class="row">if not, the number of the first impossible step: '
                    '<input type="number" min="1" step="1" data-field="first_error_step" '
                    'style="width:5em"></div>')
    else:  # pragma: no cover
        rows.append(_text("answer", "answer"))
    return '<div class="fields">' + "".join(rows) + \
        f'<div class="hint">model format: <code>{S.esc(format_instruction(record))}</code></div>' \
        '<div class="timer"></div></div>'


def item_html(index: int, record: dict, manifest: dict, image_root, out_dir) -> str:
    """One item of the form: frames (only the red marks the model sees, no evidence), question,
    options, the form. No answer."""
    key = format_key(record)
    frames = S.frames_html(record, manifest, image_root, out_dir, evidence=[], marks=True,
                           where=False)
    if key in ("V1", "V8"):
        frames = frames.replace('<figure class="frame"', '<figure class="frame drawable"')
    row = S.esc(json.dumps({"id": record["id"], "task": record["prompt"]["task"],
                            "format_key": key}, separators=(",", ":")))
    return f"""
<article class="item" id="{S.esc(record["id"])}" data-id="{S.esc(record["id"])}" data-fk="{S.esc(key)}" data-row="{row}">
<h2><span>#{index}</span><span class="task">{S.esc(record["prompt"]["task"])}</span></h2>
{frames}
{S.question_html(record)}
{fields_html(record)}
</article>"""


HUMAN_GUIDE = """
<b>What to do.</b> Answer every question from the picture(s) alone, the way you would on the
bench; there is no answer key in this page. The parts a question is about are outlined in red
on the picture (several outlines under one letter are one group of identical parts); name parts
in plain words ("the CPU cooler screws", "the PSU", "the fan connector"); verbs are the seven in
the lists. For a plan write one action per line, in order. For the box tasks drag a rectangle on
the picture (V1: one box per visible part, then pick its class; V8: one box per part asked
for).<br>
<b>Timing.</b> The page notes when an item first comes into view and when you last changed its
answer; please work through the items in order, one at a time, without going back, so the times
mean something. Your answers are kept in this browser; press <b>Export answers</b> when the page
is done and send the CSV (<code>tools/vlm_human_sheet.py convert</code> scores it like a model).
"""

JS_HUMAN = r"""
const SHEET = document.body.dataset.sheet;
const KEY = 'tda-human:' + SHEET;
let answers = load(KEY);
let CLASSES = [];
try { CLASSES = JSON.parse(document.body.dataset.classes || '[]'); } catch (e) {}
function items() { return Array.from(document.querySelectorAll('article.item')); }
function collect(a) {
  const out = {};
  a.querySelectorAll('[data-field]').forEach(el => {
    const f = el.dataset.field;
    if (el.type === 'radio') { if (el.checked) out[f] = el.value; else if (!(f in out)) out[f] = out[f] || ''; }
    else out[f] = el.value;
  });
  return out;
}
function filled(fields) { return Object.values(fields).some(v => v && v !== '[]'); }
function restore(a) {
  const st = answers[a.dataset.id]; if (!st || !st.fields) return;
  a.querySelectorAll('[data-field]').forEach(el => {
    const v = st.fields[el.dataset.field]; if (v === undefined) return;
    if (el.type === 'radio') el.checked = (el.value === v); else el.value = v;
  });
  a.querySelectorAll('figure.drawable').forEach(renderBoxes);
}
function progress() {
  const all = items(), done = all.filter(a => answers[a.dataset.id] && answers[a.dataset.id].answered_at);
  document.getElementById('progress').textContent = done.length + ' / ' + all.length;
  all.forEach(a => {
    const st = answers[a.dataset.id] || {};
    a.classList.toggle('done', !!st.answered_at);
    const t = a.querySelector('.timer');
    if (t) t.textContent = st.started_at ? ('started ' + st.started_at.slice(11, 19) +
      (st.answered_at ? ', answered ' + st.answered_at.slice(11, 19) + ' (' + st.seconds + ' s)' : '')) : '';
  });
}
function save(a) {
  const id = a.dataset.id, st = answers[id] || {}, fields = collect(a);
  st.fields = fields;
  if (filled(fields)) {
    st.answered_at = nowIso();
    if (!st.started_at) st.started_at = st.answered_at;
    st.seconds = Math.round((Date.parse(st.answered_at) - Date.parse(st.started_at)) / 1000);
  } else { delete st.answered_at; delete st.seconds; }
  answers[id] = st; store(KEY, answers); progress();
}
function started(a) {
  const id = a.dataset.id, st = answers[id] || {};
  if (!st.started_at) { st.started_at = nowIso(); answers[id] = st; store(KEY, answers); progress(); }
}
function exportCsv() {
  const cols = JSON.parse(document.body.dataset.columns);
  const rows = [cols.map(q).join(',')];
  items().forEach(a => {
    const st = answers[a.dataset.id] || {}, base = JSON.parse(a.dataset.row);
    base.fields_json = JSON.stringify(st.fields || collect(a));
    base.started_at = st.started_at || ''; base.answered_at = st.answered_at || '';
    base.seconds = (st.seconds === undefined) ? '' : st.seconds; base.reviewer = reviewer();
    rows.push(cols.map(c => q(base[c])).join(','));
  });
  download(SHEET + '.' + safeName(reviewer()) + '.csv', rows.join('\r\n') + '\r\n');
}
/* the box tool: drag on the picture, boxes in natural pixel coordinates */
function boxInput(fig) { return fig.closest('article').querySelector('input[data-field=boxes]'); }
function getBoxes(fig) { try { return JSON.parse(boxInput(fig).value || '[]'); } catch (e) { return []; } }
function setBoxes(fig, boxes) { boxInput(fig).value = JSON.stringify(boxes); renderBoxes(fig);
  save(fig.closest('article')); }
function renderBoxes(fig) {
  const svg = fig.querySelector('svg.boxes'), img = fig.querySelector('img');
  if (!img.naturalWidth) return;
  svg.setAttribute('viewBox', '0 0 ' + img.naturalWidth + ' ' + img.naturalHeight);
  svg.querySelectorAll('.human-box').forEach(n => n.remove());
  const boxes = getBoxes(fig), fs = Math.max(12, Math.round(img.naturalWidth / 70));
  boxes.forEach((b, i) => {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', b.bbox[0]); r.setAttribute('y', b.bbox[1]);
    r.setAttribute('width', b.bbox[2]); r.setAttribute('height', b.bbox[3]);
    r.setAttribute('class', 'human-box answer'); svg.appendChild(r);
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', b.bbox[0] + 4); t.setAttribute('y', Math.max(fs, b.bbox[1] - 4));
    t.setAttribute('font-size', fs); t.setAttribute('class', 'human-box');
    t.textContent = (i + 1) + (b['class'] ? ' ' + b['class'] : ''); svg.appendChild(t);
  });
  const list = fig.closest('article').querySelector('ul.boxlist');
  const multi = boxInput(fig).dataset.boxmode === 'multi';
  list.innerHTML = '';
  boxes.forEach((b, i) => {
    const li = document.createElement('li');
    li.textContent = 'box ' + (i + 1) + ' [' + b.bbox.map(v => Math.round(v)).join(', ') + '] ';
    if (multi) {
      const sel = document.createElement('select');
      sel.innerHTML = '<option value="">class ...</option>' + CLASSES.map(c =>
        '<option value="' + c[0] + '"' + (c[0] === b['class'] ? ' selected' : '') + '>' + c[1] + '</option>').join('');
      sel.addEventListener('change', () => { boxes[i]['class'] = sel.value; setBoxes(fig, boxes); });
      li.appendChild(sel);
    }
    const del = document.createElement('button'); del.type = 'button'; del.className = 'quiet';
    del.textContent = 'remove'; del.style.marginLeft = '8px';
    del.addEventListener('click', () => { boxes.splice(i, 1); setBoxes(fig, boxes); });
    li.appendChild(del); list.appendChild(li);
  });
}
function enableDrawing(fig) {
  const svg = fig.querySelector('svg.boxes'), img = fig.querySelector('img');
  svg.style.pointerEvents = 'all'; svg.style.cursor = 'crosshair';
  let start = null, draft = null;
  const nat = e => { const r = img.getBoundingClientRect();
    return [Math.min(Math.max(0, (e.clientX - r.left) / r.width * img.naturalWidth), img.naturalWidth),
            Math.min(Math.max(0, (e.clientY - r.top) / r.height * img.naturalHeight), img.naturalHeight)]; };
  svg.addEventListener('pointerdown', e => { if (!img.naturalWidth) return; e.preventDefault();
    start = nat(e); svg.setPointerCapture(e.pointerId);
    draft = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    draft.setAttribute('class', 'draft'); svg.appendChild(draft); });
  svg.addEventListener('pointermove', e => { if (!start) return; const p = nat(e);
    draft.setAttribute('x', Math.min(start[0], p[0])); draft.setAttribute('y', Math.min(start[1], p[1]));
    draft.setAttribute('width', Math.abs(p[0] - start[0])); draft.setAttribute('height', Math.abs(p[1] - start[1])); });
  svg.addEventListener('pointerup', e => { if (!start) return; const p = nat(e);
    const box = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.abs(p[0] - start[0]), Math.abs(p[1] - start[1])]
      .map(v => Math.round(v * 10) / 10);
    draft.remove(); draft = null; start = null;
    if (box[2] < 2 || box[3] < 2) return;
    const multi = boxInput(fig).dataset.boxmode === 'multi';
    const boxes = multi ? getBoxes(fig) : [];
    boxes.push(multi ? {'class': '', bbox: box} : {bbox: box});
    setBoxes(fig, boxes); });
  img.addEventListener('load', () => renderBoxes(fig));
}
document.addEventListener('DOMContentLoaded', () => {
  items().forEach(a => { restore(a); a.addEventListener('change', () => save(a));
    a.querySelectorAll('textarea, input[type=text], input[type=number]').forEach(el =>
      el.addEventListener('input', () => save(a)));
    a.querySelectorAll('figure.drawable').forEach(enableDrawing); });
  progress();
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) started(en.target); }),
                                        {threshold: 0.5});
    items().forEach(a => io.observe(a));
  }
  items().forEach(a => a.addEventListener('focusin', () => started(a)));
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('reset').addEventListener('click', () => {
    if (!confirm('Clear every answer of this page from this browser?')) return;
    answers = {}; store(KEY, answers); location.reload(); });
});
"""


def _class_list() -> list[list[str]]:
    """``[[class, label]]`` of the taxonomy, for the V1 class menu."""
    from tda.core.export.vlm_tasks import class_label
    from tda.core.taxonomy import load_taxonomy

    tax = load_taxonomy()
    return [[cls, class_label(cls, tax)] for cls in tax.classes]


def build(records: Sequence[dict], exports: Sequence, image_root, out_dir, *,
          batch: int = S.DEFAULT_BATCH, name: str = "human") -> dict:
    """Write the pages and ``<name>.ids.txt``; returns what was written."""
    records = list(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, _ = S.load_context(exports)
    classes = _class_list()
    files: list[str] = []
    parts = S.batches(records, batch)
    for b, chunk in enumerate(parts, start=1):
        body = "".join(item_html((b - 1) * batch + i + 1, r, manifest, image_root, out)
                       for i, r in enumerate(chunk))
        sheet = f"{name}.sheet.b{b:02d}"
        head = f"Teardown questions -- page {b} of {len(parts)} ({len(chunk)} items)"
        text = S.page(head, body, sheet=sheet, js=JS_HUMAN, columns=CSV_COLUMNS,
                      guide=HUMAN_GUIDE)
        text = text.replace("<body ", "<body data-classes=\""
                            + S.esc(json.dumps(classes, separators=(",", ":"))) + "\" ", 1)
        # a participant sees the model's view and nothing else: no reviewer layer, no switch
        text = text.replace(S.REVIEW_TOGGLE, "", 1)
        (out / f"{sheet}.html").write_text(text, encoding="utf-8", newline="\n")
        files.append(f"{sheet}.html")
    (out / f"{name}.ids.txt").write_text("".join(r["id"] + "\n" for r in records),
                                         encoding="utf-8", newline="\n")
    files.append(f"{name}.ids.txt")
    by_task: dict[str, int] = {}
    for r in records:
        by_task[r["prompt"]["task"]] = by_task.get(r["prompt"]["task"], 0) + 1
    return {"items": len(records), "batches": len(parts), "batch_size": batch, "files": files,
            "by_task": dict(sorted(by_task.items(), key=lambda kv: _task_order(kv[0]))),
            "desktops": len({IO.desktop_of(r) for r in records})}


def _task_order(task: str) -> tuple:
    digits = "".join(ch for ch in task[1:] if ch.isdigit())
    return (int(digits or 0), task)


# --------------------------------------------------------------------------- #
# fields <-> model answers
# --------------------------------------------------------------------------- #
def _bool(value) -> Optional[bool]:
    v = str(value or "").strip().lower()
    if v in ("yes", "true", "y", "1", "是"):
        return True
    if v in ("no", "false", "n", "0", "否"):
        return False
    return None


def _lines(text) -> list[str]:
    return [_NUMBERED.sub("", line).strip() for line in str(text or "").splitlines()
            if _NUMBERED.sub("", line).strip()]


def _actions(text) -> list[dict]:
    out = []
    for line in _lines(text):
        verb, _, target = line.partition(" ")
        out.append({"verb": verb.strip().lower(), "target": target.strip()})
    return out


def _boxes(value) -> list[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except ValueError:
            return []
    return [b for b in (value or []) if isinstance(b, dict) and isinstance(b.get("bbox"), list)
            and len(b["bbox"]) == 4]


def to_answer(key: str, fields: dict) -> Optional[dict]:
    """The model-format answer of a filled form, or ``None`` when nothing was answered."""
    f = {k: (v if v is not None else "") for k, v in (fields or {}).items()}
    if not any(str(v).strip() not in ("", "[]") for v in f.values()):
        return None
    if key == "V1":
        return {"components": [{"class": b.get("class") or "", "bbox": b["bbox"]}
                               for b in _boxes(f.get("boxes")) if b.get("class")]}
    if key == "V8":
        boxes = _boxes(f.get("boxes"))
        return {"bbox": boxes[-1]["bbox"]} if boxes else None
    if key == "V2.state":
        return {"state": str(f.get("state", "")).strip()}
    if key == "V2.count":
        try:
            return {"count": int(float(str(f.get("count")).strip()))}
        except (TypeError, ValueError):
            return None
    if key in ("V3", "V3.region"):
        tool = str(f.get("tool", "")).strip()
        return {"verb": str(f.get("verb", "")).strip().lower(), "target": str(f.get("target", "")).strip(),
                "tool": tool or None}
    if key == "V4":
        feasible = _bool(f.get("feasible"))
        if feasible is None:
            return None
        blockers = [b.strip() for b in _SPLIT.split(str(f.get("blockers", ""))) if b.strip()]
        return {"feasible": feasible, "blockers": [] if feasible else blockers}
    if key == "V6.open":
        return {"verb": str(f.get("verb", "")).strip().lower(), "target": str(f.get("target", "")).strip()}
    if key == "V6.options":
        return {"choice": str(f.get("choice", "")).strip().upper()[:1]}
    if key == "V7":
        return {"plan": _actions(f.get("plan"))}
    if key == "V10":
        return {"done": _actions(f.get("done")), "progress": str(f.get("progress", "")).strip()}
    if key == "V12":
        changed = _bool(f.get("changed"))
        return None if changed is None else {"changed": changed}
    if key == "V14":
        state = str(f.get("state", "")).strip()
        if state == CANNOT_TELL:
            return {"answerable": False, "state": None}
        return {"answerable": True, "state": state}
    if key == "V15.moment":
        same = _bool(f.get("same_moment"))
        return None if same is None else {"same_moment": same}
    if key == "V15.cross":
        return {"state": str(f.get("state", "")).strip(), "best_view": str(f.get("best_view", "")).strip()}
    if key == "V16":
        valid = _bool(f.get("valid"))
        if valid is None:
            return None
        step = None
        if not valid:
            try:
                step = int(float(str(f.get("first_error_step")).strip()))
            except (TypeError, ValueError):
                step = None
        return {"valid": valid, "first_error_step": step}
    return None  # pragma: no cover


def gold_fields(record: dict, world: Optional[World] = None) -> dict:
    """The form a perfect participant fills in (the self-test's input)."""
    label, answer = record["label"], record["label"]["answer"]
    key = format_key(record)

    def name(k):
        return S.name_of(k, world)

    def lines(items):
        return "\n".join(f"{a['verb']} {name(a['target'])}" for a in items)

    if key == "V1":
        return {"boxes": json.dumps([{"class": c["class"], "bbox": c["bbox"]}
                                     for c in answer["components"]])}
    if key == "V8":
        return {"boxes": json.dumps([{"bbox": answer["bbox"]}])}
    if key == "V2.state":
        return {"state": answer["state"]}
    if key == "V2.count":
        return {"count": str(answer["count"])}
    if key in ("V3", "V3.region"):
        return {"verb": answer["verb"], "target": name(answer["target_instance"]),
                "tool": answer.get("tool") or ""}
    if key == "V4":
        return {"feasible": "yes" if answer["feasible"] else "no",
                "blockers": "; ".join(name(b) for b in answer.get("blockers") or [])}
    if key == "V6.open":
        return {"verb": answer["verb"], "target": name(answer["target"])}
    if key == "V6.options":
        return {"choice": LETTERS[reference_option(label)]}
    if key == "V7":
        return {"plan": lines(answer["plan"])}
    if key == "V10":
        return {"done": lines(answer["done"]), "progress": answer["progress_bin"]}
    if key == "V12":
        return {"changed": "yes" if answer["changed"] else "no"}
    if key == "V14":
        return {"state": answer["state"] if answer["answerable"] else CANNOT_TELL}
    if key == "V15.moment":
        return {"same_moment": "yes" if answer["same_moment"] else "no"}
    if key == "V15.cross":
        return {"state": answer["state"], "best_view": "second"}
    if key == "V16":
        index = answer["first_error_index"]
        return {"valid": "yes" if answer["valid"] else "no",
                "first_error_step": "" if index is None else str(int(index) + 1)}
    raise ValueError(f"no form for {key}")  # pragma: no cover


def convert(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """CSV rows -> prediction rows ``{"id", "raw_text", "meta"}``; returns ``(rows, stats)``."""
    preds: list[dict] = []
    skipped = passthrough = 0
    seen: set[str] = set()
    for row in rows:
        rid = (row.get("id") or "").strip()
        if not rid or rid in seen:
            continue
        key = (row.get("format_key") or "").strip()
        answer: Optional[Any] = None
        given = (row.get("answer_json") or "").strip()
        if given:
            try:
                answer = json.loads(given)
                passthrough += 1
            except ValueError:
                answer = None
        if answer is None:
            try:
                fields = json.loads(row.get("fields_json") or "{}")
            except ValueError:
                fields = {}
            answer = to_answer(key, fields) if key else None
        if answer is None:
            skipped += 1
            continue
        seen.add(rid)
        meta = {"task": row.get("task"), "format_key": key, "reviewer": row.get("reviewer") or "",
                "started_at": row.get("started_at") or "", "answered_at": row.get("answered_at") or ""}
        try:
            meta["seconds"] = float(row["seconds"]) if str(row.get("seconds", "")).strip() else None
        except (TypeError, ValueError):
            meta["seconds"] = None
        preds.append({"id": rid, "raw_text": json.dumps(answer, ensure_ascii=False), "meta": meta})
    return preds, {"answered": len(preds), "skipped": skipped, "answer_json_rows": passthrough}


def read_csvs(paths: Sequence) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                rows.append({(k or "").strip().lower(): (v or "") for k, v in raw.items()})
    return rows


def cut_truth(exports: Sequence, ids: set[str], out_dir: Path, name: str) -> dict[str, int]:
    """Each truth file cut to ``ids`` with its sidecars: ``<out>/<name>.<view>.jsonl``."""
    import vlm_eval_set

    written: dict[str, int] = {}
    for path in exports:
        records = [r for r in IO.read_jsonl(path) if r["id"] in ids]
        if not records:
            continue
        view = Path(path).stem.replace("vlm_", "")
        target = out_dir / f"{name}.{view}.jsonl"
        IO.write_jsonl(target, records)
        vlm_eval_set._write_sidecars(path, target, records)
        written[view] = len(records)
    return written


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the pages, no answers")
    b.add_argument("--export", nargs="+", required=True, help="the truth files (eval_v1/<view>.jsonl)")
    b.add_argument("--image-root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--ids", default=None, help="a file of record ids (else a stratified draw)")
    b.add_argument("--size", type=int, default=DEFAULT_SIZE)
    b.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--batch", type=int, default=S.DEFAULT_BATCH)
    b.add_argument("--name", default="human")
    c = sub.add_parser("convert", help="exported CSVs -> prediction JSONL (+ cut truth files)")
    c.add_argument("--csv", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--export", nargs="*", default=(),
                   help="truth files to cut to the answered ids, beside the predictions")
    c.add_argument("--name", default="human")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        records = [r for p in args.export for r in IO.read_jsonl(p)]
        if args.ids:
            wanted = [l.strip() for l in Path(args.ids).read_text(encoding="utf-8").splitlines()
                      if l.strip()]
            by_id = {r["id"]: r for r in records}
            # the ids file's order is the order shown
            chosen = order_pairs([by_id[i] for i in dict.fromkeys(wanted) if i in by_id])
            missing = set(wanted) - {r["id"] for r in chosen}
            if missing:
                print(f"[vlm-human-sheet] warning: {len(missing)} ids are in no truth file")
        else:
            chosen = draw_subset(records, size=args.size, per_task=args.per_task, seed=args.seed)
        result = build(chosen, args.export, args.image_root, args.out, batch=args.batch,
                       name=args.name)
        print(f"[vlm-human-sheet] {result['items']} items ({result['desktops']} desktops) in "
              f"{result['batches']} pages -> {args.out}: "
              + " ".join(f"{t}={n}" for t, n in result["by_task"].items()))
        return 0
    rows = read_csvs(args.csv)
    preds, stats = convert(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    IO.write_jsonl(out / f"{args.name}.pred.jsonl", preds)
    msg = (f"[vlm-human-sheet] {stats['answered']} answers -> {out / (args.name + '.pred.jsonl')}"
           f" ({stats['skipped']} rows without an answer skipped)")
    if args.export:
        cut = cut_truth(args.export, {p["id"] for p in preds}, out, args.name)
        msg += "; truth cut to the answered ids: " + ", ".join(f"{v} {n}" for v, n in cut.items())
        msg += (f"\n[vlm-human-sheet] score with: python tools/vlm_score.py --truth "
                f"{out / (args.name + '.<view>.jsonl')} --pred {out / (args.name + '.pred.jsonl')} "
                f"--out {out} --name {args.name}-<view>")
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
