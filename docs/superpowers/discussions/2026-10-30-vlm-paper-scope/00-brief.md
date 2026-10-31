# Discussion brief for the Opus 5.5 discussant

You are a discussant, not an implementer. The user (a PhD student building the E3D desktop-PC disassembly dataset for a
**CVPR dataset-track paper with an evaluation section**, deadline 2026-11-16 AoE, 17 days away) asked for a thorough
two-model discussion tonight; they will read it in the morning and decide. Your counterpart is the controller (Fable),
whose opening position is in `01-fable-position.md`. Argue with it. Agreement without reasons is useless; so is
disagreement for show. Where you disagree, say what evidence would settle it and, if that evidence is in the files
below, go get it.

## Read first (in this order)

1. `.superpowers/sdd/2026-10-30-vlm-discussion/01-fable-position.md` — the position you are responding to.
2. `docs/superpowers/specs/2026-10-30-vlm-information-layers-design.md` — the information layers L0–L5, capability
   mapping, paper contributions, gaps, §6 open questions.
3. `docs/superpowers/specs/2026-09-17-teardown-annotator-design.md` §8 (lines ~647–720): the VLM task list V1–V18,
   truth principle, hard negatives, evaluation plan as written weeks ago. Also §3's `Instance` row (line ~143) for the
   meaning of `mounted_on`.
4. `.superpowers/sdd/2026-09-20-user-trial/U46-review.md` and `U46-report.md` — the precedence-graph export and the
   data findings (mounted_on never filled, CPU lever has no incoming edge, D63 stale rows, D65 hand events).
5. `.cache/tmp/u46/r2b/summary.json` — the real run: totals, candidates by class pair (with desktop lists and
   `above_removed_first` / `below_removed` counts), violations.
6. `docs/HANDOFF_2026-10-28.md` §1–§5 and §11 — what exists, what is on HF, what the user said tonight.
7. `tda/core/export/vlm.py` (skim: how samples are generated, `answer_check`), `tda/core/graph_rules.py` (how rule
   edges are declared; what adding a cooler→lever edge would touch), `tda/core/export/graph_covers.py` (what counts as
   `structural`).

Read-only database copy for any probe: `sqlite3.connect("file:D:/DataSet/.cache/tmp/hf/tda.sqlite?mode=ro", uri=True)`.
Interpreter `D:/Anaconda/envs/tda/python.exe` (bare `python` fails). Set `sys.stdout.reconfigure(encoding="utf-8")`.

## What to write

`D:/DataSet/.superpowers/sdd/2026-10-30-vlm-discussion/02-opus-response.md`, in **Simplified Chinese** (the user reads
it), English only for identifiers. Structure:

1. **逐条回应** §1–§8 of the position: 同意 / 反对 / 部分同意, the reason, your alternative, and what would settle it.
   Numbers you cite must come from the files or your probes; say where.
2. **你认为控制方漏掉的**: things a CVPR dataset-track reviewer will ask that neither document addresses (e.g. inter-annotator
   agreement with a single annotator; label noise from a model-assisted pipeline; dataset size vs. claims; license; ethics
   of hands in frame; comparison table against existing assembly/disassembly/procedural datasets — name the real ones you
   know); things a VLM-evaluation practitioner would flag (contamination between train/test templates, answer-format
   brittleness, how legal-set scoring behaves when the graph is incomplete, closed-model API nondeterminism, the
   text-only baseline being too strong because the log leaks).
3. **17 天的可行性审计**: go through the schedule in §8 and say what is not credible, with the reason. Propose the
   minimum paper that is still a real CVPR submission and what to cut first.
4. **两条全局规则改动**（§5 `mounted_on`、§6 cooler→lever edge）: argue each as a data-quality reviewer. For §5 check the
   class list against the taxonomy (`tda/core/taxonomy.py`) and against what the summary's candidate pairs actually
   contain; for §6 check in the database copy that the claim "no desktop opens the lever before removing the cooler" holds
   (and how heatsink_fan desktops look), and look at how `graph_rules.py` would express the condition.
5. **给用户的决定清单**: at the end, the decisions only the user can make, each with the two or three real options and
   which you recommend, in one line each.

Length: whatever the content needs, but no padding. Do not restate the position; respond to it.

## Rules

- Read-only everywhere. Never open `D:/DataSet/annotations/tda.sqlite`. Never write under `D:/DataSet/cache/`, the raw
  drive (`G:` / `F:`), or anywhere on `C:`. Scratch files only under `D:/DataSet/.cache/tmp/vlmdisc/` (create it).
- Do not modify repository files other than writing `02-opus-response.md`. No git commands that change state.
- Do not run the test suite or `export-graph`; the summary is already there.
- `TMP=D:/DataSet/.cache/tmp TEMP=D:/DataSet/.cache/tmp CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8` for any process.
- No downloads, no network.
