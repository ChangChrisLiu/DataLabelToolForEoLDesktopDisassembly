# VLM question redesign — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Four lanes run **in parallel**, one agent each, in its own git worktree; each lane owns
> the files listed for it and touches no other lane's files. Steps use checkbox (`- [ ]`) syntax.

**Goal:** questions name parts by a red outline on the scanner image (never by ordinal), interchangeable parts are
asked and answered as groups, action tasks show all four views of a moment, V4 loses its empty negatives, and the
model is told what each part and state means.

**Architecture:** the generator (`vlm_tasks.py`) decides groups, marks and multi-view images and writes them into the
records; the converter draws the marks and adds a glossary; the grader grades group answers; the export writes a new
`multi` unit and the subset is re-drawn as `eval_v2`.

**Tech stack:** Python 3.12 (`D:/Anaconda/envs/tda/python.exe`), pytest, Pillow, the `tda` package.

**Spec:** `docs/superpowers/specs/2026-11-03-vlm-question-redesign-design.md` (read it first; §6 holds the builder
calls, all binding).

## Global constraints

* Never open `D:/DataSet/annotations/tda.sqlite` for writing; tests use the scenes in `tests/w3b_scene.py` /
  `tests/vlm_scene.py`. Never write into `exports/`. Everything on `D:`; `PYTHONDONTWRITEBYTECODE=1`;
  `QT_QPA_PLATFORM=offscreen`; `TMP=TEMP=D:/DataSet/.cache/tmp`.
* Run tests with `D:/Anaconda/envs/tda/python.exe -m pytest -q -p no:cacheprovider --basetemp=D:/DataSet/.cache/tmp/<lane>/bt <files>`.
* No venue or conference names anywhere (the repository is public).
* A question never names a part by ordinal ("RAM module 3"); V2 count questions are the one class-level exception
  that stays as it is.
* Marks: red, never drawn in V1 / V8 / V14.
* Commit in your worktree with messages ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Shared record contract (every lane codes against this)

**Group ids.** `group_id = f"{match_class}@{anchor}"`, `anchor` = the instance key the members share (`fastens` for
screws, `attrs.of` for latches, `socket_host` + `|` + cable node for connectors, `-` for RAM modules / cages / cards /
anchor-less parts). Example: `screw.cpu_cooler@cpu_cooler.fan.01`, `ram_latch@ram_module.01`, `ram_module@-`.
A group with one member is not a group: its member is named by its instance key.

**Oracle desktop line** (`data/vlm_<unit>.oracle.jsonl`, `type: desktop`) gains
`"groups": {gid: {"members": [keys], "match_class": str, "anchor": str, "label": str}}` (label = plural class phrase,
e.g. `"CPU cooler screws"`), and every `instances[key]` gains `"group": gid | null`.

**Prompt fields** (what the model sees):

* `prompt.marks: [{"image": int, "bbox": [x, y, w, h], "tag": str}]` — `image` indexes `prompt.images`; bbox in that
  image's native pixels; `tag` `"A"`, `"B"`, … (one tag per part or group; a group's members share its tag).
  Omitted when empty.
* `prompt.terms: [class ids named in the question / options]`, `prompt.state_terms: [class ids whose states the
  question asks]` — the converter turns them into a glossary. Omitted when empty.
* `prompt.image_legend: [{"view": "scan"|"oak1"|"oak2"|"rs", "moment": "now"|"before"|"after"}]`, one per image,
  only on multi-view records (the converter writes the legend line; the pack keeps views out of the question text).

**Label fields:** multi records have `label.view = "multi"`, `label.views = [views present, in image order]`,
`label.image_steps = [step per image]`. Gold action targets (`V4.blockers`, `V6`, `V7.plan`, `V10.done`, `V16.plan`)
are instance keys **or group ids**. V3 gains `answer_check.accept_targets: [keys]`. V10 gains
`answer.started: [group actions partly done]`. V8 multi-instance records use
`answer_check.type = "components"` with `class_filter: match_class` and `answer = {"components": [...]}` (V1's shape).
Skip reasons added: `unmarkable_part`, `captive_tightness`, `invisible_action`, `mixed_group`.

## Review focus

1. A group whose members are partly removed — the marks must cover only the members still present (lane A test).
2. A record whose only side view is missing a photo — `label.views` / `image_legend` shorter than 4, marks still on
   image 0 (lane A + D tests).
3. A model that names a group by its singular ("a cooler screw") — the grader grounds it to the group (lane B test).
4. A mark on an image the converter downsizes (`MAX_SIDE` 1568 on a 4056 px OAK frame — marks are only on the scan
   image, but the scaling must be applied) (lane C test).
5. An old `eval_v1` path passed to the tools — a clear error naming `eval_v2` (lane D test).

---

### Lane A — generator (`tda/core/export/vlm_groups.py` new, `tda/core/export/vlm_tasks.py`, `tda/core/export/vlm_reasoning.py`; tests `tests/test_vlm_groups.py` new, `tests/test_vlm_tasks.py`, `tests/test_w3b_*.py`)

**Interfaces produced:** `part_groups(ctx: DesktopCtx) -> dict[str, PartGroup]` (instance key → group, only groups
of ≥ 2 members), `@dataclass(frozen=True) PartGroup(gid: str, members: tuple[str, ...], match_class: str, anchor: str,
label: str)`; `mark_box(tc: TaskCtx, frame: FrameData, key: str) -> Optional[list]` (visible-mask box from
`frame.pointable`, else the amodal box of `tda/core/vis_review.amodal_box`, else `None`); `TaskCtx.side_frames:
dict[str, dict[int, FrameData]]` (filled by lane D for the `multi` unit) and `TaskCtx.moment(step) -> list[FrameData]`
(scan frame first, then oak1, oak2, rs where present); module constant `ACTION_TASKS = ("V3", "V4", "V6", "V7",
"V10", "V12", "V16")`, `PERCEPTION_TASKS_SINGLE = ("V1", "V2", "V8", "V14", "V15")`.

- [ ] **A1 groups and marks.** Tests in `tests/test_vlm_groups.py`: `test_cooler_screws_form_one_group` (4 cooler
  screws fastening one fan → one gid `screw.cpu_cooler@cpu_cooler.fan.01`, label `"CPU cooler screws"`);
  `test_latches_group_per_module`; `test_singleton_is_no_group`; `test_mark_box_uses_amodal_when_hidden` (an
  `occluded_full` part gets its amodal box); `test_mark_box_none_when_removed`. Implement, run, commit.
- [ ] **A2 naming.** Replace ordinal labels in questions: `question_label` returns `"the <class phrase> outlined in
  red"` (or `"part A"` … when the question names several) and the record gets `prompt.marks`; groups use the plural
  (`"the CPU cooler screws outlined in red"`), members still in the asked situation only (spec §6.1). Unmarkable →
  skip `unmarkable_part`. V1 / V8 / V14 never get marks. Tests: `test_v2_question_has_mark_and_no_ordinal`,
  `test_v4_hidden_target_marked_with_amodal_box`, `test_v1_v8_v14_have_no_marks`, `test_no_ordinal_in_any_question`
  (regex `\b[A-Za-z]+ \d+\b` absent from every question of a scene export except V2 counts). Commit.
- [ ] **A3 V8 / V14 / V2.** V8 on a class with several instances → `components` record (contract); V14 asked by class
  / group only when every member gives the same answer, else skip `mixed_group`; V2: members of a group in different
  states asked one by one with their own mark; V2 counts unchanged. Tests: `test_v8_multi_instance_is_components`,
  `test_v14_mixed_group_skipped`. Commit.
- [ ] **A4 V4.** Drop `state_inapplicable` negatives; skip (`captive_tightness`) any V4 whose truth depends on
  captive / cooler screw tightness; blockers as group ids (cooler screws → the group). Tests:
  `test_v4_has_no_state_inapplicable`, `test_v4_fan_blockers_are_screw_group_and_connector`,
  `test_v4_captive_dependent_skipped`. Commit.
- [ ] **A5 V3 / V6 / V7 / V10 / V12 / V16 on groups and multi-view.** Every action-task record uses
  `tc.moment(step)` images (V3 / V12: moment(k-1) + moment(k)) and writes `image_legend`, `views`, `image_steps`;
  V3 needs the target (or group) visible ≥ `V3_MIN_PX = 400` px on the scanner before or after, or a `BIG_PARTS` part
  (else skip `invisible_action`), and writes `accept_targets` = target + parts attached to it (**superseded**, owner
  2026-11-03: `accept_targets` is the target alone; a member of its group still stands for it); V6 options / V16 plans
  / V7 plans / V10 done use group actions; V6 distractors and V16 corruptions prefer graph-blocked over
  state-inapplicable; "loosen the cooler screws" stays required in V7 / V16 / V10 until the held part comes off. Tests:
  `test_action_record_images_marked_scan_alone` (was `test_action_record_has_four_images_scan_first`), `test_missing_side_view_shrinks_legend`,
  `test_v3_tiny_target_skipped`, `test_v3_accepts_attached_part`, `test_v7_plan_uses_screw_group`,
  `test_v10_partial_group_is_started`. Commit.
- [ ] **A6 oracle groups.** `_desktop_world` (in `vlm.py`, the only `vlm.py` edit lane A makes — coordinate by
  touching only that function) writes `groups` and `instances[*].group`. Test `test_oracle_carries_groups`. Commit.

### Lane B — grader (`tda/core/export/vlm_check.py`, `scripts/vlm_score.py`; tests `tests/test_vlm_check_groups.py` new, `tests/test_w3b_scripts.py`, `tests/test_w3b_v6_scoring.py`, `tests/test_w3b_v7.py`)

**Consumes:** the contract above (groups in the oracle world, group-id targets, `components` V8, `accept_targets`,
`started`). **Produces:** `World.group_of(key) -> Optional[str]`, `World.groups: dict`; `Grounder.ground_group(text,
eligible: Sequence[str]) -> Optional[str]` (gid or key).

- [ ] **B1 grounding to groups.** Tests: `test_plural_and_singular_ground_to_group` ("the cooler screws", "a CPU
  cooler screw" → the gid), `test_instance_key_still_grounds`, `test_two_groups_one_class_graded_by_count`. Commit.
- [ ] **B2 task grades.** V4 blockers F1 over groups / keys; V7 / V16 plans and V10 done sets over group actions
  (V10 `started` graded as its own set); V3 accepts any of `accept_targets` (**superseded**, owner 2026-11-03: only the
  part acted on, or its group; a compound step accepts any of its actions); V8 `components` = V1's grade restricted
  to `class_filter`. Tests: `test_v4_group_blockers_f1`, `test_v7_group_plan_violation_free`,
  `test_v10_started_set`, `test_v3_accept_targets`, `test_v8_components_grade`. The gold reply must still grade 1.0
  for every task (`test_gold_scores_one` over a lane-B fixture). Commit.

### Lane C — what the model and the auditor see (`scripts/vlm_to_chat.py`, `scripts/vlm_glossary.py` new, `scripts/vlm_audit_sheet.py`, `scripts/vlm_human_sheet.py`, `scripts/vlm_audit_carry.py` new; tests `tests/test_vlm_to_chat_marks.py` new, `tests/test_w4e_audit.py`)

**Consumes:** `prompt.marks`, `prompt.terms`, `prompt.state_terms`, `prompt.image_legend`, `label.views`.
**Produces:** `CHAT_FORMAT_ID = "chat-v2"`; `scripts/vlm_glossary.py` with `CLASS_TEXT: dict[str, str]` (copy the
descriptions of `CLASS_DESCRIPTIONS` in `scripts/export_scanpack.py`) and `STATE_MEANINGS: dict[str, dict[str, str]]`
(one line per state of every class with ≥ 2 tracked states in `configs/taxonomy.yaml`; draft — the owner reviews it),
`glossary_text(terms, state_terms) -> str`.

- [ ] **C1 marks drawn.** `Images` draws each mark (red `#ff1744`, width `max(3, round(3 * w / 1600))`, the tag in a
  filled label at the box's top-left) on an in-memory copy, after scaling with the same factor as the image. Tests:
  `test_mark_drawn_at_scaled_box` (a 1600 px image, `MAX_SIDE` 800, mark at [100, 100, 50, 50] → red pixels around
  [50, 50, 25, 25]), `test_pack_image_untouched`. Commit.
- [ ] **C2 legend and glossary.** `user_text` starts multi-view prompts with `"Images 1–4: the same moment from the
  top-down scanner, then three side cameras (OAK 1, OAK 2, RealSense)."` (or `"Images 1–4 before, images 5–8 after
  the action, each: top-down scanner, OAK 1, OAK 2, RealSense."`; fewer images → name only those present), names the
  marked parts' tags, and ends the prompt with `glossary_text(...)`. Tests: `test_multiview_legend_lists_present_views`,
  `test_glossary_has_class_and_states`. Commit.
- [ ] **C3 audit sheet.** Draw marks (red) and show up to 8 images in two rows (before / after); keep the
  `imgbox` overlay fix. Test: `test_audit_sheet_draws_marks`. Commit.
- [ ] **C4 audit carry-over.** `scripts/vlm_audit_carry.py --old-csv <csvs> --old-records <old audit.jsonl>
  --new-records <new audit.jsonl> --out <csv>`: carries V2 verdicts whose desktop + step + part (the asked instance
  or its group) and answer match; everything else blank. Test: `test_carry_v2_by_desktop_step_part`. Commit.

### Lane D — export layout and subset (`tda/core/export/vlm.py` except `_desktop_world`, `scripts/export_vlmpack.py`, `scripts/vlm_eval_set.py`, `scripts/vlm_references.py`, `scripts/vlm_selfcheck.py`, `scripts/vlm_sample.py`, `scripts/vlm_run/*`; tests `tests/test_w4a_vlmpack.py`, `tests/test_w4a_eval_set.py`, `tests/test_w4c_references.py`, `tests/test_w4b_vlm_run.py`)

**Consumes:** `TaskCtx.side_frames`, `TaskCtx.moment`, `ACTION_TASKS` / `PERCEPTION_TASKS_SINGLE` (lane A).
**Produces:** `export_vlm(..., view="multi")`; pack files `data/vlm_multi.jsonl` (+ sidecars; images of all four
views), per-view files with `PERCEPTION_TASKS_SINGLE` only; `eval_v2/` (`scan.jsonl`, `oak1.jsonl`, `oak2.jsonl`,
`rs.jsonl`, `multi.jsonl`, `audit/`); `manifest.eval_version = "eval_v2"`.

- [ ] **D1 `multi` unit.** `export_vlm(view="multi")` reads the scanner view as the primary (truth, marks) and the
  side views into `TaskCtx.side_frames` (the `_other_views` reading, all three side views), generates
  `ACTION_TASKS` only; per-view exports generate `PERCEPTION_TASKS_SINGLE` only. Tests:
  `test_multi_unit_has_only_action_tasks`, `test_view_units_have_only_perception_tasks`. Commit.
- [ ] **D2 pack and subset.** `export_vlmpack.py` writes the five units, `eval_v2` (same per-task targets; action
  tasks drawn from `multi`), README text for the new layout (marks, multi-view, groups, glossary, `eval_v2`); the
  mask overlay paints image 0 of multi records only; the state oracle gives one table per moment; selfcheck covers
  `multi`. The pack's `tools/` also ships lane C's new `vlm_glossary.py` and `vlm_audit_carry.py`. An `eval_v1`
  path → `SystemExit` naming `eval_v2`. Tests: `test_pack_has_multi_and_eval_v2`,
  `test_overlay_paints_scan_image_only`, `test_state_oracle_one_table_per_moment`, `test_selfcheck_multi_gold_one`,
  `test_eval_v1_path_refused`. Commit.
- [ ] **D3 run scripts.** `scripts/vlm_run/run_eval.sh` and Slurm array cover `VIEW=multi`; `submit_matrix.sh`
  views `scan oak1 oak2 rs multi`. Test: `test_run_eval_accepts_multi`. Commit.

### Integration (controller, after the four lanes)

- [ ] Merge lanes A → B → C → D into `main`; resolve; full VLM + export test suite green.
- [ ] Rebuild the VLM pack from `.cache/tmp/seal3/sealed_20261103_v3.sqlite`; self-check 1.0 on all five units;
  clean-venv check; no ordinal in any question (script over the pack); marks render on 20 sampled records.
- [ ] Owner reviews `STATE_MEANINGS`; carry the 75 verdicts over; new audit pages.
- [ ] Docs (pack README via the exporter, HANDOFF §7–8 and §12, run guide, VLM explainer, card), HF upload, hash check.
