# VLM question redesign after the first author-audit pass (design)

Date: 2026-11-03. Owner: the dataset owner. Status: proposed, awaiting the owner's approval (builder check done; captive screws decided).

## 1. Why

The owner audited 75 items of `eval_v1/audit` (V2, V3, V4). The generated answers were right: no item was wrong
(the one marked "wrong" was marked only to carry a comment). What failed is how the questions are put:

1. **Instance ordinals mean nothing to a viewer.** "RAM module 3", "Connector 1", "Cover 1", "CPU cooler screw 2":
   neither a person nor a model can tell which part is meant.
2. **Interchangeable parts are asked one by one.** The four CPU-cooler screws, the motherboard screws, the latches of
   one RAM module have no order among them; asking about "screw 2" is meaningless, and because the tightness of a
   screw is invisible in a photograph, the honest answer to "can the CPU fan come off?" is always "check that all four
   cooler screws are loosened, and unplug the fan connector".
3. **Some V4 negatives are empty.** "Can you remove the PSU?" when the PSU was removed long ago, "can you open RAM
   latch 1?" when it is open, "can you unplug connector 1?" when it is out (`truth_source: state_inapplicable`). The
   current state is V2's question; as an affordance question these teach nothing.
4. **One side view often cannot show the part.** OAK / RealSense records of V3 / V4 / V6 ... are single side views in
   which the asked part is frequently invisible. Questions about what can be done should see the machine from all
   cameras at once.
5. **Class names need a definition** (what a drive cage is; what "open" means for a cable clip).
6. **V3 asks about actions nobody can see** (a tiny connector unplugged under cables), and when two parts move as one
   (D48: the shroud with its fan) only one is accepted.

No model has been run yet, so changing the questions now costs no experiment.

## 2. Decisions taken by the owner (2026-11-03)

* **Naming = mark it on the image, then ask** (owner: 「直接标出来，然后去问这个问题」). Every part a question is
  about is outlined on the scanner image and the question says "the RAM module outlined in red"; interchangeable
  parts are asked as a group with every member outlined. If the part cannot be outlined, the item is not generated.
* **Multi-view for every action task.** V3, V4, V6, V7, V10, V12 and V16 show the same moment from all four cameras
  (scanner + OAK1 + OAK2 + RealSense; two moments for V3 / V12), one record per step instead of one per view.
  V1, V2, V8, V14 stay single-view (perception); V15 (are two views the same moment) is unchanged.
* The scorer stays strict on names it cannot ground; the human baseline is out of the plan; the author audit covers
  the generated answers only (V1 / V8 not reviewed).

## 3. Design

### 3.1 Groups of interchangeable parts

A **part group** inside one desktop = the instances that share
`match_class` (class, refined by screw `role`, cooler / connector `kind`) **and** the same relational anchor:
screws by what they fasten (`fastens`), latches by what they lock (`attrs.of`), connectors by their cable owner
(the `cable` node) + host (`socket_host`), RAM modules, drive cages and expansion cards by class + kind. Examples: "the CPU cooler screws" (4),
"the motherboard screws" (6–10), "the latches of the RAM modules", "the front-panel connectors", "the PSU cable
connectors on the motherboard", "the RAM modules".

A group is used in a question only when **every member is in the same situation for that question** (same state for
V2, same feasibility and the same blocker groups for V4, all still to be done for V7 / V10). A screw of a group whose
tightness the picture cannot show (captive / cooler screws) is always reported with its whole group: "loosen the CPU
cooler screws" is one requirement, whatever the log says about each screw (the owner's rule).

Answers name groups as well: V4 `blockers`, V7 plans, V10 done sets, V6 / V16 actions are lists of **group actions**
(`{"verb": "unscrew", "target": "the CPU cooler screws"}`); the grader grounds a model's wording to a group (the
existing `match_class` grounding, extended with the anchor) and grades sets of groups. One group action replaces the
N member actions in plans (V7 / V16) — the graph's order among members is "unordered" already.

### 3.2 Naming a single part

**Every part the question names is marked on the scanner image** (owner's rule): `prompt.marks =
[{"image": i, "bbox": [x, y, w, h], "tag": "A"}]`, drawn by the converter (3.5), never by the model. The question
names it by class and mark — "the RAM module outlined in red", or with several marks "part A", "part B" — and never
by an ordinal.

* The box is the part's **visible-mask box** on the confirmed scanner frame of the question; for a part that is
  present but hidden (`occluded_full`, or under another part) it is its **amodal box** (the full outline the
  annotation keeps), so "can the CPU come off?" under the cooler can still be asked.
* A **group** (3.1) is marked member by member, all with the same tag: "the screws outlined in red".
* A part that cannot be marked — removed, out of view, or on a step without a confirmed scanner frame — is **not
  asked about** (skip reason `unmarkable_part`).
* Marks are used where a box does not give the answer away: V2, V3 (the "before" image, only when the question
  names a part), V4, V6 (targets of the options), V7 (the goal), V10 / V16 (parts named in the plan). **Never** in
  V1, V8 (the box is the answer) or V14 (a box would say the part is visible). In V1 / V8 / V14 the class name alone
  is used, as today (V8's relational phrases stay where they name exactly one part).
* Answers may still name parts in words; the grader grounds them to a class / group (3.1), so a model is not asked
  to read the tag back.

### 3.3 V4 without empty negatives

Drop `state_inapplicable` negatives (`not_present`, `already_open`, `already_unplugged`, ...). V4's "no" comes only
from the graph (`graph_blocked`, `graph_blocked_pixels`) and from real failed attempts (`failed_attempt`). The one
"no" per "yes" pairing stays (same verb and class where possible). Blockers are blocker groups (3.1); for the CPU fan
the gold is `["the CPU cooler screws", "the CPU fan connector"]`. Records whose only available negative was
state-inapplicable lose their pair (and their positive, to keep the balance), so V4 shrinks; the eval target (400)
is re-checked against the new pool.

V6's multiple-choice distractors and V16's corruptions also stop using state-inapplicable actions where a
graph-blocked one exists (an option "remove the PSU" when there is no PSU is not a distractor, it is a giveaway).

### 3.4 Multi-view records

A new export unit **`multi`**: for each step with a scanner frame, one record per action-task question whose
`prompt.images` are the frames of that step in the four views, in the fixed order scan, oak1, oak2, rs (views with
no photo of that step are left out and listed in `label.views`); V3 / V12 have the four views of step k-1 then of
step k. Truth, rationale and evidence come from the log, the graph and the scanner frame (the only confirmed view);
`view_confirmed` = the scanner frame is confirmed.

Consequences:

* `data/vlm_<view>.jsonl` for scan / oak1 / oak2 / rs keep **V1, V2, V8, V14** (perception) and **V15**; a new
  `data/vlm_multi.jsonl` holds **V3, V4, V6, V7, V10, V12, V16**. The per-view duplicates of action questions
  disappear (today one action is asked once per view).
* `eval_v1` is re-drawn: perception per view as before, action tasks from `multi`; the same per-task targets
  (V4 400, V6 600 as 300 pairs, V12 400, V16 300, V3 / V7 / V10 300) where the pool allows.
* The mask-overlay variant paints the scanner image of a record only; the state-oracle variant gives one state
  table per moment.
* Image budget: up to 4 images per record (8 for V3 / V12). The closed model answers 300 items only.

### 3.5 What the model is shown

`vlm_to_chat.py`:

* draws each `prompt.marks` box on its image (red, 3 px at 1600 px, scaled; the copy is made in memory, the pack's
  image stays as it is) before encoding;
* prefixes multi-image prompts with one line naming the images ("Image 1: top-down scanner; images 2–4: three side
  cameras, all at the same moment" / "images 1–4 before, 5–8 after");
* appends a **glossary** for every class named in the question or the options: the class description (from the
  scanner pack's `class_description`) and, for state questions, the state vocabulary with one-line meanings (a new
  table `STATE_MEANINGS`, e.g. cable clip `open` = the clip is released, the cable can be lifted out; `closed` = it
  holds the cable — **the owner reviews this table**).

`CHAT_FORMAT_ID` becomes `chat-v2`.

### 3.6 V3: only visible actions; parts that move together

* A V3 record is generated only if the acted-on part (or its group) is visible on the scanner frame before or after
  the action with at least `V3_MIN_PX` visible pixels (proposed 400 px at 1600 × 1600), or it is a large part
  (`BIG_PARTS`). The D17 connector of the audit is dropped by this rule.
* ~~When the target carries attached parts (`parent` + `attached`, e.g. D48's shroud on its fan, captive screws), the
  gold accepts the target or any part attached to it (`answer_check.accept_targets`).~~ **Withdrawn (owner,
  2026-11-03, after the real build: 156 V3 records accepted e.g. a RAM latch for the motherboard).** The gold accepts
  the part acted on alone: `answer_check.accept_targets = [target]`, and the scorer never credits an attached part.
  A member of the target's own group of interchangeable parts still stands for it (`target_group`).

### 3.7 Unchanged

The truth principle (demonstrated / graph-blocked / failed attempt), V1 / V8 / V14 / V15, the scorer's strict and
lenient parsing, the bootstrap, the reference lines, the split, every annotation and the sealed snapshot v3.

## 4. Work and order

1. Groups and marks (`vlm_tasks.py`: group key; `question_label` → class phrase + mark tag; `prompt.marks` from the
   visible or amodal box in `record()`), with unit tests on small scenes (visible part, hidden part with an amodal
   box, group, unmarkable part skipped, V1 / V8 / V14 unmarked).
2. V4 negatives (3.3) and V6 / V16 distractors.
3. Group answers in V4 / V6 / V7 / V10 / V16 and the grader (`vlm_check.py`: ground to groups, grade group sets).
4. The `multi` export unit (`vlm.py`, `export_vlmpack.py`), eval-set re-draw (`vlm_eval_set.py`), overlay and state
   oracle (`vlm_references.py`), audit sheet (marks, up to 8 images).
5. Converter: marks, image legend, glossary, `STATE_MEANINGS` (`vlm_to_chat.py`); the audit sheet shows the same.
6. V3 visibility and attached parts (3.6).
7. Rebuild the VLM pack from snapshot v3; self-check 1.0; clean-venv check; all VLM tests; the owner's 75 audit
   verdicts carried over where a record survives unchanged in substance (same readable id, same answer).
8. Docs (pack README, HANDOFF §7–8, run guide, VLM explainer), HF upload and hash check.

Estimated 1.5–2 days. Items 1–3 change records in every view; 4 changes the pack layout; 5–6 are local.

## 5. Open points for the owner


* The `STATE_MEANINGS` table (3.5) — one line per state of every askable class.
* `V3_MIN_PX` (400 px proposed).
* Whether the side views are still reported as separate rows anywhere (with this design they are not: action tasks
  are only multi-view).

## 6. Calls made after the builder check (the builder's questions, settled)

1. **A group partly done** (3 of 8 motherboard screws out; RAM latches opened one per step): the group in a question
   is **the members still in the asked situation** — "the motherboard screws outlined in red" marks the remaining
   ones; V7 / V16 plans carry one group action for the remaining members; V10's done set counts a group action as done
   only when every member is done, and grades the partly-done group as a separate "started" item; V6 accepts any
   remaining member (the existing group-equivalent grading). Captive / cooler screws: see 6.6.
2. **Anchor = the specific part** (latches of *this* RAM module): groups are per anchor instance, and the marks tell
   them apart in questions. Answers in words are grounded to the class + kind (no marks to read back); where a plan
   holds two groups of one class, the grade compares the number of group actions per class. A part without an anchor
   (no `fastens` / `socket_host` / `cable` / `of`) forms a group with the anchor-less parts of the same refined class
   on that desktop (e.g. the latches of empty slots). The anchors and group ids go into the oracle sidecar.
3. **V8 and V14 on classes with several instances:** no ordinals there either. V8 becomes "localise every visible
   <class / group>" (a set of boxes, graded like V1 restricted to that class: set F1 + IoU); V14 is asked by class or
   group only when every member gives the same answer, else not asked. Their pools shrink; the eval targets are
   re-checked.
4. **A group whose members differ** (some latches open, some closed): each member is asked on its own, with its own
   mark (V2, V4). The "no single screw" rule applies to the screws of one group, whose members are never told apart.
5. **V2 count questions** ("how many motherboard screws are still in place?") stay as they are: a class question
   with no mark and no ordinal.
6. **Captive / cooler screws** (owner, 2026-11-03): a V4 question whose answer depends on whether captive / cooler
   screws are loosened is **not asked** (neither the "yes, they are loosened" nor the "no, loosen them first" form);
   in V7 / V16 plans and V10's done set, "loosen the cooler screws" stays one required group action until the part
   they hold comes off.
7. **Audit carry-over:** V2 verdicts carry over by desktop + step + part when the new record has the same answer; V3
   and V4 (now multi-view and reworded) are audited afresh. The new audit set is a fresh seeded draw; nothing is forced
   into it.
8. **Name:** the redrawn subset is **`eval_v2`** (manifest `eval_version` bumped, pack folder name kept); no model
   has been run on `eval_v1`, and every guide is updated to the new name.

## 7. Approaches considered

* **Naming:** (a) mark every named part on the image (groups marked member by member) — chosen by the owner;
  (b) class / group names, marks only where a part must be singled out — more rules, same information; (c) groups
  and class names only, skipping the rest — loses many V2 / V4 items on multi-instance classes; (d) generated
  relational descriptions ("the RAM module nearest the CPU") — unreliable to generate and to grade.
* **Views:** (a) all four views for every action task — chosen; (b) four views for V4 / V6 / V7 / V16 only, V3 / V10 /
  V12 per view; (c) scanner + one side view — half the images, but the side view chosen is arbitrary.
