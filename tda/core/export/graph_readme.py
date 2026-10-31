"""The ``README.md`` beside the per-desktop precedence graphs (task U46).

English for the dataset's readers, with a short 中文摘要 on top, like the
scanner package's README. Generated from ``summary.json`` alone, so it says
exactly what the files beside it hold, and is the same bytes on every run.
"""
from __future__ import annotations

__all__ = ["TOP_CANDIDATE_GROUPS", "readme_text"]

#: How many candidate class pairs the README lists (summary.json has them all).
TOP_CANDIDATE_GROUPS = 25

_SUMMARY_ZH = """\
## 中文摘要

每台机器一个 `D<nn>.json`：拆解的**优先级图**，读者可以直接据此做计划。内容：
`meta`（品牌型号、`graph_version`、`export_graph_version`、像素证据来自哪个视角、计数）；
`instances`（零件、关系字段、是否推断 / 手加、状态时间线，第 0 步是初始状态）；
`steps`（步骤与动作，含失败的尝试）；`edges`（全部生效的硬约束边，及日志在第几步满足了它 `demonstrated_at`）；
`covers`（已确认帧上「谁盖住了谁」的像素证据：上面的零件可见掩码占了下面零件完整形状的多少像素、哪些帧、
被哪些边解释 `explained_by`；没有任何边（正反向都没有）、零件表里也没有关系（装在 / 插在 / 拧住 / 随父件拆下、下面是机箱）的就是**候选**，
供人工判断要不要补 `covered_by` / `blocked_by`）；
`legal_by_step`（每一步之后严格 / 放宽的合法动作集、下一步实际做了什么、是否合法）；
`plans`（从初始状态拆下主板 / CPU / 电源 / 日志拆下的全部零件的最短合法计划，和演示用的动作数对比；
每步之后「全部」还剩几步，即 V10 的进度标签）；`violations`（日志与约束图矛盾的地方，同 `constraints --validate`）；
`deadlocks` / `dead_ends`（应为空）。`summary.json` 是逐台计数、按类别对分组的候选遮挡和全部矛盾。
遮挡是边的**证据**，不是边本身：螺丝盖住主板几个像素是 `fastened_by`，线压在风扇上什么也不挡。
只读导出：只读打开数据库，不写库、不重新编译帧（读已确认帧的存储行），两次运行逐字节相同。
"""


def _fields() -> str:
    return """\
## Files

- `D<nn>.json` -- one desktop's precedence graph (the fields below).
- `summary.json` -- per-desktop counts, every candidate covering grouped by class pair, every
  violation.
- `README.md` -- this file.

Produced by `python -m tda.cli export-graph --desktops <range> --out <dir> [--view scan]` from
the annotation database, opened read-only. No frame is recompiled; nothing is written to the
database. Two runs on the same database give byte-identical files (sorted keys, sorted lists,
no clock, no machine path).

## `D<nn>.json`

### `meta`

`desktop`, `brand`, `model`, `model_family`, `chassis_type`, `chassis_platform` (null where the
sheet did not say), `graph_version` (content hash of the stored constraint graph -- the one the
VLM export quotes), `export_graph_version` (this file format, `1`), `view` (the view whose
confirmed frames the pixel evidence was read from), `pixel_views` (`[view]` when it had any
confirmed frame, else `[]`), and `counts` (instances, steps, actions, edges and edges by type,
confirmed frames, frames with a covering, covers, candidates, violations, deadlocks, dead ends).

### `instances[]`

The machine's own parts: no Label Studio draft, no object outside the machine, no part this
machine does not have. `key`, `class`, `attrs` (as annotated: `kind`, `role`, `captive`, `of`, ...),
`parent` + `attached` (leaves the chassis together with the parent), `mounted_on`, `fastens`
(screws), `socket_host` and `cable` (connectors), `implied` (created by a rule, not named by
the sheet), `extra` (added by hand in the tool), and `timeline`: `[{step, state, placement}]`,
the initial state at step 0 first, then one entry each time the state or placement changes.

### `steps[]`

`step`, `type` (`initial`, `normal`, `compound`, `failed`, `auxiliary`, `reorient`, `ignore`),
`raw_name` (the sheet's text), `actions[]` with `idx`, `verb`, `target`, `tool`, `direction`,
`result` (`success` / `failed`) and `failure_reason`.

### `edges[]`

Every **active** hard-constraint edge (rejected ones and decisions about edges no longer
derived are left out): `label` (`type(target, blocker)`), `type` (`fastened_by`,
`connected_to`, `locked_by`, `covered_by`, `blocked_by`), `target` (the part that waits),
`blocker` (what must change state first), `necessity` (`required` / `recommended`), `mode`
(`blocked_by` only: `physical_path`, `tool_access`, `cable_tension`), `reason`, `source`
(`rule`, `template`, `manual`, `labelstudio`), `status`, `evidence_step`, and
`demonstrated_at`: the first step after which the log has the blocker in a state the edge
accepts -- `null` when it already was at the start, or when the log never gets there.

### `covers[]`

Pixel evidence of who covers whom, from the stored masks of the view's **confirmed** frames
(a frame a person verified, without an open conflict). For each ordered pair `(above, below)`
of parts in the chassis where, on at least one frame, part of `below`'s whole (amodal) shape
lies under `above`'s confirmed visible mask:

    px = | amodal(below) ∩ visible(above) - visible(below) |

`below`'s whole shape is the keyframe the compiler selects on that frame, under that frame's
stored transform (the COCO export's `amodal_segmentation`); visible masks are the confirmed
rows as stored. Fields: `above`, `below`, `above_class`, `below_class` (class, refined by
`kind` or a screw's `role`), `frames` (steps), `n_frames`, `max_px`, `first`, `last`,
`explained_by` (labels of the active edges the covering agrees with: `below` waits for `above`
by any type, or a `covered_by` / `blocked_by` between the two either way round),
`reverse_edges` (active edges the other way: `above` waits for `below` -- the two are already
ordered, the pixels are one outline lying against the other), `structural` (what the instance
table says about the pair: `above.attached_to` / `below.attached_to` -- one leaves the chassis
inside the other; `above.mounted_on`, `above.socket_host`, `above.fastens` and their `below.*`
mirrors; `above.rides_with_parent` -- the upper part leaves with a third one, whose own
covering is listed; `below_fixed` -- the lower part has no verb that changes it, the chassis),
`no_edge` (`explained_by` and `reverse_edges` both empty), `candidate` (`no_edge` and
`structural` empty: a covering nothing accounts for -- the triage list), `order_source` (what
gave `above` those pixels on most of the frames: `zorder` -- the pose segment's layer order,
stored or the compiler's default; `pair_override` -- a pairwise layering exception;
`per_frame_mask` -- one of the two has a mask drawn for that frame alone), `order_sources`
(frames per source), `pose_segments`, `above_removed_at` / `below_removed_at` (the step each
part first reads `removed` in its timeline; `null`: the log never takes it out). Pairs that
never overlap are not listed.

Covering is **evidence** for a precedence edge, not the edge: a screw covering a few pixels of
the board it fastens is a `fastened_by`; a cable lying across a fan gates nothing; the
chassis, which never leaves, is covered by everything. Read candidates with
`below_removed_at` and `max_px` in hand.

### `legal_by_step[]`

One entry per step `k` of the log plus `k = 0` (the initial state), each the state **after**
step `k`: `legal` -- `legal_actions(..., strict=True)` as sorted `"verb target"` strings
(recommended edges honoured), `n_legal`, `legal_relaxed` / `n_legal_relaxed` (required edges
only), `next_step`, `next_demonstrated` (the next step's successful actions),
`next_failed` (its failed attempts), `next_is_legal` / `next_is_legal_relaxed` (every one of
the next step's successful actions was in the legal set -- a compound step's later actions
judged in the state its earlier ones left; `null` when the next step has no successful
action).

### `plans`

From the initial state, the shortest legal plan (`plan_removal`) that ends with the goal
removed, for `motherboard.01`, `cpu.01` and `psu.01` where the desktop has them: `plan`
(`"verb target"` strings, `null` when there is none), `length`, `relaxed` / `dropped`
(recommended edges that had to be set aside), `removed_at` (the step the log removes it) and
`demonstrated_length` (successful actions the log took up to then). `all`: every part the log
removes (`goals`, in the log's order), planned one goal after the other from the state the
previous plans left -- a legal plan, shortest per goal but not proven shortest overall --
with `unplanned` (goals no plan reaches), and `remaining_steps_by_step`: the length of that
plan from the state after each entry of `legal_by_step` (same order; `null` where a goal is
unplannable) -- V10's progress label.

### `violations[]`

The spec 7.4 replay of the log against the graph, the same lines `python -m tda.cli
constraints --validate` prints: `step`, `verb`, `target`, `edge`, `edge_type`, `blocker`,
`blocker_state`, `kind` -- `success_violates` (a recorded action broke a required edge:
almost always a step done and never written down) or `failed_without_unmet` (a failed attempt
nothing in the graph explains: an edge is missing) -- and the original `line`.

### `deadlocks`, `dead_ends`

Loops of actions that wait on each other (spec 7.4 forbids them) and parts no plan can take out
because nothing clears an edge in the way. Both should be empty.

## `summary.json`

`totals`, `per_desktop[]` (edges and edges by type, confirmed frames, covers, `no_edge`, candidates,
violations, deadlocks, dead ends, `next_is_legal` as legal / judged / ratio, plan lengths
against demonstrated lengths), `candidates_by_class_pair[]` (`pair` = `above_class →
below_class`, `n_desktops`, `desktops`, `n_pairs`, `pairs`, `below_removed` = pairs whose
lower part the log does remove, `above_removed_first` = of those, pairs whose upper part came
out first), `violations[]` (with `desktop`), `failed` (desktops that could not be exported).
"""


def _numbers(summary: dict) -> str:
    t = summary["totals"]
    legal = t["next_is_legal"]
    types = ", ".join(f"{k} {v}" for k, v in t["edges_by_type"].items()) or "none"
    lines = [
        "## This export",
        "",
        f"- view: `{summary['view']}`; desktops: {t['desktops']}"
        + (f"; failed: {', '.join(summary['failed'])}" if summary["failed"] else ""),
        f"- active edges: {t['edges']} ({types})",
        f"- confirmed frames read: {t['confirmed_frames']}",
        f"- covering pairs: {t['covers']}; with no edge either way: {t['no_edge']}; "
        f"candidates (nothing accounts for them): {t['candidates']} "
        f"in {t['candidate_class_pairs']} class pairs",
        f"- violations: {t['violations']}"
        + (" (" + ", ".join(f"{k} {v}" for k, v in t["violations_by_kind"].items()) + ")"
           if t["violations"] else ""),
        f"- deadlocks: {t['deadlocks']}; dead ends: {t['dead_ends']}",
        f"- next demonstrated step legal (strict): {legal['legal']} of {legal['judged']}"
        + (f" ({legal['ratio']:.4f})" if legal["ratio"] is not None else ""),
        "",
    ]
    groups = summary["candidates_by_class_pair"][:TOP_CANDIDATE_GROUPS]
    if groups:
        lines += [f"### Candidate coverings by class pair (top {len(groups)} by desktops)", "",
                  "| above → below | desktops | pairs | lower part removed | upper out first |",
                  "|---|---|---|---|---|"]
        lines += [f"| `{g['pair']}` | {g['n_desktops']} | {g['n_pairs']} | {g['below_removed']} "
                  f"| {g['above_removed_first']} |" for g in groups]
        lines.append("")
    if summary["violations"]:
        lines += ["### Violations", ""]
        lines += [f"- D{v['desktop']:02d} {v['line']}" for v in summary["violations"]]
        lines.append("")
    return "\n".join(lines)


def readme_text(summary: dict) -> str:
    """The README of one export, from its ``summary.json``."""
    head = ("# Teardown precedence graphs (`export-graph`)\n\n"
            "One JSON per desktop: the hard-constraint edges, the order the operator took, the "
            "pixel evidence of who covers whom, the legal actions after every step and the "
            "remaining plan.\n\n")
    return head + _SUMMARY_ZH + "\n" + _fields() + "\n" + _numbers(summary)
