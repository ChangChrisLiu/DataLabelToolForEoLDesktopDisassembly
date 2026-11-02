"""Task W3j: an unplugged connector is out of the way -- ``covered_by`` is satisfied by ``unplugged``.

The user answered the 20 R cards (2026-11-01) and said the questions were wrong: 「第19步的时候
这个connector一定已经不在了，是不是有问题？」, 「这个connector在第12步已经被移走了，为啥你说
还没拿走呢？」.  17 of the 19 「日志里下面那件先动了」 rows were a connector painted over a part
the log moved *after* the connector had been unplugged; ``COVER_CLEAR`` had no ``unplugged``,
although the taxonomy gives an unplugged connector no geometry (``needs_mask``, spec §6.2).

* the table: ``REQUIRED_STATES["covered_by"]`` has ``unplugged``; nothing else changes meaning
  (only connectors have the state; ``BLOCKED_REQUIRED`` and the other types untouched; the
  independent checker's transcription says the same);
* ``log_proof`` on D16's shape (connector unplugged at 3, heatsink removed at 19) is ``None``
  and the pair leaves the bucket; still plugged at 19 it stays, as before;
* ``validate_sequence`` with ``covered_by(heatsink <- connector)`` accepts that log and still
  rejects the one where the heatsink leaves while the connector is ``plugged``;
  ``legal_actions`` agrees;
* ``pixel-edges``' ``would_violate`` check agrees (``plan_desktop``, no CLI, no database);
* the docs: spec §7.2 / §4.4 / §14 v1.55 on top, the handoff.
"""
from __future__ import annotations

from pathlib import Path

import vlm_checker
from tda.core import pixel_edges as PE
from tda.core.graph import Edge, legal_actions, validate_sequence
from tda.core.graph_rules import BLOCKED_REQUIRED, REQUIRED_STATES, required_states
from tda.core.model import ActionRec, InstanceRec
from tda.core.states import events_from_actions, state_at
from tda.core.taxonomy import load_taxonomy
from tda.ui.cover_rulings import BLOCKS, COVER_CLEAR, cover_log, log_proof, triage_pairs

D = 16
TAX = load_taxonomy()
ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
HANDOFF = ROOT / "docs" / "HANDOFF_2026-10-28.md"

CONNECTOR = "connector.02"
HEATSINK = "cpu_cooler.heatsink.01"


def rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=D, cls=cls, attrs=attrs, **kw)


def act(step: int, target: str, verb: str, idx: int = 0, result: str = "success") -> ActionRec:
    return ActionRec(desktop=D, step=step, idx=idx, target=target, verb=verb, result=result)


SCENE = {r.key: r for r in (
    rec(CONNECTOR, "connector"),
    rec(HEATSINK, "cpu_cooler", attrs={"kind": "heatsink"}),
    rec("motherboard.01", "motherboard"),
)}
#: D16 (card R04): the connector is unplugged at step 3, the heatsink removed at step 19.
D16_LOG = [act(3, CONNECTOR, "disconnect"), act(19, HEATSINK, "remove")]
#: The same without the unplugging: the connector is still plugged when the heatsink leaves.
PLUGGED_LOG = [act(19, HEATSINK, "remove")]
EDGE = Edge(type="covered_by", target=HEATSINK, blocker=CONNECTOR)
PAIR = (CONNECTOR, HEATSINK, 3109)
BREACH = (f"step 19: remove {HEATSINK} violates covered_by({HEATSINK}, {CONNECTOR}) "
          f"({CONNECTOR} is 'plugged')")


def state_before(actions, step: int):
    done = [a for a in actions if a.step < step]
    return state_at(SCENE, events_from_actions(SCENE, done, TAX), 10**9, TAX)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_unplugged_frees_a_covered_by_and_nothing_else():
    assert REQUIRED_STATES["covered_by"] == {"open", "removed", "displaced", "unplugged"}
    assert COVER_CLEAR == REQUIRED_STATES["covered_by"]
    for kind in ("fastened_by", "locked_by", "blocked_by"):
        assert "unplugged" not in REQUIRED_STATES[kind], kind
    assert REQUIRED_STATES["connected_to"] == {"unplugged", "removed"}      # as it always was
    # the blocked_by modes are untouched: a drive in the cage still has to be removed
    assert BLOCKED_REQUIRED == {"mounted": {"removed"}, "contained": {"removed"}}
    assert required_states("blocked_by", "contained") == {"removed"}
    assert required_states("covered_by", "contained") == required_states("covered_by")
    # only a connector has the state, so no other class changes meaning
    with_state = sorted(cls for cls in TAX.classes if "unplugged" in TAX.states_of(cls))
    assert with_state == ["connector"]
    assert TAX.apply_verb("connector", {}, "disconnect") == ("state", "unplugged")


def test_the_independent_checker_transcribes_the_same_table():
    assert vlm_checker.REQUIRED_STATES == {k: set(v) for k, v in REQUIRED_STATES.items()}
    assert "unplugged" in vlm_checker.REQUIRED_STATES["covered_by"]


# --------------------------------------------------------------------------- #
# the bucket
# --------------------------------------------------------------------------- #
def test_a_connector_unplugged_before_the_part_moved_leaves_the_bucket():
    log = cover_log(SCENE, D16_LOG, TAX)
    assert log_proof(log, CONNECTOR, HEATSINK) is None
    tri = triage_pairs([PAIR], SCENE, [], {}, log=log)
    assert tri.proven == []
    # a connector on top without a log question is W3c's 本版不做: not asked either
    assert tri.asked == [] and tri.indirect == [] and tri.class_not_blocks == []


def test_a_connector_still_plugged_when_the_part_moved_stays_in_the_bucket():
    log = cover_log(SCENE, PLUGGED_LOG, TAX)
    move = log_proof(log, CONNECTOR, HEATSINK)
    assert (move.step, move.verb, move.target, move.before[CONNECTOR]) == \
        (19, "remove", HEATSINK, "plugged")
    [cand] = triage_pairs([PAIR], SCENE, [], {}, log=log).proven
    assert (cand.above, cand.below, cand.proof) == (CONNECTOR, HEATSINK, move)


# --------------------------------------------------------------------------- #
# the replay and the legal actions
# --------------------------------------------------------------------------- #
def test_validate_sequence_accepts_the_move_after_the_unplugging():
    assert validate_sequence(SCENE, [EDGE], D16_LOG, TAX) == []
    assert validate_sequence(SCENE, [EDGE], PLUGGED_LOG, TAX) == [BREACH]
    # unplugged then (hypothetically) still there at 19: the same answer whatever step it was
    late = [act(18, CONNECTOR, "disconnect"), act(19, HEATSINK, "remove")]
    assert validate_sequence(SCENE, [EDGE], late, TAX) == []
    # a failed attempt while the connector is still plugged is explained by the edge; one
    # after the unplugging is not (nothing in the graph was in the way any more)
    failed = [act(2, HEATSINK, "remove", result="failed"), *D16_LOG]
    assert validate_sequence(SCENE, [EDGE], failed, TAX) == []
    failed = [act(10, HEATSINK, "remove", result="failed"), *D16_LOG]
    assert validate_sequence(SCENE, [EDGE], failed, TAX) == [
        f"step 10: failed remove {HEATSINK} has no unmet constraint - missing edge?"]


def test_legal_actions_agree():
    plugged = state_before(PLUGGED_LOG, 19)
    assert plugged[CONNECTOR].state == "plugged"
    assert ("remove", HEATSINK) not in legal_actions(SCENE, [EDGE], plugged, TAX)
    unplugged = state_before(D16_LOG, 19)
    assert unplugged[CONNECTOR].state == "unplugged"
    assert ("remove", HEATSINK) in legal_actions(SCENE, [EDGE], unplugged, TAX)
    assert ("remove", HEATSINK) in legal_actions(SCENE, [EDGE], unplugged, TAX, strict=False)


# --------------------------------------------------------------------------- #
# pixel-edges
# --------------------------------------------------------------------------- #
COVER = {"above": CONNECTOR, "below": HEATSINK, "max_px": 3109, "below_amodal_px": 40_000,
         "frames": [1, 2], "n_frames": 2, "first": 1}
RULED = {(CONNECTOR, HEATSINK): {"ruling": BLOCKS, "by": "user", "ts": "2026-11-01T10:00:00",
                                 "view": "scan", "step": 2, "note": ""}}


def test_pixel_edges_no_longer_skips_the_pair_as_would_violate():
    plan = PE.plan_desktop(D, [COVER], 10, SCENE, [], D16_LOG, [], TAX, RULED, None)
    assert plan.skipped == []
    assert [(i["above"], i["below"], i["source"]) for i in plan.review] == \
        [(CONNECTOR, HEATSINK, PE.REVIEW_SOURCE)]
    assert PE.edge_of(plan.review[0]) == Edge(
        type="covered_by", target=HEATSINK, blocker=CONNECTOR, necessity="required",
        reason=plan.review[0]["reason"], source=PE.REVIEW_SOURCE, evidence_step=1,
        status="accepted")
    # the connector still plugged when the heatsink left: the edge would make the log a
    # breach, so the row is skipped as before
    plan = PE.plan_desktop(D, [COVER], 10, SCENE, [], PLUGGED_LOG, [], TAX, RULED, None)
    assert plan.review == []
    [item] = plan.skipped
    assert (item["why"], item["lines"]) == ("would_violate", [BREACH])


# --------------------------------------------------------------------------- #
# the docs
# --------------------------------------------------------------------------- #
def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    table = spec.split("### 7.2 硬约束语义", 1)[1].split("### 7.3", 1)[0]
    [row] = [line for line in table.splitlines() if line.startswith("| `covered_by(X, Y)` |")]
    assert "Y ∈ {open, removed, displaced, unplugged}" in row and "W3j" in row
    assert "§6.2" in row
    review = spec.split("### 4.4 审阅模式", 1)[1].split("### 4.5", 1)[0]
    [para] = [line for line in review.splitlines() if line.startswith("  **v1.51（W3f）")]
    for needed in ("v1.55 起 unplugged 也算让开", "v1.55（W3j", "**拔掉的接头算让开**", "`unplugged`",
                   "第19步的时候这个connector一定已经不在了"):
        assert needed in para, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:4] if e.startswith("- **v1.55（2026-11-01）**")]
    for needed in ("（W3j，", "**unplugged**", "第19步的时候这个connector一定已经不在了",
                   "为啥你说还没拿走呢", "needs_mask", "`would_violate`", "`BLOCKED_REQUIRED` 不动",
                   "**22 → 22**", "**29 → 12**", "**0 → 0**", "`tests/vlm_checker.py`",
                   "tests/test_w3j_unplugged_clears.py"):
        assert needed in entry, needed
    assert spec.index("- **v1.55") < spec.index("- **v1.54") < spec.index("- **v1.53")
    handoff = HANDOFF.read_text(encoding="utf-8")
    export = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    [line] = [l for l in export.splitlines() if "W3j" in l]
    for needed in ("`unplugged`", "v1.55", "`tests/vlm_checker.py`", "22 → 22", "29 → 12"):
        assert needed in line, needed
