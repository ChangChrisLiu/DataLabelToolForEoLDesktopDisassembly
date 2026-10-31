"""Task W3e: the fan screw holes -- the class ``screw_hole`` and ``scripts/migrate_fan_screw_holes.py``.

On the seven heatsink + fan desktops the importer made the fan's screws captive, and the
annotator drew the screw **holes** under the screws' labels on every frame after each screw
came out. The class is a detection category only (no verb, no constraint edge, no VLM
question), layered directly below its screw; the migration relabels the hole pixels to it
byte for byte and makes the screws ordinary removable screws.

The scene (desktop 7, scanner view, 128x128, three steps): the heatsink with its own captive
screw, the fan on it with two captive fan screws -- ``.02`` unscrewed at step 1, ``.01`` at
step 2 -- and the fan removed at step 3. ``.01`` has a "screw drawn in" keyframe at step 1
and its hole keyframe at 2, the hole reaching one pixel row and column outside the head;
``.02`` has only its hole keyframe and a single-frame override (the per-frame hole) at step
2. Every frame is confirmed before the migration runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.export.coco import categories, category_ids, export_coco, load_ctx, other_category
from tda.core.export.graph_export import build_desktop
from tda.core.export.vlm import export_vlm
from tda.core.graph_rules import propose_edges
from tda.core.model import (
    LATE_CLASSES,
    SCREW_HOLE_CLASS,
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.states import needs_geom, state_at
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import events_of
from tda.core.zorder_floor import entered, floor_index
from tda.ui.class_names import class_zh

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_fan_screw_holes.py"

DESKTOP = 7
VIEW = "scan"
#: Boxes below are on a 32-px grid, drawn ``K`` times larger: a screw is 16 px a side,
#: so the VLM's pointable rows are not ``too_small`` (the compiler's ladder, 6 / 12 px).
K = 4
HW = (32 * K, 32 * K)
CHASSIS = "chassis"
HEATSINK = "cpu_cooler.heatsink.01"
FAN = "cpu_cooler.fan.01"
S1, S2 = "screw.cpu_cooler.01", "screw.cpu_cooler.02"
HS_SCREW = "screw.cpu_cooler.03"
H1, H2 = "screw_hole.01", "screw_hole.02"

HEAD1 = (8, 8, 12, 12)       # screw .01 drawn in, step 1
HOLE1 = (9, 9, 13, 13)       # its hole (one row / column outside the head)
HOLE2 = (16, 16, 19, 19)     # screw .02's hole
HOLE2_FRAME2 = (16, 16, 18, 19)   # ... and the per-frame override at step 2


def _rect(box) -> np.ndarray:
    m = np.zeros(HW, dtype=bool)
    x0, y0, x1, y1 = (K * v for v in box)
    m[y0:y1, x0:x1] = True
    return m


def _kf(instance: str, box, anchor: int) -> ShapeKeyframe:
    return ShapeKeyframe(id=None, instance=instance, desktop=DESKTOP, view=VIEW,
                         pose_segment=1, anchor_step=anchor, placement="in_chassis",
                         geom_type="mask", parts=[ShapePart("main", masks.encode_rle(_rect(box)))])


def _screw(key: str, fastens: str) -> InstanceRec:
    return InstanceRec(key=key, desktop=DESKTOP, cls="screw",
                       attrs={"role": "cpu_cooler", "captive": True,
                              "captive_source": "heuristic"},
                       parent=fastens, attached=True, fastens=fastens)


def build(db: Db, tmp_path: Path) -> TruthService:
    """The scene, every frame confirmed."""
    tax = load_taxonomy()
    db.upsert_instance(InstanceRec(key=CHASSIS, desktop=DESKTOP, cls="chassis"))
    db.upsert_instance(InstanceRec(key=HEATSINK, desktop=DESKTOP, cls="cpu_cooler",
                                   attrs={"kind": "heatsink"}))
    db.upsert_instance(InstanceRec(key=FAN, desktop=DESKTOP, cls="cpu_cooler",
                                   attrs={"kind": "fan"}))
    for key, host in ((S1, FAN), (S2, FAN), (HS_SCREW, HEATSINK)):
        db.upsert_instance(_screw(key, host))
    db.replace_steps(DESKTOP, [
        StepRec(DESKTOP, 1, "normal", "CPU Fan Screw 2"),
        StepRec(DESKTOP, 2, "normal", "CPU Fan Screw 1"),
        StepRec(DESKTOP, 3, "normal", "CPU fan"),
    ], [ActionRec(DESKTOP, 1, 0, S2, "unscrew", tool="PH2"),
        ActionRec(DESKTOP, 2, 0, S1, "unscrew", tool="PH2"),
        ActionRec(DESKTOP, 3, 0, FAN, "remove", tool="hand")])
    for step in (1, 2, 3):
        image = tmp_path / f"s{step:03d}.png"
        image.write_bytes(b"png")
        db.upsert_frame(FrameKey(DESKTOP, step, VIEW), str(image), {"hw": list(HW)}, None)
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, 3, 3, None, None)
    for kf in (_kf(CHASSIS, (0, 0, 32, 32), 3), _kf(HEATSINK, (2, 2, 30, 30), 3),
               _kf(HS_SCREW, (28, 28, 30, 30), 3), _kf(FAN, (4, 4, 28, 28), 2),
               _kf(S1, HEAD1, 1), _kf(S1, HOLE1, 2), _kf(S2, HOLE2, 2)):
        db.add_keyframe(kf)
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [
        (CHASSIS, "main"), (HEATSINK, "main"), (HS_SCREW, "main"), (FAN, "main"),
        (S1, "main"), (S2, "main")]))
    db.set_frame_override(FrameOverride(FrameKey(DESKTOP, 2, VIEW), S2,
                                        masks.encode_rle(_rect(HOLE2_FRAME2)), None, None))
    truth = TruthService(db, tax)
    truth.refresh_range(DESKTOP, VIEW, [1, 2, 3])
    for step in (1, 2, 3):
        truth.verify_frame(FrameKey(DESKTOP, step, VIEW), "tester")
    return truth


def _script():
    name = "migrate_fan_screw_holes_w3e"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module        # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene(tmp_path):
    db = Db(str(tmp_path / "tda.sqlite"))
    truth = build(db, tmp_path)
    yield db, truth
    db.close()


@pytest.fixture
def migrated(scene):
    db, truth = scene
    before = {k: dict(v) for k, v in
              ((s, db.compiled(FrameKey(DESKTOP, s, VIEW))) for s in (1, 2, 3))}
    lines: list[str] = []
    result = _script().migrate_desktop(db, truth.tax, truth, DESKTOP, lines.append)
    return db, truth, result, before, lines


def _same(one: dict, two: dict) -> bool:
    """Two compiled rows alike but for ``input_hash`` (it moves with any input)."""
    return ({k: v for k, v in one.items() if k != "input_hash"}
            == {k: v for k, v in two.items() if k != "input_hash"})


def _visible(row) -> np.ndarray:
    rle = row.get("visible_rle")
    return np.zeros(HW, dtype=bool) if rle is None else masks.decode_rle(rle).astype(bool)


# --------------------------------------------------------------------------- #
# the class
# --------------------------------------------------------------------------- #
def test_the_taxonomy_has_the_class_and_no_verb_reaches_it():
    tax = load_taxonomy()
    defn = tax.classes[SCREW_HOLE_CLASS]
    assert defn["group"] == "structure" and list(defn["attrs"]) == ["of"]
    assert tax.default_state(SCREW_HOLE_CLASS) == "present"
    # `removed` is the cascade's word for "out with the fan", never a verb's
    assert tax.states_of(SCREW_HOLE_CLASS) == ["present", "removed"]
    assert tax.needs_mask(SCREW_HOLE_CLASS, "present", "in_chassis")
    assert not tax.needs_mask(SCREW_HOLE_CLASS, "removed", "in_chassis")
    for verb, spec in tax.verbs.items():
        assert SCREW_HOLE_CLASS not in spec["applies_to"], verb
        assert tax.apply_verb(SCREW_HOLE_CLASS, {}, verb) is None, verb
    # appended last: no category id moves
    assert list(tax.classes)[-1] == SCREW_HOLE_CLASS and len(tax.classes) == 25
    assert class_zh(SCREW_HOLE_CLASS) == "螺丝孔"


def test_coco_numbers_it_after_other_and_lists_it_in_sequence_files():
    tax = load_taxonomy()
    ids = category_ids(tax)
    assert [ids[c] for c in list(tax.classes)[:23]] == list(range(1, 24))
    assert other_category(tax)["id"] == 24
    assert ids["loose_cable"] == 25 and ids[SCREW_HOLE_CLASS] == 26
    assert LATE_CLASSES == {"loose_cable", SCREW_HOLE_CLASS}
    seq = {c["name"]: c for c in categories(tax)}
    assert len(seq) == 24 and "loose_cable" not in seq
    assert seq[SCREW_HOLE_CLASS] == {"id": 26, "name": SCREW_HOLE_CLASS,
                                     "supercategory": "structure"}
    gallery = [c["name"] for c in categories(tax, gallery=True)]
    assert gallery[-2:] == ["loose_cable", SCREW_HOLE_CLASS]


def test_s1_offers_the_class_and_the_gallery_does_not():
    from types import SimpleNamespace

    from tda.ui.panels.add_parts import class_choices, gallery_class_choices

    data = SimpleNamespace(tax=load_taxonomy())
    assert ("螺丝孔  screw_hole", SCREW_HOLE_CLASS) in class_choices(data)
    assert SCREW_HOLE_CLASS not in [cls for _label, cls in gallery_class_choices(data)]


def test_a_hole_enters_the_layer_order_directly_below_its_screw():
    cls_of = {"chassis": "chassis", "motherboard.01": "motherboard", "fan": "cpu_cooler",
              "s1": "screw", "s2": "screw", "h": SCREW_HOLE_CLASS}
    order = [("chassis", "main"), ("motherboard.01", "main"), ("fan", "main"),
             ("s1", "main"), ("s2", "main")]
    assert floor_index(order, SCREW_HOLE_CLASS, cls_of, under="s2") == 4
    # its screw not in the order yet: below every other part, above the floors
    assert floor_index(order, SCREW_HOLE_CLASS, cls_of, under="nope") == 2
    assert floor_index([], SCREW_HOLE_CLASS, cls_of) == 0
    # every other class as before
    assert floor_index(order, "screw", cls_of) == 5
    hole = InstanceRec(key="h", desktop=1, cls=SCREW_HOLE_CLASS, attrs={"of": "s1"})
    table = {k: InstanceRec(key=k, desktop=1, cls=c) for k, c in cls_of.items() if k != "h"}
    table["h"] = hole
    assert entered(order, "h", "main", table) == [
        ("chassis", "main"), ("motherboard.01", "main"), ("fan", "main"), ("h", "main"),
        ("s1", "main"), ("s2", "main")]
    assert entered(entered(order, "h", "main", table), "h", "main", table).count(("h", "main")) == 1


def test_no_constraint_edge_ever_names_a_hole(scene, migrated):
    db, truth, result, _before, _lines = migrated
    tax = truth.tax
    instances = db.instances(DESKTOP)
    assert {H1, H2} <= set(instances)
    edges = propose_edges(instances, tax)
    assert edges, "the scene has its fastened_by / covered_by edges"
    assert not [e for e in edges if SCREW_HOLE_CLASS in (e.target + e.blocker)]


# --------------------------------------------------------------------------- #
# the migration
# --------------------------------------------------------------------------- #
def test_the_migration_relabels_the_hole_pixels_and_takes_the_screws_out(migrated):
    db, truth, result, before, lines = migrated
    assert result["ok"], "\n".join(lines)
    assert result["holes"] == [H1, H2] and result["flipped"] == 2
    assert result["keyframes"] == {"hole": 2, "under_head": 1}
    assert result["overrides_moved"] == 1
    # the screws' rows after their unscrew: .02 at 1 and 2, .01 at 2
    assert result["closed"] == {"screw_gone": 3}
    assert result["resigned"] == 2 and result["resign_refused"] == []
    assert result["pixel_mismatch"] == 0 and result["pixel_checked"] == 3
    assert result["other_rows_changed"] == 0 and result["unexpected_rows"] == 0
    assert result["open_conflicts"] == 0 and result["pending_rechecks"] == 0

    instances = db.instances(DESKTOP)
    for hole, screw in ((H1, S1), (H2, S2)):
        rec = instances[hole]
        assert rec.cls == SCREW_HOLE_CLASS and rec.attrs["of"] == screw
        assert rec.parent == FAN and rec.attached
        assert rec.attrs["added_by"] == "controller" and rec.attrs["reason"]
        assert instances[screw].attrs["captive"] is False
    assert instances[HS_SCREW].attrs["captive"] is True      # the heatsink's stay

    rows = {s: db.compiled(FrameKey(DESKTOP, s, VIEW)) for s in (1, 2, 3)}
    # the user's pixels, relabelled byte for byte
    assert rows[1][H2]["visible_rle"] == before[1][S2]["visible_rle"]
    assert rows[2][H2]["visible_rle"] == before[2][S2]["visible_rle"]
    assert np.array_equal(_visible(rows[2][H2]), _rect(HOLE2_FRAME2))   # the moved override
    assert rows[2][H1]["visible_rle"] == before[2][S1]["visible_rle"]
    assert np.array_equal(_visible(rows[2][H1]), _rect(HOLE1))
    # before its unscrew the hole lies under the head: present, 0 px
    assert H1 in rows[1] and not _visible(rows[1][H1]).any()
    assert rows[1][H1]["visibility"] == "occluded_full"
    # the screws are out; nothing of the fan's changed; the holes left with the fan
    assert S2 not in rows[1] and S2 not in rows[2] and S1 not in rows[2]
    assert _same(rows[1][S1], before[1][S1])
    for step in (1, 2):
        assert _same(rows[step][FAN], before[step][FAN])
        assert db.get_frame(FrameKey(DESKTOP, step, VIEW))["review_status"] == "verified"
        assert all(r["status"] == "verified" for r in rows[step].values())
    assert set(rows[3]) == set(before[3]) and H1 not in rows[3]
    assert all(_same(rows[3][k], before[3][k]) for k in rows[3])
    # the per-frame override moved with the hole
    fos = db.frame_overrides(FrameKey(DESKTOP, 2, VIEW))
    assert H2 in fos and S2 not in fos
    # the events: each screw out at its unscrew; the holes out with the fan
    events = {(e.target, e.step, e.attr): (e.old, e.new) for e in db.events(DESKTOP)}
    assert events[(S1, 2, "state")] == ("fastened", "removed")
    assert events[(S1, 2, "placement")] == ("in_chassis", "on_bench")
    assert events[(S2, 1, "state")] == ("fastened", "removed")
    assert events[(H1, 3, "state")] == ("present", "removed")
    # the layer order: each hole directly below its screw
    order = [e[0] for e in db.zorder(DESKTOP, VIEW, 1).order]
    assert order[order.index(S1) - 1] == H1 and order[order.index(S2) - 1] == H2
    kinds = [row["kind"] for row in db.conn.execute(
        "SELECT kind FROM op_log WHERE desktop=? ORDER BY id", (DESKTOP,))]
    assert kinds.count("captive_fix") == 2 and kinds[-1] == "note"
    assert "add_extra_instances" in kinds


def test_a_second_run_finds_nothing_to_do(migrated):
    db, truth, _result, _before, _lines = migrated
    lines: list[str] = []
    again = _script().migrate_desktop(db, truth.tax, truth, DESKTOP, lines.append)
    assert again == {"skipped": "no captive fan screw", "ok": True}


def test_the_hole_is_layered_under_its_screw_in_the_compiler(migrated):
    db, truth, *_ = migrated
    one = truth.compile(FrameKey(DESKTOP, 1, VIEW)).instances
    # before the unscrew: the hole clipped to the head (3 x 3 cells), all of it under the head
    assert not one[H1].visible.any() and one[H1].amodal.sum() == 9 * K * K
    assert one[S1].visible.sum() == 16 * K * K
    two = truth.compile(FrameKey(DESKTOP, 2, VIEW)).instances
    # after it: the whole hole, above the fan, as the screw was
    assert np.array_equal(two[H1].visible, _rect(HOLE1)) and S1 not in two
    assert not (two[FAN].visible & _rect(HOLE1)).any()


def test_a_dry_run_leaves_the_database_alone(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "tda.sqlite")
    db = Db(path)
    build(db, tmp_path)
    db.close()
    script = _script()
    monkeypatch.setattr(script.rawroot, "configure", lambda paths: None)
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"cache_dir: '{(tmp_path / 'cache').as_posix()}'\n", encoding="utf-8")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    code = script.main(["--db", path, "--out", str(tmp_path / "out"), "--paths", str(paths),
                        "--desktops", str(DESKTOP)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "DRY RUN" in out and "all checks hold" in out
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    [summary] = (tmp_path / "out").glob("fan_holes_summary_dryrun_*.json")
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["desktops"][str(DESKTOP)]["pixel_mismatch"] == 0
    # the live database while the app runs: nothing touched
    monkeypatch.setattr(script, "LIVE", path)
    monkeypatch.setattr(script, "app_running", lambda: True)
    assert script.main(["--db", path, "--apply", "--out", str(tmp_path / "out"),
                        "--paths", str(paths)]) == 2
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


# --------------------------------------------------------------------------- #
# the exports
# --------------------------------------------------------------------------- #
def test_coco_writes_the_hole_with_its_screw(migrated, tmp_path):
    db, truth, *_ = migrated
    doc = export_coco(db, truth.tax, [DESKTOP], VIEW, str(tmp_path / "coco.json"), truth=truth)
    cats = {c["name"]: c["id"] for c in doc["categories"]}
    assert cats[SCREW_HOLE_CLASS] == 26
    holes = [a for a in doc["annotations"] if a["category_id"] == 26]
    assert {a["attributes"]["instance_key"] for a in holes} == {H1, H2}
    assert all(a["attributes"]["of"] == {H1: S1, H2: S2}[a["attributes"]["instance_key"]]
               for a in holes)
    assert not [a for a in doc["annotations"] if a["category_id"] != 26
                and "of" in a["attributes"]]


def test_vlm_never_asks_about_a_hole(migrated, tmp_path):
    db, truth, *_ = migrated
    ctx = load_ctx(db, truth.tax, DESKTOP, VIEW, holes=False)
    assert set(ctx.holes) == {H1, H2} and not set(ctx.holes) & set(ctx.instances)
    assert set(load_ctx(db, truth.tax, DESKTOP, VIEW).holes) == set()
    out = tmp_path / "vlm.jsonl"
    export_vlm(db, truth.tax, [DESKTOP], VIEW, str(out), truth=truth)
    text = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("vlm*")
                     if p.is_file())
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(records) > 5, "the scene yields records"
    assert {r["prompt"]["task"] for r in records} >= {"V1", "V8"}
    assert SCREW_HOLE_CLASS not in text and "螺丝孔" not in text


def test_without_the_exclusion_the_vlm_would_name_one(migrated, tmp_path, monkeypatch):
    """The control: the scene's holes are askable rows, so the exclusion is what keeps them out."""
    import tda.core.export.vlm as V

    db, truth, *_ = migrated
    real = V.load_ctx
    monkeypatch.setattr(V, "load_ctx", lambda db, tax, desktop, view, holes=True:
                        real(db, tax, desktop, view))
    out = tmp_path / "vlm_control.jsonl"
    V.export_vlm(db, truth.tax, [DESKTOP], VIEW, str(out), truth=truth)
    assert SCREW_HOLE_CLASS in out.read_text(encoding="utf-8")


def test_the_graph_export_lists_the_hole_without_edges(migrated):
    db, truth, *_ = migrated
    doc = build_desktop(db, truth.tax, DESKTOP)
    listed = {i["key"]: i for i in doc["instances"]}
    assert listed[H1]["class"] == SCREW_HOLE_CLASS and listed[H1]["attrs"]["of"] == S1
    assert listed[H1]["parent"] == FAN and listed[H1]["attached"] is True
    assert [p["state"] for p in listed[H1]["timeline"]] == ["present", "removed"]
    text = json.dumps(doc["edges"]) + json.dumps(doc.get("covers") or [])
    assert SCREW_HOLE_CLASS not in text


def test_the_snapshot_needs_the_hole_until_the_fan_leaves(migrated):
    db, truth, *_ = migrated
    tax = truth.tax
    instances = db.instances(DESKTOP)
    events = events_of(db, tax, DESKTOP)
    assert needs_geom(instances, state_at(instances, events, 2, tax), tax)[H1] == "mask"
    assert H1 not in needs_geom(instances, state_at(instances, events, 3, tax), tax)


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #
def test_the_spec_and_the_handoff_say_so():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "docs" / "superpowers" / "specs"
            / "2026-09-17-teardown-annotator-design.md").read_text(encoding="utf-8")
    classes = spec.split("### 6.1", 1)[1].split("### 6.3", 1)[0]
    assert "| 结构（追加） | `screw_hole`（v1.50，W3e） |" in classes
    assert classes.count("只用于风扇螺丝孔；别处的螺丝孔不做") >= 2
    assert "| `screw_hole`（v1.50，只用于风扇螺丝孔） |" in classes
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines()
               if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:2] if e.startswith("- **v1.50（2026-10-31）**")]
    for needed in ("`screw_hole`", "COCO id 26", "`scripts/migrate_fan_screw_holes.py`",
                   "D07 60、D08 59、D11 60、D14 60、D16 35、D18 55、D19 60", "共 102",
                   "`w3e:hole_under_head`", "`entered`"):
        assert needed in entry, needed
    handoff = (root / "docs" / "HANDOFF_2026-10-28.md").read_text(encoding="utf-8")
    section = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    assert sum("migrate_fan_screw_holes.py" in line for line in section.splitlines()) == 1
