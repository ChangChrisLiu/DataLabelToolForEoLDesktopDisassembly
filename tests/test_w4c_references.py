"""W4-C: the GT-mask overlay, chance / frequency rows, the state oracle without V14, the pack.

Two copies of the :mod:`tests.w3b_scene` desktop in one database (D41 with the
scanner confirmed, D42 with the scanner *and* OAK1 confirmed), exported for
both views, a flat-colour frame cache and the scene's COCO (``export_coco``) as
the scanner pack. No model, no network, no real database.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.coco import export_coco
from tda.core.export.vlm import export_vlm
from tda.core.export.vlm_check import gold_reply, grade
from tda.core.taxonomy import load_taxonomy

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for p in (SCRIPTS, SCRIPTS / "vlm_run"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import export_vlmpack as P  # noqa: E402
from test_w4a_vlmpack import pack_repo  # noqa: E402
import vlm_io as IO  # noqa: E402
import vlm_overlay as O  # noqa: E402
import vlm_references  # noqa: E402
import vlm_score  # noqa: E402

D_SCAN, D_BOTH = W.DESKTOP, W.DESKTOP + 1
VIEWS = (W.VIEW, W.OTHER)
#: The export units of those views (the multi unit holds the action tasks).
UNITS = (*VIEWS, "multi")
FLAT = {W.VIEW: (40, 90, 160), W.OTHER: (120, 60, 30)}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene_db(path: Path) -> tuple[Db, object]:
    """D41 (scanner confirmed, split train) and D42 (both views confirmed, split test)."""
    db = Db(str(path))
    original = W.DESKTOP
    try:
        W.DESKTOP = D_SCAN
        tax = W.build(db, confirm=(W.VIEW,))
        W.DESKTOP = D_BOTH
        W.build(db, confirm=(W.VIEW, W.OTHER))
    finally:
        W.DESKTOP = original
    for desktop, split in ((D_SCAN, "train"), (D_BOTH, "test")):
        meta = dict(db.get_desktop(desktop))
        meta.pop("id", None)
        meta["split"] = split
        db.upsert_desktop(desktop, meta)
    return db, tax


def _fake_cache(root: Path, exports: list[Path]) -> None:
    for export in exports:
        for entry in IO.load_manifest(export).values():
            path = root / entry["path"]
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (W.HW[1], W.HW[0]), FLAT[entry["view"]]).save(path)


@pytest.fixture(scope="module")
def scene(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("w4c")
    db, tax = _scene_db(root / "scene.sqlite")
    exports = {}
    for view in UNITS:
        out = root / "export" / f"vlm_{view}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        export_vlm(db, tax, [D_SCAN, D_BOTH], view, str(out))
        exports[view] = out
    # the "scanner pack": one COCO per view under the root (D<nn>/annotations.json layout
    # is what the real pack has; the painter globs **/annotations.json)
    coco = root / "coco"
    for view in VIEWS:
        export_coco(db, tax, [D_SCAN, D_BOTH], view, str(coco / view / "annotations.json"))
    db.close()
    cache = root / "cache"
    _fake_cache(cache, list(exports.values()))
    return {"root": root, "db": root / "scene.sqlite", "exports": exports, "coco": coco,
            "cache": cache}


@pytest.fixture(scope="module")
def refs(scene) -> dict:
    """``vlm_references.py`` on the scanner export with the overlay variant."""
    out = scene["root"] / "refs"
    code = vlm_references.main(["--export", str(scene["exports"][W.VIEW]), "--out", str(out),
                                "--assets", str(out / "overlays"),
                                "--image-root", str(scene["cache"]),
                                "--coco-root", str(scene["coco"])])
    assert code == 0
    return {"out": out, "assets": out / "overlays",
            "summary": json.loads((out / "references.summary.json").read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------- #
# the masks sidecar and the RLE decoder
# --------------------------------------------------------------------------- #
def test_the_masks_sidecar_holds_the_oracle_rows_and_decodes_like_pycocotools(scene, tmp_path):
    from pycocotools import mask as M

    export = scene["exports"][W.VIEW]
    sidecar = tmp_path / "scan.masks.jsonl"
    assert O.main(["masks", "--export", str(export), "--coco-root", str(scene["coco"]),
                   "--out", str(sidecar)]) == 0
    manifest, frames = O.load_manifest(export), O.load_frames(export)
    masks = O.load_masks(sidecar)
    confirmed = {i for i, f in frames.items() if f["confirmed"] and i in manifest}
    assert confirmed and set(masks) == confirmed  # every confirmed frame of both views
    header = next(IO.read_jsonl(sidecar))
    assert header["type"] == "header" and header["classes"][0] == "chassis"
    assert header["source"] == scene["coco"].name
    for image_id, row in masks.items():
        keys = {m["instance"] for m in row["masks"]}
        oracle_keys = {k for k, e in frames[image_id]["state"].items() if "visibility" in e}
        assert keys <= oracle_keys and keys  # the rows the export read, nothing else
        assert row["size"] == [W.HW[0], W.HW[1]]
        for m in row["masks"]:
            assert m["visibility"] == frames[image_id]["state"][m["instance"]]["visibility"]
            ours = np.array(O.mask_image(row["size"], m["counts"])) > 0
            ref = M.decode({"size": row["size"], "counts": m["counts"].encode()}).astype(bool)
            assert (ours == ref).all()
    # a frame of D42 in OAK1 is confirmed too and has masks (V15 pairs of D42 qualify)
    assert any(manifest[i]["view"] == W.OTHER for i in masks)


def test_a_scanner_pack_from_another_snapshot_is_refused(scene, tmp_path):
    coco = tmp_path / "coco"
    shutil.copytree(scene["coco"], coco)
    path = coco / W.VIEW / "annotations.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for ann in doc["annotations"]:
        if ann["attributes"]["instance_key"] == W.CHASSIS:  # never hidden in the scene
            ann["attributes"]["visibility"] = "occluded_full"
    path.write_text(json.dumps(doc), encoding="utf-8")
    export = scene["exports"][W.VIEW]
    with pytest.raises(O.OverlayError, match="not from this snapshot"):
        O.masks_from_coco(coco, O.load_manifest(export), O.load_frames(export))


# --------------------------------------------------------------------------- #
# the overlay variant
# --------------------------------------------------------------------------- #
def _eligible(records, frames, masks) -> list[dict]:
    out = []
    for r in records:
        if r["prompt"]["task"] == "V8":
            continue
        images = r["prompt"]["images"]
        if all(frames.get(i, {}).get("confirmed") and i in masks for i in images):
            out.append(r)
    return out


def test_the_overlay_variant_is_the_eligible_records_with_painted_copies(scene, refs):
    export = scene["exports"][W.VIEW]
    records = list(IO.read_jsonl(export))
    frames = O.load_frames(export)
    masks = {r["image"]: r for r in O.masks_from_coco(scene["coco"], O.load_manifest(export),
                                                      frames)[0]}
    want = _eligible(records, frames, masks)
    variant = list(IO.read_jsonl(refs["out"] / "mask_overlay.jsonl"))
    assert [r["id"] for r in variant] == [r["id"] for r in want]
    assert len(variant) < len(records)
    assert all(r["prompt"]["task"] != "V8" for r in variant)
    assert all(r["label"]["reference"] == "mask_overlay" for r in variant)
    by_id = {r["id"]: r for r in records}
    for r in variant:
        src = by_id[r["id"]]
        assert r["prompt"]["images"] == src["prompt"]["images"]  # the ids stay
        assert r["prompt"]["question"] == src["prompt"]["question"]
        assert r["label"]["answer_check"] == src["label"]["answer_check"]
    # V8 and the unconfirmed V15 pairs of D41 are the ones left out; D42's V15 pairs are in
    summary = refs["summary"]["mask_overlay"]
    assert summary["records"] == len(variant) and summary["records_in"] == len(records)
    assert summary["skipped"]["V8"] == sum(1 for r in records if r["prompt"]["task"] == "V8")
    assert summary["skipped"]["unconfirmed"] > 0 and "no_masks" not in summary["skipped"]
    v15 = [r for r in variant if r["prompt"]["task"] == "V15"]
    assert v15 and {r["label"]["desktop"] for r in v15} == {D_BOTH}
    # the overlay manifest points every image of the variant at a painted copy
    overlay_manifest = IO.load_manifest(refs["out"] / "mask_overlay.jsonl")
    used = {i for r in variant for i in r["prompt"]["images"]}
    assert set(overlay_manifest) == used
    source_manifest = IO.load_manifest(export)
    for image_id, entry in overlay_manifest.items():
        src = source_manifest[image_id]
        assert entry["path"] == Path(src["path"]).with_suffix(".png").as_posix()
        assert entry["path"].startswith(f"{src['view']}/D{src['desktop']:02d}/s{src['step']:03d}")
        assert (refs["assets"] / entry["path"]).is_file()
    # both pictures of a D42 V15 pair are painted, one per view
    pair = v15[0]["prompt"]["images"]
    views = {overlay_manifest[i]["path"].split("/")[0] for i in pair}
    assert views == set(VIEWS)
    # the oracle and summary sidecars travel with the variant, so it is a truth file of its own
    assert (refs["out"] / "mask_overlay.oracle.jsonl").is_file()
    assert (refs["out"] / "mask_overlay.summary.json").is_file()


def _blend(src, colour, alpha=O.ALPHA):
    return tuple(int(round((s * (255 - alpha) + c * alpha) / 255)) for s, c in zip(src, colour))


def test_nothing_but_the_masks_is_painted_and_the_palette_is_fixed(scene, refs):
    classes = O.taxonomy_classes()
    assert len(classes) == 25 and classes[0] == "chassis" and classes[14] == "screw"
    assert O.class_colour("screw", classes) == O.PALETTE[14]
    export = scene["exports"][W.VIEW]
    manifest, frames = O.load_manifest(export), O.load_frames(export)
    masks = {r["image"]: r for r in O.masks_from_coco(scene["coco"], manifest, frames)[0]}
    overlay_manifest = IO.load_manifest(refs["out"] / "mask_overlay.jsonl")
    screw_pixels: dict[tuple, set] = {}
    checked = 0
    for image_id, entry in overlay_manifest.items():
        src_entry = manifest[image_id]
        src = np.array(Image.open(scene["cache"] / src_entry["path"]).convert("RGB"))
        out = np.array(Image.open(refs["assets"] / entry["path"]).convert("RGB"))
        assert out.shape == src.shape
        row = masks[image_id]
        union = np.zeros(src.shape[:2], bool)
        for m in row["masks"]:
            union |= np.array(O.mask_image(row["size"], m["counts"])) > 0
        changed = (src != out).any(axis=2)
        # 1. no pixel outside a mask changed: no text, no box, no label
        assert not (changed & ~union).any()
        # 2. every mask pixel is painted (a flat source differs from every blend)
        assert (changed | ~union).all()
        # 3. the screws are the smallest masks, painted last: inside one, the pixel is the
        #    blend of the source with the screw colour, its 2 px rim the full colour
        colour = O.class_colour("screw", classes)
        for m in row["masks"]:
            if not m["instance"].startswith("screw."):
                continue
            mask = np.array(O.mask_image(row["size"], m["counts"])) > 0
            ys, xs = np.nonzero(mask)
            if not len(ys):
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            inner = mask.copy()
            inner[:, :] = False
            inner[y0 + O.OUTLINE_PX:y1 + 1 - O.OUTLINE_PX, x0 + O.OUTLINE_PX:x1 + 1 - O.OUTLINE_PX] = True
            inner &= mask
            rim = mask & ~inner
            flat = FLAT[src_entry["view"]]
            if inner.any():
                assert {tuple(p) for p in out[inner]} == {_blend(flat, colour)}
            assert {tuple(p) for p in out[rim]} == {colour}
            screw_pixels.setdefault((src_entry["view"], src_entry["desktop"]), set()).update(
                tuple(p) for p in out[rim])
            checked += 1
    assert checked > 0
    # the same class colour on every desktop and view
    assert len({frozenset(v) for v in screw_pixels.values()}) == 1
    assert len(screw_pixels) >= 2


def test_painting_is_deterministic_and_reused(scene, refs, tmp_path):
    export = scene["exports"][W.VIEW]
    again = tmp_path / "again"
    summary = O.render_export(export, again / "mask_overlay.jsonl", scene["cache"],
                              again / "overlays", coco_root=str(scene["coco"]), log=None)
    assert summary["overlays_written"] == summary["overlays"] and summary["overlays_reused"] == 0
    for path in (again / "overlays").rglob("*.png"):
        rel = path.relative_to(again / "overlays")
        assert _sha(path) == _sha(refs["assets"] / rel), rel
    assert _sha(again / "mask_overlay.jsonl") == _sha(refs["out"] / "mask_overlay.jsonl")
    # a second pass over the same assets paints nothing
    summary = O.render_export(export, again / "mask_overlay.jsonl", scene["cache"],
                              again / "overlays", coco_root=str(scene["coco"]), log=None)
    assert summary["overlays_written"] == 0 and summary["overlays_reused"] == summary["overlays"]


def test_the_overlay_needs_an_image_root_and_masks(scene, tmp_path, capsys):
    export = scene["exports"][W.VIEW]
    assert vlm_references.main(["--export", str(export), "--out", str(tmp_path / "a"),
                                "--assets", str(tmp_path / "ov")]) == 2
    assert "--image-root" in capsys.readouterr().out
    assert O.main(["render", "--export", str(export), "--image-root", str(scene["cache"]),
                   "--assets", str(tmp_path / "ov"), "--out", str(tmp_path / "b.jsonl")]) == 2
    assert "no masks" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the state oracle without V14
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def refs_multi(scene) -> dict:
    """``vlm_references.py`` on the multi export (the action tasks: the protocol prior's)."""
    out = scene["root"] / "refs_multi"
    code = vlm_references.main(["--export", str(scene["exports"]["multi"]), "--out", str(out),
                                "--assets", str(out / "overlays"),
                                "--image-root", str(scene["cache"]),
                                "--coco-root", str(scene["coco"])])
    assert code == 0
    return {"out": out, "assets": out / "overlays",
            "summary": json.loads((out / "references.summary.json").read_text(encoding="utf-8"))}


def _merged(scene, path: Path) -> Path:
    """The scanner unit and the multi unit as one truth file (records, manifest, oracle):
    every task of the scene in one place, as the scorer's rule table expects."""
    records, manifest, desks, frames = [], {}, {}, {}
    for unit in (W.VIEW, "multi"):
        export = scene["exports"][unit]
        records += list(IO.read_jsonl(export))
        manifest.update(IO.load_manifest(export))
        for row in IO.read_jsonl(IO.sidecar(export, "oracle.jsonl")):
            if row.get("type") == "desktop":
                desks.setdefault(row["desktop"], row)
            else:
                frames.setdefault(row["image"], row)
    IO.write_jsonl(path, records)
    IO.write_jsonl(IO.sidecar(path, "images_manifest.jsonl"),
                   [{"type": "header"}] + [manifest[k] for k in sorted(manifest)])
    IO.write_jsonl(IO.sidecar(path, "oracle.jsonl"),
                   [desks[k] for k in sorted(desks)] + [frames[k] for k in sorted(frames)])
    shutil.copyfile(IO.sidecar(scene["exports"][W.VIEW], "summary.json"),
                    IO.sidecar(path, "summary.json"))
    return path


def test_the_state_oracle_has_no_v14_and_says_so(scene, refs, refs_multi):
    records = list(IO.read_jsonl(scene["exports"][W.VIEW]))
    v14 = [r for r in records if r["prompt"]["task"] == "V14"]
    assert v14  # the scanner is confirmed: V14 exists
    oracle = list(IO.read_jsonl(refs["out"] / "state_oracle.jsonl"))
    assert all(r["prompt"]["task"] != "V14" for r in oracle)
    # (records naming a part by its mark tag alone, mark_tag, are left out too)
    skipped = refs["summary"]["state_oracle"]["skipped"]
    assert set(skipped) <= {"V14", "mark_tag"} and skipped["V14"] == len(v14)
    assert len(oracle) == len(records) - sum(skipped.values())
    assert refs["summary"]["state_oracle"]["records"] == len(oracle)
    assert "visibility" in refs["summary"]["state_oracle"]["skipped_why"]
    # the other two variants are complete, as before
    blind = refs["summary"]["blind"]
    n_blind = len(list(IO.read_jsonl(refs["out"] / "blind.jsonl")))
    assert n_blind == blind["records"] == len(records) - sum((blind.get("skipped") or {}).values())
    # the protocol prior answers V6 / V7 / V16, which live in the multi unit
    assert refs_multi["summary"]["protocol_prior"]["records"] > 0
    assert refs_multi["summary"]["state_oracle"]["records"] == len(
        list(IO.read_jsonl(scene["exports"]["multi"])))


# --------------------------------------------------------------------------- #
# chance and frequency
# --------------------------------------------------------------------------- #
def _synthetic_truth(path: Path) -> list[dict]:
    """Six V12 records: train 3 changed / 1 not, test 2 not; four V6-mc records."""
    rows = []
    plan = [(1, "train", True), (1, "train", True), (2, "train", True), (2, "train", False),
            (3, "test", False), (3, "test", False)]
    for i, (desk, split, changed) in enumerate(plan):
        rows.append({"id": f"rec_v12_{i}", "prompt": {"task": "V12", "images": ["a", "b"],
                                                       "question": "Did anything change?"},
                     "label": {"answer": {"changed": changed}, "answer_check": {"type": "exact"},
                               "desktop": desk, "split": split, "template_id": "V12.1"}})
    options = [{"verb": "remove", "target": f"part.0{k}", "target_label": f"Part {k}",
                "kind": "demonstrated" if k == 2 else "state_inapplicable"} for k in range(1, 5)]
    for i, (desk, split) in enumerate([(1, "train"), (1, "train"), (2, "train"), (3, "test")]):
        rows.append({"id": f"rec_v6_{i}", "prompt": {"task": "V6", "images": ["a"],
                                                      "question": "Next?", "options": options},
                     "label": {"answer": {"verb": "remove", "target": "part.02"},
                               "answer_check": {"type": "next_action",
                                                "reference": ["remove", "part.02"]},
                               "options": options, "v6_form": "mc", "v6_pair": f"p{i}",
                               "desktop": desk, "split": split, "template_id": "V6.1"}})
    IO.write_jsonl(path, rows)
    return rows


def test_chance_and_frequency_on_a_synthetic_truth_file(tmp_path):
    truth = tmp_path / "synthetic.jsonl"
    rows = _synthetic_truth(truth)
    preds = tmp_path / "pred.jsonl"
    IO.write_jsonl(preds, [{"id": r["id"], "raw_text": gold_reply(r)} for r in rows])
    assert vlm_score.main(["--truth", str(truth), "--pred", str(preds), "--out", str(tmp_path),
                           "--resamples", "50"]) == 0
    result = json.loads((tmp_path / "pred.score.json").read_text(encoding="utf-8"))
    assert result["chance"]["V12"]["mean"] == 0.5 and result["chance"]["V12"]["n"] == 6
    assert result["chance"]["V6-mc"]["mean"] == 0.25
    assert "yes / no" in result["chance"]["V12"]["rule"]
    freq = result["frequency"]
    # the train majority is "changed": right on 3 of the 6 records
    assert freq["V12"]["answer"] == json.dumps({"changed": True})
    assert freq["V12"]["fit"] == "train" and freq["V12"]["fit_records"] == 4
    assert freq["V12"]["mean"] == 0.5 and freq["V12"]["n"] == 6
    assert freq["V12"]["answer_share_in_fit"] == 0.75
    # the most frequent option position is B on every record (one option set): 1.0
    assert freq["V6-mc"]["answer"] == json.dumps({"choice": "B"})
    assert freq["V6-mc"]["mean"] == 1.0 and freq["V6-mc"]["fit_records"] == 3
    md = (tmp_path / "pred.score.md").read_text(encoding="utf-8")
    assert "| chance | frequency | text-only upper bound (in-sample) |" in md
    assert "| V12 | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.0% / 0.0% | 50.0% | " \
           "0.500 [" in md
    assert "| task | chance rule | frequency answer (fit) |" in md
    assert result["shortcut_upper_bound_note"].startswith("in-sample upper bound")


def test_chance_follows_the_rule_table_on_the_scene(scene, tmp_path):
    export = _merged(scene, tmp_path / "both.jsonl")
    records = list(IO.read_jsonl(export))
    worlds, _ = IO.load_oracle(export)
    tax = load_taxonomy()
    preds = tmp_path / "gold.jsonl"
    IO.write_jsonl(preds, [{"id": r["id"], "raw_text": gold_reply(r, worlds[IO.desktop_of(r)])}
                           for r in records])
    result = vlm_score.score(export, preds, resamples=20)
    chance, freq = result["chance"], result["frequency"]
    assert set(chance) == set(result["strict"]["by_task"]) == set(freq)
    for key, cell in chance.items():
        assert cell["rule"] == vlm_score.CHANCE_RULES[key]
        assert cell["n"] == result["strict"]["by_task"][key]["n"], key
    for key in ("V4", "V12", "V16", "V15"):
        if key in chance:
            assert chance[key]["mean"] == 0.5
    for key in ("V1", "V3", "V7", "V8", "V10", "V6"):
        if key in chance:
            assert chance[key]["mean"] == 0.0
    assert chance["V10"]["extra"] == {"bin_accuracy": 0.25}
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(vlm_score.task_key(r), []).append(r)
    mc = by_task["V6-mc"]
    assert chance["V6-mc"]["mean"] == round(sum(1 / len(r["prompt"]["options"]) for r in mc)
                                            / len(mc), 4)
    # V2 state questions: 1 / |allowed| per record; V14: 0.5, or 0.5 / |states| when answerable
    v2 = by_task["V2"]
    want = [1 / len(r["label"]["answer_check"]["allowed"]) for r in v2
            if "state" in r["label"]["answer"]]
    assert want and chance["V2"]["mean"] == pytest.approx(
        sum(vlm_score.chance_of(r, worlds[IO.desktop_of(r)], tax) for r in v2) / len(v2), abs=1e-4)
    for r in by_task["V14"]:
        p = vlm_score.chance_of(r, worlds[IO.desktop_of(r)], tax)
        if r["label"]["answer"]["answerable"]:
            states = r["label"]["answer_check"].get("allowed") or vlm_score._states_of(
                r, worlds[IO.desktop_of(r)], tax)
            assert p == 0.5 / len(states)
        else:
            assert p == 0.5
    # frequency: fitted on D41 (train), graded on both desktops, by the scorer's own grader
    for key, cell in freq.items():
        rows = [r for r in records if key in vlm_score._rows_of(vlm_score.task_key(r), r["label"])]
        train = [r for r in rows if r["label"]["split"] == "train"]
        assert cell["fit"] == "train" and cell["fit_records"] == len(train)
        scores = [grade(r, cell["answer"], worlds[IO.desktop_of(r)], tax).score for r in rows]
        assert cell["mean"] == pytest.approx(sum(scores) / len(scores), abs=1e-4), key
        assert cell["n"] == len(rows)
        assert cell["mean"] <= result["strict"]["by_task"][key]["mean"] + 1e-9


# --------------------------------------------------------------------------- #
# the pack
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pack(scene) -> dict:
    root = scene["root"]
    out = root / "pack"
    summary = P.build_pack(str(scene["db"]), str(out), cache_dir=str(scene["cache"]),
                           work_dir=str(root / "pack_work"), views=VIEWS, seed=0, audit=10,
                           live_db=str(root / "live.sqlite"), log=lambda *_: None,
                           coco_root=str(scene["coco"]), repo=pack_repo(root))
    return {"pack": out, "summary": summary,
            "manifest": json.loads((out / "manifest.json").read_text(encoding="utf-8"))}


def test_the_pack_ships_the_masks_the_overlays_and_the_painter(pack):
    out, summary = pack["pack"], pack["summary"]
    refs = summary["references"]
    assert refs["masks"]["source"] == "coco"
    for view in UNITS:
        assert (out / "data" / f"vlm_{view}.masks.jsonl").is_file()
        assert (out / "eval_v2" / f"{view}.masks.jsonl").is_file()
        row = refs["masks"]["views"][view]
        assert row["frames_with_masks"] == row["confirmed_frames"] > 0
        assert row["confirmed_without_masks"] == 0
    assert (out / "tools" / "vlm_overlay.py").read_bytes() == (SCRIPTS / "vlm_overlay.py").read_bytes()
    assert "vlm_overlay.py" in P.TOOLS and "vlm_overlay.py" in summary["tools"]
    painted = refs["mask_overlay"]
    assert set(painted) == set(UNITS)
    scan = painted[W.VIEW]
    assert 0 < scan["records"] < scan["records_in"] and scan["skipped"]["V8"] > 0
    variant = out / "eval_v2" / "mask_overlay" / f"{W.VIEW}.jsonl"
    rows = list(IO.read_jsonl(variant))
    assert len(rows) == scan["records"]
    subset_ids = {r["id"] for r in IO.read_jsonl(out / "eval_v2" / f"{W.VIEW}.jsonl")}
    assert {r["id"] for r in rows} <= subset_ids
    overlay_manifest = IO.load_manifest(variant)
    assert overlay_manifest and all((out / "overlays" / e["path"]).is_file()
                                    for e in overlay_manifest.values())
    # the subset's masks sidecar is the data one cut to the subset's images
    used = {i for r in IO.read_jsonl(out / "eval_v2" / f"{W.VIEW}.jsonl")
            for i in r["prompt"]["images"]}
    cut = O.load_masks(out / "eval_v2" / f"{W.VIEW}.masks.jsonl")
    full = O.load_masks(out / "data" / f"vlm_{W.VIEW}.masks.jsonl")
    assert set(cut) == {i for i in full if i in used} and all(cut[i] == full[i] for i in cut)
    # every overlay is in the manifest with its hash; the self-check still passes
    files = pack["manifest"]["files"]
    overlays = {k for k in files if k.startswith("overlays/")}
    # (a multi record's side cameras are not painted: their entries point at the plain frame)
    assert overlays == {"overlays/" + e["path"] for v in UNITS
                        for e in IO.load_manifest(out / "eval_v2" / "mask_overlay" / f"{v}.jsonl").values()
                        if e.get("painted", True)}
    assert summary["selfcheck"]["ok"] is True
    assert all(abs(m - 1.0) < 1e-9 for f in summary["selfcheck"]["files"] for m in f["tasks"].values())
    P.smoke_import(out / "tools")


def test_the_card_documents_the_references(pack):
    text = (pack["pack"] / "README.md").read_text(encoding="utf-8")
    assert "## Reference lines" in text
    for needle in ("**chance**", "**frequency**", "**blind**", "**protocol_prior**",
                   "**state_oracle**", "**mask_overlay**", "no V14 items", "Do **not** report",
                   "text-only upper bound (in-sample)", "pre-painted", "vlm_overlay.py",
                   "VARIANT=overlay", "`overlays/`"):
        assert needle in text, needle
    # without a scanner pack the card says what is missing instead of pretending
    summary = dict(pack["summary"])
    summary["references"] = {"skipped": "no --coco-root"}
    plain = P.readme_text(summary)
    assert "carries no masks sidecar and no overlay variant" in plain
    assert "## Reference lines" in plain


def _bash() -> str | None:
    candidates = []
    git = shutil.which("git")
    if git:
        candidates.append(str(Path(git).resolve().parents[1] / "bin" / "bash.exe"))
        candidates.append(str(Path(git).resolve().parents[1] / "usr" / "bin" / "bash.exe"))
    if shutil.which("bash"):
        candidates.append(shutil.which("bash"))
    for cand in candidates:
        try:
            proc = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip() == "ok":
            return cand
    return None


def test_run_eval_overlay_variant_scores_the_gold_replies(pack, tmp_path):
    bash = _bash()
    if bash is None:
        pytest.skip("no working bash on this machine")
    script = pack["pack"] / "tools" / "vlm_run" / "run_eval.sh"
    assert b"\r\n" not in script.read_bytes()
    assert subprocess.run([bash, "-n", str(script)], capture_output=True).returncode == 0
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update({"PACK": pack["pack"].as_posix(), "VIEW": W.VIEW, "BACKEND": "gold",
                "MODEL_TAG": "gold", "VARIANT": "overlay", "RUNS": runs.as_posix(),
                "TRUTH": (pack["pack"] / "data" / f"vlm_{W.VIEW}.jsonl").as_posix(),
                "OVERLAYS": (tmp_path / "overlays").as_posix(),  # painted here, from the sidecar
                "PYTHON": Path(sys.executable).as_posix(), "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run([bash, str(script)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = runs / "gold" / "overlay" / W.VIEW
    variant = list(IO.read_jsonl(out / "truth" / "mask_overlay.jsonl"))
    assert variant and all(r["label"]["reference"] == "mask_overlay" for r in variant)
    assert (out / "truth" / f"{W.VIEW}.masks.jsonl").is_file()  # the subset copied the sidecar
    chat = list(IO.read_jsonl(out / "chat.jsonl"))
    assert [c["id"] for c in chat] == [r["id"] for r in variant]
    shown = [Path(p) for c in chat for p in c["meta"]["images"]]
    assert shown and all(p.is_file() and (tmp_path / "overlays") in p.parents for p in shown)
    preds = list(IO.read_jsonl(out / "pred.jsonl"))
    assert len(preds) == len(variant) and all(p["raw_text"] for p in preds)
    result = json.loads((out / "score" / "gold.score.json").read_text(encoding="utf-8"))
    assert result["predicted"] == len(variant)
    # answered records all score 1; the records the variant leaves out are the parse failures
    subset = list(IO.read_jsonl(out / "truth" / f"{W.VIEW}.jsonl"))
    assert result["records"] == len(subset) > len(variant)
    assert result["strict"]["by_task"]["V2"]["mean"] == 1.0
    assert result["strict"]["parse_failure_rate"]["V8"] == 1.0
    assert "chance" in result and "frequency" in result


# --------------------------------------------------------------------------- #
# multi-view records (the redesign of 2026-11-03, lane D2)
# --------------------------------------------------------------------------- #
def _multi_scene(tmp: Path):
    """Two moments of one desktop seen by the scanner (confirmed, with a mask) and two
    side cameras (unconfirmed, no mask): a single-moment multi record, a two-moment one
    (V3-like), and a camera-unit record whose second picture is unconfirmed."""
    images, manifest, frames, masks = {}, {}, {}, {}
    for step in (4, 5):
        for view in ("scan", "oak1", "rs"):
            iid = f"img_{view}{step}"
            rel = f"{view}/D07/s{step:03d}.png"
            path = tmp / "images" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), FLAT.get(view, (10, 200, 10))).save(path)
            images[(view, step)] = iid
            manifest[iid] = {"image": iid, "path": rel, "desktop": 7, "view": view, "step": step}
            frames[iid] = {"type": "frame", "image": iid, "desktop": 7, "view": view,
                           "step": step, "confirmed": view == "scan",
                           "state": {"cpu.01": {"state": "installed" if step == 4 else "removed",
                                                "placement": "in_chassis"}}}
            if view == "scan":
                # the whole 8x8 frame is the CPU's visible mask
                masks[iid] = {"image": iid, "size": [8, 8], "masks": [
                    {"instance": "cpu.01", "class": "cpu", "visibility": "visible",
                     "counts": [0, 64]}]}

    def rec(rid, task, view, imgs, **label):
        return {"id": rid, "prompt": {"task": task, "images": imgs, "question": "?"},
                "label": {"desktop": 7, "step": 5, "view": view, "answer": {},
                          "answer_check": {}, **label}}

    one = [images[("scan", 5)], images[("oak1", 5)], images[("rs", 5)]]
    two = [images[("scan", 4)], images[("oak1", 4)], images[("rs", 4)], *one]
    records = [rec("rec_multi_one", "V4", "multi", one, views=["scan", "oak1", "rs"],
                   image_steps=[5, 5, 5]),
               rec("rec_multi_two", "V3", "multi", two, image_steps=[4, 4, 4, 5, 5, 5]),
               rec("rec_v15_pair", "V15", "scan", [images[("scan", 5)], images[("oak1", 5)]],
                   views=["scan", "oak1"])]
    return images, manifest, frames, masks, records


def test_overlay_paints_scan_image_only(tmp_path):
    images, manifest, frames, masks, records = _multi_scene(tmp_path)
    classes = O.taxonomy_classes()
    variant, rows, summary = O.render(records, manifest, frames, masks, tmp_path / "images",
                                      tmp_path / "overlays", classes)
    # the multi records are painted although their side cameras are unconfirmed; the
    # camera-unit V15 pair is not (its second frame is unconfirmed, as before)
    assert [r["id"] for r in variant] == ["rec_multi_one", "rec_multi_two"]
    assert summary["skipped"] == {"unconfirmed": 1}
    assert O.painted_images(records[0], frames) == [images[("scan", 5)]]
    assert O.painted_images(records[1], frames) == [images[("scan", 4)], images[("scan", 5)]]
    by_id = {r["image"]: r for r in rows if r.get("type") != "header"}
    for (view, step), iid in images.items():
        entry = by_id[iid]
        target = (tmp_path / "overlays" / entry["path"]).resolve()
        assert target.is_file(), entry
        if view == "scan":
            # painted: a new PNG under the assets, not the frame
            assert entry.get("painted", True) and (tmp_path / "overlays") in target.parents
            assert target.read_bytes() != (tmp_path / "images" / manifest[iid]["path"]).read_bytes()
        else:
            # a side camera: the plain frame, reached from the assets by a relative path
            assert entry["painted"] is False and entry["path"].startswith("../images/")
            assert target == (tmp_path / "images" / manifest[iid]["path"]).resolve()
    assert summary["overlays"] == 2 and summary["unpainted_side_images"] == 4
    assert not (tmp_path / "overlays" / "oak1").exists()


def test_state_oracle_one_table_per_moment(tmp_path):
    images, manifest, frames, masks, records = _multi_scene(tmp_path)
    one, two, pair = records
    worlds: dict = {}
    # a multi record of one moment: one table (the scanner's), not one per camera
    out = vlm_references.state_oracle(one, worlds, frames)
    assert out["prompt"]["images"] == [] and len(out["prompt"]["state_tables"]) == 1
    assert vlm_references.moment_images(one, frames) == [images[("scan", 5)]]
    assert "removed" in out["prompt"]["state_tables"][0]
    # V3 / V12: two moments, two tables, before then after
    out = vlm_references.state_oracle(two, worlds, frames)
    assert len(out["prompt"]["state_tables"]) == 2
    assert "installed" in out["prompt"]["state_tables"][0]
    assert "removed" in out["prompt"]["state_tables"][1]
    # without label.image_steps the oracle's step of each image groups them the same way
    bare = json.loads(json.dumps(two))
    del bare["label"]["image_steps"]
    assert vlm_references.moment_images(bare, frames) == [images[("scan", 4)], images[("scan", 5)]]
    # a camera unit keeps one table per picture: V15's two pictures of one step stay two
    assert len(vlm_references.state_oracle(pair, worlds, frames)["prompt"]["state_tables"]) == 2


def test_text_only_variants_drop_the_mark_wording(tmp_path):
    """Blind and state-oracle records carry no picture, so "outlined in red" is removed
    (the class phrase it qualified stays) and a question naming a part by its tag alone
    is left out; the run driver's blind rows are the same."""
    import vlm_collect

    marks = [{"image": 0, "bbox": [1, 1, 4, 4], "tag": "A"}]
    named = {"id": "rec_named", "prompt": {
        "task": "V4", "images": ["img_a"], "marks": marks,
        "image_legend": [{"view": "scan", "moment": "now"}],
        "question": "Can the RAM module outlined in red be removed now?"},
        "label": {"desktop": 7, "view": "multi", "answer": {}, "answer_check": {}}}
    options = {"id": "rec_opts", "prompt": {
        "task": "V6", "images": ["img_a"], "marks": marks, "question": "What next?",
        "options": [{"verb": "unscrew", "target_label": "the CPU cooler screws outlined in red"}]},
        "label": {"desktop": 7, "view": "multi", "answer": {}, "answer_check": {}}}
    tagged = {"id": "rec_tag", "prompt": {
        "task": "V4", "images": ["img_a"], "marks": marks,
        "question": "Must part A come off before part B?"},
        "label": {"desktop": 7, "view": "multi", "answer": {}, "answer_check": {}}}
    plain = {"id": "rec_plain", "prompt": {"task": "V2", "images": ["img_a"],
                                             "question": "How many motherboard screws?"},
             "label": {"desktop": 7, "view": "scan", "answer": {}, "answer_check": {}}}
    rows, skipped = vlm_references.blind_rows([named, options, tagged, plain])
    assert [r["id"] for r in rows] == ["rec_named", "rec_opts", "rec_plain"]
    assert skipped == {"mark_tag": 1}
    assert rows[0]["prompt"]["question"] == "Can the RAM module be removed now?"
    assert rows[1]["prompt"]["options"][0]["target_label"] == "the CPU cooler screws"
    assert rows[2]["prompt"]["question"] == plain["prompt"]["question"]
    for r in rows:
        assert r["prompt"]["images"] == [] and "marks" not in r["prompt"]
        assert "image_legend" not in r["prompt"] and r["label"]["reference"] == "blind"
    assert named["prompt"]["marks"] == marks  # the input is not touched
    frames = {"img_a": {"image": "img_a", "step": 3, "view": "scan", "state": {}}}
    oracle, why = vlm_references.state_oracle_rows([named, tagged], {}, frames)
    assert [r["id"] for r in oracle] == ["rec_named"] and why == {"mark_tag": 1}
    assert "outlined" not in oracle[0]["prompt"]["question"]
    # the driver's blind variant writes the same rows
    src = tmp_path / "in.jsonl"
    IO.write_jsonl(src, [named, options, tagged, plain])
    assert vlm_collect.blind(src, tmp_path / "blind.jsonl") == 3
    assert list(IO.read_jsonl(tmp_path / "blind.jsonl")) == rows
