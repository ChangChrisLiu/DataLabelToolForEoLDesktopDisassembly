"""The VLM evaluation pack (W4-A): ``scripts/export_vlmpack.py`` on the W3b scene.

The scene is one desktop with the scanner confirmed and OAK1 frames only, so
the pack is built over both views with a tiny fake frame cache; no model, no
network, no real database.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_vlmpack as P  # noqa: E402
import vlm_io as IO  # noqa: E402

VIEWS = (W.VIEW, W.OTHER)
#: The pack's export units for those views: the camera views, then the multi unit -- less
#: OAK1, which has nothing confirmed and no V15 pair of its own, so its unit is empty and
#: left out (its pictures ship through the multi unit).
UNITS = (W.VIEW, "multi")
EMPTY_UNITS = (W.OTHER,)
REPO = SCRIPTS.parent
#: Stand-ins for the redesign's two new tools while the tree under test predates them
#: (they come from another work stream); a pack built from the real tree ships the real ones.
STAND_INS = {
    "vlm_glossary.py": '"""Stand-in (test tree only)."""\nCLASS_TEXT = {}\nSTATE_MEANINGS = {}\n\n\n'
                       'def glossary_text(terms, state_terms):\n    return ""\n',
    "vlm_audit_carry.py": '"""Stand-in (test tree only)."""\n\n\ndef main(argv=None):\n    return 0\n',
}


def pack_repo(root: Path) -> Path:
    """The tree ``build_pack`` vendors its tools from: the repository itself once it has
    the redesign's glossary and audit carry-over, else a copy of the parts the build reads
    with stand-ins for those two (the build refuses a tree without them)."""
    if all((SCRIPTS / name).is_file() for name in P.REDESIGN_TOOLS):
        return REPO
    tree = Path(root) / "repo"
    if tree.is_dir():
        return tree
    shutil.copytree(SCRIPTS, tree / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in (*P.VENDORED, *P.VENDORED_CONFIGS, "docs/vlm_run_hprc.md"):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tree / rel)
    for name, text in STAND_INS.items():
        if not (tree / "scripts" / name).is_file():
            (tree / "scripts" / name).write_text(text, encoding="utf-8", newline="\n")
    return tree


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene(root: Path) -> tuple[Path, Path]:
    """The scene database and a fake frame cache with one PNG per frame."""
    from PIL import Image

    db_path = root / "scene.sqlite"
    db = Db(str(db_path))
    W.build(db)
    db.close()
    cache = root / "cache"
    for view in VIEWS:
        for step in W.STEPS:
            path = cache / view / f"D{W.DESKTOP:02d}" / f"s{step:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", W.HW, (step * 20 % 255, 90 if view == W.VIEW else 30, 120)).save(path)
    return db_path, cache


def _build(root: Path, db_path: Path, cache: Path, name: str, **kw) -> dict:
    kw.setdefault("repo", pack_repo(root))
    return P.build_pack(str(db_path), str(root / name), cache_dir=str(cache),
                        work_dir=str(root / f"{name}_work"), views=VIEWS, seed=0,
                        audit=20, live_db=str(root / "live.sqlite"), log=lambda *_: None,
                        **kw)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("w4a_pack")
    db_path, cache = _scene(root)
    before = _sha(db_path)
    summary = _build(root, db_path, cache, "pack")
    return {"root": root, "db": db_path, "db_sha_before": before, "cache": cache,
            "pack": root / "pack", "summary": summary,
            "manifest": json.loads((root / "pack" / "manifest.json").read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------- #
# the export units (redesign of 2026-11-03, lane D1)
# --------------------------------------------------------------------------- #
def _unit(tmp: Path, unit: str, tasks=None, **kw) -> tuple[dict, list[dict]]:
    from tda.core.export.vlm import export_vlm

    db = Db(str(tmp / f"{unit}.sqlite"))
    tax = W.build(db, **kw)
    out = tmp / f"vlm_{unit}.jsonl"
    summary = export_vlm(db, tax, [W.DESKTOP], unit, str(out), tasks=tasks)
    db.close()
    return summary, list(IO.read_jsonl(out))


def test_multi_unit_has_only_action_tasks(tmp_path):
    from tda.core.export.vlm import MULTI_VIEW, export_vlm
    from tda.core.export.vlm_tasks import ACTION_TASKS

    summary, records = _unit(tmp_path, MULTI_VIEW)
    tasks = {r["prompt"]["task"] for r in records}
    assert tasks and tasks <= set(ACTION_TASKS)
    assert {"V3", "V4", "V6", "V12"} <= tasks
    assert summary["tasks"] == list(ACTION_TASKS)
    assert summary["view"] == MULTI_VIEW and summary["primary_view"] == W.VIEW
    assert summary["side_views"] == [W.OTHER]
    assert all(r["label"]["view"] == MULTI_VIEW for r in records)
    assert all(f"-{MULTI_VIEW}-" in r["label"]["readable_id"] for r in records)
    # the truth is the scanner's (gold tier, confirmed)
    assert all(r["label"]["tier"] == "gold" for r in records)
    # every image a record shows is in the manifest and has a state table
    out = tmp_path / f"vlm_{MULTI_VIEW}.jsonl"
    manifest = IO.load_manifest(out)
    _, frames = IO.load_oracle(out)
    used = {img for r in records for img in r["prompt"]["images"]}
    assert used and set(manifest) == used and used <= set(frames)
    assert {e["view"] for e in manifest.values()} <= {W.VIEW, W.OTHER}
    # a non-action task is refused, not silently dropped
    db = Db(str(tmp_path / "refuse.sqlite"))
    tax = W.build(db)
    with pytest.raises(ValueError, match="action tasks"):
        export_vlm(db, tax, [W.DESKTOP], MULTI_VIEW, str(tmp_path / "x.jsonl"), tasks=["V1"])
    db.close()


def test_view_units_have_only_perception_tasks(tmp_path):
    """A camera unit (the pack's: tasks = unit_tasks(view)) holds single-view perception
    and V15 only."""
    from tda.core.export.vlm import unit_tasks
    from tda.core.export.vlm_tasks import PERCEPTION_TASKS_SINGLE

    found: dict[str, set[str]] = {}
    for view in VIEWS:
        assert unit_tasks(view) == PERCEPTION_TASKS_SINGLE
        summary, records = _unit(tmp_path, view, tasks=unit_tasks(view))
        found[view] = {r["prompt"]["task"] for r in records}
        assert found[view] <= set(PERCEPTION_TASKS_SINGLE), (view, found[view])
        assert summary["tasks"] == list(PERCEPTION_TASKS_SINGLE)
        assert summary["primary_view"] == view and summary["side_views"] == []
        assert all(r["label"]["view"] == view for r in records)
    assert {"V1", "V2", "V15"} <= found[W.VIEW]


def test_multi_unit_shows_each_moment_from_every_camera_that_has_it(tmp_path, monkeypatch):
    """The side views land in ``TaskCtx.side_frames``: ``moment(step)`` is the scanner
    frame first, then oak1 / oak2 / rs where that camera has the step (one photo missing
    -> a shorter moment)."""
    from tda.core.export import vlm as V

    db = Db(str(tmp_path / "four.sqlite"))
    tax = W.build(db, views=("scan", "oak1", "oak2", "rs"))
    db.conn.execute("UPDATE frame SET missing=1 WHERE desktop=? AND view='oak2' AND step=3",
                    (W.DESKTOP,))
    db.conn.commit()
    seen: dict[int, list[tuple[str, int]]] = {}

    def probe(tc, step):
        assert tc.view == V.MULTI_VIEW
        seen[step] = [(f.image_id, f.step) for f in tc.moment(step)]
        return []

    monkeypatch.setitem(V.GENERATORS, "V4", probe)
    summary = V.export_vlm(db, tax, [W.DESKTOP], V.MULTI_VIEW, str(tmp_path / "m.jsonl"),
                           tasks=["V4"])
    db.close()
    assert summary["side_views"] == ["oak1", "oak2", "rs"]
    from tda.core.export.vlm_tasks import opaque_image_id

    def ids(step, views):
        return [(opaque_image_id(W.DESKTOP, v, step), step) for v in views]

    assert seen[2] == ids(2, ("scan", "oak1", "oak2", "rs"))
    assert seen[3] == ids(3, ("scan", "oak1", "rs"))


# --------------------------------------------------------------------------- #
# the layout and the manifest
# --------------------------------------------------------------------------- #
def test_the_pack_has_the_promised_layout(built):
    pack = built["pack"]
    for rel in ("README.md", "LICENSE_NOTE.txt", "manifest.json", "eval_v2/ids.txt",
                "eval_v2/table.md", "eval_v2/table.json", "eval_v2/audit/audit.sheet.md",
                "tools/vlm_score.py", "tools/vlm_to_chat.py", "tools/vlm_references.py",
                "tools/vlm_selfcheck.py", "tools/vlm_eval_set.py", "tools/requirements.txt",
                "tools/tda/core/export/vlm_check.py", "tools/tda/core/export/vlm_tasks.py",
                "tools/configs/taxonomy.yaml"):
        assert (pack / rel).is_file(), rel
    for view in UNITS:
        for suffix in ("jsonl", "images_manifest.jsonl", "oracle.jsonl", "summary.json"):
            assert (pack / "data" / f"vlm_{view}.{suffix}").is_file()
            assert (pack / "eval_v2" / f"{view}.{suffix}").is_file()
    assert not (pack / "eval_v1").exists()
    assert (pack / "images" / W.VIEW / f"D{W.DESKTOP:02d}" / "s001.png").is_file()
    assert not list(pack.rglob("__pycache__"))
    assert not (pack / P.BUILDING).exists()


def test_the_manifest_lists_every_file_with_its_sha256(built):
    pack, files = built["pack"], built["manifest"]["files"]
    on_disk = {p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file()}
    assert set(files) == on_disk - {"manifest.json"}
    for rel, entry in files.items():
        path = pack / rel
        assert entry["bytes"] == path.stat().st_size, rel
        assert entry["sha256"] == _sha(path), rel
    image = next(k for k in files if k.startswith("images/"))
    assert files[image]["view"] in VIEWS and files[image]["desktop"] == W.DESKTOP
    m = built["manifest"]
    assert m["package"] == "vlmpack" and m["eval_version"] == "eval_v2"
    assert m["views"] == list(VIEWS) and m["desktops"] == [W.DESKTOP]
    assert m["units"] == list(UNITS)
    assert m["selfcheck"]["ok"] is True
    assert set(m["exports"]) == set(UNITS)
    assert m["exports"]["multi"]["by_task"]["V4"] > 0
    assert "V4" not in m["exports"][W.VIEW]["by_task"]


def test_every_manifest_image_is_in_the_pack_and_nothing_else(built):
    pack, files = built["pack"], built["manifest"]["files"]
    named = set()
    for view in UNITS:
        for entry in IO.load_manifest(pack / "data" / f"vlm_{view}.jsonl").values():
            named.add("images/" + entry["path"])
    shipped = {k for k in files if k.startswith("images/")}
    assert shipped == named
    # byte for byte the cached frame
    rel = sorted(shipped)[0]
    assert _sha(pack / rel) == _sha(built["cache"] / rel[len("images/"):])


def test_the_input_database_is_not_written(built):
    assert _sha(built["db"]) == built["db_sha_before"]
    # the build opened a copy of its own, under the work directory
    assert (built["root"] / "pack_work" / "vlmpack_work.sqlite").is_file()


def test_two_builds_are_byte_identical(built):
    root = built["root"]
    _build(root, built["db"], built["cache"], "pack2")
    a = {p.relative_to(built["pack"]).as_posix(): _sha(p)
         for p in built["pack"].rglob("*") if p.is_file()}
    b = {p.relative_to(root / "pack2").as_posix(): _sha(p)
         for p in (root / "pack2").rglob("*") if p.is_file()}
    assert a == b


def test_no_machine_path_or_clock_in_the_shipped_text(built):
    pack = built["pack"]
    root = str(built["root"])
    for rel in ("README.md", "manifest.json", "eval_v2/table.md", "eval_v2/table.json"):
        text = (pack / rel).read_text(encoding="utf-8")
        assert root not in text and root.replace("\\", "/") not in text, rel
        assert "D:/DataSet" not in text and "D:\\DataSet" not in text, rel


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_the_live_database_is_refused_by_name(tmp_path):
    with pytest.raises(P.PackError, match="live database"):
        P.refuse_live(P.LIVE_DB)
    with pytest.raises(P.PackError, match="live database"):
        P.refuse_live(P.LIVE_DB.replace("/", "\\").upper())
    live = tmp_path / "live.sqlite"
    live.write_bytes(b"")
    with pytest.raises(P.PackError, match="live database"):
        P.build_pack(str(live), str(tmp_path / "out"), live_db=str(live))
    with pytest.raises(P.PackError, match="no such database"):
        P.refuse_live(str(tmp_path / "nowhere.sqlite"))


def test_a_missing_image_stops_the_build_before_anything_is_copied(tmp_path):
    db_path, cache = _scene(tmp_path)
    gone = cache / W.VIEW / f"D{W.DESKTOP:02d}" / "s005.png"
    gone.unlink()
    with pytest.raises(P.PackError) as err:
        _build(tmp_path, db_path, cache, "pack")
    assert "1 of" in str(err.value) and f"{W.VIEW}/D{W.DESKTOP:02d}/s005.png" in str(err.value)
    assert not (tmp_path / "pack" / "images").exists()
    # the half-built directory is recognised and replaced with --force
    assert (tmp_path / "pack" / P.BUILDING).is_file()
    with pytest.raises(P.PackError, match="--force"):
        _build(tmp_path, db_path, cache, "pack")
    summary = _build(tmp_path, db_path, cache, "pack", force=True, images="none")
    assert summary["image_mode"] == "none" and summary["images"] == 0
    assert "carries no images" in (tmp_path / "pack" / "README.md").read_text(encoding="utf-8")


def test_a_directory_that_is_not_a_pack_is_never_deleted(tmp_path):
    out = tmp_path / "mine"
    out.mkdir()
    (out / "thesis.tex").write_text("precious", encoding="utf-8")
    db_path, cache = _scene(tmp_path)
    with pytest.raises(P.PackError, match="refusing to delete"):
        _build(tmp_path, db_path, cache, "mine", force=True)
    assert (out / "thesis.tex").read_text(encoding="utf-8") == "precious"


# --------------------------------------------------------------------------- #
# the subset inside the pack
# --------------------------------------------------------------------------- #
def test_the_subset_files_agree_with_ids_and_carry_their_sidecars(built):
    pack = built["pack"]
    ids = [l for l in (pack / "eval_v2" / "ids.txt").read_text(encoding="utf-8").splitlines() if l]
    assert len(ids) == len(set(ids)) == built["summary"]["eval"]["records"]
    seen: list[str] = []
    for view in UNITS:
        truth = pack / "eval_v2" / f"{view}.jsonl"
        records = list(IO.read_jsonl(truth))
        full = {r["id"]: r for r in IO.read_jsonl(pack / "data" / f"vlm_{view}.jsonl")}
        assert all(r == full[r["id"]] for r in records)
        seen += [r["id"] for r in records]
        used = {img for r in records for img in r["prompt"]["images"]}
        manifest = IO.load_manifest(truth)
        assert set(manifest) == used
        worlds, frames = IO.load_oracle(truth)
        assert set(frames) == used and set(worlds) == {W.DESKTOP}
        assert (IO.load_summary(truth)
                == IO.load_summary(pack / "data" / f"vlm_{view}.jsonl"))
        # V6 pairs travel whole
        pairs = {}
        for r in records:
            if r["prompt"]["task"] == "V6":
                pairs.setdefault(r["label"]["v6_pair"], set()).add(r["label"]["v6_form"])
        assert all(forms == {"open", "mc"} for forms in pairs.values())
    assert sorted(seen) == sorted(ids)


def test_the_audit_sample_is_drawn_from_inside_the_subset(built):
    pack = built["pack"]
    ids = set((pack / "eval_v2" / "ids.txt").read_text(encoding="utf-8").split())
    audit = [l for l in (pack / "eval_v2" / "audit" / "audit.ids.txt")
             .read_text(encoding="utf-8").splitlines() if l]
    assert audit and set(audit) <= ids
    sheet = (pack / "eval_v2" / "audit" / "audit.sheet.md").read_text(encoding="utf-8")
    assert sheet.count("\n|") >= len(audit) + 1 and "ok?" in sheet


def test_images_eval_ships_only_what_the_subset_names(tmp_path):
    db_path, cache = _scene(tmp_path)
    summary = _build(tmp_path, db_path, cache, "pack", images="eval")
    pack = tmp_path / "pack"
    named = set()
    for view in UNITS:
        named |= {"images/" + e["path"]
                  for e in IO.load_manifest(pack / "eval_v2" / f"{view}.jsonl").values()}
    files = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert {k for k in files if k.startswith("images/")} == named
    assert summary["images"] == len(named)
    assert "holds only the frames `eval_v2` names" in (pack / "README.md").read_text(
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# the tools
# --------------------------------------------------------------------------- #
def test_the_vendored_tools_stand_alone(built):
    pack = built["pack"]
    stub = (pack / "tools" / "tda" / "core" / "export" / "vlm_tasks.py").read_text(encoding="utf-8")
    assert "def class_label" in stub and "SCREW_PRESENCE" in stub and "INITIALISMS" in stub
    init = (pack / "tools" / "tda" / "core" / "export" / "__init__.py").read_text(encoding="utf-8")
    assert "import" not in init  # no exporter, no database layer
    source = pack_repo(built["root"]) / "scripts"
    for name in P.TOOLS:
        assert (pack / "tools" / name).read_bytes() == (source / name).read_bytes()
    assert {"vlm_glossary.py", "vlm_audit_carry.py"} <= set(P.TOOLS)
    # W4-B's driver, Slurm templates and requirements travel with the pack byte for byte, the run guide beside the card
    driver = SCRIPTS / "vlm_run"
    shipped = sorted(p for p in driver.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert any(p.name == "vlm_run.py" for p in shipped)
    for src in shipped:
        assert (pack / "tools" / "vlm_run" / src.relative_to(driver)).read_bytes() == src.read_bytes(), src.name
    assert (pack / "docs" / "vlm_run_hprc.md").read_bytes() == (SCRIPTS.parent / "docs" / "vlm_run_hprc.md").read_bytes()
    # the build already imported them with -I and ran the self-check from the pack
    P.smoke_import(pack / "tools")
    check = built["summary"]["selfcheck"]
    assert check["ok"] and {f["file"] for f in check["files"]} == {f"{v}.jsonl" for v in UNITS}
    assert all(abs(m - 1.0) < 1e-9 for f in check["files"] for m in f["tasks"].values())


def test_the_self_check_catches_a_wrong_oracle(built, tmp_path):
    import vlm_selfcheck

    pack = built["pack"]
    truth = pack / "eval_v2" / f"{W.VIEW}.jsonl"
    assert vlm_selfcheck.main(["--truth", str(truth), "--out", str(tmp_path / "ok")]) == 0
    # a truth file whose oracle has lost the desktop's world (parts, edges, log)
    # cannot ground a model's wording: the check says so, even though the gold
    # replies (raw keys) still score
    broken = tmp_path / "broken" / f"{W.VIEW}.jsonl"
    broken.parent.mkdir()
    shutil.copyfile(truth, broken)
    shutil.copyfile(IO.sidecar(truth, "summary.json"), IO.sidecar(broken, "summary.json"))
    shutil.copyfile(IO.sidecar(truth, "images_manifest.jsonl"),
                    IO.sidecar(broken, "images_manifest.jsonl"))
    rows = [r for r in IO.read_jsonl(IO.sidecar(truth, "oracle.jsonl"))
            if r.get("type") != "desktop"]
    IO.write_jsonl(IO.sidecar(broken, "oracle.jsonl"), rows)
    assert vlm_selfcheck.main(["--truth", str(broken), "--out", str(tmp_path / "bad")]) == 1
    report = json.loads((tmp_path / "bad" / "selfcheck.json").read_text(encoding="utf-8"))
    assert not report["ok"] and report["files"][0]["missing_worlds"] == [W.DESKTOP]
    good = json.loads((tmp_path / "ok" / "selfcheck.json").read_text(encoding="utf-8"))
    assert good["files"][0]["missing_worlds"] == [] and good["files"][0]["missing_frames"] == []


# --------------------------------------------------------------------------- #
# the data card
# --------------------------------------------------------------------------- #
def test_the_readme_is_a_data_card(built):
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    for needle in ("## Tasks", "## Records", "## Answer formats", "## Images, crops, ROI",
                   "## The evaluation subset", "## How to run", "## Known limitations",
                   "| V4 |", "| V6 |", "truth_source", "graph_blocked_pixels",
                   "prior_is_correct", "v6_pair", "view_confirmed", "--roi view",
                   "Never crop around a record's evidence", "vlm_selfcheck.py",
                   "vlm_score.py --truth", "Unconfirmed views", "V15 is the `moment` variant",
                   "Shortcut bounds are in-sample", "Captive screws"):
        assert needle in text, needle
    from tda.core.export.vlm_check import RESPONSE_FORMATS

    for key, fmt in RESPONSE_FORMATS.items():
        if not key.startswith("V5"):
            assert fmt.split("{allowed}")[0] in text, key
    # the scene has no optical drive cage: the card does not pretend it does
    assert "optical drive cage" not in text.split("## Known limitations")[1].split(
        "## Provenance")[0]


def test_the_card_names_the_optical_bracket_when_a_desktop_has_one(built):
    summary = dict(built["summary"])
    summary["composition"] = dict(summary["composition"], optical_bracket_desktops=[3, 8])
    text = P.readme_text(summary)
    issues = text.split("## Known limitations")[1]
    assert "optical drive cage" in issues and "D03, D08" in issues
    assert "\"Optical drive cage N\"" in issues and "**not** accept" in issues


# --------------------------------------------------------------------------- #
# the redesign's layout (lane D2): five units, eval_v2, the new tools
# --------------------------------------------------------------------------- #
def test_pack_has_multi_and_eval_v2(built):
    import re

    from tda.core.export.vlm_tasks import ACTION_TASKS, PERCEPTION_TASKS_SINGLE

    pack, m = built["pack"], built["manifest"]
    assert m["eval_version"] == "eval_v2" and m["units"] == list(UNITS)
    assert m["empty_units"] == list(EMPTY_UNITS)
    assert not (pack / "data" / f"vlm_{W.OTHER}.jsonl").exists()
    assert m["unit_tasks"]["multi"] == list(ACTION_TASKS)
    assert m["unit_tasks"][W.VIEW] == list(PERCEPTION_TASKS_SINGLE)
    # the action tasks live in multi only, perception in the camera units only
    for unit in UNITS:
        tasks = {r["prompt"]["task"] for r in IO.read_jsonl(pack / "data" / f"vlm_{unit}.jsonl")}
        allowed = ACTION_TASKS if unit == "multi" else PERCEPTION_TASKS_SINGLE
        assert tasks and tasks <= set(allowed), (unit, tasks)
    assert m["exports"]["multi"]["primary_view"] == W.VIEW
    assert m["exports"]["multi"]["side_views"] == [W.OTHER]
    # eval_v2 draws every action task from multi, every perception task from a camera unit
    for row in m["eval"]["rows"]:
        units = set(row["by_view"])
        if row["task"] in ACTION_TASKS:
            assert units == {"multi"}, row
        else:
            assert "multi" not in units, row
    assert {r["task"] for r in m["eval"]["rows"]} >= {"V3", "V4", "V6", "V1", "V2"}
    multi = list(IO.read_jsonl(pack / "eval_v2" / "multi.jsonl"))
    assert multi and all(r["label"]["view"] == "multi" for r in multi)
    # every picture a multi record shows ships with the pack, whichever camera took it
    for entry in IO.load_manifest(pack / "data" / "vlm_multi.jsonl").values():
        assert "images/" + entry["path"] in m["files"]
    # the redesign's tools ship and are listed
    for name in ("vlm_glossary.py", "vlm_audit_carry.py"):
        assert (pack / "tools" / name).is_file() and name in m["tools"]
    text = (pack / "README.md").read_text(encoding="utf-8")
    for needle in ("`eval_v2`", "eval_v2/multi.jsonl", "vlm_multi.jsonl", "### Multi-view records",
                   "### Marks, groups, glossary", "outlined in red", "prompt.marks",
                   "image_legend", "label.image_steps", "groups", "vlm_glossary.py",
                   "vlm_audit_carry.py", "refuse an `eval_v1` path", "--max-images 8"):
        assert needle in text, needle
    assert "eval_v1/" not in text
    venues = r"\b(CVPR|ICCV|ECCV|NeurIPS|ICLR|ICML|ICRA|IROS|CoRL|RSS|AAAI|WACV|BMVC)\b"
    for rel in ("README.md", "eval_v2/table.md", "tools/vlm_eval_set.py",
                "tools/vlm_selfcheck.py", "tools/vlm_references.py", "tools/vlm_run/run_eval.sh"):
        assert not re.search(venues, (pack / rel).read_text(encoding="utf-8")), rel


def test_selfcheck_multi_gold_one(tmp_path):
    """The gold reply of every multi record scores 1.0 with the sidecars the export
    writes (whatever images the generator gives each record)."""
    import vlm_selfcheck

    summary, records = _unit(tmp_path, "multi", views=("scan", "oak1", "oak2", "rs"))
    truth = tmp_path / "vlm_multi.jsonl"
    assert records and summary["side_views"] == ["oak1", "oak2", "rs"]
    result = vlm_selfcheck.check(truth, tmp_path / "check")
    assert result["ok"], result
    assert set(result["rows"]) >= {"V3", "V4", "V12"}
    assert all(abs(row["strict_mean"] - 1.0) < 1e-9 for row in result["rows"].values())
    assert vlm_selfcheck.main(["--truth", str(truth), "--out", str(tmp_path / "cli")]) == 0


def test_a_tree_without_the_redesign_tools_is_refused_before_the_export(tmp_path):
    tree = tmp_path / "old_tree"
    (tree / "scripts").mkdir(parents=True)
    db_path, cache = _scene(tmp_path)
    with pytest.raises(P.PackError, match="vlm_glossary.py, vlm_audit_carry.py"):
        _build(tmp_path, db_path, cache, "pack", repo=tree)
    assert not (tmp_path / "pack").exists()  # refused before anything was written


def test_the_card_runs_the_references_on_multi_and_states_the_review_notes(built):
    """Final review M1 / M4 / the deferred README line, and M2 (the shipped run guide)."""
    text = (built["pack"] / "README.md").read_text(encoding="utf-8")
    run = text.split("## How to run")[1].split("## Known limitations")[0]
    # V6 / V7 / V16 live in multi only: the protocol prior is run there
    assert "vlm_references.py --export eval_v2/multi.jsonl" in run
    assert "--pred runs/refs/multi/protocol_prior.predictions.jsonl" in run
    assert "vlm_references.py --export eval_v2/scan.jsonl" not in run
    # a mark crossing the view ROI is clipped at the crop edge
    assert "is clipped there" in text.split("## Images, crops, ROI")[1]
    # the truth principle: a V4 group "yes" is demonstrated for one member only
    assert "demonstrated for the one member the operator took next" in text
    assert "graph-permitted for the others" in text
    # eval_v2's own V4 bounds are on the card and in the manifest
    assert "eval_shortcuts" in built["manifest"]
    guide = (built["pack"] / "docs" / "vlm_run_hprc.md").read_text(encoding="utf-8")
    assert "eval_v2/audit/audit.ids.txt" in guide and "pack_units.sh" in guide
    assert "eval_v1/" not in guide and "四个视角" not in guide
    for venue in ("CVPR", "ECCV", "ICCV", "NeurIPS"):
        assert venue not in guide and venue not in text
