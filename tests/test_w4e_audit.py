"""W4-E: the fillable audit sheet, the audit summary (Wilson) and the timed answer form
(the human baseline is out of the plan since 2026-11-03; the form stays a tool).

All on the :mod:`tests.w3b_scene` export (scanner confirmed, OAK1 frames only)
with a tiny fake frame cache; the pack test builds ``scripts/export_vlmpack.py``
over it. No model, no network, no real database.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vlm_audit_sheet as S  # noqa: E402
import vlm_audit_summary as A  # noqa: E402
import vlm_eval_set as E  # noqa: E402
import vlm_human_sheet as H  # noqa: E402
import vlm_io as IO  # noqa: E402
import vlm_score  # noqa: E402


def _keys(record: dict) -> dict:
    """The answer with its keys (``answer_check.answer_keys``; ``label.answer`` names parts
    plainly since the owner's 2026-11-04 round)."""
    label = record["label"]
    return (label.get("answer_check") or {}).get("answer_keys") or label["answer"]


def _set_keys(record: dict, answer: dict) -> None:
    label = record["label"]
    if (label.get("answer_check") or {}).get("answer_keys"):
        label["answer_check"]["answer_keys"] = answer
    else:
        label["answer"] = answer


VIEWS = (W.VIEW, W.OTHER)


#: The pack's export units of the scene (the redesign of 2026-11-03): the scanner's
#: perception unit and the multi unit of the action tasks (OAK1's unit is empty: nothing
#: confirmed, no V15 pair of its own; its pictures ship in the multi unit).
UNITS = (W.VIEW, "multi")


@pytest.fixture(scope="module")
def scene(tmp_path_factory) -> dict:
    """The two-view export (every task per view), the pack's units, a fake ``images/`` tree and
    ``eval_v3`` drawn from the units with a 20-item audit."""
    from PIL import Image

    from tda.core.export.vlm import unit_tasks

    tmp = tmp_path_factory.mktemp("w4e")
    db = Db(str(tmp / "s.sqlite"))
    tax = W.build(db)
    data = tmp / "data"
    data.mkdir()
    for view in VIEWS:
        export_vlm(db, tax, [W.DESKTOP], view, str(data / f"vlm_{view}.jsonl"))
    units = tmp / "units"
    units.mkdir()
    for unit in UNITS:
        export_vlm(db, tax, [W.DESKTOP], unit, str(units / f"vlm_{unit}.jsonl"),
                   tasks=unit_tasks(unit))
    db.close()
    images = tmp / "images"
    manifests = [data / f"vlm_{v}.jsonl" for v in VIEWS] + [units / f"vlm_{u}.jsonl" for u in UNITS]
    for export in manifests:
        for entry in IO.load_manifest(export).values():
            path = images / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                Image.new("RGB", W.HW, (entry["step"] * 20 % 255, 80, 120)).save(path)
    eval_dir = tmp / "eval_v3"
    E.build([units / f"vlm_{u}.jsonl" for u in UNITS], eval_dir, seed=0, audit=20)
    exports = [eval_dir / f"{u}.jsonl" for u in UNITS]
    return {"tmp": tmp, "data": data, "images": images, "eval": eval_dir, "exports": exports,
            "audit": list(IO.read_jsonl(eval_dir / "audit" / "audit.jsonl"))}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _articles(text: str) -> list[str]:
    return re.findall(r"<article class=\"item\".*?</article>", text, re.S)


# --------------------------------------------------------------------------- #
# the audit sheet
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def audit_sheet(scene) -> dict:
    out = scene["eval"] / "audit"
    result = S.build(scene["audit"], scene["exports"], scene["images"], out, batch=8)
    return {"out": out, "result": result,
            "pages": [(out / f).read_text(encoding="utf-8") for f in result["files"]
                      if f.endswith(".html")]}


def test_the_box_overlay_covers_the_picture_only(tmp_path):
    """The SVG overlay sits with the <img> in its own box; spanning the caption too, it was stretched vertically and
    every box was drawn too low (owner, 2026-11-03, audit item 53)."""
    manifest = {"img_1": {"path": "scan/D57/s019.png"}}
    html = S.figure_html("img_1", manifest, tmp_path / "images", tmp_path, [
        {"label": "Front panel connector 2", "bbox": [679, 827, 38, 16], "kind": "evidence"}], "image 1 of 1")
    box = re.search(r'<span class="imgbox">(.*?)</span>', html, re.S)
    assert box and "<img " in box.group(1) and '<svg class="boxes"' in box.group(1)
    assert "<figcaption>" not in box.group(1) and html.index("</span>") < html.index("<figcaption>")
    assert ".imgbox { position: relative;" in S.CSS if hasattr(S, "CSS") else True


def _multi_v3(tmp_path) -> tuple[dict, dict]:
    """A hand-made multi-view V3 record (the shared contract): 4 views before, 4 after; part A marked on the scan."""
    views = ("scan", "oak1", "oak2", "rs")
    manifest, images, legend = {}, [], []
    for moment, step in (("before", 4), ("after", 5)):
        for v in views:
            image_id = f"img_{v}_{step}"
            manifest[image_id] = {"path": f"{v}/D07/s{step:03d}.png", "view": v, "desktop": 7,
                                  "step": step}
            images.append(image_id)
            legend.append({"view": v, "moment": moment})
    record = {"id": "rec_multi_v3", "prompt": {
        "task": "V3", "images": images, "image_legend": legend,
        "question": "Which action was carried out between the two moments?",
        "marks": [{"image": 0, "bbox": [120, 80, 40, 30], "tag": "A"},
                  {"image": 0, "bbox": [300, 80, 40, 30], "tag": "A"}],
        "terms": ["cpu_cooler"]},
        "label": {"readable_id": "V3-D07-multi-s005", "template_id": "V3.1", "desktop": 7,
                  "step": 5, "view": "multi", "views": list(views) * 2,
                  "image_steps": [4] * 4 + [5] * 4,
                  "answer": {"verb": "remove", "target_instance": "cpu_cooler.01",
                             "target_class": "cpu_cooler", "tool": "hand"},
                  "answer_check": {"type": "exact", "accept_targets": ["cpu_cooler.01", "cover.01"]},
                  "evidence": {"instances": ["cpu_cooler.01"],
                               "bboxes": {"cpu_cooler.01": [100, 60, 260, 120]}}}}
    return record, manifest


def test_audit_sheet_draws_marks(tmp_path):
    """Marks are red, tagged, on their own image only; a two-moment record shows a before row and an after row."""
    record, manifest = _multi_v3(tmp_path)
    article = S.item_html(1, record, manifest, None, tmp_path / "images", tmp_path / "out", 1)
    rows = re.findall(r'<div class="frames ([a-z ]+)">(.*?)</div><!--/frames-->', article, re.S)
    assert [r[0] for r in rows] == ["multi before", "multi after"]
    assert [len(re.findall(r"<figure ", r[1])) for r in rows] == [4, 4]
    figs = re.findall(r'<figure class="frame" data-index="(\d)">.*?data-boxes="([^"]*)"', article, re.S)
    assert [int(i) for i, _ in figs] == list(range(8))
    boxes = [json.loads(html_unescape(b)) for _, b in figs]
    marks = [[b for b in bs if b["kind"] == "mark"] for bs in boxes]
    assert [len(m) for m in marks] == [2, 0, 0, 0, 0, 0, 0, 0]
    assert {m["label"] for m in marks[0]} == {"A"}
    assert marks[0][0]["bbox"] == [120.0, 80.0, 40.0, 30.0]
    # the evidence is a scanner-frame box: drawn on the two scanner images, not on the side views
    evidence = [[b for b in bs if b["kind"] == "evidence"] for bs in boxes]
    assert [bool(e) for e in evidence] == [True, False, False, False, True, False, False, False]
    # the overlay fix holds for every figure; marks are red on the model's layer, the
    # reviewer's boxes on a layer of their own, hidden until asked (owner 2026-11-04)
    assert article.count('<span class="imgbox"><img ') == 8
    assert article.count('<svg class="boxes"') == article.count('<svg class="review"') == 8
    assert "svg.boxes line.mark { stroke: #ff1744; }" in S.CSS
    assert "body:not(.show-review) svg.review { display: none; }" in S.CSS
    # captions name the camera and the moment; the prompt lines the model reads are shown
    assert "image 1 of 8: top-down scanner, before (step 4" in article
    assert "image 8 of 8: RealSense, after (step 5" in article
    assert "Images 1–4 before, images 5–8 after the action" in article
    assert "Outlined in red: A (2 outlines: one group of interchangeable parts; image 1)." in article
    assert "Glossary:" in article and "CPU cooler" in article
    # V3's gold is the part acted on alone: an attached part an older record still lists
    # in accept_targets is not shown as an accepted answer (owner 2026-11-03)
    assert not any(k == "also accepted" for k, _ in S.gold_lines(record, None))
    # a single-view record keeps one row and its evidence on every image
    single = dict(record, prompt=dict(record["prompt"], images=["img_scan_4"], marks=[],
                                      image_legend=None),
                  label=dict(record["label"], view="scan", views=None, image_steps=None))
    one = S.item_html(2, single, manifest, None, tmp_path / "images", tmp_path / "out", 1)
    assert re.findall(r'<div class="frames ([a-z ]*)">', one) == ["single"]
    assert '&quot;kind&quot;:&quot;evidence&quot;' in one and "&quot;mark&quot;" not in one


def test_the_page_opens_on_the_models_view(tmp_path):
    """Owner 2026-11-04: the cyan / yellow reviewer boxes beside the red marks were taken for
    marks the model sees. The picture shows the model's view; the reviewer boxes are a layer
    of their own, off by default, behind a "show reviewer boxes / 显示审核框" switch, with a
    one-line legend saying the model does not see them; the marks are drawn with the
    converter's own mark-v2 parameters and rule."""
    import vlm_to_chat as C

    record, manifest = _multi_v3(tmp_path)
    page = S.page("t", S.item_html(1, record, manifest, None, tmp_path / "images",
                                   tmp_path / "out", 1),
                  sheet="s", js=S.JS_AUDIT, columns=S.CSV_COLUMNS, guide=S.AUDIT_GUIDE)
    assert '<input type="checkbox" id="toggle-review">' in page  # not checked
    assert "show reviewer boxes / 显示审核框" in page
    assert "not shown to the model" in page and "模型看不到" in page
    assert 'id="toggle-boxes"' not in page
    # the script draws marks into the model layer, evidence / answer into the review layer
    for needle in ("function drawModel(", "function drawReview(", "function labelPlace(",
                   "Math.max(2, Math.round(2 * W / 1600))", "Math.round(14 * W / 1600)",
                   "Math.round(8 * W / 1600)", "Math.round(18 * W / 1600)",
                   "classList.toggle('show-review', tr.checked)"):
        assert needle in page, needle
    assert f'"opacity": {C.MARK_OPACITY}' in page and C.MARK_STYLE_VERSION in page
    assert C.mark_width(1600) == 2 and C.mark_dash(1600) == (14, 8) \
        and C.label_font_size(1600) == 18  # the numbers the page transcribes
    # no reviewer box label carries an ordinal, and the human form has no reviewer switch
    article = H.item_html(1, record, manifest, tmp_path / "images", tmp_path / "out")
    assert "toggle-review" not in article


def test_v3_hint_is_on_the_models_layer(tmp_path):
    record, manifest = _multi_v3(tmp_path)
    record = dict(record, prompt=dict(record["prompt"], marks=[], hint_regions=[
        {"image": 0, "bbox": [50, 40, 300, 300]}, {"image": 4, "bbox": [50, 40, 300, 300]}]),
        label=dict(record["label"], v3_form="roi"))
    assert [b["kind"] for b in S.marks_of(record, 0)] == ["hint"]
    assert S.marks_of(record, 4)[0]["bbox"] == [50.0, 40.0, 300.0, 300.0]
    assert S.marks_of(record, 1) == []
    article = S.item_html(1, record, manifest, None, tmp_path / "images", tmp_path / "out", 1)
    assert article.count("&quot;kind&quot;:&quot;hint&quot;") == 2
    assert ("form", "roi (a yellow dashed hint box is drawn)") in S.gold_lines(record, None)


def test_audit_sheet_never_marks_a_box_task(tmp_path):
    record, manifest = _multi_v3(tmp_path)
    v8 = dict(record, prompt=dict(record["prompt"], task="V8", images=["img_scan_4"],
                                  image_legend=None),
              label=dict(record["label"], view="scan", views=None, image_steps=None,
                         answer={"instance": "cpu_cooler.01", "bbox": [1, 2, 3, 4]}))
    article = S.item_html(1, v8, manifest, None, tmp_path / "images", tmp_path / "out", 1)
    assert "&quot;mark&quot;" not in article and "Outlined in red" not in article
    assert S.marks_of(v8, 0) == []


def test_the_human_form_shows_the_marks_but_no_evidence(tmp_path):
    record, manifest = _multi_v3(tmp_path)
    article = H.item_html(1, record, manifest, tmp_path / "images", tmp_path / "out")
    boxes = [json.loads(html_unescape(b)) for b in re.findall(r'data-boxes="([^"]*)"', article)]
    assert [[b["kind"] for b in bs] for bs in boxes] == [["mark", "mark"]] + [[]] * 7
    shown = re.sub(r'(src|alt)="[^"]*"', "", article)  # the image path carries view and step (by design)
    assert "cpu_cooler.01" not in shown and "step 4" not in shown and "D07" not in shown, \
        [m.start() for m in re.finditer("step 4|D07|cpu_cooler.01", shown)]
    assert "image 1 of 8: top-down scanner, before" in article
    assert "Outlined in red: A" in article


def html_unescape(text: str) -> str:
    import html

    return html.unescape(text)


def test_the_sheet_has_one_item_per_record_in_batches(scene, audit_sheet):
    records = scene["audit"]
    result = audit_sheet["result"]
    # (vlm_sample.draw over-draws on a one-desktop pool: more than the 20 asked; W4-A's note)
    assert result["items"] == len(records) >= 20
    n_batches = math.ceil(len(records) / 8)
    assert result["batches"] == n_batches == len(audit_sheet["pages"]) >= 3
    assert result["files"] == [f"audit.sheet.b{b:02d}.html" for b in range(1, n_batches + 1)] \
        + ["audit.sheet.csv"]
    articles = [a for page in audit_sheet["pages"] for a in _articles(page)]
    assert len(articles) == len(records)
    ids = re.findall(r'data-id="(rec_[0-9a-f]+)"', "".join(audit_sheet["pages"]))
    assert ids == [r["id"] for r in records]  # the sample's order, task by task
    assert not result["images_without_manifest"]


def test_two_image_items_show_both_frames_and_paths_resolve(scene, audit_sheet):
    by_id = {r["id"]: r for r in scene["audit"]}
    two = 0
    for page in audit_sheet["pages"]:
        for article in _articles(page):
            rid = re.search(r'data-id="([^"]+)"', article).group(1)
            srcs = re.findall(r'<img src="([^"]+)"', article)
            assert len(srcs) == len(by_id[rid]["prompt"]["images"])
            two += len(srcs) == 2
            for src in srcs:
                assert src.startswith("../../images/"), src
                assert (audit_sheet["out"] / src).resolve().is_file(), src
            if len(srcs) == 2:
                assert "image 1 of 2" in article and "image 2 of 2" in article
    assert two >= 1  # V3 / V12 pairs are in a 20-item scene audit


def test_the_sheet_shows_the_gold_answer_evidence_and_rationale(scene, audit_sheet):
    manifest, worlds = S.load_context(scene["exports"])
    world = worlds[W.DESKTOP]
    text = "".join(audit_sheet["pages"])
    for r in scene["audit"]:
        article = next(a for a in _articles(text) if f'data-id="{r["id"]}"' in a)
        assert r["label"]["readable_id"] in article
        assert "Gold answer" in article and "raw JSON" in article
        # the raw JSON travels whole (escaped), the words name the parts
        assert S.esc(json.dumps(r["label"]["answer"], ensure_ascii=False, indent=1)) in article
        for key, value in S.gold_lines(r, world):
            assert S.esc(value) in article, (r["prompt"]["task"], key)
        if (r["label"].get("rationale") or {}).get("steps"):
            assert 'class="rationale"' in article
    # the gold option of a multiple choice is marked; evidence boxes are data
    mc = [r for r in scene["audit"] if r["prompt"].get("options")]
    for r in mc:
        article = next(a for a in _articles(text) if f'data-id="{r["id"]}"' in a)
        assert article.count('<li class="gold">') == 1
    v8 = next((r for r in scene["audit"] if r["prompt"]["task"] == "V8"), None)
    if v8 is not None:
        boxes = S.boxes_of(v8, world)
        assert boxes[0]["bbox"] == [float(v) for v in _keys(v8)["bbox"]]
        assert boxes[0]["kind"] == "answer"
        # a box label names the class, never an ordinal (owner 2026-11-04)
        assert not re.search(r"\s\d+$", boxes[0]["label"]), boxes[0]["label"]
    # parts are named without an ordinal: the oracle's name, never the key nor "X 1"
    v3 = next(r for r in scene["audit"] if r["prompt"]["task"] == "V3")
    lines = dict(S.gold_lines(v3, world))
    key = _keys(v3)["target_instance"]
    assert world.plain_name(key) in lines["action"] and key not in lines["action"]
    assert world.instances[key]["label"] not in lines["action"]


def test_no_ordinal_in_anything_the_sheet_says(scene, audit_sheet):
    """Owner 2026-11-04 (#97 "CPU 1", "CPU socket lever 1", "connector 3", "motherboard 1"): the
    gold lines, the rationale, the evidence line and the box labels of every item carry no
    ordinal -- a part alone of its class is its class, a marked part "part A", a group its label."""
    _, worlds = S.load_context(scene["exports"])
    world = worlds[W.DESKTOP]
    labels = {inst["label"] for inst in world.instances.values() if re.search(r"\d$", inst["label"])}
    assert labels
    for r in scene["audit"]:
        texts = [v for _, v in S.gold_lines(r, world)] + S.rationale_lines(r, world)
        texts += [b["label"] for b in S.boxes_of(r, world)]
        for text in texts:
            for lab in labels:
                assert not re.search(rf"\b{re.escape(lab)}\b", text), (r["label"]["readable_id"], text)
    # the three rules, one by one
    unique = next(k for k, i in world.instances.items()
                  if sum(1 for j in world.instances.values()
                         if j.get("match_class") == i.get("match_class")) == 1)
    assert world.plain_name(unique) == world.instances[unique]["name"]
    assert not re.search(r"\d", world.plain_name(unique))
    gid = next(iter(world.groups))
    assert S.display_name(gid, world) == world.groups[gid]["label"]
    marked = next(r for r in scene["audit"] if r["label"].get("mark_names")
                  and any(len(e["parts"]) == 1 for e in r["label"]["mark_names"].values()))
    tag, entry = next((t, e) for t, e in marked["label"]["mark_names"].items()
                      if len(e["parts"]) == 1)
    assert S.display_name(entry["parts"][0], world, marked) == f"part {tag}"
    # no oracle at all: the key's words, without its ordinal
    assert S.display_name("cpu_socket_lever.01", None) == "cpu socket lever"


def test_the_sheet_is_static_fillable_and_deterministic(scene, audit_sheet, tmp_path):
    for page in audit_sheet["pages"]:
        assert "<script src" not in page and "<link " not in page
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", page)
        for value, zh, _ in S.VERDICTS:
            assert f'value="{value}"' in page and zh in page
        assert 'id="export"' in page and "localStorage" in page and 'class="note"' in page
        assert "D:/DataSet" not in page and str(scene["tmp"]) not in page
    skeleton = (audit_sheet["out"] / "audit.sheet.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(skeleton)))
    assert [r["id"] for r in rows] == [r["id"] for r in scene["audit"]]
    assert list(rows[0]) == list(S.CSV_COLUMNS)
    assert {r["batch"] for r in rows} == {str(b) for b in range(1, audit_sheet["result"]["batches"] + 1)}
    assert rows[7]["batch"] == "1" and rows[8]["batch"] == "2"
    assert all(r["verdict"] == "" and r["note"] == "" for r in rows)
    # a second build writes the same bytes (no clock, no machine path)
    again = tmp_path / "again"
    S.build(scene["audit"], scene["exports"], scene["images"], again, batch=8)
    for name in audit_sheet["result"]["files"]:
        a, b = (audit_sheet["out"] / name).read_text(encoding="utf-8"), \
            (again / name).read_text(encoding="utf-8")
        if name.endswith(".html"):
            # only the relative image paths differ between two directories
            a = a.replace("../../images/", "IMG/")
            b = b.replace(Path(S.rel_image(scene["images"], "", again)).as_posix().rstrip("/") + "/",
                          "IMG/")
        assert a == b, name


def test_the_cli_builds_the_sheet(scene, tmp_path, capsys):
    out = tmp_path / "cli"
    rc = S.main(["--records", str(scene["eval"] / "audit" / "audit.jsonl"),
                 "--export", *map(str, scene["exports"]), "--image-root", str(scene["images"]),
                 "--out", str(out), "--batch", "50"])
    assert rc == 0
    assert (out / "audit.sheet.b01.html").is_file() and (out / "audit.sheet.csv").is_file()
    assert f"{len(scene['audit'])} items in 1 pages" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Wilson and the summary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 300, 0.0, 0.01264),        # the paper's "0 errors in 300 bounds it at about 1 %"
    (81, 263, 0.2553, 0.3662),     # Newcombe (1998), method 3
    (15, 148, 0.0624, 0.1605),     # Newcombe (1998)
    (0, 20, 0.0, 0.1611),          # Newcombe (1998)
    (1, 29, 0.0061, 0.1718),       # Newcombe (1998)
    (5, 10, 0.2366, 0.7634),       # symmetric about 0.5
    (10, 10, 0.7225, 1.0),
])
def test_the_wilson_interval_matches_known_values(k, n, lo, hi):
    p, got_lo, got_hi = A.wilson(k, n)
    assert p == pytest.approx(k / n)
    assert got_lo == pytest.approx(lo, abs=5e-5)
    assert got_hi == pytest.approx(hi, abs=5e-5)
    assert 0.0 <= got_lo <= p <= got_hi <= 1.0


def test_the_wilson_interval_edge_cases():
    assert A.wilson(0, 0) == (None, None, None)
    with pytest.raises(ValueError):
        A.wilson(11, 10)
    # wider for a smaller n, narrower for a larger one, same proportion
    _, lo_s, hi_s = A.wilson(3, 30)
    _, lo_l, hi_l = A.wilson(30, 300)
    assert hi_s - lo_s > hi_l - lo_l
    # the 90 % interval is inside the 95 % one
    _, lo90, hi90 = A.wilson(7, 50, z=1.6448536)
    _, lo95, hi95 = A.wilson(7, 50)
    assert lo95 < lo90 and hi90 < hi95


def test_verdict_spellings():
    assert A.normalise_verdict("正确") == "correct" and A.normalise_verdict(" WRONG ") == "wrong"
    assert A.normalise_verdict("?") == "doubtful" and A.normalise_verdict("存疑") == "doubtful"
    assert A.normalise_verdict("") is None and A.normalise_verdict("banana") is None


def _filled_csv(records, verdicts: dict, reviewer: str, path: Path, notes=None) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(S.CSV_COLUMNS), lineterminator="\r\n")
    writer.writeheader()
    for i, r in enumerate(records):
        row = S.csv_row(r, i // 8 + 1)
        row["verdict"] = verdicts.get(r["id"], "")
        row["note"] = (notes or {}).get(r["id"], "")
        row["reviewer"] = reviewer
        row["answered_at"] = "2026-11-03T10:00:00Z" if row["verdict"] else ""
        writer.writerow(row)
    path.write_text("\ufeff" + buf.getvalue(), encoding="utf-8", newline="")
    return path


def test_the_summary_counts_rates_flags_and_disagreements(scene, tmp_path):
    records = scene["audit"]
    ids = [r["id"] for r in records]
    tasks = {r["id"]: r["prompt"]["task"] for r in records}
    # reviewer A: item 0 wrong, item 1 doubtful, the last one not reviewed, the rest correct
    verdicts_a = {i: "正确" for i in ids[:-1]}
    verdicts_a[ids[0]] = "错误"
    verdicts_a[ids[1]] = "存疑"
    notes = {ids[0]: "the screw is already out", ids[1]: "blurred | cannot see the latch"}
    a = _filled_csv(records, verdicts_a, "A", tmp_path / "a.csv", notes)
    # reviewer B marks items 0 and 2 only: agrees on 0 (wrong), says 2 is wrong (A: correct)
    b = _filled_csv(records, {ids[0]: "wrong", ids[2]: "wrong"}, "B", tmp_path / "b.csv",
                    {ids[2]: "wrong frame"})
    rows = A.read_verdicts([a, b])
    assert len(rows) == 2 * len(records)
    result = A.summarise(rows, records)
    assert result["items_total"] == len(records)
    assert result["items_reviewed"] == len(records) - 1
    assert result["not_reviewed"] == [ids[-1]]
    o = result["overall"]
    assert (o["wrong"], o["doubtful"]) == (2, 1)  # item 2 takes B's worse verdict
    assert o["error_rate"]["k"] == 2 and o["error_rate"]["n"] == len(records) - 1
    p, lo, hi = A.wilson(2, len(records) - 1)
    assert o["error_rate"]["rate"] == pytest.approx(p, abs=1e-5)
    assert o["error_rate"]["ci95"] == [pytest.approx(lo, abs=1e-5), pytest.approx(hi, abs=1e-5)]
    assert o["error_or_doubt_rate"]["k"] == 3
    per = result["by_task"]
    assert sum(c["reviewed"] for c in per.values()) == len(records) - 1
    assert per[tasks[ids[0]]]["wrong"] >= 1
    flagged = {f["id"]: f for f in result["flagged"]}
    assert set(flagged) == {ids[0], ids[1], ids[2]}
    assert "already out" in flagged[ids[0]]["note"] and "A:" in flagged[ids[0]]["note"]
    assert flagged[ids[1]]["note"] == notes[ids[1]]
    agr = result["agreement"]
    assert agr["items_with_several_verdicts"] == 2 and agr["pairs"] == 2 and agr["agree"] == 1
    assert [d["id"] for d in agr["disagreements"]] == [ids[2]]
    md = A.markdown(result, "trial")
    assert "| **all** |" in md and "Wilson" in md
    assert "Items marked wrong or doubtful" in md and "the screw is already out" in md
    assert "blurred \\| cannot see the latch" in md
    assert "Agreement between reviewers" in md and "Not yet reviewed (1)" in md
    for task in per:
        assert f"| {task} |" in md
    # the CLI writes both files
    out = tmp_path / "summary"
    rc = A.main(["--csv", str(a), str(b), "--records",
                 str(scene["eval"] / "audit" / "audit.jsonl"), "--out", str(out), "--name", "trial"])
    assert rc == 0
    data = json.loads((out / "trial.summary.json").read_text(encoding="utf-8"))
    assert data["overall"]["wrong"] == 2
    assert (out / "trial.summary.md").read_text(encoding="utf-8") == md


def test_zero_errors_in_the_paper_sample_reads_as_one_point_three_percent():
    rows = [{"id": f"r{i}", "task": "V4", "verdict": "correct", "verdict_raw": "correct",
             "source": "x.csv", "reviewer": "A", "note": ""} for i in range(300)]
    result = A.summarise(rows)
    assert result["overall"]["error_rate"]["rate"] == 0.0
    assert result["overall"]["error_rate"]["ci95"][1] == pytest.approx(0.01264, abs=5e-5)
    assert "0.0% [0.0%, 1.3%]" in A.markdown(result)


# --------------------------------------------------------------------------- #
# carrying the first audit's V2 verdicts over (spec 6.7)
# --------------------------------------------------------------------------- #
def _v2_rec(rid, readable, desktop, step, check, answer, *, view="scan", task="V2", evidence=None):
    return {"id": rid, "prompt": {"task": task, "images": ["img_x"], "question": "?"},
            "label": {"readable_id": readable, "desktop": desktop, "step": step, "view": view,
                      "answer": answer, "answer_check": check,
                      "evidence": {"instances": evidence or []}}}


def _state(instance, **extra):
    return dict({"type": "exact", "derive": "instance_state", "instance": instance,
                 "class": instance.split(".")[0], "fields": ["state"]}, **extra)


def _count(role):
    return {"type": "exact", "derive": "screw_count", "role": role, "state": "present",
            "fields": ["count"]}


def test_carry_v2_by_desktop_step_part(tmp_path):
    import vlm_audit_carry as K

    old = [
        _v2_rec("rec_o1", "V2-D24-scan-s002-state-cable_clip.01", 24, 2, _state("cable_clip.01"),
                {"state": "open"}),
        _v2_rec("rec_o2", "V2-D46-scan-s033-count-motherboard", 46, 33, _count("motherboard"),
                {"count": 2}),
        _v2_rec("rec_o3", "V2-D41-scan-s024-state-cpu_socket_lever.01", 41, 24,
                _state("cpu_socket_lever.01"), {"state": "open"}),
        _v2_rec("rec_o4", "V2-D57-scan-s008-state-ram_latch.02", 57, 8, _state("ram_latch.02"),
                {"state": "closed"}),
        _v2_rec("rec_o5", "V2-D10-scan-s004-state-psu_latch.01", 10, 4, _state("psu_latch.01"),
                {"state": "closed"}),
        _v2_rec("rec_o6", "V3-D10-scan-s005", 10, 5, {"type": "exact"}, {"verb": "remove"},
                task="V3"),
    ]
    old_jsonl = tmp_path / "old_audit.jsonl"
    IO.write_jsonl(old_jsonl, old)
    # two exported pages, one re-exported with a later answer, one verdict in Chinese, one blank
    _filled_csv(old[:3], {"rec_o1": "correct", "rec_o2": "doubtful", "rec_o3": "correct"}, "c1",
                tmp_path / "b01.csv", notes={"rec_o2": "two screws hidden?"})
    _filled_csv(old[3:], {"rec_o4": "正确", "rec_o6": "correct"}, "c1", tmp_path / "b02.csv")
    new = [
        # the asked latch is now one member of a group with the same state: carried
        _v2_rec("rec_n1", "V2-D24-scan-s002-state-cable_clip@cover.01", 24, 2,
                _state("cable_clip@cover.01", members=["cable_clip.01", "cable_clip.02"]),
                {"state": "open"}),
        # the count, unchanged: carried (with its note)
        _v2_rec("rec_n2", "V2-D46-scan-s033-count-motherboard", 46, 33, _count("motherboard"),
                {"count": 2}),
        # same part, another answer: not carried
        _v2_rec("rec_n3", "V2-D41-scan-s024-state-cpu_socket_lever.01", 41, 24,
                _state("cpu_socket_lever.01"), {"state": "closed"}),
        # same part and answer on another step / another view: not carried
        _v2_rec("rec_n4", "V2-D57-scan-s009-state-ram_latch.02", 57, 9, _state("ram_latch.02"),
                {"state": "closed"}),
        _v2_rec("rec_n5", "V2-D57-oak1-s008-state-ram_latch.02", 57, 8, _state("ram_latch.02"),
                {"state": "closed"}, view="oak1"),
        # a group known only through its evidence (no members field): carried
        _v2_rec("rec_n6", "V2-D57-scan-s008-state-ram_latch@ram_module.01", 57, 8,
                _state("ram_latch@ram_module.01"), {"state": "closed"},
                evidence=["ram_latch.01", "ram_latch.02"]),
        # a V3 is audited afresh, whatever its old verdict
        _v2_rec("rec_n7", "V3-D10-multi-s005", 10, 5, {"type": "exact"}, {"verb": "remove"},
                task="V3", view="multi"),
    ]
    new_jsonl = tmp_path / "new_audit.jsonl"
    IO.write_jsonl(new_jsonl, new)
    out = tmp_path / "carried.csv"
    assert K.main(["--old-csv", str(tmp_path / "b01.csv"), str(tmp_path / "b02.csv"),
                   "--old-records", str(old_jsonl), "--new-records", str(new_jsonl),
                   "--out", str(out), "--batch", "4"]) == 0
    rows = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8-sig"))))
    assert list(rows[0]) == list(S.CSV_COLUMNS) == list(K.CSV_COLUMNS)
    assert [r["id"] for r in rows] == [r["id"] for r in new]
    got = {r["id"]: r for r in rows}
    assert got["rec_n1"]["verdict"] == "correct" and "V2-D24-scan-s002-state-cable_clip.01" in \
        got["rec_n1"]["note"]
    assert got["rec_n2"]["verdict"] == "doubtful" and "two screws hidden?" in got["rec_n2"]["note"]
    assert got["rec_n2"]["reviewer"] == "c1" and got["rec_n2"]["answered_at"]
    assert got["rec_n6"]["verdict"] == "correct"  # 正确, normalised
    for rid in ("rec_n3", "rec_n4", "rec_n5", "rec_n7"):
        assert got[rid]["verdict"] == "" and got[rid]["note"] == "" and got[rid]["reviewer"] == ""
    assert [r["batch"] for r in rows] == ["1"] * 4 + ["2"] * 3
    assert got["rec_n1"]["readable_id"] == "V2-D24-scan-s002-state-cable_clip@cover.01"
    # the carried file reads back through the summary like any exported page
    summary = A.summarise(A.read_verdicts([out]), new)
    assert summary["overall"]["reviewed"] == 3
    # the old CSVs are only read
    assert "rec_o1" in (tmp_path / "b01.csv").read_text(encoding="utf-8-sig")
    # an old verdict whose record is not in --old-records is reported, not guessed
    report = K.carry(A.read_verdicts([tmp_path / "b01.csv"]), old[1:], new)
    assert report["unknown_old_ids"] == ["rec_o1"]


# --------------------------------------------------------------------------- #
# the human-baseline form
# --------------------------------------------------------------------------- #
LEAKS = ("rationale", "readable_id", "evidence", "truth_source", "demonstrated",
         "state_inapplicable", "graph_blocked", "captive_screw", "hidden_events",
         "prior_is_correct", "answer_check", "different_moment", "unanswerable")


@pytest.fixture(scope="module")
def human(scene) -> dict:
    records = [r for p in scene["exports"] for r in IO.read_jsonl(p)]
    chosen = H.draw_subset(records, size=40, per_task=4, seed=0)
    out = scene["tmp"] / "human"
    result = H.build(chosen, scene["exports"], scene["images"], out, batch=16)
    pages = [(out / f).read_text(encoding="utf-8") for f in result["files"] if f.endswith(".html")]
    return {"records": chosen, "out": out, "result": result, "pages": pages}


def test_the_human_form_has_no_answer_in_it(scene, human):
    text = "".join(human["pages"])
    assert len(_articles(text)) == len(human["records"]) == human["result"]["items"]
    for r in human["records"]:
        label = r["label"]
        article = next(a for a in _articles(text) if f'data-id="{r["id"]}"' in a)
        assert json.dumps(label["answer"], ensure_ascii=False) not in article
        assert S.esc(json.dumps(label["answer"], ensure_ascii=False)) not in article
        assert label["readable_id"] not in article
        for leak in LEAKS:
            assert leak not in article, (leak, r["prompt"]["task"])
        # the question, the options and the model's format sentence are there
        assert S.esc(r["prompt"]["question"]) in article
        for o in r["prompt"].get("options") or []:
            assert S.esc(o["target_label"]) in article
        assert "model format:" in article
        assert f"D{W.DESKTOP:02d}" not in article.split("<img")[0]  # no desktop / step in the header
        # the gold option of a multiple choice is not marked
        assert 'class="gold"' not in article
        # no evidence boxes shipped with the frame: only the red outlines the model sees
        # (redesign 2026-11-03), each on its own image
        shipped = [json.loads(html_unescape(b)) for b in re.findall(r'data-boxes="([^"]*)"', article)]
        assert len(shipped) == len(r["prompt"]["images"])
        # (V3's roi form also shows its yellow hint box: the model sees it too)
        assert all(b["kind"] in ("mark", "hint") for bs in shipped for b in bs)
        assert [len(bs) for bs in shipped] == [len(S.marks_of(r, i)) for i in range(len(shipped))]
    for page in human["pages"]:
        assert "<script src" not in page and "answered_at" in page and "started_at" in page
        assert 'data-classes="' in page and "IntersectionObserver" in page


def test_the_human_form_fields_follow_the_answer_format(human):
    text = "".join(human["pages"])
    seen = set()
    for r in human["records"]:
        key = H.format_key(r)
        seen.add(key)
        article = next(a for a in _articles(text) if f'data-id="{r["id"]}"' in a)
        assert f'data-fk="{key}"' in article
        for field in H.FIELDS[key]:
            assert f'data-field="{field}"' in article, (key, field)
        n_img = len(r["prompt"]["images"])
        assert article.count("<figure") == n_img
        if key in ("V1", "V8"):
            assert article.count('class="frame drawable"') == n_img
            assert f'data-boxmode="{"multi" if key == "V1" else "single"}"' in article
        if key == "V6.options":
            assert article.count('data-field="choice"') == len(r["prompt"]["options"])
        if key == "V2.state":
            # V14 merged into V2: the state radio lists cannot_tell too
            assert f'value="{H.CANNOT_TELL}"' in article
            for state in H.allowed_states(r):
                assert f'value="{state}"' in article
        if key in ("V12", "V15.moment", "V4", "V16"):
            assert 'value="yes"' in article and 'value="no"' in article
    assert {"V12", "V4", "V6.open", "V3"} <= seen
    ids = (human["out"] / "human.ids.txt").read_text(encoding="utf-8").split()
    assert ids == [r["id"] for r in human["records"]]


def test_pairs_are_shown_open_form_first():
    def rec(form, pair, n):
        return {"id": f"r{n}", "prompt": {"task": "V6"}, "label": {"v6_form": form, "v6_pair": pair}}

    rows = [rec("mc", "p1", 1), rec("open", "p2", 2), rec("open", "p1", 3), rec("mc", "p3", 4)]
    out = H.order_pairs(rows)
    assert [r["id"] for r in out] == ["r3", "r2", "r1", "r4"]


def test_the_default_draw_is_stratified_and_not_the_audit(scene):
    records = [r for p in scene["exports"] for r in IO.read_jsonl(p)]
    chosen = H.draw_subset(records, size=40, per_task=4, seed=0)
    tasks = {}
    for r in chosen:
        tasks[r["prompt"]["task"]] = tasks.get(r["prompt"]["task"], 0) + 1
    assert len(chosen) == len({r["id"] for r in chosen}) and len(tasks) >= 8
    assert max(tasks.values()) <= 8  # the 20 % cap
    audit_ids = {r["id"] for r in scene["audit"]}
    assert {r["id"] for r in chosen} != audit_ids
    again = H.draw_subset(records, size=40, per_task=4, seed=0)
    assert [r["id"] for r in again] == [r["id"] for r in chosen]


def test_fields_map_to_the_model_formats():
    assert H.to_answer("V4", {"feasible": "no", "blockers": "CPU cooler 1; Fan connector 1"}) == \
        {"feasible": False, "blockers": ["CPU cooler 1", "Fan connector 1"]}
    assert H.to_answer("V4", {"feasible": "yes", "blockers": "ignored"}) == {"feasible": True, "blockers": []}
    assert H.to_answer("V7", {"plan": "1. unscrew Motherboard screw 2\n2) remove PSU 1\n\n- open Lever 1"}) == \
        {"plan": [{"verb": "unscrew", "target": "Motherboard screw 2"},
                  {"verb": "remove", "target": "PSU 1"}, {"verb": "open", "target": "Lever 1"}]}
    assert H.to_answer("V16", {"valid": "yes", "first_error_step": "3"}) == \
        {"valid": True, "first_error_step": None}
    assert H.to_answer("V16", {"valid": "no", "first_error_step": "3"}) == \
        {"valid": False, "first_error_step": 3}
    assert H.to_answer("V2.state", {"state": H.CANNOT_TELL}) == {"state": H.CANNOT_TELL}
    assert H.to_answer("V6.open", {"actions": "disconnect the connectors\nopen the cover"}) == \
        {"actions": [{"verb": "disconnect", "target": "the connectors"},
                     {"verb": "open", "target": "the cover"}]}
    assert H.to_answer("V2.count", {"count": "3"}) == {"count": 3}
    assert H.to_answer("V2.count", {"count": "three"}) is None
    assert H.to_answer("V1", {"boxes": json.dumps([{"class": "cpu", "bbox": [1, 2, 3, 4]},
                                                   {"class": "", "bbox": [5, 6, 7, 8]}])}) == \
        {"components": [{"class": "cpu", "bbox": [1, 2, 3, 4]}]}
    assert H.to_answer("V8", {"boxes": "[]"}) is None
    assert H.to_answer("V12", {"changed": ""}) is None
    assert H.to_answer("V3", {"verb": "Remove", "target": "PSU 1", "tool": ""}) == \
        {"verb": "remove", "target": "PSU 1", "tool": None}


def _gold_csv(records, worlds, path: Path, *, blank_first: bool = False) -> Path:
    rows = []
    for i, r in enumerate(records):
        fields = {} if (blank_first and i == 0) else H.gold_fields(r, worlds.get(IO.desktop_of(r)))
        rows.append({"id": r["id"], "task": r["prompt"]["task"], "format_key": H.format_key(r),
                     "fields_json": json.dumps(fields), "started_at": "2026-11-03T10:00:00.000Z",
                     "answered_at": "2026-11-03T10:00:42.000Z", "seconds": "42", "reviewer": "gold"})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(H.CSV_COLUMNS), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text("\ufeff" + buf.getvalue(), encoding="utf-8", newline="")
    return path


def test_gold_answers_through_the_form_score_one(scene, tmp_path):
    """CSV -> convert -> cut truth -> vlm_score: 1.0 on every task row of both views.

    On the whole scene export (every task; the one-desktop eval cut is small).
    """
    for view in VIEWS:
        truth = scene["data"] / f"vlm_{view}.jsonl"
        records = list(IO.read_jsonl(truth))
        worlds, _ = IO.load_oracle(truth)
        csv_path = _gold_csv(records, worlds, tmp_path / f"{view}.csv")
        out = tmp_path / view
        rc = H.main(["convert", "--csv", str(csv_path), "--out", str(out), "--export", str(truth),
                     "--name", "person"])
        assert rc == 0
        preds = list(IO.read_jsonl(out / "person.pred.jsonl"))
        assert [p["id"] for p in preds] == [r["id"] for r in records]
        assert all(p["meta"]["seconds"] == 42.0 and p["meta"]["reviewer"] == "gold" for p in preds)
        cut = out / f"person.{view}.jsonl"
        assert [r["id"] for r in IO.read_jsonl(cut)] == [r["id"] for r in records]
        assert IO.sidecar(cut, "oracle.jsonl").is_file() and IO.sidecar(cut, "images_manifest.jsonl").is_file()
        result = vlm_score.score(cut, out / "person.pred.jsonl", resamples=20)
        assert result["predicted"] == result["records"] == len(records)
        for mode in ("strict", "lenient"):
            assert result[mode]["parse_failure_rate_all"] == 0.0
            for task, cell in result[mode]["by_task"].items():
                assert cell["mean"] == pytest.approx(1.0), (view, mode, task)
        tasks = set(result["strict"]["by_task"])
        if view == W.VIEW:
            assert {"V1", "V8", "V2", "V6", "V7", "V10", "V16", "V12", "V3", "V4"} <= tasks


def test_convert_skips_blank_rows_and_passes_answer_json_through(scene, tmp_path):
    truth = scene["eval"] / f"{W.VIEW}.jsonl"
    records = list(IO.read_jsonl(truth))[:6]
    worlds, _ = IO.load_oracle(truth)
    csv_path = _gold_csv(records, worlds, tmp_path / "p.csv", blank_first=True)
    rows = H.read_csvs([csv_path])
    rows[1]["answer_json"] = '{"hand": "written"}'
    preds, stats = H.convert(rows)
    assert stats == {"answered": 5, "skipped": 1, "answer_json_rows": 1}
    assert [p["id"] for p in preds] == [r["id"] for r in records[1:]]
    assert json.loads(preds[0]["raw_text"]) == {"hand": "written"}
    # the cut truth holds only the answered ids
    cut = H.cut_truth([truth], {p["id"] for p in preds}, tmp_path, "p")
    assert cut == {W.VIEW: 5}
    assert [r["id"] for r in IO.read_jsonl(tmp_path / f"p.{W.VIEW}.jsonl")] == [r["id"] for r in records[1:]]


def test_the_human_cli_builds_from_an_ids_file(scene, tmp_path, capsys):
    ids = [r["id"] for r in scene["audit"][:5]]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("".join(i + "\n" for i in ids) + "rec_nowhere\n", encoding="utf-8")
    out = tmp_path / "form"
    rc = H.main(["build", "--export", *map(str, scene["exports"]), "--image-root",
                 str(scene["images"]), "--out", str(out), "--ids", str(ids_file), "--batch", "2"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "5 items" in text and "1 ids are in no truth file" in text
    assert sorted(p.name for p in out.glob("*.html")) == [f"human.sheet.b0{i}.html" for i in (1, 2, 3)]
    assert (out / "human.ids.txt").read_text(encoding="utf-8").split() == ids


# --------------------------------------------------------------------------- #
# the pack
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> dict:
    """``export_vlmpack.build_pack`` over the scene (as tests/test_w4a_vlmpack.py does)."""
    from PIL import Image

    import export_vlmpack as P

    root = tmp_path_factory.mktemp("w4e_pack")
    db_path = root / "scene.sqlite"
    db = Db(str(db_path))
    W.build(db)
    db.close()
    cache = root / "cache"
    for view in VIEWS:
        for step in W.STEPS:
            path = cache / view / f"D{W.DESKTOP:02d}" / f"s{step:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", W.HW, (step * 20 % 255, 90, 120)).save(path)
    summary = P.build_pack(str(db_path), str(root / "pack"), cache_dir=str(cache),
                           work_dir=str(root / "work"), views=VIEWS, seed=0, audit=20,
                           live_db=str(root / "live.sqlite"), log=lambda *_: None)
    return {"pack": root / "pack", "summary": summary,
            "manifest": json.loads((root / "pack" / "manifest.json").read_text(encoding="utf-8"))}


def test_the_pack_ships_the_sheet_and_the_tools(pack):
    import export_vlmpack as P

    root = pack["pack"]
    audit = root / "eval_v3" / "audit"
    assert (audit / "audit.sheet.b01.html").is_file() and (audit / "audit.sheet.csv").is_file()
    page = (audit / "audit.sheet.b01.html").read_text(encoding="utf-8")
    srcs = set(re.findall(r'<img src="([^"]+)"', page))
    assert srcs and all(s.startswith("../../images/") for s in srcs)
    assert all((audit / s).resolve().is_file() for s in srcs)
    n = len((audit / "audit.ids.txt").read_text(encoding="utf-8").split())
    assert len(_articles(page)) == n >= 20
    for name in ("vlm_audit_sheet.py", "vlm_audit_summary.py", "vlm_human_sheet.py"):
        assert name in P.TOOLS
        assert (root / "tools" / name).read_bytes() == (SCRIPTS / name).read_bytes()
    sheet = pack["summary"]["eval"]["audit"]["sheet"]
    assert sheet["batches"] == 1 and sheet["files"] == ["audit.sheet.b01.html", "audit.sheet.csv"]
    files = pack["manifest"]["files"]
    assert "eval_v3/audit/audit.sheet.b01.html" in files and "eval_v3/audit/audit.sheet.csv" in files
    assert files["eval_v3/audit/audit.sheet.csv"]["sha256"] == _sha(audit / "audit.sheet.csv")
    # the vendored tools import from the pack alone (the build already did this with -I)
    P.smoke_import(root / "tools")


def test_the_card_has_the_author_audit_paragraph(pack):
    # (redesign 2026-11-03: the human baseline is out of the plan -- spec section 2 --
    # so the card keeps the author audit only; the answer form ships as a tool, outside
    # the evaluation plan)
    text = (pack["pack"] / "README.md").read_text(encoding="utf-8")
    assert "## Author audit" in text and "## Audit and human baseline" not in text
    for needle in ("eval_v3/audit/audit.sheet.b01.html", "audit.sheet.csv",
                   "vlm_audit_summary.py", "--records eval_v3/audit/audit.jsonl",
                   "Wilson 95 % intervals", "0 of 262 -> [0, 1.4 %]", "vlm_audit_carry.py",
                   "exactly as the model gets", "show reviewer boxes / 显示审核框",
                   "do not show them to anyone answering the questions"):
        assert needle in text, needle
    for gone in ("vlm_human_sheet.py build", "vlm_human_sheet.py convert",
                 "human-baseline participant", "eval_v1/"):
        assert gone not in text, gone
    assert "vlm_human_sheet.py" in text and "not part of the evaluation plan" in text
    assert "D:/DataSet" not in text
