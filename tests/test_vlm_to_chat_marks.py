"""What the model is shown (spec 3.5): marks drawn on the images, the image legend, the glossary.

Hand-made records in the shared record contract (``prompt.marks``, ``prompt.terms``,
``prompt.state_terms``, ``prompt.image_legend``, ``label.views``) on tiny generated frames;
no database, no model, no network.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vlm_to_chat as C  # noqa: E402

RED = (0xFF, 0x17, 0x44)
GREY = (90, 90, 90)
VIEWS = ("scan", "oak1", "oak2", "rs")


def _frame(root: Path, rel: str, size) -> None:
    from PIL import Image

    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, GREY).save(path)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v2(images, marks, **prompt) -> dict:
    """A V2 state record (one image unless told otherwise) asking about the part outlined in red."""
    return {"id": "rec_test", "prompt": dict({
        "task": "V2", "images": list(images),
        "question": "What is the state of the RAM latch outlined in red?",
        "marks": marks}, **prompt),
        "label": {"readable_id": "V2-D01-scan-s003-state-ram_latch.01", "template_id": "V2.1",
                  "answer": {"state": "open"}, "desktop": 1, "step": 3, "view": "scan",
                  "answer_check": {"type": "exact", "derive": "instance_state",
                                   "instance": "ram_latch.01", "class": "ram_latch",
                                   "fields": ["state"], "allowed": ["closed", "open", "removed"]}}}


def _is_red(px) -> bool:
    """Opaque mark red: the tag label (mark-v2 draws the outline at 60 % opacity)."""
    return abs(px[0] - RED[0]) < 40 and px[1] < 80 and abs(px[2] - RED[2]) < 50


def _is_mark(px) -> bool:
    """Red, opaque or blended over the grey frame (the 60 % outline: about (189, 50, 77))."""
    return px[0] > 150 and px[1] < 90 and px[0] - px[1] > 90


def _red_box(img, test=None):
    """The bounding box of the mark pixels: ``(x0, y0, x1, y1)`` inclusive, or ``None``."""
    test = test or _is_mark
    w, h = img.size
    px = img.convert("RGB").load()
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if test(px[x, y]):
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


# --------------------------------------------------------------------------- #
# C1 marks drawn
# --------------------------------------------------------------------------- #
def test_mark_drawn_at_scaled_box(tmp_path):
    """A 1600 px frame shown at 800 px: the mark [100, 100, 50, 50] is drawn around [50, 50, 25, 25]."""
    from PIL import Image

    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (1600, 1600))
    manifest = {"img_a": {"path": "scan/D01/s003.png", "view": "scan", "desktop": 1, "step": 3}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [100, 100, 50, 50], "tag": "A"}])
    images = C.Images(str(root), str(tmp_path / "assets"), 800)
    row = C.convert(record, manifest, images, "openai")
    shown = Image.open(row["meta"]["images"][0]).convert("RGB")
    assert shown.size == (800, 800)
    assert row["meta"]["transforms"] == [{"crop": [0, 0], "scale": 0.5}]
    px = shown.load()
    # mark-v2: the outline runs just outside the scaled box [50, 50, 25, 25] (2 px at 800 px:
    # x and y 48..76), so the part itself is not painted over; it is dashed (14 / 8 at 1600 px,
    # 7 / 4 here, each side starting with a dash at its corner) and 60 % opaque
    for x, y in ((48, 50), (76, 50), (50, 48), (50, 76), (49, 60), (60, 49)):
        assert _is_mark(px[x, y]), (x, y, px[x, y])
    for x, y in ((48, 56), (56, 48), (76, 67), (67, 76)):  # the gaps of the dashes
        assert not _is_mark(px[x, y]), (x, y, px[x, y])
    assert not _is_red(px[48, 50]) and px[48, 50][0] < 230  # see-through, not opaque red
    assert not _is_mark(px[62, 62]) and not _is_mark(px[51, 51])  # inside the box
    assert not _is_mark(px[400, 400]) and not _is_mark(px[62, 120])  # elsewhere
    x0, y0, x1, y1 = _red_box(shown.crop((0, 0, 200, 200)))
    assert x0 == 48 and x1 == 76 and y1 == 76 and y0 < 47
    # the tag sits in a filled label outside the outline: above its top-left corner
    lx0, ly0, lx1, ly1 = _red_box(shown.crop((0, 0, 200, 200)), _is_red)
    assert lx0 == 48 and ly1 < 48 and ly0 >= 0


def test_mark_label_never_sits_on_the_part(tmp_path):
    """Owner 2026-11-04 (#77): the tag "A" over a CPU's printed "C" read as "APU". The label goes
    outside the outline -- below it when the box touches the top of the picture -- and a group
    of outlines under one tag carries one label, beside its top-most outline."""
    from PIL import Image

    img = Image.new("RGB", (320, 320), GREY)
    top = C.draw_marks(img.copy(), [{"image": 0, "bbox": [40, 0, 60, 30], "tag": "A"}])
    lx0, ly0, lx1, ly1 = _red_box(top, _is_red)
    assert ly0 >= 30 + C.mark_width(320)  # below the outline (its last row is 31), not on the part
    group = C.draw_marks(img.copy(), [{"image": 0, "bbox": [40, 100, 200, 12], "tag": "A"},
                                      {"image": 0, "bbox": [40, 130, 200, 12], "tag": "A"},
                                      {"image": 0, "bbox": [40, 160, 200, 12], "tag": "A"}])
    lx0, ly0, lx1, ly1 = _red_box(group, _is_red)
    assert ly1 < 100 - C.mark_width(320)  # one label, above the top-most outline only
    px = group.load()  # no opaque label pixel inside any of the three boxes
    for y0 in (100, 130, 160):
        assert not any(_is_red(px[x, y]) for x in range(40, 240) for y in range(y0, y0 + 12))
    # label_place: above, else below, else beside -- never inside
    assert C.label_place((10, 50, 60, 90), (20, 16), (200, 200)) == (10, 33)
    assert C.label_place((10, 0, 60, 90), (20, 16), (200, 200)) == (10, 91)
    assert C.label_place((30, 0, 60, 199), (20, 16), (200, 200)) == (9, 0)


def test_hint_region_is_a_dashed_box_without_tag(tmp_path):
    """V3's ``roi`` hint (``style = "hint"``): the same dashed box in yellow, no label."""
    from PIL import Image

    img = Image.new("RGB", (320, 320), GREY)
    shown = C.draw_marks(img.copy(), [{"image": 0, "bbox": [50, 50, 200, 200], "style": "hint"}])
    px = shown.load()
    yellow = lambda p: p[0] > 150 and p[1] > 130 and p[2] < 90  # noqa: E731
    assert yellow(px[48, 50]) and not yellow(px[48, 66]) and not yellow(px[150, 150])
    assert _red_box(shown) is None  # nothing red: no tag, no red outline
    assert C.HINT_RGB != C.MARK_RGB


def test_pack_image_untouched(tmp_path):
    """The marks go on an in-memory copy: the pack's frame keeps its bytes, the model gets another file."""
    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (320, 320))
    source = root / "scan/D01/s003.png"
    before = _sha(source)
    manifest = {"img_a": {"path": "scan/D01/s003.png"}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [40, 40, 30, 20], "tag": "A"}])
    images = C.Images(str(root), str(tmp_path / "assets"), None)
    row = C.convert(record, manifest, images, "openai")
    shown = Path(row["meta"]["images"][0])
    assert shown.resolve() != source.resolve() and shown.parent == tmp_path / "assets"
    assert _sha(source) == before
    assert _red_box(__import__("PIL.Image", fromlist=["Image"]).open(shown)) is not None
    assert row["meta"]["transforms"] == [{"crop": [0, 0], "scale": 1.0}]
    # an unmarked record still points at the pack's frame itself (no Pillow, no copy)
    plain = C.convert(_v2(["img_a"], []), manifest, images, "openai")
    assert Path(plain["meta"]["images"][0]) == source
    # inline: the marked copy is what is encoded
    inline = C.convert(record, manifest, images, "openai", inline=True)
    url = inline["messages"][1]["content"][1]["image_url"]["url"]
    import base64, io
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert _red_box(img) is not None


def test_mark_scaled_with_its_own_image_in_multiview(tmp_path):
    """Review focus 4: a 4056 px OAK frame beside a 1600 px scan, --max-side 1568; the mark is on the scan."""
    from PIL import Image

    root = tmp_path / "images"
    sizes = {"scan": (1600, 1600), "oak1": (4056, 3040), "oak2": (4056, 3040), "rs": (1280, 720)}
    manifest = {}
    for v in VIEWS:
        _frame(root, f"{v}/D01/s003.png", sizes[v])
        manifest[f"img_{v}"] = {"path": f"{v}/D01/s003.png", "view": v, "desktop": 1, "step": 3}
    record = _v2([f"img_{v}" for v in VIEWS], [{"image": 0, "bbox": [800, 400, 100, 100], "tag": "A"}],
                 task="V4", image_legend=[{"view": v, "moment": "now"} for v in VIEWS])
    record["label"]["view"] = "multi"
    record["label"]["views"] = list(VIEWS)
    record["label"]["answer_check"] = {"type": "exact", "fields": ["feasible"]}
    images = C.Images(str(root), str(tmp_path / "assets"), 1568)
    row = C.convert(record, manifest, images, "openai")
    shown = [Image.open(p).convert("RGB") for p in row["meta"]["images"]]
    assert [s.size for s in shown] == [(1568, 1568), (1568, 1175), (1568, 1175), (1280, 720)]
    s = 1568 / 1600
    assert row["meta"]["transforms"][0]["scale"] == pytest.approx(s)
    box = _red_box(shown[0].crop((700, 300, 1000, 600)))
    assert box is not None
    x0, _, x1, y1 = box
    assert abs((x0 + 700) - 800 * s) <= 5 and abs((x1 + 700) - 900 * s) <= 5
    assert abs((y1 + 300) - 500 * s) <= 5
    # the side views carry no mark (only the oak frame is scaled and copied, without red)
    small = shown[1].resize((392, 294))
    assert _red_box(small) is None
    assert _red_box(shown[3]) is None and row["meta"]["transforms"][3]["scale"] == 1.0


def test_mark_follows_the_view_roi_crop(tmp_path):
    from PIL import Image

    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (400, 400))
    manifest = {"img_a": {"path": "scan/D01/s003.png", "roi": [100, 50, 300, 350]}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [150, 150, 40, 40], "tag": "A"}])
    images = C.Images(str(root), str(tmp_path / "assets"), None)
    row = C.convert(record, manifest, images, "openai", roi="view")
    shown = Image.open(row["meta"]["images"][0]).convert("RGB")
    assert shown.size == (200, 300)
    px = shown.load()
    assert _is_mark(px[48, 120]) and not _is_mark(px[70, 120])  # left edge of [50, 100, 40, 40]


def test_marks_need_somewhere_to_go(tmp_path):
    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (64, 64))
    manifest = {"img_a": {"path": "scan/D01/s003.png"}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [10, 10, 10, 10], "tag": "A"}])
    with pytest.raises(SystemExit):
        C.convert(record, manifest, C.Images(str(root), None, None), "openai")
    # the CLI puts them beside its output when --assets is not given
    import json
    export = tmp_path / "x.jsonl"
    export.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "x.images_manifest.jsonl").write_text(
        json.dumps(dict(manifest["img_a"], image="img_a")) + "\n", encoding="utf-8")
    out = tmp_path / "runs" / "x.openai.jsonl"
    out.parent.mkdir()
    assert C.main(["--export", str(export), "--format", "openai", "--image-root", str(root),
                   "--out", str(out)]) == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert Path(row["meta"]["images"][0]).parent == out.parent / "x.openai_marked"


@pytest.mark.parametrize("task", ["V1", "V8", "V14"])
def test_box_tasks_never_show_marks(tmp_path, task):
    """A mark on V1 / V8 / V14 would be the answer: none is drawn or named, whatever the record holds."""
    from PIL import Image

    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (320, 320))
    manifest = {"img_a": {"path": "scan/D01/s003.png"}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [40, 40, 30, 20], "tag": "A"}], task=task,
                 question="Where is the RAM latch?")
    record["label"]["answer_check"] = {"type": "exact", "fields": ["bbox"]}
    images = C.Images(str(root), str(tmp_path / "assets"), None)
    row = C.convert(record, manifest, images, "openai")
    assert Path(row["meta"]["images"][0]) == root / "scan/D01/s003.png"  # the frame itself
    assert _red_box(Image.open(row["meta"]["images"][0])) is None
    text = row["messages"][1]["content"][0]["text"]
    assert "Outlined" not in text and C.marks_line(record) == ""
    assert not (tmp_path / "assets").exists() or not list((tmp_path / "assets").iterdir())


def test_mark_style_is_in_the_cache_key(tmp_path, monkeypatch):
    root = tmp_path / "images"
    _frame(root, "scan/D01/s003.png", (64, 64))
    manifest = {"img_a": {"path": "scan/D01/s003.png"}}
    record = _v2(["img_a"], [{"image": 0, "bbox": [10, 10, 10, 10], "tag": "A"}])
    images = C.Images(str(root), str(tmp_path / "assets"), None)
    first = C.convert(record, manifest, images, "openai")["meta"]["images"][0]
    assert C.convert(record, manifest, images, "openai")["meta"]["images"][0] == first
    monkeypatch.setattr(C, "MARK_STYLE_VERSION", "mark-test")
    restyled = C.convert(record, manifest, images, "openai")["meta"]["images"][0]
    assert restyled != first
    monkeypatch.setattr(C, "MARK_RGB", (0, 0, 255))
    blue = C.convert(record, manifest, images, "openai")["meta"]["images"][0]
    assert blue not in (first, restyled)
    monkeypatch.setattr(C, "MARK_OPACITY", 1.0)
    assert C.convert(record, manifest, images, "openai")["meta"]["images"][0] not in (
        first, restyled, blue)
    assert C.mark_style().startswith("mark-test|")


def test_the_mark_style_version_was_bumped():
    """mark-v2 (owner 2026-11-04): dashed, 60 % opaque, the tag outside the outline."""
    assert C.MARK_STYLE_VERSION == "mark-v2" and C.MARK_OPACITY == pytest.approx(0.6)
    assert C.mark_width(1600) == 2 and C.mark_dash(1600) == (14, 8)


# --------------------------------------------------------------------------- #
# C2 legend and glossary
# --------------------------------------------------------------------------- #
def _multi(views_before, views_after=None, task="V4") -> dict:
    legend = [{"view": v, "moment": "now" if views_after is None else "before"} for v in views_before]
    legend += [{"view": v, "moment": "after"} for v in views_after or []]
    images = [f"img_{e['moment']}_{e['view']}" for e in legend]
    record = _v2(images, [{"image": 0, "bbox": [10, 10, 5, 5], "tag": "A"}], task=task,
                 image_legend=legend,
                 question="Can the RAM module outlined in red be removed now?")
    record["label"].update(view="multi", views=[e["view"] for e in legend],
                           answer_check={"type": "exact", "fields": ["feasible"]})
    return record


def test_multiview_legend_lists_present_views():
    four = C.user_text(_multi(VIEWS))
    assert four.startswith("Images 1–4: the same moment from the top-down scanner, then three side "
                           "cameras (OAK 1, OAK 2, RealSense).")
    eight = C.user_text(_multi(VIEWS, VIEWS, task="V3"))
    assert eight.startswith("Images 1–4 before, images 5–8 after the action, each: top-down "
                            "scanner, OAK 1, OAK 2, RealSense.")
    # a view without a photo of that step is left out of the images and of the legend
    three = C.user_text(_multi(("scan", "oak1", "rs")))
    assert three.startswith("Images 1–3: the same moment from the top-down scanner, then two side "
                            "cameras (OAK 1, RealSense).")
    assert "OAK 2" not in three.split("\n")[0]
    two = C.user_text(_multi(("scan", "oak2")))
    assert two.startswith("Images 1–2: the same moment from the top-down scanner, then one side "
                          "camera (OAK 2).")
    assert C.user_text(_multi(("scan",))).startswith("Image 1: the top-down scanner.")
    six = C.user_text(_multi(("scan", "oak1", "rs"), ("scan", "oak1", "rs"), task="V12"))
    assert six.startswith("Images 1–3 before, images 4–6 after the action, each: top-down "
                          "scanner, OAK 1, RealSense.")
    uneven = C.user_text(_multi(("scan", "rs"), ("scan", "oak1", "rs"), task="V12"))
    assert uneven.startswith("Images 1–2 before the action (top-down scanner, RealSense), "
                             "images 3–5 after it (top-down scanner, OAK 1, RealSense).")
    # the marked part's tag is named, the question follows, the answer format after it
    assert "Outlined in red: A (image 1)." in four
    assert four.index("Outlined in red") < four.index("Can the RAM module") \
        < four.index("Answer with JSON only")
    # without image_legend the views come from label.views (moments from label.image_steps)
    record = _multi(VIEWS, VIEWS, task="V3")
    del record["prompt"]["image_legend"]
    record["label"]["image_steps"] = [4] * 4 + [5] * 4
    assert C.user_text(record).startswith("Images 1–4 before, images 5–8 after the action")
    # single-view records have no legend
    single = C.user_text(_v2(["img_a"], [{"image": 0, "bbox": [1, 1, 2, 2], "tag": "A"}]))
    assert not single.startswith("Image") and single.startswith("Outlined in red: A.")
    # a group: several outlines under one tag; two parts: two tags
    group = _v2(["img_a"], [{"image": 0, "bbox": [1, 1, 2, 2], "tag": "A"},
                            {"image": 0, "bbox": [9, 9, 2, 2], "tag": "A"},
                            {"image": 0, "bbox": [20, 20, 2, 2], "tag": "B"}])
    text = C.user_text(group)
    assert "Outlined in red: A (2 outlines: one group of interchangeable parts), B." in text
    # a text-only variant (state oracle, blind) has no images: no legend, no marks line
    blind = _multi(VIEWS)
    blind["prompt"]["images"] = []
    assert "Image" not in C.user_text(blind).split("\n")[0] and "Outlined" not in C.user_text(blind)
    assert C.CHAT_FORMAT_ID == "chat-v3"


def test_glossary_has_class_and_states():
    import vlm_glossary as G

    text = G.glossary_text(["ram_module", "cable_clip"], ["cable_clip"])
    lines = text.split("\n")
    assert lines[0] == "Glossary:"
    assert lines[1] == "- RAM module: A memory (RAM) stick."
    assert lines[2].startswith("- cable clip: A clip holding a cable. States: ")
    for state in ("closed", "open", "removed"):
        assert f"{state} = {G.STATE_MEANINGS['cable_clip'][state]}" in lines[2]
    assert "States:" not in lines[1]  # not a state question about RAM modules
    # a refined match class is glossed by its class; unknown / empty -> nothing
    assert G.glossary_text(["screw.cpu_cooler"]) == "Glossary:\n- screw: " + G.class_text("screw")
    assert G.glossary_text([], []) == "" and G.glossary_text(["no_such_class"]) == ""
    # a state class not among the terms is glossed too
    assert "- RAM latch: " in G.glossary_text(["ram_module"], ["ram_latch"])
    # no annotation jargon reaches the model
    every = G.glossary_text(list(G.CLASS_TEXT), list(G.STATE_MEANINGS))
    assert "attrs." not in every
    # the prompt ends with the glossary, after the answer format
    record = _v2(["img_a"], [{"image": 0, "bbox": [1, 1, 2, 2], "tag": "A"}],
                 terms=["ram_latch"], state_terms=["ram_latch"])
    prompt = C.user_text(record)
    assert prompt.endswith(G.glossary_text(["ram_latch"], ["ram_latch"]))
    assert prompt.index("Answer with JSON only") < prompt.index("Glossary:")
    assert "Glossary:" not in C.user_text(_v2(["img_a"], []))


def test_a_presence_question_glosses_presence_states():
    """Final review M3: V14 / V2 / V15 ask a screw present / removed (its tightness was
    never annotated), so the glossary gives those two states, not fastened / loosened."""
    import vlm_glossary as G

    record = _v2(["img_a"], [], terms=["screw.cpu_cooler"], state_terms=["screw.cpu_cooler"],
                 task="V14", question="Can this view settle whether the CPU cooler screw is "
                                       "still in place?")
    record["label"]["answer_check"] = {"type": "exact", "state_space": "presence",
                                       "fields": ["answerable", "state"]}
    assert G.is_presence(record)
    text = C.user_text(record)
    gloss = text[text.index("Glossary:"):]
    assert "present = " in gloss and "removed = " in gloss
    for word in ("fastened", "loosened"):
        assert f"{word} =" not in gloss
    # a state question on every other record keeps the tracked states
    plain = G.glossary_text(["screw"], ["screw"])
    assert "fastened = " in plain and "present = " not in plain
    latch = G.glossary_text(["ram_latch"], ["ram_latch"])
    assert G.glossary_text(["ram_latch"], ["ram_latch"], presence=True) == latch


def test_glossary_tables_follow_the_scanner_pack_and_the_taxonomy():
    """``vlm_glossary`` is self-contained (it ships without ``tda`` and the taxonomy); these keep it honest."""
    import yaml

    import export_scanpack
    import vlm_glossary as G

    assert G.CLASS_TEXT == dict(export_scanpack.CLASS_EN)
    tax = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "taxonomy.yaml")
                         .read_text(encoding="utf-8"))
    multi = {c: list(d["states"]) for c, d in tax["classes"].items()
             if len(d.get("states") or []) >= 2}
    assert {c: list(m) for c, m in G.STATE_MEANINGS.items()} == multi
    assert all(m.strip() for table in G.STATE_MEANINGS.values() for m in table.values())
    assert set(G.MODEL_TEXT) >= {c for c, t in G.CLASS_TEXT.items() if "attrs." in t}
    assert set(G.CLASS_NAME) == set(G.CLASS_TEXT)
    # the glossary module imports nothing of the repository nor PyYAML
    src = (SCRIPTS / "vlm_glossary.py").read_text(encoding="utf-8")
    assert "import tda" not in src and "from tda" not in src and "import yaml" not in src
