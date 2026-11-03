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
    return abs(px[0] - RED[0]) < 40 and px[1] < 80 and abs(px[2] - RED[2]) < 50


def _red_box(img):
    """The bounding box of the red pixels: ``(x0, y0, x1, y1)`` inclusive, or ``None``."""
    w, h = img.size
    px = img.convert("RGB").load()
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if _is_red(px[x, y]):
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
    # the outline runs just outside the scaled box: the part itself is not painted over
    for x, y in ((48, 62), (77, 62), (62, 77)):
        assert _is_red(px[x, y]), (x, y, px[x, y])
    assert not _is_red(px[62, 62])  # inside the box
    assert not _is_red(px[400, 400]) and not _is_red(px[62, 120])  # elsewhere
    x0, y0, x1, y1 = _red_box(shown.crop((0, 0, 200, 200)))
    # the outline (3 px) and, above it, the filled tag label at the box's top-left
    assert 44 <= x0 <= 50 and 75 <= x1 <= 82 and y1 <= 82 and y0 < 47


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
    assert _is_red(px[48, 120]) and not _is_red(px[70, 120])  # left edge of [50, 100, 40, 40]


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
