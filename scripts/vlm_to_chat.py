"""Turn an export-vlm JSONL (or one of its reference variants) into chat requests (W3b).

``python scripts/vlm_to_chat.py --export <vlm_scan.jsonl> [--records <variant.jsonl>]
--format openai|qwen|internvl --image-root <cache dir> --out <chat.jsonl>
[--assets <dir>] [--base64] [--roi none|view] [--composite]``

* ``openai`` -- ``{"id", "messages": [system, user]}``; the user turn is a list
  of ``text`` and ``image_url`` parts, the URL a local path or (``--base64``) a
  ``data:`` URL;
* ``qwen`` -- Qwen-VL chat messages, ``{"type": "image", "image": "file://..."}``
  parts before the text;
* ``internvl`` -- InternVL chat JSONL, ``{"id", "image", "conversations"}`` with
  ``<image>`` placeholders (``Image-1: <image>`` for several).

The question is followed by the lettered options (V6) and by the answer format
of :data:`tda.core.export.vlm_check.RESPONSE_FORMATS`. Since ``chat-v2`` a
multi-view record (``prompt.image_legend`` / ``label.views``) starts with one
line naming its images ("Images 1–4: the same moment from the top-down
scanner, then three side cameras (OAK 1, OAK 2, RealSense)."), a record with
marks names their tags ("Outlined in red: A."), and the prompt ends with the
glossary of ``prompt.terms`` / ``prompt.state_terms`` (``vlm_glossary.py``). ``--roi view`` crops
each image to its pose segment's fixed chassis ROI (from the manifest) -- the
**same crop for every record of a desktop and view**; ``--roi none`` (the
default) sends the whole frame. The crop offset and any ``--max-side`` scale
go to ``meta.transforms`` so ``scripts/vlm_score.py --chat`` can map boxes back.

There is deliberately no crop around a record's own evidence (W3b review S2):
the evidence boxes are label side, and a crop built from them is the answer
leaking into the picture -- every decidable V14 item has a box and no
undecidable one does, V12's changed pairs have boxes and its no-change pairs
none, and V8's box *is* its answer. A crop may depend on the desktop and the
view, never on the record.
A two-image question is two image parts, or with ``--composite`` one
side-by-side picture -- the choice is the run's, fixed per model.
``meta.prompt_template_id`` (chat format + question template) lets the scorer
report the template variance. A ``state_oracle`` variant puts its state tables
in the text; a ``blind`` or ``protocol_prior`` one has no image at all.

Crops and composites are written under ``--assets`` (required for them); the
source images are only read.

**Marks** (spec 3.5): every ``prompt.marks`` box ``{"image", "bbox", "tag"}`` is
drawn on its image -- red ``#ff1744``, ``max(3, round(3 * w / 1600))`` px just
outside the box, the tag in a filled label at its top-left -- on an in-memory
copy, after the same crop and scale as the picture (the pack's frame is never
drawn on). The marked copy goes to ``--assets``, or without it to
``<out dir>/<out stem>_marked/``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vlm_io as IO  # noqa: E402
from vlm_glossary import glossary_text, is_presence  # noqa: E402
from tda.core.export.vlm_check import LETTERS, format_instruction  # noqa: E402

#: Bumped whenever the wording around the question changes (chat-v2: marks, image legend,
#: glossary; chat-v3: the V6 multiple choice speaks of its one picture).
CHAT_FORMAT_ID = "chat-v3"
SYSTEM = ("You are shown photographs of a desktop PC being disassembled on a bench. "
          "Answer exactly in the JSON format asked for.")
#: The crops a run may use: none, or the view's fixed chassis ROI.
ROI_MODES = ("none", "view")
#: How the legend names a camera (``prompt.image_legend[*].view`` / ``label.views``).
VIEW_NAMES = {"scan": "top-down scanner", "oak1": "OAK 1", "oak2": "OAK 2", "rs": "RealSense"}
_COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_DASH = "–"


def image_legend(record: dict) -> list[dict]:
    """``[{"view", "moment"}]`` per image of a multi-view record, else ``[]``.

    ``prompt.image_legend`` when the record has one; otherwise, for ``label.view == "multi"``,
    rebuilt from ``label.views`` (moments from ``label.image_steps``: two steps = before /
    after, one = now).
    """
    prompt, label = record["prompt"], record.get("label") or {}
    n = len(prompt.get("images") or [])
    legend = prompt.get("image_legend")
    if legend:
        return list(legend)[:n] if n else []
    views = label.get("views") or []
    if label.get("view") != "multi" or not views or len(views) != n:
        return []
    steps = label.get("image_steps") or []
    if len(steps) == n and len(set(steps)) == 2:
        first = min(steps)
        return [{"view": v, "moment": "before" if s == first else "after"}
                for v, s in zip(views, steps)]
    return [{"view": v, "moment": "now"} for v in views]


def _span(first: int, last: int, word: str = "Images") -> str:
    if first == last:
        return f"{word[:-1]} {first}"
    return f"{word} {first}{_DASH}{last}"


def _names(views) -> str:
    return ", ".join(VIEW_NAMES.get(v, str(v)) for v in views)


def legend_line(record: dict) -> str:
    """The line naming the images of a multi-view record (spec 3.5), or ``""``.

    "Images 1–4: the same moment from the top-down scanner, then three side cameras (OAK 1,
    OAK 2, RealSense)." / "Images 1–4 before, images 5–8 after the action, each: top-down
    scanner, OAK 1, OAK 2, RealSense." -- only the views present are named.
    """
    legend = image_legend(record)
    if not legend:
        return ""
    views = [e.get("view") for e in legend]
    before = [e.get("view") for e in legend if e.get("moment") == "before"]
    after = [e.get("view") for e in legend if e.get("moment") == "after"]
    if before and after and len(before) + len(after) == len(legend):
        nb, n = len(before), len(legend)
        head = _span(1, nb)
        tail = _span(nb + 1, n, "images")
        if before == after:
            each = "each: " + _names(before) if nb > 1 else "from the " + _names(before)
            return f"{head} before, {tail} after the action, {each}."
        return (f"{head} before the action ({_names(before)}), "
                f"{tail} after it ({_names(after)}).")
    if len(views) == 1:
        return f"Image 1: the {_names(views)}."
    if views[0] == "scan":
        side = views[1:]
        k = len(side)
        cams = "side camera" if k == 1 else "side cameras"
        return (f"{_span(1, len(views))}: the same moment from the top-down scanner, then "
                f"{_COUNT.get(k, str(k))} {cams} ({_names(side)}).")
    return (f"{_span(1, len(views))}: the same moment from {_COUNT.get(len(views), len(views))} "
            f"cameras ({_names(views)}).")


#: Tasks whose answer is a box or a visibility: a mark there would give the answer away
#: (spec 3.2), so marks are never drawn nor named on them, whatever the record holds.
UNMARKED_TASKS = ("V1", "V8", "V14")


def shown_marks(record: dict) -> list[dict]:
    """The ``prompt.marks`` the model may see: none on V1 / V8 / V14."""
    prompt = record["prompt"]
    if prompt.get("task") in UNMARKED_TASKS:
        return []
    return list(prompt.get("marks") or [])


def marks_line(record: dict) -> str:
    """"Outlined in red: A (image 1), B." -- the tags of ``prompt.marks``, or ``""``.

    One entry per tag in order of first appearance; a tag on several outlines is a group of
    interchangeable parts; the image is named when the record has more than one.
    """
    prompt = record["prompt"]
    marks = shown_marks(record)
    n_images = len(prompt.get("images") or [])
    if not marks or not n_images:
        return ""
    order: list[str] = []
    boxes: dict[str, int] = {}
    where: dict[str, list[int]] = {}
    for m in marks:
        tag = str(m.get("tag") or "")
        if tag not in boxes:
            order.append(tag)
            boxes[tag], where[tag] = 0, []
        boxes[tag] += 1
        index = int(m.get("image", 0)) + 1
        if index not in where[tag]:
            where[tag].append(index)
    items = []
    for tag in order:
        extra = []
        if boxes[tag] > 1:
            extra.append(f"{boxes[tag]} outlines: one group of interchangeable parts")
        if n_images > 1:
            imgs = sorted(where[tag])
            extra.append(("image " if len(imgs) == 1 else "images ")
                         + ", ".join(str(i) for i in imgs))
        items.append(tag + (f" ({'; '.join(extra)})" if extra else ""))
    return "Outlined in red: " + ", ".join(items) + "."


def user_text(record: dict) -> str:
    """Image legend, marked tags, state tables (oracle variant), question, lettered options,
    answer format, glossary."""
    prompt = record["prompt"]
    parts = []
    if prompt.get("images"):  # a text-only variant (state oracle, blind) shows no image
        for line in (legend_line(record), marks_line(record)):
            if line:
                parts.append(line)
    tables = prompt.get("state_tables")
    if tables:
        moments = len(tables) == 2 and (record.get("label") or {}).get("view") == "multi"
        for i, table in enumerate(tables, start=1):
            if moments:
                head = "State of the machine " + ("before the action" if i == 1
                                                  else "after the action")
            else:
                head = f"State of the machine in image {i}" if len(tables) > 1 else \
                    "State of the machine"
            parts.append(f"{head} (instead of the photo):\n{table}")
    if prompt.get("previous_action"):
        pa = prompt["previous_action"]
        parts.append(f"The previous action was: {pa.get('verb')} "
                     f"{str(pa.get('match_class') or pa.get('class')).replace('_', ' ')}.")
    parts.append(prompt["question"])
    options = prompt.get("options")
    if options:
        parts.append("\n".join(f"{LETTERS[i]}. {o['verb']} {o['target_label']}"
                               for i, o in enumerate(options)))
    parts.append(format_instruction(record))
    glossary = glossary_text(prompt.get("terms"), prompt.get("state_terms"),
                             presence=is_presence(record))
    if glossary:
        parts.append(glossary)
    return "\n\n".join(parts)


#: The colour of a mark (``prompt.marks``) on the image a model is shown.
MARK_RGB = (0xFF, 0x17, 0x44)  # "#ff1744"
#: The text on a mark's tag label.
MARK_TEXT_RGB = (255, 255, 255)
#: Bumped whenever the way a mark is drawn changes (outline, label, font): it is part of the
#: cache key of a marked copy, so a new style never reuses a stale picture.
MARK_STYLE_VERSION = "mark-v1"


def mark_style() -> str:
    """Everything about a mark's look that its cached picture depends on."""
    return (f"{MARK_STYLE_VERSION}|{MARK_RGB}|{MARK_TEXT_RGB}|"
            f"w=max(3,round(3*W/1600))|{CHAT_FORMAT_ID}")


def mark_width(width: int) -> int:
    """Outline width of a mark on an image ``width`` px wide (3 px at 1600 px, never thinner)."""
    return max(3, round(3 * width / 1600))


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: the fixed bitmap font
        return ImageFont.load_default()


def draw_marks(img, marks, crop=(0, 0), scale: float = 1.0):
    """Draw ``marks`` (native ``[x, y, w, h]`` boxes) on ``img`` -- already cropped by ``crop``
    and scaled by ``scale``, so each box gets the same transform as the picture. ``img`` is
    changed in place (the caller passes its in-memory copy, never the pack's file).

    The outline runs just **outside** the box (the part itself is not painted over); the tag
    sits in a filled label at the box's top-left, above the box where there is room.
    """
    from PIL import ImageDraw

    if not marks:
        return img
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    lw = mark_width(W)
    font = _font(max(14, round(24 * W / 1600)))
    draw = ImageDraw.Draw(img)
    for m in marks:
        x, y, w, h = (float(v) for v in m["bbox"])
        x0, y0 = (x - crop[0]) * scale, (y - crop[1]) * scale
        x1, y1 = (x + w - crop[0]) * scale, (y + h - crop[1]) * scale
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue  # outside the crop: nothing of it is shown
        ox0, oy0 = round(x0) - lw, round(y0) - lw
        ox1, oy1 = round(x1) + lw - 1, round(y1) + lw - 1
        draw.rectangle((max(0, ox0), max(0, oy0), min(W - 1, ox1), min(H - 1, oy1)),
                       outline=MARK_RGB, width=lw)
        tag = str(m.get("tag") or "")
        if not tag:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        pad = max(2, lw)
        tw, th = right - left + 2 * pad, bottom - top + 2 * pad
        lx = min(max(0, ox0), max(0, W - tw))
        ly = oy0 - th if oy0 - th >= 0 else max(0, oy0)
        draw.rectangle((lx, ly, lx + tw - 1, ly + th - 1), fill=MARK_RGB)
        draw.text((lx + pad - left, ly + pad - top), tag, fill=MARK_TEXT_RGB, font=font)
    return img


class Images:
    """Reads, crops, scales, marks and composites frames; writes what it made to ``assets``.

    ``marked_dir`` is where marked copies go when no ``assets`` directory was given (the CLI
    puts them beside its output); the pack's frames are only ever read.
    """

    def __init__(self, root: Optional[str], assets: Optional[str], max_side: Optional[int],
                 marked_dir: Optional[str] = None):
        self.root = Path(root) if root else None
        self.assets = Path(assets) if assets else None
        self.max_side = max_side
        self.marked_dir = Path(marked_dir) if marked_dir else None

    def source(self, rel: str) -> Path:
        return (self.root / rel) if self.root else Path(rel)

    def _need_assets(self) -> Path:
        if self.assets is None:
            raise SystemExit("--assets is required for --roi, --max-side and --composite")
        self.assets.mkdir(parents=True, exist_ok=True)
        return self.assets

    def _marks_out(self) -> Path:
        out = self.assets or self.marked_dir
        if out is None:
            raise SystemExit("a record with prompt.marks needs --assets (where its marked copy "
                             "goes; the pack's frame is never drawn on)")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def prepare(self, rel: str, box, marks=None) -> tuple[Path, dict]:
        """The file to show, and ``{"crop": [x0, y0], "scale": s}``.

        ``marks`` (``[{"bbox", "tag"}]`` of this image) are drawn on the in-memory copy after
        the crop and the scale, with the same offset and factor as the picture.
        """
        if box is None and not self.max_side and not marks:
            return self.source(rel), {"crop": [0, 0], "scale": 1.0}
        from PIL import Image

        img = Image.open(self.source(rel))
        img.load()
        x0 = y0 = 0
        if box is not None:
            W, H = img.size
            x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
            x1, y1 = min(W, int(round(box[2]))), min(H, int(round(box[3])))
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))
            else:
                x0 = y0 = 0
        scale = 1.0
        if self.max_side and max(img.size) > self.max_side:
            scale = self.max_side / max(img.size)
            img = img.resize((max(1, round(img.size[0] * scale)),
                              max(1, round(img.size[1] * scale))))
        if marks:
            img = draw_marks(img.copy(), marks, (x0, y0), scale)
            spec = [(m.get("tag"), [float(v) for v in m["bbox"]]) for m in marks]
            tag = hashlib.sha256(f"{rel}|{box}|{self.max_side}|{spec}|{mark_style()}"
                                 .encode()).hexdigest()[:16]
            out = self._marks_out() / f"mark_{tag}.png"
        else:
            tag = hashlib.sha256(f"{rel}|{box}|{self.max_side}".encode()).hexdigest()[:16]
            out = self._need_assets() / f"crop_{tag}.png"
        if not out.exists():
            img.save(out)
        return out, {"crop": [x0, y0], "scale": scale}

    def composite(self, paths: list[Path]) -> Path:
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        h = max(i.size[1] for i in imgs)
        imgs = [i if i.size[1] == h else i.resize((round(i.size[0] * h / i.size[1]), h))
                for i in imgs]
        gap = 16
        canvas = Image.new("RGB", (sum(i.size[0] for i in imgs) + gap * (len(imgs) - 1), h),
                           (255, 255, 255))
        x = 0
        for i in imgs:
            canvas.paste(i, (x, 0))
            x += i.size[0] + gap
        tag = hashlib.sha256("|".join(str(p) for p in paths).encode()).hexdigest()[:16]
        out = self._need_assets() / f"pair_{tag}.png"
        if not out.exists():
            canvas.save(out)
        return out


def _url(path: Path, inline: bool) -> str:
    if not inline:
        return str(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    kind = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
    return f"data:image/{kind};base64,{data}"


def convert(record: dict, manifest: dict, images: Images, fmt: str, *,
            roi: str = "none", composite: bool = False, inline: bool = False) -> dict:
    """One chat request for one record.

    ``roi`` is ``none`` or ``view``; the crop is read off the manifest entry of
    the *image* (desktop, view, step), never off the record.
    """
    if roi not in ROI_MODES:
        raise ValueError(f"--roi {roi!r}: only {', '.join(ROI_MODES)} -- a crop built "
                         f"from a record's evidence would show its answer")
    shown: list[Path] = []
    transforms: list[dict] = []
    marks = shown_marks(record)
    for index, image_id in enumerate(record["prompt"].get("images") or []):
        entry = manifest[image_id]
        box = tuple(entry["roi"]) if roi == "view" and entry.get("roi") else None
        mine = [m for m in marks if int(m.get("image", 0)) == index]
        path, transform = images.prepare(entry["path"], box, mine)
        shown.append(path)
        transforms.append(transform)
    if composite and len(shown) > 1:
        shown = [images.composite(shown)]
        transforms = []  # no box task asks about two images
    text = user_text(record)
    n_images = len(record["prompt"].get("images") or [])
    if composite and n_images == 2:
        text = "The picture shows the first and the second image side by side, " \
               "left to right.\n\n" + text
    elif composite and n_images > 2:
        text = f"The picture shows images 1 to {n_images} side by side, " \
               "left to right.\n\n" + text
    label = record["label"]
    meta = {"task": record["prompt"]["task"], "template_id": label.get("template_id"),
            "prompt_template_id": f"{CHAT_FORMAT_ID}:{label.get('template_id')}",
            "format": fmt, "roi": roi, "composite": bool(composite and len(shown) == 1
                                                        and len(record["prompt"]
                                                                .get("images") or []) > 1),
            "images": [str(p) for p in shown], "transforms": transforms,
            "reference": label.get("reference")}
    if fmt == "openai":
        parts = [{"type": "text", "text": text}]
        parts += [{"type": "image_url", "image_url": {"url": _url(p, inline)}}
                  for p in shown]
        return {"id": record["id"], "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": parts}], "meta": meta}
    if fmt == "qwen":
        parts = [{"type": "image", "image": (_url(p, True) if inline
                                             else "file://" + p.as_posix())}
                 for p in shown]
        parts.append({"type": "text", "text": text})
        return {"id": record["id"], "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": parts}], "meta": meta}
    if fmt == "internvl":
        if len(shown) == 1:
            value = "<image>\n" + text
        else:
            value = "".join(f"Image-{i}: <image>\n" for i in range(1, len(shown) + 1)) + text
        out = {"id": record["id"], "conversations": [{"from": "human", "value": value}],
               "meta": meta}
        if shown:
            out["image"] = str(shown[0]) if len(shown) == 1 else [str(p) for p in shown]
        return out
    raise ValueError(f"unknown chat format {fmt!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--export", required=True, help="the export-vlm JSONL (its manifest)")
    ap.add_argument("--records", default=None,
                    help="convert these records instead (a reference variant, a sample)")
    ap.add_argument("--ids", default=None, help="a file of record ids to keep")
    ap.add_argument("--format", required=True, choices=("openai", "qwen", "internvl"))
    ap.add_argument("--image-root", default=None, help="the image cache directory")
    ap.add_argument("--assets", default=None, help="where crops / composites go")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base64", action="store_true", help="inline images as data URLs")
    ap.add_argument("--roi", default="none", choices=ROI_MODES,
                    help="none, or the view's fixed chassis ROI (one crop per "
                         "desktop and view; never one built from a record)")
    ap.add_argument("--composite", action="store_true",
                    help="two-image questions as one side-by-side picture")
    ap.add_argument("--max-side", type=int, default=None)
    args = ap.parse_args(argv)
    manifest = IO.load_manifest(args.export)
    records = IO.read_jsonl(args.records or args.export)
    keep = None
    if args.ids:
        keep = {line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines()
                if line.strip()}
    out = Path(args.out)
    images = Images(args.image_root, args.assets, args.max_side,
                    marked_dir=str(out.parent / f"{out.stem}_marked"))
    rows = (convert(r, manifest, images, args.format, roi=args.roi,
                    composite=args.composite, inline=args.base64)
            for r in records if keep is None or r["id"] in keep)
    n = IO.write_jsonl(args.out, rows)
    print(f"[vlm-to-chat] {n} {args.format} requests -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
