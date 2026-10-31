"""Turn an export-vlm JSONL (or one of its reference variants) into chat requests (W3b).

``python scripts/vlm_to_chat.py --export <vlm_scan.jsonl> [--records <variant.jsonl>]
--format openai|qwen|internvl --image-root <cache dir> --out <chat.jsonl>
[--assets <dir>] [--base64] [--roi none|view|evidence] [--composite]``

* ``openai`` -- ``{"id", "messages": [system, user]}``; the user turn is a list
  of ``text`` and ``image_url`` parts, the URL a local path or (``--base64``) a
  ``data:`` URL;
* ``qwen`` -- Qwen-VL chat messages, ``{"type": "image", "image": "file://..."}``
  parts before the text;
* ``internvl`` -- InternVL chat JSONL, ``{"id", "image", "conversations"}`` with
  ``<image>`` placeholders (``Image-1: <image>`` for several).

The question is followed by the lettered options (V6) and by the answer format
of :data:`tda.core.export.vlm_check.RESPONSE_FORMATS`. ``--roi view`` crops
each image to its pose segment's ROI (from the manifest), ``--roi evidence`` to
the union of the record's evidence boxes with a margin (falling back to the
view ROI); the crop offset and any ``--max-side`` scale go to
``meta.transforms`` so ``scripts/vlm_score.py --chat`` can map boxes back.
A two-image question is two image parts, or with ``--composite`` one
side-by-side picture -- the choice is the run's, fixed per model.
``meta.prompt_template_id`` (chat format + question template) lets the scorer
report the template variance. A ``state_oracle`` variant puts its state tables
in the text; a ``blind`` or ``protocol_prior`` one has no image at all.

Crops and composites are written under ``--assets`` (required for them); the
source images are only read.
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
from tda.core.export.vlm_check import LETTERS, format_instruction  # noqa: E402

#: Bumped whenever the wording around the question changes.
CHAT_FORMAT_ID = "chat-v1"
SYSTEM = ("You are shown photographs of a desktop PC being disassembled on a bench. "
          "Answer exactly in the JSON format asked for.")
#: Margin around the evidence boxes for ``--roi evidence``.
EVIDENCE_MARGIN = 0.25
EVIDENCE_MIN_SIDE = 96


def user_text(record: dict) -> str:
    """Question, lettered options, state tables (oracle variant), answer format."""
    prompt = record["prompt"]
    parts = []
    tables = prompt.get("state_tables")
    if tables:
        for i, table in enumerate(tables, start=1):
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
    return "\n\n".join(parts)


def _evidence_box(record: dict) -> Optional[tuple[float, float, float, float]]:
    boxes = [b for b in (record["label"].get("evidence") or {}).get("bboxes", {}).values()
             if b]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    mx = max(w * EVIDENCE_MARGIN, (EVIDENCE_MIN_SIDE - w) / 2, 0)
    my = max(h * EVIDENCE_MARGIN, (EVIDENCE_MIN_SIDE - h) / 2, 0)
    return x0 - mx, y0 - my, x1 + mx, y1 + my


class Images:
    """Reads, crops, scales and composites frames; writes what it made to ``assets``."""

    def __init__(self, root: Optional[str], assets: Optional[str], max_side: Optional[int]):
        self.root = Path(root) if root else None
        self.assets = Path(assets) if assets else None
        self.max_side = max_side

    def source(self, rel: str) -> Path:
        return (self.root / rel) if self.root else Path(rel)

    def _need_assets(self) -> Path:
        if self.assets is None:
            raise SystemExit("--assets is required for --roi, --max-side and --composite")
        self.assets.mkdir(parents=True, exist_ok=True)
        return self.assets

    def prepare(self, rel: str, box) -> tuple[Path, dict]:
        """The file to show, and ``{"crop": [x0, y0], "scale": s}``."""
        if box is None and not self.max_side:
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
    """One chat request for one record."""
    shown: list[Path] = []
    transforms: list[dict] = []
    for image_id in record["prompt"].get("images") or []:
        entry = manifest[image_id]
        box = None
        if roi == "evidence":
            box = _evidence_box(record)
        if box is None and roi in ("view", "evidence") and entry.get("roi"):
            box = tuple(entry["roi"])
        path, transform = images.prepare(entry["path"], box)
        shown.append(path)
        transforms.append(transform)
    if composite and len(shown) > 1:
        shown = [images.composite(shown)]
        transforms = []  # no box task asks about two images
    text = user_text(record)
    if composite and len(record["prompt"].get("images") or []) > 1:
        text = "The picture shows the first and the second image side by side, " \
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
    ap.add_argument("--roi", default="none", choices=("none", "view", "evidence"))
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
    images = Images(args.image_root, args.assets, args.max_side)
    rows = (convert(r, manifest, images, args.format, roi=args.roi,
                    composite=args.composite, inline=args.base64)
            for r in records if keep is None or r["id"] in keep)
    n = IO.write_jsonl(args.out, rows)
    print(f"[vlm-to-chat] {n} {args.format} requests -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
