"""Task W3k: a sequence part of an excluded class opted into the components gallery (``attrs.in_gallery``).

D02's motherboard screw 6 is also the drive latch's stop screw; the user drew it in the gallery shot (2026-11-01:
「我在D02的all component的主视角里面加入了这个螺丝 … 他实际上应该对应我说的那个motherboard screw 06」). Screws are
:data:`GALLERY_EXCLUDED_CLASSES`, so merging the gallery drawing into ``screw.motherboard.06`` made the gallery drop it
("the verified frame lost screw.motherboard.06"). ``attrs.in_gallery: true`` on that one instance makes
:func:`is_gallery_part` judge it like any other removed part; nothing else about it changes.
"""
from __future__ import annotations

from pathlib import Path

from tda.core.model import (
    GALLERY_EXCLUDED_CLASSES,
    GALLERY_ONLY_ATTR,
    IN_GALLERY_ATTR,
    InstanceRec,
    StateEvent,
    is_gallery_only,
    is_in_gallery,
)
from tda.core.states import gallery_state, is_gallery_part
from tda.core.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "superpowers" / "specs" / "2026-09-17-teardown-annotator-design.md"
HANDOFF = ROOT / "docs" / "HANDOFF_2026-10-28.md"


def _rec(key: str, cls: str, **kw) -> InstanceRec:
    attrs = kw.pop("attrs", {})
    return InstanceRec(key=key, desktop=1, cls=cls, attrs=dict(attrs), **kw)


def _removed(target: str, step: int, held: str = "installed") -> list[StateEvent]:
    return [StateEvent(1, step, target, "state", held, "removed"),
            StateEvent(1, step, target, "placement", "in_chassis", "on_bench")]


def _table(screw_attrs: dict) -> tuple[dict, list]:
    instances = {
        "chassis": _rec("chassis", "chassis"),
        "motherboard.01": _rec("motherboard.01", "motherboard"),
        "psu.01": _rec("psu.01", "psu"),
        "screw.motherboard.06": _rec("screw.motherboard.06", "screw",
                                     attrs={"role": "motherboard", **screw_attrs}, fastens="motherboard.01"),
        "screw.motherboard.05": _rec("screw.motherboard.05", "screw",
                                     attrs={"role": "motherboard"}, fastens="motherboard.01"),
        "connector.01": _rec("connector.01", "connector", attrs=dict(screw_attrs), socket_host="motherboard.01"),
    }
    events = (_removed("psu.01", 3) + _removed("motherboard.01", 9)
              + _removed("screw.motherboard.05", 4, "fastened") + _removed("screw.motherboard.06", 5, "fastened")
              + [StateEvent(1, 2, "connector.01", "state", "plugged", "unplugged")]
              + _removed("connector.01", 6, "unplugged"))
    return instances, events


def test_the_flag_reads_only_on_a_real_sequence_part():
    assert is_in_gallery(_rec("screw.motherboard.06", "screw", attrs={IN_GALLERY_ATTR: True}))
    assert not is_in_gallery(_rec("screw.motherboard.06", "screw"))
    assert not is_in_gallery(_rec("screw.motherboard.06", "screw", attrs={IN_GALLERY_ATTR: False}))
    assert not is_in_gallery(None)
    # a gallery-only part is already on the table; a context part is not a part of the machine
    only = _rec("screw.motherboard.07", "screw", attrs={IN_GALLERY_ATTR: True, GALLERY_ONLY_ATTR: True, "added_by": "x"})
    assert is_gallery_only(only) and not is_in_gallery(only)
    assert not is_in_gallery(_rec("ctx.screw.01", "screw", attrs={IN_GALLERY_ATTR: True}))


def test_an_excluded_class_stays_out_of_the_gallery_unless_opted_in():
    tax = load_taxonomy()
    instances, events = _table({})
    parts = sorted(k for k, v in gallery_state(instances, events, tax, GALLERY_EXCLUDED_CLASSES).items() if v.gallery)
    assert parts == ["motherboard.01", "psu.01"]
    instances, events = _table({IN_GALLERY_ATTR: True})
    parts = sorted(k for k, v in gallery_state(instances, events, tax, GALLERY_EXCLUDED_CLASSES).items() if v.gallery)
    # the opted-in screw and connector join; screw 05, not opted in, still does not
    assert parts == ["connector.01", "motherboard.01", "psu.01", "screw.motherboard.06"]


def test_an_opted_in_part_still_has_to_leave_on_its_own():
    tax = load_taxonomy()
    instances, events = _table({IN_GALLERY_ATTR: True})
    # never taken out: not on the table
    events = [e for e in events if e.target != "screw.motherboard.06"]
    fs = gallery_state(instances, events, tax, GALLERY_EXCLUDED_CLASSES)
    assert not fs["screw.motherboard.06"].gallery
    assert not is_gallery_part(instances, fs, "screw.motherboard.06", GALLERY_EXCLUDED_CLASSES)
    # a Label Studio draft never is
    instances["ls:Screw#6"] = _rec("ls:Screw#6", "screw", attrs={IN_GALLERY_ATTR: True})
    fs = gallery_state(instances, _table({IN_GALLERY_ATTR: True})[1], tax, GALLERY_EXCLUDED_CLASSES)
    assert not is_gallery_part(instances, fs, "ls:Screw#6", GALLERY_EXCLUDED_CLASSES)


def test_the_docs_say_it():
    spec = SPEC.read_text(encoding="utf-8")
    gallery = spec.split("### 3.2b 零件合照", 1)[1].split("### 3.3", 1)[0]
    for needed in ("attrs.in_gallery", "W3k", "GALLERY_EXCLUDED_CLASSES", "screw.motherboard.06"):
        assert needed in gallery, needed
    entries = [line for line in spec.split("## 14. 修订记录", 1)[1].splitlines() if line.startswith("- ")]
    # on top when it landed; whatever lands later goes above it
    [entry] = [e for e in entries[:2] if e.startswith("- **v1.56（2026-11-01）**")]
    for needed in ("（W3k，", "attrs.in_gallery", "`is_gallery_part`", "GALLERY_EXCLUDED_CLASSES",
                   "motherboard screw 06", "tests/test_w3k_in_gallery.py"):
        assert needed in entry, needed
    assert spec.index("- **v1.56") < spec.index("- **v1.55") < spec.index("- **v1.54")
    handoff = HANDOFF.read_text(encoding="utf-8")
    export = handoff.split("## 5. 导出与 HuggingFace 准备", 1)[1].split("\n## ", 1)[0]
    [line] = [l for l in export.splitlines() if "W3k" in l]
    for needed in ("attrs.in_gallery", "v1.56", "screw.motherboard.06"):
        assert needed in line, needed
