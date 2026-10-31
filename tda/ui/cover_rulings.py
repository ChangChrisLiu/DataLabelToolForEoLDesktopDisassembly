"""「谁压着谁」: which overlapping pairs of the frame on screen still want a ruling (task W3c).

The precedence graph misses "a rigid large part lies on X" edges (43 / 66
motherboard plans pass through a drive cage or an optical drive).  Pixels on
a confirmed frame say *who overlaps whom*; whether the overlap means
*blocked* is a physical judgement the user makes with the picture in front of
them, during Review (「谁压着谁需要我结合图去给出，这个可以在 review 的过程中做」).
The 「说明」 dock (:mod:`tda.ui.panels.notes_dock`) shows the pairs this module
picks and two buttons per pair; the answers are rows of ``cover_ruling``
(:mod:`tda.core.db_rulings`), which W3a's ``cli pixel-edges`` turns into
imported ``covered_by`` edges.

**The candidate rule** (:func:`pending_pairs`).  An overlapping pair
``(above, below, px)`` of the frame -- ``above`` paints ``px`` pixels of
``below``'s shape (:func:`tda.ui.occlusion_order.overlap_pairs`, the same
count the occlusion list shows, here *with* the motherboard as the lower part
and only between parts in the chassis, :func:`chassis_pairs`) -- is asked
when all of these hold:

* ``px >= MIN_PX`` (1000): small parts -- screws, latches, connectors, whose
  pixels are at noise level -- are W3a's thresholds' business, or later's;
* neither part is a Label Studio draft (``ls:``), a 框外零件, a gallery-only
  part (U45), an implied part, the chassis (it never moves: graph_covers'
  ``below_fixed``) or a key the instance table does not know;
* the sheet's 「本版不做」 classes: ``above`` is no screw, connector, cable
  clip, cable or latch / lever (:data:`SMALL_ABOVE`), ``below`` no screw,
  connector, cable clip or cable (:data:`SMALL_BELOW`);
* no active edge of any type joins the two, in either direction;
* no structural reason: one's ``parent`` (any depth of ``attached``
  parents), ``mounted_on``, ``fastens``, ``socket_host`` or ``cable`` is the
  other, or the class pair is in :data:`SUPPORT` (the part sits on the
  motherboard);
* no ruling yet on this desktop for ``(above, below)`` (the ordered pair:
  ``(B, A)`` is a question of its own).

Each candidate is tagged with the class-level recommendation of the covering
adjudication sheet (``docs/superpowers/discussions/2026-10-30-vlm-paper-scope/
06-covering-adjudication.md`` table A): 「建议收」 (:data:`RECOMMENDED`),
「存疑」 (:data:`UNCERTAIN`) or 「其他」, with the sheet's one-line reason --
a hint, never a ruling.  No Qt here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

from tda.core.graph import active_edges
from tda.core.model import (
    is_context,
    is_context_key,
    is_gallery_only,
    is_provisional,
)

__all__ = ["BLOCKS", "Candidate", "FIXED_CLASSES", "MIN_PX", "NOT_BLOCKS", "RECOMMENDED",
           "RULINGS", "RULING_MARK", "RULING_TEXT", "SMALL_ABOVE", "SMALL_BELOW", "SUPPORT",
           "TAGS", "TAG_OTHER",
           "TAG_RECOMMENDED", "TAG_UNCERTAIN", "UNCERTAIN", "candidate_text", "chassis_pairs",
           "class_key", "class_pairs", "excluded_part", "joined_by_edge", "pair_tag",
           "pending_pairs", "ruled_text", "status_text", "structural_reasons"]

#: The smallest covering asked about, in pixels of the lower part's shape.
#: Below it are screws, latches and connectors -- pixel noise for this
#: question; W3a's whitelist thresholds (``min_px`` / ``min_frac``) and later
#: work decide those.
MIN_PX = 1000

#: The two answers, as ``cover_ruling.ruling`` stores them (W3a reads them).
BLOCKS = "blocks"
NOT_BLOCKS = "not_blocks"
RULINGS = (BLOCKS, NOT_BLOCKS)
#: The buttons' words, and the marks of the 「已裁定」 list.
RULING_TEXT = {BLOCKS: "挡着", NOT_BLOCKS: "不挡"}
RULING_MARK = {BLOCKS: "✓", NOT_BLOCKS: "✗"}

TAG_RECOMMENDED = "建议收"
TAG_UNCERTAIN = "存疑"
TAG_OTHER = "其他"
#: The order candidates are listed in.
TAGS = (TAG_RECOMMENDED, TAG_UNCERTAIN, TAG_OTHER)

#: Class-level support: the part sits on the motherboard, so covering it is a
#: matter of build, not of order.  Mirrored from W3a's
#: ``tda.core.export.graph_covers.structural_reasons`` ``supported_by`` table
#: (task W3a §2.3); the controller unifies the two at merge.
SUPPORT: tuple[tuple[str, str], ...] = (
    ("cpu", "motherboard"),
    ("ram_module", "motherboard"),
    ("cpu_cooler", "motherboard"),
    ("expansion_card", "motherboard"),
)
#: Classes that never move (graph_covers' ``below_fixed``): no pair with them is asked.
FIXED_CLASSES = frozenset({"chassis"})
#: The sheet's 「本版不做」 rule (06-covering-adjudication.md; W3c round 2): small
#: or latch-type parts are never asked about as the part on top ...
SMALL_ABOVE = frozenset({"screw", "connector", "cable_clip", "cable", "ram_latch",
                         "drive_latch", "card_latch", "psu_latch", "cooler_latch",
                         "cpu_socket_lever"})
#: ... and screws, connectors and cables never as the part underneath (a lever
#: or a RAM latch underneath stays: cover -> lever, drive_cage -> ram_latch are
#: real questions).
SMALL_BELOW = frozenset({"screw", "connector", "cable_clip", "cable"})

_RIGID = "刚性大件压着；日志每台都先拆上面那件"
_HANGING = "上方悬着的大件下面内存可能拔得出（Dell MT 硬盘架 / 光驱位在内存插槽上方）"
_SMALL = "像素中位数 < 1000"


def _first(share: int) -> str:
    return f"日志里上面那件先拆的只有 {share}%"


#: 「建议收」 class pairs of the sheet's table A (rows 2-19): ``(above, below, reason)``,
#: each side a ``class`` or a ``class.kind``.  Row 1 (``expansion_card.gpu ->
#: motherboard``) is left out: a card is mounted on the board, so :data:`SUPPORT`
#: makes the pair structural and it is never asked (controller ruling, W3c round 2).
RECOMMENDED: tuple[tuple[str, str, str], ...] = (
    ("optical_drive", "motherboard", _RIGID),
    ("optical_drive", "drive_cage", _RIGID),
    ("cover", "cpu_socket_lever", _RIGID),
    ("optical_drive", "storage_drive.ssd", _RIGID),
    ("cover", "cpu", _RIGID),
    ("drive_cage", "cpu_socket_lever", _RIGID),
    ("optical_drive", "cpu_cooler.heatsink", _RIGID),
    ("drive_cage", "cpu_cooler.heatsink", _RIGID),
    ("optical_drive", "cpu_socket_lever", _RIGID),
    ("optical_drive", "cpu", _RIGID),
    ("drive_cage", "cpu", _RIGID),
    ("optical_drive", "storage_drive.hdd", _RIGID),
    ("drive_cage", "storage_drive.hdd", _RIGID),
    ("cpu_cooler.heatsink", "cooler_bracket", _RIGID),
    ("cover", "psu", _RIGID),
    ("cover", "cpu_cooler.heatsink", _RIGID),
    ("optical_drive", "optical_drive", _RIGID),
    ("optical_drive", "cover", _RIGID),
)
#: 「存疑」 class pairs of table A (rows 20-40): the physical question is open.
UNCERTAIN: tuple[tuple[str, str, str], ...] = (
    ("drive_cage", "motherboard", _first(65)),
    ("drive_cage", "ram_latch", _HANGING),
    ("drive_cage", "ram_module", _HANGING),
    ("optical_drive", "ram_module", _HANGING),
    ("psu", "cover", _first(71)),
    ("psu", "motherboard", "PSU 悬在主板一角上方，主板可能斜抽；HP SFF 10 台日志都先动 PSU"),
    ("optical_drive", "ram_latch", _HANGING),
    ("cover", "ram_latch", _HANGING),
    ("cover", "cover", _first(33)),
    ("expansion_card.gpu", "cover", _first(0)),
    ("cover", "ram_module", _HANGING),
    ("cpu_cooler.fan", "ram_latch", _HANGING),
    ("cpu_cooler.fan", "cooler_latch", _first(0)),
    ("expansion_card.gpu", "card_latch", _first(0)),
    ("optical_drive", "cpu_cooler.fan", _SMALL),
    ("cpu_cooler.fan", "ram_module", _HANGING),
    ("psu", "storage_drive.ssd", _first(0)),
    ("drive_cage", "optical_drive", _SMALL),
    ("psu", "cpu_socket_lever", _first(0)),
    ("cpu_cooler.fan", "drive_cage", _SMALL),
    ("expansion_card.gpu", "expansion_card.other", _SMALL),
)
_OTHER_REASON = "裁定单表 A 里没有建议：看图判断"
#: The structural relation fields: one of the two names the other.
_FIELDS = ("mounted_on", "fastens", "socket_host", "cable")

PairsLike = Union[Mapping[tuple[str, str], int], Iterable[tuple[str, str, int]]]


@dataclass(frozen=True)
class Candidate:
    """One overlapping pair of the frame that still wants a ruling."""

    above: str
    below: str
    above_class: str
    below_class: str
    #: Pixels of ``below``'s shape ``above`` paints on this frame.
    px: int
    #: 「建议收」 / 「存疑」 / 「其他」 (:data:`TAGS`) -- the sheet's hint.
    tag: str
    #: The sheet's one-line reason (the row's tooltip).
    reason: str
    #: How many *other* candidates share the class pair.  Only the frame on
    #: screen is known here -- the desktop's other frames are not compiled --
    #: so this counts the other pending pairs of **this frame**.
    same_pair_pending: int = 0

    @property
    def classes(self) -> tuple[str, str]:
        """``(above_class, below_class)``: what the batch line groups by."""
        return (self.above_class, self.below_class)


def class_key(rec) -> str:
    """``cpu_cooler.heatsink``, ``optical_drive``: the class refined by ``kind`` (or ``role``).

    The same spelling as ``tda.core.export.graph_covers.class_key`` (U46), so
    the sheet's class pairs read the same here; ``?`` without a record.
    """
    if rec is None:
        return "?"
    attrs = getattr(rec, "attrs", None) or {}
    refine = attrs.get("kind") or attrs.get("role")
    return f"{rec.cls}.{refine}" if refine else str(rec.cls)


def _matches(pattern: str, key: str) -> Optional[int]:
    """How exactly ``pattern`` names ``key``: 2 = the very ``class.kind``, 1 = its class, else ``None``."""
    if pattern == key:
        return 2
    if "." not in pattern and key.split(".", 1)[0] == pattern:
        return 1
    return None


def pair_tag(above_class: str, below_class: str) -> tuple[str, str]:
    """``(tag, reason)`` of a class pair: the most exact entry of the sheet wins.

    An entry's side names a ``class`` (every kind of it) or a ``class.kind``;
    between equally exact entries 「建议收」 comes before 「存疑」.  No entry: 「其他」.
    """
    best: Optional[tuple[int, int, str, str]] = None
    for rank, (tag, table) in enumerate(((TAG_RECOMMENDED, RECOMMENDED),
                                          (TAG_UNCERTAIN, UNCERTAIN))):
        for above, below, reason in table:
            a, b = _matches(above, above_class), _matches(below, below_class)
            if a is None or b is None:
                continue
            score = (a + b, -rank, tag, reason)
            if best is None or score[:2] > best[:2]:
                best = score
    if best is None:
        return TAG_OTHER, _OTHER_REASON
    return best[2], best[3]


def excluded_part(key: str, rec) -> Optional[str]:
    """Why a part takes no part in a ruling (``None``: it may).

    ``ls``: a Label Studio draft; ``context``: a 框外零件; ``unknown``: not in
    the instance table; ``gallery_only``: only the components gallery knows it
    (U45); ``implied``: the importer made it up from a rule; ``fixed``: the
    chassis.
    """
    if is_provisional(key):
        return "ls"
    if is_context_key(key):
        return "context"
    if rec is None:
        return "unknown"
    if is_context(rec):
        return "context"
    if is_gallery_only(rec):
        return "gallery_only"
    if (rec.attrs or {}).get("implied"):
        return "implied"
    if str(rec.cls) in FIXED_CLASSES:
        return "fixed"
    return None


def joined_by_edge(edges: Iterable, a: str, b: str) -> bool:
    """Does an active edge of any type join ``a`` and ``b``, either way round?"""
    pair = {a, b}
    return any({e.target, e.blocker} == pair for e in active_edges(list(edges)))


def _ancestors(instances: Mapping, rec) -> set[str]:
    """``rec``'s parent, and every parent up the chain while each link is ``attached``."""
    found: set[str] = set()
    if rec is None or not rec.parent:
        return found
    found.add(rec.parent)
    cur = rec
    while cur.attached:
        nxt = instances.get(cur.parent)
        if nxt is None or not nxt.parent or not nxt.attached or nxt.parent in found:
            break
        found.add(nxt.parent)
        cur = nxt
    return found


def structural_reasons(instances: Mapping, above: str, below: str) -> list[str]:
    """Why the pair needs no ruling, from the instance table (sorted; empty: none).

    ``above.parent`` / ``below.parent`` (the other is its parent, or a parent
    further up an ``attached`` chain), ``above.<field>`` / ``below.<field>``
    for ``mounted_on`` / ``fastens`` / ``socket_host`` / ``cable`` naming the
    other, and ``supported_by`` for a class pair in :data:`SUPPORT`.
    """
    out: set[str] = set()
    for name, one, other in (("above", above, below), ("below", below, above)):
        rec = instances.get(one)
        if rec is None:
            continue
        if other in _ancestors(instances, rec):
            out.add(f"{name}.parent")
        for field_name in _FIELDS:
            if getattr(rec, field_name, None) == other:
                out.add(f"{name}.{field_name}")
    up, low = instances.get(above), instances.get(below)
    if up is not None and low is not None and (str(up.cls), str(low.cls)) in SUPPORT:
        out.add("supported_by")
    return sorted(out)


def _triples(frame_pairs: PairsLike) -> list[tuple[str, str, int]]:
    if isinstance(frame_pairs, Mapping):
        return [(str(a), str(b), int(px)) for (a, b), px in frame_pairs.items()]
    return [(str(a), str(b), int(px)) for a, b, px in frame_pairs]


def chassis_pairs(compiled, pairs: Mapping[tuple[str, str], int]) -> list[tuple[str, str, int]]:
    """The overlapping pairs of a compiled frame between two parts **in the chassis**.

    ``pairs`` is :func:`tda.ui.occlusion_order.overlap_pairs` of ``compiled``;
    a part lying on another on the bench says nothing about the order they
    came out in (U46's covers read the chassis only, too).
    """
    inside = {name for name, inst in (getattr(compiled, "instances", None) or {}).items()
              if str(getattr(inst, "placement", "")) == "in_chassis"}
    return [(a, b, int(px)) for (a, b), px in sorted(pairs.items())
            if a in inside and b in inside]


def pending_pairs(frame_pairs: PairsLike, instances: Mapping, edges: Iterable,
                  rulings: Mapping, min_px: int = MIN_PX) -> list[Candidate]:
    """The candidates of one frame (see the module docstring), in the order the dock lists them.

    ``frame_pairs``: ``(above, below, px)`` triples (or ``{(above, below): px}``)
    of the frame on screen; ``instances``: the desktop's records; ``edges``:
    its relation rows as :class:`~tda.core.graph.Edge` (only active ones
    count); ``rulings``: its ``cover_ruling`` rows by ``(above, below)``.
    Ordered by tag (「建议收」 first), then pixels (most first), then keys.
    """
    edges = list(edges)
    picked = []
    for above, below, px in _triples(frame_pairs):
        if above == below or px < int(min_px) or (above, below) in rulings:
            continue
        up, low = instances.get(above), instances.get(below)
        if excluded_part(above, up) or excluded_part(below, low):
            continue
        if str(up.cls) in SMALL_ABOVE or str(low.cls) in SMALL_BELOW:
            continue
        if joined_by_edge(edges, above, below) or structural_reasons(instances, above, below):
            continue
        classes = (class_key(up), class_key(low))
        tag, reason = pair_tag(*classes)
        picked.append((above, below, classes, px, tag, reason))
    counts: dict[tuple[str, str], int] = {}
    for _a, _b, classes, *_rest in picked:
        counts[classes] = counts.get(classes, 0) + 1
    out = [Candidate(a, b, classes[0], classes[1], px, tag, reason, counts[classes] - 1)
           for a, b, classes, px, tag, reason in picked]
    out.sort(key=lambda c: (TAGS.index(c.tag), -c.px, c.above, c.below))
    return out


def class_pairs(candidates: Iterable[Candidate]) -> list[tuple[tuple[str, str], list[Candidate]]]:
    """The class pairs shared by two or more candidates, each with its candidates (the batch lines).

    In the order their first candidate is listed.
    """
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for cand in candidates:
        groups.setdefault(cand.classes, []).append(cand)
    return [(classes, group) for classes, group in groups.items() if len(group) >= 2]


def _px(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def candidate_text(cand: Candidate) -> str:
    """「optical_drive.01 → motherboard.01 · 像素 68 797 · 建议收」."""
    return f"{cand.above} → {cand.below} · 像素 {_px(cand.px)} · {cand.tag}"


def ruled_text(above: str, below: str, row: Mapping) -> str:
    """「✓ optical_drive.01 → motherboard.01 挡着（scan 第 7 帧）」: one line of 「已裁定」."""
    ruling = str(row.get("ruling"))
    where = ""
    if row.get("view") is not None and row.get("step") is not None:
        where = f"（{row['view']} 第 {row['step']} 帧）"
    return (f"{RULING_MARK.get(ruling, '?')} {above} → {below} "
            f"{RULING_TEXT.get(ruling, ruling)}{where}")


def status_text(desktop: int, step: int, pairs: list[tuple[str, str]], ruling: str,
                classes: Optional[tuple[str, str]] = None) -> str:
    """The status line after a click: 「D13 第 7 帧：optical_drive.01 压着 motherboard.01 → 挡着」.

    A batch names its class pair and the count instead of one pair.
    """
    word = RULING_TEXT.get(ruling, ruling)
    if len(pairs) == 1 and classes is None:
        (above, below), = pairs
        what = f"{above} 压着 {below}"
    else:
        a, b = classes if classes is not None else ("?", "?")
        what = f"{len(pairs)} 对 {a} 压着 {b}"
    return f"D{int(desktop):02d} 第 {int(step)} 帧：{what} → {word}"
