"""Groups of interchangeable parts, and the red marks that name a part (redesign 2026-11-03).

The owner's audit of the first pack found every answer right and many questions unaskable:
"RAM module 3" and "CPU cooler screw 2" name nothing a viewer can find. Two rules replace
the ordinal (spec ``docs/superpowers/specs/2026-11-03-vlm-question-redesign-design.md``):

* a **part group** is the instances of one desktop that share their :func:`match_class` and
  their relational anchor -- screws by what they fasten, latches by what they lock
  (``attrs.of``), connectors by host and cable, everything else by class alone (``-``).
  Its id is ``f"{match_class}@{anchor}"``; a group of one is no group, its part is named by
  its instance key (:func:`part_groups`);
* a part a question names is **outlined on the scanner image** -- its visible-mask box, or
  for a part present and hidden its amodal box -- and the question says "the RAM module
  outlined in red" (:func:`mark_box`, :class:`Namer`). A part that cannot be outlined is
  not asked about.

Pure and Qt-free; :mod:`tda.core.export.vlm_tasks` builds on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from tda.core.export.coco import ANSWERABLE, DesktopCtx
from tda.core.graph_rules import CABLE_PREFIX, verb_applies
from tda.core.implied import is_implied
from tda.core.model import is_provisional
from tda.core.states import FrameState, gone_with_parent
from tda.core.taxonomy import Taxonomy

__all__ = [
    "NO_ANCHOR",
    "Namer",
    "PartGroup",
    "class_label",
    "class_phrase",
    "group_anchor",
    "group_labels",
    "group_of",
    "mark_box",
    "match_class",
    "part_groups",
    "plural",
    "present",
    "target_class",
    "to_do",
]

CABLE_CLASS = "cable"
REMOVED = "removed"
#: The anchor of a part that has none (a RAM module, a drive cage, a latch of an empty slot).
NO_ANCHOR = "-"
#: A part in its place whose every pixel is hidden: outlined by its amodal box (spec 3.2).
HIDDEN_IN_PLACE = frozenset({"occluded_full"})

#: Class-id and attribute words that are initialisms rather than words. A
#: question reading "what state is the psu in?" is a question about a typo;
#: ``str.title()`` would make it "Psu", which is worse.
INITIALISMS = frozenset({
    "psu", "cpu", "gpu", "ram", "ssd", "hdd", "io", "atx", "sata", "usb",
    "wlan", "pc", "pcb", "24pin", "ph1", "ph2", "ph3",
})


def _words(text: str) -> list[str]:
    """``"cpu_cooler"`` -> ``["CPU", "cooler"]``: initialisms up, rest as written."""
    return [
        word.upper() if word.lower() in INITIALISMS else word
        for word in str(text).replace("_", " ").split()
    ]


def class_label(cls: str, tax: Optional[Taxonomy] = None) -> str:
    """How a person says a taxonomy class: ``ram_module`` -> "RAM module"."""
    defn = (tax.classes.get(cls) if tax is not None else None) or {}
    given = defn.get("label") or defn.get("name")
    return " ".join(_words(given if isinstance(given, str) and given else cls))


def target_class(ctx: DesktopCtx, target: str) -> Optional[str]:
    """Class of an action target: ``cable`` for a virtual node, else the taxonomy's."""
    if target.startswith(CABLE_PREFIX):
        return CABLE_CLASS
    return ctx.cls_of(target)


def match_class(ctx: DesktopCtx, target: str) -> Optional[str]:
    """The class two actions must share to count as interchangeable (V6, W3b).

    The taxonomy class, refined where the class alone would pair parts nobody
    would swap: a screw by its ``role`` (a PSU screw is not a cooler screw), a
    ``cpu_cooler`` by its ``kind`` (the fan is not the heatsink) and a
    connector by its ``kind`` (the front-panel loom is not the ATX plug). Four
    unordered cooler screws all answer ``screw.cpu_cooler``.
    """
    cls = target_class(ctx, target)
    if cls is None:
        return None
    rec = ctx.instances.get(target)
    if rec is None:
        return cls
    if cls == "screw" and rec.attrs.get("role"):
        return f"screw.{rec.attrs['role']}"
    if cls in ("cpu_cooler", "connector") and rec.attrs.get("kind"):
        return f"{cls}.{rec.attrs['kind']}"
    return cls


def plural(phrase: str) -> str:
    """``"RAM latch"`` -> ``"RAM latches"``: the last word made plural."""
    words = phrase.split(" ")
    last = words[-1]
    low = last.lower()
    if last.isupper():
        words[-1] = last + "s"                      # "PSUs", "CPUs"
    elif low.endswith(("s", "x", "ch", "sh")):
        words[-1] = last + "es"
    elif low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def class_phrase(ctx: DesktopCtx, key: str, many: bool = False) -> str:
    """How a question names a part with no ordinal: ``screw.motherboard.03`` -> "motherboard screw".

    The refining attribute (screw ``role``, cooler / connector ``kind``) and the class, as
    :func:`tda.core.export.vlm_tasks.instance_label` words them, lower case except the
    initialisms; ``many`` makes it plural ("CPU cooler screws").
    """
    rec = ctx.instances.get(key)
    if rec is None:
        cls = target_class(ctx, key)
        text = class_label(cls or key, ctx.tax)
        return plural(text) if many else text
    disc = rec.attrs.get("role") or rec.attrs.get("kind") or ""
    if rec.cls == "drive_cage" and rec.attrs.get("of") == "optical":
        disc = "optical"
    words = _words(disc) if disc else []
    words.extend(_words(class_label(str(rec.cls), ctx.tax)))
    text = " ".join(w for w in words if w)
    return plural(text) if many else text


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PartGroup:
    """Interchangeable parts of one desktop: one match class, one anchor, at least two members."""

    gid: str
    members: tuple[str, ...]
    match_class: str
    anchor: str
    label: str


def group_anchor(ctx: DesktopCtx, key: str) -> str:
    """The instance key a part's group is anchored on, or ``-`` (spec 3.1, 6.2).

    Screws by what they fasten, connectors by ``socket_host | cable``, every other part by
    ``attrs.of`` when it names an instance (the module a latch locks), else none.
    """
    rec = ctx.instances.get(key)
    if rec is None:
        return NO_ANCHOR
    if rec.cls == "screw":
        return str(rec.fastens) if rec.fastens else NO_ANCHOR
    if rec.cls == "connector":
        if not rec.socket_host and not rec.cable:
            return NO_ANCHOR
        return f"{rec.socket_host or NO_ANCHOR}|{rec.cable or NO_ANCHOR}"
    of = rec.attrs.get("of")
    if isinstance(of, str) and of in ctx.instances:
        return of
    return NO_ANCHOR


def _groupable(ctx: DesktopCtx, key: str) -> bool:
    """A real, named part: not a draft, not an implied identity, with a taxonomy class."""
    rec = ctx.instances.get(key)
    return (rec is not None and ctx.cls_of(key) is not None
            and not is_provisional(key) and not is_implied(rec))


def part_groups(ctx: DesktopCtx) -> dict[str, PartGroup]:
    """``instance key -> its group``, for every part in a group of at least two."""
    buckets: dict[tuple[str, str], list[str]] = {}
    for key in sorted(ctx.instances):
        if not _groupable(ctx, key):
            continue
        mc = match_class(ctx, key)
        if mc is None:
            continue
        buckets.setdefault((mc, group_anchor(ctx, key)), []).append(key)
    found = [(mc, anchor, members) for (mc, anchor), members in sorted(buckets.items())
             if len(members) >= 2]
    labels = group_labels(ctx, [(mc, anchor, members[0]) for mc, anchor, members in found])
    out: dict[str, PartGroup] = {}
    for (mc, anchor, members), label in zip(found, labels):
        group = PartGroup(gid=f"{mc}@{anchor}", members=tuple(members), match_class=mc,
                          anchor=anchor, label=label)
        for key in members:
            out[key] = group
    return out


def _cable_words(cable: str) -> str:
    """``cable:psu`` -> "PSU cable" (an ordinal in the node name is dropped)."""
    owner = cable.split(":", 1)[1] if ":" in cable else cable
    owner = ".".join(part for part in owner.split(".") if not part.isdigit())
    words = _words(owner)
    return " ".join([*words, "cable"]) if words else "cable"


def _anchor_phrases(ctx: DesktopCtx, mc: str, anchor: str) -> list[str]:
    """The qualifiers that tell a group from the others of its class, coarse to fine.

    A connector group: the cable it belongs to ("PSU cable"), then the part it plugs
    into ("on the motherboard"); any other group: the part it is anchored on ("of the
    fan CPU cooler"). Class phrases only -- never an ordinal.
    """
    if anchor == NO_ANCHOR:
        return []
    if mc.split(".")[0] == "connector" and "|" in anchor:
        host, cable = anchor.split("|", 1)
        out = []
        if cable != NO_ANCHOR:
            out.append(("cable", _cable_words(cable)))
        if host != NO_ANCHOR and host in ctx.instances:
            out.append(("on", class_phrase(ctx, host)))
        return out
    if anchor in ctx.instances:
        return [("of", class_phrase(ctx, anchor))]
    return []


def _qualified(base: str, quals: Sequence[tuple[str, str]]) -> str:
    text = base
    for kind, words in quals:
        if kind == "cable":
            text = f"{words} {text}"
        else:
            text = f"{text} {kind} the {words}"
    return text


def group_labels(ctx: DesktopCtx, groups: Sequence[tuple[str, str, str]]) -> list[str]:
    """How each group ``(match_class, anchor, first member)`` is said, in that order.

    The plural class phrase ("CPU cooler screws", "connectors"); where several groups of
    one desktop would read the same, the anchor's words are added until they differ or
    run out -- "PSU cable connectors", "optical drive cable connectors on the motherboard",
    "CPU cooler screws of the fan CPU cooler"; an anchor-less group beside anchored ones
    reads "RAM latches without a RAM module". Groups whose anchors read the same (the
    latches of two RAM modules) keep the plain label: words cannot tell them apart, and the
    grader counts group actions per class there (spec 6.2). Never an ordinal.
    """
    base = [class_phrase(ctx, first, many=True) for _, _, first in groups]
    quals = [_anchor_phrases(ctx, mc, anchor) for mc, anchor, _ in groups]
    labels = list(base)
    depth = max((len(q) for q in quals), default=0)
    for level in range(1, depth + 1):
        seen: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            seen.setdefault(label, []).append(i)
        clashing = [ids for ids in seen.values() if len(ids) > 1]
        if not clashing:
            break
        for ids in clashing:
            for i in ids:
                labels[i] = _qualified(base[i], quals[i][:level])
    # words that still read the same tell nothing apart: such groups keep the plain label
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    labels = [base[i] if counts[label] > 1 else label for i, label in enumerate(labels)]
    # an anchor-less group among anchored ones of its label: "... without a <anchor>"
    seen = {}
    for i, label in enumerate(labels):
        seen.setdefault(base[i], []).append(i)
    for ids in seen.values():
        if len(ids) < 2:
            continue
        anchored = {quals[i][0][1] for i in ids if quals[i] and quals[i][0][0] == "of"}
        for i in ids:
            if not quals[i] and len(anchored) == 1:
                labels[i] = f"{base[i]} without a {next(iter(anchored))}"
    return labels


def group_of(groups: dict[str, PartGroup], key: str) -> Optional[PartGroup]:
    """The group of ``key``, or ``None`` for a part that is alone (or a gid itself)."""
    return groups.get(key)


def present(state: FrameState, key: str) -> bool:
    """Is the part still on the machine -- not removed, not gone inside its parent?"""
    inst = state.get(key)
    return inst is not None and inst.state != REMOVED and not gone_with_parent(state, key)


def is_captive(ctx: DesktopCtx, key: str) -> bool:
    rec = ctx.instances.get(key)
    return bool(rec is not None and rec.cls == "screw" and rec.attrs.get("captive"))


def to_do(ctx: DesktopCtx, state: FrameState, verb: str, members: Sequence[str]) -> list[str]:
    """The members of a group for which ``verb`` is still to be done in ``state`` (spec 6.1).

    The members in the asked situation: present, and the verb applies to them. Captive
    (cooler) screws are the exception (spec 6.6): their tightness is invisible, so
    "unscrew them" is still to do for every one present while the part they hold is in.
    """
    out = []
    for key in members:
        rec = ctx.instances.get(key)
        if rec is None or not present(state, key):
            continue
        inst = state[key]
        if verb == "unscrew" and is_captive(ctx, key):
            if rec.fastens is None or present(state, rec.fastens):
                out.append(key)
            continue
        if verb_applies(ctx.tax, rec.cls, rec.attrs, verb, inst.state):
            out.append(key)
    return out


# --------------------------------------------------------------------------- #
# marks
# --------------------------------------------------------------------------- #
def mark_box(tc, frame, key: str) -> Optional[list]:
    """``[x, y, w, h]`` to outline ``key`` on ``frame``, or ``None`` when it cannot be marked.

    The visible-mask box of a part the frame can point at (``frame.pointable``); for a part
    present and wholly hidden in its place (``occluded_full``) its amodal box, which the
    export reads into the frame's rows (``rows[key]["amodal_box"]``). Nothing for a part
    that is removed or out of view, nor on a frame without confirmed rows (spec 3.2).
    """
    if frame is None or not getattr(frame, "confirmed", False):
        return None
    if not present(tc.state(frame.step), key):
        return None
    if key in frame.pointable:
        row, box = frame.pointable[key]
        if row.get("visibility") in ANSWERABLE:
            return list(box)
    row = frame.rows.get(key) or {}
    if row.get("visibility") in HIDDEN_IN_PLACE and row.get("amodal_box"):
        return list(row["amodal_box"])
    return None


@dataclass
class _Ref:
    ident: str
    members: tuple[str, ...]
    boxes: tuple[list, ...]
    tag: str
    phrase: str        # singular class phrase
    group: bool


@dataclass
class Namer:
    """The parts one question names, each outlined on image ``image`` with its own tag.

    Register every part first (:meth:`part`, :meth:`group`), then word them
    (:meth:`say`): one named part reads "the RAM module outlined in red" (a group "the CPU
    cooler screws outlined in red"); with several, "part A", "parts B" (spec 3.2).
    """

    tc: object
    frame: object
    image: int = 0
    refs: list = field(default_factory=list)

    def _find(self, ident: str) -> Optional[_Ref]:
        return next((r for r in self.refs if r.ident == ident), None)

    def part(self, key: str) -> Optional[str]:
        """Register one part; its id, or ``None`` when it cannot be marked."""
        hit = self._find(key)
        if hit is not None:
            return hit.ident
        box = mark_box(self.tc, self.frame, key)
        if box is None:
            return None
        self.refs.append(_Ref(key, (key,), (box,), chr(ord("A") + len(self.refs)),
                              class_phrase(self.tc.ctx, key), False))
        return key

    def group(self, gid: str, members: Sequence[str]) -> Optional[str]:
        """Register the members of a group still in the asked situation, under one tag."""
        hit = self._find(gid)
        if hit is not None:
            return hit.ident
        marked = [(m, mark_box(self.tc, self.frame, m)) for m in members]
        marked = [(m, b) for m, b in marked if b is not None]
        if not marked:
            return None
        self.refs.append(_Ref(gid, tuple(m for m, _ in marked), tuple(b for _, b in marked),
                              chr(ord("A") + len(self.refs)),
                              class_phrase(self.tc.ctx, marked[0][0]), True))
        return gid

    def many(self, ident: str) -> bool:
        ref = self._find(ident)
        return bool(ref is not None and len(ref.members) > 1)

    def say(self, ident: str) -> str:
        """The words for a registered part or group."""
        ref = self._find(ident)
        if ref is None:
            raise KeyError(ident)
        several = len(ref.members) > 1
        if len(self.refs) == 1:
            words = plural(ref.phrase) if several else ref.phrase
            return f"the {words} outlined in red"
        return f"parts {ref.tag}" if several else f"part {ref.tag}"

    def marks(self) -> list[dict]:
        """``prompt.marks``: one entry per marked member, a group's members sharing its tag."""
        return [{"image": int(self.image), "bbox": list(box), "tag": ref.tag}
                for ref in self.refs for box in ref.boxes]

    def members(self) -> dict[str, list[str]]:
        """``id -> the members outlined for it`` (label side)."""
        return {ref.ident: list(ref.members) for ref in self.refs}
