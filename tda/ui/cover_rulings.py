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
  connector, cable clip or cable (:data:`SMALL_BELOW`) -- except, since task
  W3h, a small part **on top** of one the log moved first: such a pair is
  never asked but is listed in 日志里下面那件先动了 for the layer check;
* no active edge of any type joins the two, in either direction;
* no structural reason: one's ``parent`` (any depth of ``attached``
  parents), ``mounted_on``, ``fastens``, ``socket_host`` or ``cable`` is the
  other, or the class pair is in :data:`SUPPORT` (the part sits on the
  motherboard);
* no ruling yet on this desktop for ``(above, below)`` (the ordered pair:
  ``(B, A)`` is a question of its own).

Each candidate is tagged with the user's class-level ruling (task W3g; it
replaced the covering adjudication sheet's 「建议收」 / 「存疑」 tables):
「用户已定：挡」 (:data:`TAG_USER_BLOCKS`) when its class pair is one of the
``pairs:`` of ``configs/covers_whitelist.yaml`` (W2 / W3, the user's answers to
``08-class-pair-questions.md`` and to the pair cards), with the user's words as
the reason, else 「其他」.  The row keeps both buttons: an instance-level
exception is still the user's to make.  No Qt here.

**Task W3f: two kinds of pair are no longer asked** (:func:`triage_pairs`; the
dock folds them away).  The user could not tell what a row was about, and on
D04 frame 21 was asked ``psu.01 → cover.01`` although the log had opened the
cover at step 22 with the PSU still in place.

* **规则已间接覆盖** (:func:`indirect_chain`): the active hard edges already
  order the pair through a third part -- ``below`` waits for X, X (perhaps
  through more parts) waits for ``above``.  ``cover.of = motherboard_screws``
  covers the board's screws and the screws fasten the board, so
  ``cover.01 → motherboard.01`` is answered by
  ``cover.01 → screw.motherboard.03 → motherboard.01`` (controller addendum,
  the user's 「盖板不打开 … 也就没法拆掉主板」).  Checked first: the graph knows.
* **日志已证明不挡** (:func:`log_proof`): the recorded log acted on ``below``
  with a verb a ``covered_by(below <- above)`` edge would gate (every gated
  verb: :data:`COVER_GATED`) while ``above`` was in none of the states that
  free it (:data:`COVER_CLEAR`: open / removed / displaced) -- exactly what
  W3a's ``pixel-edges`` refuses as ``would_violate``.  The replay is the one
  :func:`tda.core.graph.validate_sequence` does (:func:`cover_log`): actions in
  ``(step, idx)`` order, the hand-written initial states, the settled parts.

Rows also carry both parts' states at the frame (``states``) and, with the
log, the 「日志：…」 line of the tooltip (:func:`log_line`).

**Task W3g: a fourth kind is not asked** -- 「用户已定不挡（类级）」: the class
pair is in the whitelist's ``not_blocks:`` list (the user ruled it 不挡 for the
whole class: 「只是我们 disassembly 的时候有这个顺序」).  Judged after the
graph's chain and before the log; folded, no buttons.  The class rulings are
read once from the shipped whitelist (:func:`class_rulings`, through the same
loader ``cli pixel-edges`` uses), or passed in (``classes=``).

**Task W3h: the log's word is a question, not a proof** -- the W3f fold becomes
「日志里下面那件先动了（k）」 (subtitle 「不挡，或者层序画反了 —— 请看图定」).  It hid
exactly the errors the pair cards showed: ``hdd.01 → psu.01`` on D22 was
「日志已证明不挡」 because the log displaces the PSU while the HDD is still
installed -- which is a proof only if the layer order is right.  The user's
principle is the opposite (「你的遮挡关系算错了 … 要用我们的拆解的顺序作为主要参考」):
the disassembly order is the primary reference, so a covering whose lower part
the log moved first is either 不挡 or a layer order painted the wrong way round,
and the user decides with the picture (「我需要去检查一轮」).  The bucket is
exactly :func:`log_proof`'s set (every gated verb, no second replay), whatever
the class pair; a pair whose class pair the user ruled 挡 (the whitelist's
``pairs:``, **either** orientation -- ``(below, above)`` ruled 挡 means the
picture is the other way round, :func:`class_blocking`) carries the tag
「用户已定：挡 → 层序可能画反」 (:data:`TAG_LAYER`).  Two buttons per row: 「不挡」
(an ordinary ``not_blocks`` ruling, what the old fold's one button did) and
「上下翻转」 (the pair override ``below`` above ``above`` for the view's pose
segment through the session's ``flip_pair_layers``, which re-checks the frozen
frames, accepts the two parts' new values and re-signs the rows the tolerance
left behind; ``Ctrl+Z`` takes the override away and settles the rows the same
way).  The 「本版不做」 classes on top (a connector, a
clip, a latch) take part in this bucket alone (a screw / connector / cable
underneath never does): on the live copy (2026-10-31) the bucket is mostly
``connector → drive_cage`` (the Dell cage swung aside with its drives still
plugged in), the fans over their latches, the GPUs over their card latches,
connectors over a cover / drive / heatsink, and the two ``psu → cover`` the
user ruled the other way round (E4).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from tda.core.export.graph_covers import SUPPORT
from tda.core.graph import active_edges
from tda.core.graph_derive import settled_instances
from tda.core.graph_rules import HARD_TYPES, gated_verbs, required_states
from tda.core.model import (
    is_context,
    is_context_key,
    is_gallery_only,
    is_provisional,
)
from tda.core.states import REMOVED, events_from_actions, state_at
from tda.ui.class_names import state_zh

__all__ = ["BLOCKS", "COVER_CLEAR", "COVER_GATED", "Candidate", "ClassRulings", "CoverLog",
           "FIXED_CLASSES", "FLIP_LEFT_OPEN", "FLIP_NOTE", "FLIP_REFUSED_VERIFY", "FLIP_TEXT",
           "LogMove", "MIN_PX", "MOVE_ZH", "NOT_BLOCKS",
           "RULINGS", "RULING_MARK", "RULING_TEXT", "SMALL_ABOVE", "SMALL_BELOW", "SUPPORT",
           "TAGS", "TAG_LAYER", "TAG_OTHER", "TAG_USER_BLOCKS", "TAG_USER_NOT_BLOCKS", "Triage",
           "candidate_text", "chain_text", "chain_tip", "chassis_pairs", "class_blocking",
           "class_key", "class_not_blocks_text", "class_pairs", "class_rulings", "cover_log",
           "excluded_part", "flip_text", "indirect_chain", "indirect_text", "joined_by_edge",
           "log_line", "log_proof", "pair_tag", "pending_pairs", "proven_text", "proven_tip",
           "ruled_text", "status_text", "structural_reasons", "triage_pairs"]

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

#: The class pair is one the user ruled 挡 (W2 / W3 ``pairs:``, task W3g).
TAG_USER_BLOCKS = "用户已定：挡"
#: The class pair is one the user ruled 不挡 (W2 ``not_blocks:``): folded, not asked.
TAG_USER_NOT_BLOCKS = "用户已定：不挡"
#: A 日志里下面那件先动了 row whose class pair the user ruled 挡 (either way round, W3h).
TAG_LAYER = "用户已定：挡 → 层序可能画反"
#: The 日志里下面那件先动了 row's second button (W3h).
FLIP_TEXT = "上下翻转"
#: The op-log note of a flip: the pair, and the frames the override reaches.
FLIP_NOTE = "日志里下面那件先动了（W3h）：按拆解顺序把 {below} 翻到 {above} 之上，这一段第 {frames} 帧"
#: The status line's tail when a conflict of another part stayed open.
FLIP_LEFT_OPEN = "；另有 {n} 条别的零件的冲突留在复查队列（{who}）"
#: ... and when a demoted frame could not be re-signed.
FLIP_REFUSED_VERIFY = "；{n} 帧没能重签（{who}）"
TAG_OTHER = "其他"
#: The order candidates are listed in.
TAGS = (TAG_USER_BLOCKS, TAG_OTHER, TAG_USER_NOT_BLOCKS)

#: Class-level support: the part sits on the motherboard.  Since task W3d the
#: pair is a **rule edge** (``blocked_by(motherboard <- part, mode=mounted)``, spec 7.3
#: 「主板上的东西先拆」), so its order is already decided and Review never asks
#: it.  The one table (``tda.core.export.graph_covers.SUPPORT``, built from
#: ``tda.core.graph_rules.MOUNTED_ON_MOTHERBOARD``; unified at the W3a merge,
#: review M3), so Review, the rules and ``export-graph`` cannot disagree; it is
#: imported at the top of this module and re-exported as ``SUPPORT``.
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

_OTHER_REASON = "用户还没有按类别裁定这一类对：看图判断"
_USER_BLOCKS_REASON = "用户已按类别裁定：挡"
_USER_NOT_BLOCKS_REASON = "用户已按类别裁定：不挡"

ClassEntry = tuple[str, str, str]


@dataclass(frozen=True)
class ClassRulings:
    """The user's class-level rulings (task W3g): the W2 / W3 whitelist's two lists.

    ``blocks`` are its ``pairs:`` (挡), ``not_blocks`` its ``not_blocks:`` (不挡),
    each ``(above, below, note)`` with a side naming a ``class`` (every kind of
    it) or a ``class.kind``.  The loader refuses a class pair in both lists.
    """

    blocks: tuple[ClassEntry, ...] = ()
    not_blocks: tuple[ClassEntry, ...] = ()

    @classmethod
    def from_whitelist(cls, whitelist) -> "ClassRulings":
        """From a :class:`tda.core.pixel_edges.Whitelist`."""
        return cls(tuple((p.above, p.below, p.note) for p in whitelist.pairs),
                   tuple((p.above, p.below, p.note) for p in whitelist.not_blocks))

    @staticmethod
    def _best(entries: tuple[ClassEntry, ...], above_class: str,
              below_class: str) -> Optional[ClassEntry]:
        best: Optional[tuple[int, ClassEntry]] = None
        for entry in entries:
            a, b = _matches(entry[0], above_class), _matches(entry[1], below_class)
            if a is None or b is None:
                continue
            if best is None or a + b > best[0]:
                best = (a + b, entry)
        return None if best is None else best[1]

    def blocking(self, above_class: str, below_class: str) -> Optional[ClassEntry]:
        """The most exact 挡 entry naming the class pair, or ``None``."""
        return self._best(self.blocks, above_class, below_class)

    def not_blocking(self, above_class: str, below_class: str) -> Optional[ClassEntry]:
        """The most exact 不挡 entry naming the class pair, or ``None``."""
        return self._best(self.not_blocks, above_class, below_class)


#: The shipped whitelist (``configs/covers_whitelist.yaml``), next to the package.
WHITELIST_PATH = Path(__file__).resolve().parents[2] / "configs" / "covers_whitelist.yaml"


@lru_cache(maxsize=None)
def class_rulings(path: Optional[str] = None) -> ClassRulings:
    """The user's class rulings, read once through ``cli pixel-edges``' own loader (W3g).

    An unreadable or refused file gives no class rulings (every pair is asked,
    as before W3g) rather than no Review.
    """
    from tda.core.pixel_edges import WhitelistError, load_whitelist
    from tda.core.taxonomy import load_taxonomy

    try:
        return ClassRulings.from_whitelist(load_whitelist(path or WHITELIST_PATH,
                                                          load_taxonomy()))
    except (WhitelistError, OSError):
        return ClassRulings()
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
    #: 「用户已定：挡」 / 「其他」 / 「用户已定：不挡」 (:data:`TAGS`) -- the user's
    #: class-level ruling (task W3g).
    tag: str
    #: The user's words for the class ruling, or 「其他」's hint (the row's tooltip).
    reason: str
    #: How many *other* candidates share the class pair.  Only the frame on
    #: screen is known here -- the desktop's other frames are not compiled --
    #: so this counts the other pending pairs of **this frame**.
    same_pair_pending: int = 0
    #: Both parts' states at the frame on screen (task W3f; ``""``: not known).
    above_state: str = ""
    below_state: str = ""
    #: 「日志：…」, what the log did to each part (W3f; ``""`` without a log).
    log: str = ""
    #: The log's first action on ``below`` taken while ``above`` was still in
    #: the way (:func:`log_proof`), or ``None``: the row goes to
    #: 日志里下面那件先动了 (W3f's 日志已证明不挡 until task W3h).
    proof: Optional["LogMove"] = None
    #: The active edges ordering the pair through other parts, from ``below``
    #: up to ``above`` (:func:`indirect_chain`); empty: none.
    chain: tuple = ()
    #: Task W3h: the class pair as the user ruled it 挡, in the user's
    #: orientation -- ``(above_class, below_class)`` or, when the picture is the
    #: other way round, ``(below_class, above_class)``; ``()``: ruled 挡 in
    #: neither (:func:`class_blocking`).
    ruled: tuple = ()
    #: The user's words of that entry (``""`` without one).
    ruled_note: str = ""

    @property
    def classes(self) -> tuple[str, str]:
        """``(above_class, below_class)``: what the batch line groups by."""
        return (self.above_class, self.below_class)

    @property
    def ruled_reversed(self) -> bool:
        """Did the user rule the class pair 挡 the other way round from the picture?"""
        return bool(self.ruled) and self.ruled != self.classes

    @property
    def layer_tag(self) -> str:
        """「用户已定：挡 → 层序可能画反」 on a 日志里下面那件先动了 row of a ruled class pair, else ``""``."""
        return TAG_LAYER if self.proof is not None and self.ruled else ""


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


def pair_tag(above_class: str, below_class: str,
             classes: Optional[ClassRulings] = None) -> tuple[str, str]:
    """``(tag, reason)`` of a class pair from the user's class rulings (task W3g).

    A 挡 entry (W2 ``pairs:``) -> 「用户已定：挡」 with the user's words; a 不挡
    entry -> 「用户已定：不挡」; neither -> 「其他」.  ``classes``: the rulings
    (``None``: the shipped whitelist, :func:`class_rulings`).
    """
    classes = class_rulings() if classes is None else classes
    yes = classes.blocking(above_class, below_class)
    if yes is not None:
        return TAG_USER_BLOCKS, yes[2] or _USER_BLOCKS_REASON
    no = classes.not_blocking(above_class, below_class)
    if no is not None:
        return TAG_USER_NOT_BLOCKS, no[2] or _USER_NOT_BLOCKS_REASON
    return TAG_OTHER, _OTHER_REASON


def class_blocking(above_class: str, below_class: str,
                   classes: Optional[ClassRulings] = None) -> tuple[Optional[ClassEntry], bool]:
    """The 挡 entry naming the class pair in **either** orientation (task W3h).

    ``(entry, reversed)``: the picture's own orientation first; ``reversed`` is
    ``True`` when only ``(below_class, above_class)`` is ruled 挡 -- the user
    says the lower part should be on top.  ``(None, False)`` when neither is.
    """
    classes = class_rulings() if classes is None else classes
    yes = classes.blocking(above_class, below_class)
    if yes is not None:
        return yes, False
    rev = classes.blocking(below_class, above_class)
    return (rev, True) if rev is not None else (None, False)


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


# --------------------------------------------------------------------------- #
# what the log already says (task W3f)
# --------------------------------------------------------------------------- #
#: The verbs a ``covered_by`` edge holds back on its target: all the gated ones
#: (``graph_rules.GATES``) -- removing, displacing or opening ``below`` all wait.
COVER_GATED = gated_verbs("covered_by")
#: The states of the upper part that let them through (``REQUIRED_STATES``).
COVER_CLEAR = required_states("covered_by")
#: The three verbs that move a part out of the way, as the 日志 line says them.
MOVE_ZH = {"remove": "拆下", "open": "打开", "displace": "挪开"}
_ALL_STEPS = 10**9


@dataclass(frozen=True)
class LogMove:
    """One successful action of the log that a ``covered_by`` edge on its target would gate (W3f)."""

    step: int
    idx: int
    verb: str
    target: str
    #: Every part's state the moment before the action (``validate_sequence``'s
    #: replay); not compared, so two moves compare on what they are.
    before: Mapping[str, str] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class CoverLog:
    """The desktop's gated actions, each with the state before it (:func:`cover_log`, W3f)."""

    moves: tuple[LogMove, ...] = ()

    def of(self, key: str) -> list[LogMove]:
        """``key``'s own moves, in log order."""
        return [m for m in self.moves if m.target == key]


def cover_log(instances: Mapping, actions: Iterable, tax,
              initial: Optional[Mapping[str, str]] = None) -> CoverLog:
    """Replay the log the way :func:`tda.core.graph.validate_sequence` does. Pure.

    ``instances``: the desktop's records (the settled ones are replayed:
    :func:`tda.core.graph_derive.settled_instances`); ``actions``: its
    :class:`~tda.core.model.ActionRec` rows; ``initial``:
    :func:`tda.core.states.initial_overrides` of its hand-written events (a part
    whose initial state is ``removed`` is one the machine does not have: its
    actions are not replayed).  Every **successful** action whose verb is in
    :data:`COVER_GATED` becomes a :class:`LogMove` carrying the state of every
    part just before it -- earlier actions of the same compound step included.
    """
    settled = settled_instances(dict(instances))
    ordered = sorted(actions, key=lambda a: (int(a.step), int(a.idx)))
    gone = {key for key, state in (initial or {}).items() if state == REMOVED}
    moves: list[LogMove] = []
    for i, action in enumerate(ordered):
        if (action.target in gone or action.result != "success"
                or action.verb not in COVER_GATED):
            continue
        events = events_from_actions(settled, ordered[:i], tax, initial=initial)
        state = state_at(settled, events, _ALL_STEPS, tax, initial=initial)
        moves.append(LogMove(int(action.step), int(action.idx), str(action.verb),
                             str(action.target), {k: v.state for k, v in state.items()}))
    return CoverLog(tuple(moves))


def log_proof(log: Optional[CoverLog], above: str, below: str) -> Optional[LogMove]:
    """The first action on ``below`` taken while ``above`` was still in the way, or ``None``.

    "In the way": ``above`` is a part of the replay and its state just before
    the action is none of :data:`COVER_CLEAR` -- a ``covered_by(below <- above)``
    edge would make that very action a breach, so the pair can never become
    that edge (``pixel-edges`` would skip it as ``would_violate``).  Since task
    W3h this is a question, not a proof: either the pair is 不挡 or the layer
    order is painted the wrong way round (the 日志里下面那件先动了 bucket, the
    user decides).  A part opened and closed again before the action is in the
    way again, as the edge would see it.  ``below`` taken out only with its
    parent (no action of its own), or never touched, says nothing.
    """
    if log is None:
        return None
    for move in log.moves:
        if move.target != below:
            continue
        held = move.before.get(above)
        if held is not None and held not in COVER_CLEAR:
            return move
    return None


def _moves_words(log: CoverLog, key: str) -> list[str]:
    """「第 22 步 打开」, 「第 40 步 拆下」: the first of each way out of the way, up to the removal."""
    words, seen = [], set()
    for move in log.of(key):
        if move.verb not in MOVE_ZH or move.verb in seen:
            continue
        seen.add(move.verb)
        words.append(f"第 {move.step} 步 {MOVE_ZH[move.verb]}")
        if move.verb == "remove":
            break
    return words


def log_line(log: Optional[CoverLog], above: str, below: str) -> str:
    """「日志：cover.01 第 22 步 打开，psu.01 第 39 步 拆下」 (or 从未动过 / 从未拆下); ``""`` without a log."""
    if log is None:
        return ""
    up = "、".join(_moves_words(log, above)) or "从未动过"
    low = "、".join(_moves_words(log, below)) or "从未拆下"
    return f"日志：{above} {up}，{below} {low}"


def indirect_chain(edges: Iterable, above: str, below: str) -> tuple:
    """The shortest chain of active hard edges by which ``below`` already waits for ``above``.

    ``below`` is the target of an edge whose blocker is X, X the target of one
    whose blocker is ... ``above`` -- any depth, any hard type
    (:data:`~tda.core.graph_rules.HARD_TYPES`), :func:`active_edges` only.
    Returned from ``below`` upwards, as the edges; ``()`` when there is none
    or only the direct edge (which :func:`joined_by_edge` handles).  Ties go to
    the alphabetically first blocker, so the chain is deterministic.
    """
    up: dict[str, list] = {}
    for edge in active_edges(list(edges)):
        if edge.type in HARD_TYPES:
            up.setdefault(edge.target, []).append(edge)
    for kids in up.values():
        kids.sort(key=lambda e: (e.blocker, e.type))
    came: dict[str, tuple] = {below: ()}
    frontier = [below]
    while frontier:
        nxt = []
        for node in frontier:
            for edge in up.get(node, ()):
                if edge.blocker in came:
                    continue
                path = came[node] + (edge,)
                if edge.blocker == above:
                    return path if len(path) >= 2 else ()
                came[edge.blocker] = path
                nxt.append(edge.blocker)
        frontier = nxt
    return ()


@dataclass
class Triage:
    """The frame's pairs, sorted four ways (:func:`triage_pairs`, W3f, W3g, W3h)."""

    #: Open questions: the rows with 「挡着」 / 「不挡」.
    asked: list[Candidate] = field(default_factory=list)
    #: 日志里下面那件先动了 (W3f's 日志已证明不挡 until task W3h): each with its
    #: :attr:`Candidate.proof`; 「不挡」 and 「上下翻转」 per row,
    #: :attr:`Candidate.layer_tag` on the class pairs the user ruled 挡.
    proven: list[Candidate] = field(default_factory=list)
    #: 规则已间接覆盖: each with its :attr:`Candidate.chain`.
    indirect: list[Candidate] = field(default_factory=list)
    #: 用户已定不挡（类级）(task W3g): the class pair is in the whitelist's
    #: ``not_blocks:``; tag 「用户已定：不挡」, the user's words as the reason.
    class_not_blocks: list[Candidate] = field(default_factory=list)


def _order(cand: Candidate) -> tuple:
    return (TAGS.index(cand.tag), -cand.px, cand.above, cand.below)


def triage_pairs(frame_pairs: PairsLike, instances: Mapping, edges: Iterable,
                 rulings: Mapping, min_px: int = MIN_PX, *, log: Optional[CoverLog] = None,
                 states: Optional[Mapping[str, str]] = None,
                 classes: Optional[ClassRulings] = None) -> Triage:
    """The candidates of one frame (see the module docstring), sorted into the four lists.

    ``frame_pairs``: ``(above, below, px)`` triples (or ``{(above, below): px}``)
    of the frame on screen; ``instances``: the desktop's records; ``edges``:
    its relation rows as :class:`~tda.core.graph.Edge` (only active ones
    count); ``rulings``: its ``cover_ruling`` rows by ``(above, below)`` (a
    ruled pair is in none of the lists); ``log``: :func:`cover_log` of the
    desktop (``None``: nothing is proven by the log, and no 日志 line);
    ``states``: each part's state at the frame (``{key: state}``), for the row
    text; ``classes``: the user's class rulings (``None``: the shipped
    whitelist, :func:`class_rulings`; ``ClassRulings()``: none).  A pair the
    graph orders indirectly goes to ``indirect`` (whatever the rest says), else
    one whose class pair the user ruled 不挡 goes to ``class_not_blocks`` (task
    W3g), else one whose lower part the log moved first (:func:`log_proof`)
    goes to ``proven`` -- since task W3h the 日志里下面那件先动了 bucket, where a
    class pair the user ruled 挡 either way round carries
    :attr:`Candidate.ruled` and the tag 「用户已定：挡 → 层序可能画反」 -- else it
    is asked.  Each list is ordered by tag (「用户已定：挡」 first), then pixels
    (most first), then keys; :attr:`Candidate.same_pair_pending` counts the
    asked list only (the batch lines are made of it).  Pure.
    """
    edges = list(edges)
    states = states or {}
    classes = class_rulings() if classes is None else classes
    picked = []
    for above, below, px in _triples(frame_pairs):
        if above == below or px < int(min_px) or (above, below) in rulings:
            continue
        up, low = instances.get(above), instances.get(below)
        if excluded_part(above, up) or excluded_part(below, low):
            continue
        if str(low.cls) in SMALL_BELOW:
            continue
        if joined_by_edge(edges, above, below) or structural_reasons(instances, above, below):
            continue
        small = str(up.cls) in SMALL_ABOVE
        keys = (class_key(up), class_key(low))
        tag, reason = pair_tag(*keys, classes=classes)
        chain = indirect_chain(edges, above, below)
        said_no = not chain and tag == TAG_USER_NOT_BLOCKS
        entry, reversed_ = class_blocking(*keys, classes=classes)
        ruled = () if entry is None else ((keys[1], keys[0]) if reversed_ else keys)
        proof = None if chain or said_no else log_proof(log, above, below)
        if small and proof is None:
            # the sheet's 「本版不做」 (W3c): a connector / clip / latch on top is never
            # asked -- but one painted over a part the log moved first is a layer
            # check the user wants (task W3h: 日志里下面那件先动了 holds it)
            continue
        picked.append(Candidate(above, below, keys[0], keys[1], px, tag, reason, 0,
                                str(states.get(above) or ""), str(states.get(below) or ""),
                                log_line(log, above, below), proof, chain, ruled,
                                "" if entry is None else str(entry[2] or "")))
    out = Triage()
    for cand in picked:
        (out.indirect if cand.chain
         else out.class_not_blocks if cand.tag == TAG_USER_NOT_BLOCKS
         else out.proven if cand.proof else out.asked).append(cand)
    counts: dict[tuple[str, str], int] = {}
    for cand in out.asked:
        counts[cand.classes] = counts.get(cand.classes, 0) + 1
    out.asked = [replace(cand, same_pair_pending=counts[cand.classes] - 1)
                 for cand in out.asked]
    for name in ("asked", "proven", "indirect", "class_not_blocks"):
        getattr(out, name).sort(key=_order)
    return out


def pending_pairs(frame_pairs: PairsLike, instances: Mapping, edges: Iterable,
                  rulings: Mapping, min_px: int = MIN_PX, *, log: Optional[CoverLog] = None,
                  states: Optional[Mapping[str, str]] = None,
                  classes: Optional[ClassRulings] = None) -> list[Candidate]:
    """The open questions of one frame: :func:`triage_pairs`' ``asked`` list.

    Without ``log`` only the pairs the graph orders indirectly (task W3f) and
    the class pairs the user ruled 不挡 (task W3g) are left out besides W3c's rule.
    """
    return triage_pairs(frame_pairs, instances, edges, rulings, min_px, log=log,
                        states=states, classes=classes).asked


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


def _with_state(key: str, state: str) -> str:
    return f"{key}（{state_zh(state)}）" if state else key


def _pair_words(cand: Candidate) -> str:
    """「cover.01（打开）→ motherboard.01（装着）」; without states the old 「a → b」."""
    if not (cand.above_state or cand.below_state):
        return f"{cand.above} → {cand.below}"
    return f"{_with_state(cand.above, cand.above_state)}→ {_with_state(cand.below, cand.below_state)}"


def candidate_text(cand: Candidate) -> str:
    """「optical_drive.01（装着）→ motherboard.01（装着） · 像素 68 797 · 用户已定：挡」.

    The states are the parts' states at the frame on screen (task W3f); a
    candidate made without them reads 「optical_drive.01 → motherboard.01 · …」.
    """
    return f"{_pair_words(cand)} · 像素 {_px(cand.px)} · {cand.tag}"


def proven_text(cand: Candidate) -> str:
    """A 日志里下面那件先动了 row (W3h): 「psu.01（装着）→ cover.01（关上） · 像素 1 535 · 第 22 步先动了下面的 cover.01」.

    With 「 · 用户已定：挡 → 层序可能画反」 at the end when the user ruled the
    class pair 挡 either way round (:attr:`Candidate.layer_tag`).
    """
    move = cand.proof
    if move is None:
        return candidate_text(cand)
    text = f"{_pair_words(cand)} · 像素 {_px(cand.px)} · 第 {move.step} 步先动了下面的 {cand.below}"
    return f"{text} · {cand.layer_tag}" if cand.layer_tag else text


def proven_tip(cand: Candidate) -> str:
    """The row's tooltip (W3h): what the log did and when, which way the user ruled the class pair, the 日志 line."""
    move = cand.proof
    if move is None:
        return cand.log
    verb = MOVE_ZH.get(move.verb, move.verb)
    held = state_zh(move.before.get(cand.above, ""))
    what = (f"第 {move.step} 步{verb} {cand.below} 时 {cand.above} 还是{held}：要么 {cand.above} 没挡着"
            f"（点「{RULING_TEXT[NOT_BLOCKS]}」记一条裁定），要么层序画反了（点「{FLIP_TEXT}」把 "
            f"{cand.below} 翻到 {cand.above} 之上，这一视角这一段；重查已确认帧、接受这两件的新值并重签；"
            f"Ctrl+Z 撤回覆盖并同样接受 / 重签）")
    note = f"（{cand.ruled_note}）" if cand.ruled_note else ""
    if not cand.ruled:
        said = ""
    elif cand.ruled_reversed:
        said = (f"你按类别裁定 {cand.ruled[0]} → {cand.ruled[1]} 挡{note}：按拆解顺序 {cand.below} "
                f"应画在 {cand.above} 之上，这一帧却画反了")
    else:
        said = (f"你按类别裁定 {cand.ruled[0]} → {cand.ruled[1]} 挡{note}，日志却先动了下面那件："
                f"要么层序画反，要么这一对是例外")
    return "\n".join(part for part in (what, said, cand.log) if part)


def chain_text(cand: Candidate) -> str:
    """「cover.01 → screw.motherboard.03 → motherboard.01」: the chain, top part first."""
    keys = [cand.below] + [edge.blocker for edge in cand.chain]
    return " → ".join(reversed(keys))


def indirect_text(cand: Candidate) -> str:
    """A 规则已间接覆盖 row: the chain in words, then the pixels."""
    return f"{chain_text(cand)} · 像素 {_px(cand.px)}"


def class_not_blocks_text(cand: Candidate) -> str:
    """A 用户已定不挡（类级） row (W3g): 「drive_cage.01（装着）→ cpu.01（装着） · 像素 4 406 · C 不挡：…」."""
    return f"{_pair_words(cand)} · 像素 {_px(cand.px)} · {cand.reason}"


def chain_tip(cand: Candidate) -> str:
    """The edges of the chain, one per line (the row's tooltip)."""
    return "\n".join(f"{edge.target} 等 {edge.blocker}（{edge.type}）" for edge in cand.chain)


def flip_text(desktop: int, step: int, above: str, below: str, result: Mapping) -> str:
    """The status line after 「上下翻转」 (W3h).

    「D22 第 5 帧：psu.01 翻到 storage_drive.hdd.01 之上 · 这一段 12 帧（已确认 9 帧）· 接受新值 18 条 · 重签 0 帧 9 行」
    -- ``resigned`` frames re-signed whole (``verify_frame``), ``rewritten`` rows
    the tolerance left behind and U32's refreeze re-signed -- plus
    :data:`FLIP_LEFT_OPEN` when a conflict of another part stayed open and
    :data:`FLIP_REFUSED_VERIFY` when a demoted frame could not be re-signed.
    """
    frames = list(result.get("frames") or [])
    frozen = list(result.get("frozen") or [])
    text = (f"D{int(desktop):02d} 第 {int(step)} 帧：{below} 翻到 {above} 之上 · 这一段 {len(frames)} 帧"
            f"（已确认 {len(frozen)} 帧）· 接受新值 {int(result.get('accepted') or 0)} 条 · "
            f"重签 {int(result.get('resigned') or 0)} 帧 {int(result.get('rewritten') or 0)} 行")
    for key, words in (("left_open", FLIP_LEFT_OPEN), ("verify_refused", FLIP_REFUSED_VERIFY)):
        items = list(result.get(key) or [])
        if items:
            who = "、".join(f"第 {s} 帧 {what}" for s, what in items[:4])
            if len(items) > 4:
                who += " …"
            text += words.format(n=len(items), who=who)
    return text


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
