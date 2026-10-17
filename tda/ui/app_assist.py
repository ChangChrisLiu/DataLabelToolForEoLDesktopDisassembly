"""The window half of the model assist: SAM prompts, candidates, the heat map.

:mod:`tda.ui.app_diff` owns the comparison itself and its worker thread; this
mixes into :class:`tda.ui.app.MainWindow` and is the part that knows what a
task card is -- which instances a blob may be explained by, which item the
annotator activated, and where a prompt box should come from.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from tda.core.diffmap import DiffBlob, explain_blobs
from tda.core.model import is_context_key
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui import prompt_gate as PG
from tda.ui import session_api as api
from tda.ui.app_diff import (
    ALT_DEDUP_IOU,
    AssistController,
    Box,
    _box_iou,
    alternate_parts,
    best_unexplained,
    blob_boxes,
    expected_boxes,
    expected_payload,
    heat_rgba,
)
from tda.models.load_gate import Turn, heavy
from tda.ui.app_detect import DetCandidate, DetectMixin, detector_asked
from tda.ui.guide import DRAW_NO_SAM as G_DRAW_NO_SAM
from tda.ui.app_roi_worker import RoiProposer
from tda.ui.class_names import class_of_key, class_zh
from tda.ui.context_text import NO_PROMPT_BOX as NO_PROMPT_BOX_CONTEXT

__all__ = ["ASSIST_CONFIRM_WAIT", "BOX_REFUSED", "DET_CHIP", "DET_CHIP_NO_SAM",
           "DET_CHIP_RANK", "EDITING_NO_BOX", "HEAD_OFF_NOTE", "NO_ALTERNATE", "PROMPT_ARMED",
           "PROMPT_ARMED_DET", "PROMPT_ARMED_DET_NO_SAM", "PROMPT_CHIP",
           "PROMPT_CHIP_RANK", "PROMPT_TOO_BIG", "PROMPT_WITHHELD", "AssistMixin",
           "DiffOffer", "short_reason"]

class _SamLoader(QObject):
    """Carries the outcome of the background SAM load onto the GUI thread."""

    sigLoaded = Signal(object)


#: How long ``Space`` waits for an unfinished comparison before giving up on it.
ASSIST_CONFIRM_WAIT = 1.0

#: A window with no SAM queue that has not started loading one this long after
#: it was built loads none: the detector's build stops waiting for SAM's turn
#: (U6 round 2).  ``tda.cli app`` starts SAM right after ``show()``.
SAM_TURN_GRACE_MS = 2000

#: A diff box bigger than this share of the ROI is not used as a prompt box:
#: it says "everything changed", which narrows nothing for SAM.  The rule holds
#: at every rank: ``Shift+C`` may not reach a box the difference map itself
#: would have refused to arm.
MAX_PROMPT_BOX_FRAC = 0.6

#: Why ``Shift+C`` did nothing, in the four cases where something else owns the
#: canvas or the keys.  A key that silently does nothing is a key the annotator
#: presses again harder.
NO_ALTERNATE = ("差异图没有别的候选框 / no other box prompt from the "
                "difference map")
ALT_WRONG_MODE = "只有标注模式有提示框 / only Annotate mode has a prompt box"
ALT_FLASHING = "松开 Tab 再换提示框 / release Tab first: another frame is on screen"
ALT_ROI_EDITING = ("ROI 框正开着，先 Enter 或 Esc / the ROI rectangle owns the "
                   "keys right now")
ALT_GHOST = ("草稿幽灵正开着，先 Enter 或 Esc / the draft ghost owns the keys "
             "right now")
#: Appended when ``Shift+C`` drops a prompt that had already put pixels on
#: screen.  Since round 1 a result is *composed* into the layer rather than
#: replacing it, so "it stays" is no longer news -- what the annotator needs to
#: know is that the previous box's pixels are now part of the shape they are
#: building, and which key takes them back out.
APPLIED_MASK_STAYS = ("；上一个框的掩码已经并进图层，不要就 Ctrl+Z / the "
                      "previous box's mask is part of the layer now: Ctrl+Z "
                      "takes it off")
#: The armed box's chip on the canvas (task U2g, addendum A): what it is, and
#: that it is a guess -- on D13/scan 37 it sat on a cable, not on the screw.
PROMPT_CHIP = "SAM 提示框（程序猜的位置）"
#: ... and after ``Shift+C``: which of how many.
PROMPT_CHIP_RANK = "SAM 提示框 {rank}/{total}"
#: The arrival line, when the difference map arms a box (addendum C).  The
#: box's coordinates and size go to the log, not here.  "虚线框", not "小框":
#: the box is whatever changed, and on D13/scan 37 that was 120 x 350 px.
PROMPT_ARMED = ("虚线框是程序猜的位置（这一帧和下一帧差别最大的地方）：对就直接在零件上点 S；"
                "不对就直接点零件，框会自动不用，或按 Shift+C 换一个 / the dashed box is "
                "a guess: click the part; a click outside it drops the box, "
                "Shift+C offers another")
#: ... and when the change is too big to be a box at all.
PROMPT_TOO_BIG = ("这一帧差不多整块机箱范围都变了，这次没有提示框：直接在零件上点 S 就行 / "
                  "almost the whole chassis range changed: no prompt box this "
                  "time, just click the part")
#: ... and when its size is not the size of the part the card asks for (task
#: U2h, :mod:`tda.ui.prompt_gate`).  ``{name}`` is the part's Chinese name;
#: several are joined by :data:`WITHHELD_JOIN`.
PROMPT_WITHHELD = ("这一步要补的是「{name}」，程序找到的变化大小不像它，这次不给提示框："
                   "直接在零件上点 S，或用 B 涂 / no guess this time: click the part")
WITHHELD_JOIN = "」或「"
#: Appended to "正在画 X" when an edit starts on a frame whose box was
#: withheld, which would otherwise overwrite the one line saying so (U2i);
#: in both languages, like every other status line (U3).
EDITING_NO_BOX = (" — 这一帧没有提示框：直接在零件上点 S / no prompt box on this "
                  "frame: click the part")
#: The chip of a box the small-part detector armed (task U3): what it is and
#: who found it.  ``{name}`` is the class's Chinese name (螺丝).
DET_CHIP = "SAM 提示框（检测器：{name}）"
#: ... while SAM is not loaded (task U6): the box is still worth showing -- it
#: is where ``Y`` goes -- but "SAM 提示框" on a box SAM will never see is what
#: confused the D13 annotator the day SAM's import deadlocked.
DET_CHIP_NO_SAM = "检测器找到的{name}（SAM 未加载：Y 拖一下就画好）"
#: The arrival line in the same case.
PROMPT_ARMED_DET_NO_SAM = ("虚线框是检测器找到的{name}：SAM 没加载，按 Y 在{name}中心按下、拖到"
                           "边缘松手就画好（或 B 画笔 / P 多边形） / the dashed box is a {cls} "
                           "the detector found; SAM is not loaded: Y and drag")
#: ... and after ``Shift+C``: which of how many.
DET_CHIP_RANK = "SAM 提示框 {rank}/{total}（检测器：{name}）"
#: The other boxes while SAM is not loaded (U6 round 2): the same rule as
#: :data:`DET_CHIP_NO_SAM` -- the box stays, it is where ``Y`` or ``B`` goes, and
#: it is not called a SAM prompt.  All four go back the moment SAM loads.
PROMPT_CHIP_NO_SAM = "程序猜的位置（SAM 未加载：Y 拖一下 / B 涂）"
PROMPT_CHIP_RANK_NO_SAM = "提示框 {rank}/{total}（SAM 未加载：Y 拖一下 / B 涂）"
DET_CHIP_RANK_NO_SAM = "检测器找到的{name} {rank}/{total}（SAM 未加载：Y 拖一下 / B 涂）"
#: :data:`PROMPT_ARMED` with no SAM to click for.
PROMPT_ARMED_NO_SAM = ("虚线框是程序猜的位置（这一帧和下一帧差别最大的地方）：SAM 没加载，零件在框里就"
                       "按 Y 拖一下，或用 B 涂 / P 围出它 / the dashed box is a guess; SAM is "
                       "not loaded: Y, B or P")
#: chip kind -> (with SAM, without SAM); a chip on screen is kept as
#: ``(kind, fields)`` so it can be said again either way (:meth:`chip_text`).
CHIPS = {
    "diff": (PROMPT_CHIP, PROMPT_CHIP_NO_SAM),
    "diff_rank": (PROMPT_CHIP_RANK, PROMPT_CHIP_RANK_NO_SAM),
    "det": (DET_CHIP, DET_CHIP_NO_SAM),
    "det_rank": (DET_CHIP_RANK, DET_CHIP_RANK_NO_SAM),
}
#: The chip of the difference map's own box.
DIFF_CHIP = ("diff", {})
#: The arrival line when the detector armed the box.  Its confidence and dE
#: go to the log, not here.
PROMPT_ARMED_DET = ("虚线框是检测器找到的{name}（这一帧和下一帧差别最大的那个）：对就直接在零件上"
                    "点 S；不对就直接点零件，框会自动不用，或按 Shift+C 换一个 / the dashed box "
                    "is a {cls} the detector found (the one that changes most in the next "
                    "frame): click the part; a click outside it drops the box, Shift+C "
                    "offers another")
#: The first click of a prompt landed outside the armed box (addendum B).
BOX_REFUSED = ("你点在提示框外：这次只用你的点，提示框不用了 / clicked outside the "
               "prompt box: point only")
#: ... held against SAM's "+N px" line, which lands a fraction of a second
#: later and would otherwise wipe the one sentence saying why the box went.
BOX_REFUSED_HOLD_MS = 4000
#: After "SAM ready" while SAM's fine-tuned head is off (task U19): the masks
#: are the stock head's, as before U19.  The log line says why.
HEAD_OFF_NOTE = "（原版 SAM：微调头没加载）"


def short_reason(text: str, limit: int = 48) -> str:
    """A failure's message in a few words, for the status badge (task U6).

    ``_DeadlockError: deadlock detected by _ModuleLock('torchvision...') at
    2226949304144`` -> ``_DeadlockError: deadlock detected by _ModuleLock(…``:
    the exception's name first, a memory address never, and the whole line
    stays in the log.
    """
    cleaned = re.sub(r"\s+at 0x[0-9a-fA-F]+|\s+at \d{6,}", "", str(text)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1].rstrip() + "…"


class DiffOffer:
    """The difference map's own box, offered under ``Shift+C`` behind a detector box.

    What rank 1 would have been without the detector (task U3): the strongest
    unexplained blob, when the 60 %-of-the-ROI rule and the U2h gate would
    have armed it.  No "click here" cross, like rank 1 itself.
    """

    def __init__(self, box: tuple, area: int) -> None:
        self.box = tuple(float(v) for v in box)
        self.area = int(area)
        self.point = None


#: Two boxes both at most this many image pixels are "small" (a screw and the
#: difference map's blob around it), and are one candidate at a looser match.
SMALL_BOX_PX = 64 * 64
#: ... namely IoU above this, or the centre of one inside the other when the
#: two are of a size (areas within this factor): a blob four times a screw's
#: size is the screw plus something else.
SMALL_BOX_SAME_IOU = 0.5
SMALL_BOX_SIZE_RATIO = 4.0


def _same_candidate(a, b) -> bool:
    """Are two ``Shift+C`` boxes one place (round 3, review item 5)?

    IoU above :data:`~tda.ui.app_diff.ALT_DEDUP_IOU` always; for two small
    boxes also IoU above :data:`SMALL_BOX_SAME_IOU`, or the centre of one
    inside the other when they are of a size -- on D13/scan frame 40 the
    detector's screw and the difference map's box around it were IoU 0.76
    apart, and the same screw was offered three times.  A small box inside a
    much bigger one is not the same candidate: the big one is something else
    too.
    """
    if _box_iou(a, b) > ALT_DEDUP_IOU:
        return True

    def area(box) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    small, large = sorted((area(a), area(b)))
    if large > SMALL_BOX_PX or small <= 0.0:
        return False
    if _box_iou(a, b) > SMALL_BOX_SAME_IOU:
        return True
    if large > SMALL_BOX_SIZE_RATIO * small:
        return False

    def holds(box, other) -> bool:
        cx, cy = 0.5 * (other[0] + other[2]), 0.5 * (other[1] + other[3])
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    return holds(a, b) or holds(b, a)


def _covers_most(box, roi, limit: float = MAX_PROMPT_BOX_FRAC) -> bool:
    """Does ``box`` take up more than ``limit`` of the ROI's area?"""
    bw, bh = max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])
    rw, rh = max(1.0, float(roi[2] - roi[0])), max(1.0, float(roi[3] - roi[1]))
    return (bw * bh) > limit * (rw * rh)


class AssistMixin(DetectMixin):
    """The window half of the assist: SAM tools, prompt boxes, the heat map."""

    # ------------------------------------------------------------------ setup
    def _init_assist(self, sam_queue: Any) -> None:
        from tda.ui.canvas.sam_tools import SamBoxTool, SamPointTool

        self.sam_queue = sam_queue
        self._owns_queue = False
        self._sam_loading = False
        self.sam_available = sam_queue is not None
        self.sam_reason = "" if sam_queue is not None else "not loaded yet"
        #: Why SAM's last load failed, in a few words; ``""`` while it has not
        #: failed (task U6).  ``F5`` retries while it is set.
        self._sam_load_failed = ""
        #: SAM's import phase goes before the detector's model build (U6): the
        #: detector waits for this, the SAM loader releases it -- at once when
        #: a queue was handed in and this window loads no SAM of its own.
        self._sam_turn = Turn(released=sam_queue is not None)
        #: :meth:`start_sam` was called at least once in this window.
        self._sam_started = False
        #: Set when the window shuts down: a load that finishes afterwards stops
        #: its own queue instead of handing it to a window that is gone.
        self._sam_closing = threading.Event()
        if sam_queue is None:
            # Round 2: a window that never starts SAM must not keep the
            # detector waiting for the turn's ceiling (three minutes).
            QTimer.singleShot(SAM_TURN_GRACE_MS, self,
                              self._release_turn_unless_sam_started)
        self.sam_point = SamPointTool(self.canvas, None, queue=sam_queue, refine=True)
        self.sam_box = SamBoxTool(self.canvas, None, queue=sam_queue)
        #: Why the last prompt failed, until one is armed again.  Empty means
        #: the label may say "SAM ready" -- which it used to say after a submit
        #: raised, so the annotator clicked the same broken thing over and over.
        self._sam_failure = ""
        for tool in (self.sam_point, self.sam_box):
            tool.sigStroke.connect(self.on_stroke)
            tool.sigHint.connect(self.report)
            tool.sigError.connect(self.on_sam_error)
            # What the annotator has deliberately rubbed out is never put back
            # by an add-only result (ruling E1); the session owns the set.
            tool.erased_provider = self.erased_mask
        # Direct: the window takes the box off both tools before the click
        # that turned it down is submitted.
        self.sam_point.sigBoxRefused.connect(self.on_prompt_box_refused)

        self.assist = AssistController(self)
        self.assist.sigBlobs.connect(self._on_blobs)
        self.assist.sigFailed.connect(self._on_assist_failed)
        # the chassis ROI is measured on three frames of the pose segment, which
        # is three decodes and three detections: not something to spend before
        # the first paint (tda/ui/app_roi_worker.py)
        self.roi_proposer = RoiProposer(self)
        self.roi_proposer.sigProposed.connect(self._on_roi_proposed)
        self.roi_proposer.sigFailed.connect(self.report_error)
        self.assist_result: Optional[dict] = None
        #: What the comparison last asked for was *about* -- see
        #: :meth:`_assist_subject`.  A frame re-announced for any other reason
        #: (a commit, an undo, ``F5``) asks the same question again, and at
        #: 12 MP the answer costs 130 ms to arrive at twice.
        self._assist_asked: Optional[tuple] = None
        self._unexplained: dict[int, list[Box]] = {}
        #: Steps confirmed while the comparison had not landed (spec 4.4).
        self.unanalysed: set[int] = set()
        #: The diff-map box prompt of the frame, kept by the window because
        #: ``SamToolBase.detach()`` clears the tool's own copy: switching to the
        #: brush and back must not silently downgrade point+box to point-only.
        self._prompt_box: Optional[tuple[float, float, float, float]] = None
        #: Rank 1: the box :meth:`begin_add_shape` armed by itself, remembered
        #: separately from :attr:`_prompt_box` so that every reset can put it
        #: back after ``Shift+C`` has walked away from it.
        self._rank_one: Optional[tuple[float, float, float, float]] = None
        #: 0 = rank 1 = exactly today's behaviour; 1..3 index
        #: :meth:`prompt_alternates`.  It is *only* ever non-zero because the
        #: annotator pressed ``Shift+C``.
        self._prompt_rank = 0
        #: Who armed rank 1 (task U3): ``"diff"``, or ``"det:<class>"`` when the
        #: small-part detector did -- and the chip every reset puts back with it,
        #: as a ``(kind, fields)`` spec of :data:`CHIPS` (U6 round 2).
        self._rank_one_source = "diff"
        self._rank_one_chip: tuple = DIFF_CHIP
        #: The chip of the box on screen now, the same way: said again when SAM
        #: comes or goes (:meth:`refresh_prompt_chip`).
        self._prompt_chip: Optional[tuple] = None
        #: The detector's other candidates behind a detector rank 1, in dE order.
        self._det_behind: list = []
        #: Who armed the box on screen now: ``"diff"``, ``"det:<class>"`` or
        #: ``None`` (the hover names it).
        self._prompt_source: Optional[str] = None
        #: "No box when unsure" (task U2h): which views, ``k`` and the bands.
        self.prompt_gate = PG.load_prompt_gate()
        #: The rank-1 box the gate withheld on this frame, or ``None``.  Never
        #: armed or drawn; kept so that ``Shift+C``'s alternates are the ones
        #: they would have been behind it (a split piece that *is* the
        #: withheld box is not an alternative to it).
        self._withheld_box: Optional[tuple[float, float, float, float]] = None
        #: The frame on which a click outside the armed box turned it down, or
        #: ``None``.  Until the next frame visit nothing arms a box by itself
        #: there again (task U2h); ``Shift+C`` is the annotator asking.
        self._refused_on: Optional[Any] = None

        self.heat_visible = False
        self.heat_item = QGraphicsPixmapItem()
        self.heat_item.setZValue(0.5)  # over the frame, under the label overlay
        self.heat_item.setVisible(False)
        self.canvas.scene().addItem(self.heat_item)
        self._sam_loader = _SamLoader(self)
        self._sam_loader.sigLoaded.connect(self._on_sam_loaded,
                                           Qt.ConnectionType.QueuedConnection)
        # Last: the detector's worker starts loading its model now, off the
        # GUI thread, and the first frame's plan waits for it (task U3).
        self._init_detect()

    def _sam_tool(self):
        return self.sam_box if self._tool_name == "sam_box" else self.sam_point

    @S.guard
    def on_sam_error(self, text: str) -> None:
        """A SAM tool could not do what was asked; say it in both places.

        The status line is transient and the label is not: a prompt that failed
        has to leave a mark on the label, or the next click repeats it.  A
        dropped *result* (the frame moved on) is not a failure of the tool, so
        it only gets the line.
        """
        self.logger.info("SAM: %s", text)
        self.report_error(text)
        if not str(text).startswith("SAM result dropped"):
            self.note_sam_failure(text)

    def set_sam_instance(self, instance: Optional[str]) -> None:
        """Name the instance an applied mask belongs to.

        The tools have a ``"editing"`` fallback for when nothing says; relying
        on it would stamp two different edits with the same identity, so the
        window always answers explicitly.

        Changing it also throws the half-built prompt away: the points clicked
        so far were about the *previous* part, and sending them with the next
        click made every mask after the first commit a union of the two.

        The tools are told the instance's class with it (task U19): SAM picks
        its decoder head by it for a prompt without a box.  Looked up when the
        instance changes, not on every frame that re-announces the same one.
        """
        changed = any(tool.instance != instance
                      for tool in (self.sam_point, self.sam_box))
        cls = self.sam_class_of(instance) if changed else self.sam_point.instance_cls
        for tool in (self.sam_point, self.sam_box):
            if tool.instance != instance:
                tool.reset_prompt()
            tool.instance = instance
            tool.instance_cls = cls
        if changed:
            self.reset_prompt_rank()
        self.hand_prompt_box_to_tools()

    def sam_class_of(self, instance: Optional[str]) -> Optional[str]:
        """The taxonomy class of ``instance`` as SAM is told it, ``None`` when unknown.

        The card's row first (a sequence frame, the components gallery), then
        the desktop's instance table, then what the key says -- which for a
        框外零件 is the class after ``ctx.`` (task U18), ``other`` included.
        An unknown class is a small part as far as SAM's routing goes.
        """
        if not instance:
            return None
        try:
            for row in self.task_card.rows():
                if str(row.get("instance")) == str(instance) and row.get("cls"):
                    return str(row["cls"])
        except Exception:  # noqa: BLE001 - a class is never worth a failure
            pass
        try:
            rec = self.db.instances(int(self.session.desktop)).get(str(instance))
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None and getattr(rec, "cls", None):
            return str(rec.cls)
        return class_of_key(str(instance)) or None

    def reset_sam_prompt(self) -> None:
        """Forget the points, the box drag and the candidates of both SAM tools.

        Called wherever the **editing layer is replaced from outside the tool**
        -- a commit, ``Esc``, an undo/redo, a restored sidecar.  That is the one
        rule worth remembering: a prompt refines the layer it produced, so the
        moment somebody else writes that layer the prompt describes nothing.

        The armed *box* is not the prompt's but the frame's, and it stays: the
        tools are handed it back (:meth:`hand_prompt_box_to_tools`), so the
        next click carries the box the canvas is showing.
        """
        for tool in (self.sam_point, self.sam_box):
            tool.reset_prompt()
        # Somebody wrote the layer: a detector answer landing late must not
        # move the box under what they are doing now (task U3).
        self.note_prompt_touched()
        self.reset_prompt_rank()
        self.hand_prompt_box_to_tools()

    def hand_prompt_box_to_tools(self) -> None:
        """Give both SAM tools the window's armed box: the one the canvas shows.

        :meth:`~tda.ui.canvas.sam_tools.SamToolBase.reset_prompt` -- which a
        detach, an ``Esc``, an undo and an instance change all end in -- drops
        the tool's copy of the box, and the canvas kept drawing the window's.
        After ``Esc`` at rank 2 the canvas showed rank 1 and neither tool held
        it; after an undo the next click went out point-only under a box on
        screen (task U2h).  Every funnel that resets a tool ends here instead,
        so the canvas, ``S`` and ``X`` hold one box -- :attr:`_prompt_box` --
        on every path.  Setting a box touches neither the token nor the
        candidates, so this cancels nothing.
        """
        box = self._tool_box(self._prompt_box)
        for tool in (self.sam_point, self.sam_box):
            if tool.prompt_box != box:
                tool.set_prompt_box(box)

    def _tool_box(self, box: Optional[tuple]) -> Optional[tuple]:
        """The armed box as the SAM tools get it: none while a 框外零件 is drawn.

        The box is the difference map's or the detector's guess about a part
        of the machine, inside the ROI; a context object (task U18) is outside
        it, and a click on it must not be refused for landing outside a box
        that was never about it. The canvas still shows the box.
        """
        instance = getattr(self.sam_point, "instance", None)
        return None if is_context_key(instance) else box

    def clear_prompt_box(self) -> None:
        """Forget the box prompt; the next frame's diff map proposes its own.

        Including the *alternates* the annotator may have been walking: the
        difference map that produced them is about the pair of frames being
        left, and rank 1 of the next frame is the next frame's own blob.  And
        a box the gate withheld: that was the pair being left too.
        """
        self._prompt_box = None
        self._rank_one = None
        self._withheld_box = None
        self._prompt_rank = 0
        self._rank_one_source = "diff"
        self._rank_one_chip = DIFF_CHIP
        self._prompt_chip = None
        self._det_behind = []
        self._prompt_source = None
        self.canvas.set_prompt_point(None)
        self.canvas.set_rubber_band(None, kind="prompt")
        self.hand_prompt_box_to_tools()

    def reset_prompt_rank(self) -> None:
        """Put the armed box back to rank 1 -- the difference map's own offer.

        Hooked into the window-level funnels that already reset the prompt --
        :meth:`reset_sam_prompt` (a commit, ``Esc``, an undo),
        :meth:`set_sam_instance`, the pause ``Tab`` applies,
        :meth:`~tda.ui.app_roi.RoiMixin.start_roi_edit`,
        :meth:`~tda.ui.app_roi.RoiMixin.restore_pending`,
        :meth:`~tda.ui.app.MainWindow.set_mode` and
        :meth:`clear_prompt_box` (a frame or view change) -- rather than into
        the dozen actions that call them.  The rule is
        :meth:`tda.ui.canvas.sam_tools.SamToolBase.reset_prompt`'s and it is
        the same rule: a prompt describes **one part on one frame**, and an
        alternate prompt is a prompt.

        Putting rank 1's box back is itself a box change, so it goes through
        :meth:`_drop_prompt_for_new_box` as well.  Most of the funnels above
        reset the tools on their own -- but ``set_mode`` into Steps detaches
        nothing, and without this an answer to the alternate would have landed
        on the way back into Annotate.

        At rank 1 this returns without touching anything, which is what makes
        an annotator who never presses ``Shift+C`` see byte-identical
        behaviour: the whole feature is unreachable from here.

        **A refused box is not put back** (task U2h).  When a click outside the
        armed box turned it down on this frame, the rank goes to 1 with *no*
        box armed: the rule "a refused box stays off until the next frame
        visit" has no exception.  ``Tab``, a Review round trip, ``Esc``, a
        restored sidecar and an ROI re-edit all used to re-arm rank 1 here,
        which the annotator had just said was wrong -- or had walked past with
        ``Shift+C`` to the alternate they then also turned down.
        """
        if self._prompt_rank == 0:
            return
        refused = self._box_refused_here()
        self._prompt_rank = 0
        self._drop_prompt_for_new_box()
        # With rank 1's own chip: the detector's (task U3), or today's.
        self._arm_prompt_box(None if refused else self._rank_one,
                             chip=self._rank_one_chip, source=self._rank_one_source)

    def _box_refused_here(self) -> bool:
        """Has a click outside the armed box turned it down on this frame?

        The prompt's own flag (:attr:`~tda.ui.canvas.sam_tools.SamPointTool.box_refused`)
        goes with the prompt; the frame's (:attr:`_refused_on`) with the frame.
        """
        if self.sam_point.box_refused:
            return True
        return (self._refused_on is not None and compat.is_open(self.session)
                and self._refused_on == self.session.current())

    def _drop_prompt_for_new_box(self) -> str:
        """Invalidate the prompt because a **different** box is about to be armed.

        :meth:`~tda.ui.canvas.sam_tools.SamToolBase.set_prompt_box` touches
        neither the request token nor the candidates, so changing the armed box
        is the one thing that changes a prompt without going through
        :meth:`~tda.ui.canvas.sam_tools.SamToolBase.reset_prompt`.  Without
        this the answer to the *previous* box still matched the identity check
        and was painted into the editing layer while the new box and its
        crosshair were on screen, and ``C`` afterwards walked the old box's
        three masks.

        Deliberately **not** inside :meth:`_arm_prompt_box`: that is also the
        path :meth:`begin_add_shape` arms rank 1 with, and rank 1 has to stay
        byte-identical to what the difference map did before ``Shift+C``
        existed.

        The **editing layer is not touched**, which is exactly what
        ``reset_prompt`` does after ``Esc`` or an instance change: a mask that
        already landed stays as ordinary uncommitted pixels and ``Ctrl+Z``
        takes it off.  Rolling it back here is the option that loses work --
        the annotator may have brushed on top of it already, and one undoable
        step cannot tell the mask from the mask plus their strokes, which is
        the same reason ``cycle_candidate`` drops the candidates rather than
        the edit.  Returns the note the status line owes them when there was
        such a mask, and ``""`` when there was not.
        """
        applied = self._candidate_tool() is not None
        for tool in (self.sam_point, self.sam_box):
            tool.reset_prompt()
        return APPLIED_MASK_STAYS if applied else ""

    def _arm_prompt_box(self, box: Optional[tuple],
                        point: Optional[tuple] = None,
                        chip: tuple = DIFF_CHIP, source: str = "diff") -> None:
        """Put one box -- and optionally the pixel to click inside it -- on both tools.

        Both, always: ``S`` and ``X`` share the armed box, and arming only the
        tool that happens to be active is how switching to the other one
        silently downgraded the prompt to point-only.  ``None`` takes it off
        both, and off the canvas -- the one funnel for that too.  ``source``
        says who proposed it (``"diff"`` or ``"det:<class>"``) for the hover.
        ``chip`` is a ``(kind, fields)`` spec of :data:`CHIPS`, said with or
        without SAM (:meth:`chip_text`) and again whenever SAM comes or goes.
        """
        self._prompt_box = box
        self._prompt_source = None if box is None else str(source)
        self._prompt_chip = None if box is None else chip
        for tool in (self.sam_point, self.sam_box):
            tool.set_prompt_box(self._tool_box(box))
        # The difference map's box, not a drag: its own band, colour and chip
        # saying it is a guess (U2g addendum A).  The canvas keeps it off the
        # screen by itself while the ROI rectangle is being edited.
        self.canvas.set_rubber_band(box, kind="prompt", label=self.chip_text(chip))
        self.canvas.set_prompt_point(point)

    def chip_text(self, chip: tuple) -> str:
        """A chip spec as the canvas says it: the SAM wording or the no-SAM one.

        Every box keeps showing while SAM is down -- it is still where the part
        probably is, and ``Y`` / ``B`` go there -- but a box called "SAM 提示框"
        that SAM will never see is what confused the D13 annotator (U6).
        """
        kind, fields = chip
        with_sam, without = CHIPS[kind]
        return (with_sam if self.sam_available else without).format(**fields)

    def refresh_prompt_chip(self) -> None:
        """SAM came or went: the box on screen is said again, nothing else moves.

        Not the box, not the rank, not the prompt -- the box is the same box
        either way; what it is *for* changed.  Rank 1's own chip is kept as a
        spec, so a reset after this says it the new way too.
        """
        if self._prompt_box is not None and self._prompt_chip is not None:
            self.canvas.set_rubber_band(self._prompt_box, kind="prompt",
                                        label=self.chip_text(self._prompt_chip))

    @S.guard
    def on_prompt_box_refused(self, box: object) -> None:
        """The first click of a prompt landed outside the armed box: point only.

        :class:`~tda.ui.canvas.sam_tools.SamPointTool` has already left the box
        out of the click; this takes it off **both** tools and the canvas
        through :meth:`_arm_prompt_box`, so ``X``, a tool switch
        (:meth:`rearm_sam`) and the hover all agree that there is no box now.
        ``Shift+C`` still offers the alternates: it starts a new prompt.
        Nothing else arms a box on this frame again until it is left and
        visited anew (task U2h): not a late comparison, not ``Esc``, not the
        next part of the card.
        """
        self.logger.info("prompt box %s dropped: the first click was outside it",
                         None if box is None else tuple(int(round(v)) for v in box))
        # Off until the next frame visit, whatever resets the prompt meanwhile.
        self._refused_on = (self.session.current() if compat.is_open(self.session)
                            else None)
        self.note_prompt_touched()
        # And the walk starts again (task U2i): a refused *alternate* left the
        # rank at 2 with nothing armed, and the next Shift+C skipped to 3.
        self._prompt_rank = 0
        self._arm_prompt_box(None)
        self.report(BOX_REFUSED, hold_ms=BOX_REFUSED_HOLD_MS)

    def rearm_sam(self) -> None:
        """Give a freshly attached SAM tool its frame token and prompt box back.

        ``detach()`` cancels the in-flight prompt and clears the box, which is
        what makes a tool switch safe; the cost is that re-arming has to be
        explicit, or the next click would go out point-only.
        """
        tool = self._sam_tool()
        if not self.sam_available or self._tool_name not in ("sam_point", "sam_box"):
            return
        self.clear_sam_failure()     # arming a SAM tool is "try again"
        if compat.is_open(self.session) and self.session.image() is not None:
            tool.set_frame_token(self.session.current())
        if self._prompt_box is not None:
            tool.set_prompt_box(self._prompt_box)

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_assist(self, key) -> None:
        """Re-stamp the SAM tools and start the comparison for the new frame.

        The token is **mandatory**: without one the tools refuse to prompt, so a
        frame with no image gets ``None`` (prompting is meaningless there) and
        every other frame gets its :class:`~tda.core.model.FrameKey`.
        """
        image = self.session.image()
        token = key if image is not None else None
        if self._refused_on is not None and self._refused_on != key:
            # A visit to another frame: the box turned down was that frame's.
            # The same frame announced again (a commit, F5) is not a visit.
            self._refused_on = None
        self.clear_prompt_box()
        for tool in (self.sam_point, self.sam_box):
            tool.overlay = self.overlay
            # The frame to crop from, explicitly: the canvas may be showing the
            # neighbour (``Tab``) and a prompt must never be about that one.
            tool.image = image
            tool.set_frame_token(token)
            # set_frame_token drops the box only when the token really changes,
            # and re-arming an attached tool may already have set it: say it.
            tool.set_prompt_box(None)
        self.set_sam_instance(getattr(self.session, "editing_instance", None))
        # Both bands: the armed box and any drag were about the frame just left.
        self.canvas.set_rubber_band(None)
        self.canvas.set_rubber_band(None, kind="prompt")
        # The detector's pass (task U3): a dictionary lookup when this frame's
        # answer is on hand, a priority request when it is not -- never a wait.
        self.on_frame_changed_detect(key)
        if self._assist_subject() == self._assist_asked:
            # The same two frames, inside the same ROI: the difference between
            # them cannot have changed, so the comparison on hand (or the one
            # on its way) is still the answer.  A **commit** arrives here --
            # the session re-announces the frame -- and re-running a 12 MP
            # comparison because an annotation changed put 130 ms of waiting
            # into the ``Space`` that followed. What the annotation changes is
            # which blobs are *explained*, and that is a re-split of blobs
            # already in hand.
            #
            # Everything :meth:`_on_blobs` does when a comparison lands has to
            # be done here too, and for the same reason: the lines above have
            # just cleared the prompt box, both SAM tools' copies and the
            # rubber band, and on this path nothing is coming back to put them
            # there again. Without it every SAM click *after a commit* went out
            # point-only -- the configuration that returned the whole chassis
            # on 7 of 13 real frames -- and ``Shift+A`` lost the difference box
            # it falls back to.
            self._settle_assist()
            return
        self.assist_result = None
        self.heat_item.setVisible(False)
        self.request_assist()

    def _settle_assist(self) -> None:
        """Re-split this frame's blobs and show what they mean.

        The two things a comparison's arrival is good for once the blobs
        themselves are known: the heat map under ``D``, and the box prompt the
        task card's open item is armed with.  Shared by :meth:`_on_blobs`, for
        a comparison that has just landed, and by the frame hook, for one that
        landed earlier and is still the answer.
        """
        payload = self.re_explain()
        if self.heat_visible and payload is not None:
            self._paint_heat()
        self._arm_from_card()

    def _assist_subject(self) -> Optional[tuple]:
        """What a comparison would be *about*: the frame, its neighbour, the ROI.

        Everything the difference map reads, and nothing the annotator can
        change by drawing.  ``None`` when there is no frame to compare.
        """
        if not compat.is_open(self.session):
            return None
        return (self.session.current(), compat.task_neighbour(self.session),
                self.roi())

    def request_assist(self) -> None:
        """Compare the open frame with its task-card neighbour, off the GUI thread.

        The neighbour is the frame the card is written against -- ``j + 1``,
        the one the annotator came from -- so a blob marks the part that
        *re-appears* in the image on screen, which is exactly the part the card
        is asking for.
        """
        if not compat.is_open(self.session):
            return
        key = self.session.current()
        image = self.session.image()
        neighbour = compat.task_neighbour(self.session)
        self.report_neighbour_gap(key, neighbour)
        previous = None if neighbour is None else self._neighbour_pixels(neighbour)
        self._assist_asked = self._assist_subject()
        self.assist.request(key, image, previous, self.roi(), self.expected_now())

    def _neighbour_pixels(self, neighbour: int):
        """The neighbour frame for the comparison: its pixels, or its path.

        Stepping back one frame, ``k+1`` is the frame just left and is already
        decoded, so the worker gets the array.  A timeline click lands on a
        frame whose neighbour nobody has opened, and decoding 12 MP of it on
        the GUI thread -- 46 ms, purely to hand a worker something it could
        have read itself -- was a tenth of the click.  The path goes instead;
        :func:`tda.ui.app_diff._pixels_of` reads it on the worker.
        """
        held = compat.peek_image_at(self.session, neighbour)
        if held is not None:
            return held
        path = getattr(self.session, "image_path", None)
        found = path(neighbour) if callable(path) else None
        return found or self.session.image_at(neighbour)

    def report_neighbour_gap(self, key, neighbour: Optional[int]) -> None:
        """Say so when the neighbour is not the adjacent step.

        ``task_neighbour`` skips a step this view never photographed, and then
        the card spans more than one action -- which changes what the annotator
        is being asked to draw, so it has to be visible rather than inferred
        from a gap in the timeline.
        """
        if neighbour is None or abs(int(neighbour) - int(key.step)) == 1:
            return
        self.report(f"本视图缺帧：任务卡对照的是第 {neighbour} 帧，跨了 "
                    f"{abs(int(neighbour) - int(key.step))} 个步骤 / the card is "
                    f"diffed against step {neighbour}, not the adjacent one")

    def expected_now(self) -> dict:
        """What the frame already accounts for, as the worker wants it.

        A blob is explained when it sits on an instance whose geometry here
        differs from the neighbour's -- i.e. on one of the task card's own
        instances (and the children that come out with them) that already has a
        shape on this frame.  Reading it from the *visible masks* rather than
        from the stored ``box`` column is the whole fix: only bench rows carry a
        box, so the old version was always empty and nothing was ever explained.
        """
        wanted = self._card_instances()
        masks, _order, _windows = compat.overlay_layers(self.session)
        chosen = {key: mask for key, mask in masks.items() if key in wanted}
        boxes = []
        for key, inst in self.session.compiled().instances.items():
            if key in wanted and inst.visible is None and inst.box:
                boxes.append(tuple(int(round(float(v))) for v in inst.box))
        return expected_payload(chosen, boxes)

    def _card_instances(self) -> set[str]:
        """The task card's instances plus the children attached to them."""
        from tda.core.truth_inputs import instances_of

        wanted = {str(row["instance"]) for row in self.session.task_card()
                  if row.get("instance")}
        try:
            records = instances_of(self.db, int(self.session.desktop))
        except Exception:  # noqa: BLE001 - the card alone is still usable
            return wanted
        wanted |= {key for key, rec in records.items()
                   if getattr(rec, "attached", False) and rec.parent in wanted}
        return wanted

    # --------------------------------------------------------------- results
    @S.guard
    def _on_assist_failed(self, text: str) -> None:
        """The comparison raised on the worker: say so, and do not call it asked.

        ``_assist_asked`` is what stops a commit asking the same question
        twice; a question that *raised* has not been answered, so the stamp
        comes off and the next re-announcement -- a commit, ``F5`` -- asks
        again, which is what main did with no stamp at all.
        """
        self._assist_asked = None
        self.report_error(text)

    @S.guard
    def _on_blobs(self, payload: object) -> None:
        """A comparison came back on the GUI thread."""
        self.assist_result = payload if isinstance(payload, dict) else None
        if self.assist_result is None:
            # No comparison to be had -- no neighbour, or one that could not be
            # read. Not an answer, so it is not remembered as one.
            self._assist_asked = None
            return
        if not compat.is_open(self.session):
            return
        if self.assist_result.get("key") != self.session.current():
            self.assist_result = None
            return
        self._settle_assist()

    def re_explain(self) -> Optional[dict]:
        """Re-split the blobs of this frame against what it holds *now*.

        Called after every commit and again at confirm time: a part that has
        just been drawn stops being an unexplained difference, and the review
        queue must not be told about it.
        """
        payload = self.assist_result
        if not payload or payload.get("key") != self.session.current():
            return payload
        boxes = expected_boxes(self.expected_now())
        explained, unexplained = explain_blobs(list(payload.get("blobs") or []), boxes)
        payload["explained"], payload["unexplained"] = explained, unexplained
        payload["expected"] = boxes
        return payload

    def unexplained_at_confirm(self) -> Optional[list[Box]]:
        """Boxes still unexplained on this frame, or ``None`` when not analysed.

        ``None`` is not the same as an empty list: an empty list says "this
        frame was compared and everything is accounted for", while ``None`` says
        "nobody looked", which the review queue has to be able to tell apart.
        """
        if not self._assist_ready():
            return None
        payload = self.re_explain()
        return blob_boxes((payload or {}).get("unexplained") or [])

    def _assist_ready(self, timeout: float = ASSIST_CONFIRM_WAIT) -> bool:
        """Make sure this frame's comparison has landed, within a short cap."""
        key = self.session.current()
        if (self.assist_result or {}).get("key") == key:
            return True
        if not self.assist.wait(timeout):
            return False
        QApplication.processEvents()          # let the queued result through
        return (self.assist_result or {}).get("key") == key

    def _arm_from_card(self) -> None:
        """Offer the strongest unexplained blob when the card wants a shape drawn.

        Or the small-part detector's best candidate, when it has one for this
        frame (task U3) -- even if the difference map found nothing.
        """
        rows = self.session.task_card()
        index = self.task_card.current_index()
        if not (0 <= index < len(rows)) or rows[index].get("kind") != api.KIND_ADD_SHAPE:
            return
        blob = best_unexplained(self.assist_result)
        if blob is not None or self._detector_offers(rows):
            self.begin_add_shape(blob, rows=rows)

    def arm_prompt_box_for(self, instance: str) -> None:
        """Arm the box prompt for the instance the annotator actually activated.

        ``_on_blobs`` can only guess from the card's current row, and the
        annotator may well have double-clicked three rows further down; this is
        called from ``on_request_edit`` with the answer.
        """
        payload = self.assist_result
        if not payload or payload.get("key") != self.session.current():
            return
        card = self.session.task_card()
        rows = {str(r.get("instance")): r for r in card}
        row = rows.get(str(instance))
        if row is None or row.get("kind") != api.KIND_ADD_SHAPE:
            return
        blob = best_unexplained(payload)
        if blob is not None or self._detector_offers(card):
            self.begin_add_shape(blob, rows=card)

    def _detector_offers(self, rows: Optional[list]) -> bool:
        """Might the small-part detector have a rank 1 here (task U3)?  Cheap, a superset.

        Only whether the frame's answer holds a candidate of a class the card
        asks for; :meth:`_arm_from_detector` ranks and skips, and falls through
        to the difference map when nothing survives.
        """
        config = self.det_config
        if self.det_worker is None or config is None or not self._det_frames:
            return False
        try:
            if not compat.is_open(self.session):
                return False
            key = self.session.current()
            frame = self._det_frames.get(int(key.step))
            if frame is None or not config.applies(key.view) or not frame.change:
                return False
            asked = detector_asked(rows or [])
            return any(frame.dets[i].cls in asked for i in frame.change)
        except Exception:  # noqa: BLE001 - the difference map's box must still come
            self.logger.exception("detector: the frame's answer could not be read")
            return False

    def begin_add_shape(self, blob: Optional[DiffBlob], rows: Optional[list] = None) -> None:
        """Feed a changed region's box to SAM as the box half of point+box.

        Only once the segment has a stored ROI.  Without one the difference map
        covers the whole frame, and its strongest region is as likely to be a
        scan-bed artefact at the edge as the part being drawn -- which is
        exactly what happened on 7 of 13 real frames.

        Not on a view :mod:`tda.ui.prompt_gate` is on for when the blob is not
        the size of any part the card asks to add back (task U2h): no box
        rather than a likely-wrong one.  ``rows`` is the task card the caller
        already read; ``None`` reads it, and only on a gated view.

        **The small-part detector goes first** (task U3): when the card's open
        ✚ rows ask for a class it is configured for -- rows that come back on
        their own, not with a parent (round 2: captive cooler screws) -- on one
        of its views, and a candidate survives the "already drawn" skip, rank 1
        is its best candidate -- the gate does not judge it, and ``blob`` may
        be ``None``.
        Otherwise everything below runs exactly as it did before it existed.
        """
        roi = self.roi()
        if roi is None:
            return
        if is_context_key(getattr(self.session, "editing_instance", None)):
            # a 框外零件 is being drawn (task U18): the guesses are about the
            # machine's parts, inside the ROI -- none is offered for it
            return
        if self._box_refused_here():
            # A click outside the armed box has already turned it down on this
            # frame; a comparison landing late, an ``Esc`` or the next part of
            # the card must not put one back under the annotator's next click
            # (U2g addendum B; until the next frame visit, U2h).
            return
        if self._arm_from_detector(rows):
            return
        if blob is None:
            return
        box = tuple(float(v) for v in blob.box)
        if _covers_most(box, roi):
            # "Everything changed" is not a prompt: it narrows nothing, and it
            # is exactly when SAM's single answer came back as the whole
            # chassis.  A point on its own does better.
            self.clear_prompt_box()
            for tool in (self.sam_point, self.sam_box):
                tool.set_prompt_box(None)
            self.logger.info("the changed region covers most of the ROI: no prompt "
                             "box, point-only (%d px)", int(blob.area))
            self.report(PROMPT_TOO_BIG)
            return
        if self._withhold(blob, box, rows):
            return
        # This is rank 1 by definition: the difference map arming its own blob.
        # Whatever ``Shift+C`` was walking belonged to the previous offer.
        self._rank_one = box
        self._withheld_box = None
        self._prompt_rank = 0
        self._rank_one_source = "diff"
        self._rank_one_chip = DIFF_CHIP
        self._det_behind = []
        self._arm_prompt_box(box)
        self.logger.info("prompt box from the difference map: %s (%d px)",
                         tuple(int(v) for v in blob.box), int(blob.area))
        # "点 S" only when there is a SAM to click for (U6 round 2)
        self.report(PROMPT_ARMED if self.sam_available else PROMPT_ARMED_NO_SAM)

    def _arm_from_detector(self, rows: Optional[list]) -> bool:
        """Arm the detector's best candidate as rank 1 (task U3); ``True`` if it did.

        No gate, no 60 %-of-the-ROI rule: those are about the difference map's
        blob, and this is a box the size of the part.  One log line per arming
        with every candidate's box, confidence and dE -- to validate a dE
        threshold later, never to apply one now.  A failure here is logged with
        its traceback and the difference map arms as before (round 3).
        """
        if self.det_worker is None:
            return False
        try:
            ranking = self.detector_ranking(rows)
        except Exception:  # noqa: BLE001 - the difference map's box must still come
            self.logger.exception("detector: ranking failed; the difference map's box")
            return False
        if ranking is None or not ranking.chosen:
            return False
        top = ranking.chosen[0]
        name = class_zh(top.cls)
        self._rank_one = top.box
        self._withheld_box = None
        self._prompt_rank = 0
        self._rank_one_source = f"det:{top.cls}"
        self._rank_one_chip = ("det", {"name": name})
        self._det_behind = list(ranking.chosen[1:])
        self._arm_prompt_box(top.box, chip=self._rank_one_chip,
                             source=self._rank_one_source)
        key = ranking.key
        self.logger.info(
            "prompt box from the detector%s: D%s/%s step %s (dE against step %s) chose "
            "%s conf %.3f dE %.2f; candidates (box, conf, dE) %s; skipped as drawn %s",
            " (late answer)" if getattr(self, "_det_arming_late", False) else "",
            key.desktop, key.view, key.step, ranking.neighbour, top.int_box, top.conf,
            top.change, [c.describe() for c in ranking.chosen],
            [c.describe() for c in ranking.skipped])
        armed = PROMPT_ARMED_DET if self.sam_available else PROMPT_ARMED_DET_NO_SAM
        self.report(armed.format(name=name, cls=top.cls))
        return True

    def det_chip(self, cls: str) -> str:
        """The detector box's chip: :data:`DET_CHIP`, or the no-SAM one (task U6)."""
        return self.chip_text(("det", {"name": class_zh(cls)}))

    def no_box_note(self) -> str:
        """:data:`EDITING_NO_BOX` while the gate's withholding is what is on the frame."""
        if self._withheld_box is not None and self._prompt_box is None:
            return EDITING_NO_BOX
        return ""

    def _gate_verdict(self, blob: DiffBlob, rows: Optional[list]) -> tuple:
        """``(verdict, the open ✚ rows, view)``; the verdict is ``None`` to keep the box."""
        gate = self.prompt_gate
        view = getattr(self.session.current(), "view", None)
        if not gate.applies(view):
            return None, [], view
        card = self.session.task_card() if rows is None else rows
        wanted = [r for r in card if r.get("kind") == api.KIND_ADD_SHAPE
                  and not r.get("done") and r.get("instance")]
        classes = [str(r.get("cls") or str(r["instance"]).split(".", 1)[0])
                   for r in wanted]
        return gate.judge(view, classes, int(blob.area)), wanted, view

    def _withhold(self, blob: DiffBlob, box: tuple, rows: Optional[list]) -> bool:
        """Withhold the box when the gate says so (task U2h); ``True`` if it did.

        The classes are those of the card's open ✚ rows -- what this frame asks
        to add back -- and the box stays if it fits any one of their bands.
        Withheld means withheld everywhere: nothing armed on either tool,
        nothing on the canvas, and :meth:`_frame_diff_box` offers it to no
        hover.  The heat map and ``Shift+C`` stay.
        """
        gate = self.prompt_gate
        verdict, wanted, view = self._gate_verdict(blob, rows)
        if verdict is None:
            return False
        self.clear_prompt_box()
        self._withheld_box = box
        names = list(dict.fromkeys(
            class_zh(r.get("cls"), r.get("attrs"), str(r["instance"])) for r in wanted))
        self.logger.info("prompt box withheld: blob %s (%d px) is %.1fx outside the "
                         "area band of %s on %s (k=%g)",
                         tuple(int(v) for v in blob.box), int(blob.area),
                         verdict.factor, "/".join(verdict.classes), view, gate.k)
        self.report(PROMPT_WITHHELD.format(name=WITHHELD_JOIN.join(names)))
        return True

    # ------------------------------------------------- alternate box prompts
    def prompt_rank(self) -> int:
        """Which box prompt is armed, 1-based; 1 is the difference map's own."""
        return self._prompt_rank + 1

    def prompt_alternates(self) -> list:
        """The split proposals ``Shift+C`` offers behind the armed box.

        Empty unless a comparison **of the open frame** is on hand: a payload
        whose key has been superseded is an answer about a frame that is no
        longer on screen, and arming a box out of it would put the previous
        pair's geometry on this one.  The 60 %-of-the-ROI rule is applied here
        as well as inside the splitter, because it belongs to the prompt rather
        than to the proposal: "everything changed" is not an alternative to
        anything.
        """
        if self._rank_one_source.startswith("det"):
            return self._alternates_behind_detector()
        payload = self.assist_result
        if not compat.is_open(self.session) or not payload:
            return []
        if payload.get("key") != self.session.current():
            return []
        roi = self.roi()
        if roi is None:
            return []
        inside = [part for part in (payload.get("proposals") or [])
                  if not _covers_most(tuple(float(v) for v in part.box), roi)]
        # Behind a withheld rank 1 the alternates are the ones they would have
        # been behind it: the withheld box spelled twice is not an alternative.
        armed = self._rank_one if self._rank_one is not None else self._withheld_box
        return alternate_parts(inside, armed)

    def _alternates_behind_detector(self) -> list:
        """``Shift+C`` behind a detector rank 1 (task U3): its other candidates first.

        In dE order, then what the difference map offers: its own box -- what
        rank 1 would have been without the detector, when the 60 % rule and the
        gate would have armed it (:class:`DiffOffer`) -- and its split
        proposals behind that, exactly as they would stand behind it.  A diff
        box that repeats one of the detector's, or one already offered
        (:func:`_same_candidate`), is not offered twice.
        """
        out: list = list(self._det_behind)
        taken = [self._rank_one] + [c.box for c in out]
        payload = self.assist_result
        if (not compat.is_open(self.session) or not payload
                or payload.get("key") != self.session.current()):
            return out
        roi = self.roi()
        if roi is None:
            return out
        diff: list = []
        own = withheld = None
        blob = best_unexplained(payload)
        if blob is not None:
            box = tuple(float(v) for v in blob.box)
            if not _covers_most(box, roi):
                if self._gate_verdict(blob, None)[0] is None:
                    own = box
                    diff.append(DiffOffer(box, int(blob.area)))
                else:
                    withheld = box
        inside = [part for part in (payload.get("proposals") or [])
                  if not _covers_most(tuple(float(v) for v in part.box), roi)]
        diff += alternate_parts(inside, own if own is not None else withheld)
        for part in diff:
            box = tuple(float(v) for v in part.box)
            if any(t is not None and _same_candidate(box, t) for t in taken):
                continue
            out.append(part)
            taken.append(box)
        return out

    def _alternate_refusal(self) -> str:
        """Why ``Shift+C`` can do nothing right now, or ``""``.

        The same shape as ``Shift+A``'s refusal list and for the same reason:
        each of these already owns the canvas or the two keys that would settle
        what is on it, and a second owner is how a rectangle gets stored by a
        press meant for something else.
        """
        if self.mode != "annotate":
            return ALT_WRONG_MODE
        if self.is_flashing():
            return ALT_FLASHING
        if self.roi_editing:
            return ALT_ROI_EDITING
        if self.showing_draft_ghost():
            return ALT_GHOST
        if is_context_key(getattr(self.session, "editing_instance", None)):
            return NO_PROMPT_BOX_CONTEXT       # the guesses are not about it (U18)
        if not self.sam_available:
            # The box prompt exists to be sent to SAM.  ``act_cycle_candidate``
            # says the reason rather than going quiet, and so does this.
            return f"SAM 用不了：{self.sam_reason} / SAM unavailable: {self.sam_reason}"
        return ""

    @S.guard
    def act_cycle_prompt_box(self) -> None:
        """``Shift+C``: the next box prompt, from the split difference map.

        ``C`` cycles SAM's three answers to one prompt; this cycles the
        *prompt*.  Rank 1 is the blob the difference map armed by itself and is
        never replaced -- on 201 real removal events the split proposal was a
        large win on parts above 100 px (median SAM IoU 0.320 -> 0.762) and a
        bad loss on nine large parts it over-split, the CPU cooler landing on
        the fan hub (0.870 -> 0.131).  Behind a key the annotator presses when
        the default box is visibly wrong, that trade is all upside.

        When the small-part detector armed rank 1 (task U3) the walk is its
        other candidates in dE order first, then the difference map's own box
        and its split proposals (:meth:`_alternates_behind_detector`).
        """
        refusal = self._alternate_refusal()
        if refusal:
            self.report(refusal)
            return
        # The annotator asked for another box: a detector answer landing late
        # must not replace whatever they are about to walk to (task U3).
        self.note_prompt_touched()
        alternates = self.prompt_alternates()
        if not alternates:
            self.report(NO_ALTERNATE)
            return
        # A new box is a new prompt: whatever is in flight for the old one must
        # not land, and its candidates are not this box's answers.
        note = self._drop_prompt_for_new_box()
        total = len(alternates) + 1
        self._prompt_rank = (self._prompt_rank + 1) % total
        chip = ("diff_rank", {"rank": self._prompt_rank + 1, "total": total})
        if self._prompt_rank == 0 and self._rank_one_source.startswith("det"):
            name = class_zh(self._rank_one_source.split(":", 1)[1])
            self._arm_prompt_box(self._rank_one, source=self._rank_one_source,
                                 chip=("det_rank", {"rank": 1, "total": total,
                                                    "name": name}))
            self.report(f"提示框 1/{total}：检测器找到的那个{name}（差别最大） / prompt box "
                        f"1/{total}: the detector's own{note}")
            return
        if self._prompt_rank == 0:
            self._arm_prompt_box(self._rank_one, chip=chip)
            if self._rank_one is None:
                # Rank 1 was no box at all -- withheld (U2h) or the whole ROI.
                self.report(f"提示框 1/{total}：这次不给框，直接在零件上点 / "
                            f"prompt box 1/{total}: none this time, click the "
                            f"part{note}")
                return
            self.report(f"提示框 1/{total}：差异图原本给的那块 / "
                        f"prompt box 1/{total}: the difference map's own{note}")
            return
        part = alternates[self._prompt_rank - 1]
        box = tuple(float(v) for v in part.box)
        rank = self._prompt_rank + 1
        if isinstance(part, DetCandidate):
            name = class_zh(part.cls)
            self._arm_prompt_box(box, chip=("det_rank", {"rank": rank, "total": total,
                                                         "name": name}),
                                 source=f"det:{part.cls}")
            self.logger.info("prompt box %d/%d from the detector: %s conf %.3f dE %.2f",
                             rank, total, part.int_box, part.conf, part.change)
            self.report(f"提示框 {rank}/{total}：检测器找到的另一个{name}，对就在零件上点 S / "
                        f"prompt box {rank}/{total}: another {part.cls} the detector "
                        f"found: click the part{note}")
            return
        self._arm_prompt_box(box, part.point, chip=chip)
        self.logger.info("prompt box %d/%d box=%s point=%s area=%d",
                         self._prompt_rank + 1, total,
                         tuple(int(v) for v in part.box), part.point, part.area)
        if isinstance(part, DiffOffer):
            self.report(f"提示框 {rank}/{total}：差异图原本给的那块 / prompt box "
                        f"{rank}/{total}: the difference map's own{note}")
            return
        # Where and how big is in the log line above; the status line says
        # what to do (U2g addendum C).
        self.report(f"提示框 {self._prompt_rank + 1}/{total}：差异图拆出的另一块，"
                    f"在十字处点一下 / prompt box {self._prompt_rank + 1}/{total} "
                    f"from the split difference map: click the cross{note}")

    def unexplained_boxes(self) -> list[Box]:
        """Boxes of the changes nothing on this frame accounts for."""
        return blob_boxes((self.assist_result or {}).get("unexplained") or [])

    def hand_over_unexplained(self, step: int, boxes: Optional[list[Box]]) -> None:
        """Give the review queue what the frame never explained (spec 4.4).

        ``None`` means the comparison never finished: the step is recorded as
        not analysed instead of being filed as "nothing to see", which would
        quietly claim a check that never happened.
        """
        if boxes is None:
            self.unanalysed.add(int(step))
            self.report_error(f"step {step}: the difference map did not finish; "
                              f"it is recorded as not analysed")
            return
        self.unanalysed.discard(int(step))
        if not compat.set_unexplained(self.session, step, boxes):
            self._unexplained[int(step)] = list(boxes)

    # ------------------------------------------------------------ SAM status
    def sam_status_text(self) -> str:
        """The SAM part of the status bar: state plus the candidate counter.

        The counter follows whichever tool actually *has* candidates, not the
        armed one: picking up the brush to tidy a proposal must not make the
        ``2/3`` disappear while ``C`` would still work.
        """
        if self._sam_load_failed and not self._sam_loading:
            return f"SAM 加载失败：{self._sam_load_failed}（F5 重试）"
        if not self.sam_available:
            return f"SAM unavailable: {self.sam_reason}"
        if self._sam_failure:
            # A failed prompt with "SAM ready" still on the label is how an
            # annotator clicks the same broken thing twenty times.
            return f"SAM error: {self._sam_failure} (S to retry)"
        tool = self._candidate_tool()
        if tool is not None:
            return (f"SAM ready · {tool.candidate_index + 1}/{tool.candidate_count}"
                    f"{self.sam_head_note()}")
        return f"SAM ready{self.sam_head_note()}"

    def sam_head_reason(self) -> str:
        """Why SAM's fine-tuned head is off (task U19), ``""`` when it is on.

        Also ``""`` when the queue does not say -- one handed in by a test or
        a smoke run, which never asked for a head.
        """
        service = getattr(self.sam_queue, "service", None)
        reason = getattr(service, "head_off_reason", "")
        return reason if isinstance(reason, str) else ""

    def sam_head_note(self) -> str:
        """:data:`HEAD_OFF_NOTE` while the fine-tuned head is off, else ``""``."""
        return HEAD_OFF_NOTE if self.sam_head_reason() else ""

    def note_sam_failure(self, text: str) -> None:
        """Remember that the last prompt failed, until the next one is armed."""
        self._sam_failure = str(text).strip()[:60]
        self.update_status()

    def clear_sam_failure(self) -> None:
        """``S`` (or any fresh arming) means "try again"."""
        if self._sam_failure:
            self._sam_failure = ""
            self.update_status()

    def _candidate_tool(self):
        """The SAM tool holding candidates right now, preferring the armed one."""
        armed = self._sam_tool()
        for tool in (armed, self.sam_point, self.sam_box):
            if tool.candidate_count:
                return tool
        return None

    def set_sam_unavailable(self, reason: str) -> None:
        """Disable the SAM tools and say why; everything else keeps working."""
        self.sam_available = False
        self.sam_reason = str(reason)
        for tool in (self.sam_point, self.sam_box):
            tool.queue = None
        if self._tool_name in ("sam_point", "sam_box"):
            self._tool_name = "brush"
            self._attach_tool()
        if not self._sam_loading:
            # SAM is off and nothing is loading it: the detector's build has
            # nobody to wait for (U6 round 2)
            self._sam_turn.release()
        self.refresh_prompt_chip()
        self.update_status()

    def set_sam_queue(self, queue: Any, owns: bool = True) -> None:
        """Arm the SAM tools once a checkpoint has finished loading.

        A queue handed in from outside -- a smoke run, a test -- is also this
        window deciding it loads no SAM of its own, so the detector's build
        stops waiting for one (U6 round 2).
        """
        self.sam_queue = queue
        self._owns_queue = bool(owns)
        self.sam_available = queue is not None
        self.sam_reason = ""
        for tool in (self.sam_point, self.sam_box):
            tool.queue = queue
        self._sam_turn.release()
        self.refresh_prompt_chip()
        self.update_status()

    def decline_sam(self, reason: str = "not loaded in this window") -> None:
        """This window will load no SAM of its own (``--no-sam``, a probe, a test).

        The detector's model build waits for SAM's import phase (U6); a window
        that will never start one says so here and the build goes ahead at
        once, instead of after :data:`~tda.models.load_gate.TURN_TIMEOUT_S`.
        Nothing else changes: SAM can still be loaded later with
        :meth:`start_sam` or :meth:`set_sam_queue`.
        """
        self._sam_turn.release()
        if self.sam_queue is None and not self._sam_loading:
            self.sam_reason = str(reason)
            self.update_status()
        self.logger.info("SAM: not loaded in this window (%s); the detector does "
                         "not wait for it", reason)

    def _release_turn_unless_sam_started(self) -> None:
        """The grace timer: no SAM load started -- the detector need not wait.

        ``tda.cli app`` calls :meth:`start_sam` right after ``show()``, long
        before this fires; a window built by anything else that never starts
        one would otherwise keep the detector off for the turn's ceiling.
        """
        if self.closed or self._sam_started or self.sam_queue is not None:
            return
        if not self._sam_turn.released:
            self.decline_sam("no SAM load was started")

    def start_sam(self) -> None:
        """Load SAM on a background thread; the first frame must not wait for it.

        The import and the model build run inside
        :func:`~tda.models.load_gate.heavy` -- the one lock the detector's
        loader holds too -- and the window's turn is released the moment they
        are over, whichever way, so the detector builds after SAM (task U6:
        on 2026-10-12 the two imports deadlocked and SAM was off all day).
        """
        if self.sam_queue is not None or self._sam_loading:
            return
        self._sam_started = True
        self._sam_loading = True
        self._sam_load_failed = ""
        self.sam_reason = "loading"
        self.logger.info("SAM: loading the checkpoint")
        self.update_status()
        loader = self._sam_loader
        paths = dict(self.paths)
        turn = self._sam_turn
        closing = self._sam_closing

        def load() -> None:
            try:
                with heavy("sam"):
                    from tda.models import sam_heads, sam_service

                    # The paths the app was *started* with, not the repo's own
                    # configs/paths.yaml: --paths was being ignored for the weights.
                    checkpoint = sam_service.checkpoint_in(paths)
                    # Task U19: the fine-tuned head, or a WARNING and the stock
                    # head alone -- configs/sam.yaml (TDA_SAM_HEAD overrides); a
                    # head named without a directory is in the same weights_dir.
                    heads = sam_heads.load_sam_head_config(
                        weights_dir=paths.get("weights_dir"))
                    service = (sam_service.SamService(checkpoint=str(checkpoint), heads=heads)
                               if checkpoint is not None
                               else sam_service.SamService(heads=heads))
                outcome: object = sam_service.SamQueue(service)
            except Exception as exc:  # noqa: BLE001 - the app works without SAM
                outcome = f"{type(exc).__name__}: {exc}"
            finally:
                turn.release()          # loaded or failed: the detector's turn
            if closing.is_set():
                # The window shut while this loaded (round 2): nobody will take
                # the queue, and one nobody stops keeps a thread and a CUDA
                # context alive for the life of the process.
                stop = getattr(outcome, "stop", None)
                if callable(stop):
                    stop()
                return
            loader.sigLoaded.emit(outcome)

        self._sam_loader_thread = threading.Thread(target=load, name="tda-sam-load",
                                                   daemon=True)
        self._sam_loader_thread.start()

    @S.guard
    def _on_sam_loaded(self, outcome: object) -> None:
        """The background load finished -- possibly after the window closed."""
        self._sam_loading = False
        if isinstance(outcome, str):
            self.logger.info("SAM: unavailable (%s)", outcome)
            # Said on the badge as a failure with its key, not as "not loaded
            # yet" for the rest of the session (task U6).
            self._sam_load_failed = short_reason(outcome)
            self.set_sam_unavailable(outcome)
            self.report(self.sam_status_text())    # the badge keeps saying it
            return
        if self.closed:
            # Nobody is going to use it: a queue nobody stops keeps a thread and
            # a CUDA context alive for the life of the process.
            outcome.stop()
            return
        self._sam_load_failed = ""
        self.set_sam_queue(outcome, owns=True)
        heads = getattr(getattr(outcome, "service", None), "heads", None)
        why = self.sam_head_reason()
        self.logger.info("SAM: ready (heads: %s%s)", ", ".join(heads or ("stock",)),
                         f"; fine-tuned head off: {why}" if why else "")
        self.report(f"SAM ready{self.sam_head_note()}")

    def retry_sam(self) -> bool:
        """``F5`` after a failed load: one more load, once per press (task U6).

        ``False`` -- and nothing started -- unless the last load failed and no
        load is running: a press during a load does not start a second one,
        and a SAM that loaded is not reloaded.
        """
        if not self._sam_load_failed or self._sam_loading or self.sam_queue is not None:
            return False
        self.logger.info("SAM: retrying the load (F5) after: %s", self._sam_load_failed)
        self.start_sam()
        self.report("正在重新加载 SAM… / loading SAM again")
        return True

    def sam_failure_reason(self) -> str:
        """Why SAM's last load failed, in a few words; ``""`` unless it did."""
        return self._sam_load_failed

    def sam_refusal(self) -> str:
        """What ``S`` / ``X`` -- and their buttons -- say when SAM's load failed."""
        return f"{G_DRAW_NO_SAM}（SAM 加载失败：{self._sam_load_failed}；F5 重试）"

    # -------------------------------------------------------------- actions
    @S.guard
    def act_cycle_candidate(self) -> None:
        """``C``: the next of SAM's proposals for the same click."""
        tool = self._candidate_tool()
        if tool is None or tool.candidate_count < 2:
            self.report("no other SAM candidate"
                        + ("" if self.sam_available else f": {self.sam_reason}"))
            return
        tool.cycle_candidate()
        self.update_status()

    @S.guard
    def act_toggle_heat(self) -> None:
        """``D``: the frame-difference heat map over the image."""
        self.heat_visible = not self.heat_visible
        if self.heat_visible:
            self._paint_heat()
        else:
            self.heat_item.setVisible(False)
        self.report("difference heat map " + ("on" if self.heat_visible else "off"))
        self.refresh_guidance()     # the palette's 差异图 button is checked with it

    def _paint_heat(self) -> None:
        payload = self.assist_result
        if not payload or payload.get("delta") is None:
            self.report("no difference map for this frame yet")
            self.heat_item.setVisible(False)
            return
        rgba = np.ascontiguousarray(heat_rgba(payload["delta"], payload.get("roi")))
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, 4 * width,
                       QImage.Format.Format_RGBA8888)
        self.heat_item.setPixmap(QPixmap.fromImage(image.copy()))
        self.heat_item.setVisible(True)

    # ------------------------------------------------------------- lifecycle
    def shutdown_assist(self) -> None:
        """Stop every assist thread: the detector, the SAM queue and loader, the diff worker.

        **The detector first** (U6 round 2).  Its model build waits for SAM's
        turn, and SAM's loader releases that turn when it finishes; joining the
        loader before stopping the detector let a window closed during SAM's
        load wake the build *after* the close and spend seconds building
        RF-DETR for nobody.  Now the worker is stopped, the turn is released
        (the build wakes, sees the stop and builds nothing), and only then is
        the loader waited for -- which, finding the window closing, stops the
        queue it built itself rather than handing it over.
        """
        for tool in (self.sam_point, self.sam_box):
            tool.queue = None       # nothing new may be submitted from here on
        # The detector's thread writes its cache on the way out (task U3).
        self.shutdown_detector()
        self._sam_turn.release()
        self._sam_closing.set()
        if self._owns_queue and self.sam_queue is not None:
            try:
                self.sam_queue.stop()
            except Exception:  # pragma: no cover - a wedged GPU call
                pass
            self.sam_queue = None
        loader = getattr(self, "_sam_loader_thread", None)
        if loader is not None and loader.is_alive():
            loader.join(30.0)       # a half-built SamQueue must not outlive us
        self.assist.shutdown()
        self.roi_proposer.shutdown()
