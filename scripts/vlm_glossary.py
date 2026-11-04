"""The glossary a chat prompt ends with (spec 3.5): what each named class is, and what its states mean.

``glossary_text(terms, state_terms)`` -- ``terms`` = the class ids the question or its options
name (``prompt.terms``), ``state_terms`` = the classes whose state the question asks
(``prompt.state_terms``) -- gives one line per class with its description and, for a state
class, one plain line per state, written from what the photograph shows.

Self-contained on purpose: the pack's ``tools/`` ship this file without the ``tda`` package and
without ``configs/taxonomy.yaml``. So the three tables are literals:

* :data:`CLASS_TEXT` -- the scanner pack's ``class_description`` per class, copied verbatim from
  ``CLASS_EN`` in ``scripts/export_scanpack.py`` (a test keeps them equal);
* :data:`MODEL_TEXT` -- the model-facing wording of the few of those lines that talk about the
  annotation's attributes (``attrs.kind`` ...), which mean nothing to a model;
* :data:`STATE_MEANINGS` -- one line per state of every class with two or more tracked states in
  ``configs/taxonomy.yaml`` (authored from it; a test keeps the state sets equal).
  **A draft: the owner reviews it.**
"""
from __future__ import annotations

from typing import Iterable, Optional

__all__ = ["CLASS_NAME", "CLASS_TEXT", "MODEL_TEXT", "PRESENCE_MEANINGS", "STATE_MEANINGS",
           "class_text", "glossary_text", "is_presence"]

#: How the questions name a class (``vlm_tasks.class_label``: initialisms up, underscores out).
CLASS_NAME: dict[str, str] = {
    "chassis": "chassis", "drive_cage": "drive cage", "cover": "cover",
    "cooler_bracket": "cooler bracket", "motherboard": "motherboard", "cpu": "CPU",
    "cpu_cooler": "CPU cooler", "ram_module": "RAM module", "psu": "PSU",
    "storage_drive": "storage drive", "optical_drive": "optical drive",
    "expansion_card": "expansion card", "case_fan": "case fan", "misc_part": "misc part",
    "screw": "screw", "ram_latch": "RAM latch", "cpu_socket_lever": "CPU socket lever",
    "psu_latch": "PSU latch", "drive_latch": "drive latch", "card_latch": "card latch",
    "cooler_latch": "cooler latch", "cable_clip": "cable clip", "connector": "connector",
    "loose_cable": "loose cable", "screw_hole": "screw hole", "other": "other",
}

#: The scanner pack's class descriptions, verbatim (``export_scanpack.CLASS_EN``).
CLASS_TEXT: dict[str, str] = {
    "chassis": "The computer case: the metal enclosure everything else sits in.",
    "drive_cage": "A cage or bracket holding storage or optical drives.",
    "cover": "A removable cover, shroud or duct (attrs.of says over what, e.g. the air shroud "
             "over the CPU cooler).",
    "cooler_bracket": "The bracket that holds the CPU cooler down.",
    "motherboard": "The main circuit board.",
    "cpu": "The processor in its socket.",
    "cpu_cooler": "The CPU cooler: heatsink and/or fan (attrs.kind).",
    "ram_module": "A memory (RAM) stick.",
    "psu": "The power supply unit.",
    "storage_drive": "A hard disk or SSD (attrs.kind).",
    "optical_drive": "A CD/DVD drive.",
    "expansion_card": "An add-in card in a slot (GPU, Wi-Fi card, ...).",
    "case_fan": "A case fan.",
    "misc_part": "Another removable part (attrs.name says which).",
    "screw": "A screw (attrs.role: what it fastens; a captive screw stays on its part).",
    "ram_latch": "The latch at the end of a RAM slot; moulded into the motherboard.",
    "cpu_socket_lever": "The CPU socket's retention lever; part of the motherboard.",
    "psu_latch": "A latch or bracket holding the power supply.",
    "drive_latch": "A latch holding a drive or a drive cage.",
    "card_latch": "A latch holding an expansion card.",
    "cooler_latch": "What holds the CPU cooler in place: a latch, a clip, or (on some Dell "
                    "machines) a frame around the fan.",
    "cable_clip": "A clip holding a cable.",
    "connector": "A cable plug in its socket (tracked only while plugged in).",
    "loose_cable": "A cable laid out loose with the removed parts (attrs.kind: sata, power, fan, "
                   "front_panel, other); drawn in the components gallery only.",
    "screw_hole": "Where a removed CPU-fan screw was (attrs.of: the screw); only the fan screws "
                  "of the heatsink + fan machines D07 D08 D11 D14 D16 D18 D19, a detection "
                  "category only.",
    "other": "An object outside the machine (context), named by its label.",
}

#: The model-facing wording of the descriptions that speak of annotation attributes.
MODEL_TEXT: dict[str, str] = {
    "cover": "A removable cover, shroud or duct over some part (e.g. the air shroud over the "
             "CPU cooler).",
    "cpu_cooler": "The CPU cooler: a heatsink, a fan, or a heatsink with its fan.",
    "storage_drive": "A hard disk or SSD.",
    "misc_part": "Another removable part of the machine.",
    "screw": "A screw; a captive screw stays on its part when undone.",
    "connector": "A cable plug in its socket.",
    "loose_cable": "A cable laid out loose with the removed parts.",
    "screw_hole": "The empty hole where a removed CPU-fan screw was.",
}

#: One plain line per state, from what the photograph shows. DRAFT -- the owner reviews it.
#: Authored from ``configs/taxonomy.yaml`` (every class with two or more states; ``chassis``
#: has one and is left out).
#:
#: ``displaced`` (owner decision (a), 2026-11-04): every class says what it concretely means for
#: that part. Four classes use it in the 2026-11-02 pack -- gold ``displaced`` in V2 / V14 of
#: ``vlm_scan`` and the ``displace`` verb or a ``displaced`` state anywhere in ``vlm_multi``:
#: ``drive_cage`` (V2 128, V14 6; multi 619, almost all the optical drive in its hinged
#: bracket), ``psu`` (V2 102; multi 511), ``storage_drive`` (V2 5; multi 44) and ``cpu_cooler``
#: (V2 2, V14 1, the M91P's fan; multi 34). No other class has it as a gold state.
STATE_MEANINGS: dict[str, dict[str, str]] = {
    "drive_cage": {
        "installed": "the cage sits in its place in the case",
        "displaced": "the cage -- on most machines here the optical drive in its bracket -- "
                     "has been swung up or tilted out of its seat on its hinge but is still "
                     "attached to the case",
        "removed": "the cage is out of the case (it may lie on the bench beside it)",
    },
    "cover": {
        "closed": "the cover, shroud or duct is in place over what it covers",
        "open": "it is swung or lifted open but still attached, so what was under it shows",
        "removed": "it has been taken off the machine",
    },
    "cooler_bracket": {
        "installed": "the bracket sits in its place, holding the CPU cooler down",
        "displaced": "the bracket has been lifted off its seat over the CPU cooler but is still "
                     "in the case",
        "removed": "the bracket is out of the case",
    },
    "motherboard": {
        "installed": "the board is mounted in the case",
        "displaced": "the board has been lifted or slid off its mounting posts but is still in "
                     "the case",
        "removed": "the board is out of the case",
    },
    "cpu": {
        "installed": "the processor sits in its socket",
        "displaced": "the processor has been lifted out of its socket but still lies in the "
                     "machine",
        "removed": "the processor is out of the machine; its socket is empty",
    },
    "cpu_cooler": {
        "installed": "the cooler sits on the CPU",
        "displaced": "the cooler (on the machines that use this state, its fan) has been lifted "
                     "or swung off the CPU but is still in the case, held by its cable or hinge",
        "removed": "the cooler is out of the case",
    },
    "ram_module": {
        "installed": "the stick is seated in its slot",
        "displaced": "the stick has been tilted or pulled partly out of its slot but is still "
                     "in it",
        "removed": "the stick is out of the machine; its slot is empty",
    },
    "psu": {
        "installed": "the power supply sits in its bay",
        "displaced": "the power supply has been pulled or swung out of its bay (on many small "
                     "cases it tips up on a hinge) but is still in the case",
        "removed": "the power supply is out of the case",
    },
    "storage_drive": {
        "installed": "the drive sits in its bay or cage",
        "displaced": "the drive has been slid or pulled partly out of its bay or cage but is "
                     "still in the case",
        "removed": "the drive is out of the case",
    },
    "optical_drive": {
        "installed": "the drive sits in its bay",
        "displaced": "the drive has been pushed or slid partly out of its bay but is still in "
                     "the case",
        "removed": "the drive is out of the case",
    },
    "expansion_card": {
        "installed": "the card is seated in its slot",
        "displaced": "the card has been lifted partly out of its slot but is still in the case",
        "removed": "the card is out of the case; its slot is empty",
    },
    "case_fan": {
        "installed": "the fan is mounted in its place on the case",
        "displaced": "the fan has been taken off its mount but still hangs in the case (e.g. by "
                     "its cable)",
        "removed": "the fan is out of the case",
    },
    "misc_part": {
        "installed": "the part is in its place",
        "displaced": "the part has been moved off its place but is still in the machine",
        "removed": "the part is out of the machine",
    },
    "screw": {
        "fastened": "the screw is in its hole, screwed in",
        "loosened": "a captive screw that has been undone but is still held in its part (it can "
                    "look the same as a fastened one)",
        "removed": "the screw has been taken out; its hole is empty",
    },
    "ram_latch": {
        "closed": "the latch stands upright, locking the end of the RAM stick (or the empty slot)",
        "open": "the latch is pushed outward, releasing the RAM stick",
        "removed": "the latch has left the case with the motherboard",
    },
    "cpu_socket_lever": {
        "closed": "the lever lies down, hooked under its catch, holding the socket shut",
        "open": "the lever is unhooked and raised, so the socket can be opened",
        "removed": "the lever has left the case with the motherboard",
    },
    "psu_latch": {
        "closed": "the latch or bracket holds the power supply in place",
        "open": "it is released (pushed, pulled or swung aside) and no longer holds the power "
                "supply",
        "removed": "it has been taken off the machine",
    },
    "drive_latch": {
        "closed": "the latch holds its drive or drive cage in place",
        "open": "it is released (lever lifted, tab pushed), so the drive or cage can slide out",
        "removed": "it has been taken off, or has left with its drive cage",
    },
    "card_latch": {
        "closed": "the latch holds the expansion card in its slot",
        "open": "it is released, so the card can be lifted out",
        "removed": "it has been taken off the machine",
    },
    "cooler_latch": {
        "closed": "the latch, clip or frame holds the CPU cooler down",
        "open": "it is released, so the cooler can be lifted",
        "removed": "it has been taken off the machine",
    },
    "cable_clip": {
        "closed": "the clip holds its cable in place",
        "open": "the clip is open and its cable has been taken out of it",
        "removed": "the clip has left the machine with the part it sits on",
    },
    "connector": {
        "plugged": "the plug is seated in its socket",
        "unplugged": "the plug has been pulled out of its socket (the cable may still hang "
                     "nearby)",
        "removed": "the plug and its cable are gone from the machine, or left with the part "
                   "they were plugged into",
    },
    "loose_cable": {
        "installed": "the loose cable lies in the picture with the removed parts",
        "removed": "the cable is no longer in the picture",
    },
    "screw_hole": {
        "present": "the empty hole where a removed CPU-fan screw was can be seen",
        "removed": "the hole has left the case with its fan",
    },
}


#: The states of a question asked on presence alone (``answer_check.state_space ==
#: "presence"``: a screw is asked present / removed, its tightness was never annotated --
#: final review M3). Every other class keeps :data:`STATE_MEANINGS`.
PRESENCE_MEANINGS: dict[str, dict[str, str]] = {
    "screw": {
        "present": "the screw is still in its hole (screwed in, or for a captive screw "
                   "undone but held -- a photograph cannot tell these apart)",
        "removed": "the screw has been taken out; its hole is empty",
    },
}


def is_presence(record: Optional[dict]) -> bool:
    """Is ``record`` a question on presence alone (``answer_check.state_space``)?"""
    check = ((record or {}).get("label") or {}).get("answer_check") or {}
    return check.get("state_space") == "presence"


def _base(term) -> Optional[str]:
    """``"screw.cpu_cooler"`` (a refined match class) -> ``"screw"``; unknown -> ``None``."""
    cls = str(term or "").strip().split(".", 1)[0]
    return cls if cls in CLASS_TEXT or cls in STATE_MEANINGS else None


def class_text(cls: str) -> str:
    """The description a model reads for ``cls`` (no annotation attributes in it)."""
    return MODEL_TEXT.get(cls) or CLASS_TEXT.get(cls, "")


def glossary_text(terms: Optional[Iterable] = None, state_terms: Optional[Iterable] = None,
                  presence: bool = False) -> str:
    """The glossary block, or ``""`` when no known class is named.

    One bullet per class, in the order named (``terms`` first, then the ``state_terms`` not
    already there); a class in ``state_terms`` gets its states, one ``state = meaning`` each.
    ``presence`` (the record asks present / removed, :func:`is_presence`) gives a class of
    :data:`PRESENCE_MEANINGS` those two states instead of its tracked ones.
    """
    order: list[str] = []
    for term in list(terms or []) + list(state_terms or []):
        cls = _base(term)
        if cls and cls not in order:
            order.append(cls)
    states = {c for c in (_base(t) for t in state_terms or []) if c}
    if not order:
        return ""
    lines = ["Glossary:"]
    for cls in order:
        name = CLASS_NAME.get(cls, cls.replace("_", " "))
        line = f"- {name}: {class_text(cls)}".rstrip()
        meanings = (PRESENCE_MEANINGS.get(cls) if presence else None) or STATE_MEANINGS.get(cls)
        if cls in states and meanings:
            line += " States: " + "; ".join(f"{s} = {m}" for s, m in meanings.items()) + "."
        lines.append(line)
    return "\n".join(lines)
