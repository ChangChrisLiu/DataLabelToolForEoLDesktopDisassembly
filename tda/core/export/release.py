"""What a published file may say about a part and a machine (tasks U30-PACK, U46).

Two rules every file that leaves the lab follows, kept here so the scanner
package (``scripts/export_scanpack.py``) and the precedence-graph export
(:mod:`tda.core.export.graph_export`) cannot drift apart:

* :data:`DROP_ATTRS` -- instance attributes that are our bookkeeping (who added
  a part, when, how a flag was guessed, a note to ourselves) rather than facts
  about the part; :func:`public_attrs` leaves them out. Whether a part was
  added by hand or implied by a rule is said by the files' own ``extra`` /
  ``implied`` flags instead.
* :data:`MODEL_NAMES` / :func:`model_names` -- the machine's real model name
  where the teardown sheet misspells it, the sheet's text kept beside it and a
  caveat (``model_note``) where the name itself is uncertain.

Moved out of the scanner package's script in task U46 unchanged; the script
re-exports both names.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import IMPLIED_RULE_ATTR

__all__ = ["DROP_ATTRS", "MODEL_NAMES", "NAMEPLATE", "model_names", "public_attrs"]

#: Instance attrs that are our bookkeeping (who added a part, when, how a flag
#: was guessed, a note to ourselves) rather than facts about the part.
DROP_ATTRS = ("added_by", "added_at", "captive_source", "reason", "note",
              "implied", IMPLIED_RULE_ATTR)

#: ``(brand, model as the teardown sheet spells it) -> (the model's real name,
#: a caveat or None)``. The sheet's text is kept beside it (``model_sheet``), as
#: written; the caveat goes to ``model_note``, never into the name.
NAMEPLATE = "to be verified on the nameplate: the teardown sheet says '{sheet}'"
MODEL_NAMES: dict[tuple[str, str], tuple[str, Optional[str]]] = {
    ("Dell", "Optiplex 7020"): ("Dell OptiPlex 7020", None),
    ("Dell", "Optiplex 7010"): ("Dell OptiPlex 7010", None),
    ("Dell", "Optiplex 3020"): ("Dell OptiPlex 3020", None),
    ("Dell", "Optiplex 990"): ("Dell OptiPlex 990", None),
    ("HP", "Compoq 8100 Elite Small Form Fador"): ("HP Compaq 8100 Elite SFF", None),
    ("HP", "Compoq 8300 Elite Small Form Fador"): ("HP Compaq 8300 Elite SFF", None),
    ("HP", "Compoq 8200 Elite Small Form Fador"): ("HP Compaq 8200 Elite SFF", None),
    # HP made no "Compaq 4300 Elite": the nearest is the Compaq Pro 4300 SFF
    ("HP", "Compoq 4300 Elite Small Form Fador"): (
        "HP Compaq Pro 4300 SFF", NAMEPLATE.format(sheet="Compoq 4300 Elite Small Form Fador")),
    ("HP", "EliteDesk 800 GI SFF"): ("HP EliteDesk 800 G1 SFF", None),
    ("HP", "ProDesk 400 GI SFF"): ("HP ProDesk 400 G1 SFF", None),
    ("HP", "ProDesk 4300 GI SFF"): ("HP Compaq Pro 4300 SFF",
                                    NAMEPLATE.format(sheet="ProDesk 4300 GI SFF")),
    ("HP", "ProDesk 4300"): ("HP Compaq Pro 4300 SFF", NAMEPLATE.format(sheet="ProDesk 4300")),
}


def public_attrs(attrs: Optional[dict]) -> dict:
    """``attrs`` without :data:`DROP_ATTRS`."""
    return {a: v for a, v in (attrs or {}).items() if a not in DROP_ATTRS}


def model_names(meta: dict) -> tuple[str, str, Optional[str]]:
    """``(model, model_sheet, model_note)``: the real name, the teardown sheet's
    text as is, and a caveat about the name (or ``None``)."""
    brand = str(meta.get("brand") or "").strip()
    family = str(meta.get("model_family") or "").strip()
    sheet = str(meta.get("brand_model_raw") or f"{brand} {family}").strip()
    name, note = MODEL_NAMES.get((brand, family), (f"{brand} {family}".strip(), None))
    return name, sheet, note
