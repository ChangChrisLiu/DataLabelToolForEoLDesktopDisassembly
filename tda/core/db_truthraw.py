"""Truth rows exactly as stored, for an op whose inverse must restore them (task U32).

:meth:`tda.core.db.Db.put_compiled` takes decoded values and stamps
``verified_at`` with the time of the write, which is right for a row being
signed and wrong for a row being *put back*: 「重签」's undo
(:func:`tda.core.truth_refreeze.undo_refreeze`) promises the rows it restores
byte for byte -- the RLE text as it was stored, the signer and the time they
signed.  So this reads and writes the ``compiled_mask`` columns verbatim, and
the same for the frame's digest row, which the re-signing stamps and the undo
puts back as it was (or removes, when there was none).
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey

__all__ = ["RAW_COLUMNS", "RawTruthMixin"]

#: The ``compiled_mask`` columns a row is, besides its key.
RAW_COLUMNS = ("visible_rle_json", "occlusion_ratio", "visibility", "placement", "status",
               "input_hash", "verified_by", "verified_at", "geom_type", "box_json")


class RawTruthMixin:
    """Verbatim reads and writes of truth rows and digests, for ``tda.core.db.Db``."""

    def compiled_raw(self, key: FrameKey) -> dict[str, dict]:
        """One frame's truth rows keyed by instance, every column as stored (no decoding)."""
        rows = self.conn.execute(
            "SELECT instance, " + ", ".join(RAW_COLUMNS) + " FROM compiled_mask "
            "WHERE desktop=? AND step=? AND view=? ORDER BY instance",
            (key.desktop, key.step, key.view),
        ).fetchall()
        return {str(r["instance"]): {c: r[c] for c in RAW_COLUMNS} for r in rows}

    def put_compiled_raw(self, key: FrameKey, instance: str, raw: dict) -> None:
        """Write one truth row with exactly these column values (every one of them)."""
        missing = [c for c in RAW_COLUMNS if c not in raw]
        if missing:
            raise ValueError(f"a raw truth row needs every column; missing {missing}")
        self._upsert("compiled_mask", self._fk(key) | {"instance": str(instance)},
                     {c: raw[c] for c in RAW_COLUMNS}, desktop=key.desktop)

    def put_frame_digest_raw(self, key: FrameKey, row: Optional[dict]) -> None:
        """Put one frame's digest row back as it was: these values, or none at all."""
        if row is None:
            self.clear_frame_digest(key)
            return
        self.set_frame_digest(key, str(row["digest"]), str(row["compiler_version"]),
                              int(row["n_rows"]))
