"""SAM 2.1 with two decoder heads: stock, and the annotator's fine-tuned one (task U19).

L5 (``experiments/l5_sam/REPORT.md``) fine-tuned SAM 2.1's prompt encoder and
mask decoder on the annotator's verified scan masks.  On two held-out
desktops the small parts gained (macro IoU +0.10, boundary F +0.11 ... +0.13:
screws drawn as tight as the annotator draws them, a connector's housing from
a single click instead of a bundle of cables), and one cell lost: a single
click, no box, on a **large** part (-0.012 ... -0.021, in a cell where neither
head is usable).  The controller's ruling (REPORT.md section 0) is one image
encoder and two decoder heads, routed by prompt type and class so that the
regression cannot happen by construction:

* a prompt with a box (detector box, difference-map box, a drag, point+box)
  -> the tuned head;
* point-only, and the part being drawn is of a ``large_classes`` class
  -> the stock head;
* point-only, any other class, or no class known -> the tuned head (the
  common case is a small part);
* a refinement (``mask_input``) -> the head that produced the mask it refines.

This module is the half that needs neither Qt nor torch at import: the
configuration (``configs/sam.yaml``, :func:`load_sam_head_config`), the routing
rule (:func:`choose_head`), and the head file's format (:func:`read_head`,
written by ``experiments/l5_sam/export_head.py``).
:class:`tda.models.sam_service.SamService` builds the tuned modules.

**Never silently, never fatal.**  ``enabled: false`` and ``TDA_SAM_HEAD=off``
are one INFO line each; a configuration that cannot be used -- missing,
malformed, a head file that is not there -- is one WARNING saying why, and
SAM runs with the stock head everywhere, exactly as before U19.  A head file
that is there but does not load (wrong format, tensors that do not fit, made
for another SAM checkpoint) is the service's one WARNING, with the same
outcome.  ``TDA_SAM_HEAD`` names another configuration file, or ``off``.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "DEFAULT_FILE", "ENV_VAR", "FINGERPRINT_KEYS", "HEADS", "HEAD_FORMAT",
    "HEAD_MODULES", "STOCK", "TUNED", "Routing", "SamHeadConfig",
    "SamHeadConfigError", "SamHeadError", "choose_head", "fingerprint",
    "large_classes_from_bands", "load_sam_head_config", "parse_sam_head_config",
    "read_head", "tensor_digest",
]

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO / "configs" / "sam.yaml"
#: Another configuration file for this process, or ``off``.
ENV_VAR = "TDA_SAM_HEAD"

#: The two heads, by the name :attr:`SamResult.head` reports.
STOCK = "stock"
TUNED = "tuned"
HEADS = (STOCK, TUNED)

#: ``format`` of a head file; bumped whenever its layout changes.
HEAD_FORMAT = "tda-sam2-head/1"
#: The modules of ``SAM2Base`` a head may replace -- and nothing else.
HEAD_MODULES = ("sam_prompt_encoder", "sam_mask_decoder")
#: Tensors of the stock checkpoint whose bytes identify it.  A head is a delta
#: on one particular SAM 2.1 checkpoint; on another one it is not the model
#: that was measured.  ``no_mem_embed`` is added to the image embedding both
#: heads decode from; the image encoder's first and last weights stand for the
#: rest of it.  The export adds every tensor of :data:`HEAD_MODULES` the head
#: does *not* replace, since the tuned head uses the stock copies of those.
FINGERPRINT_KEYS = (
    "no_mem_embed",
    "image_encoder.trunk.patch_embed.proj.weight",
    "image_encoder.trunk.pos_embed",
    "image_encoder.neck.convs.0.conv.weight",
    "image_encoder.neck.convs.3.conv.weight",
)

_TOP_KEYS = ("enabled", "head", "route", "large_classes")
_ROUTE_KEYS = ("box_prompts", "point_only")
_POINT_KEYS = ("large", "other")


class SamHeadConfigError(ValueError):
    """Why ``configs/sam.yaml`` cannot be used; the sentence the WARNING says."""


class SamHeadError(RuntimeError):
    """Why a head file cannot be loaded; the sentence the WARNING says."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Routing:
    """Which head answers which prompt (``route`` and ``large_classes``)."""

    box_prompts: str = TUNED
    point_large: str = STOCK
    point_other: str = TUNED
    large_classes: frozenset = field(default_factory=frozenset)


@dataclass(frozen=True)
class SamHeadConfig:
    """``configs/sam.yaml``, checked.

    ``off`` says why the tuned head is not to be loaded (``""`` when it is);
    a configuration that is off still carries a :class:`Routing`, but nothing
    reads it -- with one head every prompt goes to the stock one.
    """

    routing: Routing = field(default_factory=Routing)
    head: Optional[Path] = None
    source: Optional[Path] = None
    off: str = ""

    @property
    def wanted(self) -> bool:
        """Should the service load the tuned head?"""
        return not self.off and self.head is not None


def choose_head(routing: Routing, *, has_box: bool, instance_cls: Optional[str] = None,
                mask_input_head: Optional[str] = None) -> str:
    """The head that answers one prompt (the ruling of L5 section 0).

    ``mask_input_head`` -- the head that produced the mask a refinement sends
    back -- wins: a correcting click goes to the head whose answer it corrects.
    Without one, a box (with or without points) goes to ``box_prompts``; a
    prompt with no box to ``point_large`` when the part is of a large class,
    to ``point_other`` otherwise -- an unknown class included.
    """
    if mask_input_head in HEADS:
        return str(mask_input_head)
    if has_box:
        return routing.box_prompts
    if instance_cls is not None and str(instance_cls) in routing.large_classes:
        return routing.point_large
    return routing.point_other


def _head_name(raw: Any, what: str) -> str:
    if not isinstance(raw, str) or raw.strip() not in HEADS:
        raise SamHeadConfigError(f"{what} = {raw!r} is not one of {', '.join(HEADS)}")
    return raw.strip()


def _mapping(raw: Any, what: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(raw, dict):
        raise SamHeadConfigError(f"{what} must be a mapping ({what}: {raw!r})")
    unknown = sorted(str(k) for k in raw if k not in keys)
    if unknown:
        raise SamHeadConfigError(f"{what} has unknown keys {unknown} "
                                 f"(it takes {', '.join(keys)})")
    missing = [k for k in keys if k not in raw]
    if missing:
        raise SamHeadConfigError(f"{what} is missing {missing}")
    return raw


def _known_classes() -> frozenset:
    from tda.core.taxonomy import load_taxonomy

    return frozenset(str(c) for c in load_taxonomy().classes)


def parse_sam_head_config(data: Any, source: Optional[Path] = None,
                          known_classes: Optional[Iterable[str]] = None) -> SamHeadConfig:
    """A :class:`SamHeadConfig`, or :class:`SamHeadConfigError` saying why not.

    Refused: anything but a mapping; a key it does not know (``large_class``
    would otherwise route every motherboard to the tuned head without a word);
    ``enabled`` that is not true/false; a ``head`` that is not a path; a route
    that is not ``stock`` or ``tuned``; ``large_classes`` that is not a list of
    taxonomy classes.  Whether the head file exists is :func:`load_sam_head_config`'s
    question, not this one's.
    """
    if not isinstance(data, dict):
        raise SamHeadConfigError("the SAM head file is not a mapping of keys")
    unknown = sorted(str(k) for k in data if k not in _TOP_KEYS)
    if unknown:
        raise SamHeadConfigError(f"unknown keys {unknown} (it takes {', '.join(_TOP_KEYS)})")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SamHeadConfigError(f"enabled = {enabled!r} is not true or false")
    raw_head = data.get("head")
    if not isinstance(raw_head, str) or not raw_head.strip():
        raise SamHeadConfigError(f"head must be a path (head: {raw_head!r})")
    head = Path(raw_head.strip())
    if not head.is_absolute() and source is not None:
        head = source.parent / head
    route = _mapping(data.get("route"), "route", _ROUTE_KEYS)
    point = _mapping(route["point_only"], "route.point_only", _POINT_KEYS)
    raw_large = data.get("large_classes")
    if (not isinstance(raw_large, list) or not raw_large
            or not all(isinstance(c, str) and c.strip() for c in raw_large)):
        raise SamHeadConfigError(f"large_classes must be a non-empty list of class "
                                 f"names (large_classes: {raw_large!r})")
    large = tuple(dict.fromkeys(c.strip() for c in raw_large))
    known = frozenset(known_classes) if known_classes is not None else _known_classes()
    strangers = [c for c in large if c not in known]
    if strangers:
        raise SamHeadConfigError(f"large_classes {strangers} are not taxonomy classes")
    routing = Routing(
        box_prompts=_head_name(route["box_prompts"], "route.box_prompts"),
        point_large=_head_name(point["large"], "route.point_only.large"),
        point_other=_head_name(point["other"], "route.point_only.other"),
        large_classes=frozenset(large),
    )
    off = "" if enabled else f"enabled: false in {source}"
    return SamHeadConfig(routing=routing, head=head, source=source, off=off)


def _read_yaml(path: Path) -> Any:
    import yaml

    if not path.is_file():
        raise SamHeadConfigError(f"the SAM head file {path} does not exist")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SamHeadConfigError(f"the SAM head file {path} cannot be read: "
                                 f"{type(exc).__name__}: {exc}") from exc


def load_sam_head_config(path: Optional[str] = None) -> SamHeadConfig:
    """``configs/sam.yaml`` (or ``path``) checked; never raises, never ``None``.

    Without ``path`` the environment's :data:`ENV_VAR` is read first: ``off``
    is a configuration that is off, anything else a file to read instead.
    An unusable file is one WARNING and a configuration that is off (its
    ``off`` says why); ``enabled: false`` and ``off`` are one INFO line.  The
    line that says the head is **on** comes from the service, once the head
    has loaded: until then it is not on.
    """
    if path is None:
        chosen = os.environ.get(ENV_VAR, "").strip()
        if chosen.lower() == "off":
            log.info("SAM fine-tuned head off: %s=off; the stock head answers every prompt",
                     ENV_VAR)
            return SamHeadConfig(off=f"{ENV_VAR}=off")
        path = chosen or None
    target = Path(path) if path else DEFAULT_FILE
    try:
        config = parse_sam_head_config(_read_yaml(target), target)
        if config.wanted and not config.head.is_file():
            raise SamHeadConfigError(f"the head file {config.head} does not exist")
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("SAM fine-tuned head OFF, the stock head answers every prompt: %s", exc)
        return SamHeadConfig(source=target, off=str(exc))
    if config.off:
        log.info("SAM fine-tuned head off: %s; the stock head answers every prompt",
                 config.off)
    return config


def large_classes_from_bands(bands: Mapping, view: str = "scan",
                             threshold_px: float = 20_000.0) -> list[str]:
    """Classes whose ``view`` band's upper bound exceeds ``threshold_px``.

    The rule ``large_classes`` was written down from (``prompt_box_bands.yaml``,
    task U19); kept as a function so the test suite can say when the bands are
    re-fitted and the list in ``configs/sam.yaml`` no longer follows from them.
    """
    classes = ((bands.get("views") or {}).get(view) or {}).get("classes") or {}
    return sorted(str(cls) for cls, band in classes.items()
                  if float((band or {}).get("hi_px", 0.0)) > float(threshold_px))


# --------------------------------------------------------------------------- #
# the head file
# --------------------------------------------------------------------------- #
def tensor_digest(tensor: Any) -> str:
    """SHA-256 of a tensor's dtype, shape and raw bytes (any device, any dtype)."""
    import torch

    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    head = f"{flat.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return hashlib.sha256(head + raw).hexdigest()


def fingerprint(state: Mapping[str, Any], keys: Iterable[str]) -> dict[str, str]:
    """``{key: tensor_digest}`` for ``keys`` of a state dict; a missing key raises."""
    return {str(k): tensor_digest(state[k]) for k in keys}


def read_head(path: Path) -> dict:
    """The head file at ``path``, checked; :class:`SamHeadError` says why not.

    Loaded with ``weights_only=True`` (tensors and plain containers, no code).
    Checked here: the format, that every tensor belongs to one of
    :data:`HEAD_MODULES`, and that the base fingerprint is there.  Whether the
    tensors fit the model, and the fingerprint matches it, is checked where
    the model is (:class:`~tda.models.sam_service.SamService`).
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise SamHeadError(f"the head file {path} does not exist")
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - a corrupt or foreign file
        raise SamHeadError(f"the head file {path} does not load: "
                           f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(blob, dict) or blob.get("format") != HEAD_FORMAT:
        found = blob.get("format") if isinstance(blob, dict) else type(blob).__name__
        raise SamHeadError(f"the head file {path} is not a {HEAD_FORMAT} file "
                           f"(format: {found!r})")
    state = blob.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise SamHeadError(f"the head file {path} has no tensors")
    prefixes = tuple(f"{m}." for m in HEAD_MODULES)
    foreign = sorted(k for k in state if not str(k).startswith(prefixes))
    if foreign:
        raise SamHeadError(f"the head file {path} replaces tensors outside "
                           f"{', '.join(HEAD_MODULES)}: {foreign[:3]}")
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise SamHeadError(f"the head file {path} holds something that is not a tensor")
    base = blob.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("fingerprint"), dict) \
            or not base["fingerprint"]:
        raise SamHeadError(f"the head file {path} does not say which SAM checkpoint "
                           f"it was made for (no base fingerprint)")
    return blob
