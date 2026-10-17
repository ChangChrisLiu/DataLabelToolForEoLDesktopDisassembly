"""Task U19: ``SamService`` with two decoder heads, driven on a toy ``SAM2Base``.

The real model is 900 MB and a GPU; the plumbing is not.  sam2's two entry
points (``build_sam2``, ``SAM2ImagePredictor``) are replaced by a model of a
few 1x1 convolutions with the attributes the service and the predictor touch,
and a predictor that keeps sam2's state (``_features``, ``_orig_hw``) and
scores each answer by its decoder -- so the tests can see which head answered,
that the head went onto copies, that the two heads share one embedding and
that the tuned one projects the high-res levels with its own convolutions,
exactly as the whole fine-tuned model would.  ``tests/test_u19_weights.py``
checks the same on the real weights where they are.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from tda.models import sam_heads as H
from tda.models.sam_heads import STOCK, TUNED, SamHeadConfig, load_sam_head_config
from test_u19_heads import ROUTING, _warnings, _with, _write

torch = pytest.importorskip("torch")
nn = torch.nn


class _Encoder(nn.Module):
    """Three levels, like Hiera-L's FPN with ``scalp: 1``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        f = self.proj(x)
        return {"vision_features": f[..., ::4, ::4], "vision_pos_enc": [f, f, f],
                "backbone_fpn": [f, f[..., ::2, ::2], f[..., ::4, ::4]]}


class _PromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point = nn.Parameter(torch.zeros(2))
        self.register_buffer("gauss", torch.ones(2))


class _MaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_s0 = nn.Conv2d(8, 2, 1)
        self.conv_s1 = nn.Conv2d(8, 4, 1)
        self.iou = nn.Parameter(torch.tensor([0.25]))
        self.obj = nn.Parameter(torch.zeros(3))      # a tensor the head leaves alone


class _Sam(nn.Module):
    """What ``SamService`` and ``SAM2ImagePredictor`` touch of ``SAM2Base``."""

    image_size = 32
    use_high_res_features_in_sam = True
    num_feature_levels = 3

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.image_encoder = _Encoder()
        self.sam_prompt_encoder = _PromptEncoder()
        self.sam_mask_decoder = _MaskDecoder()
        self.no_mem_embed = nn.Parameter(torch.full((1, 8, 1, 1), 0.5))

    def forward_image(self, x):                      # SAM2Base.forward_image
        out = self.image_encoder(x)
        out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(out["backbone_fpn"][0])
        out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(out["backbone_fpn"][1])
        return out


class _Predictor:
    """``SAM2ImagePredictor``'s state and the calls the service makes on it."""

    def __init__(self, model) -> None:
        self.model = model
        self._transforms = object()
        self.reset_predictor()

    def reset_predictor(self) -> None:
        self._features = None
        self._orig_hw = None
        self._is_image_set = False
        self._is_batch = False

    def set_image(self, image) -> None:
        self.reset_predictor()
        x = torch.as_tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1)[None] / 255
        fpn = self.model.forward_image(x)["backbone_fpn"]
        self._features = {"image_embed": fpn[2] + self.model.no_mem_embed,
                          "high_res_feats": [fpn[0], fpn[1]]}
        self._orig_hw = [tuple(np.asarray(image).shape[:2])]
        self._is_image_set = True

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None,
                multimask_output=True, normalize_coords=True, return_logits=False):
        assert self._is_image_set, "decoded before set_image"
        h, w = self._orig_hw[0]
        n = 3 if multimask_output else 1
        masks = np.zeros((n, h, w), np.float32)
        masks[:, : h // 2, : w // 2] = 1.0
        score = float(self.model.sam_mask_decoder.iou.detach()[0])
        return (masks, np.array([score - 0.01 * i for i in range(n)], np.float32),
                np.zeros((n, 8, 8), np.float32))


TUNED_IOU = 0.75


@pytest.fixture
def toy(monkeypatch, tmp_path):
    """Patch sam2's two entry points with the toy; returns the checkpoint path."""
    pytest.importorskip("sam2.build_sam")
    import sam2.build_sam
    import sam2.sam2_image_predictor

    built: list = []

    def build(config, ckpt, device="cuda", **kwargs):
        built.append(_Sam())
        return built[-1]

    monkeypatch.setattr(sam2.build_sam, "build_sam2", build)
    monkeypatch.setattr(sam2.sam2_image_predictor, "SAM2ImagePredictor", _Predictor)
    ckpt = tmp_path / "stock.pt"
    ckpt.write_bytes(b"not read: build_sam2 is the toy")
    return ckpt, built


def _delta(model: _Sam) -> dict:
    """What L5 changed, in miniature: the prompt encoder, the convs, the IoU head."""
    md = model.sam_mask_decoder
    return {
        "sam_prompt_encoder.point": torch.tensor([1.0, 2.0]),
        "sam_mask_decoder.conv_s0.weight": -2.0 * md.conv_s0.weight.detach().clone(),
        "sam_mask_decoder.conv_s0.bias": md.conv_s0.bias.detach().clone() + 1.0,
        "sam_mask_decoder.conv_s1.weight": 3.0 * md.conv_s1.weight.detach().clone(),
        "sam_mask_decoder.iou": torch.tensor([TUNED_IOU]),
    }


def _blob(**overrides) -> dict:
    model = _Sam()
    state = model.state_dict()
    keys = ["no_mem_embed", "image_encoder.proj.weight", "sam_mask_decoder.obj",
            "sam_prompt_encoder.gauss"]
    blob = {"format": H.HEAD_FORMAT, "modules": list(H.HEAD_MODULES),
            "base": {"file": "stock.pt", "sha256": "0" * 64,
                     "fingerprint": H.fingerprint(state, keys)},
            "state_dict": _delta(model)}
    blob.update(overrides)
    return blob


def _head(tmp_path: Path, blob=None, name: str = "head.pt", manifest: bool = True) -> Path:
    import hashlib

    path = tmp_path / name
    torch.save(_blob() if blob is None else blob, str(path))
    if manifest:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".json").write_text(json.dumps({"head": {"sha256": digest}}),
                                             encoding="utf-8")
    return path


def _config(head: Path) -> SamHeadConfig:
    return SamHeadConfig(routing=ROUTING, head=head, source=None)


def _service(ckpt: Path, heads):
    from tda.models.sam_service import SamService

    return SamService(checkpoint=str(ckpt), device="cpu", heads=heads)


def _req(crop, **kwargs):
    from tda.models.sam_service import SamRequest

    return SamRequest(image_crop=crop, **kwargs)


CROP = (np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3) % 251).astype(np.uint8)
BOX = (4.0, 4.0, 20.0, 20.0)
POINT = [(10.0, 10.0, 1)]


def test_the_tuned_head_loads_and_answers_by_the_routing(toy, tmp_path, caplog):
    ckpt, built = toy
    with caplog.at_level(logging.INFO):
        svc = _service(ckpt, _config(_head(tmp_path)))
    assert svc.heads == (STOCK, TUNED) and svc.tuned_loaded and svc.head_off_reason == ""
    assert _warnings(caplog) == []
    assert any("SAM fine-tuned head on: head.pt (5 tensors)" in r.getMessage()
               for r in caplog.records)

    prior = np.zeros((32, 32), bool)
    prior[5:15, 5:15] = True
    cases = [
        (dict(box=BOX, instance_cls="screw"), TUNED),
        (dict(box=BOX, points=POINT, instance_cls="motherboard"), TUNED),
        (dict(points=POINT, instance_cls="motherboard"), STOCK),
        (dict(points=POINT, instance_cls="screw"), TUNED),
        (dict(points=POINT), TUNED),
        (dict(points=POINT, instance_cls="other"), TUNED),
        (dict(points=POINT, box=BOX, mask_input=prior, mask_input_head=STOCK), STOCK),
        (dict(points=POINT, mask_input=prior, mask_input_head=TUNED,
              instance_cls="motherboard"), TUNED),
        # the tag counts only with the mask it describes
        (dict(box=BOX, mask_input_head=STOCK), TUNED),
        # a refinement nobody tagged -- a layer painted by hand -- is routed by the rule
        (dict(points=POINT, mask_input=prior, instance_cls="psu"), STOCK),
    ]
    for kwargs, head in cases:
        req = _req(CROP, multimask=True, **kwargs)
        result = svc.predict(req)
        assert (result.head, svc.head_for(req)) == (head, head), kwargs
        # every candidate is the answering head's: the toy scores by its decoder
        expected = TUNED_IOU if head == TUNED else 0.25
        assert result.score == pytest.approx(expected), kwargs
    # the stock modules are untouched: the head went onto copies
    assert float(built[0].sam_mask_decoder.iou.detach()[0]) == 0.25
    assert float(built[0].sam_prompt_encoder.point.detach()[1]) == 0.0


def test_the_two_heads_share_one_embedding_and_project_their_own_high_res_levels(
        toy, tmp_path):
    ckpt, built = toy
    svc = _service(ckpt, _config(_head(tmp_path)))
    model = built[0]
    svc.predict(_req(CROP, box=BOX))
    svc.predict(_req(CROP, points=POINT, instance_cls="psu"))
    svc.predict(_req(CROP.copy(), points=POINT, instance_cls="screw"))  # an equal copy
    assert model.image_encoder.calls == 1, "the image encoder ran more than once"

    stock, tuned = svc._predictors[STOCK], svc._predictors[TUNED]
    assert tuned._features["image_embed"] is stock._features["image_embed"]
    assert tuned._orig_hw == stock._orig_hw and tuned._is_image_set

    # the whole fine-tuned model, as L5 saved it: the stock weights with the delta
    whole = _Sam()
    state = whole.state_dict()
    state.update(_delta(whole))
    whole.load_state_dict(state)
    reference = _Predictor(whole)
    reference.set_image(CROP)
    for got, want, other in zip(tuned._features["high_res_feats"],
                                reference._features["high_res_feats"],
                                stock._features["high_res_feats"]):
        assert torch.equal(got, want)
        assert not torch.equal(got, other)      # L5 trained conv_s0 / conv_s1 too
    assert torch.equal(tuned._features["image_embed"], reference._features["image_embed"])

    # a new crop: both heads follow it
    other = CROP[::-1].copy()
    svc.predict(_req(other, box=BOX))
    assert model.image_encoder.calls == 2
    assert tuned._features["image_embed"] is stock._features["image_embed"]
    assert svc._raw_fpn is None, "the raw levels outlived set_image"


def test_without_heads_the_service_is_todays(toy, caplog):
    ckpt, _built = toy
    with caplog.at_level(logging.INFO):
        svc = _service(ckpt, None)
    assert svc.heads == (STOCK,) and svc.head_off_reason == "" and _warnings(caplog) == []
    for kwargs in (dict(box=BOX), dict(points=POINT, instance_cls="screw")):
        assert svc.predict(_req(CROP, **kwargs)).head == STOCK


def test_no_head_file_is_one_warning_and_the_stock_head_everywhere(toy, tmp_path, caplog):
    ckpt, _built = toy
    path = _write(tmp_path, _with(("head",), str(tmp_path / "gone.pt")))
    with caplog.at_level(logging.INFO):
        config = load_sam_head_config(str(path))
        svc = _service(ckpt, config)
    assert len(_warnings(caplog)) == 1
    assert svc.heads == (STOCK,) and "gone.pt does not exist" in svc.head_off_reason
    for kwargs in (dict(box=BOX, instance_cls="screw"), dict(points=POINT, instance_cls="screw"),
                   dict(points=POINT, instance_cls="motherboard")):
        result = svc.predict(_req(CROP, multimask=True, **kwargs))
        assert result.head == STOCK and len(result.candidates) == 3


def _foreign(blob):
    blob["state_dict"]["image_encoder.proj.weight"] = torch.zeros(8, 3, 1, 1)
    return blob


def _shape(blob):
    blob["state_dict"]["sam_mask_decoder.iou"] = torch.zeros(2)
    return blob


def _dtype(blob):
    blob["state_dict"]["sam_mask_decoder.iou"] = torch.tensor([0.75], dtype=torch.float64)
    return blob


def _stranger(blob):
    blob["state_dict"]["sam_mask_decoder.not_there"] = torch.zeros(1)
    return blob


def _other_checkpoint(blob):
    blob["base"]["fingerprint"]["no_mem_embed"] = "0" * 64
    return blob


def _no_fingerprint(blob):
    blob["base"] = {"file": "stock.pt"}
    return blob


@pytest.mark.parametrize("spoil, says", [
    ("garbage", "does not load"),
    (lambda b: {**b, "format": "something-else/9"}, "is not a tda-sam2-head/1 file"),
    (lambda b: {**b, "state_dict": {}}, "has no tensors"),
    (_foreign, "replaces tensors outside sam_prompt_encoder, sam_mask_decoder"),
    (_no_fingerprint, "does not say which SAM checkpoint"),
    (_other_checkpoint, "made for another SAM 2.1 checkpoint"),
    (_shape, "sam_mask_decoder.iou is torch.float32 (2,)"),
    (_dtype, "sam_mask_decoder.iou is torch.float64"),
    (_stranger, "sam_mask_decoder.not_there is not a tensor of this model"),
    ("manifest", "describes another file"),
])
def test_a_head_that_does_not_load_is_one_warning_and_the_stock_head(
        toy, tmp_path, caplog, spoil, says):
    ckpt, _built = toy
    if spoil == "garbage":
        head = tmp_path / "head.pt"
        head.write_bytes(b"\x00 not a torch file")
    elif spoil == "manifest":
        head = _head(tmp_path)
        head.with_suffix(".json").write_text(json.dumps({"head": {"sha256": "ab" * 32}}),
                                             encoding="utf-8")
    else:
        head = _head(tmp_path, spoil(_blob()))
    with caplog.at_level(logging.INFO):
        svc = _service(ckpt, _config(head))
    warned = _warnings(caplog)
    assert len(warned) == 1, [r.getMessage() for r in warned]
    assert "stock head answers every prompt" in warned[0].getMessage()
    assert says in svc.head_off_reason, svc.head_off_reason
    assert svc.heads == (STOCK,) and svc._hook is None
    assert svc.predict(_req(CROP, box=BOX, instance_cls="screw")).head == STOCK


def test_a_head_without_a_manifest_still_loads(toy, tmp_path, caplog):
    ckpt, _built = toy
    with caplog.at_level(logging.INFO):
        svc = _service(ckpt, _config(_head(tmp_path, manifest=False)))
    assert svc.tuned_loaded and _warnings(caplog) == []


def test_a_head_that_fails_mid_session_is_dropped_once(toy, tmp_path, caplog):
    """The projection cannot happen (the hook is gone): one WARNING, SAM keeps working."""
    ckpt, _built = toy
    svc = _service(ckpt, _config(_head(tmp_path)))
    svc._hook.remove()
    with caplog.at_level(logging.INFO):
        first = svc.predict(_req(CROP, box=BOX))
        again = svc.predict(_req(CROP[::-1].copy(), box=BOX))
    assert (first.head, again.head) == (STOCK, STOCK)
    assert len(_warnings(caplog)) == 1 and "were not captured" in svc.head_off_reason
    assert svc.heads == (STOCK,)


def test_the_queue_says_which_service_it_runs(toy):
    from tda.models.sam_service import SamQueue

    ckpt, _built = toy
    svc = _service(ckpt, None)
    queue = SamQueue(svc)
    try:
        assert queue.service is svc
    finally:
        queue.stop()
