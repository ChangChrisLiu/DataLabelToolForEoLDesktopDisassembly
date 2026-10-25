"""Task U19 on the real weights: the two-head service is the two models it replaces.

* the tuned head's logits equal those of ``sam2.1_hiera_large_tda_f5_v2.pt``
  loaded as a whole model (the 898 MB checkpoint L5 measured), and
* the stock head's equal today's service (``SamService()``, no head),

for four prompts on the fixture crop -- a point, a box, point+box, and a
refinement with ``mask_input`` -- in the app's bf16 and in fp32.  Then the
routing end to end: ``predict`` answers a box with the fine-tuned model's mask
and a lone click on a large part with today's.

Skipped unless CUDA, sam2, the stock checkpoint, the head file and the full
f5_v2 checkpoint are all there.  The three models are loaded one after the
other (one at a time on the GPU, which the annotator's app shares): about
fifteen seconds.  Measured 2026-10-17: every difference exactly 0.0.
"""
from __future__ import annotations

import contextlib
import gc
from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.models.sam_heads import DEFAULT_FILE, STOCK, TUNED, parse_sam_head_config
from tda.models.sam_service import (
    SamRequest,
    SamService,
    default_checkpoint,
    mask_to_low_res_logits,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "images" / "scanner_crop_native.png"
#: The whole fine-tuned model the shipped head was extracted from (L6's v3; L5's
#: was sam2.1_hiera_large_tda_f5_v2.pt).
FULL_F5_V2 = Path("D:/DataSet/models/weights/sam2.1_hiera_large_tda_l6_v3.pt")
FAN_POINT = (760.0, 270.0)            # tests/test_sam_service.py: a CPU cooler fan blade
FAN_BOX = (660.0, 95.0, 899.0, 335.0)
#: Logits are compared to this (the ruling's bf16 tolerance); measured: 0.0.
ATOL = 1e-3


def _heads():
    import yaml

    return parse_sam_head_config(yaml.safe_load(DEFAULT_FILE.read_text(encoding="utf-8")),
                                 DEFAULT_FILE)


def _missing() -> str:
    if not SamService.available():
        return "SAM 2.1 unavailable (needs CUDA torch, sam2 and the checkpoint)"
    head = _heads().head
    for path in (head, head.with_suffix(".json"), FULL_F5_V2):
        if not Path(path).is_file():
            return f"{path} is not here"
    return ""


pytestmark = pytest.mark.skipif(bool(_missing()), reason=_missing() or "-")


def _prior(shape) -> np.ndarray:
    prior = np.zeros(shape, bool)
    prior[120:320, 680:880] = True
    return prior


def _prompts(shape) -> dict:
    point = dict(point_coords=np.array([FAN_POINT], np.float32),
                 point_labels=np.array([1], np.int32))
    box = dict(box=np.array([FAN_BOX], np.float32))
    return {
        "point": dict(point, multimask_output=True),
        "box": dict(box, multimask_output=True),
        "point_box": dict(point, **box, multimask_output=True),
        "refine": dict(point_coords=np.array([FAN_POINT, (700.0, 300.0)], np.float32),
                       point_labels=np.array([1, 0], np.int32),
                       mask_input=mask_to_low_res_logits(_prior(shape)),
                       multimask_output=False),
    }


def _logits(svc: SamService, head: str, crop: np.ndarray, fp32: bool) -> dict:
    """``{prompt: (mask logits, scores, low-res logits)}`` from one head's predictor."""
    import torch

    if fp32:
        svc._autocast = contextlib.nullcontext
    else:
        svc.__dict__.pop("_autocast", None)   # the service's own: bf16 autocast
    svc._image_id = None                      # a fresh embedding in this precision
    svc.set_image(crop)
    out = {}
    for name, kwargs in _prompts(crop.shape[:2]).items():
        with torch.inference_mode(), svc._autocast():
            masks, scores, low = svc._predictors[head].predict(return_logits=True, **kwargs)
        out[name] = (np.asarray(masks), np.asarray(scores), np.asarray(low))
    return out


def _requests(crop) -> dict:
    return {
        "box": SamRequest(image_crop=crop, box=FAN_BOX, multimask=True, instance_cls="cpu_cooler"),
        "lone_click_large": SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)], multimask=True,
                                       instance_cls="cpu_cooler"),
        "lone_click_small": SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)], multimask=True,
                                       instance_cls="screw"),
    }


def _free(svc) -> None:
    import torch

    del svc
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def measured():
    bgr = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    crop = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = {"crop": crop}

    whole = SamService(checkpoint=str(FULL_F5_V2))      # L5's model, as a whole
    out["whole_masks"] = {k: whole.predict(r) for k, r in _requests(crop).items()}
    out["whole"] = {p: _logits(whole, STOCK, crop, fp32=p == "fp32") for p in ("bf16", "fp32")}
    _free(whole)

    today = SamService()                                 # the stock head alone
    out["today_masks"] = {k: today.predict(r) for k, r in _requests(crop).items()}
    out["today"] = {p: _logits(today, STOCK, crop, fp32=p == "fp32") for p in ("bf16", "fp32")}
    _free(today)

    two = SamService(checkpoint=str(default_checkpoint()), heads=_heads())
    out["heads"] = two.heads
    out["two_masks"] = {k: two.predict(r) for k, r in _requests(crop).items()}
    for head in (STOCK, TUNED):
        out[head] = {p: _logits(two, head, crop, fp32=p == "fp32") for p in ("bf16", "fp32")}
    _free(two)
    return out


def _max_diff(a: dict, b: dict) -> dict:
    return {k: max(float(np.abs(x - y).max()) for x, y in zip(a[k], b[k])) for k in a}


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_the_tuned_head_is_the_whole_tuned_model(measured, precision):
    assert measured["heads"] == (STOCK, TUNED)
    diff = _max_diff(measured[TUNED][precision], measured["whole"][precision])
    assert all(v <= ATOL for v in diff.values()), diff
    # and it is not the stock head: L5 changed the answers
    apart = _max_diff(measured[TUNED][precision], measured[STOCK][precision])
    assert all(v > 0.01 for v in apart.values()), apart


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_the_stock_head_is_todays_service(measured, precision):
    diff = _max_diff(measured[STOCK][precision], measured["today"][precision])
    assert all(v <= ATOL for v in diff.values()), diff


def test_predict_routes_each_prompt_to_the_model_it_stands_for(measured):
    two, whole, today = (measured["two_masks"], measured["whole_masks"],
                         measured["today_masks"])
    expect = {"box": (TUNED, whole), "lone_click_large": (STOCK, today),
              "lone_click_small": (TUNED, whole)}
    for name, (head, reference) in expect.items():
        got = two[name]
        assert got.head == head, name
        assert len(got.candidates) == len(reference[name].candidates) == 3
        for mine, theirs in zip(got.candidates, reference[name].candidates):
            assert np.array_equal(mine, theirs), name
        assert got.scores == pytest.approx(reference[name].scores, abs=ATOL)
