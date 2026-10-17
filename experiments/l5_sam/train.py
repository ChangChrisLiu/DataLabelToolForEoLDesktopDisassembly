"""Fine-tune SAM 2.1 Hiera-L's prompt encoder + mask decoder on verified scan masks.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.train --fold f13
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.train --fold f5
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.train --fold all
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l5_sam.train --check   # forward == predictor

The recipe is ``decision_rule.txt``'s (fixed before any result).  The image
encoder runs frozen under ``no_grad``; the decoder's high-res projections
(``conv_s0``/``conv_s1``, which ``SAM2Base.forward_image`` applies) are applied
here with gradients so the whole decoder trains.  Prompts are built the way the
predictor builds them (box corners as points labelled 2/3 first, then clicks,
``boxes=None`` so the prompt encoder appends its padding point), so the trained
weights see exactly the token layout the app sends.

Output: ``D:/DataSet/models/weights/sam2.1_hiera_large_tda_<fold>.pt`` -- a full
state dict ``{"model": ...}`` that ``sam2.build_sam.build_sam2`` loads with the
app's config -- plus ``OUT/train/<fold>/log.jsonl`` and ``manifest.json``.
"""
from __future__ import annotations

from experiments.l5_sam import env  # noqa: F401  (side effects first)

import argparse
import json
import math
import queue
import random
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

from experiments.l5_sam import common as C
from experiments.l5_sam import gt as G

FOLDS = {"f13": (2, 3, 4, 5), "f5": (2, 3, 4, 13), "all": (2, 3, 4, 5, 13)}

# --- the recipe (decision_rule.txt) -----------------------------------------
LR = 2e-5
WD = 0.01
WARMUP = 200
LR_FLOOR = 0.1
CLIP = 0.1
CROPS_PER_STEP = 4
MAX_TARGETS_PER_CROP = 8
EPOCHS = 4
ROUNDS = 3
W_FOCAL, W_DICE, W_IOU = 20.0, 1.0, 1.0
CTX_RANGE = (1.2, 3.0)
SHIFT = 0.25
BOX_NOISE_FRAC, BOX_NOISE_MIN, BOX_NOISE_MAX = 0.1, 2.0, 12.0
#: Recipe v2 (decision_rule.txt, amendment 1): share of loose box prompts and
#: the largest outward push per edge, as a fraction of the box side.
RECIPES = {"v1": 0.0, "v2": 0.5}
LOOSE_MAX = 0.25
POINT_DEPTH = 0.3
MASK_LOGIT_SCALE = 8.0      # tda.models.sam_service.MASK_LOGIT_SCALE
RES = 1024
SEED = 0
PROMPT_KINDS = ("box", "point", "point_box")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Sampler:
    """Builds one training crop with up to MAX_TARGETS_PER_CROP targets (CPU)."""

    def __init__(self, recs: list[G.Rec], frames: dict, seed: int, loose_frac: float = 0.0) -> None:
        self.loose_frac = loose_frac
        self.recs = recs
        self.frames = frames
        self.by_frame = defaultdict(list)
        for i, r in enumerate(recs):
            self.by_frame[r.frame].append(i)
        self.rng = np.random.default_rng(seed)

    def window(self, rec: G.Rec) -> C.Window:
        h, w = rec.frame_hw
        x0, y0, x1, y1 = rec.bbox
        longest = max(x1 - x0, y1 - y0)
        if longest * CTX_RANGE[0] <= C.MAX_WINDOW:
            side = int(round(longest * self.rng.uniform(*CTX_RANGE)))
            side = min(max(side, C.MIN_WINDOW), C.MAX_WINDOW)
        else:                                       # a large part: keep it whole
            side = max(longest, int(round(longest * self.rng.uniform(1.05, 1.3))))
        side = min(side, min(h, w))
        scale = min(1.0, C.MAX_WINDOW / float(side))
        cx = (x0 + x1) / 2.0 + self.rng.uniform(-SHIFT, SHIFT) * side
        cy = (y0 + y1) / 2.0 + self.rng.uniform(-SHIFT, SHIFT) * side
        wx0 = int(round(cx - side / 2.0))
        wy0 = int(round(cy - side / 2.0))
        # keep the target inside (when it fits) and the window inside the frame
        lo_x, hi_x = (x1 - side, x0) if x1 - x0 <= side else (x0, x0)
        lo_y, hi_y = (y1 - side, y0) if y1 - y0 <= side else (y0, y0)
        wx0 = min(max(wx0, lo_x), hi_x)
        wy0 = min(max(wy0, lo_y), hi_y)
        wx0 = max(0, min(wx0, w - side))
        wy0 = max(0, min(wy0, h - side))
        return C.Window(wx0, wy0, wx0 + side, wy0 + side, scale)

    def _prompts(self, m: np.ndarray, kind: str):
        """(points [(x, y, label)], box or None) in model-crop pixels for mask ``m``."""
        ys, xs = np.nonzero(m)
        pts, box = [], None
        if kind in ("box", "point_box"):
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            if self.loose_frac > 0 and self.rng.random() < self.loose_frac:
                # v2 (amendment 1): the difference-map case, each edge pushed out
                u = self.rng.uniform(0, LOOSE_MAX, size=4)
                bw, bh = x1 - x0, y1 - y0
                x0, y0 = x0 - u[0] * bw, y0 - u[1] * bh
                x1, y1 = x1 + u[2] * bw, y1 + u[3] * bh
            nx = min(BOX_NOISE_MAX, max(BOX_NOISE_MIN, BOX_NOISE_FRAC * (x1 - x0)))
            ny = min(BOX_NOISE_MAX, max(BOX_NOISE_MIN, BOX_NOISE_FRAC * (y1 - y0)))
            d = self.rng.uniform(-1, 1, size=4) * np.array([nx, ny, nx, ny])
            bx0, by0, bx1, by1 = x0 + d[0], y0 + d[1], x1 + d[2], y1 + d[3]
            if bx1 - bx0 < 2:
                bx0, bx1 = x0, x1
            if by1 - by0 < 2:
                by0, by1 = y0, y1
            s = m.shape[0]
            box = (float(np.clip(bx0, 0, s)), float(np.clip(by0, 0, s)),
                   float(np.clip(bx1, 0, s)), float(np.clip(by1, 0, s)))
        if kind in ("point", "point_box"):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            sub = np.pad(m[y0:y1, x0:x1].astype(np.uint8), 1)
            dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)[1:-1, 1:-1]
            cand = np.argwhere(dist >= POINT_DEPTH * dist.max())
            cy, cx = cand[self.rng.integers(len(cand))]
            pts.append((float(x0 + cx + 0.5), float(y0 + cy + 0.5), 1))
        return pts, box

    def crop(self, primary: int, kind: str) -> dict:
        rec = self.recs[primary]
        win = self.window(rec)
        frame = self.frames[rec.frame]
        img = C.crop_image(frame, win)
        others = [i for i in self.by_frame[rec.frame] if i != primary
                  and self.recs[i].bbox[0] >= win.x0 and self.recs[i].bbox[1] >= win.y0
                  and self.recs[i].bbox[2] <= win.x1 and self.recs[i].bbox[3] <= win.y1
                  and self.recs[i].area >= 16]
        self.rng.shuffle(others)
        chosen = [primary] + others[:MAX_TARGETS_PER_CROP - 1]
        hflip, vflip = self.rng.random() < 0.5, self.rng.random() < 0.5
        if hflip:
            img = img[:, ::-1]
        if vflip:
            img = img[::-1]
        side = img.shape[0]
        masks, prompts = [], []
        for i in chosen:
            m = self.recs[i].mask_in(win.rect)
            if win.oversize:
                m = cv2.resize(m.astype(np.uint8), (side, side),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if hflip:
                m = m[:, ::-1]
            if vflip:
                m = m[::-1]
            m = np.ascontiguousarray(m)
            if not m.any():
                continue
            masks.append(m)
            prompts.append(self._prompts(m, kind))
        return {"img": np.ascontiguousarray(img), "masks": masks, "prompts": prompts,
                "side": side, "kind": kind}


def batches(sampler: Sampler, order: list[int], step0: int, n_steps: int):
    for s in range(step0, step0 + n_steps):
        kind = PROMPT_KINDS[s % len(PROMPT_KINDS)]
        idx = order[(s - step0) * CROPS_PER_STEP:(s - step0 + 1) * CROPS_PER_STEP]
        yield [sampler.crop(i, kind) for i in idx]


def prefetch(gen, depth: int = 3):
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    end = object()

    def run():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(end)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            return
        yield item


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(ckpt: str, device: str = "cuda"):
    from sam2.build_sam import build_sam2

    model = build_sam2(env.SAM2_CONFIG, ckpt, device=device, mode="eval")
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(model.sam_prompt_encoder.parameters()) + list(model.sam_mask_decoder.parameters())
    for p in trainable:
        p.requires_grad_(True)
    return model, trainable


_MEAN = None
_STD = None


def preprocess(torch, imgs: list[np.ndarray], device: str = "cuda"):
    """The predictor's SAM2Transforms on the GPU: [0,1], bilinear antialiased resize, normalise."""
    import torch.nn.functional as F

    global _MEAN, _STD
    if _MEAN is None:
        _MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for im in imgs:
        x = torch.from_numpy(im).to(device).permute(2, 0, 1).float().div_(255.0)[None]
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
        out.append(x)
    return (torch.cat(out) - _MEAN) / _STD


def image_features(torch, model, x):
    """(image_embed, [feat_s0, feat_s1]) as the predictor's set_image, decoder convs with grad."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bo = model.image_encoder(x)
    fpn = [f.detach() for f in bo["backbone_fpn"]]
    dec = model.sam_mask_decoder
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s0 = dec.conv_s0(fpn[0])
        s1 = dec.conv_s1(fpn[1])
    emb = fpn[-1] + model.no_mem_embed.view(1, -1, 1, 1)
    return emb, [s0, s1]


def decode(torch, model, emb, hr, coords, labels, mask_in, multimask: bool):
    """Prompt encoder + mask decoder, the predictor's ``_predict`` minus the upsampling."""
    pe = model.sam_prompt_encoder
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sparse, dense = pe(points=(coords, labels), boxes=None, masks=mask_in)
        low, iou_pred, _, _ = model.sam_mask_decoder(
            image_embeddings=emb, image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=multimask, repeat_image=False, high_res_features=hr)
    return low.float(), iou_pred.float()


def focal(torch, logits, target, alpha=0.25, gamma=2.0):
    import torch.nn.functional as F

    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.flatten(-2).mean(-1)


def dice(torch, logits, target):
    p = logits.sigmoid().flatten(-2)
    t = target.flatten(-2)
    return 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)


def round_loss(torch, low, iou_pred, gt1024):
    """SAM 2's loss: min over the masks of 20*focal + dice, L1 IoU on every mask."""
    import torch.nn.functional as F

    up = F.interpolate(low, size=(RES, RES), mode="bilinear", align_corners=False)  # T,K,H,W
    tgt = gt1024[:, None].expand_as(up)
    lf = focal(torch, up, tgt)
    ld = dice(torch, up, tgt)
    comb = W_FOCAL * lf + W_DICE * ld                               # T,K
    best = comb.argmin(dim=1)
    ar = torch.arange(up.shape[0], device=up.device)
    with torch.no_grad():
        pb = up > 0
        inter = (pb & (tgt > 0.5)).flatten(-2).sum(-1).float()
        union = (pb | (tgt > 0.5)).flatten(-2).sum(-1).float()
        actual = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    li = (iou_pred - actual).abs().mean(dim=1)
    loss = comb[ar, best] + W_IOU * li
    return loss.mean(), up, actual


def to_mask_input(torch, binary1024):
    """The app's mask_to_low_res_logits: 256x256 bilinear-ish resize, +/-8 logits."""
    import torch.nn.functional as F

    small = F.interpolate(binary1024[:, None].float(), size=(256, 256), mode="area")
    return (small - 0.5) * (2.0 * MASK_LOGIT_SCALE)


def train_step(torch, model, batch, device="cuda"):
    import torch.nn.functional as F

    x = preprocess(torch, [b["img"] for b in batch], device)
    emb, (s0, s1) = image_features(torch, model, x)
    img_idx, gts, coords, labels = [], [], [], []
    for bi, b in enumerate(batch):
        f = RES / float(b["side"])
        for m, (pts, box) in zip(b["masks"], b["prompts"]):
            img_idx.append(bi)
            gts.append(torch.from_numpy(m))
            c, l = [], []
            if box is not None:
                c += [(box[0] * f, box[1] * f), (box[2] * f, box[3] * f)]
                l += [2, 3]
            for (px, py, lab) in pts:
                c.append((px * f, py * f))
                l.append(lab)
            coords.append(c)
            labels.append(l)
    T = len(img_idx)
    idx = torch.tensor(img_idx, device=device)
    emb_t, s0_t, s1_t = emb[idx], s0[idx], s1[idx]
    gt1024 = torch.stack([
        F.interpolate(g[None, None].float().to(device), size=(RES, RES), mode="bilinear",
                      align_corners=False)[0, 0] for g in gts]) > 0.5
    gt1024 = gt1024.float()
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.int32, device=device)

    total = 0.0
    stats = {}
    mask_in = None
    for r in range(ROUNDS):
        multimask = r == 0
        low, iou_pred = decode(torch, model, emb_t, [s0_t, s1_t], coords_t, labels_t, mask_in,
                               multimask)
        loss, up, actual = round_loss(torch, low, iou_pred, gt1024)
        total = total + loss
        with torch.no_grad():
            shown = iou_pred.argmax(dim=1)                       # what the app shows
            ar = torch.arange(T, device=device)
            stats[f"r{r + 1}_iou"] = float(actual[ar, shown].mean())
            if r == 0:
                stats["r1_oracle"] = float(actual.max(dim=1).values.mean())
            if r + 1 < ROUNDS:
                pred = up[ar, shown] > 0                         # T,H,W
                gt_b = gt1024 > 0.5
                union = pred | gt_b
                rows = union.any(dim=2).cpu().numpy()           # T,H
                cols = union.any(dim=1).cpu().numpy()           # T,W
                pred_np = pred.cpu().numpy()
                gt_np = gt_b.cpu().numpy()
                new_c, new_l = [], []
                for t in range(T):
                    ry, rx = np.flatnonzero(rows[t]), np.flatnonzero(cols[t])
                    y0, y1, x0, x1 = ry[0], ry[-1] + 1, rx[0], rx[-1] + 1
                    p_roi, g_roi = pred_np[t, y0:y1, x0:x1], gt_np[t, y0:y1, x0:x1]
                    click = C.corrective_click(p_roi, g_roi)
                    if click is None:   # perfect: a positive click deep inside
                        ys, xs = np.nonzero(g_roi)
                        k = len(ys) // 2
                        click = (int(xs[k]), int(ys[k]), 1)
                    new_c.append((x0 + click[0] + 0.5, y0 + click[1] + 0.5))
                    new_l.append(click[2])
                coords_t = torch.cat([coords_t, torch.tensor(new_c, device=device)[:, None]], dim=1)
                labels_t = torch.cat([labels_t, torch.tensor(new_l, dtype=torch.int32,
                                                             device=device)[:, None]], dim=1)
                mask_in = to_mask_input(torch, pred)
        stats[f"loss_r{r + 1}"] = float(loss.detach())
    stats["targets"] = T
    return total, stats


# --------------------------------------------------------------------------- #
# check: this forward path == SAM2ImagePredictor
# --------------------------------------------------------------------------- #
def check() -> int:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    env.limit_threads()
    model, _ = build_model(str(env.SAM2_CKPT))
    model.eval()
    pred = SAM2ImagePredictor(model)
    recs, images = G.load()
    rec = [r for r in recs if r.cls == "screw" and r.desktop == 2][5]
    frame = G.load_frames(images, {rec.frame})[rec.frame]
    win = C.crop_window(rec.bbox, rec.frame_hw)
    crop = C.crop_image(frame, win)
    px, py = C.to_crop((rec.point[0] + 0.5, rec.point[1] + 0.5), win)
    box = C.box_in_crop(C.jittered_box(rec.bbox, 1), win)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(crop)
        m_ref, s_ref, low_ref = pred.predict(point_coords=np.array([[px, py]]), point_labels=np.array([1]),
                                             box=np.array(box), multimask_output=True,
                                             normalize_coords=True, return_logits=True)
    with torch.no_grad():
        x = preprocess(torch, [crop])
        emb, hr = image_features(torch, model, x)
        f = RES / float(crop.shape[0])
        coords = torch.tensor([[(box[0] * f, box[1] * f), (box[2] * f, box[3] * f), (px * f, py * f)]],
                              device="cuda")
        labels = torch.tensor([[2, 3, 1]], dtype=torch.int32, device="cuda")
        low, iou = decode(torch, model, emb, hr, coords, labels, None, True)
        low = low.clamp(-32.0, 32.0)
    d_low = float((low.cpu() - torch.as_tensor(low_ref).float().cpu()).abs().max())
    d_iou = float((iou.cpu() - torch.as_tensor(s_ref).float().cpu()).abs().max())
    agree = float(((low.cpu() > 0) == (torch.as_tensor(low_ref).cpu() > 0)).float().mean())
    print(f"[check] max |dlogit| {d_low:.4f}, max |dIoU| {d_iou:.4f}, sign agreement {agree:.5f}")
    print(f"[check] predictor iou {np.round(np.asarray(s_ref), 4)} vs mine {iou.cpu().numpy().round(4)}")
    return 0 if agree > 0.995 else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", choices=sorted(FOLDS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="v1")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    if not args.fold:
        ap.error("--fold is required")

    import torch

    env.limit_threads()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    free = env.gpu_free_gb()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    budget = free - env.MIN_FREE_GB
    if budget < 8:
        raise SystemExit(f"only {free:.1f} GiB free; {env.MIN_FREE_GB} must stay free")
    torch.cuda.set_per_process_memory_fraction(min(0.75, budget / total_gb))
    print(f"[train] {free:.1f} GiB free; capping this process at {min(0.75, budget / total_gb) * total_gb:.1f} GiB",
          flush=True)

    tag = args.tag or ("" if args.recipe == "v1" else args.recipe)
    name = args.fold + (f"_{tag}" if tag else "")
    out = env.OUT / "train" / name
    out.mkdir(parents=True, exist_ok=True)
    recs, images = G.load()
    recs = [r for r in recs if r.desktop in FOLDS[args.fold]]
    frames = G.load_frames(images, {r.frame for r in recs})
    sampler = Sampler(recs, frames, SEED, loose_frac=RECIPES[args.recipe])
    steps_per_epoch = len(recs) // CROPS_PER_STEP
    n_steps = int(round(args.epochs * steps_per_epoch))
    print(f"[train] fold {args.fold}: desktops {FOLDS[args.fold]}, {len(recs)} masks, "
          f"{len(frames)} frames, {steps_per_epoch} steps/epoch, {n_steps} steps", flush=True)

    model, trainable = build_model(str(env.SAM2_CKPT))
    model.sam_mask_decoder.train()
    model.sam_prompt_encoder.train()
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WD)

    def lr_at(s):
        if s < WARMUP:
            return args.lr * (s + 1) / WARMUP
        p = (s - WARMUP) / max(1, n_steps - WARMUP)
        return args.lr * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

    rng = np.random.default_rng(SEED)
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    t0 = time.perf_counter()
    step = 0
    acc = defaultdict(float)
    acc_n = 0
    stopped = False
    epoch_means = []
    while step < n_steps and not stopped:
        order = list(rng.permutation(len(recs)))
        this = min(steps_per_epoch, n_steps - step)
        ep_loss, ep_n = 0.0, 0
        for batch in prefetch(batches(sampler, order, step, this)):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, st = train_step(torch, model, batch)
            if not torch.isfinite(loss):
                raise SystemExit(f"[train] non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, CLIP)
            opt.step()
            step += 1
            ep_loss += float(loss.detach())
            ep_n += 1
            for k, v in st.items():
                acc[k] += v
            acc["grad_norm"] += float(gn)
            acc_n += 1
            if step % 50 == 0 or step == n_steps:
                rec = {k: round(v / acc_n, 5) for k, v in acc.items()}
                rec.update(step=step, lr=lr_at(step), s=round(time.perf_counter() - t0, 1),
                           mem_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                el = time.perf_counter() - t0
                print(f"[train] {step}/{n_steps} loss {rec['loss_r1']:.3f}/{rec['loss_r2']:.3f}/"
                      f"{rec['loss_r3']:.3f} r1 IoU {rec['r1_iou']:.3f} (oracle {rec['r1_oracle']:.3f}) "
                      f"r3 IoU {rec['r3_iou']:.3f} | {el:.0f} s, ETA {el / step * (n_steps - step):.0f} s, "
                      f"peak {rec['mem_gb']} GiB", flush=True)
                acc.clear()
                acc_n = 0
            if step % 100 == 0 and env.stop_requested():
                print("[train] STOP file present: saving and stopping", flush=True)
                stopped = True
                break
        epoch_means.append(ep_loss / max(1, ep_n))
        print(f"[train] epoch {len(epoch_means)} mean loss {epoch_means[-1]:.4f}", flush=True)
    log.close()

    ckpt = env.WEIGHTS / f"sam2.1_hiera_large_tda_{name}{'_partial' if stopped else ''}.pt"
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": sd}, ckpt)
    manifest = {
        "fold": args.fold, "desktops": FOLDS[args.fold], "masks": len(recs), "steps": step,
        "planned_steps": n_steps, "epochs": args.epochs, "lr": args.lr, "stopped": stopped,
        "epoch_mean_loss": epoch_means, "train_s": round(time.perf_counter() - t0, 1),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        "checkpoint": str(ckpt), "init": str(env.SAM2_CKPT), "config": env.SAM2_CONFIG,
        "recipe": {"name": args.recipe, "loose_frac": RECIPES[args.recipe], "loose_max": LOOSE_MAX,
                   "wd": WD, "warmup": WARMUP, "lr_floor": LR_FLOOR, "clip": CLIP,
                   "crops_per_step": CROPS_PER_STEP, "max_targets": MAX_TARGETS_PER_CROP,
                   "rounds": ROUNDS, "loss": [W_FOCAL, W_DICE, W_IOU], "ctx": CTX_RANGE,
                   "shift": SHIFT, "seed": SEED},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[train] saved {ckpt} after {step} steps in {manifest['train_s']:.0f} s; "
          f"peak {manifest['peak_alloc_gb']} GiB allocated / {manifest['peak_reserved_gb']} reserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
