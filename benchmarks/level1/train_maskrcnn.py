#!/usr/bin/env python
"""Baseline (b): torchvision Mask R-CNN ResNet-50-FPN, fine-tuned from COCO weights on a make_coco.py split.

    python train_maskrcnn.py --train coco/train.json [--val coco/val.json] --run runs/maskrcnn_visible \
        [--size 1280] [--epochs 24] [--batch 4] [--lr 0.02] [--seed 0] [--weights coco] [--arch v2]

* Input: every frame scaled to `--size` on its longer side (the frames are 1600 x 1600); random horizontal flip.
* Schedule: SGD, momentum 0.9, weight decay 1e-4, lr = --lr x batch / 16, linear warm-up over the first
  min(1000, 1/10 of all) iterations, x 0.1 at 2/3 and 8/9 of the epochs; mixed precision on CUDA.
* `iscrowd = 1` rows (ignore regions) are not training targets (torchvision has no ignore regions).
* COCO weights come from torchvision's model zoo into $TORCH_HOME/hub/checkpoints (offline: copy the
  file `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` there first, or pass --weights <path>).
* Every --val-every epochs the model is run on --val and its segm AP logged; best.pt keeps the best one.
* The run folder gets run.json, train.log, metrics.jsonl (one line per logged step / validation),
  last.pt (resumable with --resume) and best.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1common import Logger, dump_json, env_report, set_seed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--arch", choices=("v2", "v1"), default="v2")
    ap.add_argument("--weights", default="coco", help="coco | imagenet | none | path to a .pth / checkpoint")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02, help="base lr for batch 16 (scaled linearly)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | cuda:1 | cpu (default: cuda if available)")
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-iters", type=int, default=0, help="stop after this many iterations (smoke tests)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from maskrcnn_lib import CocoInstanceDataset, build_model, collate, predict

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    log = Logger(run / "train.log")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda" and not args.no_amp

    ds = CocoInstanceDataset(args.train, args.size, train=True, image_root_override=args.image_root)
    num_classes = len(ds.cat_ids)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    generator=g, drop_last=len(ds) >= args.batch, persistent_workers=args.workers > 0,
                    pin_memory=device.type == "cuda")
    log(f"train {args.train}: {len(ds)} images, {num_classes} classes, device {device}, amp {amp}")
    model = build_model(num_classes, args.size, args.arch, args.weights).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr * args.batch / 16
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    iters_per_epoch = len(dl)
    total = iters_per_epoch * args.epochs
    warm = max(1, min(1000, total // 10))
    steps = [int(round(args.epochs * 2 / 3)) * iters_per_epoch, int(round(args.epochs * 8 / 9)) * iters_per_epoch]

    def lr_at(it):
        f = 0.1 ** sum(it >= s for s in steps)
        if it < warm:
            f *= 0.001 + (1 - 0.001) * it / warm
        return lr * f

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, it, best_ap = 0, 0, -1.0
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, it, best_ap = ck["epoch"] + 1, ck["iter"], ck.get("best_ap", -1.0)
        log(f"resumed from epoch {ck['epoch']} (iter {it})")

    meta = {"baseline": "torchvision_maskrcnn_r50_fpn_" + args.arch, "args": vars(args), "device": str(device),
            "num_classes": num_classes, "label_to_category_id": ds.label_to_cat, "lr": lr,
            "iters_per_epoch": iters_per_epoch, "lr_steps": steps, "warmup": warm, "env": env_report(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump_json(meta, run / "run.json", indent=1)
    mlog = open(run / "metrics.jsonl", "a", encoding="utf-8")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "iter": it, "best_ap": best_ap, "num_classes": num_classes, "arch": args.arch,
                    "size": args.size, "label_to_category_id": ds.label_to_cat}, path)

    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        te = time.time()
        for images, targets in dl:
            for gp in opt.param_groups:
                gp["lr"] = lr_at(it)
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
                       for t in targets]
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                log(f"non-finite loss at iter {it}: {({k: v.item() for k, v in losses.items()})}; stopping")
                sys.exit(1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if it % args.log_every == 0:
                rec = {"iter": it, "epoch": epoch, "lr": lr_at(it), "loss": round(loss.item(), 4),
                       **{k: round(v.item(), 4) for k, v in losses.items()}}
                mlog.write(json.dumps(rec) + "\n")
                mlog.flush()
                log(f"ep {epoch} it {it}/{total} loss {loss.item():.3f} "
                    + " ".join(f"{k[5:]}={v.item():.3f}" for k, v in losses.items()) + f" lr {lr_at(it):.2e}"
                    + (f" gpu_peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB"
                       if device.type == "cuda" else "")
                    + f" {(time.time() - te) / max(1, it - epoch * iters_per_epoch + 1):.2f}s/it")
            it += 1
            if args.max_iters and it >= args.max_iters:
                stop = True
                break
        log(f"epoch {epoch} done in {time.time() - te:.0f}s")
        save(run / "last.pt", epoch)
        last_epoch = epoch + 1 == args.epochs or stop
        if args.val and ((epoch + 1) % args.val_every == 0 or last_epoch):
            from eval_coco import evaluate
            preds = predict(model, args.val, args.size, device, ds.label_to_cat, args.image_root, log=log, amp=amp)
            res = evaluate(args.val, preds, ["segm"], [])
            summ = res["metrics"]["segm"]["summary"] if res["metrics"]["segm"] else {"AP": 0.0, "AP50": 0.0}
            ap = summ["AP"]
            mlog.write(json.dumps({"epoch": epoch, "val_segm_AP": ap, "val_segm_AP50": summ["AP50"]}) + "\n")
            mlog.flush()
            log(f"epoch {epoch}: val segm AP {ap:.4f}")
            if ap > best_ap:
                best_ap = ap
                save(run / "best.pt", epoch)
        if stop:
            break
    if not (run / "best.pt").exists():
        save(run / "best.pt", epoch)  # no validation: the last weights
    secs = time.time() - t0
    meta.update({"train_seconds": round(secs, 1), "iters": it, "best_val_segm_AP": best_ap,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    dump_json(meta, run / "run.json", indent=1)
    log(f"training done in {secs / 60:.1f} min, {it} iterations")
    mlog.close()


if __name__ == "__main__":
    main()
