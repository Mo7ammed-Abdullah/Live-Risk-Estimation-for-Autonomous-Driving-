"""
Fine-tune UniDepthV2 on your nuScenes LiDAR depth (train split), validate on the validation split.

Built for an 8 GB laptop GPU: mixed precision, one image per forward pass, gradients accumulated over several images.
The camera intrinsics are given to the model (as in your zero-shot tests). The loss is computed only at the pixels that
have a LiDAR measurement (about 0.2 % of the image): scale-invariant log loss + relative error.

Run with venv311 active, from "Testing 3":
    python train_depth.py --epochs 1 --limit_train 32 --limit_val 16      # 2-minute smoke test first
    python train_depth.py                                                  # real run (6 epochs)
    python train_depth.py --resume                                         # continue after an interruption
    python train_depth.py --train_pixels 300000                            # if you run out of GPU memory
    python train_depth.py --freeze_encoder                                 # train only the decoder (less memory, less overfitting)

Outputs in runs/<name>/ :
    best.pt          best model by validation AbsRel; works as --depth_model runs\\depth_s\\best.pt in every other script
    last_state.pt    everything needed to resume
    history.json     validation numbers after every epoch (epoch 0 = the untouched pretrained model)
    config.json      the settings used

Validation here is the validation split only. The test split is never touched by this script.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import evalkit as ek
# Reuse the model loader and the inference function from eval_depth.py.
from eval_depth import load_depth_model, run_depth

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# One training sample: the photo, the pixels that have a LiDAR depth (row, column, depth in meters) and the camera intrinsics.
# LiDAR is sparse (about 0.2% of the pixels), so the loss is computed only at those pixels.
class DepthFrames(torch.utils.data.Dataset):
    """One frame: RGB uint8, the valid LiDAR pixels (row, col, depth in m) and the 3x3 intrinsics. Flip augmentation here."""

    def __init__(self, stems, images, depth_gt, meta, max_depth, max_points, train):
        self.stems, self.images, self.depth_gt, self.meta = stems, images, depth_gt, meta
        self.max_depth, self.max_points, self.train = max_depth, max_points, train

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        rgb = np.array(Image.open(self.images / f"{stem}.jpg").convert("RGB"))
        gt = np.array(Image.open(self.depth_gt / f"{stem}.png")).astype(np.float32) / 256.0
        K = np.array(self.meta["cs"][self.meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"],
                     dtype=np.float32)
        # Valid LiDAR pixels; if there are more than max_points, a random subset is used.
        ys, xs = np.nonzero((gt > 0) & (gt <= self.max_depth))
        if len(ys) > self.max_points:
            sel = np.random.choice(len(ys), self.max_points, replace=False)
            ys, xs = ys[sel], xs[sel]
        d = gt[ys, xs]
        H, W = gt.shape
        # Augmentation: horizontal flip. The camera's principal point is flipped with the image, otherwise depth would be wrong.
        if self.train and random.random() < 0.5:                      # horizontal flip (intrinsics flip with it)
            rgb = rgb[:, ::-1].copy()
            xs = W - 1 - xs
            K = K.copy()
            K[0, 2] = W - K[0, 2]
        return (torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(ys.astype(np.float32)),
                torch.from_numpy(xs.astype(np.float32)), torch.from_numpy(d), torch.from_numpy(K), stem)


# The same preparation as the model's own infer() (padding, resize, normalise, camera), but WITH gradients so the model can learn.
# Returns the predicted depth map and the padding information.
def forward_depth(model, rgb_u8, K, device, train_aug=False, fp16=True):
    """Same pre-processing as model.infer(), but with gradients. Returns (depth map (1,1,h,w), padded size)."""
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor
    from unidepth.utils.camera import BatchCamera, Pinhole
    from unidepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
    import torchvision.transforms.v2.functional as TF

    sc = model.shape_constraints
    H, W = rgb_u8.shape[-2:]
    paddings, (pH, pW) = get_paddings((H, W), sc["ratio_bounds"])
    pl, pr, pt, pb = paddings
    factor, (nH, nW) = get_resize_factor((pH, pW), (sc["pixels_min"], sc["pixels_max"]))
    x = rgb_u8.unsqueeze(0).to(device).float() / 255.0
    if train_aug:                                                     # light brightness / contrast jitter
        # Light brightness / contrast jitter during training.
        x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5) * random.uniform(0.8, 1.2)
        x = x.clamp(0, 1)
    x = TF.normalize(x, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD)
    x = F.pad(x, (pl, pr, pt, pb), value=0.0)
    x = F.interpolate(x, size=(nH, nW), mode="bilinear", align_corners=False)
    # The camera intrinsics must be cropped and resized exactly like the image.
    cam = BatchCamera.from_camera(Pinhole(K=K.to(device))).to(device)
    cam = cam.crop(left=-pl, top=-pt, right=-pr, bottom=-pb).resize(factor)
    with torch.autocast(device_type="cuda", enabled=fp16 and device.type == "cuda", dtype=torch.float16):
        _, out = model.encode_decode(inputs={"image": x, "camera": cam}, image_metas=[])
    return out["depth"].float(), (pH, pW), (pl, pt)


# Read the predicted depth at the LiDAR pixels. The predicted map covers the padded and resized image, so coordinates are converted first.
def sample_at(depth_map, ys, xs, padded_hw, pad_lt):
    """Read the predicted depth at original-image pixel (ys, xs); depth_map covers the padded, resized image."""
    pH, pW = padded_hw
    gx = ((xs + pad_lt[0] + 0.5) / pW) * 2 - 1
    gy = ((ys + pad_lt[1] + 0.5) / pH) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    return F.grid_sample(depth_map, grid, mode="bilinear", align_corners=False).view(-1)


# Loss = scale-invariant log error (silog) + relative error, both weighted. It is computed only at LiDAR pixels.
def depth_loss(pred, gt, w_silog, w_rel, lam=0.85):
    pred = pred.clamp(min=1e-3)
    d = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt((d ** 2).mean() - lam * d.mean() ** 2 + 1e-8)
    rel = ((pred - gt).abs() / gt).mean()
    return w_silog * silog + w_rel * rel, float(rel)


@torch.no_grad()
# Validation: run the model on the validation frames with the real camera and collect the depth metrics.
def validate(model, stems, args, meta, device):
    model.eval()
    meter = ek.DepthMeter(args.max_depth)
    for stem in tqdm(stems, desc="validating", leave=False):
        rgb = np.array(Image.open(args.images / f"{stem}.jpg").convert("RGB"))
        gt = np.array(Image.open(args.depth_gt / f"{stem}.png")).astype(np.float32) / 256.0
        K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        depth = run_depth(model, rgb, K)
        if depth.shape == gt.shape:
            meter.update(depth, gt, tags=(ek.city_tag(stem),))
    model.train()
    return meter.results()


# One-line summary of the validation: AbsRel, RMSE, d1 and AbsRel by distance range.
def short(res):
    a = res["all"]
    bins = {k: round(v["AbsRel"], 3) for k, v in res.items() if k.endswith(" m")}
    return f"AbsRel {a['AbsRel']:.4f} | RMSE {a['RMSE']:.2f} | d1 {a['d1']:.3f} | by distance {bins}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vits14", help="starting model (hub id)")
    ap.add_argument("--name", default="depth_s")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--grad_accum", type=int, default=8, help="images per optimizer step (one image per forward pass)")
    ap.add_argument("--lr_dec", type=float, default=5e-5)
    ap.add_argument("--lr_enc", type=float, default=5e-6)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=50, help="optimizer steps")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--w_silog", type=float, default=10.0)
    ap.add_argument("--w_rel", type=float, default=10.0)
    ap.add_argument("--max_points", type=int, default=6000, help="LiDAR pixels used per image in the loss")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--train_pixels", type=int, default=0, help="cap the training resolution (pixels); 0 = model default")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--skip_first_eval", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--depth_gt", type=Path, default=HERE / "depth_gt" / "depth" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_root", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    # Fixed random seeds for repeatable runs.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = ek.get_device()
    if device.type != "cuda":
        print("WARNING: no GPU found, this will be very slow")
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(args).items()}, open(out / "config.json", "w"), indent=2)

    meta = json.load(open(args.meta_cache))
    train_stems = ek.load_split("train", False, 1, args.limit_train)
    val_stems = ek.load_split("val", False, 1, args.limit_val)
    # Use only frames that have both a photo and a LiDAR depth map.
    ok = lambda s: (args.images / f"{s}.jpg").exists() and (args.depth_gt / f"{s}.png").exists()
    train_stems, val_stems = [s for s in train_stems if ok(s)], [s for s in val_stems if ok(s)]
    print(f"train {len(train_stems)} images | val {len(val_stems)} images | device {device}")

    # Start from the pretrained UniDepthV2-small and fine-tune it.
    model = load_depth_model(args.model, device)
    # Option for small GPUs: cap the training resolution.
    if args.train_pixels:
        model.shape_constraints = dict(model.shape_constraints, pixels_max=args.train_pixels,
                                       pixels_min=min(model.shape_constraints["pixels_min"], args.train_pixels))
        print(f"training resolution capped at {args.train_pixels} pixels")
    default_sc = dict(model.shape_constraints)
    print("shape constraints:", default_sc)
    model.train()
    if args.freeze_encoder:
        for p in model.pixel_encoder.parameters():
            p.requires_grad = False
    # Two learning rates: the decoder learns faster than the encoder (the encoder can also be frozen).
    enc = [p for p in model.pixel_encoder.parameters() if p.requires_grad]
    dec = [p for p in model.pixel_decoder.parameters() if p.requires_grad]
    groups = [{"params": dec, "lr": args.lr_dec}] + ([{"params": enc, "lr": args.lr_enc}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in groups]
    # Mixed precision (fp16) to fit in 8 GB of GPU memory.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.fp32)
    steps_per_epoch = max(len(train_stems) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs

    # Learning-rate schedule: linear warm-up, then cosine decay.
    def set_lr(step):
        f = (step + 1) / args.warmup if step < args.warmup else 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(total_steps - args.warmup, 1)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(f, 0.0)

    # Resume from last_state.pt if asked; otherwise first measure the untouched pretrained model (epoch 0) as the reference.
    history, best, start_epoch, step = [], float("inf"), 1, 0
    if args.resume and (out / "last_state.pt").exists():
        st = torch.load(out / "last_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        history, best, start_epoch, step = st["history"], st["best"], st["epoch"] + 1, st["step"]
        print(f"resumed after epoch {st['epoch']} (best AbsRel so far {best:.4f})")
    elif not args.skip_first_eval:
        res = validate(model, val_stems, args, meta, device)
        print(f"epoch 0 (pretrained, untouched): {short(res)}")
        history.append({"epoch": 0, "val": res})
        best = res["all"]["AbsRel"]
        json.dump(history, open(out / "history.json", "w"), indent=2)

    # Training loop. One image per forward pass; gradients of grad_accum images are added before each optimiser step.
    ds = DepthFrames(train_stems, args.images, args.depth_gt, meta, args.max_depth, args.max_points, train=True)
    for epoch in range(start_epoch, args.epochs + 1):
        loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True, num_workers=args.workers,
                                             persistent_workers=False)
        model.train()
        run_loss = run_rel = 0.0
        n_done, t0 = 0, time.time()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(loader, total=len(ds), desc=f"epoch {epoch}/{args.epochs}")
        for rgb, ys, xs, gt, K, stem in bar:
            # Skip frames with almost no LiDAR points.
            if len(gt) < 20:
                continue
            ys, xs, gt = ys.to(device), xs.to(device), gt.to(device)
            try:
                depth_map, padded_hw, pad_lt = forward_depth(model, rgb, K, device, train_aug=True, fp16=not args.fp32)
                pred = sample_at(depth_map, ys, xs, padded_hw, pad_lt)
                loss, rel = depth_loss(pred, gt, args.w_silog, args.w_rel)
                # Safety: a non-finite loss is skipped instead of breaking the weights.
                if not torch.isfinite(loss):
                    print(f"\nnon-finite loss on {stem}, skipping")
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                sys.exit("Out of GPU memory. Try: --train_pixels 300000   or   --freeze_encoder")
            run_loss += float(loss); run_rel += rel; n_done += 1
            # Optimiser step with gradient clipping (max norm 1.0).
            if n_done % args.grad_accum == 0:
                set_lr(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            if n_done % 20 == 0:
                bar.set_postfix(loss=f"{run_loss / n_done:.3f}", rel=f"{run_rel / n_done:.3f}")
        print(f"epoch {epoch}: train loss {run_loss / max(n_done, 1):.3f}, train AbsRel at LiDAR pixels "
              f"{run_rel / max(n_done, 1):.3f}, {(time.time() - t0) / 60:.1f} min")
        torch.cuda.empty_cache()
        res = validate(model, val_stems, args, meta, device)
        print(f"epoch {epoch} validation: {short(res)}")
        history.append({"epoch": epoch, "train_loss": run_loss / max(n_done, 1), "val": res})
        # After each epoch: validate, and keep the checkpoint with the lowest AbsRel as best.pt.
        a = res["all"]["AbsRel"]
        if a < best:
            best = a
            torch.save({"base": args.model, "state": model.state_dict(), "epoch": epoch, "val": res["all"]}, out / "best.pt")
            print(f"  new best ({best:.4f}) saved to {out / 'best.pt'}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "history": history, "best": best, "step": step}, out / "last_state.pt")
        json.dump(history, open(out / "history.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        torch.cuda.empty_cache()

    print(f"\nDone. Best validation AbsRel {best:.4f}. Use it with:  --depth_model {out / 'best.pt'}")
    if not (out / "best.pt").exists():
        print("(no epoch beat the pretrained model, so best.pt was not written - the zero-shot model stays)")


if __name__ == "__main__":
    main()
