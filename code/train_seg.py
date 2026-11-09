"""
Fine-tune SegFormer on your segmentation masks (train split) and validate on the validation split.

Built for an 8 GB laptop GPU (RTX 4060): mixed precision, 512x1024 crops, batch 4.
Rare classes (person, rider, bus, motorcycle, bicycle) get extra attention in two ways:
  * repeat-factor sampling: images that contain rare classes are drawn more often
  * crop centring: half of the crops are centred on a random pixel of a rare class (when the image has one)

Run with venv311 active, from "Testing 3" (needs evalkit.py, make_splits.py and the splits/ folder):
    python train_seg.py --epochs 1 --limit_train 64 --limit_val 32 --eval_every 1   # 2-minute smoke test first
    python train_seg.py                                                             # real run (20 epochs)
    python train_seg.py --resume                                                    # continue after an interruption
    python train_seg.py --batch_size 2 --grad_accum 2                               # if you run out of GPU memory

Outputs in runs/<name>/ :
    best/            the best model by validation score (loadable with eval_seg.py --model runs\\seg_b2\\best)
    last_state.pt    everything needed to resume
    history.json     loss and validation metrics after every evaluation
    config.json      the settings used

The labels are pseudo-labels from Mask2Former, so validation scores here mean "agreement with those labels".
Never use the test split while developing (this script does not touch it).
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

import evalkit as ek

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Pixel normalisation used by the pretrained SegFormer (ImageNet mean and standard deviation).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Rare classes: half of the training crops are centred on a pixel of one of these, so the model sees them often.
RARE_FOR_CROPS = ["person", "rider", "bus", "motorcycle", "bicycle"]
IGNORE = 255


# ----------------------------------------------------------------------------
# Class statistics -> sampling weights / loss weights (plain numpy, easy to test)
# ----------------------------------------------------------------------------
# For each training image, which classes it contains (at least min_px pixels) and the total pixels per class.
def load_class_stats(csv_path, stems, names, min_px=500):
    """Per-image class pixel counts for the given stems, from splits/per_image_class_pixels.csv."""
    if not csv_path.exists():
        return None, None
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or rows[0][2:] != names:
        print("WARNING: per_image_class_pixels.csv does not match the class list; skipping class-aware sampling.")
        return None, None
    table = {r[0]: np.array(r[2:], dtype=np.int64) for r in rows[1:]}
    counts = np.stack([table.get(s, np.zeros(len(names), dtype=np.int64)) for s in stems])
    return counts >= min_px, counts.sum(0)


# Repeat-factor sampling: an image with a rare class gets a sampling weight above 1 (set by its rarest class), so it is drawn more often.
def repeat_factors(presence, t=0.3):
    """Repeat-factor sampling: images containing rare classes get weight > 1 (weight of an image = its rarest class)."""
    freq = presence.mean(0)                                   # share of images containing each class
    r_class = np.where(freq > 0, np.maximum(1.0, np.sqrt(t / np.maximum(freq, 1e-9))), 1.0)
    return np.where(presence, r_class[None, :], 1.0).max(1)


# Optional loss weights: rare classes count more (inverse frequency to a power, capped at 10). Off by default (power 0).
def class_loss_weights(pixels, power, max_weight=10.0):
    """Inverse-frequency^power weights, rescaled so the pixel-weighted mean is 1."""
    freq = pixels / max(pixels.sum(), 1)
    w = np.minimum((1.0 / np.maximum(freq, 1e-8)) ** power, max_weight)
    return w / max((w * freq).sum(), 1e-8)


def class_balanced_weights(pixels, beta, max_weight=10.0):
    """Class-balanced weights (Cui et al. 2019): weight = 1 / effective number of pixels of the class,
    effective number = (1 - beta^n) / (1 - beta). beta close to 1 (e.g. 0.9999999) because classes have millions of pixels.
    Rescaled so the pixel-weighted mean is 1, capped at max_weight."""
    n = np.maximum(pixels.astype(np.float64), 1.0)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = np.minimum(1.0 / np.maximum(eff, 1e-12), np.inf)
    freq = n / n.sum()
    w = w / max((w * freq).sum(), 1e-12)
    return np.minimum(w, max_weight)


def seg_loss(logits, masks, class_w=None, kind="ce", gamma=2.0, ignore_index=IGNORE):
    """Pixel loss. kind = 'ce' (cross-entropy, the original) or 'focal' (Lin et al. 2017): the cross-entropy of each pixel is
    multiplied by (1 - p_correct)^gamma, so easy pixels (road, sky) count little and hard pixels (small, rare objects) count more.
    class_w (optional) multiplies each pixel by the weight of its true class. Pixels with the ignore label are skipped."""
    import torch
    import torch.nn.functional as F
    if kind == "ce":
        return F.cross_entropy(logits, masks, weight=class_w, ignore_index=ignore_index)
    valid = masks != ignore_index
    safe = torch.where(valid, masks, torch.zeros_like(masks))
    ce = F.cross_entropy(logits, safe, reduction="none")
    p_correct = torch.exp(-ce)
    loss = ((1.0 - p_correct) ** gamma) * ce
    w = valid.float() if class_w is None else class_w[safe] * valid.float()
    return (loss * w).sum() / w.sum().clamp(min=1e-8)


# ----------------------------------------------------------------------------
# Dataset (returns numpy arrays; the DataLoader turns them into tensors)
# ----------------------------------------------------------------------------
# Training data: returns an image and its class-id mask. Training images are randomly cropped and augmented;
# validation images are returned whole.
class SegDataset:
    def __init__(self, stems, images_dir, masks_dir, names, train, crop=(512, 1024), scale=(0.75, 1.5),
                 rare_prob=0.5, jitter=True):
        self.stems, self.images_dir, self.masks_dir = stems, Path(images_dir), Path(masks_dir)
        self.train, self.crop, self.scale, self.rare_prob, self.jitter = train, crop, scale, rare_prob, jitter
        self.rare_ids = [names.index(c) for c in RARE_FOR_CROPS if c in names]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        mask = np.array(Image.open(self.masks_dir / f"{stem}.png"))
        if not self.train:
            return np.ascontiguousarray(np.array(img).transpose(2, 0, 1)), mask, ek.city_tag(stem)
        return self._augment(img, mask)

    def _augment(self, img, mask):
        W, H = img.size
        ch, cw = self.crop
        s = np.random.uniform(*self.scale)
        # Random scale: take a window of the original image so that after resizing it becomes the crop size (512 x 1024).
        win_h, win_w = min(H, int(round(ch / s))), min(W, int(round(cw / s)))   # window in original pixels
        y0 = np.random.randint(0, H - win_h + 1)
        x0 = np.random.randint(0, W - win_w + 1)

        # With probability rare_prob, move the crop to be centred on a random pixel of a rare class (if the image has one).
        if self.rare_ids and np.random.rand() < self.rare_prob:      # centre the crop on a rare-class pixel
            present = [c for c in self.rare_ids if (mask == c).any()]
            if present:
                c = present[np.random.randint(len(present))]
                ys, xs = np.nonzero(mask == c)
                k = np.random.randint(len(ys))
                y0 = int(np.clip(ys[k] - win_h // 2, 0, H - win_h))
                x0 = int(np.clip(xs[k] - win_w // 2, 0, W - win_w))

        # Cut and resize the crop (bilinear for the photo, nearest for the mask so class ids stay exact).
        img_c = img.crop((x0, y0, x0 + win_w, y0 + win_h)).resize((cw, ch), Image.BILINEAR)
        mask_c = Image.fromarray(mask[y0:y0 + win_h, x0:x0 + win_w]).resize((cw, ch), Image.NEAREST)
        # Augmentation: random horizontal flip, then brightness / contrast / colour jitter.
        if np.random.rand() < 0.5:
            img_c, mask_c = img_c.transpose(Image.FLIP_LEFT_RIGHT), mask_c.transpose(Image.FLIP_LEFT_RIGHT)
        if self.jitter and np.random.rand() < 0.8:
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                img_c = enhancer(img_c).enhance(np.random.uniform(0.8, 1.2))
        return np.ascontiguousarray(np.array(img_c).transpose(2, 0, 1)), np.array(mask_c)


# Give every data-loading worker its own random seed, so workers do not produce identical crops.
def worker_init(worker_id):
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
# Validation: run the model on whole validation images (896 x 1600) and collect the metrics with SegMeter. Training mode is restored afterwards.
def evaluate(model, loader, device, names, use_amp, eval_size=(896, 1600)):
    import torch
    import torch.nn.functional as F
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    meter = ek.SegMeter(len(names), names)
    model.eval()
    with torch.inference_mode():
        for imgs, masks, cities in loader:
            x = imgs.to(device).float() / 255.0
            x = (F.interpolate(x, size=eval_size, mode="bilinear", align_corners=False) - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(pixel_values=x).logits
            logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for p, m, c in zip(pred, masks.numpy(), cities):
                meter.update(p, m, tags=(c,))
    model.train()
    torch.cuda.empty_cache()  # give the evaluation memory back before training continues
    return meter.results()


# One-line training log: loss, pixel accuracy, mIoU, risk-class mIoU and the IoU of each risk class.
def summarize(res, names, epoch, loss, seconds):
    r = res["all"]
    risk = " ".join(f"{c[:4]} {r['iou'][names.index(c)]:.2f}" for c in ek.RISK_CLASSES if c in names)
    return (f"[epoch {epoch}] loss {loss:.4f} | pixAcc {r['pixel_acc']:.3f} mIoU {r['mIoU']:.3f} "
            f"riskmIoU {r['risk_mIoU']:.3f} | {risk} | {seconds:.0f}s")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# Training settings. Defaults suit an 8 GB laptop GPU: mixed precision, 512 x 1024 crops, batch 4.
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--name", default="seg_b2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5, help="learning rate of the encoder")
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="decode head learning rate = lr x this")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=150, help="warm-up optimizer steps")
    ap.add_argument("--crop", type=int, nargs=2, default=[512, 1024], metavar=("H", "W"))
    ap.add_argument("--scale", type=float, nargs=2, default=[0.75, 1.5], metavar=("MIN", "MAX"))
    ap.add_argument("--rare_crop_prob", type=float, default=0.5)
    ap.add_argument("--rfs_t", type=float, default=0.3, help="repeat-factor sampling threshold (0 = off)")
    ap.add_argument("--class_weight_power", type=float, default=0.0, help="loss class weights = inverse freq^power (0 = off)")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"], help="pixel loss: ce = cross-entropy (original), focal = focal loss")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="focal loss focusing strength (0 = same as cross-entropy)")
    ap.add_argument("--cb_beta", type=float, default=0.0, help="class-balanced loss weights with this beta (0 = off, try 0.9999999); overrides --class_weight_power")
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_batch", type=int, default=1)
    ap.add_argument("--select", default="miou", choices=["miou", "risk_miou"], help="metric used to pick the best epoch")
    ap.add_argument("--workers", type=int, default=4, help="data loading processes (use 0 if Windows complains)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_eval_at_start", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_train", type=int, default=0, help="debug: use only N training images")
    ap.add_argument("--limit_val", type=int, default=0, help="debug: use only N validation images")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--counts_csv", type=Path, default=HERE / "splits" / "per_image_class_pixels.csv",
                    help="per-image class pixel counts of the masks in --seg_dir (make_splits.py --counts_only)")
    ap.add_argument("--runs_dir", type=Path, default=HERE / "runs")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ek.get_device()
    if device.type != "cuda":
        sys.exit("ERROR: no CUDA GPU visible to torch. Training on CPU would take days.")
    use_amp = not args.no_amp
    out = args.runs_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    n = len(names)
    # Train and validation frames come from the drive-level split; the test split is never loaded here.
    train_stems = ek.load_split("train", False, 1, args.limit_train)
    val_stems = ek.load_split("val", False, 1, args.limit_val)
    print(f"Train images: {len(train_stems)} | validation images: {len(val_stems)} | classes: {n}")

    masks_dir = args.seg_dir / "masks"
    # Datasets for training (with augmentation) and validation.
    train_ds = SegDataset(train_stems, args.images, masks_dir, names, True, tuple(args.crop), tuple(args.scale),
                          args.rare_crop_prob)
    val_ds = SegDataset(val_stems, args.images, masks_dir, names, False)

    presence, pixels = load_class_stats(args.counts_csv, train_stems, names)
    # Repeat-factor sampling: images with rare classes are drawn more often during training.
    sampler, shuffle = None, True
    if presence is not None and args.rfs_t > 0:
        w = repeat_factors(presence, args.rfs_t)
        print(f"Repeat-factor sampling on: mean weight {w.mean():.2f}, max {w.max():.2f}, "
              f"{(w > 1).mean() * 100:.0f}% of images up-weighted")
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(train_ds), replacement=True)
        shuffle = False
    class_w = None
    if presence is not None and args.cb_beta > 0:
        class_w = torch.tensor(class_balanced_weights(pixels, args.cb_beta), dtype=torch.float32, device=device)
        print(f"Class-balanced loss weights (beta {args.cb_beta}):", {nm: round(float(x), 2) for nm, x in zip(names, class_w.cpu())})
    elif presence is not None and args.class_weight_power > 0:
        class_w = torch.tensor(class_loss_weights(pixels.astype(np.float64), args.class_weight_power),
                               dtype=torch.float32, device=device)
        print("Loss class weights:", {nm: round(float(x), 2) for nm, x in zip(names, class_w.cpu())})

    # Data loaders that feed batches to the GPU (several worker processes for speed).
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
                              num_workers=args.workers, drop_last=True, pin_memory=True,
                              persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False, num_workers=min(args.workers, 2))

    print(f"Loading {args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    # Start from SegFormer-B2 pretrained on Cityscapes and fine-tune it on our masks.
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model).to(device)
    if model.config.num_labels != n:
        sys.exit(f"ERROR: model has {model.config.num_labels} classes but your masks have {n}.")
    model.train()

    # Two learning rates: the pretrained encoder learns slowly, the decode head 5x faster.
    head = [p for nm, p in model.named_parameters() if nm.startswith("decode_head")]
    body = [p for nm, p in model.named_parameters() if not nm.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    warm = min(args.warmup, max(total_steps // 10, 1))

    # Learning-rate schedule: linear warm-up, then cosine decay down to 1% of the start value.
    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.5 * (1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1))), 0.01)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    # Mixed precision (fp16) with loss scaling saves GPU memory and time.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # --resume restores model, optimiser, schedule and history from last_state.pt, and continues from the next epoch.
    history, best, start_epoch = [], -1.0, 1
    state_path = out / "last_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        history, best, start_epoch = st["history"], st["best"], st["epoch"] + 1
        print(f"Resumed from epoch {st['epoch']} (best {args.select} so far: {best:.3f})")

    # Save the best model (weights + processor) so it can be loaded later with eval_seg.py or the demo.
    def save_best():
        model.save_pretrained(out / "best")
        processor.save_pretrained(out / "best")

    # Evaluate once before training, as a starting point to compare against.
    if start_epoch == 1 and not args.no_eval_at_start:
        t0 = time.time()
        res = evaluate(model, val_loader, device, names, use_amp)
        print(summarize(res, names, 0, float("nan"), time.time() - t0) + "   <- starting point (before training)")
        history.append({"epoch": 0, "loss": None, **{k: res["all"][k] for k in ("pixel_acc", "mIoU", "risk_mIoU")},
                        "iou": res["all"]["iou"].tolist()})

    # Training loop: forward pass, cross-entropy loss against the mask (255 ignored), backward pass.
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        running, count = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for it, (imgs, masks) in enumerate(bar):
            try:
                imgs = (imgs.to(device, non_blocking=True).float() / 255.0 - mean) / std
                masks = masks.to(device, non_blocking=True).long()
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(pixel_values=imgs).logits
                logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = seg_loss(logits, masks, class_w, args.loss, args.focal_gamma)
                # Gradient accumulation: gradients of several small batches are added before one optimiser step.
                scaler.scale(loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                sys.exit("\nOut of GPU memory. Try: --batch_size 2 --grad_accum 2   (or --crop 384 768)")
            # Optimiser step with gradient clipping (max norm 1.0), then update the learning rate.
            if (it + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if math.isfinite(loss.item()):
                running, count = running + loss.item(), count + 1
            bar.set_postfix(loss=f"{running / max(count, 1):.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
            if epoch == start_epoch and it == 0:
                print(f"  first step OK, GPU memory so far {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        train_loss = running / max(count, 1)

        # Every few epochs: validate, and save the model if it is the best so far (by mIoU or risk-class mIoU).
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res = evaluate(model, val_loader, device, names, use_amp)
            r = res["all"]
            score = r["mIoU"] if args.select == "miou" else r["risk_mIoU"]
            tag = ""
            if score > best:
                best = score
                save_best()
                tag = "   <- new best, saved"
            print(summarize(res, names, epoch, train_loss, time.time() - t0) + tag)
            history.append({"epoch": epoch, "loss": train_loss, "pixel_acc": r["pixel_acc"], "mIoU": r["mIoU"],
                            "risk_mIoU": r["risk_mIoU"], "iou": r["iou"].tolist()})
        else:
            print(f"[epoch {epoch}] loss {train_loss:.4f} | {time.time() - t0:.0f}s")
            history.append({"epoch": epoch, "loss": train_loss})

        # After every epoch save everything needed to resume, plus the history log.
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best": best, "epoch": epoch}, state_path)
        ek.save_json(out / "history.json", {"class_names": names, "history": history})

    print(f"\nDone. Best validation {args.select}: {best:.3f}")
    print(f"Evaluate it the same way as the baselines:\n  python eval_seg.py --model {out / 'best'}")


if __name__ == "__main__":
    main()
