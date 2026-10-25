"""
Generate semantic-segmentation PSEUDO ground truth for the nuScenes front camera on a LOCAL NVIDIA GPU.

What it writes (next to this script by default):
    seg_gt/masks/<image_name>.png        class-index mask, 8-bit palette PNG (pixel value = class id, 255 = unlabeled)
    seg_gt/overlays/<image_name>_overlay.png   colored preview (for looking at only, NOT ground truth)
    seg_gt/classes.json                  class id -> name / color, and which model made the labels
    seg_gt/class_pixel_counts.json       how often each class occurs (useful for class imbalance)
    seg_gt/meta.json                     settings used for this run

Two stages:
    1. A semantic labeler (default: Mask2Former Swin-L trained on Cityscapes) decides WHICH CLASS each pixel is.
    2. Optional (--refine): SAM 2 proposes class-agnostic masks with sharp boundaries. Each SAM mask takes the
       class that most of its pixels already have (if that class covers >= --sam_min_frac of the mask). SAM 2
       cannot name classes itself, it only sharpens the edges.

Install (inside your venv, torch with CUDA is already installed):
    python -m pip install transformers accelerate opencv-python pillow numpy tqdm
    # only for --refine:
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    python -m pip install -e .

Usage examples (run from the "Testing 3" folder, venv active):
    python generate_seg_gt.py --limit 10                 # quick test on 10 images, check overlays + timing
    python generate_seg_gt.py --limit 10 --refine        # same, with SAM 2 boundary refinement
    python generate_seg_gt.py                            # all images (skips ones already done, so it can be resumed)

Read a mask back (use PIL, NOT cv2.imread, which would turn the palette into colors):
    mask = np.array(Image.open(path))        # 2-D array of class ids, 255 = ignore
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Pixel value that means 'no label'; every metric skips these pixels.
IGNORE_INDEX = 255
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Download link of the optional SAM 2 checkpoint (only needed with --refine).
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

# Official Cityscapes colors for the 19 training classes (used when the labeler has 19 classes)
CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]


# ----------------------------------------------------------------------------
# Colors / saving
# ----------------------------------------------------------------------------
# 256-entry colour table (class id -> RGB), used only to make masks and overlays viewable.
def make_color_table(n_classes):
    """(256, 3) uint8 table: class id -> RGB. Index 255 (ignore) is white."""
    if n_classes == 19:
        colors = np.array(CITYSCAPES_COLORS, dtype=np.uint8)
    else:
        colors = np.random.default_rng(0).integers(30, 230, size=(n_classes, 3), dtype=np.uint8)
    table = np.zeros((256, 3), dtype=np.uint8)
    table[:n_classes] = colors
    table[IGNORE_INDEX] = (255, 255, 255)
    return table


def save_mask(pred, path, color_table):
    """Save class ids as an 8-bit palette PNG: pixel values stay exact class ids, viewers show colors."""
    h, w = pred.shape
    # Palette PNG: the stored pixel values are the exact class ids, a viewer just shows them in colour.
    img = Image.frombytes("P", (w, h), pred.astype(np.uint8).tobytes())
    img.putpalette(color_table.reshape(-1).tolist())
    img.save(path)


def make_overlay(image_np, pred, color_table, names, label_regions=True):
    import cv2
    # Blend the colour mask over the photo (55% photo, 45% colour) and write each region's class name on it.
    color_mask = color_table[pred]
    overlay = cv2.addWeighted(image_np, 0.55, color_mask, 0.45, 0)
    if label_regions:
        for class_id in np.unique(pred):
            if class_id >= len(names):
                continue
            binary = (pred == class_id).astype(np.uint8)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for comp in range(1, count):
                if stats[comp, cv2.CC_STAT_AREA] < 1500:
                    continue
                x, y = centroids[comp].astype(int)
                text = names[int(class_id)]
                cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


# ----------------------------------------------------------------------------
# Stage 1: semantic labeler (Hugging Face)
# ----------------------------------------------------------------------------
# Load the Hugging Face labeller model (Mask2Former) and its image processor.
def load_labeler(model_id, device):
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        from transformers import AutoModelForSemanticSegmentation as AutoCls
        model = AutoCls.from_pretrained(model_id)
    except (ValueError, KeyError):  # e.g. Mask2Former is registered as "universal segmentation"
        from transformers import AutoModelForUniversalSegmentation as AutoCls
        model = AutoCls.from_pretrained(model_id)
    return processor, model.to(device).eval()


def predict_semantic(processor, model, image, device, use_amp, short_side):
    """Resize ourselves (keeping the aspect ratio, multiples of 32) instead of trusting the processor's size
    argument: newer transformers versions silently ignored it and fed 512x512 squashed images.
    short_side = image height the model sees (900 = native for nuScenes; 0 = native)."""
    import torch
    import torch.nn.functional as F
    width, height = image.size
    s = short_side or height
    # Round the input size to multiples of 32 (the network needs this), keeping the aspect ratio.
    h, w = max(32, int(round(s / 32) * 32)), max(32, int(round(s * width / height / 32) * 32))
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    # Normalise with the model's own mean/std; the resized image goes through the network in fp16 when allowed.
    x = (x - mean) / std
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(pixel_values=x)
        pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    return pred.to(torch.uint8).cpu().numpy()


def predict_with_retry(processor, model, image, device, use_amp, short_side):
    """Run the labeler; if the 8 GB GPU runs out of memory, retry at a smaller input size."""
    import torch
    # If the GPU runs out of memory, retry at 75% and then 50% of the input size.
    scales = [short_side, int(short_side * 0.75), int(short_side * 0.5)]
    for i, s in enumerate(scales):
        try:
            return predict_semantic(processor, model, image, device, use_amp, s)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if i == len(scales) - 1:
                raise
            print(f"\n  GPU out of memory, retrying at short side {scales[i + 1]}")


# ----------------------------------------------------------------------------
# Stage 2 (optional): SAM 2 boundary refinement
# ----------------------------------------------------------------------------
def download_checkpoint(path):
    print(f"Downloading SAM 2.1 Hiera Large checkpoint (~0.9 GB) to {path} ...")
    last = [0]

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and done - last[0] > 50e6:
            last[0] = done
            print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB")

    urllib.request.urlretrieve(SAM_URL, path, hook)


# Create the optional SAM 2 mask generator (downloads the checkpoint on first use).
def build_sam(args, device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        sys.exit("ERROR: the 'sam2' package is not installed, so --refine cannot run.\n"
                 "Install it (see the top of this file) or run without --refine.")
    if not args.sam_ckpt.exists():
        download_checkpoint(args.sam_ckpt)
    sam = build_sam2(args.sam_cfg, str(args.sam_ckpt), device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(sam, points_per_side=args.sam_points)


def sam_masks(generator, image_np):
    import torch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return generator.generate(image_np)


# Sharpen boundaries: each SAM mask gets the majority class of the pixels it covers (if enough agree).
def refine_with_sam(pred, masks, n_classes, min_frac, min_area):
    """Snap boundaries: every SAM mask takes the class most of its pixels already have.
    Big masks are painted first so smaller, more specific masks override them."""
    out = pred.copy()
    pred_i = pred.astype(np.int64)
    for m in sorted(masks, key=lambda d: d["area"], reverse=True):
        if m["area"] < min_area:
            continue
        seg = m["segmentation"]
        counts = np.bincount(pred_i[seg], minlength=n_classes)
        total = counts.sum()
        if total == 0:
            continue
        cls = int(counts.argmax())
        if counts[cls] / total >= min_frac:
            out[seg] = cls
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
# Count how many pixels each class has over all masks and save/print the shares (shows class imbalance).
def compute_stats(mask_dir, names, out_path):
    files = sorted(mask_dir.glob("*.png"))
    counts = np.zeros(256, dtype=np.int64)
    for f in tqdm(files, desc="class statistics"):
        counts += np.bincount(np.array(Image.open(f)).ravel(), minlength=256)
    n = len(names)
    valid = max(int(counts[:n].sum()), 1)
    table = [{"id": i, "name": names[i], "pixels": int(counts[i]), "fraction": float(counts[i] / valid)}
             for i in range(n)]
    with open(out_path, "w") as f:
        json.dump({"num_masks": len(files), "ignored_pixels": int(counts[IGNORE_INDEX]), "classes": table}, f, indent=2)
    print(f"\nClass frequencies over {len(files)} masks:")
    for row in sorted(table, key=lambda r: -r["fraction"]):
        print(f"  {row['name']:<15s} {100 * row['fraction']:6.2f}%")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--out", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-cityscapes-semantic",
                    help="Hugging Face semantic-segmentation model id (a Cityscapes model gives 19 driving classes)")
    ap.add_argument("--short_side", type=int, default=900, help="input short side in px; lower = faster, less VRAM")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--refine", action="store_true", help="refine boundaries with SAM 2 (slower)")
    ap.add_argument("--sam_ckpt", type=Path, default=HERE / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam_points", type=int, default=16, help="SAM grid points per side (more = finer, much slower)")
    ap.add_argument("--sam_min_frac", type=float, default=0.6, help="min share of a SAM mask that must agree on a class")
    ap.add_argument("--sam_min_area", type=int, default=200, help="ignore SAM masks smaller than this (pixels)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (after --every)")
    ap.add_argument("--every", type=int, default=1, help="process every Nth image (spreads a subset across scenes)")
    ap.add_argument("--overlay_every", type=int, default=20, help="save a preview overlay for every Nth image (0 = none)")
    ap.add_argument("--no_region_labels", action="store_true", help="no class names written on overlays")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch

    if not args.images.exists():
        sys.exit(f"ERROR: image folder not found: {args.images}\nUse --images to point to your CAM_FRONT folder.")
    # List the images to label (every Nth, first N if --limit).
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:: max(args.every, 1)]
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"ERROR: no images found in {args.images}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA GPU found (torch.cuda.is_available() is False). This will be extremely slow. "
              "You probably have the CPU build of torch in this venv.")
    # Mixed precision (fp16) on the GPU for speed, unless --fp32 is given.
    use_amp = device.type == "cuda" and not args.fp32

    mask_dir = args.out / "masks"
    overlay_dir = args.out / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_every:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labeler: {args.model}")
    processor, model = load_labeler(args.model, device)
    id2label = model.config.id2label
    # Class names come from the model itself (19 Cityscapes classes).
    names = [str(id2label[i]) if i in id2label else str(id2label[str(i)]) for i in range(len(id2label))]
    n_classes = len(names)
    color_table = make_color_table(n_classes)
    print(f"{n_classes} classes: {', '.join(names)}")

    sam = build_sam(args, device) if args.refine else None

    with open(args.out / "classes.json", "w") as f:
        json.dump({"labeler": args.model, "refined_with_sam2": bool(args.refine), "kind": "pseudo-label (model predictions)",
                   "ignore_index": IGNORE_INDEX,
                   "classes": [{"id": i, "name": names[i], "color": color_table[i].tolist()} for i in range(n_classes)]},
                  f, indent=2)
    with open(args.out / "meta.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    # Main loop: label each image and save its mask; images already done are skipped, so a run can be resumed.
    times, failed, skipped = [], [], 0
    for i, path in enumerate(tqdm(images, desc="labeling")):
        out_mask = mask_dir / f"{path.stem}.png"
        if out_mask.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            t0 = time.time()
            image = Image.open(path).convert("RGB")
            pred = predict_with_retry(processor, model, image, device, use_amp, args.short_side)
            image_np = np.array(image)
            if sam is not None:
                masks = sam_masks(sam, image_np)
                pred = refine_with_sam(pred, masks, n_classes, args.sam_min_frac, args.sam_min_area)
            save_mask(pred, out_mask, color_table)
            if args.overlay_every and i % args.overlay_every == 0:
                overlay = make_overlay(image_np, pred, color_table, names, not args.no_region_labels)
                Image.fromarray(overlay).save(overlay_dir / f"{path.stem}_overlay.png")
            times.append(time.time() - t0)
        # A failed image is recorded and the run continues.
        except Exception as e:  # keep going; report at the end
            failed.append((path.name, repr(e)))

    print(f"\nLabeled {len(times)} images, skipped {skipped} already done, failed {len(failed)}.")
    if times:
        print(f"Average {np.mean(times):.2f} s/image  ->  about {np.mean(times) * 3376 / 3600:.1f} h for 3376 images")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    if failed:
        with open(args.out / "failed.txt", "w") as f:
            f.write("\n".join(f"{n}\t{e}" for n, e in failed))
        print(f"First failure: {failed[0]}  (all listed in {args.out / 'failed.txt'})")

    compute_stats(mask_dir, names, args.out / "class_pixel_counts.json")
    print(f"\nMasks: {mask_dir}\nCheck the overlays in {overlay_dir} before using these labels.")
    print("Reminder: these are model-generated pseudo-labels. Keep a hand-corrected test subset for honest mIoU.")


if __name__ == "__main__":
    main()