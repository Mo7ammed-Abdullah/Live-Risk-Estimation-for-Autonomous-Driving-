"""
Evaluate a semantic-segmentation model against your class-index masks (seg_gt/masks).

Reports: pixel accuracy, mean class accuracy, mIoU, frequency-weighted IoU, per-class IoU,
mIoU over the 7 risk classes (person, rider, car, truck, bus, motorcycle, bicycle), by city,
plus latency/FPS and peak GPU memory.

IMPORTANT: your masks are PSEUDO-labels (made by Mask2Former). A high score here means "agrees with
Mask2Former", not "is correct". Use a hand-checked test subset for the numbers you publish.

Run with venv311 active, from "Testing 3" (needs generate_seg_gt.py and evalkit.py in the same folder):
    python eval_seg.py                                                   # SegFormer-B2 Cityscapes, validation split
    python eval_seg.py --model nvidia/segformer-b0-finetuned-cityscapes-1024-1024
    python eval_seg.py --short_side 512                                  # faster, less accurate
    python eval_seg.py --limit 50
    python eval_seg.py --split test --final                              # ONLY for your final numbers
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
# Reuse the labeller loader and the out-of-memory-safe predictor from the label generator.
from generate_seg_gt import load_labeler, predict_with_retry

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--short_side", type=int, default=900, help="input short side in px (900 = native)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    # Frames to evaluate (the test split only opens with --final).
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    # Class names of our masks (19 Cityscapes classes).
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    device = ek.get_device()
    use_amp = device.type == "cuda" and not args.fp32
    print(f"Model {args.model} | split {args.split} ({len(stems)} images) | device: {device}")

    processor, model = load_labeler(args.model, device)
    # Safety check: the model's class order must match the mask class order, or every score would be wrong.
    model_names = [str(model.config.id2label[i]).lower() for i in range(len(model.config.id2label))]
    if len(model_names) != len(names):
        sys.exit(f"ERROR: the model has {len(model_names)} classes but your masks have {len(names)}. "
                 "Use a Cityscapes model, or add a class mapping.")
    if model_names != [n.lower() for n in names]:
        print("WARNING: model class names differ from classes.json - class ids may not line up:")
        print("  model:", model_names)
        print("  masks:", names)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # SegMeter builds a confusion matrix over all frames; Timer measures resize + model + post-processing.
    meter, timer = ek.SegMeter(len(names), names), ek.Timer(warmup=3)
    skipped = 0
    for stem in tqdm(stems, desc="evaluating segmentation"):
        img_path, mask_path = args.images / f"{stem}.jpg", args.seg_dir / "masks" / f"{stem}.png"
        if not img_path.exists() or not mask_path.exists():
            skipped += 1
            continue
        image = Image.open(img_path).convert("RGB")
        # Ground truth = class-id mask (pseudo-labels from Mask2Former, 255 = ignore).
        gt = np.array(Image.open(mask_path))

        ek.cuda_sync()
        t0 = time.time()
        # Predicted class id for every pixel.
        pred = predict_with_retry(processor, model, image, device, use_amp, args.short_side)
        ek.cuda_sync()
        timer.add(time.time() - t0)

        if pred.shape != gt.shape:
            skipped += 1
            continue
        # Add this frame to the confusion matrices (overall and per city).
        meter.update(pred, gt, tags=(ek.city_tag(stem),))

    # Final metrics and latency; printed and saved to results/seg_*.json.
    res, lat = meter.results(), timer.summary()
    ek.print_seg_report(res, names)
    if lat:
        print(f"\nLatency (resize + model + post-processing, batch 1): {lat['mean_ms']:.1f} ± {lat['std_ms']:.1f} ms, "
              f"p95 {lat['p95_ms']:.1f} ms, {lat['fps']:.1f} FPS | peak GPU memory {ek.peak_gpu_gb():.2f} GB | "
              f"{n_params:.1f} M parameters")
    if skipped:
        print(f"Skipped {skipped} images (missing file or size mismatch)")

    # File-name tag: local folder name for our fine-tuned model, otherwise the hub name.
    p = Path(args.model)
    model_tag = f"{p.parent.name}_{p.name}" if p.exists() else args.model.split("/")[-1]  # local folder vs hub id
    tag = f"{model_tag}_{args.split}"
    ek.save_json(args.out_dir / f"seg_{tag}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "params_M": n_params, "class_names": names,
                  "latency": lat, "peak_gpu_gb": ek.peak_gpu_gb(), "metrics": res,
                  "note": "ground truth = pseudo-labels from Mask2Former"})


if __name__ == "__main__":
    main()
