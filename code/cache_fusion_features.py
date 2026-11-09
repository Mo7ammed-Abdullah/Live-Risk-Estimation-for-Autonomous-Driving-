"""
Run the segmentation and depth models ONCE per frame and store small feature maps for the learned fusion.

Per frame (at 1/8 of the image size, 112 x 200), 12 channels in float16:
    0-7   probability of person, rider, car, truck, bus, train, motorcycle, bicycle   (SegFormer softmax)
    8     probability of everything else (1 - sum of the above)
    9     segmentation confidence (highest class probability)
    10    metric depth in metres (nearest surface in each 8x8 block, clipped to 0-80 m)
    11    depth confidence (UniDepthV2's own confidence output, block average)

Output:  cache/fusion_<split>.npy     (frames, 12, 112, 200) float16, opened later as a memory map
         cache/fusion_<split>.json    frame names in the same order + channel names

Run (venv311 active, project folder):
    python code/cache_fusion_features.py --split train     # ~10 min on the laptop GPU, ~1.2 GB file
    python code/cache_fusion_features.py --split val       # ~2 min
    python code/cache_fusion_features.py --split test --final   # only at the very end
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
from eval_depth import load_depth_model
from generate_seg_gt import load_labeler

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
RISK = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
CHANNELS = [f"p_{c}" for c in RISK] + ["p_other", "seg_conf", "depth_m", "depth_conf"]
OUT_H, OUT_W = 112, 200
MAX_DEPTH = 80.0


def seg_probs(processor, model, image, device, risk_ids):
    """SegFormer at native resolution -> class probabilities resized to OUT_H x OUT_W, as (9 + 1) channels."""
    import torch
    import torch.nn.functional as F
    width, height = image.size
    h, w = int(round(height / 32) * 32), int(round(width / 32) * 32)      # native size, multiples of 32
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(pixel_values=(x - mean) / std).logits
    logits = F.interpolate(logits.float(), size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
    p = logits.softmax(1)[0]                                     # (classes, OUT_H, OUT_W)
    risk = p[risk_ids]                                           # (8, OUT_H, OUT_W)
    other = (1.0 - risk.sum(0, keepdim=True)).clamp(min=0.0)
    conf = p.max(0, keepdim=True).values
    return torch.cat([risk, other, conf], 0).cpu().numpy()


def depth_maps(model, image, K):
    """UniDepthV2 -> (nearest depth per 8x8 block, mean confidence per block), both OUT_H x OUT_W."""
    import torch
    import torch.nn.functional as F
    from unidepth.utils.camera import Pinhole
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    with torch.inference_mode():
        pred = model.infer(rgb, Pinhole(K=torch.tensor(K, dtype=torch.float32))) if K is not None else model.infer(rgb)
    d = torch.nan_to_num(pred["depth"].float(), nan=MAX_DEPTH, posinf=MAX_DEPTH).clamp(0, MAX_DEPTH).view(1, 1, *pred["depth"].shape[-2:])
    near = -F.adaptive_max_pool2d(-d, (OUT_H, OUT_W))           # nearest surface in each block (safety first)
    c = pred.get("confidence")
    if c is None:
        conf = torch.zeros_like(near)
    else:
        conf = F.adaptive_avg_pool2d(torch.nan_to_num(c.float()).view(1, 1, *c.shape[-2:]), (OUT_H, OUT_W))
    return torch.cat([near, conf], 1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt_native", help="for the class list")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "cache")
    args = ap.parse_args()

    stems = ek.load_split(args.split, args.final, 1, args.limit)
    stems = [s for s in stems if (args.images / f"{s}.jpg").exists()]
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    risk_ids = [names.index(c) for c in RISK]
    meta = json.load(open(args.meta_cache)) if args.meta_cache.exists() else None
    device = ek.get_device()
    print(f"Split {args.split}: {len(stems)} frames | device {device}\n  seg   {args.seg_model}\n  depth {args.depth_model}")
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = args.out_dir / f"fusion_{args.split}.npy"
    feats = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float16, shape=(len(stems), len(CHANNELS), OUT_H, OUT_W))
    for i, stem in enumerate(tqdm(stems, desc=f"caching {args.split}")):
        img = Image.open(args.images / f"{stem}.jpg").convert("RGB")
        K = None
        if meta is not None and f"{stem}.jpg" in meta["sd"]:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        feats[i, :10] = seg_probs(processor, seg_model, img, device, risk_ids)
        feats[i, 10:] = depth_maps(depth_model, img, K)
    feats.flush()
    json.dump({"stems": stems, "channels": CHANNELS, "size": [OUT_H, OUT_W], "image_size": [900, 1600],
               "seg_model": args.seg_model, "depth_model": args.depth_model}, open(args.out_dir / f"fusion_{args.split}.json", "w"))
    print(f"Saved {out_npy} ({feats.shape}, {out_npy.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
