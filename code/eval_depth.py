"""
Evaluate a depth model against your LiDAR depth maps on the validation (or, at the very end, test) split.

Reports: AbsRel, SqRel, MAE, RMSE, RMSE-log, SILog, d1/d2/d3 - overall, by true distance (0-20, 20-50, 50-80 m)
and by city - plus latency (mean/std/p95), FPS and peak GPU memory.

Run with venv311 active, from "Testing 3":
    python eval_depth.py                                   # UniDepthV2-Small, real camera intrinsics, validation split
    python eval_depth.py --camera unknown                  # let the model guess the camera
    python eval_depth.py --model lpiccinelli/unidepth-v2-vitb14
    python eval_depth.py --limit 50                        # quick check on 50 images
    python eval_depth.py --split test --final              # ONLY for your final numbers
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# evalkit holds the shared metrics, timer and split loader.
import evalkit as ek

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)


# Load UniDepthV2 either from the hub (pretrained) or from our fine-tuned .pt checkpoint (base model + saved weights).
def load_depth_model(name, device, resolution_level=None):
    """name = hub id (e.g. lpiccinelli/unidepth-v2-vits14) or a fine-tuned checkpoint written by train_depth.py (a .pt file)."""
    import torch
    from unidepth.models import UniDepthV2
    if str(name).endswith(".pt") and Path(str(name)).exists():
        ck = torch.load(str(name), map_location="cpu", weights_only=False)
        model = UniDepthV2.from_pretrained(ck["base"])
        model.load_state_dict(ck["state"])
    else:
        model = UniDepthV2.from_pretrained(name)
    model = model.to(device).eval()
    if resolution_level is not None:
        model.resolution_level = resolution_level
    return model


# Run the depth model on one RGB image and return metric depth (meters).
# With camera intrinsics K the scale is anchored to the real camera; without K the model guesses the camera.
def run_depth(model, rgb_np, K=None, fp16=False):
    """RGB uint8 (H, W, 3) -> metric depth in meters (H, W) as numpy. K = 3x3 intrinsics or None."""
    import torch
    from unidepth.utils.camera import Pinhole
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and torch.cuda.is_available()):
        if K is None:
            pred = model.infer(rgb)
        else:
            pred = model.infer(rgb, Pinhole(K=torch.tensor(K, dtype=torch.float32)))
    return pred["depth"].squeeze().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--camera", default="known", choices=["known", "unknown"])
    ap.add_argument("--resolution_level", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", help="run the depth model under fp16 autocast")
    ap.add_argument("--input_size", default="", help="WxH, e.g. 640x320: shrink the picture before the model, then scale the depth back up to full size")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--every", type=int, default=1, help="use every Nth image")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--depth_gt", type=Path, default=HERE / "depth_gt" / "depth" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    # Frames to evaluate (the test split only opens with --final).
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    # Camera intrinsics come from the metadata cache when the camera is 'known'.
    meta = json.load(open(args.meta_cache)) if args.camera == "known" else None
    device = ek.get_device()
    print(f"Model {args.model} | split {args.split} ({len(stems)} images) | camera: {args.camera} | device: {device}")
    model = load_depth_model(args.model, device, args.resolution_level)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # DepthMeter pools the errors over all frames; Timer measures the model call only.
    meter, timer = ek.DepthMeter(args.max_depth), ek.Timer(warmup=3)
    skipped = 0
    for stem in tqdm(stems, desc="evaluating depth"):
        img_path, gt_path = args.images / f"{stem}.jpg", args.depth_gt / f"{stem}.png"
        if not img_path.exists() or not gt_path.exists():
            skipped += 1
            continue
        # Load the photo and its LiDAR depth map (stored as meters x 256, 0 = no measurement).
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 256.0
        K = None
        if meta is not None:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]

        # Optional: shrink the picture before the model (--input_size) and scale the intrinsics by the same factors.
        in_rgb, in_K = rgb, K
        if args.input_size:
            W, H = (int(v) for v in args.input_size.split("x"))
            in_rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
            if K is not None:
                sx, sy = W / rgb.shape[1], H / rgb.shape[0]
                in_K = [[K[0][0] * sx, 0, K[0][2] * sx], [0, K[1][1] * sy, K[1][2] * sy], [0, 0, 1]]
        # Synchronise the GPU before and after, so the timer measures the real model time.
        ek.cuda_sync()
        t0 = time.time()
        depth = run_depth(model, in_rgb, in_K, args.fp16)
        ek.cuda_sync()
        timer.add(time.time() - t0)
        # Scale the predicted depth back up to the full image size so it can be compared with the LiDAR map.
        if args.input_size:
            depth = np.array(Image.fromarray(depth.astype(np.float32)).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))

        if depth.shape != gt.shape:
            skipped += 1
            continue
        # Score this frame (tagged by city); distance ranges are added inside the meter.
        meter.update(depth, gt, tags=(ek.city_tag(stem),))

    # Final metrics and latency; printed as a table and saved to results/depth_*.json.
    res, lat = meter.results(), timer.summary()
    ek.print_depth_table(res)
    if lat:
        print(f"\nLatency (model call only, batch 1): {lat['mean_ms']:.1f} ± {lat['std_ms']:.1f} ms, "
              f"p95 {lat['p95_ms']:.1f} ms, {lat['fps']:.1f} FPS | peak GPU memory {ek.peak_gpu_gb():.2f} GB | "
              f"{n_params:.1f} M parameters")
    if skipped:
        print(f"Skipped {skipped} images (missing file or size mismatch)")

    tag = f"{ek.model_tag(args.model)}_{args.camera}_{args.split}" + (f"_{args.input_size}" if args.input_size else "")
    ek.save_json(args.out_dir / f"depth_{tag}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "params_M": n_params,
                  "latency": lat, "peak_gpu_gb": ek.peak_gpu_gb(), "metrics": res})


if __name__ == "__main__":
    main()
