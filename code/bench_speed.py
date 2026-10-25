"""
Speed of the WHOLE pipeline (segmentation + depth + risk + decision) at several input sizes.
Uses validation images only (the test split is not touched). Every stage is timed with a GPU sync.

    python bench_speed.py                       # sizes 1600x900, 960x540, 640x360, 640x320
    python bench_speed.py --n 100
    python bench_speed.py --sizes 640x320 --resolution_level 5

The paper reports 8.6-11.2 FPS at 640x320 on an A1000. Compare the "total" column.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import evalkit as ek
import risk as rk
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, predict_with_retry

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="images to time (after warm-up)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sizes", nargs="*", default=["1600x900", "960x540", "640x360", "640x320"])
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--resolution_level", type=int, default=None, help="UniDepthV2 internal resolution (0-9), lower = faster")
    ap.add_argument("--depth_fp16", action="store_true", help="run UniDepth under fp16 autocast")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    import torch
    device = ek.get_device()
    # Speed is timed on validation images only; the test split is never touched.
    stems = ek.load_split("val", False, 1, args.n + args.warmup)
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    w_vec = rk.class_weight_vector(names)
    # The timing includes the real decision step (fitted rule: max per region).
    alphas = rk.fitted_alphas(rk.DEFAULT_REGIONS)
    # Load the images once, so disk reading is not part of the timing.
    imgs = [Image.open(args.images / f"{s}.jpg").convert("RGB") for s in stems]

    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device, args.resolution_level)
    print(f"device {device} | {len(imgs)} val images ({args.warmup} warm-up) | depth resolution_level {args.resolution_level} | depth fp16 {args.depth_fp16}")

    results = {}
    # Time the whole pipeline at each input size (default 1600x900 down to 640x320).
    for size in args.sizes:
        W, H = (int(v) for v in size.split("x"))
        frames = [im.resize((W, H), Image.BILINEAR) for im in imgs]
        t_seg, t_dep, t_risk, t_tot = [], [], [], []
        for i, img in enumerate(frames):
            rgb = np.array(img)
            # GPU work is asynchronous: synchronise before each timestamp so the times are real.
            ek.cuda_sync(); t0 = time.perf_counter()
            # Stage 1: segmentation (time t0 -> t1).
            seg = predict_with_retry(processor, seg_model, img, device, device.type == "cuda", H)
            ek.cuda_sync(); t1 = time.perf_counter()
            # Stage 2: depth, with real metric output (time t1 -> t2). The camera is unknown here (K = None), as for any video.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.depth_fp16):
                depth = run_depth(depth_model, rgb, None)
            ek.cuda_sync(); t2 = time.perf_counter()
            # Make sure depth and segmentation have the same size before fusing.
            if seg.shape != depth.shape:
                depth = np.array(Image.fromarray(depth).resize((seg.shape[1], seg.shape[0]), Image.BILINEAR))
            # Stage 3: risk map and region decision (time t2 -> t3).
            r = rk.compute_risk(np.minimum(seg, len(names) - 1), depth, w_vec, "metric")
            rk.region_urgency(r, rk.DEFAULT_REGIONS, alphas, p=rk.FITTED_P)
            t3 = time.perf_counter()
            # The first frames (warm-up, GPU start-up) are not counted.
            if i >= args.warmup:
                t_seg.append(t1 - t0); t_dep.append(t2 - t1); t_risk.append(t3 - t2); t_tot.append(t3 - t0)
        m = lambda x: 1000 * float(np.mean(x))
        # Mean time of each stage, total mean and 95th percentile, and FPS = 1 / mean total time.
        results[size] = {"seg_ms": m(t_seg), "depth_ms": m(t_dep), "risk_ms": m(t_risk), "total_ms": m(t_tot),
                         "total_p95_ms": 1000 * float(np.percentile(t_tot, 95)), "fps": 1.0 / float(np.mean(t_tot))}
        r_ = results[size]
        print(f"{size:>9s}  seg {r_['seg_ms']:6.1f} ms | depth {r_['depth_ms']:6.1f} ms | risk {r_['risk_ms']:5.1f} ms | "
              f"total {r_['total_ms']:6.1f} ms (p95 {r_['total_p95_ms']:.0f}) = {r_['fps']:.1f} FPS")
    print(f"peak GPU memory {ek.peak_gpu_gb():.2f} GB")
    args.out_dir.mkdir(exist_ok=True)
    ek.save_json(args.out_dir / ("speed_pipeline_fp16.json" if args.depth_fp16 else "speed_pipeline.json"), {"resolution_level": args.resolution_level, "sizes": results})
    print("Saved results\\speed_pipeline.json")


if __name__ == "__main__":
    main()
