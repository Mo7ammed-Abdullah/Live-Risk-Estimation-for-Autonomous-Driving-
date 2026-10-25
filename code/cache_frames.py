"""
Run the models ONCE and store a compact summary of every frame, so that decision rules can be tried on the CPU in seconds.

Per frame it keeps, for 6 horizontal strips of the image, a histogram: how many pixels have (predicted class, predicted depth bin).
From that histogram the risk map's mean / max / any power-mean for ANY class weights, lambda or region weights can be
recomputed exactly (depth resolution 0.5 m). It also keeps the real objects (nuScenes 3D boxes): class, nearest distance, image row
of the object's bottom edge.

Run (venv311 active, inside "Testing 3"):
    python cache_frames.py                       # validation split  -> cache/frames_val.npz
    python cache_frames.py --split test --final  # only at the very end
Then:  python fit_decision.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, predict_with_retry
from make_splits import RISK_CLASSES

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# The image is cut into 6 horizontal strips; depth is binned in 0.5 m steps from 0 to 80 m (160 bins).
N_STRIPS, BIN_M, N_BINS = 6, 0.5, 160        # 160 bins x 0.5 m = 0..80 m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    best = HERE / "runs" / "seg_b2_native" / "best"
    ap.add_argument("--seg_model", default=str(best) if best.exists() else "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--depth_model", default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "cache")
    args = ap.parse_args()

    if not args.boxes.exists():
        sys.exit("box_gt/boxes.json not found - run: python generate_box_gt.py")
    # Real objects from the 3D boxes (box_gt/boxes.json).
    boxes = json.load(open(args.boxes))
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    n_cls = len(names)
    meta = json.load(open(args.meta_cache)) if args.meta_cache.exists() else None
    device = ek.get_device()
    print(f"Split {args.split} ({len(stems)} images) | device {device}")
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    # kept = frames cached; counts = one histogram per frame; o_* = real objects (frame, class, distance, row of bottom edge, visibility).
    kept, counts = [], []
    o_frame, o_cls, o_zmin, o_bottom, o_vis = [], [], [], [], []
    H_used = None
    for stem in tqdm(stems, desc="caching frames"):
        img_p = args.images / f"{stem}.jpg"
        if not img_p.exists() or stem not in boxes:
            continue
        img = Image.open(img_p).convert("RGB")
        K = None
        if meta is not None and f"{stem}.jpg" in meta["sd"]:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        # Run both models: predicted class per pixel and predicted metric depth (cleaned and limited to 0-80 m).
        seg = np.minimum(predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height), n_cls - 1)
        depth = np.clip(np.nan_to_num(run_depth(depth_model, np.array(img), K), nan=args.max_depth, posinf=args.max_depth),
                        0, args.max_depth)
        if seg.shape != depth.shape:
            continue
        H, W = seg.shape
        H_used = H
        # Per strip, count how many pixels have each (predicted class, depth bin).
        # From this histogram the risk mean/max of a strip can be recomputed exactly for ANY class weights, lambda or alpha, without running the models again.
        hist = np.zeros((N_STRIPS, n_cls, N_BINS), dtype=np.uint32)
        bins = np.minimum((depth / BIN_M).astype(np.int64), N_BINS - 1)
        for s, rows in enumerate(np.array_split(np.arange(H), N_STRIPS)):
            # One bincount over (class x depth bin) for the strip.
            idx = seg[rows[0]:rows[-1] + 1].astype(np.int64) * N_BINS + bins[rows[0]:rows[-1] + 1]
            hist[s] = np.bincount(idx.ravel(), minlength=n_cls * N_BINS).reshape(n_cls, N_BINS)
        fi = len(kept)
        kept.append(stem)
        counts.append(hist)
        # Also store the real objects that are visible enough, within range and tall enough.
        for o in boxes[stem]:
            if not o["cls"] or o["cls"] not in names or o["vis"] < args.min_vis or o["zmin"] > args.max_depth:
                continue
            y0, y1 = max(int(round(o["box"][1])), 0), min(int(round(o["box"][3])), H)
            if y1 - y0 < args.min_box_px:
                continue
            o_frame.append(fi)
            o_cls.append(names.index(o["cls"]))
            o_zmin.append(o["zmin"])
            o_bottom.append(y1)
            o_vis.append(o["vis"])

    if not kept:
        sys.exit("Nothing cached.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"frames_{args.split}.npz"
    # Save everything compactly in cache/frames_<split>.npz (used by fit_decision.py).
    np.savez_compressed(out, counts=np.stack(counts), stems=np.array(kept), class_names=np.array(names), H=H_used,
                        bin_m=BIN_M, obj_frame=np.array(o_frame), obj_cls=np.array(o_cls), obj_zmin=np.array(o_zmin),
                        obj_bottom=np.array(o_bottom), obj_vis=np.array(o_vis))
    print(f"Saved {out}: {len(kept)} frames, {len(o_frame)} objects")


if __name__ == "__main__":
    main()
