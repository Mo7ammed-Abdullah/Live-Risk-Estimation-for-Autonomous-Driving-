"""
Save the segmentation and depth maps of every frame of a split once (half size, 450 x 800), so that object finding,
tracking and TTC variants can be tried on the CPU in seconds without running the models again.

Output:  cache/segdepth_<split>_seg.npy    (frames, 450, 800) uint8   class id per pixel
         cache/segdepth_<split>_depth.npy  (frames, 450, 800) float16 metric depth (m)
         cache/segdepth_<split>.json       frame names in the same order

Run (venv311 active, project folder):
    python code/cache_segdepth.py --split val       # ~2 min, ~430 MB
"""
import argparse
import json
from pathlib import Path

import numpy as np

import evalkit as ek

HERE = Path(__file__).resolve().parent.parent
OUT_H, OUT_W = 450, 800


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt_native")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    args = ap.parse_args()

    import cv2
    from PIL import Image
    from tqdm import tqdm
    from eval_depth import load_depth_model, run_depth
    from generate_seg_gt import load_labeler, predict_with_retry

    stems = [s for s in sorted(ek.load_split(args.split, args.final)) if (args.images / f"{s}.jpg").exists()]
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    meta = json.load(open(args.meta_cache))
    device = ek.get_device()
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)
    out = HERE / "cache"
    out.mkdir(exist_ok=True)
    seg_mm = np.lib.format.open_memmap(out / f"segdepth_{args.split}_seg.npy", mode="w+", dtype=np.uint8, shape=(len(stems), OUT_H, OUT_W))
    dep_mm = np.lib.format.open_memmap(out / f"segdepth_{args.split}_depth.npy", mode="w+", dtype=np.float16, shape=(len(stems), OUT_H, OUT_W))
    for i, stem in enumerate(tqdm(stems, desc=f"caching {args.split}")):
        img = Image.open(args.images / f"{stem}.jpg").convert("RGB")
        sd = meta["sd"].get(f"{stem}.jpg")
        K = meta["cs"][sd["calibrated_sensor_token"]]["camera_intrinsic"] if sd else None
        seg = np.minimum(predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height), len(names) - 1)
        depth = np.clip(np.nan_to_num(run_depth(depth_model, np.array(img), K), nan=80.0, posinf=80.0), 0, 80)
        seg_mm[i] = cv2.resize(seg.astype(np.uint8), (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
        dep_mm[i] = cv2.resize(depth.astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    seg_mm.flush(); dep_mm.flush()
    json.dump({"stems": stems, "size": [OUT_H, OUT_W], "scale": 0.5, "names": names}, open(out / f"segdepth_{args.split}.json", "w"))
    print(f"Saved cache/segdepth_{args.split}_*.npy ({len(stems)} frames)")


if __name__ == "__main__":
    main()
