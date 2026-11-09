"""
Time-to-collision (TTC) from two consecutive camera frames, using the EXISTING models (no training).

For every real object (risk class, located with its nuScenes box in both frames):
  1. depth method   the object's depth in the previous and the current frame (UniDepth, on the pixels that SegFormer
                    labels as the object's class inside its box, a low percentile = nearest surface)
                    closing speed = (depth_prev - depth_now) / dt,   TTC = depth_now / closing speed
  2. scale method   how much the object's segmented area grew between the frames (no depth needed):
                    s = sqrt(area_now / area_prev),   TTC = dt / (s - 1)
  3. combined       median of the two when both exist, otherwise whichever exists
"Not approaching" (closing <= --min_closing m/s, or s <= 1) means no TTC.

Truth: box_gt/motion.json (made by build_motion_labels.py) - the real closing speed and TTC from nuScenes positions.
Object locations come from the nuScenes boxes (which pixels belong to which object, and the same object in both
frames); the depth / size MEASUREMENTS come only from the camera models. A fully automatic version would need its own
object tracking; that is stated as a limitation.

Metrics (objects with motion truth):
    closing-speed error (m/s);  TTC error for objects truly approaching with TTC <= 10 s: median error (s),
    share within 1 s and within 20 %;  danger detection (true TTC < 4 s): AP, AUROC, precision / recall of
    "estimated TTC < 4 s";  compared with the fixed risk formula and distance only.  Split by distance and class.

Run (venv311 active, project folder):
    python code/estimate_ttc.py --split val            # runs the models (~3 min), saves cache/ttc_meas_val.json, prints metrics
    python code/estimate_ttc.py --split val --eval_only --percentile 25   # re-score saved measurements with other settings
Final, once the settings are chosen on validation:
    python code/estimate_ttc.py --split test --final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import evalkit as ek
import risk as rk

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
PERCENTILES = [5, 10, 25, 50]
# Which predicted classes count as "the object" for each annotated class.
GROUP = {"car": {"car", "truck", "bus", "train"}, "truck": {"car", "truck", "bus", "train"},
         "bus": {"car", "truck", "bus", "train"}, "train": {"car", "truck", "bus", "train"},
         "person": {"person", "rider"}, "rider": {"person", "rider", "bicycle", "motorcycle"},
         "bicycle": {"bicycle", "rider", "person"}, "motorcycle": {"motorcycle", "rider", "person"}}


def ts_of(stem):
    return int(stem.split("__")[2]) / 1e6


def measure(seg, depth, box, cls_ids, names, w_vec, max_depth=80.0, min_px=20):
    """Depth percentiles, segmented area and fixed-formula score for one object box in one frame."""
    H, W = seg.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), W), min(max(y1, y0 + 1), H)
    s, d = seg[y0:y1, x0:x1], depth[y0:y1, x0:x1]
    mask = np.isin(s, cls_ids)
    n = int(mask.sum())
    if n >= min_px:
        vals, fallback = d[mask], False
    else:                                           # too few object pixels: use the lower half of the box
        vals, fallback = d[(y1 - y0) // 2:, :].ravel(), True
    vals = np.clip(vals, 0, max_depth)
    risk = w_vec[s] * np.exp(-rk.DEFAULT_LAMBDA_METRIC * np.clip(d, 0, max_depth))
    return {"n_px": n, "fallback": fallback, "area": float(max(n, 1)),
            "depth": {str(p): float(np.percentile(vals, p)) for p in PERCENTILES},
            "fixed": float(risk.max()), "dist_only": float(-np.percentile(vals, 10))}


def run_models(args, stems, boxes, motion):
    from PIL import Image
    from tqdm import tqdm
    from eval_depth import load_depth_model, run_depth
    from generate_seg_gt import load_labeler, predict_with_retry

    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    w_vec = rk.class_weight_vector(names)
    meta = json.load(open(args.meta_cache))
    device = ek.get_device()
    print(f"Split {args.split}: {len(stems)} frames | device {device}\n  seg   {args.seg_model}\n  depth {args.depth_model}")
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    stems = sorted(stems)
    rows, prev = [], None                          # prev = (stem, seg, depth)
    for stem in tqdm(stems, desc="measuring"):
        img_p = args.images / f"{stem}.jpg"
        if not img_p.exists():
            prev = None
            continue
        img = Image.open(img_p).convert("RGB")
        sd = meta["sd"].get(f"{stem}.jpg")
        K = meta["cs"][sd["calibrated_sensor_token"]]["camera_intrinsic"] if sd else None
        seg = np.minimum(predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height), len(names) - 1)
        depth = np.nan_to_num(run_depth(depth_model, np.array(img), K), nan=80.0, posinf=80.0)
        ok_prev = (prev is not None and prev[0].split("__")[0] == stem.split("__")[0]
                   and 0 < ts_of(stem) - ts_of(prev[0]) <= args.max_gap)
        if ok_prev:
            dt = ts_of(stem) - ts_of(prev[0])
            prev_boxes = {o["inst"]: o for o in boxes.get(prev[0], [])}
            truth = {r["inst"]: r for r in motion.get(stem, [])}
            for o in boxes.get(stem, []):
                if o["cls"] not in GROUP or o["inst"] not in prev_boxes or o["inst"] not in truth:
                    continue
                if o["vis"] < args.min_vis or o["zmin"] > 80 or (o["box"][3] - o["box"][1]) < args.min_box_px:
                    continue
                t = truth[o["inst"]]
                if not t["has_motion"]:
                    continue
                ids = [names.index(c) for c in GROUP[o["cls"]] if c in names]
                now = measure(seg, depth, o["box"], ids, names, w_vec)
                before = measure(prev[1], prev[2], prev_boxes[o["inst"]]["box"], ids, names, w_vec)
                rows.append({"stem": stem, "inst": o["inst"], "cls": o["cls"], "dt": dt, "z_true": o["zmin"],
                             "closing_true": t["closing_z"], "ttc_true": t["ttc_z"], "now": now, "prev": before})
        prev = (stem, seg, depth)
    return rows


# ----------------------------------------------------------------------------- scoring
def estimates(r, p, min_closing):
    z_now, z_prev = r["now"]["depth"][str(p)], r["prev"]["depth"][str(p)]
    dt = r["dt"]
    closing_d = (z_prev - z_now) / dt
    ttc_d = z_now / closing_d if closing_d > min_closing else None
    s = np.sqrt(r["now"]["area"] / max(r["prev"]["area"], 1.0))
    ok_scale = r["now"]["n_px"] >= 20 and r["prev"]["n_px"] >= 20
    ttc_s = dt / (s - 1) if ok_scale and s > 1.0 + 1e-3 else None
    closing_s = z_now * (s - 1) / dt if ok_scale else None
    if ttc_d is not None and ttc_s is not None:
        ttc_c = float(np.median([ttc_d, ttc_s]))
    else:
        ttc_c = ttc_d if ttc_d is not None else ttc_s
    return {"depth": (closing_d, ttc_d), "scale": (closing_s, ttc_s), "combined": (None, ttc_c)}


def average_precision(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = ((np.cumsum(counts) - counts) + (counts + 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(rows, p, min_closing, danger_s=4.0, max_ttc=10.0):
    out = {}
    y = np.array([r["ttc_true"] is not None and r["ttc_true"] < danger_s for r in rows], dtype=float)
    true_close = np.array([r["closing_true"] for r in rows])
    true_ttc = np.array([r["ttc_true"] if r["ttc_true"] is not None else np.inf for r in rows])
    est = [estimates(r, p, min_closing) for r in rows]
    for m in ("depth", "scale", "combined"):
        closing = np.array([e[m][0] if e[m][0] is not None else np.nan for e in est])
        ttc = np.array([e[m][1] if e[m][1] is not None else np.inf for e in est])
        sel = np.isfinite(true_ttc) & (true_ttc <= max_ttc)
        err = np.abs(ttc[sel] - true_ttc[sel])
        found = np.isfinite(ttc[sel])
        score = np.where(np.isfinite(ttc), 1.0 / np.maximum(ttc, 1e-3), 0.0)
        flag = ttc < danger_s
        res = {"objects": len(rows), "approaching_ttc<=10s": int(sel.sum()),
               "TTC found (approaching)": float(found.mean()) if sel.any() else np.nan,
               "median TTC error (s)": float(np.median(err[found])) if found.any() else np.nan,
               "within 1 s": float((err <= 1.0).mean()) if sel.any() else np.nan,
               "within 20%": float((err <= 0.2 * true_ttc[sel]).mean()) if sel.any() else np.nan,
               "AP (TTC<4s)": average_precision(y, score), "AUROC (TTC<4s)": auroc(y, score),
               "precision (TTC<4s)": float(y[flag].mean()) if flag.any() else np.nan,
               "recall (TTC<4s)": float(flag[y == 1].mean()) if y.any() else np.nan}
        if np.isfinite(closing).any():
            ok = np.isfinite(closing)
            res["closing speed MAE (m/s)"] = float(np.mean(np.abs(closing[ok] - true_close[ok])))
            res["closing speed median err (m/s)"] = float(np.median(np.abs(closing[ok] - true_close[ok])))
        out[m] = res
    for name, key in (("fixed formula", "fixed"), ("distance only", "dist_only")):
        s = np.array([r["now"][key] for r in rows])
        out[name] = {"objects": len(rows), "AP (TTC<4s)": average_precision(y, s), "AUROC (TTC<4s)": auroc(y, s)}
    out["_danger_objects"] = int(y.sum())
    return out


def print_table(res, title):
    cols = ["TTC found (approaching)", "median TTC error (s)", "within 1 s", "within 20%", "closing speed median err (m/s)",
            "AP (TTC<4s)", "AUROC (TTC<4s)", "precision (TTC<4s)", "recall (TTC<4s)"]
    short = ["found", "med err s", "<=1 s", "<=20%", "speed err", "AP", "AUROC", "prec", "recall"]
    print(f"\n=== {title} ===")
    print(f"{'method':16s}" + "".join(f"{c:>11s}" for c in short))
    for m in ("depth", "scale", "combined", "fixed formula", "distance only"):
        r = res[m]
        print(f"{m:16s}" + "".join(f"{r[c]:11.3f}" if c in r and r[c] == r[c] else f"{'-':>11s}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="re-score cache/ttc_meas_<split>.json without running models")
    ap.add_argument("--percentile", type=int, default=10, choices=PERCENTILES, help="depth percentile inside the object mask")
    ap.add_argument("--min_closing", type=float, default=0.5, help="closing speed below this (m/s) = not approaching")
    ap.add_argument("--max_gap", type=float, default=0.7, help="max seconds between the two frames")
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt_native")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The TEST split is reserved for your final numbers. Add --final only when everything is frozen.")

    meas = HERE / "cache" / f"ttc_meas_{args.split}.json"
    if args.eval_only:
        rows = json.load(open(meas))
    else:
        stems = ek.load_split(args.split, args.final)
        boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
        motion = json.load(open(HERE / "box_gt" / "motion.json"))
        rows = run_models(args, stems, boxes, motion)
        meas.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(meas, "w"))
        print(f"Saved measurements: {meas} ({len(rows)} objects)")

    res = evaluate(rows, args.percentile, args.min_closing)
    print_table(res, f"{args.split}: {len(rows)} objects, {res['_danger_objects']} with true TTC < 4 s "
                     f"(depth percentile {args.percentile}, min closing {args.min_closing} m/s)")
    # By true distance and by class (combined method).
    for label, sel in [("0-20 m", lambda r: r["z_true"] < 20), ("20-50 m", lambda r: 20 <= r["z_true"] < 50),
                       ("50-80 m", lambda r: r["z_true"] >= 50),
                       ("vehicles", lambda r: r["cls"] in ("car", "truck", "bus", "train")),
                       ("people", lambda r: r["cls"] == "person"),
                       ("two-wheelers", lambda r: r["cls"] in ("bicycle", "motorcycle", "rider"))]:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 10:
            continue
        rr = evaluate(sub, args.percentile, args.min_closing)["combined"]
        print(f"  {label:12s} n={len(sub):5d}  TTC<=10s n={rr['approaching_ttc<=10s']:5d}  median err {rr['median TTC error (s)']:.2f} s"
              f"  within 1 s {rr['within 1 s']:.2f}  within 20% {rr['within 20%']:.2f}  AUROC {rr['AUROC (TTC<4s)']:.3f}")
    out = HERE / "results" / f"ttc_{args.split}_p{args.percentile}.json"
    json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": res}, open(out, "w"), indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
