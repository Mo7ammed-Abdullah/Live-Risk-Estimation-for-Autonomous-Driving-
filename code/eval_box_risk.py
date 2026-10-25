"""
Does the system's risk match the TRUE risk of real objects?   (ground truth = nuScenes 3D boxes)

For every annotated person / rider / bicycle / motorcycle / car / truck / bus the true risk is the paper's formula
with the true class and the true distance (nearest surface along the camera axis, from the 3D box):
        R_true = w_class * exp(-lambda * distance_m)
The system's risk is read from its own risk map (predicted segmentation + predicted depth) inside the object's 2D box.

Reported (validation split only, unless --final):
  1. Object level   - correlation / error between true and system risk, by distance and by class.
                      Was the object recognised as the right class, and is the depth at the object right?
  2. Detection      - of the objects whose true risk is above a threshold, how many does the system flag?
                      And of the system's high-risk pixels, how many lie on a real object (pixel precision)?
  3. Decision       - which horizontal region (far / mid / near) contains the truly most urgent object, and does the
                      system choose the same region?  Compared with the trivial "always near" guess.
Every object is also written to a table (results/object_table_<split>.csv) so class weights and lambda can be fitted
later WITHOUT running the models again.

Run (venv311 active, inside "Testing 3"), after python generate_box_gt.py:
    python eval_box_risk.py --limit 50
    python eval_box_risk.py
    python eval_box_risk.py --split test --final      # only at the very end

Notes: the 2D box of a 3D box also contains some background, and a nearer object can overlap it, so "system risk in box"
(the maximum inside the box) is slightly optimistic. Objects hidden by more than 60 % are skipped (--min_vis 3 keeps
only >60 % visible; default 2 keeps >40 %).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
import risk as rk
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, predict_with_retry
from make_splits import RISK_CLASSES

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)


# Rank of each value (used for Spearman correlation = Pearson on ranks).
def ranks(a):
    return np.argsort(np.argsort(a)).astype(np.float64)


# Pearson correlation; NaN when there are too few objects or no variation.
def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# Summary for a group of objects:
# correlation and error between true and system risk, share of objects with the right class,
# depth error at the object, and detection (flagged / false alarms) at each threshold.
def summarize(rows, tau_list):
    """rows: list of dicts with r_true, r_sys, cls, pred_cls, zmin, depth_sys"""
    if not rows:
        return None
    t = np.array([r["r_true"] for r in rows])
    s = np.array([r["r_sys"] for r in rows])
    out = {"objects": len(rows), "MAE": float(np.abs(t - s).mean()), "pearson": corr(t, s),
           "spearman": corr(ranks(t), ranks(s)),
           "class_correct": float(np.mean([r["pred_cls"] == r["cls"] for r in rows]))}
    # objects closer than 2 m are left out of the depth check: their 3D box straddles the camera plane, so the
    # "nearest corner" distance is meaningless there and would blow up the relative error
    # Depth check only for objects farther than 2 m (closer boxes straddle the camera plane, so their distance is unreliable).
    d_ok = [r for r in rows if np.isfinite(r["depth_sys"]) and r["zmin"] > 2.0]
    if d_ok:
        z = np.array([r["zmin"] for r in d_ok])
        ds = np.array([r["depth_sys"] for r in d_ok])
        out["depth_AbsRel_at_object"] = float(np.median(np.abs(ds - z) / z))
        out["depth_found"] = len(d_ok) / len(rows)
    det = {}
    for tau in tau_list:
        # Detection: objects whose TRUE risk is >= tau are 'risky'; flagged = the system's risk also >= tau; false alarm = flagged but not truly risky.
        pos = t >= tau
        if pos.any():
            det[str(tau)] = {"true_risky": int(pos.sum()), "flagged": float((s[pos] >= tau).mean()),
                             "false_alarm_on_non_risky": float((s[~pos] >= tau).mean()) if (~pos).any() else float("nan")}
    out["detection"] = det
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    best = HERE / "runs" / "seg_b2_native" / "best"
    ap.add_argument("--seg_model", default=str(best) if best.exists() else "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--depth_model", default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--lam_metric", type=float, default=rk.DEFAULT_LAMBDA_METRIC)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--pixel_tau", type=float, default=0.2, help="risk level counted as HIGH for pixel precision")
    ap.add_argument("--min_vis", type=int, default=2, help="1..4, nuScenes visibility level (2 = more than 40 %% visible)")
    ap.add_argument("--min_box_px", type=int, default=8, help="skip objects whose box is smaller than this (pixels high)")
    ap.add_argument("--n_regions", type=int, default=rk.DEFAULT_REGIONS)
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--background", type=float, default=rk.BACKGROUND_WEIGHT)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    if not args.boxes.exists():
        sys.exit("box_gt/boxes.json not found - run: python generate_box_gt.py")
    boxes = json.load(open(args.boxes))
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    w_vec = rk.class_weight_vector(names, json.load(open(args.weights)) if args.weights else None, args.background)
    # Class ids of the risk classes, used to find which class the model saw inside each box.
    risk_ids = [names.index(c) for c in RISK_CLASSES if c in names]
    id2name = {i: names[i] for i in risk_ids}
    # Region weights alpha (default 0.4 / 0.7 / 1.0), used for both the true and the system decision.
    alphas = rk.default_alphas(args.n_regions) if args.alphas is None else np.asarray(args.alphas, dtype=np.float64)
    meta = json.load(open(args.meta_cache)) if args.meta_cache.exists() else None
    device = ek.get_device()
    print(f"Split {args.split} ({len(stems)} images) | lambda {args.lam_metric}/m | min visibility {args.min_vis} | device {device}")

    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    # rows = one entry per real object. The frame_* lists store the true region and the system's choice (mean rule and max rule) per frame.
    rows, skipped = [], 0
    frame_gt, frame_sys_mean, frame_sys_max = [], [], []
    pix_hi = pix_hi_on_obj = 0
    n_no_obj = 0
    zn = rk.region_names(args.n_regions)

    for stem in tqdm(stems, desc="box risk evaluation"):
        img_p = args.images / f"{stem}.jpg"
        if not img_p.exists() or stem not in boxes:
            skipped += 1
            continue
        img = Image.open(img_p).convert("RGB")
        K = None
        if meta is not None and f"{stem}.jpg" in meta["sd"]:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        # Run both models on the photo, then build the system's risk map from them.
        seg = predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height)
        depth = run_depth(depth_model, np.array(img), K)
        if seg.shape != depth.shape:
            skipped += 1
            continue
        H, W = seg.shape
        # The system's risk map (predicted segmentation + predicted depth, metric mode).
        rmap = rk.compute_risk(np.minimum(seg, len(names) - 1), depth, w_vec, "metric", lam_metric=args.lam_metric,
                               max_depth=args.max_depth)
        city = ek.city_tag(stem)

        risky_boxes = [o for o in boxes[stem] if o["cls"]]
        # Mark every real risk object's box, to measure how many high-risk pixels lie on real objects (pixel precision).
        cover = np.zeros((H, W), dtype=bool)               # any real risk-class object, any visibility
        for o in risky_boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in o["box"]]
            cover[max(y0, 0):y1, max(x0, 0):x1] = True
        hi = rmap >= args.pixel_tau
        pix_hi += int(hi.sum())
        pix_hi_on_obj += int((hi & cover).sum())

        # Score each real object that is visible enough, within range and big enough in the image.
        objs = []
        for o in risky_boxes:
            if o["vis"] < args.min_vis or o["zmin"] > args.max_depth:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in o["box"]]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if y1 - y0 < args.min_box_px or x1 - x0 < 2:
                continue
            reg_r, reg_s, reg_d = rmap[y0:y1, x0:x1], seg[y0:y1, x0:x1], depth[y0:y1, x0:x1]
            # Which risk class did the model mostly see inside this box, and what depth did it give there?
            counts = {i: int((reg_s == i).sum()) for i in risk_ids}
            top = max(counts, key=counts.get) if counts else None
            if top is not None and counts[top] > 0:
                pc, pdep = id2name[top], float(np.median(reg_d[reg_s == top]))
                pfrac = counts[top] / reg_s.size
            else:
                pc, pdep, pfrac = "none", float("nan"), 0.0
            w = float(w_vec[names.index(o["cls"])]) if o["cls"] in names else args.background
            # TRUE risk = class weight of the real class x exp(-lambda x true nearest-surface distance from the 3D box).
            # Note: uses the same weights and lambda as the system, so it is independent of perception errors but not of the formula.
            r_true = w * float(np.exp(-args.lam_metric * min(o["zmin"], args.max_depth)))
            # One row per object: truth (class, distance, true risk) and system (risk in box = maximum inside the box, predicted class, depth).
            row = {"stem": stem, "city": city, "cls": o["cls"], "vis": o["vis"], "zmin": o["zmin"], "lateral": o["lateral"],
                   "bottom_row": y1, "r_true": r_true, "r_sys": float(reg_r.max()), "pred_cls": pc,
                   "pred_frac": pfrac, "depth_sys": pdep, "npts": o["npts"]}
            rows.append(row)
            objs.append(row)

        if not objs:
            n_no_obj += 1
            continue
        # Decision truth: the region (far/mid/near) of the object with the highest region-weighted true risk, judged by the bottom edge of its box.
        band_edges = np.linspace(0, H, args.n_regions + 1)
        gt_best = max(objs, key=lambda r: alphas[min(int(np.searchsorted(band_edges, r["bottom_row"] - 1, "right") - 1),
                                                     args.n_regions - 1)] * r["r_true"])
        frame_gt.append(min(int(np.searchsorted(band_edges, gt_best["bottom_row"] - 1, "right") - 1), args.n_regions - 1))
        # System decisions for the same frame: paper rule (mean per region) and max-per-region rule.
        frame_sys_mean.append(rk.region_urgency(rmap, args.n_regions, alphas)[1])
        bands = np.array_split(np.arange(H), args.n_regions)
        frame_sys_max.append(int(np.argmax([alphas[i] * rmap[b].max() for i, b in enumerate(bands)])))

    if not rows:
        sys.exit("No objects evaluated.")
    taus = args.taus
    # Summaries overall, by distance range, by class and by city.
    result = {"all": summarize(rows, taus)}
    for lo, hi_ in ek.DIST_BINS:
        result[f"{lo}-{hi_} m"] = summarize([r for r in rows if lo < r["zmin"] <= hi_], taus)
    for cls in RISK_CLASSES:
        result[f"class: {cls}"] = summarize([r for r in rows if r["cls"] == cls], taus)
    for c in sorted({r["city"] for r in rows}):
        result[f"city: {c}"] = summarize([r for r in rows if r["city"] == c], taus)

    print(f"\n{len(rows)} real risk-class objects in {len(frame_gt)} frames "
          f"({n_no_obj} frames had none, {skipped} skipped)\n")
    print("=== object level: true risk vs system risk in the object's box ===")
    print(f"{'group':<18s}{'objects':>8s}{'Pearson':>9s}{'Spearman':>10s}{'MAE':>8s}{'class ok':>10s}{'depth AbsRel (median)':>23s}")
    for k, v in result.items():
        if v:
            dr = v.get("depth_AbsRel_at_object", float("nan"))
            print(f"{k:<18s}{v['objects']:>8d}{v['pearson']:>9.3f}{v['spearman']:>10.3f}{v['MAE']:>8.3f}"
                  f"{v['class_correct']:>10.2f}{dr:>23.3f}")
    print("\n=== detection: objects whose TRUE risk >= tau that the system also flags (system risk in box >= tau) ===")
    for key in ["all"] + [f"{lo}-{hi_} m" for lo, hi_ in ek.DIST_BINS]:
        v = result.get(key)
        if not v:
            continue
        print(f"\n--- {key} ---")
        print(f"{'tau':>6s}{'true risky':>12s}{'flagged':>9s}{'false alarm on non-risky':>26s}")
        for tau, d in v["detection"].items():
            print(f"{float(tau):>6.2f}{d['true_risky']:>12d}{d['flagged']:>9.2f}{d['false_alarm_on_non_risky']:>26.2f}")
    # Pixel precision: share of the system's high-risk pixels that lie inside a real object's box.
    prec = pix_hi_on_obj / max(pix_hi, 1)
    print(f"\nPixel precision: {100 * prec:.1f}% of the system's high-risk pixels (risk >= {args.pixel_tau}) lie inside the "
          f"box of a real person/rider/vehicle ({pix_hi} pixels).")

    # Decision accuracy: how often the system picks the truly most urgent region, versus always answering the most common one.
    gt = np.array(frame_gt)
    dec = {}
    for name, sysd in (("paper rule (mean per region)", np.array(frame_sys_mean)),
                       ("max per region", np.array(frame_sys_max))):
        conf = np.zeros((args.n_regions, args.n_regions), dtype=np.int64)
        for g, s in zip(gt, sysd):
            conf[g, s] += 1
        dec[name] = {"agreement": float(np.trace(conf) / max(len(gt), 1)), "confusion": conf}
    base_zone = int(np.bincount(gt, minlength=args.n_regions).argmax())
    base = float(np.mean(gt == base_zone))
    print(f"\n=== decision: region of the truly most urgent object vs the system's choice ({len(gt)} frames) ===")
    print(f"True most-urgent region: " + ", ".join(f"{n} {int((gt == i).sum())}" for i, n in enumerate(zn)))
    print(f"Trivial baseline 'always {zn[base_zone]}': {100 * base:.1f}%")
    for name, d in dec.items():
        print(f"System, {name}: {100 * d['agreement']:.1f}%  (gain over baseline {100 * (d['agreement'] - base):+.1f} points)")
        print("   rows = true region, columns = system region")
        print("             " + "".join(f"{n:>8s}" for n in zn))
        for i, n in enumerate(zn):
            print(f"   {n:<10s}" + "".join(f"{c:>8d}" for c in d["confusion"][i]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save every object to a table so weights and lambda can be fitted later without running the models again.
    with open(args.out_dir / f"object_table_{args.split}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    ek.save_json(args.out_dir / f"box_risk_{args.split}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "objects": result,
                  "pixel_precision": prec, "decision": dec, "baseline_always_most_common": base,
                  "true_region_counts": np.bincount(gt, minlength=args.n_regions)})
    print(f"\nSaved results to {args.out_dir}")


if __name__ == "__main__":
    main()
