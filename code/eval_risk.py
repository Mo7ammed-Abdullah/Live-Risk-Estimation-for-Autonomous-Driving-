"""
How good is the risk map, measured against the SAME formula fed with ground truth?

For every frame four risk maps are built with the paper's formula (metric depth mode):
    O  oracle     : ground-truth segmentation + LiDAR depth                 <- the reference
    P  system     : predicted segmentation + predicted depth                <- what your pipeline outputs
    S  seg=GT     : ground-truth segmentation + predicted depth             -> error caused by DEPTH only
    D  depth=GT   : predicted segmentation + LiDAR depth                    -> error caused by SEGMENTATION only
LiDAR depth exists only at a few pixels (V), so everything is compared on those pixels.

Reported for each of P, S, D against O:
    MAE        mean absolute difference of risk
    corr       Pearson correlation of risk
    prec/rec/F1  detection of HIGH-risk pixels (oracle risk >= --tau)
    decision   how often the most urgent horizontal region (paper Eq. 8) is the same as the oracle's
Also split by distance (true LiDAR distance) and by city.

Caveats: "ground-truth segmentation" = your pseudo-labels; LiDAR depth is sparse and noisy at object edges.
This measures perception error inside the formula. It does NOT say whether the formula itself captures real danger.

Run with venv311 active, from "Testing 3":
    python eval_risk.py                    # validation split
    python eval_risk.py --limit 50         # quick check
    python eval_risk.py --split test --final   # ONLY at the very end
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
# risk.py = the fusion (risk formula and region decision).
import risk as rk
# Reuse the depth model loader/runner and the segmentation loader/predictor.
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, predict_with_retry

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# P = what the system outputs; S and D swap in ground truth for one input at a time,
# so the error can be blamed on depth (S) or segmentation (D).
VARIANTS = {"P": "system (pred seg + pred depth)", "S": "GT seg + pred depth (depth error only)",
            "D": "pred seg + LiDAR depth (segmentation error only)"}
# Running sums kept per group: count, sums and squares (for MAE and correlation), and TP/FP/FN of high-risk pixels.
FIELDS = ["n", "sx", "sy", "sxx", "syy", "sxy", "sabs", "tp", "fp", "fn"]


# Compares one risk map (x) with the oracle (o) over many pixels. Sums are pooled, so every pixel counts equally.
class PairMeter:
    """Pooled comparison of a variant X against the oracle O over many pixels."""

    def __init__(self):
        self.acc = {}

    def add(self, key, x, o, tau):
        a = self.acc.setdefault(key, np.zeros(len(FIELDS), dtype=np.float64))
        x, o = x.astype(np.float64), o.astype(np.float64)
        # High-risk = risk at or above tau. Count agreements and disagreements for precision, recall and F1.
        hx, ho = x >= tau, o >= tau
        a += [len(x), x.sum(), o.sum(), (x * x).sum(), (o * o).sum(), (x * o).sum(), np.abs(x - o).sum(),
              (hx & ho).sum(), (hx & ~ho).sum(), (~hx & ho).sum()]

    def results(self):
        out = {}
        for k, a in self.acc.items():
            n, sx, sy, sxx, syy, sxy, sabs, tp, fp, fn = a
            if n == 0:
                continue
            # Pearson correlation computed from the running sums.
            den = np.sqrt(max(n * sxx - sx ** 2, 0) * max(n * syy - sy ** 2, 0))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out[k] = {"pixels": int(n), "high_pixels": int(tp + fn), "MAE": sabs / n, "corr": float((n * sxy - sx * sy) / den) if den > 0 else float("nan"),
                      "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12)}
        return out


# Choose the most urgent region using only pixels that have a LiDAR distance.
# Uses the base paper's MEAN rule with region weights alpha (0.4 / 0.7 / 1.0 by default), the same for oracle and system.
def masked_decision(risk_map, valid, n_regions, alphas):
    """Paper Eq. 7/8 using only the pixels where LiDAR gives a distance."""
    alphas = rk.default_alphas(n_regions) if alphas is None else np.asarray(alphas, dtype=np.float64)
    bands = np.array_split(np.arange(risk_map.shape[0]), n_regions)
    u = np.array([alphas[i] * (risk_map[b][valid[b]].mean() if valid[b].any() else 0.0) for i, b in enumerate(bands)])
    return int(u.argmax())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    # Default segmentation model = our fine-tuned SegFormer-B2 (falls back to the hub model if it is missing).
    best = HERE / "runs" / "seg_b2_native" / "best"
    ap.add_argument("--seg_model", default=str(best) if best.exists() else "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--depth_model", default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--lam_metric", type=float, default=rk.DEFAULT_LAMBDA_METRIC)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--tau", type=float, default=0.3, help="risk level counted as HIGH (a choice, vary it)")
    ap.add_argument("--taus", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3, 0.5],
                    help="thresholds for the sweep table (the main tau is added automatically)")
    ap.add_argument("--n_regions", type=int, default=rk.DEFAULT_REGIONS)
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--background", type=float, default=rk.BACKGROUND_WEIGHT)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--depth_gt", type=Path, default=HERE / "depth_gt" / "depth" / "CAM_FRONT")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    # Class weight per class id (people 1.0, two-wheelers 0.9, vehicles 0.8, everything else 0.1).
    w_vec = rk.class_weight_vector(names, json.load(open(args.weights)) if args.weights else None, args.background)
    meta = json.load(open(args.meta_cache)) if args.meta_cache.exists() else None
    device = ek.get_device()
    print(f"Split {args.split} ({len(stems)} images) | tau {args.tau} | lambda {args.lam_metric}/m | device {device}")

    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    # Risk map for a given segmentation and depth (metric mode, lambda 0.05 per meter by default).
    def risk(seg, depth):
        return rk.compute_risk(np.minimum(seg, len(names) - 1), depth, w_vec, "metric", lam_metric=args.lam_metric,
                               max_depth=args.max_depth)

    # taus = thresholds for 'high risk'; one meter per variant and threshold.
    taus = sorted(set(args.taus) | {args.tau})
    meters = {v: {t: PairMeter() for t in taus} for v in VARIANTS}
    agree = {v: {"all": [0, 0]} for v in VARIANTS}
    # Counts of which region was most urgent (row 0 = oracle, row 1 = system) and the confusion matrix between them.
    zone_counts = np.zeros((2, args.n_regions), dtype=np.int64)   # row 0 oracle, row 1 system
    confusion = np.zeros((args.n_regions, args.n_regions), dtype=np.int64)   # rows oracle zone, cols system zone
    skipped = 0
    for stem in tqdm(stems, desc="risk evaluation"):
        img_p, d_p, m_p = args.images / f"{stem}.jpg", args.depth_gt / f"{stem}.png", args.seg_dir / "masks" / f"{stem}.png"
        if not (img_p.exists() and d_p.exists() and m_p.exists()):
            skipped += 1
            continue
        img = Image.open(img_p).convert("RGB")
        # Ground truth for this frame: LiDAR depth (meters x 256) and the label mask.
        gt_depth = np.array(Image.open(d_p)).astype(np.float32) / 256.0
        gt_seg = np.array(Image.open(m_p))
        K = None
        if meta is not None and f"{stem}.jpg" in meta["sd"]:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        # The two model predictions: segmentation and depth (with the real camera intrinsics K).
        pred_seg = predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height)
        pred_depth = run_depth(depth_model, np.array(img), K)
        if not (pred_depth.shape == gt_depth.shape == pred_seg.shape == gt_seg.shape):
            skipped += 1
            continue

        # Everything is compared only where LiDAR gives a distance.
        valid = (gt_depth > 0) & (gt_depth <= args.max_depth)
        if not valid.any():
            continue
        # The four risk maps: O (all ground truth), P (all predicted), S (GT seg + predicted depth), D (predicted seg + LiDAR depth).
        maps = {"O": risk(gt_seg, gt_depth), "P": risk(pred_seg, pred_depth),
                "S": risk(gt_seg, pred_depth), "D": risk(pred_seg, gt_depth)}
        o = maps["O"][valid]
        g = gt_depth[valid]
        city = ek.city_tag(stem)
        # Oracle decision, then each variant's decision; agreement with the oracle is counted overall and per city.
        dec_o = masked_decision(maps["O"], valid, args.n_regions, args.alphas)
        zone_counts[0, dec_o] += 1
        for v in VARIANTS:
            x = maps[v][valid]
            for t in taus:
                meters[v][t].add("all", x, o, t)
                meters[v][t].add(city, x, o, t)
                for lo, hi in ek.DIST_BINS:
                    m = (g > lo) & (g <= hi)
                    if m.any():
                        meters[v][t].add(f"{lo}-{hi} m", x[m], o[m], t)
            dec = masked_decision(maps[v], valid, args.n_regions, args.alphas)
            if v == "P":
                zone_counts[1, dec] += 1
                confusion[dec_o, dec] += 1
            for key in ("all", city):
                a = agree[v].setdefault(key, [0, 0])
                a[0] += int(dec == dec_o)
                a[1] += 1

    # Final numbers per variant, group and threshold; then printed as tables.
    sweep = {v: {t: meters[v][t].results() for t in taus} for v in VARIANTS}
    res = {v: sweep[v][args.tau] for v in VARIANTS}
    if not res["P"].get("all"):
        sys.exit("No frames evaluated.")
    for key in ["all"] + [f"{lo}-{hi} m" for lo, hi in ek.DIST_BINS] + sorted(k for k in res["P"] if k not in
              ["all"] + [f"{lo}-{hi} m" for lo, hi in ek.DIST_BINS]):
        print(f"\n=== {key} ===")
        print(f"{'variant':<9s}{'pixels':>10s}{'MAE':>8s}{'corr':>8s}{'prec':>7s}{'rec':>7s}{'F1':>7s}{'decision':>10s}   meaning")
        for v in VARIANTS:
            r = res[v].get(key)
            if r:
                a = agree[v].get(key)
                d = f"{100 * a[0] / a[1]:.0f}%" if a and a[1] else "-"
                print(f"{v:<9s}{r['pixels']:>10d}{r['MAE']:>8.3f}{r['corr']:>8.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
                      f"{r['F1']:>7.3f}{d:>10s}   {VARIANTS[v]}")
    zn = rk.region_names(args.n_regions)
    print("\nMost urgent region, oracle:  " + ", ".join(f"{n} {c}" for n, c in zip(zn, zone_counts[0])))
    print("Most urgent region, system:  " + ", ".join(f"{n} {c}" for n, c in zip(zn, zone_counts[1])))
    # Decision check: how often the system picks the same region as the oracle, versus the lazy baseline of always answering the most common region.
    n_frames = int(zone_counts[0].sum())
    base_zone = int(zone_counts[0].argmax())
    base = zone_counts[0].max() / max(n_frames, 1)
    sys_acc = np.trace(confusion) / max(n_frames, 1)
    print(f"\nDecision check: system agrees with oracle on {100 * sys_acc:.1f}% of frames. "
          f"Trivial baseline 'always {zn[base_zone]}' gets {100 * base:.1f}%. Gain over baseline: {100 * (sys_acc - base):+.1f} points.")
    print("Confusion (rows = oracle zone, columns = system zone):")
    print("          " + "".join(f"{n:>8s}" for n in zn))
    for i, n in enumerate(zn):
        print(f"{n:<10s}" + "".join(f"{c:>8d}" for c in confusion[i]))

    print("\nThreshold sweep (system P vs oracle). 'pos' = oracle pixels at or above the threshold; "
          "F1 with few positives is not reliable.")
    for key in ["all"] + [f"{lo}-{hi} m" for lo, hi in ek.DIST_BINS]:
        print(f"\n--- {key} ---")
        print(f"{'tau':>6s}{'pos':>10s}{'prec':>8s}{'rec':>8s}{'F1':>8s}")
        for t in taus:
            r = sweep["P"][t].get(key)
            if r:
                print(f"{t:>6.2f}{r['high_pixels']:>10d}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['F1']:>8.3f}")
    if skipped:
        print(f"Skipped {skipped} images")

    for v in VARIANTS:
        for key, a in agree[v].items():
            if key in res[v]:
                res[v][key]["decision_agreement"] = a[0] / max(a[1], 1)
    # Save everything (settings, metrics, confusion matrix, sweep) to results/risk_*.json.
    ek.save_json(args.out_dir / f"risk_{ek.model_tag(args.depth_model)}_{args.split}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "variants": VARIANTS, "results": res,
                  "oracle_zone_counts": zone_counts[0], "system_zone_counts": zone_counts[1],
                  "zone_confusion": confusion, "baseline_always_most_common_zone": float(base),
                  "sweep": {v: {str(t): sweep[v][t] for t in taus} for v in VARIANTS}})


if __name__ == "__main__":
    main()
