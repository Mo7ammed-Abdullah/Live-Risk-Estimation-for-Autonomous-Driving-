"""
How good is the AUTOMATIC time-to-collision (ttc_tracker.py: finds and follows objects by itself, no nuScenes boxes)?

Runs the two models over every frame of a split in time order, feeds the tracker, and compares each real object's
estimated TTC with the true TTC from box_gt/motion.json. The nuScenes boxes are used ONLY to score: a real object
counts as "seen" when a tracked object of the same kind overlaps its box (IoU >= --match_iou).

Reported:
    seen          share of real objects that the tracker found
    TTC accuracy  for objects truly approaching with TTC <= 10 s (unseen or no-TTC objects count as misses)
    danger        AP / AUROC / precision / recall for "true TTC < 4 s" (unseen objects get score 0)
    false alarms  tracked objects with TTC < 4 s that overlap no real object's box
and, on the same objects, the box-based numbers from estimate_ttc.py (cache/ttc_meas_<split>.json) for comparison.

Run (venv311 active, project folder):
    python code/eval_auto_ttc.py --split val
    python code/eval_auto_ttc.py --split test --final     # once, at the end
"""
import argparse
import json
import sys
from pathlib import Path

import warnings

import numpy as np

import evalkit as ek

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty subsets just print as "-"
from estimate_ttc import GROUP, average_precision, auroc, estimates, ts_of
from ttc_tracker import KINDS, TTCTracker, box_iou

HERE = Path(__file__).resolve().parent.parent
KIND_OF = {c: k for k, cs in KINDS.items() for c in cs}
KIND_OF["rider"] = "two-wheeler"          # annotated riders sit on two-wheelers; the tracker may call them person
OK_KINDS = {"person": {"person"}, "vehicle": {"vehicle"}, "two-wheeler": {"two-wheeler", "person"}}


def metrics(true_ttc, est_ttc, danger_s=4.0, max_ttc=10.0):
    if len(true_ttc) == 0:
        return {k: np.nan for k in ("TTC found (approaching)", "median TTC error (s)", "within 1 s", "within 20%", "AP (TTC<4s)",
                                    "AUROC (TTC<4s)", "precision (TTC<4s)", "recall (TTC<4s)")} | {"n": 0}
    true_ttc = np.array([t if t is not None else np.inf for t in true_ttc])
    est = np.array([t if t is not None else np.inf for t in est_ttc])
    y = (true_ttc < danger_s).astype(float)
    sel = np.isfinite(true_ttc) & (true_ttc <= max_ttc)
    err = np.abs(est[sel] - true_ttc[sel])
    found = np.isfinite(est[sel])
    score = np.where(np.isfinite(est), 1.0 / np.maximum(est, 1e-3), 0.0)
    flag = est < danger_s
    return {"n": len(est), "TTC found (approaching)": float(found.mean()), "median TTC error (s)": float(np.median(err[found])) if found.any() else np.nan,
            "within 1 s": float((err <= 1).mean()), "within 20%": float((err <= 0.2 * true_ttc[sel]).mean()),
            "AP (TTC<4s)": average_precision(y, score), "AUROC (TTC<4s)": auroc(y, score),
            "precision (TTC<4s)": float(y[flag].mean()) if flag.any() else np.nan, "recall (TTC<4s)": float(flag[y == 1].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--max_gap", type=float, default=0.7)
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt_native")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The TEST split is reserved for your final numbers. Add --final only when everything is frozen.")

    from PIL import Image
    from tqdm import tqdm
    from eval_depth import load_depth_model, run_depth
    from generate_seg_gt import load_labeler, predict_with_retry

    stems = sorted(ek.load_split(args.split, args.final))
    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    motion = json.load(open(HERE / "box_gt" / "motion.json"))
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    meta = json.load(open(args.meta_cache))
    device = ek.get_device()
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)
    tracker = TTCTracker(names)

    rows, false_alarms, tracked_ttc_lt4, prev_stem = [], 0, 0, None
    for stem in tqdm(stems, desc=f"auto TTC {args.split}"):
        img_p = args.images / f"{stem}.jpg"
        if not img_p.exists():
            continue
        new_drive = (prev_stem is None or prev_stem.split("__")[0] != stem.split("__")[0]
                     or not 0 < ts_of(stem) - ts_of(prev_stem) <= args.max_gap)
        if new_drive:
            tracker.reset()
        img = Image.open(img_p).convert("RGB")
        sd = meta["sd"].get(f"{stem}.jpg")
        K = meta["cs"][sd["calibrated_sensor_token"]]["camera_intrinsic"] if sd else None
        seg = np.minimum(predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height), len(names) - 1)
        depth = np.nan_to_num(run_depth(depth_model, np.array(img), K), nan=80.0, posinf=80.0)
        objs = tracker.update(seg, depth, ts_of(stem))
        prev_ok, prev_stem = not new_drive, stem
        if not prev_ok:
            continue                                   # first frame of a drive: no history, like estimate_ttc.py
        truth = {r["inst"]: r for r in motion.get(stem, [])}
        gts = [o for o in boxes.get(stem, []) if o["cls"] in GROUP and o["inst"] in truth and truth[o["inst"]]["has_motion"]
               and o["vis"] >= args.min_vis and o["zmin"] <= 80 and (o["box"][3] - o["box"][1]) >= args.min_box_px]
        # Greedy match real objects <-> tracked objects (compatible kinds, box IoU).
        pairs = sorted(((box_iou(g["box"], t["box"]), gi, ti) for gi, g in enumerate(gts) for ti, t in enumerate(objs)
                        if t["kind"] in OK_KINDS[KIND_OF[g["cls"]]]), reverse=True)
        used_g, used_t, match = set(), set(), {}
        for iou, gi, ti in pairs:
            if iou < args.match_iou:
                break
            if gi in used_g or ti in used_t:
                continue
            used_g.add(gi); used_t.add(ti); match[gi] = ti
        for gi, g in enumerate(gts):
            t = objs[match[gi]] if gi in match else None
            rows.append({"stem": stem, "inst": g["inst"], "cls": g["cls"], "z_true": g["zmin"], "ttc_true": truth[g["inst"]]["ttc_z"],
                         "seen": t is not None, "ttc_est": None if t is None else t["ttc"]})
        for ti, t in enumerate(objs):
            if t["ttc"] is not None and t["ttc"] < 4:
                tracked_ttc_lt4 += 1
                if ti not in used_t and not any(box_iou(t["box"], g["box"]) > 0.1 for g in boxes.get(stem, [])):
                    false_alarms += 1

    out = {"automatic": metrics([r["ttc_true"] for r in rows], [r["ttc_est"] for r in rows])}
    out["automatic"]["seen"] = float(np.mean([r["seen"] for r in rows]))
    out["automatic"]["false alarms among tracked TTC<4s"] = false_alarms / max(tracked_ttc_lt4, 1)
    # Same objects, box-based method (estimate_ttc.py), when its measurements exist.
    meas = HERE / "cache" / f"ttc_meas_{args.split}.json"
    if meas.exists():
        bb = {(r["stem"], r["inst"]): r for r in json.load(open(meas))}
        common = [r for r in rows if (r["stem"], r["inst"]) in bb]
        out["automatic (same objects as box-based)"] = metrics([r["ttc_true"] for r in common], [r["ttc_est"] for r in common])
        out["box-based (estimate_ttc.py)"] = metrics([bb[(r["stem"], r["inst"])]["ttc_true"] for r in common],
                                                     [estimates(bb[(r["stem"], r["inst"])], 10, 0.5)["combined"][1] for r in common])
    cols = ["n", "TTC found (approaching)", "median TTC error (s)", "within 1 s", "within 20%", "AP (TTC<4s)", "AUROC (TTC<4s)",
            "precision (TTC<4s)", "recall (TTC<4s)"]
    short = ["objects", "found", "med err s", "<=1 s", "<=20%", "AP", "AUROC", "prec", "recall"]
    print(f"\n=== {args.split}: automatic TTC (objects found and followed by the system itself) ===")
    print(f"{'method':40s}" + "".join(f"{c:>10s}" for c in short))
    for k, r in out.items():
        print(f"{k:40s}" + "".join(f"{r[c]:10.0f}" if c == "n" else (f"{r[c]:10.3f}" if r[c] == r[c] else f"{'-':>10s}") for c in cols))
    a = out["automatic"]
    print(f"real objects seen by the tracker: {a['seen']:.1%} | tracked objects with TTC < 4 s that match no real object: "
          f"{a['false alarms among tracked TTC<4s']:.1%}")
    for label, sel in [("0-20 m", lambda r: r["z_true"] < 20), ("20-50 m", lambda r: 20 <= r["z_true"] < 50), ("50-80 m", lambda r: r["z_true"] >= 50)]:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 10:
            continue
        m = metrics([r["ttc_true"] for r in sub], [r["ttc_est"] for r in sub])
        print(f"  {label:8s} n={len(sub):5d}  seen {np.mean([r['seen'] for r in sub]):.2f}  median err {m['median TTC error (s)']:.2f} s"
              f"  within 1 s {m['within 1 s']:.2f}  AUROC {m['AUROC (TTC<4s)']:.3f}")
    res = HERE / "results" / f"ttc_auto_{args.split}.json"
    json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": out}, open(res, "w"), indent=1)
    print(f"Saved {res}")


if __name__ == "__main__":
    main()
