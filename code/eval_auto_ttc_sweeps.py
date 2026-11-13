"""
Automatic time-to-collision at the camera's full rate: the same tracker and scoring as eval_auto_ttc.py, but the
tracker also sees the CAM_FRONT *sweeps* (the ~12 Hz frames nuScenes records between the 2 Hz keyframes).

Why: at 2 Hz (0.5 s between frames) most objects could not be linked from one frame to the next (diag_auto_ttc.py:
25% of dangerous objects found but not linked). At ~12 Hz objects move little between frames, so they should stay linked,
and each 0.6 s window holds ~7 measurements instead of 2.

Scoring is unchanged and happens ONLY at the keyframes of the split (where box_gt/motion.json has the true TTC):
same matching (box IoU >= --match_iou), same metrics, same danger threshold (true TTC < 4 s).
Sweeps are used only as extra input frames; frames from scenes outside the split are skipped.

Needs the sweeps in dataset/sweeps/CAM_FRONT (from "Camera blobs only for part 1" of nuScenes trainval).

Run (venv311 active, project folder):
    python code/eval_auto_ttc_sweeps.py --split val                 # all sweeps (~12 Hz)
    python code/eval_auto_ttc_sweeps.py --split val --every 2       # every 2nd sweep (~6 Hz), optional check
    python code/eval_auto_ttc_sweeps.py --split test --final        # once, at the end, with the settings chosen on val
Output: results/ttc_auto_sweeps_<split>[_every<N>].json, and cache/ttc_objs_sweeps_<split>.json (objects per frame,
used by replay_auto_ttc.py to try tracker settings in seconds without re-running the models)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import evalkit as ek
from eval_auto_ttc import KIND_OF, OK_KINDS, metrics
from estimate_ttc import GROUP, ts_of
from ttc_tracker import TTCTracker, box_iou

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent.parent


def build_sequence(key_stems, sweep_dir, max_gap, every):
    """Ordered list of (stem, path, is_key) per contiguous run of split keyframes, with the sweeps that fall inside each run."""
    by_log = {}
    for s in key_stems:
        by_log.setdefault(s.split("__")[0], []).append(s)
    sweeps_by_log = {}
    for p in sweep_dir.glob("*.jpg"):
        sweeps_by_log.setdefault(p.stem.split("__")[0], []).append(p)
    runs = []
    for log, keys in by_log.items():
        keys = sorted(keys, key=ts_of)
        run = [keys[0]]
        for a, b in zip(keys, keys[1:]):
            if 0 < ts_of(b) - ts_of(a) <= max_gap:
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        runs.append(run)
    seqs, n_sweeps = [], 0
    for run in runs:
        log = run[0].split("__")[0]
        t0, t1 = ts_of(run[0]), ts_of(run[-1])
        sw = sorted((p for p in sweeps_by_log.get(log, []) if t0 < ts_of(p.stem) < t1), key=lambda p: ts_of(p.stem))
        sw = sw[:: max(every, 1)]
        n_sweeps += len(sw)
        frames = [(s, None, True) for s in run] + [(p.stem, p, False) for p in sw]
        seqs.append(sorted(frames, key=lambda f: ts_of(f[0])))
    return seqs, n_sweeps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="use every N-th sweep (1 = all, ~12 Hz)")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--max_gap", type=float, default=0.7, help="keyframes further apart than this start a new sequence")
    ap.add_argument("--window", type=float, default=0.6, help="tracker history window in seconds (as at 2 Hz)")
    ap.add_argument("--min_span", type=float, default=0.3)
    ap.add_argument("--seg_model", default=str(HERE / "runs" / "seg_b2_native" / "best"))
    ap.add_argument("--depth_model", default=str(HERE / "runs" / "depth_s" / "best.pt"))
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt_native")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--sweeps", type=Path, default=HERE / "dataset" / "sweeps" / "CAM_FRONT")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The TEST split is reserved for your final numbers. Add --final only when everything is frozen.")
    if not args.sweeps.is_dir() or not any(args.sweeps.glob("*.jpg")):
        sys.exit(f"No sweeps found in {args.sweeps}. Extract sweeps/CAM_FRONT from the nuScenes camera blobs first.")

    from PIL import Image
    from tqdm import tqdm
    from eval_depth import load_depth_model, run_depth
    from generate_seg_gt import load_labeler, predict_with_retry

    key_stems = [s for s in ek.load_split(args.split, args.final) if (args.images / f"{s}.jpg").exists()]
    seqs, n_sweeps = build_sequence(key_stems, args.sweeps, args.max_gap, args.every)
    n_frames = sum(len(s) for s in seqs)
    print(f"{args.split}: {len(key_stems)} keyframes + {n_sweeps} sweeps = {n_frames} frames in {len(seqs)} sequences")

    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    motion = json.load(open(HERE / "box_gt" / "motion.json"))
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    meta = json.load(open(args.meta_cache))
    # Camera intrinsics per drive (the front camera does not change within a drive; sweeps share the keyframes' calibration).
    K_of_log = {}
    for s in key_stems:
        sd = meta["sd"].get(f"{s}.jpg")
        if sd and s.split("__")[0] not in K_of_log:
            K_of_log[s.split("__")[0]] = meta["cs"][sd["calibrated_sensor_token"]]["camera_intrinsic"]

    device = ek.get_device()
    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)
    tracker = TTCTracker(names, window=args.window, min_span=args.min_span)

    rows, false_alarms, tracked_ttc_lt4 = [], 0, 0
    bar = tqdm(total=n_frames, desc=f"auto TTC with sweeps ({args.split})")
    obj_cache = []
    for seq in seqs:
        tracker.reset()
        cache_seq = []
        obj_cache.append(cache_seq)
        first_key_done = False
        for stem, path, is_key in seq:
            bar.update(1)
            img = Image.open(path if path is not None else args.images / f"{stem}.jpg").convert("RGB")
            K = K_of_log.get(stem.split("__")[0])
            seg = np.minimum(predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height), len(names) - 1)
            depth = np.nan_to_num(run_depth(depth_model, np.array(img), K), nan=80.0, posinf=80.0)
            # Save the objects found in this frame so replay_auto_ttc.py can re-run the tracker without the models.
            cache_seq.append({"stem": stem, "key": is_key, "t": ts_of(stem), "objs": tracker._objects(seg, depth)})
            objs = tracker.update(seg, depth, ts_of(stem))
            if not is_key:
                continue
            if not first_key_done:                     # first keyframe of a sequence is skipped, as at 2 Hz
                first_key_done = True
                continue
            truth = {r["inst"]: r for r in motion.get(stem, [])}
            gts = [o for o in boxes.get(stem, []) if o["cls"] in GROUP and o["inst"] in truth and truth[o["inst"]]["has_motion"]
                   and o["vis"] >= args.min_vis and o["zmin"] <= 80 and (o["box"][3] - o["box"][1]) >= args.min_box_px]
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
                             "seen": t is not None, "ttc_est": None if t is None else t["ttc"],
                             "history_s": None if t is None else t["history_s"]})
            for ti, t in enumerate(objs):
                if t["ttc"] is not None and t["ttc"] < 4:
                    tracked_ttc_lt4 += 1
                    if ti not in used_t and not any(box_iou(t["box"], g["box"]) > 0.1 for g in boxes.get(stem, [])):
                        false_alarms += 1
    bar.close()
    tag0 = "" if args.every == 1 else f"_every{args.every}"
    cache_f = HERE / "cache" / f"ttc_objs_sweeps_{args.split}{tag0}.json"
    cache_f.parent.mkdir(exist_ok=True)
    json.dump({"names": names, "sequences": obj_cache}, open(cache_f, "w"))
    print(f"Saved per-frame objects for fast replays: {cache_f}")

    a = metrics([r["ttc_true"] for r in rows], [r["ttc_est"] for r in rows])
    a["seen"] = float(np.mean([r["seen"] for r in rows])) if rows else float("nan")
    a["false alarms among tracked TTC<4s"] = false_alarms / max(tracked_ttc_lt4, 1)
    danger = [r for r in rows if r["ttc_true"] is not None and r["ttc_true"] < 4]
    a["dangerous objects with a TTC"] = float(np.mean([r["ttc_est"] is not None for r in danger])) if danger else float("nan")
    a["n dangerous"] = len(danger)
    out = {"automatic, with sweeps": a}
    prev = HERE / "results" / f"ttc_auto_{args.split}.json"
    if prev.exists():
        out["automatic, keyframes only (eval_auto_ttc.py)"] = json.load(open(prev))["results"]["automatic"]

    cols = ["n", "TTC found (approaching)", "median TTC error (s)", "within 1 s", "AP (TTC<4s)", "AUROC (TTC<4s)",
            "precision (TTC<4s)", "recall (TTC<4s)", "seen"]
    short = ["objects", "found", "med err s", "<=1 s", "AP", "AUROC", "prec", "recall", "seen"]
    print(f"\n=== {args.split}: automatic TTC, keyframes + sweeps (every {args.every}) vs keyframes only ===")
    print(f"{'method':45s}" + "".join(f"{c:>10s}" for c in short))
    for k, r in out.items():
        print(f"{k:45s}" + "".join(f"{r.get(c, float('nan')):10.0f}" if c == "n" else
                                   (f"{r.get(c, float('nan')):10.3f}" if r.get(c, float("nan")) == r.get(c, float("nan")) else f"{'-':>10s}")
                                   for c in cols))
    print(f"dangerous objects (true TTC < 4 s): {a['n dangerous']}, of which with an automatic TTC: {a['dangerous objects with a TTC']:.1%}")
    for label, sel in [("0-20 m", lambda r: r["z_true"] < 20), ("20-50 m", lambda r: 20 <= r["z_true"] < 50), ("50-80 m", lambda r: r["z_true"] >= 50)]:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 10:
            continue
        m = metrics([r["ttc_true"] for r in sub], [r["ttc_est"] for r in sub])
        print(f"  {label:8s} n={len(sub):5d}  seen {np.mean([r['seen'] for r in sub]):.2f}  median err {m['median TTC error (s)']:.2f} s"
              f"  within 1 s {m['within 1 s']:.2f}  AUROC {m['AUROC (TTC<4s)']:.3f}")
    tag = "" if args.every == 1 else f"_every{args.every}"
    res = HERE / "results" / f"ttc_auto_sweeps_{args.split}{tag}.json"
    json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "frames": n_frames, "sweeps": n_sweeps,
               "results": out, "rows": rows}, open(res, "w"), indent=1)
    print(f"Saved {res}")


if __name__ == "__main__":
    main()
