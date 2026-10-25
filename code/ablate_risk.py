"""
How much do our ASSUMED choices matter?  Ablations + sensitivity of the decision, scored against the 3D boxes.

The models (segmentation + depth) run ONCE per frame. For every setting below we only re-do the cheap risk formula
on the same predictions, so adding settings costs almost nothing.

Reference truth (never changes, so every setting is scored against the same thing):
    for every real object   R_true = w_class * exp(-0.05 * distance_m)   (default class weights)
    truly most urgent region = argmax_i alpha_i * R_true of the objects whose box bottom lies in region i
    alphas = 0.4 / 0.7 / 1.0 (the project default, the same as eval_box_risk.py)
Each setting is scored by:
    acc   = how often its decision equals the reference truth      (with a 95 % bootstrap interval over frames)
    same  = how often its decision equals the decision of the reference setting (decision stability)

Groups:
    rule        mean vs max per region, three alpha choices
    function    exponential (paper) vs linear vs inverse depth attenuation
    lambda      0.02 / 0.05 / 0.1 / 0.25 per meter
    weights     other class-weight choices
    component   semantic only (no depth) / depth only (no class) / combined   (the base paper's Table IV idea)

NOTE: the truth uses the reference weights and lambda, so settings close to the reference score higher by construction.
Read "acc" as "how much does the decision change if the assumption changes", NOT as proof that one setting is right.

Run with venv311 active, from "Testing 3":
    python ablate_risk.py                     # validation split (explore here)
    python ablate_risk.py --limit 50          # quick check
    python ablate_risk.py --split test --final   # once, at the end; nothing is selected from it
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek
import risk as rk
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, predict_with_retry

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Region weights: default (0.4 / 0.7 / 1.0), fitted-on-validation (0.6 / 0.8 / 1.0) and flat (all equal).
A_DEFAULT = np.array([0.4, 0.7, 1.0])
A_FITTED = np.array([0.6, 0.8, 1.0])
A_FLAT = np.array([1.0, 1.0, 1.0])
# Reference class weights from risk.py; the truth is always built with these.
DEFAULT_W = dict(rk.DEFAULT_CLASS_WEIGHTS)


# The list of settings to compare. Each entry changes one thing against the reference:
# attenuation function (fn), lambda, class weights, background weight, rule (p: infinity = max per region, 1 = mean) and region weights.
def build_configs():
    cfgs = []

    # Helper that adds one setting; defaults equal the reference setting.
    def add(name, group, fn="exp", lam=0.05, w=None, bg=0.1, p=np.inf, alphas=A_DEFAULT, use_w=True, use_d=True):
        cfgs.append(dict(name=name, group=group, fn=fn, lam=lam, w=DEFAULT_W if w is None else w, bg=bg, p=p,
                         alphas=alphas, use_w=use_w, use_d=use_d))

    # The reference setting every other one is compared to.
    add("REFERENCE: exp, lambda 0.05, default weights, max, alpha 0.4-1", "reference")
    add("mean per region, alpha 0.4-1 (base paper rule)", "rule", p=1.0)
    add("mean per region, alpha flat", "rule", p=1.0, alphas=A_FLAT)
    add("max per region, alpha 0.6-1 (fitted on val; used in demo)", "rule", alphas=A_FITTED)
    add("max per region, alpha flat", "rule", alphas=A_FLAT)
    add("linear attenuation 1 - d/80", "function", fn="lin")
    add("inverse attenuation 1/d", "function", fn="inv")
    # Lambda sweep: how fast risk falls with distance.
    for lam in (0.02, 0.1, 0.25):
        add(f"exponential, lambda {lam}", "lambda", lam=lam)
    # Classes treated as risky in the 'all weights 1.0' setting.
    risky = ["person", "rider", "bicycle", "motorcycle", "car", "truck", "bus", "train"]
    add("all risk classes weight 1.0", "weights", w={c: 1.0 for c in risky})
    add("people 1.0, vehicles 0.5, two-wheelers 0.7", "weights",
        w=dict(person=1.0, rider=1.0, bicycle=0.7, motorcycle=0.7, car=0.5, truck=0.5, bus=0.5, train=0.5))
    add("two-wheelers 1.0", "weights", w={**DEFAULT_W, "bicycle": 1.0, "motorcycle": 1.0})
    add("background weight 0 (objects only)", "weights", bg=0.0)
    add("background weight 0.3", "weights", bg=0.3)
    # Component test: class weights without depth, and depth without class weights.
    add("semantic only (class weight, no depth)", "component", fn="none")
    add("depth only (no class weight)", "component", use_w=False)
    return cfgs


# Class weight per class id (classes not in the table get the background weight bg).
def weight_vector(names, w, bg):
    return np.array([w.get(n, bg) for n in names], dtype=np.float32)


# How risk falls with distance d (meters): exponential (reference), linear, inverse, or none (semantic only).
def depth_term(fn, d, lam, max_depth):
    if fn == "exp":
        return np.exp(-lam * d)
    if fn == "lin":
        return np.clip(1.0 - d / max_depth, 0.0, 1.0)
    if fn == "inv":
        return 1.0 / np.maximum(d, 1.0)
    return np.ones_like(d)


# Decision of one setting for one frame: urgency of each band = alpha x (max or mean of risk in the band); pick the highest.
def decide(rmap, bands, p, alphas):
    u = []
    for a, (r0, r1) in zip(alphas, bands):
        x = rmap[r0:r1]
        u.append(a * (float(x.max()) if np.isinf(p) else float(x.mean())))
    return int(np.argmax(u))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    best = HERE / "runs" / "seg_b2_native" / "best"
    ap.add_argument("--seg_model", default=str(best))
    ap.add_argument("--depth_model", default="runs\\depth_s\\best.pt")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--n_regions", type=int, default=3)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    boxes = json.load(open(args.boxes))
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    meta = json.load(open(args.meta_cache)) if args.meta_cache.exists() else None
    device = ek.get_device()
    cfgs = build_configs()
    for c in cfgs:
        c["wvec"] = weight_vector(names, c["w"], c["bg"])
    # Reference weights used to build the truth.
    ref_w = weight_vector(names, DEFAULT_W, 0.1)
    n = args.n_regions
    print(f"Split {args.split} ({len(stems)} images) | {len(cfgs)} settings | device {device}")

    processor, seg_model = load_labeler(args.seg_model, device)
    depth_model = load_depth_model(args.depth_model, device)

    # truth = true most urgent region per frame; decs = each setting's decision per frame.
    truth = []
    decs = {c["name"]: [] for c in cfgs}
    for stem in tqdm(stems, desc="ablation"):
        img_p = args.images / f"{stem}.jpg"
        if not img_p.exists() or stem not in boxes:
            continue
        img = Image.open(img_p).convert("RGB")
        K = None
        if meta is not None and f"{stem}.jpg" in meta["sd"]:
            K = meta["cs"][meta["sd"][f"{stem}.jpg"]["calibrated_sensor_token"]]["camera_intrinsic"]
        # The models run ONCE per frame; every setting then only re-does the cheap risk formula on the same predictions.
        seg = predict_with_retry(processor, seg_model, img, device, device.type == "cuda", img.height)
        depth = run_depth(depth_model, np.array(img), K)
        if seg.shape != depth.shape:
            continue
        H, W = seg.shape
        seg = np.minimum(seg, len(names) - 1)

        # Truth: each real object's reference risk, assigned to the region where its box bottom lies; the best weighted object decides.
        edges = np.linspace(0, H, n + 1)
        objs = []
        for o in boxes[stem]:
            if not o["cls"] or o["vis"] < args.min_vis or o["zmin"] > args.max_depth:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in o["box"]]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if y1 - y0 < args.min_box_px or x1 - x0 < 2:
                continue
            w = float(ref_w[names.index(o["cls"])]) if o["cls"] in names else 0.1
            r_true = w * float(np.exp(-0.05 * min(o["zmin"], args.max_depth)))
            reg = min(int(np.searchsorted(edges, y1 - 1, "right") - 1), n - 1)
            objs.append((A_DEFAULT[reg] * r_true, reg))
        if not objs:
            continue
        truth.append(max(objs)[1])

        # Clean the predicted depth (replace invalid values, limit to 0-80 m).
        d = np.clip(np.where(np.isfinite(depth), depth, args.max_depth), 0, args.max_depth).astype(np.float32)
        bands = [(int(b[0]), int(b[-1]) + 1) for b in np.array_split(np.arange(H), n)]
        # Cache of the depth term, so settings with the same function and lambda share one computation.
        cache = {}
        for c in cfgs:
            key = (c["fn"], c["lam"])
            if key not in cache:
                cache[key] = depth_term(c["fn"], d, c["lam"], args.max_depth).astype(np.float32)
            dt = cache[key]
            w = c["wvec"][seg] if c["use_w"] else None
            # Risk map = class weight x depth term (or just one of them for the component tests).
            rmap = (dt if w is None else w * dt) if c["use_d"] else w
            decs[c["name"]].append(decide(rmap, bands, c["p"], c["alphas"]))

    if not truth:
        sys.exit("No frames evaluated.")
    truth = np.array(truth)
    ref_dec = np.array(decs[cfgs[0]["name"]])
    # Baseline: always answering the most common true region.
    base = float(np.mean(truth == np.bincount(truth, minlength=n).argmax()))
    # Bootstrap: 1000 resamples of the frames give a 95% interval for each accuracy.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(truth), (1000, len(truth)))
    print(f"\n{len(truth)} frames with real objects. Always-most-common-region baseline: {100 * base:.1f}%")
    print(f"{'setting':<62s}{'acc':>7s}{'95% interval':>16s}{'same as ref':>13s}")
    out, last = [], None
    for c in cfgs:
        dec = np.array(decs[c["name"]])
        eq = (dec == truth).astype(np.float64)
        accs = eq[idx].mean(axis=1)
        lo, hi = np.percentile(accs, [2.5, 97.5])
        # 'Same as ref' = how often the setting makes the same decision as the reference (stability).
        same = float(np.mean(dec == ref_dec))
        if c["group"] != last:
            print(f"--- {c['group']}")
            last = c["group"]
        print(f"{c['name']:<62s}{100 * eq.mean():>6.1f}%{f'{100 * lo:.1f}-{100 * hi:.1f}':>16s}{100 * same:>12.1f}%")
        out.append(dict(name=c["name"], group=c["group"], acc=float(eq.mean()), ci=[float(lo), float(hi)], same_as_reference=same))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ek.save_json(args.out_dir / f"ablation_{args.split}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "frames": int(len(truth)),
                  "baseline_always_most_common": base, "results": out})
    print(f"\nSaved results to {args.out_dir / f'ablation_{args.split}.json'}")


if __name__ == "__main__":
    main()
