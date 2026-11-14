"""
Learned baselines for danger ranking, on exactly the same objects and truth as estimate_ttc.py.

Question for the paper: does our training-free camera TTC beat a model that is TRAINED to predict danger?

Task (per real object, nuScenes boxes give where the object is, as for box-based TTC):
    danger = true time-to-collision < 4 s   (box_gt/motion.json)
Learned models (trained on the train split, settings chosen on validation, test once):
    learned, one frame   class, measured depth (percentiles), segmented size, image position and size of the box,
                         fixed-formula risk  -> what a static, single-image risk model can learn
    learned, two frames  the same + the previous frame's depth and size, closing speed, growth rate and our two TTC
                         estimates as inputs  -> an upper reference for "learned fusion" with motion
Compared with (no training): fixed risk formula, distance only, and our box-based TTC (combined).
Models: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier) and logistic regression; the variant and
its settings are picked by validation AP only.

Inputs: cache/ttc_meas_<split>.json for train / val / test (made by estimate_ttc.py), box_gt/boxes.json.
Run (any Python with numpy + scikit-learn, no GPU):
    python code/learned_risk_baseline.py                 # train on train, report validation
    python code/learned_risk_baseline.py --final         # also report the locked test split (once)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from estimate_ttc import auroc, average_precision, estimates

HERE = Path(__file__).resolve().parent.parent
W_IMG, H_IMG = 1600.0, 900.0
KIND = {"car": 0, "truck": 0, "bus": 0, "train": 0, "person": 1, "rider": 2, "bicycle": 2, "motorcycle": 2}
P, MIN_CLOSING = 10, 0.5                          # the settings used for the reported box-based TTC


def load(split, boxes):
    f = HERE / "cache" / f"ttc_meas_{split}.json"
    if not f.exists():
        sys.exit(f"{f} not found. Run: python code/estimate_ttc.py --split {split}" + (" --final" if split == "test" else ""))
    rows = json.load(open(f))
    box_of = {(s, o["inst"]): o["box"] for s, objs in boxes.items() for o in objs}
    X1, X2, y, base = [], [], [], {"fixed formula": [], "distance only": [], "box-based TTC (ours)": []}
    for r in rows:
        bx = box_of.get((r["stem"], r["inst"]))
        if bx is None:
            continue
        now, prev = r["now"], r["prev"]
        k = np.eye(3)[KIND.get(r["cls"], 0)]
        d10, d50 = now["depth"]["10"], now["depth"]["50"]
        x0, y0, x1, y1 = bx
        static = [*k, np.log1p(d10), np.log1p(d50), d50 - d10, np.log1p(now["area"]), float(now["fallback"]),
                  (x0 + x1) / 2 / W_IMG - 0.5, y1 / H_IMG, (x1 - x0) / W_IMG, (y1 - y0) / H_IMG, now["fixed"]]
        e = estimates(r, P, MIN_CLOSING)
        closing_d = e["depth"][0]
        growth = np.sqrt(now["area"] / max(prev["area"], 1.0))
        inv = lambda t: 0.0 if t is None else 1.0 / max(t, 1e-3)
        motion = [np.log1p(prev["depth"]["10"]), closing_d, np.log(max(growth, 1e-3)) / r["dt"], r["dt"],
                  inv(e["depth"][1]), inv(e["scale"][1]), inv(e["combined"][1]), float(prev["fallback"])]
        X1.append(static)
        X2.append(static + motion)
        y.append(float(r["ttc_true"] is not None and r["ttc_true"] < 4.0))
        base["fixed formula"].append(now["fixed"])
        base["distance only"].append(now["dist_only"])
        base["box-based TTC (ours)"].append(inv(e["combined"][1]))
    return np.array(X1), np.array(X2), np.array(y), {k: np.array(v) for k, v in base.items()}


def candidates(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for lr, leaves, l2 in itertools.product([0.03, 0.1], [7, 15, 31], [0.0, 1.0]):
        out.append((f"trees lr={lr} leaves={leaves} l2={l2}",
                    lambda lr=lr, leaves=leaves, l2=l2: HistGradientBoostingClassifier(
                        learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2, max_iter=300,
                        early_stopping=False, random_state=seed)))
    for c in (0.1, 1.0, 10.0):
        out.append((f"logistic C={c}", lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))))
    return out


def scores(y, s):
    return {"AP (TTC<4s)": average_precision(y, s), "AUROC (TTC<4s)": auroc(y, s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="also evaluate the locked test split (do this once)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    tr1, tr2, ytr, _ = load("train", boxes)
    va1, va2, yva, bva = load("val", boxes)
    print(f"objects: train {len(ytr)} ({int(ytr.sum())} dangerous) | val {len(yva)} ({int(yva.sum())} dangerous)")

    chosen, report = {}, {"val": {}, "test": {}}
    for name, Xtr, Xva in (("learned, one frame", tr1, va1), ("learned, two frames", tr2, va2)):
        best = None
        for label, make in candidates(args.seed):
            m = make().fit(Xtr, ytr)
            ap_val = average_precision(yva, m.predict_proba(Xva)[:, 1])
            if best is None or ap_val > best[0]:
                best = (ap_val, label, make)
        chosen[name] = best
        print(f"{name:22s} chosen on validation: {best[1]} (val AP {best[0]:.3f})")

    def table(split, X1, X2, y, base):
        res = {k: scores(y, v) for k, v in base.items()}
        for name, Xtr, X in (("learned, one frame", tr1, X1), ("learned, two frames", tr2, X2)):
            m = chosen[name][2]().fit(Xtr, ytr)
            res[name] = scores(y, m.predict_proba(X)[:, 1])
        print(f"\n=== {split}: danger ranking (true TTC < 4 s), {len(y)} objects, {int(y.sum())} dangerous ===")
        print(f"{'method':26s}{'AP':>8s}{'AUROC':>8s}   training")
        for k in ("fixed formula", "distance only", "learned, one frame", "box-based TTC (ours)", "learned, two frames"):
            tr = {"learned, one frame": "trained", "learned, two frames": "trained"}.get(k, "none")
            print(f"{k:26s}{res[k]['AP (TTC<4s)']:8.3f}{res[k]['AUROC (TTC<4s)']:8.3f}   {tr}")
        return res

    report["val"] = table("validation", va1, va2, yva, bva)
    if args.final:
        te1, te2, yte, bte = load("test", boxes)
        report["test"] = table("TEST (locked)", te1, te2, yte, bte)
    report["chosen"] = {k: v[1] for k, v in chosen.items()}
    out = HERE / "results" / ("learned_baseline_test.json" if args.final else "learned_baseline_val.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
