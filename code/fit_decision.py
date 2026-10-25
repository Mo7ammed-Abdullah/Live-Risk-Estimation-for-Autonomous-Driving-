"""
Which decision rule picks the truly most urgent region?   (CPU only, uses cache/frames_<split>.npz)

TARGET (fixed, uses the paper formula with the default weights and lambda, no region weights):
    the region that contains the real object with the highest true risk  R = w_class * exp(-lambda * distance)
    (an object belongs to the region where its bottom edge is).

SYSTEM RULE (what we vary), for region i:
    U_i = alpha_i * ( mean over the region's pixels of R^p )^(1/p)
    p = 1       -> the paper's rule (plain mean, Eq. 7)
    p -> large  -> approaches "highest risk in the region"
    alpha_i rises linearly from a_min (top region) to 1 (bottom region); a_min = 1 means all regions count the same.
The decision is argmax U_i (paper Eq. 8).

Why only the rule is fitted: the target is built with the same class weights and lambda, so fitting those against it would
be circular. They need an independent target (lane position, stopping distance) - a later step.

Honesty: the validation set is ONE drive. The fit is judged with block cross-validation (4 contiguous time blocks: choose the
rule on 3 blocks, test on the 4th). Neighbouring frames are similar, so treat it as an optimistic estimate; the real
test is the held-out test split at the very end (cache_frames.py --split test --final, then --split test --final here).

Run:  python fit_decision.py          # after python cache_frames.py
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

import risk as rk

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Candidate rules to try: p = 1 is the plain mean (base paper), larger p approaches the maximum, infinity = maximum.
P_GRID = [1, 2, 4, 8, 16, 32, np.inf]
# Candidate values of a_min, the weight of the top (far) region relative to the bottom (near) one.
A_GRID = [1.0, 0.8, 0.6, 0.4, 0.2]


# Load the cache written by cache_frames.py (per-frame class x depth histograms plus the real objects).
def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# Sort key: (drive, timestamp), so frames can be put in driving order.
def stem_key(s):
    log, _, ts = str(s).split("__")[:3]
    return log, int(ts.split(".")[0].split("_")[0])


# Merge the 6 cached strips into the wanted number of regions (2, 3 or 6).
def region_counts(counts, n_regions):
    strips = np.array_split(np.arange(counts.shape[1]), n_regions)
    return np.stack([counts[:, s[0]:s[-1] + 1].sum(axis=1) for s in strips], axis=1)   # (F, R, C, K)


# Decision of a rule for every frame, computed from the histograms:
# risk of a (class, depth bin) cell = class weight x exp(-lambda x depth); a region's score is its max (p = infinity) or power mean (p);
# then multiply by alpha and take the best region.
def decisions(rc, w, lam, bin_m, p, alphas):
    """rc: (F, R, C, K) pixel counts. Returns the argmax region per frame."""
    d = (np.arange(rc.shape[-1]) + 0.5) * bin_m
    r = w[:, None] * np.exp(-lam * d)[None, :]                    # (C, K) risk of a (class, depth bin) cell
    n = rc.sum(axis=(2, 3)).astype(np.float64)                    # (F, R)
    if np.isinf(p):
        agg = np.where(rc > 0, r[None, None], 0.0).max(axis=(2, 3))
    else:
        agg = ((rc * (r ** p)[None, None]).sum(axis=(2, 3)) / np.maximum(n, 1)) ** (1.0 / p)
    return (agg * alphas[None, :]).argmax(axis=1)


# The truth: the region that holds the real object with the highest true risk (judged by the bottom edge of its box).
def target_regions(data, w, lam, H, n_regions):
    """Region holding the object with highest true risk, per frame (-1 if the frame has no object)."""
    F = len(data["stems"])
    best_r, best_reg = np.full(F, -1.0), np.full(F, -1, dtype=int)
    edges = np.linspace(0, H, n_regions + 1)
    r_true = w[data["obj_cls"]] * np.exp(-lam * data["obj_zmin"])
    reg = np.minimum(np.searchsorted(edges, data["obj_bottom"] - 1, side="right") - 1, n_regions - 1)
    for f, rt, rg in zip(data["obj_frame"], r_true, reg):
        if rt > best_r[f]:
            best_r[f], best_reg[f] = rt, rg
    return best_reg


# Accuracy averaged over the true classes, so a rule that always says 'near' gets no free credit.
def balanced_acc(y, yhat, n):
    rec = [np.mean(yhat[y == k] == k) for k in range(n) if (y == k).any()]
    return float(np.mean(rec)) if rec else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--n_regions", type=int, default=3, choices=[2, 3, 6])
    ap.add_argument("--lam", type=float, default=rk.DEFAULT_LAMBDA_METRIC)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--fit_from", type=Path, default=None,
                    help="cache of a DIFFERENT split to choose the rule on (use with --split test: fit on val, test on test)")
    args = ap.parse_args()

    path = args.cache or HERE / "cache" / f"frames_{args.split}.npz"
    data = load(path)
    names = [str(n) for n in data["class_names"]]
    w = rk.class_weight_vector(names)
    H, bin_m, R = int(data["H"]), float(data["bin_m"]), args.n_regions
    order = np.array(sorted(range(len(data["stems"])), key=lambda i: stem_key(data["stems"][i])))
    rc = region_counts(data["counts"], R)
    tgt = target_regions(data, w, args.lam, H, R)
    use = np.where(tgt >= 0)[0]
    zn = rk.region_names(R)
    print(f"{path.name}: {len(use)} frames with a real risk object (of {len(order)}) | {R} regions | lambda {args.lam}")
    print("true most-urgent region: " + ", ".join(f"{n} {int((tgt[use] == i).sum())}" for i, n in enumerate(zn)))
    # Baseline: always answering the most common true region.
    base_zone = int(np.bincount(tgt[use], minlength=R).argmax())
    print(f"baseline 'always {zn[base_zone]}': {100 * np.mean(tgt[use] == base_zone):.1f}%\n")

    def alpha_vec(a_min):
        return np.linspace(a_min, 1.0, R) if R > 1 else np.array([1.0])

    # Score every combination of rule (p) and region weight (a_min) on the whole split.
    grid = list(itertools.product(P_GRID, A_GRID))
    acc = {}
    for p, a in grid:
        dec = decisions(rc[use], w, args.lam, bin_m, p, alpha_vec(a))
        acc[(p, a)] = (float(np.mean(dec == tgt[use])), balanced_acc(tgt[use], dec, R))

    # The base paper's rule (plain mean) with this project's default alphas, for comparison.
    paper = (1, 0.4)    # paper rule with this project's default alphas (linspace 0.4..1)
    dec = decisions(rc[use], w, args.lam, bin_m, 1, rk.default_alphas(R))
    paper_acc = (float(np.mean(dec == tgt[use])), balanced_acc(tgt[use], dec, R))
    print("In-sample (whole split) accuracy / balanced accuracy of every rule  [paper rule = p 1, a_min 0.4]")
    print(f"{'p':>6s}" + "".join(f"{'a_min ' + str(a):>16s}" for a in A_GRID))
    for p in P_GRID:
        print(f"{p:>6}" + "".join(f"{100 * acc[(p, a)][0]:>9.1f}/{100 * acc[(p, a)][1]:<5.1f}" for a in A_GRID))
    print(f"\npaper rule: accuracy {100 * paper_acc[0]:.1f}%, balanced {100 * paper_acc[1]:.1f}%")

    # --- honest estimate: block cross-validation, or fit on another split ----------------------------------------
    # Pick the best (p, a_min) on a set of frames (ties go to the larger a_min, i.e. the simpler rule).
    def best_rule(idx):
        sc = {(p, a): np.mean(decisions(rc[idx], w, args.lam, bin_m, p, alpha_vec(a)) == tgt[idx]) for p, a in grid}
        return max(sc, key=lambda k: (sc[k], -k[1]))

    # Option: choose the rule on a DIFFERENT split (e.g. fit on val) and apply it to this one (e.g. test).
    if args.fit_from is not None:
        fit = load(args.fit_from)
        rc_f, tgt_f = region_counts(fit["counts"], R), target_regions(fit, w, args.lam, int(fit["H"]), R)
        uf = np.where(tgt_f >= 0)[0]
        sc = {(p, a): np.mean(decisions(rc_f[uf], w, args.lam, bin_m, p, alpha_vec(a)) == tgt_f[uf]) for p, a in grid}
        rule = max(sc, key=lambda k: (sc[k], -k[1]))
        dec = decisions(rc[use], w, args.lam, bin_m, rule[0], alpha_vec(rule[1]))
        print(f"\nRule chosen on {args.fit_from.name}: p={rule[0]}, a_min={rule[1]}")
        print(f"Applied to {path.name}: accuracy {100 * np.mean(dec == tgt[use]):.1f}%, "
              f"balanced {100 * balanced_acc(tgt[use], dec, R):.1f}%  (paper rule {100 * paper_acc[0]:.1f}%)")
        return

    # Honest estimate on one split: cut the drive into time blocks, choose the rule on the other blocks, score it on the held-out block.
    pos = {f: i for i, f in enumerate(order)}
    blocks = np.array_split(order, args.blocks)
    in_block = np.zeros(len(order), dtype=int)
    for b, idxs in enumerate(blocks):
        in_block[idxs] = b
    # Hit counters for the fitted rule, the paper rule and the plain max rule.
    hit = {"cv": 0, "paper": 0, "max": 0, "n": 0}
    print(f"\nBlock cross-validation ({args.blocks} time blocks): rule chosen on the other blocks, scored on the held-out block")
    print(f"{'block':>6s}{'frames':>8s}{'chosen rule':>18s}{'held-out acc':>14s}{'paper rule':>12s}{'max rule':>10s}")
    for b in range(args.blocks):
        test = np.array([i for i in use if in_block[i] == b])
        train = np.array([i for i in use if in_block[i] != b])
        if len(test) == 0:
            continue
        p, a = best_rule(train)
        d_cv = decisions(rc[test], w, args.lam, bin_m, p, alpha_vec(a))
        d_pa = decisions(rc[test], w, args.lam, bin_m, 1, rk.default_alphas(R))
        d_mx = decisions(rc[test], w, args.lam, bin_m, np.inf, alpha_vec(1.0))
        for k, d in (("cv", d_cv), ("paper", d_pa), ("max", d_mx)):
            hit[k] += int((d == tgt[test]).sum())
        hit["n"] += len(test)
        print(f"{b + 1:>6d}{len(test):>8d}{'p=' + str(p) + ', a=' + str(a):>18s}{100 * np.mean(d_cv == tgt[test]):>13.1f}%"
              f"{100 * np.mean(d_pa == tgt[test]):>11.1f}%{100 * np.mean(d_mx == tgt[test]):>9.1f}%")
    n = max(hit["n"], 1)
    print(f"\nOverall held-out: fitted rule {100 * hit['cv'] / n:.1f}% | paper rule {100 * hit['paper'] / n:.1f}% | "
          f"plain max {100 * hit['max'] / n:.1f}% | always '{zn[base_zone]}' {100 * np.mean(tgt[use] == base_zone):.1f}%")
    p, a = best_rule(use)
    print(f"\nBest rule on the whole split: p={p}, a_min={a}  -> in-sample accuracy {100 * acc[(p, a)][0]:.1f}%")


if __name__ == "__main__":
    main()
