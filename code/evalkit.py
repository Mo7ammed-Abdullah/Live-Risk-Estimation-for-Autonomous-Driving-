"""
Shared helpers for eval_depth.py and eval_seg.py (keep this file in "Testing 3" next to them).

 - load_split():    reads splits/<val|test>.txt  (test is locked unless you pass --final)
 - DepthMeter:      AbsRel, SqRel, MAE, RMSE, RMSE-log, SILog, d1/d2/d3, overall + by distance + by city
 - SegMeter:        pixel accuracy, mean class accuracy, per-class IoU, mIoU, risk-class mIoU, by city
 - Timer:           latency (mean / std / p95) and FPS, ignoring the first warm-up calls
 - printing + JSON saving helpers
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# RISK_CLASSES = the classes that carry risk (people, vehicles, two-wheelers); city_of() maps a drive to Boston or Singapore.
from make_splits import RISK_CLASSES, city_of

# Project root (one level above code/); all data paths are built from it.
HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Distance ranges (meters) used to report depth and risk errors by range. Range is judged by the true LiDAR depth.
DIST_BINS = [(0, 20), (20, 50), (50, 80)]
# Label value meaning 'no label here'; those pixels are skipped in every segmentation metric.
IGNORE_INDEX = 255
# classes that share a risk weight in practice: mix-ups inside a group do not change the risk
RISK_GROUPS = {"vehicle": ["car", "truck", "bus"], "vru": ["person", "rider", "bicycle", "motorcycle"]}


# ----------------------------------------------------------------------------
# Data / environment helpers
# ----------------------------------------------------------------------------
# Read the list of frame names for 'val' or 'test'.
# The test set is locked: it only loads when --final is given, so it is not used for tuning by accident.
def load_split(split, final, every=1, limit=0, split_dir=HERE / "splits"):
    if split == "test" and not final:
        sys.exit("The TEST split is reserved for your final numbers.\n"
                 "Use --split val while developing. Add --final only when everything is frozen.")
    f = split_dir / f"{split}.txt"
    if not f.exists():
        sys.exit(f"ERROR: {f} not found. Run make_splits.py first.")
    stems = [s.strip() for s in f.read_text().splitlines() if s.strip()]
    # Optional sub-sampling (--every) and limit (--limit) for quick tests.
    stems = stems[:: max(every, 1)]
    return stems[:limit] if limit else stems


# Frame names look like <drive>__<frame>; the drive tells us the city.
def city_tag(stem):
    return city_of(stem.split("__")[0])


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_gpu_gb():
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# Short, file-name-safe label of a model, used in result file names.
def model_tag(name):
    """Short file-name-safe label for a model id or a local path (works for hub ids and Windows paths)."""
    p = Path(str(name).replace("\\", "/"))
    return f"{p.parent.name}_{p.stem}" if (p.suffix == ".pt" or p.name == "best") else p.name


# Write results to JSON; numpy numbers/arrays are converted to plain Python first.
def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    print(f"Saved results to {path}")


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
# Collects per-frame times. The first few calls (warm-up, GPU start-up) are ignored in the summary.
class Timer:
    def __init__(self, warmup=3):
        self.warmup, self.t = warmup, []

    def add(self, seconds):
        self.t.append(seconds)

    def summary(self):
        t = np.array(self.t[self.warmup:] if len(self.t) > self.warmup else self.t) * 1000.0
        if len(t) == 0:
            return {}
        return {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "p95_ms": float(np.percentile(t, 95)),
                "fps": float(1000.0 / t.mean()), "n_timed": int(len(t))}


# ----------------------------------------------------------------------------
# Depth metrics (pooled over all valid LiDAR pixels)
# ----------------------------------------------------------------------------
# Accumulates depth-error sums over many frames, so metrics are pooled over all valid pixels
# (not averaged per image). Only pixels with a LiDAR depth are scored.
class DepthMeter:
    # Order of the running sums kept for each group (all / city / distance range).
    FIELDS = ["n", "abs_rel", "sq_rel", "abs", "se", "se_log", "log", "log2", "d1", "d2", "d3"]

    def __init__(self, max_depth=80.0):
        self.max_depth = max_depth
        self.acc = defaultdict(lambda: np.zeros(len(self.FIELDS), dtype=np.float64))
        self.images = defaultdict(int)

    # Add one set of pixels (p = predicted, g = true depth, both in meters) to the running sums for group `key`.
    def _add(self, key, p, g):
        diff = p - g
        ld = np.log(p) - np.log(g)
        # Threshold ratio used for d1/d2/d3: max(pred/true, true/pred) below 1.25, 1.25^2, 1.25^3.
        ratio = np.maximum(p / g, g / p)
        self.acc[key] += np.array([
            len(g), np.sum(np.abs(diff) / g), np.sum(diff ** 2 / g), np.sum(np.abs(diff)), np.sum(diff ** 2),
            np.sum(ld ** 2), np.sum(ld), np.sum(ld ** 2),
            np.sum(ratio < 1.25), np.sum(ratio < 1.25 ** 2), np.sum(ratio < 1.25 ** 3)])

    def update(self, pred, gt, tags=()):
        # Score only pixels that have a LiDAR depth within range and a finite prediction.
        valid = (gt > 0) & (gt <= self.max_depth) & np.isfinite(pred)
        if not valid.any():
            return
        p = np.maximum(pred[valid].astype(np.float64), 1e-3)
        g = gt[valid].astype(np.float64)
        self._add("all", p, g)
        self.images["all"] += 1
        for tag in tags:
            self._add(tag, p, g)
            self.images[tag] += 1
        # Also score each distance range separately (by true depth).
        for lo, hi in DIST_BINS:  # distance is judged by the TRUE (LiDAR) depth
            m = (g > lo) & (g <= hi)
            if m.any():
                self._add(f"{lo}-{hi} m", p[m], g[m])

    # Turn the running sums into the final metrics: AbsRel, SqRel, MAE, RMSE, RMSE-log, SILog, d1-d3.
    def results(self):
        out = {}
        for key, a in self.acc.items():
            n = a[0]
            if n == 0:
                continue
            mean_log = a[6] / n
            out[key] = {"pixels": int(n), "images": self.images.get(key, None),
                        "AbsRel": a[1] / n, "SqRel": a[2] / n, "MAE": a[3] / n, "RMSE": float(np.sqrt(a[4] / n)),
                        "RMSE_log": float(np.sqrt(a[5] / n)),
                        "SILog": float(100 * np.sqrt(max(a[7] / n - mean_log ** 2, 0))),
                        "d1": a[8] / n, "d2": a[9] / n, "d3": a[10] / n}
        return out


# Print the depth results as an aligned table: 'all' first, then distance ranges, then cities.
def print_depth_table(res):
    order = ["all"] + [f"{lo}-{hi} m" for lo, hi in DIST_BINS] + sorted(k for k in res if k not in
            ["all"] + [f"{lo}-{hi} m" for lo, hi in DIST_BINS])
    print(f"\n{'group':<12s}{'pixels':>11s}{'AbsRel':>8s}{'SqRel':>8s}{'MAE':>7s}{'RMSE':>7s}{'RMSElog':>9s}"
          f"{'SILog':>7s}{'d1':>7s}{'d2':>7s}{'d3':>7s}")
    for k in order:
        if k in res:
            r = res[k]
            print(f"{k:<12s}{r['pixels']:>11d}{r['AbsRel']:>8.3f}{r['SqRel']:>8.3f}{r['MAE']:>7.2f}{r['RMSE']:>7.2f}"
                  f"{r['RMSE_log']:>9.3f}{r['SILog']:>7.2f}{r['d1']:>7.3f}{r['d2']:>7.3f}{r['d3']:>7.3f}")


# ----------------------------------------------------------------------------
# Segmentation metrics (confusion matrix based)
# ----------------------------------------------------------------------------
# Accumulates a confusion matrix (true class x predicted class) per group; IoU, accuracy etc. come from it.
class SegMeter:
    def __init__(self, n_classes, names):
        self.n, self.names = n_classes, list(names)
        self.conf = defaultdict(lambda: np.zeros((n_classes, n_classes), dtype=np.int64))

    def update(self, pred, gt, tags=()):
        # Ignore unlabeled pixels, then count each (true, predicted) pair with one bincount.
        valid = (gt != IGNORE_INDEX) & (gt < self.n) & (pred < self.n)
        idx = gt[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        cm = np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)
        self.conf["all"] += cm
        for tag in tags:
            self.conf[tag] += cm

    # Metrics from a confusion matrix: pixel accuracy, mean class accuracy, IoU per class, mIoU,
    # frequency-weighted IoU, risk-class mIoU, and grouped IoU for vehicles and people.
    def scores(self, cm):
        tp = np.diag(cm).astype(np.float64)
        gt_px, pr_px = cm.sum(1).astype(np.float64), cm.sum(0).astype(np.float64)
        union = gt_px + pr_px - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            # IoU = TP / (TP + FP + FN). Classes that never appear in the ground truth are left out (NaN).
            iou = np.where(gt_px > 0, tp / union, np.nan)      # classes absent from the GT are left out
            cacc = np.where(gt_px > 0, tp / gt_px, np.nan)
        total = cm.sum()
        group_iou = {}
        # Grouped IoU: a car mistaken for a truck is not counted as an error, since both carry the same risk weight.
        for g, members in RISK_GROUPS.items():
            idx = [self.names.index(c) for c in members if c in self.names]
            if idx and gt_px[idx].sum() > 0:
                g_tp = cm[np.ix_(idx, idx)].sum()
                group_iou[g] = float(g_tp / (gt_px[idx].sum() + pr_px[idx].sum() - g_tp))
        risk_idx = [self.names.index(c) for c in RISK_CLASSES if c in self.names]
        return {"pixel_acc": float(tp.sum() / total) if total else float("nan"),
                "mean_class_acc": float(np.nanmean(cacc)), "mIoU": float(np.nanmean(iou)),
                "fwIoU": float(np.nansum(gt_px / total * np.nan_to_num(iou))) if total else float("nan"),
                "risk_mIoU": float(np.nanmean(iou[risk_idx])) if risk_idx else float("nan"),
                "group_iou": group_iou, "iou": iou, "class_acc": cacc, "gt_fraction": gt_px / max(total, 1)}

    def results(self):
        return {k: self.scores(cm) for k, cm in self.conf.items()}


# Print the segmentation results: overall scores, grouped IoU, then one line per class.
def print_seg_report(res, names):
    print(f"\n{'group':<12s}{'pixAcc':>8s}{'mAcc':>8s}{'mIoU':>8s}{'fwIoU':>8s}{'riskmIoU':>10s}")
    for k in ["all"] + sorted(k for k in res if k != "all"):
        r = res[k]
        print(f"{k:<12s}{r['pixel_acc']:>8.3f}{r['mean_class_acc']:>8.3f}{r['mIoU']:>8.3f}{r['fwIoU']:>8.3f}{r['risk_mIoU']:>10.3f}")
    r = res["all"]
    if r.get("group_iou"):
        print("\nGrouped IoU (confusions inside a group do not count as errors): " +
              "  ".join(f"{g} {v:.3f}" for g, v in r["group_iou"].items()) +
              "   [vehicle = car+truck+bus, vru = person+rider+bicycle+motorcycle]")
    print(f"\n{'class':<15s}{'share of GT':>12s}{'IoU':>8s}{'acc':>8s}")
    for i, n in enumerate(names):
        mark = "  <- risk class" if n in RISK_CLASSES else ""
        print(f"{n:<15s}{100 * r['gt_fraction'][i]:>11.2f}%{r['iou'][i]:>8.3f}{r['class_acc'][i]:>8.3f}{mark}")
