"""
Why does the automatic TTC lose objects, and which tracker fix helps?  CPU only, uses cache/segdepth_<split>_*.npy
(made by cache_segdepth.py). Development on VALIDATION only.

For every real object (same filters as eval_auto_ttc.py) it records what happened:
    seg miss         segmentation found (almost) none of its pixels
    merged           its pixels belong to a blob much bigger than the object (touching objects joined)
    not found        other reasons it was not matched
    found, no link   found now, but the tracker did not connect it to the previous frame -> no history, no TTC
                     (split into: it WAS found in the previous frame = matching failure / it was not)
    found, linked    has history; TTC exists or the object measured as not approaching

Variants (same TTC measurement, different object finding / matching):
    base        what ttc_tracker.py does now: connected blobs, match by box overlap IoU >= 0.3
    predict     match by predicted position: previous box grown by the object's expected scale change, then
                accept if centres are close relative to the box size and sizes / depths are consistent
    split       cut blobs where depth jumps (touching objects at different distances become separate objects)
    split+predict  both

Run:  python code/diag_auto_ttc.py --split val
"""
import argparse
import json
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent.parent
from estimate_ttc import GROUP, ts_of          # noqa: E402
from eval_auto_ttc import KIND_OF, OK_KINDS, metrics   # noqa: E402
from ttc_tracker import KINDS, box_iou, slope  # noqa: E402


class Tracker:
    def __init__(self, names, split=False, predict=False, min_area_frac=0.0003, iou_min=0.3, percentile=10, window=0.6,
                 min_span=0.3, min_closing=0.5, max_age=0.7, jump=0.15):
        self.ids = {k: [names.index(c) for c in v if c in names] for k, v in KINDS.items()}
        self.split, self.predict, self.jump = split, predict, jump
        self.min_area_frac, self.iou_min, self.percentile = min_area_frac, iou_min, percentile
        self.window, self.min_span, self.min_closing, self.max_age = window, min_span, min_closing, max_age
        self.tracks, self.next_id = [], 0

    def reset(self):
        self.tracks, self.next_id = [], 0

    def objects(self, seg, depth):
        import cv2
        H, W = seg.shape
        min_area = max(int(self.min_area_frac * H * W), 12)
        objs = []
        if self.split:
            # depth edges: relative depth change between neighbouring pixels above `jump` (e.g. 15 %)
            ld = np.log(np.maximum(depth.astype(np.float32), 0.5))
            gx = np.zeros_like(ld); gy = np.zeros_like(ld)
            gx[:, 1:] = np.abs(np.diff(ld, axis=1)); gy[1:, :] = np.abs(np.diff(ld, axis=0))
            edge = (np.maximum(gx, gy) > np.log1p(self.jump))
        for kind, ids in self.ids.items():
            mask = np.isin(seg, ids)
            if self.split:
                mask = mask & ~edge
            if not mask.any():
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4 if self.split else 8)
            for j in range(1, n):
                x, y, w, h, area = stats[j]
                if area < min_area:
                    continue
                d = depth[lab == j].astype(np.float32)
                objs.append({"kind": kind, "box": (float(x), float(y), float(x + w), float(y + h)), "area": float(area),
                             "depth": float(np.percentile(d, self.percentile)), "label": j})
        return objs

    def _score(self, o, tr, t):
        """Match quality between a new object and a track (higher = better, < 0 = impossible)."""
        if tr["kind"] != o["kind"]:
            return -1
        if not self.predict:
            iou = box_iou(o["box"], tr["box"])
            return iou if iou >= self.iou_min else -1
        # predicted box: grow the last box around its bottom centre by the expected scale change
        dt = t - tr["last_t"]
        g = tr.get("growth", 0.0)
        s = float(np.exp(np.clip(g * dt, -0.7, 0.7)))
        x0, y0, x1, y1 = tr["box"]
        cx, by, w, h = (x0 + x1) / 2, y1, (x1 - x0) * s, (y1 - y0) * s
        pred = (cx - w / 2, by - h, cx + w / 2, by)
        iou = box_iou(o["box"], pred)
        ow, oh = o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]
        ocx, ocy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
        pcy = by - h / 2
        dist = np.hypot(ocx - cx, ocy - pcy) / max(np.sqrt(max(w * h, 1.0)), 1.0)
        size_ratio = np.sqrt(max(ow * oh, 1.0) / max(w * h, 1.0))
        dz = abs(o["depth"] - tr["depth"]) / max(tr["depth"], 1.0)
        if dist > 0.8 or not (0.5 <= size_ratio <= 2.0) or dz > 0.5:
            return -1
        return iou + 0.5 * (1 - dist / 0.8) + 0.2 * (1 - min(dz / 0.5, 1))

    def update(self, seg, depth, t):
        objs = self.objects(seg, depth)
        self.tracks = [tr for tr in self.tracks if t - tr["last_t"] <= self.max_age]
        pairs = sorted(((self._score(o, tr, t), i, k) for i, o in enumerate(objs) for k, tr in enumerate(self.tracks)), reverse=True)
        used_o, used_t = set(), set()
        for sc, i, k in pairs:
            if sc < 0:
                break
            if i in used_o or k in used_t:
                continue
            used_o.add(i); used_t.add(k)
            objs[i]["track"] = self.tracks[k]
        for o in objs:
            tr = o.get("track")
            if tr is None:
                tr = {"id": self.next_id, "kind": o["kind"], "hist": deque()}
                self.next_id += 1
                self.tracks.append(tr)
            tr["box"], tr["last_t"], tr["depth"] = o["box"], t, o["depth"]
            tr["hist"].append((t, o["depth"], 0.5 * np.log(max(o["area"], 1.0))))
            while tr["hist"] and t - tr["hist"][0][0] > self.window + 1e-6:
                tr["hist"].popleft()
            h = list(tr["hist"])
            o["id"], o["history_s"] = tr["id"], h[-1][0] - h[0][0]
            o["ttc"] = None
            if len(h) >= 2 and o["history_s"] >= self.min_span - 1e-6:
                ts = [x[0] for x in h]
                closing = -slope(ts, [x[1] for x in h])
                growth = slope(ts, [x[2] for x in h])
                tr["growth"] = growth
                vals = [v for v in ((o["depth"] / closing) if closing > self.min_closing else None,
                                    (1.0 / growth) if growth > 1e-3 else None) if v is not None]
                o["ttc"] = float(np.mean(vals)) if vals else None
        return objs


def run(variant, args, meta, seg_mm, dep_mm, boxes, motion, names):
    tr = Tracker(names, split="split" in variant, predict="predict" in variant)
    sc = meta["scale"]
    rows, prev_stem, prev_found = [], None, set()
    for i, stem in enumerate(meta["stems"]):
        new = (prev_stem is None or prev_stem.split("__")[0] != stem.split("__")[0] or not 0 < ts_of(stem) - ts_of(prev_stem) <= 0.7)
        if new:
            tr.reset(); prev_found = set()
        seg, depth = np.asarray(seg_mm[i]), np.asarray(dep_mm[i])
        objs = tr.update(seg, depth, ts_of(stem))
        truth = {r["inst"]: r for r in motion.get(stem, [])}
        gts = [o for o in boxes.get(stem, []) if o["cls"] in GROUP and o["inst"] in truth and truth[o["inst"]]["has_motion"]
               and o["vis"] >= 2 and o["zmin"] <= 80 and (o["box"][3] - o["box"][1]) >= 8]
        gb = [tuple(v * sc for v in g["box"]) for g in gts]
        pairs = sorted(((box_iou(gb[gi], t["box"]), gi, ti) for gi in range(len(gts)) for ti, t in enumerate(objs)
                        if t["kind"] in OK_KINDS[KIND_OF[gts[gi]["cls"]]]), reverse=True)
        ug, ut, match = set(), set(), {}
        for iou, gi, ti in pairs:
            if iou < 0.3:
                break
            if gi in ug or ti in ut:
                continue
            ug.add(gi); ut.add(ti); match[gi] = ti
        found_now = set()
        for gi, g in enumerate(gts):
            if gi in match:
                found_now.add(g["inst"])
        if not new:
            for gi, g in enumerate(gts):
                x0, y0, x1, y1 = [int(round(v)) for v in gb[gi]]
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(max(x1, x0 + 1), seg.shape[1]), min(max(y1, y0 + 1), seg.shape[0])
                ids = [names.index(c) for c in GROUP[g["cls"]] if c in names]
                n_px = int(np.isin(seg[y0:y1, x0:x1], ids).sum())
                t_true = truth[g["inst"]]["ttc_z"]
                if gi in match:
                    o = objs[match[gi]]
                    if o["history_s"] < 0.3 - 1e-6:
                        status = "found, no link (was found before)" if g["inst"] in prev_found else "found, no link (not found before)"
                    else:
                        status = "found, linked, TTC" if o["ttc"] is not None else "found, linked, no TTC"
                    est = o["ttc"]
                else:
                    est = None
                    box_area = max((x1 - x0) * (y1 - y0), 1)
                    if n_px < max(10, 0.05 * box_area):
                        status = "seg miss"
                    else:
                        # the blob holding most of its pixels: is it much bigger than the object?
                        best = max((t for t in objs if t["kind"] in OK_KINDS[KIND_OF[g["cls"]]]),
                                   key=lambda t: box_iou(gb[gi], t["box"]) if True else 0, default=None)
                        big = best is not None and box_iou(gb[gi], best["box"]) > 0 and \
                            (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1]) > 2.0 * box_area
                        status = "merged" if big else "not found (other)"
                rows.append({"stem": stem, "inst": g["inst"], "z": g["zmin"], "status": status, "ttc_true": t_true, "ttc_est": est})
        prev_found, prev_stem = found_now, stem
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--variants", nargs="*", default=["base", "predict", "split", "split+predict"])
    args = ap.parse_args()
    meta = json.load(open(HERE / "cache" / f"segdepth_{args.split}.json"))
    seg_mm = np.load(HERE / "cache" / f"segdepth_{args.split}_seg.npy", mmap_mode="r")
    dep_mm = np.load(HERE / "cache" / f"segdepth_{args.split}_depth.npy", mmap_mode="r")
    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    motion = json.load(open(HERE / "box_gt" / "motion.json"))
    names = meta["names"]
    order = ["seg miss", "merged", "not found (other)", "found, no link (was found before)", "found, no link (not found before)",
             "found, linked, no TTC", "found, linked, TTC"]
    for v in args.variants:
        rows = run(v, args, meta, seg_mm, dep_mm, boxes, motion, names)
        c = Counter(r["status"] for r in rows)
        m = metrics([r["ttc_true"] for r in rows], [r["ttc_est"] for r in rows])
        print(f"\n=== {args.split} | {v}: {len(rows)} real objects ===")
        for k in order:
            print(f"  {k:36s} {c[k]:5d}  ({c[k] / len(rows):5.1%})")
        print(f"  TTC found (approaching) {m['TTC found (approaching)']:.3f} | median err {m['median TTC error (s)']:.2f} s | within 1 s "
              f"{m['within 1 s']:.3f} | AUROC {m['AUROC (TTC<4s)']:.3f} | AP {m['AP (TTC<4s)']:.3f}")


if __name__ == "__main__":
    main()
