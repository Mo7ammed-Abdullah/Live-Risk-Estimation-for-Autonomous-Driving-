"""
Replay the automatic TTC tracker on saved per-frame objects (cache/ttc_objs_sweeps_<split>.json, written by
eval_auto_ttc_sweeps.py), so tracker settings can be compared in seconds without running the two networks again.

Scoring is identical to eval_auto_ttc_sweeps.py (keyframes only, same matching and metrics).

Examples (no GPU needed):
    python code/replay_auto_ttc.py --split val                          # default settings, all sweeps (~12 Hz)
    python code/replay_auto_ttc.py --split val --rate key               # same tracker, keyframes only (2 Hz baseline)
    python code/replay_auto_ttc.py --split val --grid                   # compare window / rate settings on validation
    python code/replay_auto_ttc.py --split test --final --window 0.6    # once, with the setting chosen on validation
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from eval_auto_ttc import KIND_OF, OK_KINDS, metrics
from estimate_ttc import GROUP
from ttc_tracker import TTCTracker, box_iou

HERE = Path(__file__).resolve().parent.parent


class ReplayTracker(TTCTracker):
    """Same tracker; the objects of each frame come from the cache instead of the segmentation and depth maps."""

    def update_objs(self, objs, t):
        self._next = [dict(o, box=tuple(o["box"])) for o in objs]
        return self.update(None, None, t)

    def _objects(self, seg, depth):
        return self._next


def run(cache, boxes, motion, window, min_span, rate, max_age=0.7, predict=True, min_vis=2, min_box_px=8, match_iou=0.3):
    tracker = ReplayTracker(cache["names"], window=window, min_span=min_span, max_age=max_age, predict=predict)
    rows = []
    for seq in cache["sequences"]:
        tracker.reset()
        if rate == "key":
            frames = [f for f in seq if f["key"]]
        elif rate == "all":
            frames = seq
        else:                                          # keep every N-th sweep plus all keyframes
            n, sweeps_seen, frames = int(rate), 0, []
            for f in seq:
                if f["key"]:
                    frames.append(f)
                else:
                    if sweeps_seen % n == 0:
                        frames.append(f)
                    sweeps_seen += 1
        first_key_done = False
        for f in frames:
            objs = tracker.update_objs(f["objs"], f["t"])
            if not f["key"]:
                continue
            if not first_key_done:
                first_key_done = True
                continue
            stem = f["stem"]
            truth = {r["inst"]: r for r in motion.get(stem, [])}
            gts = [o for o in boxes.get(stem, []) if o["cls"] in GROUP and o["inst"] in truth and truth[o["inst"]]["has_motion"]
                   and o["vis"] >= min_vis and o["zmin"] <= 80 and (o["box"][3] - o["box"][1]) >= min_box_px]
            pairs = sorted(((box_iou(g["box"], t["box"]), gi, ti) for gi, g in enumerate(gts) for ti, t in enumerate(objs)
                            if t["kind"] in OK_KINDS[KIND_OF[g["cls"]]]), reverse=True)
            used_g, used_t, match = set(), set(), {}
            for iou, gi, ti in pairs:
                if iou < match_iou:
                    break
                if gi in used_g or ti in used_t:
                    continue
                used_g.add(gi); used_t.add(ti); match[gi] = ti
            for gi, g in enumerate(gts):
                t = objs[match[gi]] if gi in match else None
                rows.append({"z_true": g["zmin"], "ttc_true": truth[g["inst"]]["ttc_z"], "seen": t is not None,
                             "ttc_est": None if t is None else t["ttc"]})
    m = metrics([r["ttc_true"] for r in rows], [r["ttc_est"] for r in rows])
    danger = [r for r in rows if r["ttc_true"] is not None and r["ttc_true"] < 4]
    m["dangerous with TTC"] = float(np.mean([r["ttc_est"] is not None for r in danger])) if danger else float("nan")
    m["seen"] = float(np.mean([r["seen"] for r in rows])) if rows else float("nan")
    return m, rows


COLS = [("n", "objects"), ("TTC found (approaching)", "found"), ("dangerous with TTC", "danger w/TTC"), ("median TTC error (s)", "med err s"),
        ("within 1 s", "<=1 s"), ("AP (TTC<4s)", "AP"), ("AUROC (TTC<4s)", "AUROC"), ("precision (TTC<4s)", "prec"), ("recall (TTC<4s)", "recall")]


def fmt(m):
    return "".join(f"{m[c]:>13.0f}" if c == "n" else (f"{m[c]:>13.3f}" if m[c] == m[c] else f"{'-':>13s}") for c, _ in COLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--window", type=float, default=0.6)
    ap.add_argument("--min_span", type=float, default=0.3)
    ap.add_argument("--rate", default="all", help="all (~12 Hz), key (2 Hz keyframes only), or N = every N-th sweep")
    ap.add_argument("--grid", action="store_true", help="compare several settings (validation only)")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The TEST split is reserved for your final numbers. Add --final only when everything is frozen.")
    if args.split == "test" and args.grid:
        sys.exit("Choose settings on validation; --grid is not allowed on test.")
    cache = json.load(open(HERE / "cache" / f"ttc_objs_sweeps_{args.split}.json"))
    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    motion = json.load(open(HERE / "box_gt" / "motion.json"))

    print(f"{'setting':34s}" + "".join(f"{h:>13s}" for _, h in COLS))
    if args.grid:
        out = {}
        settings = [("key", 0.6, 0.3)] + [(r, w, s) for r, w, s in itertools.product(["all", "2", "3"], [0.4, 0.6, 0.8, 1.0, 1.5], [0.3])]
        for rate, w, s in settings:
            m, _ = run(cache, boxes, motion, w, s, rate)
            label = f"rate={rate:3s} window={w:.1f}s min_span={s:.1f}s"
            out[label] = m
            print(f"{label:34s}" + fmt(m))
        res = HERE / "results" / f"ttc_auto_replay_grid_{args.split}.json"
    else:
        m, _ = run(cache, boxes, motion, args.window, args.min_span, args.rate)
        label = f"rate={args.rate:3s} window={args.window:.1f}s min_span={args.min_span:.1f}s"
        print(f"{label:34s}" + fmt(m))
        out = {label: m}
        res = HERE / "results" / f"ttc_auto_replay_{args.split}_{args.rate}_w{args.window}.json"
    json.dump(out, open(res, "w"), indent=1)
    print(f"Saved {res}")


if __name__ == "__main__":
    main()
