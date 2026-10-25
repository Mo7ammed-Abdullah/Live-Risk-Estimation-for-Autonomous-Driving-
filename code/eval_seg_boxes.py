"""
Judge a segmentation model (or a label folder) by the REAL objects in the nuScenes 3D boxes - independent of any pseudo-label.

For every real person / rider / car / truck / bus / motorcycle / bicycle (visible > 40 %, box at least 8 px high, within 80 m):
    coverage = share of the object's 2D box that carries a label of the right group
               (person+rider, car+truck+bus, motorcycle+bicycle)
    found    = coverage >= 10 %   (also reported at 30 %)
Also: "stray" = share of all person / rider / vehicle label pixels that lie outside every real object's box (10 % margin).
Reported overall, by group, by distance (nearest surface of the 3D box) and by city.

Because the reference is the 3D boxes, models trained on different labels can be compared fairly.

Run (venv311 active, inside "Testing 3"):
    python eval_seg_boxes.py                                          # newest model, validation split
    python eval_seg_boxes.py --seg_model runs\\seg_b2\\best            # the model trained on the OLD labels
    python eval_seg_boxes.py --seg_model runs\\seg_b2_native\\best     # the model trained on the NEW labels
    python eval_seg_boxes.py --labels seg_gt_native                   # score the label folder itself (no model)
    python eval_seg_boxes.py --labels seg_gt                          # ... the old labels
    add --split test --final only for the very last numbers.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import evalkit as ek

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# A detected object counts as right if it has the right GROUP of labels; mix-ups inside a group (car vs truck) do not matter.
GROUPS = {"person": ["person", "rider"], "vehicle": ["car", "truck", "bus"], "twowheel": ["motorcycle", "bicycle"]}
# Reverse lookup: class name -> its group.
CLS_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seg_model", default="")
    ap.add_argument("--labels", type=Path, default=None, help="score a label folder (masks/ inside) instead of a model")
    ap.add_argument("--min_vis", type=int, default=2)
    ap.add_argument("--min_box_px", type=int, default=8)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt", help="classes.json (class names)")
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    if not args.boxes.exists():
        sys.exit("box_gt/boxes.json not found - run: python generate_box_gt.py")
    # Ground truth objects from the 3D boxes (box_gt/boxes.json), independent of any pseudo-label.
    boxes = json.load(open(args.boxes))
    stems = ek.load_split(args.split, args.final, args.every, args.limit)
    names = [c["name"] for c in json.load(open((args.labels or args.seg_dir) / "classes.json"))["classes"]]
    # Class ids of each group, so predicted pixels can be checked against them.
    ids = {c: names.index(c) for c in CLS_GROUP if c in names}
    group_ids = {g: [ids[c] for c in cs if c in ids] for g, cs in GROUPS.items()}
    all_risk_ids = [i for v in group_ids.values() for i in v]

    # Two modes: score a trained model (run it on each photo) or score a label folder directly (no model).
    predict = None
    if args.labels is None:
        from generate_seg_gt import load_labeler, predict_with_retry
        cand = [HERE / "runs" / "seg_b2_native" / "best", HERE / "runs" / "seg_b2" / "best"]
        model_id = args.seg_model or next((str(p) for p in cand if p.exists()), "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        device = ek.get_device()
        print(f"Model {model_id} | split {args.split} ({len(stems)} images) | device {device}")
        processor, model = load_labeler(model_id, device)
        predict = lambda img: np.minimum(predict_with_retry(processor, model, img, device, device.type == "cuda", img.height), len(names) - 1)
        tag = ek.model_tag(model_id)
    else:
        print(f"Label folder {args.labels} | split {args.split} ({len(stems)} images)")
        tag = f"labels_{args.labels.name}"

    # Main loop. rows = one entry per real object with its coverage; stray_* count person/vehicle label pixels outside every real box.
    rows, stray_out, stray_all, used = [], 0, 0, 0
    for stem in tqdm(stems, desc="scoring"):
        if stem not in boxes or (predict is not None and not (args.images / f"{stem}.jpg").exists()):
            continue
        if predict is not None:
            pred = predict(Image.open(args.images / f"{stem}.jpg").convert("RGB"))
        else:
            mp = args.labels / "masks" / f"{stem}.png"
            if not mp.exists():
                continue
            pred = np.array(Image.open(mp))
        H, W = pred.shape
        used += 1
        # Mark every real risk object's box (plus a 10% margin) on a mask.
        cover = np.zeros((H, W), dtype=bool)
        for o in boxes[stem]:
            if not o["cls"]:
                continue
            x0, y0, x1, y1 = o["box"]
            dx, dy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            cover[max(int(y0 - dy), 0):int(y1 + dy), max(int(x0 - dx), 0):int(x1 + dx)] = True
        # Stray labels: person/rider/vehicle pixels the model predicted where no real object's box is.
        v = np.isin(pred, all_risk_ids)
        stray_out += int((v & ~cover).sum())
        stray_all += int(v.sum())
        city = ek.city_tag(stem)
        # Score each real object that is clearly visible (visibility >= --min_vis), within 80 m and tall enough in the image.
        for o in boxes[stem]:
            if not o["cls"] or o["cls"] not in CLS_GROUP or o["vis"] < args.min_vis or o["zmin"] > 80:
                continue
            x0, y0, x1, y1 = [int(round(t)) for t in o["box"]]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if y1 - y0 < args.min_box_px or x1 - x0 < 2:
                continue
            g = CLS_GROUP[o["cls"]]
            # Coverage = share of the object's 2D box that carries a label of the right group.
            cov = float(np.isin(pred[y0:y1, x0:x1], group_ids[g]).mean())
            rows.append({"g": g, "z": o["zmin"], "city": city, "cov": cov})

    if not rows:
        sys.exit("No objects scored.")
    res = {}

    # Summarise a set of objects: share found (coverage >= 10% and >= 30%) and mean coverage.
    def add(name, sel):
        if sel:
            c = np.array([r["cov"] for r in sel])
            res[name] = {"objects": len(sel), "found_10pct": float(np.mean(c >= 0.10)),
                         "found_30pct": float(np.mean(c >= 0.30)), "mean_coverage": float(c.mean())}

    # Report overall, by group, by distance (nearest surface of the box), by group and distance, and by city.
    add("all", rows)
    for g in GROUPS:
        add(f"group: {g}", [r for r in rows if r["g"] == g])
    for lo, hi in ek.DIST_BINS:
        add(f"{lo}-{hi} m", [r for r in rows if lo < r["z"] <= hi])
    for g in GROUPS:
        for lo, hi in ek.DIST_BINS:
            add(f"{g} {lo}-{hi} m", [r for r in rows if r["g"] == g and lo < r["z"] <= hi])
    for c in sorted({r["city"] for r in rows}):
        add(f"city: {c}", [r for r in rows if r["city"] == c])
    stray = stray_out / max(stray_all, 1)

    print(f"\n{used} images, {len(rows)} real objects scored\n")
    print(f"{'group':<26s}{'objects':>8s}{'found>=10%':>12s}{'found>=30%':>12s}{'mean coverage':>15s}")
    for k, v in res.items():
        print(f"{k:<26s}{v['objects']:>8d}{v['found_10pct']:>12.2f}{v['found_30pct']:>12.2f}{v['mean_coverage']:>15.3f}")
    print(f"\nStray labels: {100 * stray:.1f}% of all person/rider/vehicle label pixels lie outside every real object's box "
          f"({stray_all} pixels)")
    ek.save_json(args.out_dir / f"segboxes_{tag}_{args.split}.json",
                 {"settings": {k: str(v) for k, v in vars(args).items()}, "results": res, "stray_fraction": stray})


if __name__ == "__main__":
    main()
