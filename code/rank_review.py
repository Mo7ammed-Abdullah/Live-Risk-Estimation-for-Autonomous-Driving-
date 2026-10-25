"""
Rank the 100 review pictures by how badly the model fails, using the 3D boxes as the referee.
For each picture it counts (on the full-size image, test split, nothing tuned):
    persons_missed / vehicles_missed : real objects (visible, >=8 px, <=80 m) with < 10 % correct label inside their box
    persons_false  / vehicles_false  : separate label blobs (>= 150 px) of that group that lie outside every real box of that group
Writes review\\ranked.csv (worst first) and copies the 20 worst + 10 random pictures into review\\check30\\.

Run (venv311 active, inside "Testing 3"):
    python rank_review.py
    python rank_review.py --cpu
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import evalkit as ek

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Only people and vehicles are ranked (two-wheelers are too rare per picture).
GROUPS = {"person": ["person", "rider"], "vehicle": ["car", "truck", "bus"]}
CLS_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seg_model", default="")
    ap.add_argument("--min_blob", type=int, default=150)
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--review", type=Path, default=HERE / "review")
    args = ap.parse_args()

    boxes = json.load(open(args.boxes))
    # The 100 pictures chosen by make_review_set.py (read from review.csv).
    rows = list(csv.DictReader(open(args.review / "review.csv", encoding="utf-8-sig")))
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    # Class ids belonging to each group.
    gid = {g: [names.index(c) for c in cs if c in names] for g, cs in GROUPS.items()}

    import torch
    from generate_seg_gt import load_labeler, predict_with_retry
    # Use the same model that made the review pictures, unless another one is given.
    model_id = args.seg_model or rows[0]["model"]
    device = torch.device("cpu") if args.cpu else ek.get_device()
    print(f"Model {model_id} | device {device} | {len(rows)} pictures")
    processor, model = load_labeler(model_id, device)

    out = []
    # Score every picture with the 3D boxes as referee (test split, nothing is tuned).
    for r in rows:
        stem = r["image"]
        img = Image.open(args.images / f"{stem}.jpg").convert("RGB")
        # Run the segmentation model on the full-size photo.
        pred = np.minimum(predict_with_retry(processor, model, img, device, device.type == "cuda", img.height), len(names) - 1)
        H, W = pred.shape
        res = {}
        for g in GROUPS:
            # 'cover' marks the real boxes of this group (plus a 10% margin), to find labels that lie where no real object is.
            cover = np.zeros((H, W), dtype=bool)
            missed = 0
            for o in boxes[stem]:
                if not o["cls"] or CLS_GROUP.get(o["cls"]) != g:
                    continue
                x0, y0, x1, y1 = o["box"]
                dx, dy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
                cover[max(int(y0 - dy), 0):int(y1 + dy), max(int(x0 - dx), 0):int(x1 + dx)] = True
                if o["vis"] < 2 or o["zmin"] > 80:
                    continue
                a, b, c, d = [int(round(t)) for t in (x0, y0, x1, y1)]
                a, b, c, d = max(a, 0), max(b, 0), min(c, W), min(d, H)
                if d - b < 8 or c - a < 2:
                    continue
                # Missed = a clearly visible real object with less than 10% correct label inside its box.
                if np.isin(pred[b:d, a:c], gid[g]).mean() < 0.10:
                    missed += 1
            # False blobs = connected label regions of this group outside every real box, at least --min_blob pixels large.
            lab, n = ndimage.label(np.isin(pred, gid[g]) & ~cover)
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else []
            res[g] = (missed, int(np.sum(np.array(sizes) >= args.min_blob)) if n else 0)
        # Score of a picture = total misses + false blobs (higher = worse).
        score = sum(res[g][0] + res[g][1] for g in GROUPS)
        out.append({"id": r["id"], "picture": r["picture"], "city": r["city"],
                    "true_persons": r["true_persons"], "true_vehicles": r["true_vehicles"],
                    "persons_missed": res["person"][0], "persons_false": res["person"][1],
                    "vehicles_missed": res["vehicle"][0], "vehicles_false": res["vehicle"][1], "score": score})
        print(f"{r['picture']}: persons miss {res['person'][0]} false {res['person'][1]} | vehicles miss {res['vehicle'][0]} false {res['vehicle'][1]}")

    # Worst pictures first; saved to ranked.csv.
    out.sort(key=lambda x: -x["score"])
    with open(args.review / "ranked.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # Totals over all pictures. 'Found' is a lower bound, because tiny objects are counted as real but skipped by the scorer.
    tp = sum(int(x["true_persons"]) for x in out)
    tv = sum(int(x["true_vehicles"]) for x in out)
    pm = sum(x["persons_missed"] for x in out)
    vm = sum(x["vehicles_missed"] for x in out)
    print(f"\nTotals over {len(out)} pictures (auto count, 3D-box referee):")
    print(f"  persons: {tp} real, {pm} missed ({100 * (1 - pm / max(tp, 1)):.0f}% found), {sum(x['persons_false'] for x in out)} false blobs")
    print(f"  vehicles: {tv} real, {vm} missed ({100 * (1 - vm / max(tv, 1)):.0f}% found), {sum(x['vehicles_false'] for x in out)} false blobs")
    print("  (true counts include small objects the scorer skips, so 'found' is a floor)")

    # Copy the 20 worst and 10 random pictures into check30/ for a quick manual look.
    chk = args.review / "check30"
    chk.mkdir(exist_ok=True)
    worst = out[:20]
    rest = [x for x in out[20:] if x["score"] > 0 or True]
    rnd = random.Random(0).sample(rest, min(10, len(rest)))
    for tag, lst in (("worst", worst), ("random", rnd)):
        for x in lst:
            shutil.copy(args.review / x["picture"], chk / f"{tag}_{x['picture']}")
    print(f"\nCopied 20 worst + 10 random pictures to {chk}")
    print("Worst 20:", ", ".join(x["picture"] for x in worst))


if __name__ == "__main__":
    main()
