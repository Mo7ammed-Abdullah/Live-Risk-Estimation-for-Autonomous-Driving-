"""
Pick 100 TEST images for a hand check and make one picture per image to look at.

Each picture (review/NNN_<city>.jpg) has two halves:
    left : the photo with the real objects from the nuScenes 3D boxes (coloured boxes: person / vehicle / two-wheeler,
           with true distance). This is the independent reference.
    right: what the segmentation model says (colour overlay with class names).
You compare the two halves and note what is wrong in review/review.csv (one row per image, already filled with the
number of real people / vehicles so you know what to look for).

Selection: half the images from each city, spread evenly over the drive (one random image from each time slice),
only images that contain at least one real person / rider / vehicle. Fixed random seed -> the same 100 every time.
This only LOOKS at the test images; nothing is tuned on them.

Run (venv311 active, inside "Testing 3"):
    python make_review_set.py                  # uses runs\\seg_b2_native\\best if it exists, else runs\\seg_b2\\best
    python make_review_set.py --seg_model runs\\seg_b2\\best
    python make_review_set.py --cpu            # model on the CPU (slower) while a GPU job is running
    python make_review_set.py --no_model       # only the left half (3D boxes), no GPU needed

How to fill review.csv (open it in Excel, one row per image, look at both halves of the picture):
    persons_missed    real people that have no 'person' label on them in the right half
    persons_false     'person' labels on things that are not people
    vehicles_missed   real cars / trucks / buses with no vehicle label on them
    vehicles_false    vehicle labels on things that are not vehicles
    wrong_class       real objects labelled as the wrong class (e.g. person labelled as car)
    notes             anything else (whole region wrong, very dark image, ...)
Skip objects that are tiny (a few pixels) or almost completely hidden: nobody could label those.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import evalkit as ek

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Group of each risk class, and the box colour used for each group in the pictures.
GROUP = {"person": "person", "rider": "person", "bicycle": "twowheel", "motorcycle": "twowheel",
         "car": "vehicle", "truck": "vehicle", "bus": "vehicle"}
COLOR = {"person": (255, 60, 60), "vehicle": (60, 160, 255), "twowheel": (60, 220, 120)}


# Real risk objects worth checking: visible enough, within 80 m and not tiny.
def risk_objects(objs, min_vis=2, min_px=8):
    out = []
    for o in objs:
        if not o["cls"] or o["vis"] < min_vis or o["zmin"] > 80:
            continue
        x0, y0, x1, y1 = o["box"]
        if y1 - y0 < min_px or x1 - x0 < 2:
            continue
        out.append(o)
    return out


# Load a readable font; older Pillow versions only have a small fixed one.
def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # old Pillow: fixed small font
        return ImageFont.load_default()


# Draw the real objects (3D boxes projected into the image) with class name and true distance on the photo.
def draw_truth(img, objs):
    d = ImageDraw.Draw(img)
    f = font(26)
    for o in objs:
        if not o["cls"] or o["vis"] < 1:
            continue
        x0, y0, x1, y1 = o["box"]
        if y1 - y0 < 8 or x1 - x0 < 2:
            continue
        g = GROUP[o["cls"]]
        c = COLOR[g]
        d.rectangle(o["box"], outline=c, width=3 if o["vis"] >= 2 else 1)
        d.text((x0 + 3, max(y0 - 28, 0)), f"{o['cls']} {o['zmin']:.0f}m", fill=c, font=f)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_model", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="run the model on the CPU (slower, about 5-8 minutes) so a GPU job keeps running")
    ap.add_argument("--seg_model", default="")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt", help="only for classes.json (class names)")
    ap.add_argument("--test_list", type=Path, default=HERE / "splits" / "test.txt")
    ap.add_argument("--out", type=Path, default=HERE / "review")
    ap.add_argument("--no_raw", action="store_true", help="do not copy the original photos into review/raw")
    args = ap.parse_args()

    if not args.boxes.exists():
        sys.exit("box_gt/boxes.json not found - run: python generate_box_gt.py")
    # Real objects from box_gt/boxes.json (the independent reference).
    boxes = json.load(open(args.boxes))
    stems = [s.strip() for s in args.test_list.read_text().splitlines() if s.strip()]
    # Candidate test images per city: sorted by drive and time, only those with at least one real risk object.
    by_city = {}
    for s in sorted(stems, key=lambda s: (s.split("__")[0], int(s.split("__")[2]))):
        if s in boxes and risk_objects(boxes[s]) and (args.images / f"{s}.jpg").exists():
            by_city.setdefault(ek.city_tag(s), []).append(s)
    cities = sorted(by_city)
    if not cities:
        sys.exit("No usable test images found.")
    # Pick the images: equal numbers per city, spread evenly over the drive (one random image per time slice).
    # The fixed seed gives the same selection every time.
    rng = random.Random(args.seed)
    chosen = []
    for k, c in enumerate(cities):
        n_c = args.n // len(cities) + (1 if k < args.n % len(cities) else 0)
        chunks = np.array_split(np.arange(len(by_city[c])), min(n_c, len(by_city[c])))
        chosen += [by_city[c][rng.choice(list(ch))] for ch in chunks if len(ch)]
    print(f"{len(chosen)} images selected: " + ", ".join(f"{c} {sum(ek.city_tag(s) == c for s in chosen)}" for c in cities))

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_raw:
        (args.out / "raw").mkdir(exist_ok=True)

    # Optionally load the segmentation model so the right half of each picture shows its prediction.
    seg = None
    model_name = "no model"
    if not args.no_model:
        from generate_seg_gt import load_labeler, make_color_table, make_overlay, predict_with_retry
        cand = [HERE / "runs" / "seg_b2_native" / "best", HERE / "runs" / "seg_b2" / "best"]
        model_id = args.seg_model or next((str(p) for p in cand if p.exists()), "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        model_name = model_id
        import torch
        device = torch.device("cpu") if args.cpu else ek.get_device()
        print(f"Segmentation model: {model_id} (device: {device})")
        processor, model = load_labeler(model_id, device)
        names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
        table = make_color_table(len(names))
        seg = (processor, model, device, names, table, predict_with_retry, make_overlay)

    # For each chosen image: left half = photo with real objects, right half = model overlay, one header line with the counts.
    rows = []
    for i, s in enumerate(chosen, 1):
        img = Image.open(args.images / f"{s}.jpg").convert("RGB")
        city = ek.city_tag(s)
        objs = boxes[s]
        real = risk_objects(objs)
        n_person = sum(GROUP[o["cls"]] == "person" for o in real)
        n_vehicle = sum(GROUP[o["cls"]] == "vehicle" for o in real)
        n_two = sum(GROUP[o["cls"]] == "twowheel" for o in real)
        left = draw_truth(img.copy(), objs).resize((800, 450))
        if seg is not None:
            processor, model, device, names, table, predict, overlay = seg
            pred = np.minimum(predict(processor, model, img, device, device.type == "cuda", img.height), len(names) - 1)
            right = Image.fromarray(overlay(np.array(img), pred, table, names)).resize((800, 450))
        else:
            right = Image.new("RGB", (800, 450), (40, 40, 40))
        sheet = Image.new("RGB", (1600, 450))
        sheet.paste(left, (0, 0))
        sheet.paste(right, (800, 0))
        d = ImageDraw.Draw(sheet)
        d.rectangle([0, 0, 1599, 22], fill=(0, 0, 0))
        d.text((4, 3), f"#{i:03d} {city} | truth (3D boxes): person {n_person}, vehicle {n_vehicle}, bike/motorbike {n_two}"
                       f"   |   right: model {Path(model_name).parent.name + '/' + Path(model_name).name if model_name != 'no model' else model_name}",
               fill=(255, 255, 255), font=font(15))
        fn = f"{i:03d}_{city}.jpg"
        sheet.save(args.out / fn, quality=92)
        if not args.no_raw:
            shutil.copy(args.images / f"{s}.jpg", args.out / "raw" / f"{i:03d}_{s}.jpg")
        rows.append([i, fn, s, city, model_name, n_person, n_vehicle, n_two, "", "", "", "", "", ""])
    # review.csv: one row per image with the true counts and empty columns to fill in by hand while checking the pictures.
    with open(args.out / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "picture", "image", "city", "model", "true_persons", "true_vehicles", "true_twowheelers", "persons_missed",
                    "persons_false", "vehicles_missed", "vehicles_false", "wrong_class", "notes"])
        w.writerows(rows)
    print(f"Done. Pictures and review.csv are in {args.out}\nOpen the pictures in order and fill review.csv.")


if __name__ == "__main__":
    main()
