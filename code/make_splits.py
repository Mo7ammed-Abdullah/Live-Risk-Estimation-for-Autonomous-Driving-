"""
Build a train / val / test split BY DRIVE LOG (never by single frame) and check class coverage.

Why by log: frames from the same drive look almost identical, so putting neighbours in different splits
inflates every score. Whole logs go to one split. Because rare classes (rider, bicycle, motorcycle, person...)
sit in only a few logs, the script searches ALL ways of assigning logs to splits and picks the one that
(1) matches the target image fractions and (2) leaves enough images of every risk class in every split.

Needs: the masks made by generate_seg_gt.py (seg_gt/masks + seg_gt/classes.json). Put this file in "Testing 3".

Usage:
    python make_splits.py                                  # propose a split, print tables, write splits/
    python make_splits.py --test_logs n015-2018-07-18-11-50-34+0800   # force a log into test, search the rest
    python make_splits.py --review train truck             # also save overlays of the images with the most 'train'/'truck' pixels

Outputs (in splits/):
    splits.json, train.txt, val.txt, test.txt     image names (without extension), one per line in the .txt files
    per_image_class_pixels.csv                    pixels of every class in every image
    review/<class>/                               (only with --review) overlays to eyeball suspicious classes
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
# Classes that carry collision risk; the split makes sure each of them appears in train, val and test.
RISK_CLASSES = ["person", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]


def city_of(log):
    """nuScenes log names end in the local UTC offset: +0800 = Singapore, -0400 = Boston."""
    if log.endswith("+0800"):
        return "singapore"
    if log.endswith("-0400") or log.endswith("-0500"):
        return "boston"
    return "unknown"


# Count how many pixels of each class every mask image has (slow the first time, so the result is cached in a CSV).
def scan_masks(mask_dir, n_classes):
    files = sorted(mask_dir.glob("*.png"))
    if not files:
        sys.exit(f"ERROR: no masks found in {mask_dir}. Run generate_seg_gt.py first.")
    counts = np.zeros((len(files), n_classes), dtype=np.int64)
    for i, f in enumerate(tqdm(files, desc="reading masks")):
        # One bincount per mask = pixels per class id.
        counts[i] = np.bincount(np.array(Image.open(f)).ravel(), minlength=256)[:n_classes]
    stems = [f.stem for f in files]
    logs = [s.split("__")[0] for s in stems]
    return stems, logs, counts


def load_cached_counts(csv_path, mask_dir, names):
    """Reuse per_image_class_pixels.csv from an earlier run if it matches the masks on disk."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or rows[0][2:] != names:
        return None
    stems = [r[0] for r in rows[1:]]
    if set(stems) != {p.stem for p in mask_dir.glob("*.png")}:
        return None
    logs = [r[1] for r in rows[1:]]
    counts = np.array([[int(x) for x in r[2:]] for r in rows[1:]], dtype=np.int64)
    return stems, logs, counts


# Choose which drives go to train / val / test. Whole drives stay together, because neighbouring frames look almost the same.
# With few drives, every possible assignment is tried; each gets a cost and the lowest cost wins.
def search_split(log_names, n_img, pix, img_with, names, args):
    """Try every assignment of logs to train(0)/val(1)/test(2); return the best one."""
    L = len(log_names)
    cidx = {n: i for i, n in enumerate(names)}
    risk = [cidx[c] for c in RISK_CLASSES if c in cidx]
    tot_img = img_with.sum(0)

    # 3 choices (train/val/test) for each of L drives: try all combinations if there are few, otherwise a random sample.
    if 3 ** L <= 600_000:
        A = np.indices((3,) * L).reshape(L, -1).T.astype(np.int8)
    else:  # many logs: random search instead of full enumeration
        A = np.random.default_rng(0).integers(0, 3, size=(400_000, L)).astype(np.int8)

    # Apply any drives forced into a split from the command line (--train_logs / --val_logs / --test_logs).
    keep = np.ones(len(A), dtype=bool)
    for name, split in [(n, 1) for n in args.val_logs] + [(n, 2) for n in args.test_logs] + [(n, 0) for n in args.train_logs]:
        if name not in log_names:
            sys.exit(f"ERROR: log '{name}' not found. Known logs:\n  " + "\n  ".join(log_names))
        keep &= A[:, log_names.index(name)] == split
    A = A[keep]
    for s in range(3):
        # Every split must contain at least one drive.
        A = A[(A == s).any(1)]  # every split non-empty
    if len(A) == 0:
        sys.exit("ERROR: no valid assignment (check --val_logs/--test_logs/--train_logs).")

    total = n_img.sum()
    targets = [1 - args.val_frac - args.test_frac, args.val_frac, args.test_frac]
    # Cost of each candidate = how far the image share is from the target (70/15/15), how unevenly each risk class is spread,
    # how few images of a risk class a split gets, and whether both cities appear in val and test.
    cost = np.zeros(len(A))
    cities = np.array([city_of(l) for l in log_names])
    for s in range(3):
        m = (A == s).astype(np.float64)
        frac = m @ n_img / total
        cost += 20 * (frac - targets[s]) ** 2
        # pixel share of each risk class should roughly follow the image share
        share = (m @ pix[:, risk]) / np.maximum(pix[:, risk].sum(0), 1)
        cost += ((share - targets[s]) ** 2).mean(1)
        # enough images containing each risk class (only for classes frequent enough overall)
        imgs = m @ img_with[:, risk]
        for j, c in enumerate(risk):
            if tot_img[c] >= 3 * args.min_images:
                cost += 10 * (imgs[:, j] < args.min_images)
        # both cities represented in val and test (soft)
        if s > 0:
            for city in set(cities) - {"unknown"}:
                has = (m[:, cities == city].sum(1) > 0)
                cost += 0.05 * (~has)
    # Lowest cost = best assignment of drives to splits.
    best = A[int(np.argmin(cost))]
    return best, float(cost.min())


# Save overlay pictures of the images with the most pixels of a class, to check by eye that the labels are right.
def save_review(classes_json, images_dir, out_dir, class_names, stems, counts, names, top_n):
    from generate_seg_gt import make_color_table, make_overlay
    import cv2  # noqa: F401  (make_overlay needs it)
    table = make_color_table(len(names))
    for cname in class_names:
        if cname not in names:
            print(f"  (skip review: unknown class '{cname}')")
            continue
        c = names.index(cname)
        order = np.argsort(-counts[:, c])[:top_n]
        dest = out_dir / "review" / cname
        dest.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(order):
            if counts[i, c] == 0:
                break
            img_path = next((p for p in (images_dir / f"{stems[i]}{e}" for e in (".jpg", ".jpeg", ".png")) if p.exists()), None)
            if img_path is None:
                continue
            mask = np.array(Image.open(args_masks / f"{stems[i]}.png"))
            overlay = make_overlay(np.array(Image.open(img_path).convert("RGB")), mask, table, names, label_regions=True)
            Image.fromarray(overlay).save(dest / f"{rank:02d}_{counts[i, c]}px_{stems[i]}.png")
        print(f"  review overlays for '{cname}' -> {dest}")


def main():
    # args_masks is shared with save_review() as the mask folder.
    global args_masks
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt")
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--out", type=Path, default=HERE / "splits")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--min_px", type=int, default=500, help="pixels needed for an image to 'contain' a class")
    ap.add_argument("--min_images", type=int, default=15, help="minimum images per risk class in every split")
    ap.add_argument("--train_logs", nargs="*", default=[], help="force these logs into train")
    ap.add_argument("--val_logs", nargs="*", default=[], help="force these logs into val")
    ap.add_argument("--test_logs", nargs="*", default=[], help="force these logs into test")
    ap.add_argument("--review", nargs="*", default=[], help="class names to export review overlays for, e.g. train truck")
    ap.add_argument("--review_n", type=int, default=12)
    ap.add_argument("--rescan", action="store_true", help="re-read all masks instead of using the saved pixel counts")
    ap.add_argument("--counts_only", action="store_true",
                    help="only count class pixels per image (writes per_image_class_pixels.csv into --out) and stop; "
                         "the existing split files are not touched")
    args = ap.parse_args()

    # Read class names from the label files, then load (or re-count) the per-image class pixel counts.
    args_masks = args.seg_dir / "masks"
    with open(args.seg_dir / "classes.json") as f:
        names = [c["name"] for c in json.load(f)["classes"]]
    n = len(names)

    cached = None if args.rescan else load_cached_counts(args.out / "per_image_class_pixels.csv", args_masks, names)
    if cached:
        stems, logs, counts = cached
        print("Using saved pixel counts from the last run (add --rescan to re-read the masks).")
    else:
        stems, logs, counts = scan_masks(args_masks, n)
    # Per-drive totals: images, pixels of each class, and images that contain at least --min_px pixels of each class.
    log_names = sorted(set(logs))
    li = np.array([log_names.index(l) for l in logs])
    n_img = np.bincount(li, minlength=len(log_names)).astype(np.float64)
    pix = np.zeros((len(log_names), n))
    img_with = np.zeros((len(log_names), n))
    for k in range(len(log_names)):
        sel = counts[li == k]
        pix[k] = sel.sum(0)
        img_with[k] = (sel >= args.min_px).sum(0)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_image_class_pixels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "log"] + names)
        for s, l, c in zip(stems, logs, counts):
            w.writerow([s, l] + c.tolist())

    # Print a table of how many images each drive has per risk class, and warn about classes that are too rare to test.
    risk = [c for c in RISK_CLASSES if c in names]
    print(f"\n{len(stems)} images in {len(log_names)} logs. Images containing each risk class (>= {args.min_px} px):")
    print(f"{'log':<34s}{'city':<11s}{'imgs':>5s}  " + " ".join(f"{c[:5]:>5s}" for c in risk))
    for k, l in enumerate(log_names):
        print(f"{l:<34s}{city_of(l):<11s}{int(n_img[k]):>5d}  " + " ".join(f"{int(img_with[k, names.index(c)]):>5d}" for c in risk))
    for c in risk:
        total = int(img_with[:, names.index(c)].sum())
        if total < 3 * args.min_images:
            print(f"WARNING: '{c}' appears in only {total} images, too few to evaluate it reliably in val/test.")

    if args.counts_only:
        print(f"\nCounts written to {args.out / 'per_image_class_pixels.csv'}; no split was made or changed.")
        return
    # Search for the best split and build the lists of image names for train / val / test.
    assign, cost = search_split(log_names, n_img, pix, img_with, names, args)
    split_names = ["train", "val", "test"]
    result = {s: [] for s in split_names}
    result_logs = {s: [] for s in split_names}
    for k, l in enumerate(log_names):
        result_logs[split_names[assign[k]]].append(l)
    for stem, l in zip(stems, logs):
        result[split_names[assign[log_names.index(l)]]].append(stem)

    print(f"\nChosen split (cost {cost:.3f}):")
    for s in split_names:
        m = np.isin(log_names, result_logs[s])
        imgs = int(n_img[m].sum())
        row = " ".join(f"{int(img_with[m, names.index(c)].sum()):>5d}" for c in risk)
        print(f"  {s:<6s}{imgs:>5d} images ({100 * imgs / len(stems):4.1f}%)  logs: {len(result_logs[s])}   risk-class images: {row}")
        for l in result_logs[s]:
            print(f"        {l}  [{city_of(l)}]")

    # Report any split that ended up with too few images of a risk class.
    problems = []
    for s in split_names:
        m = np.isin(log_names, result_logs[s])
        for c in risk:
            have = int(img_with[m, names.index(c)].sum())
            if img_with[:, names.index(c)].sum() >= 3 * args.min_images and have < args.min_images:
                problems.append(f"  {s}: only {have} images contain '{c}' (wanted at least {args.min_images})")
    if problems:
        print("\nCOVERAGE WARNINGS (no better assignment of whole logs exists with these settings):")
        print("\n".join(problems))
        print("  Options: lower --min_images, force a different log with --test_logs/--val_logs, or hand-label "
              "extra images of that class for the test set.")

    # Save the split: splits.json plus train.txt, val.txt and test.txt (one image name per line).
    with open(args.out / "splits.json", "w") as f:
        json.dump({"logs": result_logs, "images": result, "min_px": args.min_px}, f, indent=1)
    for s in split_names:
        (args.out / f"{s}.txt").write_text("\n".join(result[s]) + "\n")
    print(f"\nSaved split files to {args.out}")

    if args.review:
        print("\nExporting review overlays:")
        save_review(args.seg_dir / "classes.json", args.images, args.out, args.review, stems, counts, names, args.review_n)

    print("\nNote: logs from the same city often drive the same streets on different days, so a log-level test set "
          "means 'unseen drives', not necessarily 'unseen places'. Say so when you report results.")


if __name__ == "__main__":
    main()
