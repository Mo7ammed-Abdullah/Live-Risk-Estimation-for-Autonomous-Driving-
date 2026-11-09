"""
Learned fusion (supervisor point 3): a small network that turns the two models' outputs into a danger map,
trained against REAL danger labels (time-to-collision, box_gt/ttc.json) instead of using fixed class weights.

Inputs per pixel (from cache/fusion_<split>.npy, 1/8 image size):
    class probabilities (8 risk classes + other), segmentation confidence, metric depth, log depth confidence,
    the fixed formula's risk  w_c * exp(-0.05 * depth)  (so the network can start from it),
    and the pixel's position in the image (row, column).
Network: three small convolutions (about 5 000 weights) -> one danger value per pixel.
    It is "uncertainty-aware" because both models' confidences are inputs: it can learn to trust
    a pixel's class or depth less when the model is unsure.

Training signal (per real object of a risk class):
    label 1 = on a collision course within --horizon s (TTC), 0 = not.
    The object's predicted danger = the highest danger inside its box (multiple-instance style),
    plus a weak push towards 0 for pixels outside every object box.

Compared on the SAME objects against:
    fixed formula      max of  w_c * exp(-0.05 * depth)  inside the box   (what the system uses today)
    distance only      the closer the object (predicted depth), the higher the score
    oracle static risk the fixed formula with the TRUE class and TRUE distance (best a static rule can do)

Metrics: AP (average precision) and AUROC for finding collision-course objects; recall at 10 % false alarms;
         hit@1 = in frames with a dangerous object, the top-scored object is a dangerous one.

Run (venv311 active, project folder; needs cache/fusion_train.* and cache/fusion_val.*):
    python code/train_fusion.py                 # trains 3 seeds, picks epochs on validation, prints the comparison
    python code/train_fusion.py --load          # re-print the validation comparison of the saved models (no training)
Final, after everything is frozen (never trains: it loads the saved models from runs/fusion and only scores them):
    python code/train_fusion.py --eval_split test --final
Ablations: add  --drop conf  or  --drop position  (saved in runs/fusion_no_<group>), same for the test look.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import evalkit as ek
import risk as rk

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
RISK = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
SCALE = 8                       # feature maps are 1/8 of the 900 x 1600 image
# Input groups that can be switched off (--drop) to see what the learned fusion relies on.
GROUPS = {"conf": [9, 11], "position": [13, 14], "fixed": [12], "depth": [10], "class": list(range(9))}
DROP = []                       # channel indices zeroed in every input (set from --drop)


# ----------------------------------------------------------------------------- data
def load_split(split, boxes, ttc, horizon, min_vis):
    """Feature memmap + per-object labels for one split."""
    meta = json.load(open(HERE / "cache" / f"fusion_{split}.json"))
    feats = np.load(HERE / "cache" / f"fusion_{split}.npy", mmap_mode="r")
    H, W = meta["size"]
    objs = []                   # (frame index, y0, y1, x0, x1, label, class id, true zmin)
    for fi, stem in enumerate(meta["stems"]):
        lab = {r["inst"]: r for r in ttc.get(stem, []) if r.get("has_motion")}
        for o in boxes.get(stem, []):
            if not o["cls"] or o["cls"] not in RISK or o["inst"] not in lab or o["vis"] < min_vis:
                continue
            x0, y0, x1, y1 = o["box"]
            x0, x1 = int(np.floor(x0 / SCALE)), int(np.ceil(x1 / SCALE))
            y0, y1 = int(np.floor(y0 / SCALE)), int(np.ceil(y1 / SCALE))
            x0, y0 = min(max(x0, 0), W - 1), min(max(y0, 0), H - 1)
            x1, y1 = max(min(x1, W), x0 + 1), max(min(y1, H), y0 + 1)
            t = lab[o["inst"]]["ttc"]
            objs.append((fi, y0, y1, x0, x1, int(t is not None and t <= horizon), RISK.index(o["cls"]), o["zmin"]))
    return feats, np.array(objs, dtype=np.float64).reshape(-1, 8), meta


def make_inputs(f, torch):
    """(B, 12, H, W) float16 cache -> (B, 15, H, W) network input."""
    f = f.float()
    probs = f[:, :9]
    depth = f[:, 10:11]
    w_vec = torch.tensor(rk.class_weight_vector(RISK + ["other"]), device=f.device).view(1, 9, 1, 1)
    cls_w = (w_vec * (probs == probs.max(1, keepdim=True).values)).sum(1, keepdim=True)   # weight of the top class
    fixed = cls_w * torch.exp(-rk.DEFAULT_LAMBDA_METRIC * depth)                          # the fixed formula
    B, _, H, W = f.shape
    yy = torch.linspace(0, 1, H, device=f.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(-1, 1, W, device=f.device).view(1, 1, 1, W).expand(B, 1, H, W)
    # UniDepth's confidence is unbounded (mostly 0.6-2.6, rare values up to ~240): use its log so a few
    # extreme values cannot dominate the network.
    depth_conf = torch.log(f[:, 11:12].clamp(min=1e-3))
    x = torch.cat([probs, f[:, 9:10], depth / 80.0, depth_conf, fixed, yy, xx], 1)
    if DROP:
        x[:, DROP] = 0.0
    return x, fixed


def build_net(torch, width=16):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(15, width, 3, padding=1), nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
        nn.Conv2d(width, 1, 1))


# ----------------------------------------------------------------------------- metrics
def average_precision(y, s):
    order = np.argsort(-s, kind="stable")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with average ranks for ties.
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts                     # 0-based start rank of each distinct value
    ranks = (first + (counts + 1) / 2.0)[inv]              # 1-based average rank
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(y, s, fpr=0.10):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = neg[min(int(np.floor(fpr * len(neg))), len(neg) - 1)]
    return float((s[y == 1] > thr).mean())


def hit_at_1(objs, s):
    hits, n = 0, 0
    for fi in np.unique(objs[:, 0]):
        m = objs[:, 0] == fi
        if objs[m, 5].sum() == 0:
            continue
        n += 1
        hits += objs[m, 5][np.argmax(s[m])] == 1
    return hits / max(n, 1), n


def report(name, objs, s):
    y = objs[:, 5]
    h, n = hit_at_1(objs, s)
    return {"method": name, "AP": average_precision(y, s), "AUROC": auroc(y, s),
            "recall@10%FA": recall_at_fpr(y, s), "hit@1": h, "frames_with_danger": n}


# ----------------------------------------------------------------------------- scoring
def object_scores(torch, net, feats, objs, device, batch=32):
    """Max of the learned danger map and of the fixed-formula map inside each object box."""
    learned, fixed_s, dist_s = np.zeros(len(objs)), np.zeros(len(objs)), np.zeros(len(objs))
    frames = np.unique(objs[:, 0]).astype(int)
    for i in range(0, len(frames), batch):
        fr = frames[i:i + batch]
        f = torch.from_numpy(np.ascontiguousarray(feats[fr])).to(device)
        x, fixed = make_inputs(f, torch)
        with torch.no_grad():
            d = torch.sigmoid(net(x)) if net is not None else fixed
        d, fixed, depth = d[:, 0].cpu().numpy(), fixed[:, 0].cpu().numpy(), f[:, 10].float().cpu().numpy()
        for j, fi in enumerate(fr):
            for k in np.nonzero(objs[:, 0] == fi)[0]:
                _, y0, y1, x0, x1 = objs[k, :5].astype(int)
                learned[k] = d[j, y0:y1, x0:x1].max()
                fixed_s[k] = fixed[j, y0:y1, x0:x1].max()
                dist_s[k] = -depth[j, y0:y1, x0:x1].min()
    return learned, fixed_s, dist_s


# ----------------------------------------------------------------------------- training
def train_one(torch, seed, tr_feats, tr_objs, va_feats, va_objs, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_net(torch, args.width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    n_pos = max(tr_objs[:, 5].sum(), 1)
    pos_w = torch.tensor(min((len(tr_objs) - n_pos) / n_pos, args.max_pos_weight), device=device)
    frames = np.unique(tr_objs[:, 0]).astype(int)
    by_frame = {fi: np.nonzero(tr_objs[:, 0] == fi)[0] for fi in frames}
    best, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        net.train()
        rng.shuffle(frames)
        total = 0.0
        for i in range(0, len(frames), args.batch):
            fr = np.sort(frames[i:i + args.batch])
            f = torch.from_numpy(np.ascontiguousarray(tr_feats[fr])).to(device)
            x, _ = make_inputs(f, torch)
            if args.flip and rng.random() < 0.5:                          # left-right flip augmentation
                x = torch.flip(x, dims=[3]); x[:, 14] = -x[:, 14]
                flipped = True
            else:
                flipped = False
            logit = net(x)[:, 0]
            obj_logits, obj_y = [], []
            outside = torch.ones_like(logit, dtype=torch.bool)
            W = logit.shape[2]
            for j, fi in enumerate(fr):
                for k in by_frame[fi]:
                    _, y0, y1, x0, x1 = tr_objs[k, :5].astype(int)
                    if flipped:
                        x0, x1 = W - x1, W - x0
                    obj_logits.append(logit[j, y0:y1, x0:x1].max())
                    obj_y.append(tr_objs[k, 5])
                    outside[j, y0:y1, x0:x1] = False
            obj_logits = torch.stack(obj_logits)
            obj_y = torch.tensor(obj_y, dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(obj_logits, obj_y, pos_weight=pos_w)
            bg = logit[outside]
            if bg.numel():
                loss = loss + args.bg_weight * torch.nn.functional.binary_cross_entropy_with_logits(bg, torch.zeros_like(bg))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(fr)
        net.eval()
        s, _, _ = object_scores(torch, net, va_feats, va_objs, device)
        ap = average_precision(va_objs[:, 5], s)
        history.append({"epoch": ep, "train_loss": total / len(frames), "val_AP": ap})
        print(f"  seed {seed} epoch {ep:2d}  train loss {total / len(frames):.4f}  val AP {ap:.3f}")
        if ap > best:
            best, best_state = ap, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best, history


def fmt(rows):
    keys = ["AP", "AUROC", "recall@10%FA", "hit@1"]
    out = [f"{'method':32s}" + "".join(f"{k:>14s}" for k in keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            cells.append(f"{v:14.3f}" if not isinstance(v, str) else f"{v:>14s}")
        out.append(f"{r['method']:32s}" + "".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    ap.add_argument("--horizon", type=float, default=4.0, help="TTC up to this many seconds = dangerous")
    ap.add_argument("--min_vis", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--bg_weight", type=float, default=0.1)
    ap.add_argument("--max_pos_weight", type=float, default=20.0)
    ap.add_argument("--flip", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--drop", nargs="*", default=[], choices=list(GROUPS),
                    help="ablation: switch off input groups, e.g. --drop conf  or  --drop position")
    ap.add_argument("--out_dir", type=Path, default=None, help="default runs/fusion (or runs/fusion_no_<groups>)")
    ap.add_argument("--load", action="store_true", help="do not train: score the models saved in --out_dir "
                    "(always on for --eval_split test, so the test look uses exactly the validated models)")
    args = ap.parse_args()
    DROP[:] = sorted(i for g in args.drop for i in GROUPS[g])
    if args.out_dir is None:
        args.out_dir = HERE / "runs" / ("fusion" + ("_no_" + "_".join(args.drop) if args.drop else ""))
    if args.eval_split == "test" and not args.final:
        sys.exit("The TEST split is reserved for your final numbers. Add --final only when everything is frozen.")
    if args.eval_split == "test":
        args.load = True
    config = {"drop": sorted(args.drop), "width": args.width, "depth_conf": "log", "horizon": args.horizon,
              "seeds": args.seeds}

    import torch
    device = ek.get_device()
    boxes = json.load(open(HERE / "box_gt" / "boxes.json"))
    ttc = json.load(open(HERE / "box_gt" / "ttc.json"))
    tr_feats, tr_objs, _ = load_split("train", boxes, ttc, args.horizon, args.min_vis)
    va_feats, va_objs, _ = load_split("val", boxes, ttc, args.horizon, args.min_vis)
    if args.drop:
        print(f"ABLATION: inputs switched off: {', '.join(args.drop)} (channels {DROP})")
    print(f"device {device} | train: {len(tr_objs)} objects, {int(tr_objs[:, 5].sum())} dangerous | "
          f"val: {len(va_objs)} objects, {int(va_objs[:, 5].sum())} dangerous")

    nets, logs = [], {}
    if args.load:
        # Score the saved models; refuse if they were trained with different settings.
        cfg_file = args.out_dir / "config.json"
        if not cfg_file.exists():
            sys.exit(f"No trained models in {args.out_dir} (config.json missing). Train first without --load / --final.")
        saved = json.load(open(cfg_file))
        if saved != config:
            sys.exit(f"Saved models in {args.out_dir} were trained with {saved}, not {config}. Retrain or match the options.")
        for seed in args.seeds:
            net = build_net(torch, args.width).to(device)
            net.load_state_dict(torch.load(args.out_dir / f"fusion_seed{seed}.pt", map_location=device))
            nets.append(net.eval())
        logs = json.load(open(args.out_dir / "fusion_val.json")).get("training", {}) if (args.out_dir / "fusion_val.json").exists() else {}
        print(f"Loaded {len(nets)} saved models from {args.out_dir} (no training)")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            net, best, hist = train_one(torch, seed, tr_feats, tr_objs, va_feats, va_objs, args, device)
            nets.append(net.eval())
            logs[seed] = {"best_val_AP": best, "history": hist}
            torch.save(net.state_dict(), args.out_dir / f"fusion_seed{seed}.pt")
            print(f"seed {seed}: best val AP {best:.3f}")
        json.dump(config, open(args.out_dir / "config.json", "w"), indent=1)

    # Evaluate on the chosen split (validation by default; test only with --final).
    if args.eval_split == "test":
        ev_feats, ev_objs, _ = load_split("test", boxes, ttc, args.horizon, args.min_vis)
    else:
        ev_feats, ev_objs = va_feats, va_objs
    rows, per_seed = [], []
    fixed_s = dist_s = None
    for seed, net in zip(args.seeds, nets):
        s, fixed_s, dist_s = object_scores(torch, net, ev_feats, ev_objs, device)
        per_seed.append(report(f"learned fusion (seed {seed})", ev_objs, s))
    mean_row = {"method": f"learned fusion (mean of {len(nets)})"}
    for k in ["AP", "AUROC", "recall@10%FA", "hit@1"]:
        vals = np.array([r[k] for r in per_seed])
        mean_row[k] = float(np.nanmean(vals))
        mean_row[k + "_std"] = float(np.nanstd(vals))
    w_vec = rk.class_weight_vector(RISK)
    oracle = w_vec[ev_objs[:, 6].astype(int)] * np.exp(-rk.DEFAULT_LAMBDA_METRIC * np.clip(ev_objs[:, 7], 0, 80))
    rows = [report("fixed formula (today's system)", ev_objs, fixed_s),
            report("distance only (predicted)", ev_objs, dist_s),
            report("oracle static risk (true cls+dist)", ev_objs, oracle),
            *per_seed, mean_row]
    print(f"\n=== {args.eval_split}: finding objects on a collision course within {args.horizon:.0f} s "
          f"({int(ev_objs[:, 5].sum())} of {len(ev_objs)} objects; hit@1 over {rows[0]['frames_with_danger']} frames) ===")
    print(fmt(rows))
    print("std over seeds: " + ", ".join(f"{k} {mean_row[k + '_std']:.3f}" for k in ["AP", "AUROC", "recall@10%FA", "hit@1"]))
    out = args.out_dir / f"fusion_{args.eval_split}.json"
    json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": rows, "training": logs}, open(out, "w"), indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
