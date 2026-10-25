"""
Run the whole system: segmentation + depth -> risk map -> region urgency -> decision.

For every frame it saves a picture like Figure 3 of the base paper:
    [ segmentation | depth in meters | risk map (with the 3 horizontal regions) ]
with the numbers on top, plus a CSV with the risk values and timing of every frame, and optionally an MP4.

Run with venv311 active, from "Testing 3" (needs risk.py, evalkit.py, eval_depth.py, generate_seg_gt.py, make_splits.py):
    python demo_risk.py                                         # first 40 frames of your dataset, fine-tuned SegFormer
    python demo_risk.py --log n015-2018-07-24-11-22-45 --limit 0 --video     # a whole drive as a video
    python demo_risk.py --input C:\\path\\to\\my_video.mp4 --video            # your own video (camera unknown)
    python demo_risk.py --resize_height 360                     # paper-like small size: faster, less accurate
    python demo_risk.py --depth_mode metric                     # use real meters in the risk formula

Output goes to demo_out/ : frames/*.jpg, risk_per_frame.csv, demo.mp4 (with --video).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import evalkit as ek
import risk as rk
from eval_depth import load_depth_model, run_depth
from generate_seg_gt import load_labeler, make_color_table, predict_with_retry

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
# Read frames one at a time from a folder of images, a single image or a video file.
# --log keeps only one drive, --every skips frames, --limit stops after N frames.
def iter_frames(path, log, every, limit):
    """Yield (name, PIL image) from a folder of images, a single image or a video file."""
    path = Path(path)
    count = 0
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if log:
            files = [p for p in files if log in p.name]
        for p in files[:: max(every, 1)]:
            yield p.stem, Image.open(p).convert("RGB")
            count += 1
            if limit and count >= limit:
                return
    elif path.suffix.lower() in VIDEO_EXTS:
        import cv2
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(every, 1) == 0:
                yield f"frame{idx:05d}", Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                count += 1
                if limit and count >= limit:
                    break
            idx += 1
        cap.release()
    elif path.suffix.lower() in IMAGE_EXTS:
        yield path.stem, Image.open(path).convert("RGB")
    else:
        sys.exit(f"ERROR: don't know how to read {path}")


# Camera intrinsics of a nuScenes frame (from the metadata cache). Other videos have none, so the depth model must guess the camera.
def intrinsics_for(stem, meta):
    if meta is None:
        return None
    sd = meta["sd"].get(f"{stem}.jpg")
    return meta["cs"][sd["calibrated_sensor_token"]]["camera_intrinsic"] if sd else None


# ----------------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------------
# Draw readable text: a dark box behind it, then a black outline, then the coloured text on top.
def _txt(img, s, org, scale=0.8, color=(255, 255, 255), th=2):
    import cv2
    (tw, tht), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x0, y0 = max(org[0] - 4, 0), max(org[1] - tht - 6, 0)
    x1, y1 = min(org[0] + tw + 4, img.shape[1]), min(org[1] + 6, img.shape[0])
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(img.dtype)       # dark box so text is readable
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


# Build the picture shown on screen and saved in the video:
#   top left  = camera with the risk overlay, region lines and per-region numbers
#   top right = segmentation
#   bottom left = depth map (red = near)
#   bottom right = decision panel (most urgent region, risk level, bars, FPS).
def make_panel(rgb, seg, depth, risk_map, urgency, best, bands, names, color_table, text, max_depth, tile_w=800, zone_names=None, fps=0.0):
    """2x2 layout: [camera + risk overlay + regions | segmentation] / [depth | decision panel]."""
    import cv2
    H, W = rgb.shape[:2]
    tile_h = int(round(tile_w * H / W))
    k = tile_h / H
    zone_names = zone_names or ["far", "mid", "near"]
    # Risk overlay: only risk above a small threshold is drawn so the picture stays readable.
    rm = np.clip(risk_map, 0, 1)

    # risk overlay: only show risk above a small threshold so the picture stays readable
    heat = cv2.cvtColor(cv2.applyColorMap((255 * np.clip(rm / max(rm.max(), 0.3), 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    a = (np.clip((rm - 0.08) / 0.5, 0, 1) * 0.85)[..., None]
    risk_vis = (rgb * (1 - a) + heat * a).astype(np.uint8)
    seg_vis = cv2.addWeighted(rgb, 0.45, color_table[seg], 0.55, 0)
    # Depth colours are scaled per frame (2nd to 98th percentile) so near and far are always visible.
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)           # per-frame contrast so near/far is visible
    d = (255 * (1 - np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1))).astype(np.uint8)
    depth_vis = cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    # Resize every tile to the same size.
    def fit(arr):
        return cv2.resize(arr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    t_risk, t_seg, t_depth = fit(risk_vis), fit(seg_vis), fit(depth_vis)
    cols = [(90, 90, 90)] * len(bands)
    for i, (r0, r1) in enumerate(bands):
        if i > 0:
            cv2.line(t_risk, (0, int(r0 * k)), (tile_w, int(r0 * k)), (255, 255, 255), 1)
    # Green frame around the most urgent region.
    r0, r1 = bands[best]
    cv2.rectangle(t_risk, (3, int(r0 * k) + 3), (tile_w - 4, int(r1 * k) - 3), (0, 255, 0), 3)
    # numbers on screen: per region risk + distance (m) of the most dangerous pixel, marked on both tiles
    # For each region: find the pixel with the highest risk, mark it on the risk and depth tiles with its distance in meters.
    # If the region's maximum risk is below 0.15 it is shown as 'clear'.
    for i, (b0, b1) in enumerate(bands):
        sub = risk_map[b0:b1]
        if sub.size == 0:
            continue
        yy, xx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        dist = float(depth[b0 + yy, xx])
        px, py = int(xx * tile_w / W), int((b0 + yy) * k)
        for tile in ((t_risk, t_depth) if float(sub.max()) > 0.15 else ()):
            cv2.circle(tile, (px, py), 9, (255, 255, 255), 2)
            cv2.circle(tile, (px, py), 3, (0, 255, 0) if i == best else (255, 255, 255), -1)
            _txt(tile, f"{dist:.1f} m", (min(px + 12, tile_w - 110), max(py - 8, 50)), 0.8, (255, 255, 0), 2)
        _txt(t_risk, f"{zone_names[i]}: U={urgency[i]:.2f}" + (f"  {dist:.0f} m" if float(sub.max()) > 0.15 else "  clear"), (10, int(b0 * k) + 60 if i else 64), 0.8,
             (0, 255, 0) if i == best else (255, 255, 255), 2)
    _txt(t_risk, "CAMERA + RISK (bright = dangerous)", (10, 28))
    _txt(t_seg, "SEGMENTATION (SegFormer)", (10, 28))
    _txt(t_depth, f"DEPTH (UniDepthV2)  red = near   {lo:.0f}-{hi:.0f} m", (10, 28))

    # decision panel
    # Decision panel: level LOW / MEDIUM / HIGH from the winning urgency (above 0.6 = HIGH, above 0.3 = MEDIUM), one bar per region.
    t_dec = np.full((tile_h, tile_w, 3), 25, np.uint8)
    _txt(t_dec, "DECISION", (20, 40), 0.9)
    danger = urgency[best]
    level = "HIGH" if danger > 0.6 else ("MEDIUM" if danger > 0.3 else "LOW")
    colr = (230, 60, 60) if level == "HIGH" else ((240, 170, 40) if level == "MEDIUM" else (60, 200, 90))
    _txt(t_dec, f"most urgent: {zone_names[best].upper()}", (20, 100), 1.2, (255, 255, 255), 3)
    _txt(t_dec, f"risk {level}  (U = {danger:.2f})", (20, 150), 0.95, colr, 2)
    bar_x, bar_w, y = 20, tile_w - 60, 200
    for i in range(len(urgency)):
        j = len(urgency) - 1 - i                                         # near on top of the list
        u = float(urgency[j])
        _txt(t_dec, f"{zone_names[j]}", (bar_x, y + 24), 0.8)
        cv2.rectangle(t_dec, (bar_x + 90, y), (bar_x + 90 + bar_w - 90, y + 34), (70, 70, 70), -1)
        c = (0, 220, 90) if j == best else (150, 150, 150)
        cv2.rectangle(t_dec, (bar_x + 90, y), (bar_x + 90 + int((bar_w - 90) * min(u, 1.0)), y + 34), c, -1)
        _txt(t_dec, f"{u:.2f}", (bar_x + 100, y + 25), 0.7)
        y += 55
    _txt(t_dec, f"{fps:.1f} FPS  seg+depth+risk", (20, tile_h - 20), 0.7, (200, 200, 200), 1)

    # Join the four tiles into the 2 x 2 picture.
    top = np.concatenate([t_risk, t_seg], axis=1)
    bot = np.concatenate([t_depth, t_dec], axis=1)
    return np.concatenate([top, bot], axis=0)


# Earlier 3-tile layout, kept for reference (not used any more).
def make_panel_old(rgb, seg, depth, risk_map, urgency, best, bands, names, color_table, text, max_depth, tile_w=640):
    import cv2
    H, W = rgb.shape[:2]
    tile_h = int(round(tile_w * H / W))

    seg_vis = cv2.addWeighted(rgb, 0.5, color_table[seg], 0.5, 0)
    d = (255 * (1 - np.clip(depth, 0, max_depth) / max_depth)).astype(np.uint8)       # near = warm
    depth_vis = cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heat = cv2.cvtColor(cv2.applyColorMap((255 * np.clip(risk_map, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)
    a = (0.2 + 0.7 * np.clip(risk_map, 0, 1))[..., None]
    risk_vis = (rgb * (1 - a) + heat * a).astype(np.uint8)

    scale = tile_h / H
    for i, (r0, r1) in enumerate(bands):                                              # region lines + urgency
        k = tile_h / H
        if i > 0:
            cv2.line(risk_vis, (0, int(r0 * k)), (W, int(r0 * k)), (255, 255, 255), max(1, int(k * 2)))
    tiles = []
    for title, arr in (("segmentation", seg_vis), ("depth (red = near, blue = far)", depth_vis),
                       ("risk map  (bright = dangerous)", risk_vis)):
        tile = cv2.resize(arr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    risk_tile = tiles[2]
    for i, (r0, r1) in enumerate(bands):                                              # urgency numbers on the risk tile
        y_mid = int((r0 + r1) / 2 * scale)
        label = f"U={urgency[i]:.3f}"
        cv2.putText(risk_tile, label, (tile_w - 110, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(risk_tile, label, (tile_w - 110, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    r0, r1 = bands[best]
    cv2.rectangle(risk_tile, (2, int(r0 * scale) + 2), (tile_w - 3, int(r1 * scale) - 2), (0, 255, 0), 2)

    header = Image.new("RGB", (tile_w * 3, 54), (20, 20, 20))
    draw = ImageDraw.Draw(header)
    draw.text((8, 6), text[0], fill=(255, 255, 255))
    draw.text((8, 28), text[1], fill=(180, 180, 180))
    canvas = Image.new("RGB", (tile_w * 3, tile_h + 54))
    canvas.paste(header, (0, 0))
    for i, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t), (tile_w * i, 54))
    return np.array(canvas)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# All settings. Defaults: our fine-tuned SegFormer, metric depth, the fitted rule (max risk per region), 3 regions.
# The depth model default is the hub model; pass --depth_model runs\depth_s\best.pt for our fine-tuned one.
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "dataset" / "CAM_FRONT"), help="image folder, image or video file")
    ap.add_argument("--log", default="", help="only frames whose file name contains this text (one drive)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=40, help="number of frames (0 = all)")
    best = HERE / "runs" / "seg_b2_native" / "best"
    ap.add_argument("--seg_model", default=str(best) if best.exists() else "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--depth_model", default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--camera", default="auto", choices=["auto", "known", "unknown"],
                    help="known = real nuScenes intrinsics (only for your dataset frames); auto = use them when available")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--resize_height", type=int, default=0, help="shrink frames to this height first (0 = original size)")
    ap.add_argument("--depth_mode", default="metric", choices=["paper", "metric"],
                    help="metric (default, what the decision rule was fitted with) or paper")
    ap.add_argument("--rule", default="fitted", choices=["fitted", "paper"],
                    help="fitted: highest risk per region, top region weighted 0.6 (see risk.py); paper: plain mean, alphas 0.4..1")
    ap.add_argument("--lam", type=float, default=rk.DEFAULT_LAMBDA_PAPER, help="lambda for --depth_mode paper")
    ap.add_argument("--lam_metric", type=float, default=rk.DEFAULT_LAMBDA_METRIC, help="lambda (1/m) for --depth_mode metric")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--n_regions", type=int, default=rk.DEFAULT_REGIONS)
    ap.add_argument("--alphas", type=float, nargs="*", default=None, help="one weight per region, top (far) to bottom (near)")
    ap.add_argument("--weights", type=Path, default=None, help="JSON file {class name: weight}; other classes = background")
    ap.add_argument("--background", type=float, default=rk.BACKGROUND_WEIGHT)
    ap.add_argument("--seg_dir", type=Path, default=HERE / "seg_gt", help="folder with classes.json")
    ap.add_argument("--out", type=Path, default=HERE / "demo_out")
    ap.add_argument("--video", action="store_true", help="also write demo.mp4")
    ap.add_argument("--show", action="store_true", help="live window while it runs (press q to stop)")
    ap.add_argument("--no_frames", action="store_true", help="do not save a JPG per frame (faster)")
    ap.add_argument("--video_fps", type=float, default=4.0)
    ap.add_argument("--crop_top", type=float, default=0.0, help="cut this fraction off the top (mirror, windshield frame)")
    ap.add_argument("--max_panel_h", type=int, default=980, help="shrink the picture so it fits your screen")
    ap.add_argument("--crop_bottom", type=float, default=0.0, help="cut this fraction off the bottom (own car hood/dashboard), e.g. 0.2")
    return ap.parse_args()


def main():
    args = parse_args()
    device = ek.get_device()
    use_amp = device.type == "cuda"
    # Class names, a colour for each class and the class weight vector from risk.py.
    names = [c["name"] for c in json.load(open(args.seg_dir / "classes.json"))["classes"]]
    color_table = make_color_table(len(names))
    weights = json.load(open(args.weights)) if args.weights else None
    w_vec = rk.class_weight_vector(names, weights, args.background)
    print("Class weights:", {n: round(float(w), 2) for n, w in zip(names, w_vec) if w != args.background} or "all background")
    print(f"Background weight {args.background} | depth mode: {args.depth_mode} | regions: {args.n_regions} | decision rule: {args.rule}")

    print(f"Loading segmentation model: {args.seg_model}")
    # Load both models once, before the frame loop.
    processor, seg_model = load_labeler(args.seg_model, device)
    if len(seg_model.config.id2label) != len(names):
        sys.exit("ERROR: segmentation model and classes.json have different numbers of classes")
    print(f"Loading depth model: {args.depth_model}")
    depth_model = load_depth_model(args.depth_model, device)
    # Real camera intrinsics are used only for nuScenes frames (not for other videos).
    meta = json.load(open(args.meta_cache)) if (args.camera != "unknown" and args.meta_cache.exists()) else None

    frame_dir = args.out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    zone_names = rk.region_names(args.n_regions)
    # One timer per stage; the first 3 frames (warm-up) are ignored in the averages.
    t_seg, t_depth, t_risk, t_all = ek.Timer(3), ek.Timer(3), ek.Timer(3), ek.Timer(3)
    rows, writer, warned = [], None, False

    # MAIN LOOP: for every frame -> segmentation -> depth -> risk map -> region decision -> picture.
    for i, (stem, img) in enumerate(tqdm(iter_frames(args.input, args.log, args.every, args.limit), desc="frames")):
        K = None
        if args.camera != "unknown":
            K = intrinsics_for(stem, meta)
            if K is None and args.camera == "known" and not warned:
                print("WARNING: no intrinsics for this input; the depth model will guess the camera.")
                warned = True
        W0, H0 = img.size
        # Optional shrink (e.g. height 320 for speed). The intrinsics are scaled by the same factors.
        if args.resize_height and args.resize_height != H0:
            h = args.resize_height
            w = int(round(W0 * h / H0 / 2) * 2)
            img = img.resize((w, h), Image.BILINEAR)
            if K is not None:
                K = np.array(K, dtype=np.float64)
                K[0] *= w / W0
                K[1] *= h / H0
        # Optional crop (fractions of the height) to hide your own car's dashboard / hood or the sky.
        if args.crop_bottom > 0 or args.crop_top > 0:
            img = img.crop((0, int(round(img.height * args.crop_top)), img.width, int(round(img.height * (1 - args.crop_bottom)))))
            if K is not None:
                K = np.array(K, dtype=np.float64)
        rgb = np.array(img)

        # Segmentation, then depth, timed separately (GPU synchronised so the times are real).
        ek.cuda_sync(); t0 = time.time()
        seg = predict_with_retry(processor, seg_model, img, device, use_amp, img.height)
        ek.cuda_sync(); t1 = time.time()
        depth = run_depth(depth_model, rgb, None if K is None else np.asarray(K).tolist())
        ek.cuda_sync(); t2 = time.time()
        # Make the depth map the same size as the segmentation.
        if depth.shape != seg.shape:
            import cv2
            depth = cv2.resize(depth, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Fusion: per-pixel risk from class weights and depth (risk.py).
        risk_map = rk.compute_risk(seg, depth, w_vec, args.depth_mode, args.lam, args.lam_metric, args.max_depth)
        # Region decision: 'fitted' = highest risk per region with alphas 0.6 / 0.8 / 1.0 (the demo default);
        # 'paper' = plain mean per region with alphas 0.4 / 0.7 / 1.0.
        if args.rule == "fitted":
            urgency, best, bands = rk.region_urgency(
                risk_map, args.n_regions, args.alphas if args.alphas is not None else rk.fitted_alphas(args.n_regions), rk.FITTED_P)
        else:
            urgency, best, bands = rk.region_urgency(risk_map, args.n_regions, args.alphas)
        t3 = time.time()
        t_seg.add(t1 - t0); t_depth.add(t2 - t1); t_risk.add(t3 - t2); t_all.add(t3 - t0)

        # Numbers shown in the header line and saved in the CSV.
        mean_r, max_r = float(risk_map.mean()), float(risk_map.max())
        text = (f"{stem[-40:]}   |   mean risk {mean_r:.3f}   max risk {max_r:.2f}   |   most urgent: "
                f"{zone_names[best]} (U={urgency[best]:.3f})   |   camera: {'known' if K is not None else 'guessed'}",
                f"segmentation {1000 * (t1 - t0):.0f} ms   depth {1000 * (t2 - t1):.0f} ms   risk {1000 * (t3 - t2):.0f} ms   "
                f"total {1000 * (t3 - t0):.0f} ms ({1 / (t3 - t0):.1f} FPS)   |   depth mode: {args.depth_mode}")
        # Draw the picture, shrunk if it is taller than --max_panel_h so it fits the screen.
        panel = make_panel(rgb, seg, depth, risk_map, urgency, best, bands, names, color_table, text, args.max_depth,
                           zone_names=zone_names, fps=1.0 / max(t3 - t0, 1e-6))
        if panel.shape[0] > args.max_panel_h:
            import cv2
            s = args.max_panel_h / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * s) // 2 * 2, args.max_panel_h // 2 * 2), interpolation=cv2.INTER_AREA)
        if not args.no_frames:
            Image.fromarray(panel).save(frame_dir / f"{i:05d}_{stem[-30:]}.jpg", quality=88)
        # Live window; press q in it to stop.
        if args.show:
            import cv2
            cv2.imshow("risk estimation (q = stop)", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write the picture as one frame of demo.mp4.
        if args.video:
            import cv2
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
                                         (panel.shape[1], panel.shape[0]))
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        # One CSV row per frame: risk values, most urgent region, urgency per region and the time of each stage.
        rows.append([i, stem, mean_r, max_r, best] + [float(u) for u in urgency] +
                    [1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t3 - t0)])

    if writer is not None:
        writer.release()
    if not rows:
        sys.exit("No frames were processed (check --input / --log).")
    # Save risk_per_frame.csv, then print the timing summary.
    with open(args.out / "risk_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "frame", "mean_risk", "max_risk", "most_urgent_region"] +
                   [f"U_{n}" for n in zone_names] + ["seg_ms", "depth_ms", "risk_ms", "total_ms"])
        w.writerows(rows)

    print(f"\nProcessed {len(rows)} frames. Pictures: {frame_dir}   CSV: {args.out / 'risk_per_frame.csv'}"
          + (f"   Video: {args.out / 'demo.mp4'}" if args.video else ""))
    for name, t in (("segmentation", t_seg), ("depth", t_depth), ("risk formula", t_risk), ("whole pipeline", t_all)):
        s = t.summary()
        if s:
            print(f"  {name:<15s} {s['mean_ms']:7.1f} ± {s['std_ms']:5.1f} ms   p95 {s['p95_ms']:7.1f} ms")
    s = t_all.summary()
    if s:
        print(f"  => about {s['fps']:.1f} FPS end to end, peak GPU memory {ek.peak_gpu_gb():.2f} GB "
              f"(frame size {W0}x{H0}{' -> resized to height ' + str(args.resize_height) if args.resize_height else ''})")
    counts = np.bincount([r[4] for r in rows], minlength=args.n_regions)
    print("  most urgent region over all frames: " + ", ".join(f"{n} {c}" for n, c in zip(zone_names, counts)))


if __name__ == "__main__":
    main()
