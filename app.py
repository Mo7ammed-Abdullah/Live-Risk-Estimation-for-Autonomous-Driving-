"""
Web app for the vision-only driving-risk system (SegFormer-B2 + UniDepthV2-small + fixed risk fusion + time-to-collision).

Anyone opening the page can upload an IMAGE (risk map, most urgent region, objects with distance) or a short VIDEO
(the same per frame, plus time-to-collision per object). It reuses the project's own code (code/risk.py, code/ttc_tracker.py,
the model loaders and the picture from code/demo_risk.py), so the results match run.py.

Run on your laptop (venv311 active, project folder):
    python -m pip install gradio==6.31.0 imageio imageio-ffmpeg
    python app.py                 # opens http://127.0.0.1:7860
    python app.py --share         # also prints a temporary public https://....gradio.live link (up to 72 h, while it runs)
Live version: https://huggingface.co/spaces/Mohammed-Abdullah/Driving-risk-estimation (ZeroGPU; started automatically).

Weights are looked for in this order:  SEG_MODEL / DEPTH_MODEL environment variables, then weights/ (Space layout),
then runs/ (project layout). If none exist (e.g. a copy from GitHub), weights/ is downloaded once from the Space.

Licence: UniDepth is CC BY-NC 4.0 -> this demo is for non-commercial research use only.
"""
import argparse
import csv
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
if (HERE / "UniDepth").is_dir():                     # bundled UniDepth package (Space layout)
    sys.path.insert(0, str(HERE / "UniDepth"))
try:                                                  # UniDepth imports wandb only for training logs
    import wandb  # noqa: F401
except ImportError:
    import types
    sys.modules["wandb"] = types.ModuleType("wandb")

import gradio as gr  # noqa: E402

import risk as rk  # noqa: E402
from ttc_tracker import TTCTracker  # noqa: E402

# ZeroGPU on Hugging Face: a shared GPU is lent to the app only while a request runs. Each request asks for a time
# budget; visitors have a small daily quota (2 min without a Hugging Face login), so the budgets are kept short.
ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))


def video_gpu_seconds(video=None, max_seconds=15, proc_fps=8, *args, **kwargs):
    """GPU time to ask for: about 0.15 s per analysed frame plus start-up, at most 2 minutes."""
    try:
        frames = float(max_seconds) * float(proc_fps)
    except (TypeError, ValueError):
        frames = 120
    return int(min(120, 15 + 0.15 * frames))


try:
    import spaces
    GPU_IMAGE = spaces.GPU(duration=20)
    GPU_VIDEO = spaces.GPU(duration=video_gpu_seconds)
except Exception:                                     # not on a Space, or no ZeroGPU: no effect
    def GPU_IMAGE(fn):
        return fn
    GPU_VIDEO = GPU_IMAGE


def first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None


SEG_MODEL = os.environ.get("SEG_MODEL") or first_existing(HERE / "weights" / "seg_b2_native", HERE / "runs" / "seg_b2_native" / "best")
DEPTH_MODEL = os.environ.get("DEPTH_MODEL") or first_existing(HERE / "weights" / "depth_s.pt", HERE / "runs" / "depth_s" / "best.pt")
RESIZE_HEIGHT = int(os.environ.get("RESIZE_HEIGHT", "320"))        # same as run.py
MAX_DEPTH = 80.0

# The trained weights are too large for GitHub; a copy without them downloads them once from the Hugging Face Space.
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "Mohammed-Abdullah/Driving-risk-estimation")


def download_weights():
    """Fetch weights/ (about 245 MB) from the Hugging Face Space into this folder and return the two model paths."""
    from huggingface_hub import snapshot_download
    print(f"Trained weights not found here: downloading them once from huggingface.co/spaces/{WEIGHTS_REPO} ...")
    try:
        snapshot_download(WEIGHTS_REPO, repo_type="space", allow_patterns=["weights/*", "weights/**/*"], local_dir=str(HERE))
    except Exception as e:
        raise gr.Error(f"Could not download the trained weights ({e}). Put them in weights/ or set SEG_MODEL and DEPTH_MODEL.")
    seg, depth = HERE / "weights" / "seg_b2_native", HERE / "weights" / "depth_s.pt"
    if not (seg.exists() and depth.exists()):
        raise gr.Error("Download finished but weights/seg_b2_native or weights/depth_s.pt is missing.")
    return str(seg), str(depth)


# ----------------------------------------------------------------------------- models (loaded once)
_models, _lock = {}, threading.Lock()


def models():
    """Load both networks once; later calls reuse them."""
    with _lock:
        if not _models:
            import torch
            from eval_depth import load_depth_model
            from generate_seg_gt import load_labeler, make_color_table
            seg_path, depth_path = SEG_MODEL, DEPTH_MODEL
            if seg_path is None or depth_path is None:
                seg_path, depth_path = download_weights()
            # On ZeroGPU the models are placed on "cuda" at start-up (emulated until a request gets the real GPU).
            device = torch.device("cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu")
            processor, seg_model = load_labeler(seg_path, device)
            id2label = seg_model.config.id2label
            names = [str(id2label[i]).lower() for i in sorted(int(k) for k in id2label)]
            _models.update(device=device, processor=processor, seg=seg_model, depth=load_depth_model(depth_path, device),
                           names=names, colors=make_color_table(len(names)), w_vec=rk.class_weight_vector(names))
            print(f"Models loaded on {device}: {seg_path} | {depth_path}")
        return _models


def on_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ----------------------------------------------------------------------------- one frame
def analyse(img, tracker, t):
    """Run segmentation + depth + risk (+ tracker) on one PIL image. Returns the 2x2 picture and the numbers."""
    from demo_risk import make_panel
    from eval_depth import run_depth
    from generate_seg_gt import predict_with_retry
    m = models()
    img = img.convert("RGB")
    if RESIZE_HEIGHT and img.height != RESIZE_HEIGHT:
        w = int(round(img.width * RESIZE_HEIGHT / img.height / 2) * 2)
        img = img.resize((w, RESIZE_HEIGHT), Image.BILINEAR)
    rgb = np.array(img)
    t0 = time.time()
    seg = predict_with_retry(m["processor"], m["seg"], img, m["device"], m["device"].type == "cuda", img.height)
    seg = np.minimum(seg, len(m["names"]) - 1)
    depth = np.nan_to_num(run_depth(m["depth"], rgb, None), nan=MAX_DEPTH, posinf=MAX_DEPTH)   # camera unknown
    if depth.shape != seg.shape:
        import cv2
        depth = cv2.resize(depth, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_LINEAR)
    risk_map = rk.compute_risk(seg, depth, m["w_vec"], "metric", rk.DEFAULT_LAMBDA_PAPER, rk.DEFAULT_LAMBDA_METRIC, MAX_DEPTH)
    urgency, best, bands = rk.region_urgency(risk_map, rk.DEFAULT_REGIONS, rk.fitted_alphas(rk.DEFAULT_REGIONS), rk.FITTED_P)
    tracks = tracker.update(seg, depth, t)
    dt = time.time() - t0
    zones = rk.region_names(rk.DEFAULT_REGIONS)
    panel = make_panel(rgb, seg, depth, risk_map, urgency, best, bands, m["names"], m["colors"], "", MAX_DEPTH,
                       zone_names=zones, fps=1.0 / max(dt, 1e-6), tracks=tracks, clean=True)
    u = float(urgency[best])
    level = "HIGH" if u > 0.6 else ("MEDIUM" if u > 0.3 else "LOW")
    return panel, {"zone": zones[best], "level": level, "urgency": u, "tracks": tracks, "seconds": dt}


# ----------------------------------------------------------------------------- result widgets (HTML)
ZONE_TEXT = {"near": "right in front of the car", "mid": "at middle distance", "far": "far ahead"}
LEVEL_TEXT = {"HIGH": "High risk", "MEDIUM": "Medium risk", "LOW": "Low risk"}


def sign(level, title, detail):
    """The result, drawn as a road sign: green = low, amber = medium, red = high, grey = waiting."""
    return (f'<div class="sign sign-{level}" role="status"><div class="sign-face">'
            f'<div class="sign-title">{title}</div><div class="sign-detail">{detail}</div></div></div>')


def stats(items):
    return '<dl class="stats">' + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items) + "</dl>"


def object_list(tracks, limit=12):
    """Objects nearest first, with a bar showing distance on the 0-80 m scale."""
    if not tracks:
        return '<p class="muted">No cars, people or two-wheelers found.</p>'
    rows = []
    for o in sorted(tracks, key=lambda o: o["depth"])[:limit]:
        d = float(o["depth"])
        tone = "near" if d < 10 else ("mid" if d < 25 else "far")
        rows.append(f'<li><span class="obj-kind">{o["kind"].capitalize()}</span>'
                    f'<span class="obj-bar"><span class="obj-fill obj-{tone}" style="width:{max(3, 100 - d / MAX_DEPTH * 100):.0f}%"></span></span>'
                    f'<span class="obj-dist">{d:.1f} m</span></li>')
    more = len(tracks) - limit
    tail = f'<p class="muted">and {more} more</p>' if more > 0 else ""
    return '<ul class="objects">' + "".join(rows) + "</ul>" + tail


def device_name():
    return "GPU" if on_gpu() else "CPU"


TIPS = """
<div class="tips">
  <p class="tips-title">What to upload</p>
  <ul>
    <li class="ok">A dashcam or windshield camera looking straight ahead at the road.</li>
    <li class="ok">Only the road scene in the picture, from edge to edge.</li>
    <li class="no">No dashboard, bonnet (hood), wipers, mirror or car interior in the frame. Crop them out first.</li>
  </ul>
  <p class="tips-why">Why: the models learned from clean front-camera images. Parts of your own car look like an
  object right in front and give a false high risk.</p>
</div>
"""

WAIT_IMAGE = sign("wait", "Waiting for a photo", "Upload a clear view of the road ahead, with no part of your own car in it, then press Check risk.")
WAIT_VIDEO = sign("wait", "Waiting for a video", "Upload a short dashcam clip of the road ahead, with no dashboard or bonnet in view, then press Check video.")


# ----------------------------------------------------------------------------- image tab
@GPU_IMAGE
def run_image(image):
    if image is None:
        raise gr.Error("No photo yet. Upload a photo taken from the driver's seat, then press Check risk.")
    panel, info = analyse(image, TTCTracker(models()["names"]), 0.0)
    verdict = sign(info["level"].lower(), LEVEL_TEXT[info["level"]],
                   f"The most urgent area is {ZONE_TEXT.get(info['zone'], info['zone'])}.")
    verdict += stats([("Urgency", f"{info['urgency']:.2f}"), ("Objects", len(info["tracks"])),
                      ("Time", f"{info['seconds']:.1f} s on {device_name()}")])
    return panel, verdict, object_list(info["tracks"])


# ----------------------------------------------------------------------------- video tab
@GPU_VIDEO
def run_video(video, max_seconds, proc_fps, progress=gr.Progress()):
    import cv2
    import imageio.v2 as imageio
    if not video:
        raise gr.Error("No video yet. Upload a short dashcam clip, then press Check video.")
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / float(proc_fps))))
    last = int(min(total if total > 0 else 10 ** 9, float(max_seconds) * src_fps))
    n_out = max(1, (last + step - 1) // step)
    models()
    tracker = TTCTracker(models()["names"])
    out_dir = Path(tempfile.mkdtemp(prefix="risk_"))
    out_mp4, out_csv = out_dir / "risk_ttc.mp4", out_dir / "ttc_per_object.csv"
    writer, rows, done, shortest, t_start = None, [], 0, None, time.time()
    levels = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    idx = 0
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / src_fps
            panel, info = analyse(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), tracker, t)
            levels[info["level"]] += 1
            if writer is None:
                writer = imageio.get_writer(str(out_mp4), fps=src_fps / step, codec="libx264", quality=7,
                                            pixelformat="yuv420p", macro_block_size=2)
            writer.append_data(panel)
            for o in info["tracks"]:
                rows.append([round(t, 2), o["id"], o["kind"], round(o["depth"], 2),
                             "" if o["closing"] is None else round(o["closing"], 2),
                             "" if o["ttc"] is None else round(o["ttc"], 2)])
                if o["ttc"] is not None and (shortest is None or o["ttc"] < shortest[0]):
                    shortest = (o["ttc"], o["kind"], t)
            done += 1
            progress(done / n_out, desc=f"Frame {done} of {n_out}")
        idx += 1
    cap.release()
    if writer is None:
        raise gr.Error("Could not read any frame from this video. Try an MP4 file.")
    writer.close()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "track_id", "kind", "distance_m", "closing_speed_m_s", "ttc_s"])
        w.writerows(rows)
    took = time.time() - t_start
    if shortest is None:
        verdict = sign("low", "No object closing in", "Nothing was measured approaching the car in this clip.")
    else:
        lvl = "high" if shortest[0] < 2 else ("medium" if shortest[0] < 4 else "low")
        verdict = sign(lvl, f"Closest call: {shortest[0]:.1f} s",
                       f"A {shortest[1]} {shortest[2]:.1f} s into the clip would be reached in {shortest[0]:.1f} s at that speed.")
    verdict += stats([("High-risk frames", f"{levels['HIGH']} of {done}"), ("Frames checked", done),
                      ("Time", f"{took:.0f} s on {device_name()}")])
    return str(out_mp4), verdict, str(out_csv)


# ----------------------------------------------------------------------------- page
HEADER = """
<header class="hero">
  <div class="hero-text">
    <h1>How risky is this road scene?</h1>
    <p>One camera, no radar or LiDAR. The app finds cars, people and bikes, estimates how far away each one is,
    and tells you which part of the road needs attention first. In a video it also estimates the
    <b>time to collision</b>: how many seconds until an approaching object would be reached.</p>
  </div>
  <ul class="legend" aria-label="Colour key">
    <li><span class="dot dot-high"></span>High risk, or collision in under 2 s</li>
    <li><span class="dot dot-medium"></span>Medium risk, or under 4 s</li>
    <li><span class="dot dot-low"></span>Low risk</li>
  </ul>
</header>
"""

FOOTER = """
<section class="how">
  <h2>How it works</h2>
  <ol>
    <li><b>Label every pixel.</b> SegFormer-B2, fine-tuned on nuScenes front-camera images, marks road, cars, people, bikes and so on.</li>
    <li><b>Measure distance.</b> UniDepthV2, fine-tuned on nuScenes LiDAR, estimates each pixel's distance in metres.</li>
    <li><b>Combine.</b> Risk = class weight &times; e<sup>&minus;0.05 &times; distance</sup>. People count most, then bikes, then vehicles;
    nearer means riskier. The road is split into far, middle and near areas, and the most urgent one is reported.</li>
  </ol>
  <p class="fine">On the nuScenes test set the app picks the right area 89.7% of the time (depth error 7.1%).
  Distances from other cameras are approximate because the camera's calibration is unknown.
  Research demo for non-commercial use only (UniDepth is CC BY-NC 4.0). Uploads are processed and not kept.</p>
</section>
"""

CSS = """
:root { --ink:#1d2b36; --muted:#5b6b77; --line:#d3dae0; --page:#e9edf0; --panel:#ffffff;
        --go:#0a6b4b; --warn:#f2b200; --stop:#c3271f; --idle:#7c8a95; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { font-variant-numeric: tabular-nums; }
.hero { display:flex; gap:32px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;
        padding: 28px 4px 20px; border-bottom: 3px solid var(--ink); margin-bottom: 8px; }
.hero-text { flex: 1 1 560px; }
.hero-text p { max-width: 68ch; }
.hero h1 { font-size: 2.6rem; line-height: 1.05; font-weight: 800; letter-spacing: -0.01em; margin: 0 0 12px; color: var(--ink); }
.hero p { font-size: 1.02rem; line-height: 1.55; color: var(--muted); margin: 0; }
.legend { list-style:none; margin:0; padding:0; display:grid; gap:6px; font-size:.92rem; color: var(--ink); }
.legend li { display:flex; align-items:center; gap:10px; }
.dot { width:14px; height:14px; border-radius:3px; display:inline-block; }
.dot-high { background: var(--stop); } .dot-medium { background: var(--warn); } .dot-low { background: var(--go); }

/* the road sign: the one loud element on the page */
.sign { border-radius: 14px; padding: 6px; background: var(--idle); color:#fff; }
.sign-face { border: 3px solid rgba(255,255,255,.92); border-radius: 10px; padding: 18px 22px 20px; }
.sign .sign-title { font-size: 2.1rem; font-weight: 800; line-height: 1.1; color: inherit !important; }
.sign .sign-detail { font-size: 1.05rem; line-height: 1.4; margin-top: 6px; color: inherit !important; }
.sign, .sign * { color: #ffffff; }
.sign-medium, .sign-medium * { color: #1d2b36 !important; }
.sign-high { background: var(--stop); } .sign-low { background: var(--go); }
.sign-medium { background: var(--warn); }
.sign-medium .sign-face { border-color: #1d2b36; }
.sign-wait .sign-title { font-size: 1.5rem; }
@media (prefers-reduced-motion: no-preference) { .sign:not(.sign-wait) { animation: flip .35s ease-out; } }
@keyframes flip { from { transform: perspective(600px) rotateX(-70deg); opacity:0; } to { transform:none; opacity:1; } }

.stats { display:flex; flex-wrap:wrap; gap: 8px 28px; margin: 14px 2px 0; }
.stats div { min-width: 90px; }
.stats dt { font-size:.8rem; color: var(--muted); }
.stats dd { margin:0; font-size:1.15rem; font-weight:700; color: var(--ink); }

.objects { list-style:none; padding:0; margin: 0; display:grid; gap:8px; }
.objects li { display:grid; grid-template-columns: 110px 1fr 64px; align-items:center; gap:12px; font-size:.95rem; color: var(--ink); }
.obj-bar { height: 8px; background: #e3e8ec; border-radius: 4px; overflow:hidden; }
.obj-fill { display:block; height:100%; border-radius: 4px; }
.obj-near { background: var(--stop); } .obj-mid { background: var(--warn); } .obj-far { background: #8fa3b1; }
.obj-dist { text-align:right; font-weight:700; }
.muted { color: var(--muted); font-size: .92rem; margin: 6px 0 0; }
.section-note { color: var(--muted); font-size: .9rem; margin: 2px 0 0; }
.tips { margin-bottom: 4px; border-left: 4px solid var(--warn); background: var(--panel); padding: 12px 16px; border-radius: 0 6px 6px 0; color: var(--ink); }
.tips .tips-title { font-weight: 800; margin: 0 0 6px; font-size: 1rem; }
.tips ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: .93rem; line-height: 1.45; }
.tips li { padding-left: 24px; position: relative; }
.tips li::before { position: absolute; left: 0; top: 0; font-weight: 800; }
.tips li.ok::before { content: "✓"; color: var(--go); }
.tips li.no::before { content: "✕"; color: var(--stop); }
.dark .tips li.ok::before { color: #4fd1a0; } .dark .tips li.no::before { color: #ff6b5e; }
.tips .tips-why { color: var(--muted); font-size: .85rem; margin: 8px 0 0; line-height: 1.45; }

.how { margin-top: 28px; padding: 22px 4px 8px; border-top: 1px solid var(--line); color: var(--ink); }
.how h2 { font-size: 1.25rem; margin: 0 0 10px; }
.how ol { margin: 0 0 12px 1.2em; padding:0; display:grid; gap:8px; max-width: 80ch; line-height:1.5; }
.how .fine { color: var(--muted); font-size: .86rem; max-width: 90ch; line-height: 1.5; }
button.primary { font-weight: 700 !important; font-size: 1.02rem !important; }

.dark { --ink:#e6ecf0; --muted:#9fb0bc; --line:#3a4651; --panel:#1f2830; }
.dark .hero { border-bottom-color: var(--ink); }
.dark .obj-bar { background:#33404b; }
.dark button[role=tab][aria-selected=true] { color: #4fd1a0 !important; }
@media (max-width: 640px) {
  .hero h1 { font-size: 1.9rem; } .sign-title { font-size: 1.6rem; }
  .objects li { grid-template-columns: 90px 1fr 56px; }
}
"""


def theme():
    return gr.themes.Base(
        font=[gr.themes.GoogleFont("Overpass"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill="#e9edf0", body_background_fill_dark="#151c22",
        block_background_fill="#ffffff", block_background_fill_dark="#1f2830",
        block_border_color="#d3dae0", block_border_color_dark="#3a4651",
        block_shadow="none", block_shadow_dark="none",
        button_primary_background_fill="#0a6b4b", button_primary_background_fill_hover="#08593e",
        button_primary_background_fill_dark="#12875f", button_primary_background_fill_hover_dark="#0f7752",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        slider_color="#0a6b4b", slider_color_dark="#12875f",
        color_accent="#0a6b4b", color_accent_soft="#d7ebe3",
        body_text_color="#1d2b36", body_text_color_subdued="#5b6b77",
    )


def build(max_seconds_default, max_seconds_limit, fps_default):
    with gr.Blocks(title="Driving risk from one camera") as demo:
        gr.HTML(HEADER)
        with gr.Tabs():
            with gr.Tab("Photo"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML(TIPS)
                        img_in = gr.Image(type="pil", label="Road ahead, from a dashcam or windshield camera", height=330)
                        img_btn = gr.Button("Check risk", variant="primary", size="lg")
                        gr.HTML('<p class="section-note">Time to collision needs movement, so it only appears for videos.</p>')
                    with gr.Column(scale=6):
                        img_verdict = gr.HTML(WAIT_IMAGE)
                        img_objects = gr.HTML(label="Objects, nearest first")
                img_out = gr.Image(label="What the model sees: objects, labels, distance and the risky area", interactive=False)
                img_btn.click(run_image, inputs=img_in, outputs=[img_out, img_verdict, img_objects])
            with gr.Tab("Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML(TIPS)
                        vid_in = gr.Video(label="Dashcam clip of the road ahead", height=330)
                        with gr.Accordion("Speed settings", open=False):
                            secs = gr.Slider(2, max_seconds_limit, value=max_seconds_default, step=1,
                                             label="Seconds of video to check")
                            fps = gr.Slider(1, 15, value=fps_default, step=1,
                                            label="Frames checked per second (more = smoother, slower)")
                        vid_btn = gr.Button("Check video", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        vid_verdict = gr.HTML(WAIT_VIDEO)
                        vid_csv = gr.File(label="Download time to collision per object (CSV)")
                vid_out = gr.Video(label="Result: boxes show time to collision (red under 2 s, orange under 4 s)",
                                   interactive=False)
                vid_btn.click(run_video, inputs=[vid_in, secs, fps], outputs=[vid_out, vid_verdict, vid_csv])
                examples = sorted(p for p in (HERE / "examples").glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi"}) \
                    if (HERE / "examples").is_dir() else []
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=vid_in, label="Or try an example clip")
        gr.HTML(FOOTER)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    ap.add_argument("--open", action="store_true", help="open the app in the web browser when it is ready")
    args, _ = ap.parse_known_args()
    if ZERO_GPU:
        models()                                      # ZeroGPU wants the models loaded on cuda before any request
        demo = build(max_seconds_default=15, max_seconds_limit=30, fps_default=8)
    elif on_gpu():
        demo = build(max_seconds_default=20, max_seconds_limit=120, fps_default=10)
    else:                                             # CPU: keep videos short and sparse (about a minute per request)
        demo = build(max_seconds_default=10, max_seconds_limit=20, fps_default=2)
    demo.queue(max_size=16, default_concurrency_limit=1)
    in_space = bool(os.environ.get("SPACE_ID"))
    demo.launch(share=args.share or os.environ.get("SHARE") == "1",
                server_name="0.0.0.0" if in_space else None, server_port=args.port,
                theme=theme(), css=CSS, inbrowser=args.open and not in_space)


if __name__ == "__main__":
    main()
