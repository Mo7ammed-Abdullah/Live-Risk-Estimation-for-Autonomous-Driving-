"""
Run the risk demo on one video. Edit the settings below, save, then run:

    python run.py

(venv311 active, from the project folder.)  Output goes to demo_out/ (demo.mp4 + risk_per_frame.csv, and ttc_per_object.csv when SHOW_TTC is on).
"""
import subprocess
import sys
from pathlib import Path

# ===================== EDIT THESE =====================
VIDEO_PATH = r"C:\Downloads\All testings\Testing 3\risk estimation for autonomous driving (Segformer+Unidepth)\demo_in\driving.mp4"   # <- put your video path here
CROP_TOP = 0.0        # fraction to cut from the top, 0 to 0.5 (e.g. 0.1 = top 10%)
CROP_BOTTOM = 0.0     # fraction to cut from the bottom (e.g. 0.2 = bottom 20%, hides your own dashboard/hood)
RESIZE_HEIGHT = 320   # smaller = faster (320 gives ~14 FPS)
EVERY = 5             # process 1 frame in N. 60 FPS video: 5 -> 12 frames per second (best for time-to-collision)
VIDEO_FPS = 12        # saved video speed = video FPS / EVERY (60 / 5 = 12)
LIMIT = 0             # 0 = whole video, or a number to test on the first N frames
SHOW_WINDOW = True    # live window; press q in it to stop
SAVE_FRAMES = False   # True also saves one JPG per frame (large)
SHOW_TTC = True       # find and follow objects and show their time-to-collision (red < 2 s, orange < 4 s)
CLEAN_CAMERA = True   # top-left picture: plain camera (True) or with risk colours and far/mid/near region boxes (False)
# ======================================================

# Folder where this file lives (the project root).
HERE = Path(__file__).resolve().parent

# Check the video exists, then call the demo script with the settings from above.
def main():
    if not Path(VIDEO_PATH).is_file():
        sys.exit(f"Video not found: {VIDEO_PATH}\nEdit VIDEO_PATH at the top of run.py")
    # sys.executable = the same Python (venv311) that is running this file; the demo uses our fine-tuned segmentation and depth models.
    cmd = [sys.executable, str(HERE / "code" / "demo_risk.py"),
           "--input", VIDEO_PATH, "--video",
           "--video_fps", str(VIDEO_FPS),
           "--resize_height", str(RESIZE_HEIGHT),
           "--every", str(EVERY),
           "--limit", str(LIMIT),
           "--crop_top", str(CROP_TOP),
           "--crop_bottom", str(CROP_BOTTOM),
           "--depth_model", str(HERE / "runs" / "depth_s" / "best.pt")]
    if SHOW_WINDOW:
        cmd.append("--show")
    if not SAVE_FRAMES:
        cmd.append("--no_frames")
    if SHOW_TTC:
        cmd.append("--ttc")
    if CLEAN_CAMERA:
        cmd.append("--clean_camera")
    # Run it from the project root so all relative paths work.
    subprocess.run(cmd, cwd=HERE, check=True)

if __name__ == "__main__":
    main()
