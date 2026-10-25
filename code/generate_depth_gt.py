"""
Generate sparse depth ground truth from nuScenes LiDAR (.bin) + front camera images.

Works with YOUR layout (no nuscenes-devkit needed):

    Downloads/All testings/Testing 3/dataset/CAM_FRONT/*.jpg
    Downloads/All testings/Testing 3/dataset/LIDAR_TOP/*.pcd.bin
    Downloads/v1.0-trainval_meta/v1.0-trainval/*.json     (metadata)

Output:
    Downloads/All testings/Testing 3/depth_gt/depth/CAM_FRONT/<image_name>.png
        16-bit PNG, pixel value = depth_in_meters * 256, 0 = no LiDAR measurement
    Downloads/All testings/Testing 3/depth_gt/vis/CAM_FRONT/<image_name>.jpg   (overlay, first N)

Read a depth map back:
    depth = np.array(Image.open(path)).astype(np.float32) / 256.0   # meters, 0 = invalid

Install:
    pip install numpy pillow tqdm ijson

Run (defaults already point at your folders):
    python generate_depth_gt.py --vis 20

The huge metadata files (sample_data.json 1.3 GB, ego_pose.json 630 MB) are STREAMED, not loaded
into RAM, and only the records for your files are kept. The result is cached in
<out>/meta_cache.json so later runs start instantly.
"""
import argparse
import bisect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Default paths (Path.home() = C:\Users\<you>, so no username needed)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)
DEFAULT_DATASET = HERE / "dataset"
DEFAULT_META = HERE / "v1.0-trainval_meta" / "v1.0-trainval"
DEFAULT_OUT = HERE / "depth_gt"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# Image names look like <drive>__<camera>__<timestamp>; split them into those three parts.
def parse_name(fname):
    """'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg' -> (log, channel, timestamp_us)"""
    parts = fname.split(".")[0].split("__")
    return parts[0], parts[1], int(parts[2])


# Turn a nuScenes rotation (quaternion w, x, y, z) into a 3x3 rotation matrix.
def quat_to_rot(q):
    """nuScenes quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# A camera image and a LiDAR sweep are never taken at the exact same instant.
# For each image, pick the LiDAR file from the same drive closest in time and keep it only if it is within max_dt.
def pair_files(cam_files, lidar_files, max_dt_us):
    """Pair each camera image with the LiDAR file of the same log closest in time."""
    by_log = defaultdict(list)
    for name in lidar_files:
        log, _, ts = parse_name(name)
        by_log[log].append((ts, name))
    for log in by_log:
        by_log[log].sort()

    pairs, dts = [], []
    for cam in cam_files:
        log, _, ts = parse_name(cam)
        lst = by_log.get(log)
        if not lst:
            continue
        stamps = [t for t, _ in lst]
        i = bisect.bisect_left(stamps, ts)
        cands = [j for j in (i - 1, i) if 0 <= j < len(lst)]
        j = min(cands, key=lambda k: abs(stamps[k] - ts))
        dt = abs(stamps[j] - ts)
        if dt <= max_dt_us:
            pairs.append((cam, lst[j][1]))
            dts.append(dt)
    return pairs, dts


# ----------------------------------------------------------------------------
# Metadata (streamed + cached)
# ----------------------------------------------------------------------------
# The nuScenes metadata files are huge, so they are streamed and only the records for our files are kept.
# The result is cached in meta_cache.json, so later runs start instantly.
def load_metadata(meta_dir, wanted_names, cache_path):
    wanted_names = set(wanted_names)

    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        if set(cache["scanned"]) == wanted_names:
            print(f"Loaded cached metadata from {cache_path}")
            return cache

    # Step 1: for each of our files, find which sensor calibration and vehicle pose belong to it.
    # 1) sample_data.json -> keep only records for files we actually have
    sd = {}
    path = meta_dir / "sample_data.json"
    print(f"Streaming {path.name} (this can take a few minutes the first time)...")
    with open(path, "rb") as f:
        for e in tqdm(ijson.items(f, "item", use_float=True), desc="sample_data.json", unit=" rec"):
            b = os.path.basename(e["filename"])
            if b in wanted_names:
                sd[b] = {
                    "calibrated_sensor_token": e["calibrated_sensor_token"],
                    "ego_pose_token": e["ego_pose_token"],
                    "width": e.get("width"),
                    "height": e.get("height"),
                }
                if len(sd) == len(wanted_names):
                    break

    need_cs = {r["calibrated_sensor_token"] for r in sd.values()}
    need_ego = {r["ego_pose_token"] for r in sd.values()}

    # Step 2: sensor calibration (position of the sensor on the car, camera intrinsics).
    # 2) calibrated_sensor.json (small)
    with open(meta_dir / "calibrated_sensor.json") as f:
        cs = {r["token"]: r for r in json.load(f) if r["token"] in need_cs}

    # Step 3: pose of the car in the world at the time each file was recorded.
    # 3) ego_pose.json -> stream, keep only needed poses
    ego = {}
    path = meta_dir / "ego_pose.json"
    print(f"Streaming {path.name}...")
    with open(path, "rb") as f:
        for e in tqdm(ijson.items(f, "item", use_float=True), desc="ego_pose.json", unit=" rec"):
            if e["token"] in need_ego:
                ego[e["token"]] = {"rotation": e["rotation"], "translation": e["translation"]}
                if len(ego) == len(need_ego):
                    break

    cache = {"scanned": sorted(wanted_names), "sd": sd, "cs": cs, "ego": ego}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"Cached metadata to {cache_path}")
    return cache


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------
# Project the 3D LiDAR points into the camera image to get a sparse depth map.
# The chain of transforms: LiDAR -> car -> world -> car at camera time -> camera -> image pixels.
# The car moved between the two recordings, which is why the world frame is used in between.
def project_lidar_to_camera(lidar_path, sd_l, sd_c, meta, min_d, max_d):
    """Return u, v, depth (1-D arrays) of LiDAR points that land inside the camera image."""
    # Each LiDAR point has 5 values (x, y, z, intensity, ring); keep x, y, z.
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64).T  # 3 x N

    cs_l = meta["cs"][sd_l["calibrated_sensor_token"]]
    pose_l = meta["ego"][sd_l["ego_pose_token"]]
    cs_c = meta["cs"][sd_c["calibrated_sensor_token"]]
    pose_c = meta["ego"][sd_c["ego_pose_token"]]

    # Helper: turn a 3-vector into a column so it can be added to a 3 x N point array.
    col = lambda x: np.asarray(x, dtype=np.float64)[:, None]

    p = quat_to_rot(cs_l["rotation"]) @ pts + col(cs_l["translation"])              # LiDAR -> ego (LiDAR time)
    p = quat_to_rot(pose_l["rotation"]) @ p + col(pose_l["translation"])            # ego -> global
    p = quat_to_rot(pose_c["rotation"]).T @ (p - col(pose_c["translation"]))        # global -> ego (camera time)
    p = quat_to_rot(cs_c["rotation"]).T @ (p - col(cs_c["translation"]))            # ego -> camera

    # Depth = distance along the camera's viewing axis. Keep only points in front of the camera within min/max range.
    depth = p[2]
    front = (depth > min_d) & (depth < max_d)
    p, depth = p[:, front], depth[front]

    # Camera intrinsics K turn 3D camera coordinates into pixel coordinates (u, v).
    K = np.asarray(cs_c["camera_intrinsic"], dtype=np.float64)
    uvw = K @ p
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    W = sd_c.get("width") or 1600
    H = sd_c.get("height") or 900
    # Keep only points that land inside the image.
    inside = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    u = np.round(u[inside]).astype(np.int64)
    v = np.round(v[inside]).astype(np.int64)
    return u, v, depth[inside].astype(np.float32), (H, W)


# Write the points into an image-sized depth map. If several points land on one pixel, keep the nearest.
def rasterize(u, v, d, shape):
    """Sparse depth map; if several points fall on one pixel keep the nearest. 0 = no measurement."""
    depth = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), d)
    depth[np.isinf(depth)] = 0.0
    return depth


# Depth is stored as a 16-bit PNG: pixel value = meters x 256, 0 = no LiDAR measurement.
def save_depth_png(depth, path):
    Image.fromarray(np.clip(depth * 256.0, 0, 65535).astype(np.uint16)).save(path)


# Debug picture: coloured dots (red near, green middle, blue far) drawn on the photo to check the alignment.
def save_overlay(img_path, u, v, d, out_path, max_d):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    t = np.clip(d / max_d, 0, 1)
    for x, y, tt in zip(u, v, t):  # near = red, mid = green, far = blue
        color = (int(255 * (1 - tt)), int(255 * (1 - abs(2 * tt - 1))), int(255 * tt))
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    img.save(out_path, quality=90)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="folder with CAM_FRONT/ and LIDAR_TOP/")
    ap.add_argument("--meta", type=Path, default=DEFAULT_META, help="folder with sample_data.json, ego_pose.json, ...")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--camera", default="CAM_FRONT", help="camera folder name inside --dataset")
    ap.add_argument("--min_depth", type=float, default=1.0)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--max_dt", type=float, default=0.05, help="max camera-LiDAR time gap in seconds")
    ap.add_argument("--vis", type=int, default=0, help="save overlay images for the first N frames")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pairs (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (args.meta / "sample_data.json").exists() and (args.meta / "v1.0-trainval" / "sample_data.json").exists():
        args.meta = args.meta / "v1.0-trainval"  # pointed one level too high

    cam_dir = args.dataset / args.camera
    lidar_dir = args.dataset / "LIDAR_TOP"
    for p in (cam_dir, lidar_dir, args.meta / "sample_data.json", args.meta / "ego_pose.json",
              args.meta / "calibrated_sensor.json"):
        if not p.exists():
            sys.exit(f"ERROR: not found: {p}\nUse --dataset / --meta to point to the right folders.")

    # List the camera and LiDAR files, then pair them up in time.
    cam_files = sorted(p.name for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    lidar_files = sorted(p.name for p in lidar_dir.iterdir() if p.name.endswith(".bin"))
    print(f"Found {len(cam_files)} camera images and {len(lidar_files)} LiDAR files")

    pairs, dts = pair_files(cam_files, lidar_files, int(args.max_dt * 1e6))
    if not pairs:
        sys.exit("ERROR: no camera/LiDAR pairs found (they must come from the same scene/log "
                 "and be within --max_dt seconds of each other).")
    print(f"Paired {len(pairs)} images with LiDAR (median time gap {np.median(dts) / 1000:.1f} ms, "
          f"max {max(dts) / 1000:.1f} ms)")
    if args.limit:
        pairs = pairs[: args.limit]

    # Load only the metadata records for the files we are using.
    wanted = {n for pair in pairs for n in pair}
    meta = load_metadata(args.meta, wanted, args.out / "meta_cache.json")

    depth_dir = args.out / "depth" / args.camera
    vis_dir = args.out / "vis" / args.camera
    depth_dir.mkdir(parents=True, exist_ok=True)
    if args.vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Main loop: project each LiDAR sweep into its camera image and save the depth map (existing ones are skipped).
    done = missing_meta = n_vis = 0
    coverage = []
    for cam_name, lid_name in tqdm(pairs, desc="projecting"):
        if cam_name not in meta["sd"] or lid_name not in meta["sd"]:
            missing_meta += 1  # file not listed in the metadata you pointed to
            continue
        stem = Path(cam_name).stem
        out_png = depth_dir / f"{stem}.png"
        if out_png.exists() and not args.overwrite:
            continue

        u, v, d, shape = project_lidar_to_camera(
            lidar_dir / lid_name, meta["sd"][lid_name], meta["sd"][cam_name], meta,
            args.min_depth, args.max_depth)
        depth = rasterize(u, v, d, shape)
        save_depth_png(depth, out_png)
        coverage.append((depth > 0).mean())
        done += 1

        if n_vis < args.vis:
            save_overlay(cam_dir / cam_name, u, v, d, vis_dir / f"{stem}.jpg", args.max_depth)
            n_vis += 1

    print(f"\nSaved {done} depth maps to {depth_dir}")
    if coverage:
        print(f"Average pixel coverage: {100 * np.mean(coverage):.2f}% (expect roughly 1-5%)")
    if missing_meta:
        print(f"Skipped {missing_meta} pairs whose files were not found in the metadata JSON.")
    if args.vis:
        print(f"Check the overlays in {vis_dir} before using the depth maps.")


if __name__ == "__main__":
    main()