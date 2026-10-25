"""
Ground-truth objects for every CAM_FRONT frame, from the nuScenes 3D boxes (no devkit needed).

For each image it stores every annotated object that is in front of the camera:
    category, risk class (person/rider/car/...), visibility level, number of LiDAR points,
    distances (nearest-surface depth along the camera axis, centre depth, straight-line distance),
    the 2D box in the image (projection of the 3D box), and the object's position in the camera frame.

Output:   box_gt/boxes.json      { "<image stem>": [ {object}, ... ], ... }

Run from "Testing 3" (venv311 active):
    python generate_box_gt.py
Takes a few minutes: it streams sample_annotation.json (556 MB) once and keeps only your frames.

How frames are matched to annotations: every image is a keyframe, so it is matched to the nuScenes
"sample" of the same drive (log) whose timestamp is closest (the camera fires within ~50 ms of it).
"""
import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in code/)

# nuScenes category -> class name used by the segmentation model / risk.py (None = not a risk class)
RISK_CLASS = {
    "human.pedestrian.adult": "person", "human.pedestrian.child": "person",
    "human.pedestrian.police_officer": "person", "human.pedestrian.construction_worker": "person",
    "human.pedestrian.wheelchair": "person", "human.pedestrian.stroller": "person",
    "human.pedestrian.personal_mobility": "rider",
    "vehicle.bicycle": "bicycle", "vehicle.motorcycle": "motorcycle",
    "vehicle.car": "car", "vehicle.emergency.ambulance": "car", "vehicle.emergency.police": "car",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck", "vehicle.construction": "truck", "vehicle.trailer": "truck",
}


# nuScenes quaternion [w, x, y, z] -> 3x3 rotation matrix.
def quat_to_rot(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# Return (drive name, timestamp) from an image file name.
def parse_name(fname):
    parts = fname.split(".")[0].split("__")
    return parts[0], int(parts[2])


def box_corners_local(size):
    """nuScenes size = [width, length, height]; local frame x = length (forward), y = width, z = up."""
    w, l, h = [float(v) for v in size]
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return np.stack([x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta", type=Path, default=HERE / "v1.0-trainval_meta" / "v1.0-trainval")
    ap.add_argument("--meta_cache", type=Path, default=HERE / "depth_gt" / "meta_cache.json")
    ap.add_argument("--out", type=Path, default=HERE / "box_gt")
    ap.add_argument("--max_dt", type=float, default=0.1, help="max seconds between image and matched sample")
    ap.add_argument("--max_range", type=float, default=100.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The metadata cache (made by generate_depth_gt.py) holds each image's sensor calibration and car pose.
    cache = json.load(open(args.meta_cache))
    names = sorted(p.name for p in args.images.glob("*.jpg"))
    print(f"{len(names)} images")

    # --- samples per log --------------------------------------------------------------------
    # Group the nuScenes key samples by drive, with their timestamps.
    scenes = {s["token"]: s["log_token"] for s in json.load(open(args.meta / "scene.json"))}
    logs = {l["token"]: l["logfile"] for l in json.load(open(args.meta / "log.json"))}
    per_log = defaultdict(list)
    for s in json.load(open(args.meta / "sample.json")):
        per_log[logs[scenes[s["scene_token"]]]].append((int(s["timestamp"]), s["token"]))
    for v in per_log.values():
        v.sort()
    stems, sample_of, dt_list = {}, {}, []
    # Match every image to the key sample of the same drive with the closest timestamp (within --max_dt).
    for n in names:
        log, ts = parse_name(n)
        lst = per_log.get(log)
        if not lst:
            continue
        keys = [t for t, _ in lst]
        i = bisect.bisect_left(keys, ts)
        cand = [j for j in (i - 1, i) if 0 <= j < len(lst)]
        j = min(cand, key=lambda j: abs(keys[j] - ts))
        dt = abs(keys[j] - ts) / 1e6
        if dt <= args.max_dt:
            stems[n] = lst[j][1]
            dt_list.append(dt)
    print(f"matched {len(stems)}/{len(names)} images to samples (median gap {np.median(dt_list) * 1000:.0f} ms, "
          f"max {max(dt_list) * 1000:.0f} ms)")
    wanted = set(stems.values())

    # --- annotations ------------------------------------------------------------------------
    # Look up each annotation's category name (car, pedestrian, ...).
    inst_cat = {i["token"]: i["category_token"] for i in json.load(open(args.meta / "instance.json"))}
    cat_name = {c["token"]: c["name"] for c in json.load(open(args.meta / "category.json"))}
    # Stream the huge annotation file once and keep the 3D boxes of the samples we use.
    anns = defaultdict(list)
    with open(args.meta / "sample_annotation.json", "rb") as f:
        for a in tqdm(ijson.items(f, "item"), desc="streaming sample_annotation.json", unit=" ann"):
            if a["sample_token"] in wanted:
                anns[a["sample_token"]].append({
                    "t": [float(v) for v in a["translation"]], "size": [float(v) for v in a["size"]],
                    "r": [float(v) for v in a["rotation"]], "vis": int(a["visibility_token"] or 0),
                    "npts": int(a["num_lidar_pts"]), "inst": a["instance_token"],
                    "cat": cat_name[inst_cat[a["instance_token"]]]})
    print(f"kept annotations for {len(anns)} samples, {sum(len(v) for v in anns.values())} boxes")

    # --- project into each camera frame ----------------------------------------------------------------
    # For each image: express every 3D box in the camera frame (world -> car -> camera) and drop objects behind the camera or beyond --max_range.
    out, n_obj = {}, 0
    for n in tqdm(sorted(stems), desc="projecting boxes"):
        sd = cache["sd"].get(n)
        if sd is None:
            continue
        cs = cache["cs"][sd["calibrated_sensor_token"]]
        ego = cache["ego"][sd["ego_pose_token"]]
        K = np.array(cs["camera_intrinsic"], dtype=np.float64)
        Re, te = quat_to_rot(ego["rotation"]), np.array(ego["translation"], dtype=np.float64)
        Rc, tc = quat_to_rot(cs["rotation"]), np.array(cs["translation"], dtype=np.float64)
        objs = []
        for a in anns.get(stems[n], []):
            c_glob = np.array(a["t"])
            c_cam = Rc.T @ (Re.T @ (c_glob - te) - tc)
            if c_cam[2] < 0.5 or np.linalg.norm(c_cam) > args.max_range:
                continue
            # Rotate the box into the camera frame, then place its 8 corners around its centre.
            R_cam = Rc.T @ Re.T @ quat_to_rot(a["r"])
            corners = R_cam @ box_corners_local(a["size"]) + c_cam[:, None]
            z = corners[2]
            # Corners behind the camera are clamped so the projection stays finite.
            zc = np.maximum(z, 0.1)                       # corners behind the camera are clamped
            u = K[0, 0] * corners[0] / zc + K[0, 2]
            v = K[1, 1] * corners[1] / zc + K[1, 2]
            # 2D box in the image = the bounding rectangle of the 8 projected corners, clipped to the 1600 x 900 image.
            box = [float(np.clip(u.min(), 0, 1599)), float(np.clip(v.min(), 0, 899)),
                   float(np.clip(u.max(), 0, 1599)), float(np.clip(v.max(), 0, 899))]
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            # Record per object: category, risk class, visibility, LiDAR points, nearest-surface depth (zmin), centre depth (zc),
            # straight-line distance, sideways offset, size, 2D box and position in the camera frame.
            objs.append({"cat": a["cat"], "cls": RISK_CLASS.get(a["cat"]), "vis": a["vis"], "npts": a["npts"],
                         "inst": a["inst"], "zmin": float(max(z.min(), 0.1)), "zc": float(c_cam[2]),
                         "dist": float(np.linalg.norm(c_cam)), "lateral": float(c_cam[0]),
                         "size": a["size"], "box": box, "pos_cam": [float(v) for v in c_cam]})
        out[Path(n).stem] = objs
        n_obj += len(objs)
    # Save everything as box_gt/boxes.json: { image name: [objects] }.
    json.dump(out, open(args.out / "boxes.json", "w"))
    n_risk = sum(1 for v in out.values() for o in v if o["cls"])
    print(f"Saved {args.out / 'boxes.json'}: {len(out)} frames, {n_obj} objects in front of the camera, "
          f"{n_risk} of a risk class (person/rider/vehicles)")
    cnt = defaultdict(int)
    for v in out.values():
        for o in v:
            cnt[o["cls"] or "other"] += 1
    print("objects per class:", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
