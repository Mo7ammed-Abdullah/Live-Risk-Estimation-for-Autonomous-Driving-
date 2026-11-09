"""
Real danger labels (time-to-collision) for every object in box_gt/boxes.json, from nuScenes motion.

Why: the risk formula only looks at class and distance. A car 10 m ahead driving away is not dangerous;
a pedestrian 15 m ahead walking into our lane is. These labels say which objects are ACTUALLY on a
collision course, so a learned fusion can be trained and judged against something real.

How (per object, per frame):
  1. Find the object's annotation in this frame and in the previous / next annotated frame (nuScenes keeps
     a linked track per object, about 0.5 s apart).
  2. Express each of those positions in OUR camera frame at that moment (using our car's pose then).
     The change over time is the object's velocity RELATIVE to our car (it includes our own speed and turning).
  3. Assume both keep moving the same way (constant relative velocity) for up to --horizon seconds and
     find the first moment the object touches our car's footprint (a rectangle around our car, plus the
     object's half width). That moment is the time-to-collision (TTC). No touch within the horizon -> no TTC.

Output:   box_gt/ttc.json   { "<image stem>": [ {inst, cls, ttc, v_rel, speed_rel, closing, cpa_dist, cpa_t, has_motion}, ... ] }
          One entry per object of a risk class, matched to boxes.json by "inst".

Run from the project folder (venv311 active):
    python code/build_ttc_labels.py
Takes several minutes: it streams sample_data.json (1.3 GB), ego_pose.json and sample_annotation.json once each.

Assumptions (stated in the paper):
  - constant relative velocity over the horizon (standard for TTC);
  - our car's footprint: Renault ZOE, about 4.1 m x 1.73 m; in the nuScenes car frame (origin = rear axle,
    x forward, y left) it spans x in [-0.7, 3.45] m and y in [-0.87, 0.87] m;
  - objects are treated as circles of radius half their width (their heading is not used), plus --margin.
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

# Our car's footprint in the nuScenes ego frame (origin = rear axle, x forward, y left), in metres.
EGO_X = (-0.7, 3.45)
EGO_HALF_W = 0.87


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


def to_cam(p_glob, pose):
    """World point -> camera frame of one moment. pose = (R_ego, t_ego, R_cam, t_cam)."""
    Re, te, Rc, tc = pose
    return Rc.T @ (Re.T @ (np.asarray(p_glob, dtype=np.float64) - te) - tc)


def ego_footprint_cam(Rc, tc):
    """Our car's rectangle expressed in the camera's ground plane: (x_min, x_max, z_min, z_max).
    Camera axes: x right, y down, z forward.  Ego axes: x forward, y left."""
    corners = []
    for ex in EGO_X:
        for ey in (-EGO_HALF_W, EGO_HALF_W):
            c = Rc.T @ (np.array([ex, ey, 0.0]) - tc)
            corners.append(c)
    corners = np.array(corners)
    return corners[:, 0].min(), corners[:, 0].max(), corners[:, 2].min(), corners[:, 2].max()


def dist_to_rect(x, z, rect):
    """Distance from points (x, z) to the rectangle (0 inside)."""
    x0, x1, z0, z1 = rect
    dx = np.maximum(np.maximum(x0 - x, 0.0), x - x1)
    dz = np.maximum(np.maximum(z0 - z, 0.0), z - z1)
    return np.hypot(dx, dz)


def time_to_collision(p, v, radius, rect, horizon, step=0.02):
    """Constant relative velocity in the ground plane (camera x, z).
    Returns (ttc or None, closest distance within the horizon, time of that closest approach)."""
    t = np.arange(0.0, horizon + 1e-9, step)
    x = p[0] + v[0] * t
    z = p[2] + v[2] * t
    d = dist_to_rect(x, z, rect) - radius
    hit = np.nonzero(d <= 0.0)[0]
    i = int(np.argmin(d))
    return (float(t[hit[0]]) if len(hit) else None), float(max(d[i], 0.0)), float(t[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta", type=Path, default=HERE / "v1.0-trainval_meta" / "v1.0-trainval")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--out", type=Path, default=HERE / "box_gt" / "ttc.json")
    ap.add_argument("--max_dt", type=float, default=0.1, help="max seconds between image and matched sample")
    ap.add_argument("--max_track_dt", type=float, default=1.5, help="ignore neighbouring annotations further apart than this (s)")
    ap.add_argument("--horizon", type=float, default=4.0, help="look this many seconds ahead")
    ap.add_argument("--margin", type=float, default=0.3, help="extra safety margin added to the object radius (m)")
    args = ap.parse_args()

    boxes = json.load(open(args.boxes))
    stems = set(boxes)
    names = sorted(p.name for p in args.images.glob("*.jpg") if p.stem in stems)
    print(f"{len(names)} images with objects in boxes.json")

    # --- match images to nuScenes samples (same rule as generate_box_gt.py) -----------------------
    scenes = {s["token"]: s["log_token"] for s in json.load(open(args.meta / "scene.json"))}
    logs = {l["token"]: l["logfile"] for l in json.load(open(args.meta / "log.json"))}
    samples = {}
    per_log = defaultdict(list)
    for s in json.load(open(args.meta / "sample.json")):
        samples[s["token"]] = s
        per_log[logs[scenes[s["scene_token"]]]].append((int(s["timestamp"]), s["token"]))
    for v in per_log.values():
        v.sort()
    sample_of = {}
    for n in names:
        log, ts = parse_name(n)
        lst = per_log.get(log)
        if not lst:
            continue
        keys = [t for t, _ in lst]
        i = bisect.bisect_left(keys, ts)
        cand = [j for j in (i - 1, i) if 0 <= j < len(lst)]
        j = min(cand, key=lambda j: abs(keys[j] - ts))
        if abs(keys[j] - ts) / 1e6 <= args.max_dt:
            sample_of[Path(n).stem] = lst[j][1]
    print(f"matched {len(sample_of)} images to samples")

    # Samples we need poses and annotations for: our frames plus their neighbours (one step either way).
    current = set(sample_of.values())
    needed = set(current)
    for s in current:
        for k in ("prev", "next"):
            if samples[s][k]:
                needed.add(samples[s][k])

    # --- annotations of those samples -------------------------------------------------------------
    ann = {}                       # token -> (sample, instance, translation, size, prev, next)
    with open(args.meta / "sample_annotation.json", "rb") as f:
        for a in tqdm(ijson.items(f, "item"), desc="streaming sample_annotation.json", unit=" ann"):
            if a["sample_token"] in needed:
                ann[a["token"]] = (a["sample_token"], a["instance_token"], [float(v) for v in a["translation"]],
                                   [float(v) for v in a["size"]], a["prev"], a["next"])
    by_sample_inst = {(v[0], v[1]): k for k, v in ann.items() if v[0] in current}
    print(f"kept {len(ann)} annotations")

    # --- our car's pose at each needed sample (from its CAM_FRONT key frame) ------------------------
    cam_sd = {}                    # sample -> (ego_pose_token, calibrated_sensor_token)
    with open(args.meta / "sample_data.json", "rb") as f:
        for d in tqdm(ijson.items(f, "item"), desc="streaming sample_data.json", unit=" rec"):
            if d["is_key_frame"] and d["sample_token"] in needed and d["filename"].startswith("samples/CAM_FRONT/"):
                cam_sd[d["sample_token"]] = (d["ego_pose_token"], d["calibrated_sensor_token"])
    want_ego = {v[0] for v in cam_sd.values()}
    ego = {}
    with open(args.meta / "ego_pose.json", "rb") as f:
        for e in tqdm(ijson.items(f, "item"), desc="streaming ego_pose.json", unit=" pose"):
            if e["token"] in want_ego:
                ego[e["token"]] = (quat_to_rot(e["rotation"]), np.array([float(v) for v in e["translation"]]),
                                   int(e["timestamp"]))
    cs_all = {c["token"]: c for c in json.load(open(args.meta / "calibrated_sensor.json"))}
    pose, stamp = {}, {}           # sample -> (Re, te, Rc, tc), timestamp in s
    for s, (et, ct) in cam_sd.items():
        if et in ego:
            c = cs_all[ct]
            pose[s] = (ego[et][0], ego[et][1], quat_to_rot(c["rotation"]),
                       np.array([float(v) for v in c["translation"]]))
            stamp[s] = ego[et][2] / 1e6
    print(f"poses for {len(pose)}/{len(needed)} needed samples")

    # --- per object: relative velocity and time-to-collision ---------------------------------------
    out = {}
    stats = defaultdict(int)
    for stem, objs in tqdm(sorted(boxes.items()), desc="computing TTC"):
        s = sample_of.get(stem)
        if s is None or s not in pose:
            continue
        Rc, tc = pose[s][2], pose[s][3]
        rect = ego_footprint_cam(Rc, tc)
        rows = []
        for o in objs:
            if not o["cls"]:
                continue
            stats["objects"] += 1
            a_tok = by_sample_inst.get((s, o["inst"]))
            p = np.asarray(o["pos_cam"], dtype=np.float64)
            # Neighbouring positions of this object, each in our camera frame at its own moment.
            track = [(stamp[s], p)]
            if a_tok is not None:
                for k in (4, 5):                      # prev, next annotation tokens
                    nb = ann.get(ann[a_tok][k])
                    if nb and nb[0] in pose and abs(stamp[nb[0]] - stamp[s]) <= args.max_track_dt:
                        track.append((stamp[nb[0]], to_cam(nb[2], pose[nb[0]])))
            track.sort(key=lambda x: x[0])
            row = {"inst": o["inst"], "cls": o["cls"], "has_motion": len(track) >= 2}
            if len(track) >= 2:
                # Central difference when both neighbours exist, otherwise one-sided.
                (t0, p0), (t1, p1) = track[0], track[-1]
                v = (p1 - p0) / (t1 - t0)
                radius = 0.5 * float(o["size"][0]) + args.margin      # half the object's width
                ttc, cpa_d, cpa_t = time_to_collision(p, v, radius, rect, args.horizon)
                rng = float(np.hypot(p[0], p[2]))
                closing = float(-(p[0] * v[0] + p[2] * v[2]) / max(rng, 1e-6))   # >0 = getting closer
                row.update({"ttc": ttc, "v_rel": [float(v[0]), float(v[2])], "speed_rel": float(np.hypot(v[0], v[2])),
                            "closing": closing, "cpa_dist": cpa_d, "cpa_t": cpa_t})
                stats["with_motion"] += 1
                stats["ttc_found"] += ttc is not None
                stats["ttc<=2s"] += ttc is not None and ttc <= 2.0
            else:
                row.update({"ttc": None})
            rows.append(row)
        out[stem] = rows
        stats["frames"] += 1

    json.dump(out, open(args.out, "w"))
    print(f"Saved {args.out}: {stats['frames']} frames, {stats['objects']} risk-class objects, "
          f"{stats['with_motion']} with motion, {stats['ttc_found']} on a collision course within "
          f"{args.horizon:.0f} s ({stats['ttc<=2s']} within 2 s)")


if __name__ == "__main__":
    main()
