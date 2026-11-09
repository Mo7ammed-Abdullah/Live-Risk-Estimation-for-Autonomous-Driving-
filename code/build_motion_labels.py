"""
Assumption-free motion labels for every risk-class object in box_gt/boxes.json, from what REALLY happened in nuScenes.

Two kinds of truth per object and frame:

1. True time-to-collision (TTC), standard definition: distance / speed at which that distance shrinks.
     z_near      nearest-surface depth along the camera axis now (m)         [as in boxes.json]
     closing_z   how fast the object's depth shrinks (m/s, > 0 = getting closer), from its real positions in the
                 previous and next frames, each expressed in our camera at that moment (central difference)
     ttc_z       z_near / closing_z  (only when closing_z > 0.1 m/s; otherwise null = not approaching)
     range, range_rate, ttc_range   the same with the straight-line distance between our car and the object centre
   This is what a camera-based TTC estimate is compared against.

2. Actual near-miss (what happened next, nothing predicted):
     min_dist    the smallest real gap (m) between our car's footprint and the object's real box over the next
                 --horizon seconds, using our car's REAL future poses and the object's REAL future annotations
     t_min       when that smallest gap happened (s from now)
     n_future    how many future frames the object was seen in
   No constant-velocity or constant-turn assumption: our real path and the object's real path are used.

Output:  box_gt/motion.json   { "<image stem>": [ {inst, cls, ...}, ... ] }

Run from the project folder (venv311 active), CPU only, about 1-2 minutes:
    python code/build_motion_labels.py
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
# Our car's footprint in the nuScenes ego frame (origin = rear axle, x forward, y left), Renault ZOE, metres.
EGO_X = (-0.7, 3.45)
EGO_HALF_W = 0.87


def quat_to_rot(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_name(fname):
    parts = fname.split(".")[0].split("__")
    return parts[0], int(parts[2])


def ego_rect():
    """Our footprint as a polygon (4 x 2) in the ego frame."""
    return np.array([[EGO_X[1], EGO_HALF_W], [EGO_X[1], -EGO_HALF_W], [EGO_X[0], -EGO_HALF_W], [EGO_X[0], EGO_HALF_W]])


def box_polygon_ego(center_w, rot_w, size, R_ego, t_ego):
    """Object's real oriented box (ground footprint) in our ego frame at that moment, 4 x 2."""
    w, l = size[0], size[1]
    local = np.array([[l / 2, w / 2, 0], [l / 2, -w / 2, 0], [-l / 2, -w / 2, 0], [-l / 2, w / 2, 0]])
    corners_w = (rot_w @ local.T).T + center_w
    return (R_ego.T @ (corners_w - t_ego).T).T[:, :2]


def _seg_point_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(P, Q):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pp, qq = P @ n, Q @ n
            if pp.max() < qq.min() or qq.max() < pp.min():
                return False
    return True


def poly_distance(P, Q):
    """Smallest gap between two convex polygons (0 if they overlap)."""
    if _overlap(P, Q):
        return 0.0
    d = np.inf
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            for i in range(len(B)):
                d = min(d, _seg_point_dist(p, B[i], B[(i + 1) % len(B)]))
    return float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=HERE / "dataset" / "CAM_FRONT")
    ap.add_argument("--meta", type=Path, default=HERE / "v1.0-trainval_meta" / "v1.0-trainval")
    ap.add_argument("--boxes", type=Path, default=HERE / "box_gt" / "boxes.json")
    ap.add_argument("--out", type=Path, default=HERE / "box_gt" / "motion.json")
    ap.add_argument("--max_dt", type=float, default=0.1)
    ap.add_argument("--max_track_dt", type=float, default=1.5)
    ap.add_argument("--horizon", type=float, default=4.0, help="look this many seconds into the real future")
    ap.add_argument("--min_closing", type=float, default=0.1, help="below this closing speed (m/s) TTC is null")
    args = ap.parse_args()

    boxes = json.load(open(args.boxes))
    names = sorted(p.name for p in args.images.glob("*.jpg") if p.stem in boxes)

    # --- samples and their neighbours ------------------------------------------------------------
    scenes = {s["token"]: s["log_token"] for s in json.load(open(args.meta / "scene.json"))}
    logs = {l["token"]: l["logfile"] for l in json.load(open(args.meta / "log.json"))}
    samples, per_log = {}, defaultdict(list)
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
        j = min([j for j in (i - 1, i) if 0 <= j < len(lst)], key=lambda j: abs(keys[j] - ts))
        if abs(keys[j] - ts) / 1e6 <= args.max_dt:
            sample_of[Path(n).stem] = lst[j][1]
    current = set(sample_of.values())
    # Needed: one frame back, and every frame up to the horizon ahead (keyframes are ~0.5 s apart).
    n_ahead = int(np.ceil(args.horizon / 0.5)) + 1
    needed = set(current)
    for s in current:
        if samples[s]["prev"]:
            needed.add(samples[s]["prev"])
        nxt = samples[s]["next"]
        for _ in range(n_ahead):
            if not nxt:
                break
            needed.add(nxt)
            nxt = samples[nxt]["next"]
    print(f"matched {len(sample_of)} images; {len(needed)} samples needed (incl. neighbours and future)")

    # --- annotations, poses ----------------------------------------------------------------------
    ann = {}            # token -> (sample, instance, centre (3,), rotation (3x3), size, prev, next)
    with open(args.meta / "sample_annotation.json", "rb") as f:
        for a in tqdm(ijson.items(f, "item"), desc="streaming sample_annotation.json", unit=" ann"):
            if a["sample_token"] in needed:
                ann[a["token"]] = (a["sample_token"], a["instance_token"], np.array([float(v) for v in a["translation"]]),
                                   quat_to_rot(a["rotation"]), [float(v) for v in a["size"]], a["prev"], a["next"])
    by_sample_inst = {(v[0], v[1]): k for k, v in ann.items() if v[0] in current}
    cam_sd = {}
    with open(args.meta / "sample_data.json", "rb") as f:
        for d in tqdm(ijson.items(f, "item"), desc="streaming sample_data.json", unit=" rec"):
            if d["is_key_frame"] and d["sample_token"] in needed and d["filename"].startswith("samples/CAM_FRONT/"):
                cam_sd[d["sample_token"]] = (d["ego_pose_token"], d["calibrated_sensor_token"])
    want = {v[0] for v in cam_sd.values()}
    ego = {}
    with open(args.meta / "ego_pose.json", "rb") as f:
        for e in tqdm(ijson.items(f, "item"), desc="streaming ego_pose.json", unit=" pose"):
            if e["token"] in want:
                ego[e["token"]] = (quat_to_rot(e["rotation"]), np.array([float(v) for v in e["translation"]]), int(e["timestamp"]) / 1e6)
    cs_all = {c["token"]: c for c in json.load(open(args.meta / "calibrated_sensor.json"))}
    pose = {}           # sample -> (R_ego, t_ego, R_cam, t_cam, time)
    for s, (et, ct) in cam_sd.items():
        if et in ego:
            c = cs_all[ct]
            pose[s] = (ego[et][0], ego[et][1], quat_to_rot(c["rotation"]), np.array([float(v) for v in c["translation"]]), ego[et][2])

    def to_cam(p, s):
        Re, te, Rc, tc, _ = pose[s]
        return Rc.T @ (Re.T @ (p - te) - tc)

    rect = ego_rect()
    out, stats = {}, defaultdict(int)
    for stem, objs in tqdm(sorted(boxes.items()), desc="labelling"):
        s = sample_of.get(stem)
        if s is None or s not in pose:
            continue
        t_now = pose[s][4]
        rows = []
        for o in objs:
            if not o["cls"]:
                continue
            stats["objects"] += 1
            row = {"inst": o["inst"], "cls": o["cls"], "z_near": o["zmin"], "has_motion": False,
                   "closing_z": None, "ttc_z": None, "range": None, "range_rate": None, "ttc_range": None,
                   "min_dist": None, "t_min": None, "n_future": 0}
            a = by_sample_inst.get((s, o["inst"]))
            if a is not None:
                # ---- 1. true closing speed / TTC from real positions (central difference when possible)
                p_now = to_cam(ann[a][2], s)
                track = [(t_now, p_now)]
                for k in (5, 6):
                    nb = ann.get(ann[a][k])
                    if nb and nb[0] in pose and abs(pose[nb[0]][4] - t_now) <= args.max_track_dt:
                        track.append((pose[nb[0]][4], to_cam(nb[2], nb[0])))
                track.sort(key=lambda x: x[0])
                if len(track) >= 2:
                    (t0, p0), (t1, p1) = track[0], track[-1]
                    closing_z = -float((p1[2] - p0[2]) / (t1 - t0))
                    rng = float(np.hypot(p_now[0], p_now[2]))
                    rng_rate = float((np.hypot(p1[0], p1[2]) - np.hypot(p0[0], p0[2])) / (t1 - t0))
                    row.update({"has_motion": True, "closing_z": closing_z, "range": rng, "range_rate": rng_rate,
                                "ttc_z": float(o["zmin"] / closing_z) if closing_z > args.min_closing else None,
                                "ttc_range": float(rng / -rng_rate) if -rng_rate > args.min_closing else None})
                    stats["with_motion"] += 1
                    stats["approaching"] += row["ttc_z"] is not None
                # ---- 2. actual near-miss: real future of both our car and the object
                best, best_t, n_f = np.inf, None, 0
                cur = a
                while True:
                    s_k = ann[cur][0]
                    if s_k in pose:
                        dt = pose[s_k][4] - t_now
                        if dt > args.horizon + 1e-6:
                            break
                        if dt >= -1e-6:
                            poly = box_polygon_ego(ann[cur][2], ann[cur][3], ann[cur][4], pose[s_k][0], pose[s_k][1])
                            d = poly_distance(rect, poly)
                            n_f += dt > 1e-6
                            if d < best:
                                best, best_t = d, float(dt)
                    nxt = ann[cur][6]
                    if not nxt or nxt not in ann:
                        break
                    cur = nxt
                if np.isfinite(best):
                    row.update({"min_dist": float(best), "t_min": best_t, "n_future": int(n_f)})
            rows.append(row)
        out[stem] = rows
        stats["frames"] += 1

    json.dump(out, open(args.out, "w"))
    md = np.array([r["min_dist"] for v in out.values() for r in v if r["min_dist"] is not None and r["n_future"] > 0])
    print(f"Saved {args.out}: {stats['frames']} frames, {stats['objects']} risk-class objects, {stats['with_motion']} with motion, "
          f"{stats['approaching']} approaching (TTC defined)")
    for thr in (0.5, 1.0, 2.0):
        print(f"  real gap to our car within {args.horizon:.0f} s below {thr} m: {(md < thr).sum()} objects")


if __name__ == "__main__":
    main()
