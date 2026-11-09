"""Diagnostic (train split only): are the 'dangerous' labels caused by our own car turning?
For every train frame: our speed and yaw rate (from ego poses of the neighbouring samples).
For every object: its own speed in the world (static or moving).
Then recompute TTC with a turning-aware model: our car follows its arc (constant speed + yaw rate),
the object moves in a straight line at its world velocity; collision = object touches our footprint."""
import bisect, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import ijson, numpy as np

P = Path(sys.argv[1]); META = P / "v1.0-trainval_meta" / "v1.0-trainval"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "train"
EGO_X, EGO_HALF_W, MARGIN, H = (-0.7, 3.45), 0.87, 0.3, 4.0

def quat_to_rot(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def yaw(R): return np.arctan2(R[1, 0], R[0, 0])

stems = [l.strip() for l in open(P / "splits" / f"{SPLIT}.txt") if l.strip()]
boxes = json.load(open(P / "box_gt" / "boxes.json")); ttc = json.load(open(P / "box_gt" / "ttc.json"))
scenes = {s["token"]: s["log_token"] for s in json.load(open(META / "scene.json"))}
logs = {l["token"]: l["logfile"] for l in json.load(open(META / "log.json"))}
samples, per_log = {}, defaultdict(list)
for s in json.load(open(META / "sample.json")):
    samples[s["token"]] = s; per_log[logs[scenes[s["scene_token"]]]].append((int(s["timestamp"]), s["token"]))
for v in per_log.values(): v.sort()
sample_of = {}
for st in stems:
    log, ts = st.split("__")[0], int(st.split("__")[2])
    lst = per_log[log]; keys = [t for t, _ in lst]; i = bisect.bisect_left(keys, ts)
    j = min([j for j in (i-1, i) if 0 <= j < len(lst)], key=lambda j: abs(keys[j]-ts))
    sample_of[st] = lst[j][1]
cur = set(sample_of.values()); needed = set(cur)
for s in cur:
    for k in ("prev", "next"):
        if samples[s][k]: needed.add(samples[s][k])
ann = {}
with open(META / "sample_annotation.json", "rb") as f:
    for a in ijson.items(f, "item"):
        if a["sample_token"] in needed:
            ann[a["token"]] = (a["sample_token"], a["instance_token"], np.array([float(v) for v in a["translation"]]), a["prev"], a["next"], [float(v) for v in a["size"]])
by = {(v[0], v[1]): k for k, v in ann.items() if v[0] in cur}
cam = {}
with open(META / "sample_data.json", "rb") as f:
    for d in ijson.items(f, "item"):
        if d["is_key_frame"] and d["sample_token"] in needed and d["filename"].startswith("samples/CAM_FRONT/"):
            cam[d["sample_token"]] = d["ego_pose_token"]
want = set(cam.values()); ego = {}
with open(META / "ego_pose.json", "rb") as f:
    for e in ijson.items(f, "item"):
        if e["token"] in want:
            ego[e["token"]] = (quat_to_rot(e["rotation"]), np.array([float(v) for v in e["translation"]]), int(e["timestamp"]) / 1e6)
pose = {s: ego[t] for s, t in cam.items() if t in ego}

rows = []
for st in stems:
    s = sample_of[st]
    nb = [x for x in (samples[s]["prev"], s, samples[s]["next"]) if x and x in pose]
    if len(nb) < 2: continue
    (R0, t0, ts0), (R1, t1, ts1) = pose[nb[0]], pose[nb[-1]]
    dt = ts1 - ts0
    v_ego_w = (t1 - t0) / dt
    speed = float(np.hypot(*v_ego_w[:2]))
    yr = float(np.angle(np.exp(1j * (yaw(R1) - yaw(R0))))) / dt          # yaw rate rad/s
    Rc, tc, _ = pose[s]
    lab = {r["inst"]: r for r in ttc.get(st, [])}
    for o in boxes.get(st, []):
        if not o["cls"] or o["inst"] not in lab or not lab[o["inst"]].get("has_motion"): continue
        a = by.get((s, o["inst"]))
        if a is None: continue
        tr = [(pose[s][2], ann[a][2])]
        for k in (3, 4):
            n = ann.get(ann[a][k])
            if n and n[0] in pose and abs(pose[n[0]][2] - pose[s][2]) <= 1.5: tr.append((pose[n[0]][2], n[2]))
        tr.sort(key=lambda x: x[0])
        if len(tr) < 2: continue
        v_obj = (tr[-1][1] - tr[0][1]) / (tr[-1][0] - tr[0][0])
        # turning-aware TTC in the CURRENT ego frame: ego moves along an arc, object straight in the world
        p0 = Rc.T @ (ann[a][2] - tc)                         # object in ego frame now (x fwd, y left)
        vo = Rc.T @ v_obj                                     # object world velocity in ego axes now
        r = 0.5 * ann[a][5][0] + MARGIN
        hit = None
        for t in np.arange(0, H + 1e-9, 0.02):
            th = yr * t                                       # our heading change
            if abs(yr) > 1e-4: ex, ey = speed / yr * np.sin(th), speed / yr * (1 - np.cos(th))
            else: ex, ey = speed * t, 0.0
            q = p0[:2] + vo[:2] * t - np.array([ex, ey])       # object relative to our position, world-aligned axes of 'now'
            c, sn = np.cos(-th), np.sin(-th)
            qx, qy = c * q[0] - sn * q[1], sn * q[0] + c * q[1]   # into our rotated frame at time t
            dx = max(EGO_X[0] - qx, 0, qx - EGO_X[1]); dy = max(-EGO_HALF_W - qy, 0, qy - EGO_HALF_W)
            if np.hypot(dx, dy) <= r: hit = round(float(t), 2); break
        old = lab[o["inst"]]["ttc"]
        rows.append({"stem": st, "cls": o["cls"], "old": old, "new": hit, "speed": speed, "yaw_deg_s": np.degrees(yr),
                     "obj_speed": float(np.hypot(*v_obj[:2])), "lat": p0[1], "fwd": p0[0]})

def summary(name, sel):
    sel = list(sel)
    if not sel: print(f"{name}: none"); return
    turn = np.mean([abs(r["yaw_deg_s"]) > 5 for r in sel]); static = np.mean([r["obj_speed"] < 0.5 for r in sel])
    inlane = np.mean([abs(r["lat"]) < 2 for r in sel])
    print(f"{name}: {len(sel)} | our car turning (>5 deg/s): {turn:.0%} | object static (<0.5 m/s): {static:.0%} | in our lane (|y|<2 m): {inlane:.0%} | drives {len({r['stem'].split('__')[0] for r in sel})}")

print(f"[{SPLIT}] {len(rows)} objects with motion")
old_pos = [r for r in rows if r["old"] is not None and r["old"] <= H]
new_pos = [r for r in rows if r["new"] is not None]
summary("OLD labels (straight-line)  dangerous", old_pos)
summary("NEW labels (turning-aware)  dangerous", new_pos)
both = [r for r in old_pos if r["new"] is not None]
print(f"old dangerous that stay dangerous: {len(both)}/{len(old_pos)} | new dangerous that were not old: {len(new_pos) - len(both)}")
print("all frames: share with our car turning >5 deg/s:", f"{np.mean([abs(r['yaw_deg_s']) > 5 for r in rows]):.0%}", "| our speed median", f"{np.median([r['speed'] for r in rows]):.1f} m/s")
print("new dangerous by class:", dict(Counter(r["cls"] for r in new_pos)))
print("examples of OLD-only (dropped):")
for r in [r for r in old_pos if r["new"] is None][:6]:
    print(f"   {r['cls']:10s} old ttc {r['old']:.2f}  turn {r['yaw_deg_s']:6.1f} deg/s  obj speed {r['obj_speed']:.1f}  fwd {r['fwd']:5.1f} lat {r['lat']:5.1f}")
print("examples of NEW dangerous:")
for r in new_pos[:6]:
    print(f"   {r['cls']:10s} new ttc {r['new']:.2f}  turn {r['yaw_deg_s']:6.1f} deg/s  obj speed {r['obj_speed']:.1f}  fwd {r['fwd']:5.1f} lat {r['lat']:5.1f}")
