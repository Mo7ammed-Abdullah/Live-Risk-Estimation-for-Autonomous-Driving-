"""
Automatic time-to-collision (TTC) per object, from the segmentation and depth maps of consecutive frames.
No nuScenes boxes needed, so it also works on your own videos. No training.

Per frame:
  1. find objects: connected groups of pixels that SegFormer labels as a vehicle (car/truck/bus/train),
     a person (person/rider) or a two-wheeler (bicycle/motorcycle); tiny groups are ignored
  2. follow them (predicted matching): each track's last box is grown by its expected scale change (from its own
     growth rate) to predict where it should be now; a new object of the same kind is matched to it when the centres
     are close relative to the box size (< 0.8 box sizes), the sizes agree (ratio 0.5-2) and the depths agree
     (within 50 %); best scores first. Unmatched objects start a new track.
     (On nuScenes validation this kept more tracks alive than plain box overlap: objects with a TTC 26 % -> 33 %.)
  3. measure: object depth = a low percentile of its pixels' depth (nearest surface); object size = sqrt(pixel count)
  4. TTC over the last `window` seconds of the track (least-squares slope, at least `min_span` s of history):
        depth   closing speed = -d(depth)/dt,        TTC = depth / closing speed
        scale   growth rate   = d(ln size)/dt,       TTC = 1 / growth rate
        combined = mean of the two when both exist, otherwise whichever exists
     Not approaching (closing <= min_closing m/s, or not growing) -> no TTC.

Known limits: touching objects of the same kind (e.g. a row of parked cars) form one object; an object hidden
for a frame starts a new track; TTC needs at least `min_span` seconds of history.
"""
from collections import deque

import numpy as np

KINDS = {"vehicle": ("car", "truck", "bus", "train"), "person": ("person", "rider"), "two-wheeler": ("bicycle", "motorcycle")}


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def slope(t, y):
    """Least-squares slope of y over t."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    tc = t - t.mean()
    den = (tc * tc).sum()
    return float((tc * (y - y.mean())).sum() / den) if den > 0 else 0.0


class TTCTracker:
    def __init__(self, class_names, min_area_frac=0.0003, iou_min=0.3, percentile=10, window=0.6, min_span=0.3,
                 min_closing=0.5, max_age=0.7, max_depth=80.0, predict=True):
        self.ids = {k: [class_names.index(c) for c in v if c in class_names] for k, v in KINDS.items()}
        self.min_area_frac, self.iou_min, self.percentile = min_area_frac, iou_min, percentile
        self.window, self.min_span, self.min_closing, self.max_age, self.max_depth = window, min_span, min_closing, max_age, max_depth
        self.predict = predict             # False = the original plain box-overlap matching
        self.tracks, self.next_id = [], 0

    def reset(self):
        self.tracks, self.next_id = [], 0

    def _objects(self, seg, depth):
        import cv2
        H, W = seg.shape
        min_area = max(int(self.min_area_frac * H * W), 12)
        objs = []
        for kind, ids in self.ids.items():
            mask = np.isin(seg, ids).astype(np.uint8)
            if not mask.any():
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for j in range(1, n):
                x, y, w, h, area = stats[j]
                if area < min_area:
                    continue
                d = np.clip(depth[lab == j], 0, self.max_depth)
                objs.append({"kind": kind, "box": (float(x), float(y), float(x + w), float(y + h)), "area": float(area),
                             "depth": float(np.percentile(d, self.percentile))})
        return objs

    def _score(self, o, tr, t):
        """How well a new object fits a track (higher = better, < 0 = not the same object)."""
        if tr["kind"] != o["kind"]:
            return -1.0
        if not self.predict:
            iou = box_iou(o["box"], tr["box"])
            return iou if iou >= self.iou_min else -1.0
        # Predicted box: last box grown around its bottom centre by the track's expected scale change.
        s = float(np.exp(np.clip(tr.get("growth", 0.0) * (t - tr["last_t"]), -0.7, 0.7)))
        x0, y0, x1, y1 = tr["box"]
        cx, by, w, h = (x0 + x1) / 2, y1, (x1 - x0) * s, (y1 - y0) * s
        pred = (cx - w / 2, by - h, cx + w / 2, by)
        ow, oh = o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]
        ocx, ocy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
        dist = np.hypot(ocx - cx, ocy - (by - h / 2)) / max(np.sqrt(max(w * h, 1.0)), 1.0)
        size_ratio = np.sqrt(max(ow * oh, 1.0) / max(w * h, 1.0))
        dz = abs(o["depth"] - tr["depth"]) / max(tr["depth"], 1.0)
        if dist > 0.8 or not (0.5 <= size_ratio <= 2.0) or dz > 0.5:
            return -1.0
        return box_iou(o["box"], pred) + 0.5 * (1 - dist / 0.8) + 0.2 * (1 - min(dz / 0.5, 1))

    def update(self, seg, depth, t):
        """seg, depth: (H, W) maps of the same size; t: time in seconds. Returns the list of current objects with TTC."""
        objs = self._objects(seg, depth)
        self.tracks = [tr for tr in self.tracks if t - tr["last_t"] <= self.max_age]
        # Greedy matching, best fits first, same kind only.
        pairs = sorted(((self._score(o, tr, t), i, k) for i, o in enumerate(objs) for k, tr in enumerate(self.tracks)), reverse=True)
        used_o, used_t = set(), set()
        for sc, i, k in pairs:
            if sc < 0:
                break
            if i in used_o or k in used_t:
                continue
            used_o.add(i); used_t.add(k)
            objs[i]["track"] = self.tracks[k]
        out = []
        for i, o in enumerate(objs):
            tr = o.get("track")
            if tr is None:
                tr = {"id": self.next_id, "kind": o["kind"], "hist": deque()}
                self.next_id += 1
                self.tracks.append(tr)
            tr["box"], tr["last_t"], tr["depth"] = o["box"], t, o["depth"]
            tr["hist"].append((t, o["depth"], 0.5 * np.log(max(o["area"], 1.0))))
            while tr["hist"] and t - tr["hist"][0][0] > self.window + 1e-6:
                tr["hist"].popleft()
            h = list(tr["hist"])
            closing = ttc_d = ttc_s = None
            if len(h) >= 2 and h[-1][0] - h[0][0] >= self.min_span - 1e-6:
                ts = [x[0] for x in h]
                closing = -slope(ts, [x[1] for x in h])
                growth = slope(ts, [x[2] for x in h])
                tr["growth"] = growth
                ttc_d = o["depth"] / closing if closing > self.min_closing else None
                ttc_s = 1.0 / growth if growth > 1e-3 else None
            vals = [v for v in (ttc_d, ttc_s) if v is not None]
            out.append({"id": tr["id"], "kind": o["kind"], "box": o["box"], "area": o["area"], "depth": o["depth"],
                        "closing": closing, "ttc_depth": ttc_d, "ttc_scale": ttc_s,
                        "ttc": float(np.mean(vals)) if vals else None, "history_s": h[-1][0] - h[0][0]})
        return out


def draw_tracks(tile, objs, k, max_label=12):
    """Draw boxes with TTC on an RGB tile that is k times the size of the maps. Red < 2 s, orange < 4 s."""
    import cv2
    shown = sorted(objs, key=lambda o: (o["ttc"] is None, o["ttc"] if o["ttc"] is not None else 0, o["depth"]))[:max_label]
    for o in shown:
        x0, y0, x1, y1 = [int(v * k) for v in o["box"]]
        if o["ttc"] is not None and o["ttc"] < 2:
            col = (255, 40, 40)
        elif o["ttc"] is not None and o["ttc"] < 4:
            col = (255, 165, 0)
        else:
            col = (200, 200, 200)
        cv2.rectangle(tile, (x0, y0), (x1, y1), col, 2)
        label = f"{o['kind']} {o['ttc']:.1f}s" if o["ttc"] is not None else f"{o['kind']} {o['depth']:.0f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        yy = min(y1 + th + 6, tile.shape[0] - 4)            # label under the box (the risk marker uses the space above)
        tile[max(yy - th - 4, 0):yy + 3, x0:min(x0 + tw + 6, tile.shape[1])] = (0, 0, 0)
        cv2.putText(tile, label, (x0 + 3, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return tile
