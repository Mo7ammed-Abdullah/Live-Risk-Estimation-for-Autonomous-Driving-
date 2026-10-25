"""

    R(x, y)  = w_c(x, y) * exp(-lambda * D_norm(x, y))          paper Eq. 5  (pixel risk)
    D_norm   = D / (max(D) + eps)                               paper Eq. 4  (depth normalised per frame)
    U_i      = alpha_i * mean(R over horizontal region i)       paper Eq. 7  (region urgency)
    decision = argmax_i U_i                                     paper Eq. 8

Two depth modes:
    "paper"  : exactly the paper's Eq. 4 / Eq. 5 (depth divided by the frame's largest depth).
    "metric" : uses real meters from UniDepth: R = w_c * exp(-lambda_m * depth_in_meters).  (your extension)

!! Defaults marked ASSUMPTION are mine. The paper text does not list them. Change them HERE, or pass your own
!! values from the command line / a JSON file. Later we will fit them against ground-truth risk from nuScenes.
"""
# All maths here is plain numpy, so this file can be imported and tested without a GPU or any model.
# If you change class weights or lambda, change them here: every script imports this file.
import numpy as np

# ASSUMPTION: the paper's Fig. 1 shows a class-weight table with three groups (pedestrian, vehicle, background).
# Check the exact numbers in your PDF and edit them here. Bicycle/motorcycle/rider sit in between: my choice.
DEFAULT_CLASS_WEIGHTS = {
    "person": 1.0, "rider": 1.0, "bicycle": 0.9, "motorcycle": 0.9,
    "car": 0.8, "truck": 0.8, "bus": 0.8, "train": 0.8,
}
BACKGROUND_WEIGHT = 0.1          # road, sidewalk, buildings, poles, sky, ... (ASSUMPTION, see above)
DEFAULT_LAMBDA_PAPER = 0.25      # paper: lambda = 0.25 with normalised depth
DEFAULT_LAMBDA_METRIC = 0.05     # ASSUMPTION: per meter, risk halves roughly every 14 m
DEFAULT_REGIONS = 3              # ASSUMPTION: the paper does not say how many regions N it uses


# Turn the {class name: weight} table into an array indexed by class id.
# Any class not in the table (road, sky, ...) gets the background weight.
def class_weight_vector(names, weights=None, background=BACKGROUND_WEIGHT):
    """One weight per class id, in the order of `names` (the class list of your masks)."""
    weights = DEFAULT_CLASS_WEIGHTS if weights is None else weights
    return np.array([weights.get(n, background) for n in names], dtype=np.float32)


# Per-pixel risk: class weight times exponential decay with distance (closer = riskier).
def pixel_risk(w, d_norm, lam):
    """Paper Eq. 5 on arrays that are already aligned."""
    return w * np.exp(-lam * d_norm)


def compute_risk(seg, depth, w_vec, mode="paper", lam=DEFAULT_LAMBDA_PAPER, lam_metric=DEFAULT_LAMBDA_METRIC,
                 max_depth=80.0, eps=1e-6):
    """seg: (H, W) class ids, depth: (H, W) meters, w_vec: weight per class id. Returns the (H, W) risk map."""
    # 1) Clean the depth map: replace NaN/inf values with the largest valid depth so they count as far away.
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.all():
        depth = np.where(finite, depth, depth[finite].max() if finite.any() else max_depth)
    # 2) Look up each pixel's class weight (fancy indexing: seg holds class ids).
    w = w_vec[seg]
    # 3) Apply the formula. 'paper' normalises depth by the frame maximum (relative depth).
    # 'metric' uses real meters, clipped to max_depth, and is what all reported results use.
    if mode == "paper":
        return pixel_risk(w, depth / (depth.max() + eps), lam).astype(np.float32)
    if mode == "metric":
        return pixel_risk(w, np.clip(depth, 0.0, max_depth), lam_metric).astype(np.float32)
    raise ValueError("mode must be 'paper' or 'metric'")


# Region weights alpha: 0.4 (far/top) rising to 1.0 (near/bottom). This is the default used in all reported test results.
def default_alphas(n_regions):
    """ASSUMPTION: regions near the car (bottom of the image) matter more than far ones (top)."""
    return np.linspace(0.4, 1.0, n_regions) if n_regions > 1 else np.array([1.0])


# FITTED on the validation drive (fit_decision.py, 4-block cross-validation): the paper's plain mean (p = 1) picks the
# truly most urgent region only 47 % of the time (50 % for a coin flip); taking the highest risk in the region
# (p = infinity) with the top region weighted 0.6 of the bottom one gets about 82 %.  Metric depth mode, lambda 0.05/m.
# Fitted rule settings (used only in the live demo and fit_decision.py): p = infinity means 'highest risk in the band'.
FITTED_P = float("inf")
FITTED_A_MIN = 0.6


# Region weights for the fitted rule: a_min (far) rising to 1.0 (near).
def fitted_alphas(n_regions, a_min=FITTED_A_MIN):
    return np.linspace(a_min, 1.0, n_regions) if n_regions > 1 else np.array([1.0])


def region_urgency(risk, n_regions=DEFAULT_REGIONS, alphas=None, p=1.0):
    """Split the risk map into N horizontal bands (band 0 = top/far ... last = bottom/near).
    U_i = alpha_i * (mean over band of R^p)^(1/p).   p = 1 is the paper's Eq. 7 (plain mean);
    p = infinity is the highest risk in the band (the fitted rule, see FITTED_P).
    Returns (urgency per band, index of the most urgent band, list of (row_start, row_end))."""
    # Use default weights unless the caller passes its own (one value per region).
    alphas = default_alphas(n_regions) if alphas is None else np.asarray(alphas, dtype=np.float64)
    if len(alphas) != n_regions:
        raise ValueError("need one alpha per region")
    # Cut the image rows into n equal horizontal bands: band 0 = top (far), last band = bottom (near).
    bands = [(int(b[0]), int(b[-1]) + 1) for b in np.array_split(np.arange(risk.shape[0]), n_regions)]

    # How to summarise the risk values of one band:
    # infinity -> maximum; p = 1 -> plain mean (base paper); other p -> generalised mean.
    def aggregate(x):
        if np.isinf(p):
            return float(x.max())
        return float(np.mean(np.power(x, p)) ** (1.0 / p)) if p != 1 else float(x.mean())

    # Urgency of each band = its alpha times its summarised risk. The decision is the band with the highest urgency.
    urgency = np.array([alphas[i] * aggregate(risk[r0:r1]) for i, (r0, r1) in enumerate(bands)])
    return urgency, int(urgency.argmax()), bands


# Human-readable band names; with 3 bands: far / mid / near.
def region_names(n_regions):
    if n_regions == 3:
        return ["far", "mid", "near"]
    return [f"zone {i + 1}" + (" (far)" if i == 0 else " (near)" if i == n_regions - 1 else "") for i in range(n_regions)]
