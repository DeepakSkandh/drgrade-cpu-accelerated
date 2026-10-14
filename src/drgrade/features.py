"""
Feature families v2 (30 features), all computed from preprocess() output.

  color      (6)  illumination-robust color descriptors (ratios, coefficients of variation)
  texture    (5)  GLCM on the quantized z-map, retina pixels only, 4 directions averaged
  vessel     (5)  dark, elongated / large components of the z-map
  red_lesion (8)  dark, small, compact components (microaneurysm- / hemorrhage-like CANDIDATES)
  bright     (6)  bright components outside the optic disc (exudate-like CANDIDATES)

All candidates are image-derived and NOT validated lesions.
"""

import cv2
import numpy as np

from .config import DEFAULT
from .preprocess import preprocess, _timer

FEATURE_FAMILIES = {
    "color": ["color_chroma_rg", "color_chroma_bg", "color_cv_r", "color_cv_g",
              "color_cv_b", "color_brightness"],
    "texture": ["tex_contrast", "tex_dissimilarity", "tex_homogeneity",
                "tex_energy", "tex_correlation"],
    "vessel": ["vessel_density", "vessel_n_components", "vessel_mean_component_area",
               "vessel_largest_frac", "vessel_mean_contrast"],
    "red_lesion": ["red_count", "red_small_count", "red_large_count", "red_area_ratio",
                   "red_mean_area", "red_mean_contrast", "red_quadrant_min_count",
                   "red_quadrants_over_thresh"],
    "bright": ["bright_count", "bright_area_ratio", "bright_mean_area",
               "bright_largest_area_ratio", "bright_mean_contrast",
               "bright_quadrants_over_thresh"],
}
FEATURE_NAMES = [f for fam in FEATURE_FAMILIES.values() for f in fam]


# ----------------------------------------------------------------------------
# color
# ----------------------------------------------------------------------------
def _hist_median(img_u8, channel, mask_u8):
    """Exact median of one uint8 channel inside the mask, via a 256-bin histogram (O(N), no sort)."""
    hist = cv2.calcHist([img_u8], [channel], mask_u8, [256], [0, 256]).ravel()
    c = np.cumsum(hist)
    return float(np.searchsorted(c, (c[-1] + 1) / 2))


def color_features(rgb, inner):
    m = inner.astype(np.uint8)
    med_r, med_g, med_b = (_hist_median(rgb, c, m) for c in range(3))
    mean, std = cv2.meanStdDev(rgb, mask=m)          # one pass, no float copy of the image
    means, stds = mean.ravel(), std.ravel()
    eps = 1e-6
    return {
        "color_chroma_rg": med_r / (med_g + eps),
        "color_chroma_bg": med_b / (med_g + eps),
        "color_cv_r": float(stds[0] / (means[0] + eps)),
        "color_cv_g": float(stds[1] / (means[1] + eps)),
        "color_cv_b": float(stds[2] / (means[2] + eps)),
        "color_brightness": float(means.mean() / 255.0),
    }


# ----------------------------------------------------------------------------
# texture
# ----------------------------------------------------------------------------
GLCM_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1)]       # 0, 45, 90, 135 degrees


def quantize_z(z, cfg=DEFAULT):
    c = cfg.glcm_z_clip
    q = np.floor((np.clip(z, -c, c - 1e-6) + c) / (2 * c) * cfg.glcm_levels)
    return q.astype(np.uint8)


def glcm_masked(q, inner, levels, dy, dx):
    """Symmetric, normalized GLCM counting only pixel pairs that are both inside the retina."""
    h, w = q.shape
    y0, y1 = 0, h - dy
    x0, x1 = max(0, -dx), w - max(0, dx)
    a, ma = q[y0:y1, x0:x1], inner[y0:y1, x0:x1]
    b, mb = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx], inner[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    both = ma & mb
    idx = a[both].astype(np.int32) * levels + b[both]
    m = np.bincount(idx, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    m = m + m.T
    return m / max(m.sum(), 1.0)


def texture_features(z, inner, cfg=DEFAULT):
    q = quantize_z(z, cfg)
    L = cfg.glcm_levels
    i, j = np.indices((L, L))
    props = []
    for dy, dx in GLCM_OFFSETS:
        P = glcm_masked(q, inner, L, dy, dx)
        mu_i, mu_j = (P * i).sum(), (P * j).sum()
        sd_i = np.sqrt((P * (i - mu_i) ** 2).sum())
        sd_j = np.sqrt((P * (j - mu_j) ** 2).sum())
        corr = ((P * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j)) if sd_i * sd_j > 0 else 1.0
        props.append([
            (P * (i - j) ** 2).sum(),
            (P * np.abs(i - j)).sum(),
            (P / (1.0 + (i - j) ** 2)).sum(),
            np.sqrt((P ** 2).sum()),
            corr,
        ])
    p = np.mean(props, axis=0)
    return dict(zip(FEATURE_FAMILIES["texture"], map(float, p)))


# ----------------------------------------------------------------------------
# dark structures: vessels vs red-lesion candidates
# ----------------------------------------------------------------------------
def _shape(labels, stats, i):
    """Elongation (rotated bounding box) and solidity of component i."""
    x, y, w, h = stats[i, :4]
    comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    elongation = max(rw, rh) / max(min(rw, rh), 1.0)
    area = cv2.contourArea(contour)
    hull = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull if hull > 0 else 1.0
    return elongation, solidity


def split_dark_components(z, inner, cfg=DEFAULT):
    """
    Threshold dark pixels and classify each connected component as
    vessel-like (large, elongated or irregular) or red-lesion candidate
    (small, compact). Returns label image, stats, centroids and two index arrays.
    """
    dark = ((z < -cfg.dark_k) & inner).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    vessel_ids, red_ids = [], []
    for i in range(1, n):
        a = areas[i]
        if a < cfg.red_min_area:
            continue
        if a > cfg.red_max_area:
            vessel_ids.append(i)
            continue
        if a >= cfg.shape_check_min_area:
            elong, solid = _shape(labels, stats, i)
            if elong > cfg.max_elongation or solid < cfg.min_solidity:
                vessel_ids.append(i)
                continue
        red_ids.append(i)
    return labels, stats, cents, np.array(vessel_ids, int), np.array(red_ids, int)


def _mean_z_over(labels, z, ids, n_labels):
    """Mean z of each listed component, all components in one pass (bincount)."""
    if len(ids) == 0:
        return np.array([])
    flat = labels.ravel()
    sums = np.bincount(flat, weights=z.ravel(), minlength=n_labels)
    counts = np.bincount(flat, minlength=n_labels)
    return sums[ids] / np.maximum(counts[ids], 1)


def _quadrant_counts(cents, ids, size):
    c = size / 2.0
    q = (cents[ids, 0] >= c).astype(int) + 2 * (cents[ids, 1] >= c).astype(int)
    return np.bincount(q, minlength=4)


def vessel_and_red_features(z, inner, cfg=DEFAULT, od=None, return_parts=False):
    """od = (x, y) optic-disc centre: red candidates inside the disc circle are dropped
    (the bright disc makes its surroundings look dark relative to the background)."""
    labels, stats, cents, v_ids, r_ids = split_dark_components(z, inner, cfg)
    if od is not None and len(r_ids):
        d2 = (cents[r_ids, 0] - od[0]) ** 2 + (cents[r_ids, 1] - od[1]) ** 2
        r_ids = r_ids[d2 > cfg.od_radius ** 2]
    n_labels = stats.shape[0]
    inner_area = float(inner.sum())
    areas = stats[:, cv2.CC_STAT_AREA]

    v_areas = areas[v_ids].astype(float)
    v_total = v_areas.sum()
    v_contrast = _mean_z_over(labels, z, v_ids, n_labels)
    feats = {
        "vessel_density": v_total / inner_area,
        "vessel_n_components": int(len(v_ids)),
        "vessel_mean_component_area": float(v_areas.mean()) if len(v_ids) else 0.0,
        "vessel_largest_frac": float(v_areas.max() / v_total) if v_total > 0 else 0.0,
        "vessel_mean_contrast": float(-(v_contrast * v_areas).sum() / v_total) if v_total > 0 else 0.0,
    }

    r_areas = areas[r_ids].astype(float)
    r_contrast = _mean_z_over(labels, z, r_ids, n_labels)
    if len(r_ids):
        qc = _quadrant_counts(cents, r_ids, z.shape[0])
        feats.update({
            "red_count": int(len(r_ids)),
            "red_small_count": int((r_areas <= cfg.red_small_max_area).sum()),
            "red_large_count": int((r_areas > cfg.red_small_max_area).sum()),
            "red_area_ratio": float(r_areas.sum() / inner_area),
            "red_mean_area": float(r_areas.mean()),
            "red_mean_contrast": float(-r_contrast.mean()),
            "red_quadrant_min_count": int(qc.min()),
            "red_quadrants_over_thresh": int((qc >= cfg.red_quadrant_thresh).sum()),
        })
    else:
        feats.update({k: 0 for k in FEATURE_FAMILIES["red_lesion"]})

    if return_parts:
        return feats, {"labels": labels, "stats": stats, "centroids": cents,
                       "vessel_ids": v_ids, "red_ids": r_ids}
    return feats


# ----------------------------------------------------------------------------
# bright structures: exudate-like candidates, optic disc excluded
# ----------------------------------------------------------------------------
def locate_optic_disc(green, inner, cfg=DEFAULT):
    """
    Brightest region of a heavily smoothed green channel = optic disc estimate.
    Simple heuristic: can be fooled by large bright lesions or reflections;
    q_od_peak_z reports how distinct the peak is.
    """
    d = cfg.bg_downscale
    g = cv2.resize(green, (green.shape[1] // d, green.shape[0] // d),
                   interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(inner.astype(np.uint8), (g.shape[1], g.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    g[~m] = float(np.median(g[m]))
    s = cv2.GaussianBlur(g, (0, 0), cfg.od_smooth_sigma)
    s[~m] = -np.inf
    y, x = np.unravel_index(np.argmax(s), s.shape)
    vals = s[m]
    peak_z = float((s[y, x] - np.median(vals)) / (vals.std() + 1e-6))
    return (x + 0.5) * d, (y + 0.5) * d, peak_z


def bright_features(z, inner, od, cfg=DEFAULT, return_parts=False):
    size = z.shape[0]
    od_x, od_y, od_peak_z = od
    yy, xx = np.ogrid[:size, :size]
    outside_od = (xx - od_x) ** 2 + (yy - od_y) ** 2 > cfg.od_radius ** 2

    bright = ((z > cfg.bright_k) & inner & outside_od).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    ids = np.array([i for i in range(1, n) if areas[i] >= cfg.bright_min_area], int)
    inner_area = float(inner.sum())

    if len(ids):
        a = areas[ids].astype(float)
        contrast = _mean_z_over(labels, z, ids, n)
        qc = _quadrant_counts(cents, ids, size)
        feats = {
            "bright_count": int(len(ids)),
            "bright_area_ratio": float(a.sum() / inner_area),
            "bright_mean_area": float(a.mean()),
            "bright_largest_area_ratio": float(a.max() / inner_area),
            "bright_mean_contrast": float(contrast.mean()),
            "bright_quadrants_over_thresh": int((qc >= cfg.bright_quadrant_thresh).sum()),
        }
    else:
        feats = {k: 0 for k in FEATURE_FAMILIES["bright"]}

    od = {"q_od_x": float(od_x), "q_od_y": float(od_y), "q_od_peak_z": od_peak_z}
    if return_parts:
        return feats, od, {"labels": labels, "ids": ids, "centroids": cents, "stats": stats}
    return feats, od


# ----------------------------------------------------------------------------
# everything
# ----------------------------------------------------------------------------
def extract_features(path, cfg=DEFAULT, timings=None):
    """
    Full pipeline for one image.
    Returns (features: dict of the 30 model inputs, quality: dict of q_* columns).
    """
    pre = preprocess(path, cfg, timings)
    z, inner, rgb = pre["z"], pre["inner"], pre["rgb"]
    feats = {}
    with _timer(timings, "color"):
        feats.update(color_features(rgb, inner))
    with _timer(timings, "texture"):
        feats.update(texture_features(z, inner, cfg))
    with _timer(timings, "optic_disc"):
        disc = locate_optic_disc(pre["green"], inner, cfg)
    with _timer(timings, "vessel_red"):
        feats.update(vessel_and_red_features(z, inner, cfg, od=disc[:2]))
    with _timer(timings, "bright"):
        b, od = bright_features(z, inner, disc, cfg)
        feats.update(b)
    quality = {**pre["quality"], **od}
    return {k: feats[k] for k in FEATURE_NAMES}, quality
