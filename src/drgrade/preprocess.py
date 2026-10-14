"""
Preprocessing v2: resolution-independent and illumination-normalized.

    file ──► decode (reduced if large) ──► FOV mask ──► crop to retina ──► pad square
         ──► resize so retina = work_size px ──► final FOV mask ──► inner mask (rim removed)
         ──► green channel ──► background (median, at 1/4 res) ──► residual = green - background
         ──► robust z-map  z = (residual - median) / (1.4826 * MAD)

The z-map is the single input for vessel, red-lesion, bright-lesion and texture
features: negative = darker than the local background, positive = brighter.
"""

import time
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT


@contextmanager
def _timer(timings, key):
    t = time.perf_counter()
    yield
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t) * 1000


def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def fov_mask(rgb, threshold):
    """Retina = pixels where any channel exceeds `threshold`, cleaned."""
    mask = (rgb.max(axis=2) > threshold).astype(np.uint8) * 255
    k = max(3, (min(mask.shape) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _fill_holes(_largest_component(mask))


def decode(path, cfg=DEFAULT):
    """
    Decode an image as RGB. For large images, use the decoder's built-in
    downscaling (IMREAD_REDUCED_*): for JPEG this skips most of the decoding work.
    """
    with Image.open(path) as im:               # reads the header only
        w, h = im.size
    factor = 1
    if cfg.fast_decode:
        for f in (8, 4, 2):
            if min(w, h) / f >= 1.15 * cfg.work_size:
                factor = f
                break
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
    bgr = cv2.imread(str(path), flag)
    if bgr is None:
        raise IOError(f"Could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), factor


def crop_to_retina(rgb, cfg=DEFAULT):
    """Crop to the FOV bounding box, pad to a square, resize to work_size."""
    h, w = rgb.shape[:2]
    # locate the FOV on a small copy (fast), then crop the full-resolution image
    s = 512 / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    m = fov_mask(small, cfg.mask_threshold)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise ValueError("No retina found (image too dark?)")
    x0, x1 = int(xs.min() / s), int(np.ceil((xs.max() + 1) / s))
    y0, y1 = int(ys.min() / s), int(np.ceil((ys.max() + 1) / s))
    crop = rgb[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

    # pad to square (keeps the circle geometry when top/bottom of the retina are cut off)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    square = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                cv2.BORDER_CONSTANT, value=0)
    interp = cv2.INTER_AREA if side > cfg.work_size else cv2.INTER_LINEAR   # cubic overshoot creates false spots
    out = cv2.resize(square, (cfg.work_size, cfg.work_size), interpolation=interp)
    return out, side


def extend_outside(img, mask, iterations):
    """
    Grow the image outward from the mask: each step fills the pixels just outside
    the current region with the mean of their filled neighbours. Gives the
    background filter realistic values beyond the retina edge, so the rim is not
    mistaken for a dark structure.
    """
    out = img.astype(np.float32)
    m = mask.astype(np.float32)
    out *= m
    for _ in range(iterations):
        num = cv2.blur(out, (3, 3))
        den = cv2.blur(m, (3, 3))
        grow = (m == 0) & (den > 0)
        if not grow.any():
            break
        out[grow] = num[grow] / den[grow]
        m[grow] = 1.0
    out[m == 0] = out[m > 0].mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def shade_correct(green, inner, cfg=DEFAULT):
    """
    Background = median filter of the green channel at 1/bg_downscale resolution,
    upsampled. Returns (background, robust z-map, robust sigma).
    """
    d = cfg.bg_downscale
    size = (green.shape[1] // d, green.shape[0] // d)
    small = cv2.resize(green, size, interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(inner.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    m_small = cv2.erode(m_small, np.ones((3, 3), np.uint8),          # drop mixed rim pixels
                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    small = extend_outside(small, m_small, iterations=cfg.bg_kernel)
    bg_small = cv2.medianBlur(small, cfg.bg_kernel)
    # keep the upsampled background in float: integer residuals make the
    # robust sigma (MAD) jump in whole steps, which shifts every threshold
    bg = cv2.resize(bg_small.astype(np.float32), (green.shape[1], green.shape[0]),
                    interpolation=cv2.INTER_LINEAR)

    resid = green.astype(np.float32) - bg
    vals = resid[inner]
    med = float(np.median(vals))
    sigma = max(1.4826 * float(np.median(np.abs(vals - med))), 0.5)
    z = (resid - med) / sigma
    z[~inner] = 0.0
    return bg, z, sigma


def structure_hash(z, inner, size=16):
    """
    256-bit difference hash of the shade-corrected z-map, for near-duplicate detection.
    Hashing the vessel/lesion pattern (not raw brightness) separates different eyes,
    which all look alike at low resolution, while staying stable under rescaling,
    re-compression and exposure changes. Returned as a 64-character hex string.
    """
    zz = np.where(inner, np.clip(z, -4, 4), 0).astype(np.float32)
    small = cv2.resize(zz, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def preprocess(path, cfg=DEFAULT, timings=None):
    """
    Returns a dict with the normalized image and everything the feature
    families need, plus quality measurements (keys starting with q_).
    """
    with _timer(timings, "decode"):
        rgb_full, (orig_w, orig_h), factor = decode(path, cfg)
    with _timer(timings, "crop_resize"):
        rgb, side = crop_to_retina(rgb_full, cfg)
        mask = fov_mask(rgb, cfg.mask_threshold) > 0
        k = 2 * cfg.border_px + 1
        inner = cv2.erode(mask.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0   # also erode at the image edge
        if inner.sum() < 0.2 * cfg.work_size ** 2:
            raise ValueError("Retina region too small after cropping")
    with _timer(timings, "shade_correct"):
        green = rgb[:, :, 1]
        bg, z, sigma = shade_correct(green, inner, cfg)

    with _timer(timings, "quality"):
        mask_u8 = mask.astype(np.uint8)
        n_fov = max(int(mask_u8.sum()), 1)
        chan_max = cv2.max(cv2.max(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
        lap = cv2.Laplacian(green, cv2.CV_32F)
        _, lap_std = cv2.meanStdDev(lap, mask=inner.astype(np.uint8))
        quality = {
            "q_orig_w": orig_w,
            "q_orig_h": orig_h,
            "q_decode_factor": factor,
            "q_fov_diameter_px": int(side * factor),                 # retina size in the original file
            "q_fov_coverage": float(n_fov / (np.pi * (cfg.work_size / 2) ** 2)),
            "q_brightness": float(np.mean(cv2.mean(rgb, mask=mask_u8)[:3])),
            "q_dark_frac": float(np.count_nonzero((chan_max < 25) & mask) / n_fov),
            "q_saturated_frac": float(np.count_nonzero((chan_max >= 250) & mask) / n_fov),
            "q_noise_sigma": sigma,                                  # robust std of the residual
            "q_sharpness": float(lap_std[0, 0] ** 2),
            "q_dhash": structure_hash(z, inner),
        }

    return {"rgb": rgb, "mask": mask, "inner": inner, "green": green,
            "background": bg, "z": z, "quality": quality}
