"""
Debug overlay: one PNG showing every stage, for checking the detectors by eye.

    python -m src.drgrade.debug <image> <out.png>
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT
from .features import vessel_and_red_features, bright_features, locate_optic_disc
from .preprocess import preprocess


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_debug(path, out_path, cfg=DEFAULT):
    pre = preprocess(path, cfg)
    rgb, z, inner, green = pre["rgb"], pre["z"], pre["inner"], pre["green"]
    disc = locate_optic_disc(green, inner, cfg)
    dr, dparts = vessel_and_red_features(z, inner, cfg, od=disc[:2], return_parts=True)
    br, od, bparts = bright_features(z, inner, disc, cfg, return_parts=True)

    # 1. normalized retina
    p1 = _label(rgb, f"retina {cfg.work_size}px (orig {pre['quality']['q_orig_w']}x{pre['quality']['q_orig_h']})")

    # 2. z-map: blue = darker than background, red = brighter
    zc = np.clip(z / 6.0, -1, 1)
    zmap = np.zeros_like(rgb)
    zmap[..., 0] = (np.clip(zc, 0, 1) * 255).astype(np.uint8)
    zmap[..., 2] = (np.clip(-zc, 0, 1) * 255).astype(np.uint8)
    p2 = _label(zmap, "z-map: blue dark, red bright")

    # 3. vessels (green) + red-lesion candidates (yellow small, magenta large)
    p3 = rgb.copy()
    lab = dparts["labels"]
    p3[np.isin(lab, dparts["vessel_ids"])] = (0, 255, 0)
    areas = dparts["stats"][:, cv2.CC_STAT_AREA]
    for i in dparts["red_ids"]:
        cx, cy = dparts["centroids"][i]
        r = int(np.sqrt(areas[i] / np.pi)) + 5
        col = (255, 255, 0) if areas[i] <= cfg.red_small_max_area else (255, 0, 255)
        cv2.circle(p3, (int(cx), int(cy)), r, col, 1)
    p3 = _label(p3, f"vessels (green), red cand.: {dr['red_count']} "
                    f"({dr['red_small_count']} small)")

    # 4. bright candidates (cyan) + optic disc exclusion (white circle)
    p4 = rgb.copy()
    p4[np.isin(bparts["labels"], bparts["ids"])] = (0, 255, 255)
    cv2.circle(p4, (int(od["q_od_x"]), int(od["q_od_y"])), cfg.od_radius, (255, 255, 255), 2)
    p4 = _label(p4, f"bright cand.: {br['bright_count']} | disc peak z {od['q_od_peak_z']:.1f}")

    panel = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return out_path


if __name__ == "__main__":
    render_debug(sys.argv[1], sys.argv[2])
    print("Saved", sys.argv[2])
