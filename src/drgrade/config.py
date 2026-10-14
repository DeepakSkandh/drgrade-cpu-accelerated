"""
All pipeline parameters in one place.

Every image is cropped to its retina (field of view, FOV) and resized so the
retina is WORK_SIZE pixels across. All pixel parameters below are defined at
that size, so they mean the same physical scale for every dataset and camera.

The parameters are starting values chosen to transfer the earlier IDRiD
settings to this resolution; tune them by inspecting debug overlays.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # --- geometry -------------------------------------------------------
    work_size: int = 768            # retina diameter after crop + resize (px)
    mask_threshold: int = 10        # FOV mask: max(R, G, B) > threshold
    border_px: int = 18             # outer rim of the FOV ignored (vignetting)

    # --- shade correction -----------------------------------------------
    bg_downscale: int = 4           # background estimated at 1/4 resolution (speed)
    bg_kernel: int = 17             # median kernel at that resolution (~0.09 D)

    # --- dark structures: vessels vs red-lesion candidates --------------
    dark_k: float = 3.0             # dark if z < -dark_k (robust z units)
    red_min_area: int = 4           # smaller components = noise
    red_max_area: int = 3800        # larger components = vessel tree
    red_small_max_area: int = 64    # <= this: microaneurysm-sized candidate
    max_elongation: float = 3.0     # longer shapes are vessel-like
    min_solidity: float = 0.5       # very irregular shapes are vessel-like
    shape_check_min_area: int = 12  # shape measures unreliable below this
    red_quadrant_thresh: int = 5    # quadrant counts as "affected" at >= this

    # --- bright structures: exudate candidates, optic disc excluded ------
    bright_k: float = 3.5           # bright if z > bright_k
    bright_min_area: int = 4
    od_smooth_sigma: float = 6.0    # Gaussian sigma (at bg resolution) to locate the disc
    od_radius: int = 85             # excluded radius around the disc centre (~0.11 D)
    bright_quadrant_thresh: int = 3

    # --- texture ----------------------------------------------------------
    glcm_levels: int = 8
    glcm_z_clip: float = 3.0        # z-map clipped to [-clip, clip] before quantization

    # --- decoding -------------------------------------------------------
    fast_decode: bool = True        # use JPEG/PNG reduced decoding when the image is large

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
