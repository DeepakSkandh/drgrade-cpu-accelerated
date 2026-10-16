"""
Presentation figures: run the DRGrade feature pipeline on ONE image and save a
slide-ready PNG for every stage. Uses the same functions that produced the
feature CSVs (src/drgrade), so the pictures show exactly what the model sees.

Examples (from the repository root):

  # random image from a folder
  python scripts/visualize_pipeline.py --images-dir ~/datasets/aptos/val_images/val_images

  # random image of a chosen grade (needs the labels CSV)
  python scripts/visualize_pipeline.py --images-dir ~/datasets/aptos/val_images/val_images \
      --labels ~/datasets/aptos/valid.csv --grade 3

  # a specific image, feature bars compared with the training set, no titles
  python scripts/visualize_pipeline.py --image ~/datasets/aptos/val_images/val_images/000c1434d8d7.png \
      --reference-csv data/processed/aptos_train_features_v2.csv --no-titles

Output folder (default results/figures/pipeline_<image_id>/):
  00_overview.png            all key stages on one slide
  01_original.png            input photograph
  02_retina_768.png          retina cropped, padded to square, resized to 768 px
  03_masks.png               field-of-view mask and inner analysis region
  04_green_channel.png       green channel (used for all structure features)
  05_background.png          estimated illumination background (median filter)
  06_zmap.png                robust z-map: blue = darker, red = brighter than background
  07_dark_pixels.png         all dark pixels (z < -3)
  08_vessels_red.png         vessel-like structures vs red-lesion candidates
  09_optic_disc.png          optic-disc estimate and excluded zone
  10_bright_candidates.png   bright (exudate-like) candidates
  11_texture.png             quantized z-map and its GLCM
  12_quadrants.png           candidate counts per image quadrant
  13_features.png            the 30 feature values (z-scores vs training set if given)
  features.csv               feature and quality values for this image
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Circle                                    # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from src.drgrade.config import DEFAULT                                   # noqa: E402
from src.drgrade.datasets import IMAGE_EXTS                              # noqa: E402
from src.drgrade.features import (                                       # noqa: E402
    FEATURE_FAMILIES, FEATURE_NAMES, GLCM_OFFSETS, bright_features, color_features,
    glcm_masked, locate_optic_disc, quantize_z, texture_features, vessel_and_red_features,
)
from src.drgrade.preprocess import decode, preprocess                    # noqa: E402

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
FAMILY_COLORS = {"color": "#d1495b", "texture": "#3c6e9f", "vessel": "#2e9e5b",
                 "red_lesion": "#8f3fb0", "bright": "#e09f1f"}
DPI = 200
ZMAP_CMAP = matplotlib.colormaps["RdBu_r"].copy()
ZMAP_CMAP.set_bad("black")                          # outside the retina: black, like the other panels


# ------------------------------------------------------------------ image choice
def pick_image(args):
    if args.image:
        p = Path(args.image).expanduser()
        if not p.is_file():
            sys.exit(f"Image not found: {p}")
        return p, None
    d = Path(args.images_dir).expanduser()
    if not d.is_dir():
        sys.exit(f"Folder not found: {d}")
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images in {d}")
    grades = {}
    if args.labels:
        lab = pd.read_csv(Path(args.labels).expanduser())
        lab.columns = lab.columns.str.strip()
        grades = dict(zip(lab["id_code"].astype(str).str.strip(), lab["diagnosis"].astype(int)))
        if args.grade is not None:
            files = [f for f in files if grades.get(f.stem) == args.grade]
            if not files:
                sys.exit(f"No images of grade {args.grade} in {d}")
    rng = random.Random(args.seed)
    f = rng.choice(files)
    return f, grades.get(f.stem)


# ------------------------------------------------------------------ drawing helpers
class Figs:
    def __init__(self, out_dir, titles):
        self.out, self.titles = out_dir, titles

    def image(self, name, img, title, cmap=None, overlay=None, colorbar=None, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if overlay:
            overlay(ax)
        ax.set_axis_off()
        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(colorbar, fontsize=10)
        if self.titles:
            ax.set_title(title, fontsize=13, pad=8)
        fig.tight_layout()
        fig.savefig(self.out / name, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def tint(rgb, mask, color, alpha=0.75):
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def dim_outside(rgb, inner, factor=0.35):
    out = rgb.astype(np.float32).copy()
    out[~inner] *= factor
    return out.astype(np.uint8)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="one image file")
    src.add_argument("--images-dir", help="folder: a random image is chosen")
    ap.add_argument("--labels", help="labels CSV (id_code, diagnosis) to show / filter the grade")
    ap.add_argument("--grade", type=int, choices=range(5), help="pick a random image of this grade")
    ap.add_argument("--seed", type=int, help="random seed (repeatable choice)")
    ap.add_argument("--reference-csv", help="training feature CSV: feature bars become z-scores")
    ap.add_argument("--out-dir", help="output folder (default results/figures/pipeline_<id>)")
    ap.add_argument("--no-titles", action="store_true", help="clean images without titles")
    args = ap.parse_args()
    if args.grade is not None and not args.labels:
        ap.error("--grade needs --labels")

    path, grade = pick_image(args)
    image_id = path.stem
    out = Path(args.out_dir) if args.out_dir else ROOT / "results" / "figures" / f"pipeline_{image_id}"
    out.mkdir(parents=True, exist_ok=True)
    F = Figs(out, not args.no_titles)
    cfg = DEFAULT
    gtxt = f"grade {grade} ({GRADE_NAMES[grade]})" if grade is not None else "grade unknown"
    print(f"image: {path}  |  {gtxt}")

    # ---------------- run the real pipeline ----------------
    rgb_orig, (ow, oh), factor = decode(path, cfg)
    pre = preprocess(path, cfg)
    rgb, mask, inner, green, bg, z = (pre[k] for k in ("rgb", "mask", "inner", "green", "background", "z"))
    disc = locate_optic_disc(green, inner, cfg)
    dr, dp = vessel_and_red_features(z, inner, cfg, od=disc[:2], return_parts=True)
    br, od, bp = bright_features(z, inner, disc, cfg, return_parts=True)
    feats = {**color_features(rgb, inner), **texture_features(z, inner, cfg), **dr, **br}
    quality = {**pre["quality"], **od}
    labels, stats, cents = dp["labels"], dp["stats"], dp["centroids"]
    areas = stats[:, 4]
    vessel_mask = np.isin(labels, dp["vessel_ids"])
    red_mask = np.isin(labels, dp["red_ids"])
    bright_mask = np.isin(bp["labels"], bp["ids"])
    size = cfg.work_size

    # ---------------- stage figures ----------------
    F.image("01_original.png", rgb_orig, f"Input photograph  ({ow} x {oh} px)")
    F.image("02_retina_768.png", rgb, f"Retina cropped, padded, resized to {size} px")

    m_vis = np.zeros((*mask.shape, 3), np.uint8)
    m_vis[mask] = (90, 90, 90)
    m_vis[inner] = (235, 235, 235)
    F.image("03_masks.png", m_vis, f"Field of view (grey) and inner region (white, rim {cfg.border_px} px removed)")
    F.image("04_green_channel.png", np.where(mask, green, 0), "Green channel", cmap="gray", vmin=0, vmax=255)
    F.image("05_background.png", np.where(mask, bg, 0), f"Illumination background (median {cfg.bg_kernel} px at 1/{cfg.bg_downscale} scale)",
            cmap="gray", vmin=0, vmax=max(1, float(bg[mask].max())))
    zshow = np.ma.masked_where(~inner, z)
    F.image("06_zmap.png", zshow, "Robust z-map (blue = darker, red = brighter)", cmap=ZMAP_CMAP,
            vmin=-6, vmax=6, colorbar="z (robust std units)")

    dark = (z < -cfg.dark_k) & inner
    F.image("07_dark_pixels.png", np.where(inner, ~dark, 0.5).astype(float),
            f"Dark pixels  z < -{cfg.dark_k:g}  ({int(dark.sum()):,} px)", cmap="gray", vmin=0, vmax=1)

    def red_circles(ax, ids=dp["red_ids"]):
        for i in ids:
            small = areas[i] <= cfg.red_small_max_area
            ax.add_patch(Circle(cents[i], np.sqrt(areas[i] / np.pi) + 6, fill=False, lw=1.4,
                                ec="#ffd400" if small else "#ff2bd6"))
    vr = tint(dim_outside(rgb, inner, 0.6), vessel_mask, (0, 220, 90), 0.8)
    F.image("08_vessels_red.png", vr,
            f"Vessel-like (green): {dr['vessel_n_components']} parts  |  red candidates: {dr['red_count']} "
            f"({dr['red_small_count']} small, yellow; {dr['red_large_count']} large, magenta)",
            overlay=red_circles)

    def disc_overlay(ax):
        ax.add_patch(Circle((od["q_od_x"], od["q_od_y"]), cfg.od_radius, fill=False, lw=2.2, ec="white", ls="--"))
        ax.plot(od["q_od_x"], od["q_od_y"], "+", color="white", ms=16, mew=2.5)
    F.image("09_optic_disc.png", rgb, f"Optic-disc estimate (peak z {od['q_od_peak_z']:.1f}) and excluded zone",
            overlay=disc_overlay)

    def bright_overlay(ax):
        disc_overlay(ax)
    F.image("10_bright_candidates.png", tint(rgb, bright_mask, (0, 230, 255), 0.85),
            f"Bright candidates (cyan): {br['bright_count']}  (z > {cfg.bright_k:g}, outside the disc)",
            overlay=bright_overlay)

    # texture: quantized map + direction-averaged GLCM
    q = quantize_z(z, cfg)
    P = np.mean([glcm_masked(q, inner, cfg.glcm_levels, dy, dx) for dy, dx in GLCM_OFFSETS], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 0.95]},
                             layout="constrained")
    axes[0].imshow(np.ma.masked_where(~inner, q), cmap="viridis", vmin=0, vmax=cfg.glcm_levels - 1,
                   interpolation="nearest")
    axes[0].set_axis_off()
    im = axes[1].imshow(P, cmap="magma")
    for i in range(cfg.glcm_levels):
        for j in range(cfg.glcm_levels):
            axes[1].text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if P[i, j] < P.max() * 0.6 else "black")
    axes[1].set_xlabel("neighbour level j")
    axes[1].set_ylabel("level i")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    if F.titles:
        axes[0].set_title(f"z-map quantized to {cfg.glcm_levels} levels", fontsize=12)
        axes[1].set_title("GLCM (4 directions averaged)", fontsize=12)
        fig.suptitle(f"Texture:  contrast {feats['tex_contrast']:.2f}  |  homogeneity {feats['tex_homogeneity']:.2f}  |  "
                     f"energy {feats['tex_energy']:.2f}  |  correlation {feats['tex_correlation']:.2f}", fontsize=12)
    fig.savefig(out / "11_texture.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # quadrants
    c = size / 2

    def quad_overlay(ax):
        red_circles(ax)
        ax.axhline(c, color="white", lw=1.5, ls="--")
        ax.axvline(c, color="white", lw=1.5, ls="--")
        rc = np.zeros(4, int)
        bc = np.zeros(4, int)
        for i in dp["red_ids"]:
            rc[int(cents[i, 0] >= c) + 2 * int(cents[i, 1] >= c)] += 1
        for i in bp["ids"]:
            bc[int(bp["centroids"][i, 0] >= c) + 2 * int(bp["centroids"][i, 1] >= c)] += 1
        for k, (x, y) in enumerate([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]):
            ax.text(x * size, y * size, f"red {rc[k]}\nbright {bc[k]}", ha="center", va="center",
                    fontsize=12, color="white", weight="bold",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.55, lw=0))
    F.image("12_quadrants.png", tint(rgb, bright_mask, (0, 230, 255), 0.85),
            "Candidates per image quadrant", overlay=quad_overlay)

    # feature values
    vals = pd.Series(feats)[FEATURE_NAMES].astype(float)
    fam = {f: k for k, v in FEATURE_FAMILIES.items() for f in v}
    fig, ax = plt.subplots(figsize=(9, 10))
    if args.reference_csv:
        ref = pd.read_csv(Path(args.reference_csv).expanduser())[FEATURE_NAMES].astype(float)
        zv = (vals - ref.mean()) / ref.std().replace(0, 1)
        plot, xlabel = zv, "z-score vs training set (0 = average image)"
        ax.axvline(0, color="black", lw=0.8)
    else:
        plot, xlabel = vals, "raw value (features have different scales)"
    order = list(reversed(FEATURE_NAMES))
    ax.barh(order, plot[order], color=[FAMILY_COLORS[fam[f]] for f in order])
    for k, f in enumerate(order):
        ax.text(plot[f], k, f"  {vals[f]:.3g}  ", va="center", fontsize=7,
                ha="left" if plot[f] >= 0 else "right")
    ax.set_xlabel(xlabel)
    ax.margins(x=0.18)                                  # room for the value labels inside the frame
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in FAMILY_COLORS.values()]
    ax.legend(handles, [k.replace("_", " ") for k in FAMILY_COLORS], loc="lower right", fontsize=9)
    if F.titles:
        ax.set_title(f"30 extracted features  |  {image_id}  |  {gtxt}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "13_features.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # overview slide
    panels = [
        ("Input", rgb_orig, None, {}),
        ("Retina 768 px", rgb, None, {}),
        ("Green channel", np.where(mask, green, 0), "gray", {"vmin": 0, "vmax": 255}),
        ("Background", np.where(mask, bg, 0), "gray", {"vmin": 0}),
        ("z-map", zshow, ZMAP_CMAP, {"vmin": -6, "vmax": 6}),
        ("Vessels + red candidates", vr, None, {}),
        ("Optic disc", rgb, None, {}),
        ("Bright candidates", tint(rgb, bright_mask, (0, 230, 255), 0.85), None, {}),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), layout="constrained")
    for k, (ax, (t, img, cmap, kw)) in enumerate(zip(axes.ravel(), panels)):
        ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
        if t == "Vessels + red candidates":
            red_circles(ax)
        if t in ("Optic disc", "Bright candidates"):
            disc_overlay(ax)
        ax.set_title(f"{k + 1}. {t}", fontsize=13)
        ax.set_axis_off()
    fig.suptitle(f"DRGrade feature pipeline  |  {image_id}  |  {gtxt}", fontsize=15)
    fig.savefig(out / "00_overview.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    row = {"image_id": image_id, "grade": grade, **quality, **{k: feats[k] for k in FEATURE_NAMES}}
    pd.DataFrame([row]).to_csv(out / "features.csv", index=False)
    print(f"red candidates {dr['red_count']} | bright candidates {br['bright_count']} | "
          f"vessel density {dr['vessel_density']:.3f} | disc at ({od['q_od_x']:.0f}, {od['q_od_y']:.0f})")
    print(f"saved {len(list(out.glob('*.png')))} figures + features.csv -> {out}")


if __name__ == "__main__":
    main()
