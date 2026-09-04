#!/usr/bin/env python
"""Fig. 3 — qualitative comparison (GT / bi-GRU / SS2D) composed from the existing 4-way visualisation
results/vis/v8_pure_eternet_compare/compare_3368.png (visualize_v8_pure_compare.py; per-slice LS scale-aligned,
brain-masked |error| with a shared 0–0.10 colour scale on the [0,1]-normalised GT scale).

The leaderboard U-Net column is dropped on purpose (its weights were trained on train+val — reference line
only, see the paper's baseline caveat). Row 2 applies a x4 display gain to reveal the background ringing of
the GRU arm outside the skull (not penalised by the brain-masked metrics). No inference is run here (CPU only).
Output: paper/figs/fig3_qualitative.{png,pdf}
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/vis/v8_pure_eternet_compare/compare_3368.png")
OUT = os.path.join(ROOT, "paper/figs")
# panel boxes (x0, x1, y0, y1) detected from the source PNG (dark-pixel projections)
TOP = {"GT": (14, 666), "GRU": (1474, 2126), "SS2D": (2203, 2855)}; TOP_Y = (147, 798)
ERR = {"GRU": (1474, 2063), "SS2D": (2203, 2792)}; ERR_Y = (874, 1462)
CBAR = (2825, 2926, 862, 1475)          # SS2D colour bar incl. tick labels
# per-slice metrics printed in the source panel titles (slice #3368, brain-masked, scale-aligned)
METR = {"GRU": "PSNR 34.02 dB / SSIM 0.9450", "SS2D": "PSNR 34.28 dB / SSIM 0.9483"}
GAIN = 4.0


def crop(arr, x0, x1, y0, y1):
    return arr[y0:y1 + 1, x0:x1 + 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    rgb = np.asarray(Image.open(SRC).convert("RGB"))
    gray = np.asarray(Image.open(SRC).convert("L")).astype(np.float32) / 255.0
    cols = [("GT", "Ground truth (RSS)"), ("GRU", "bi-GRU (ETER-Net original)"), ("SS2D", "SS2D (proposed replacement)")]
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.16], wspace=0.04, hspace=0.10,
                          left=0.06, right=0.98, top=0.95, bottom=0.01)
    row_labels = ["reconstruction", f"×{GAIN:.0f} display gain\n(background)", "masked |error|"]
    for j, (key, title) in enumerate(cols):
        x0, x1 = TOP[key]
        img = crop(gray, x0, x1, *TOP_Y)
        ax = fig.add_subplot(gs[0, j]); ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(title + ("\n" + METR[key] if key in METR else "\n"), fontsize=8.5,
                     color={"GRU": "#c62828", "SS2D": "#1565c0"}.get(key, "black"))
        ax = fig.add_subplot(gs[1, j]); ax.imshow(np.clip(img * GAIN, 0, 1), cmap="gray", vmin=0, vmax=1); ax.axis("off")
        if key in ERR:
            ex0, ex1 = ERR[key]
            ax = fig.add_subplot(gs[2, j]); ax.imshow(crop(rgb, ex0, ex1, *ERR_Y)); ax.axis("off")
        else:
            ax = fig.add_subplot(gs[2, j]); ax.axis("off")
            ax.text(0.5, 0.5, "|recon − GT| inside the\nbrain mask, both arms on\nthe same 0–0.10 colour scale\n(GT normalised to [0,1])",
                    ha="center", va="center", fontsize=8, color="0.35", transform=ax.transAxes)
    ax = fig.add_subplot(gs[2, 3]); ax.imshow(crop(rgb, *CBAR)); ax.axis("off")
    for i, lab in enumerate(row_labels):
        fig.text(0.015, 0.95 - (i + 0.5) * (0.95 - 0.01) / 3, lab, rotation=90, va="center", ha="center", fontsize=8.5)
    png = os.path.join(OUT, "fig3_qualitative.png"); pdf = os.path.join(OUT, "fig3_qualitative.pdf")
    fig.savefig(png, dpi=300); fig.savefig(pdf)
    print("saved:", png, pdf)


if __name__ == "__main__":
    main()
