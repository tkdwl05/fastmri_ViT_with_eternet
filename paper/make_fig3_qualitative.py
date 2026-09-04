#!/usr/bin/env python
"""Fig. 3 — multi-model qualitative comparison (professor's ETER-Net paper layout):
   GT | Zero-filled | U-Net† | E2E-VarNet† | PromptMR+ | bi-GRU (original) | SS2D (controlled) | Enhanced SS2D
   for several validation slices of different contrasts; per slice two rows — reconstruction (RSS magnitude,
   per-slice LS scale-aligned inside the brain mask) and brain-masked |error| on a shared 0–0.10 colour scale
   of the [0,1]-normalised GT — plus one ×GAIN display-gain row of a single slice that reveals the background
   ringing of the bi-GRU arm outside the skull (not penalised by the brain-masked metrics).

   Composition only: reconstructions/metrics come from `visualize_multimodel_compare.py`
   (results/vis/multimodel_compare/recon_<idx>.npz + metrics_<idx>.json; CPU fp32 inference, same
   384/R4/16-coil pipeline for every method). † = leaderboard weights trained on train+val (reference only).
Output: paper/figs/fig3_qualitative.{png,pdf}
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/vis/multimodel_compare")
OUT = os.path.join(ROOT, "paper/figs")

COLS = [  # (npz key, header line 1, header line 2)
    ("gt",       "Ground truth", ""),
    ("zf",       "Zero-filled", ""),
    ("unet",     "U-Net†", ""),
    ("varnet",   "E2E-VarNet†", ""),
    ("promptmr", "PromptMR+", ""),
    ("gru",      "bi-GRU", "(original)"),
    ("ss2d",     "SS2D", "(controlled)"),
    ("v9",       "Enhanced", "SS2D"),
]
ERR_VMAX = 0.10


def load_slice(idx):
    z = np.load(os.path.join(SRC, f"recon_{idx:04d}.npz"))
    meta = json.load(open(os.path.join(SRC, f"metrics_{idx:04d}.json")))
    rec = {k[4:]: z[k] for k in z.files if k.startswith("rec_")}
    return z["gt"], z["brain_mask"], rec, meta["meta"], meta["metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="3368,1321,0", help="dataset indices (rows: recon + |error| per slice)")
    ap.add_argument("--gain-slice", type=int, default=3368, help="slice shown in the ×gain row (-1 = none)")
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--panel-in", type=float, default=0.86, help="panel width in inches")
    ap.add_argument("--out-name", default="fig3_qualitative")
    args = ap.parse_args()
    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    os.makedirs(OUT, exist_ok=True)

    n_rows = 2 * len(slices) + (1 if args.gain_slice >= 0 else 0)
    n_cols = len(COLS)
    W = args.panel_in
    fig_w = n_cols * W + 0.62            # + colourbar column / row labels
    fig_h = n_rows * W + 0.42            # + column headers
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.10],
                          wspace=0.03, hspace=0.03,
                          left=0.045, right=0.945, top=1 - 0.40 / fig_h, bottom=0.01)
    hot = plt.get_cmap("hot").copy(); hot.set_bad("black")
    txt_fx = [pe.withStroke(linewidth=1.6, foreground="black")]
    im_err = None
    letters = "abcdefgh"

    def show_metrics(ax, m):
        if m is None:
            return
        ax.text(0.03, 0.03, f"{m['psnr']:.2f} dB / {m['ssim']:.3f}", transform=ax.transAxes,
                fontsize=5.4, color="white", ha="left", va="bottom", path_effects=txt_fx)

    r = 0
    for si, idx in enumerate(slices):
        gt, bm, rec, meta, met = load_slice(idx)
        gmax = max(float(gt.max()), 1e-8)
        mb = bm > 0.5
        for ci, (key, h1, h2) in enumerate(COLS):
            ax0 = fig.add_subplot(gs[r, ci]); ax1 = fig.add_subplot(gs[r + 1, ci])
            for ax in (ax0, ax1):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if key == "gt":
                ax0.imshow(gt / gmax, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax1.imshow(np.where(mb, 0.0, np.nan), cmap=hot, vmin=0, vmax=ERR_VMAX, interpolation="nearest")
                ax1.text(0.5, 0.5, "|error|\n(brain mask)", transform=ax1.transAxes, fontsize=5.6,
                         color="white", ha="center", va="center")
                # row labels (contrast) on the left
                ax0.text(-0.06, 0.5, f"{meta['contrast']}\nslice {meta['slice']}", transform=ax0.transAxes,
                         rotation=90, fontsize=6.0, ha="right", va="center")
            elif key in rec:
                x = rec[key] / gmax
                ax0.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                im_err = ax1.imshow(np.where(mb, np.abs(x - gt / gmax), np.nan), cmap=hot,
                                    vmin=0, vmax=ERR_VMAX, interpolation="nearest")
                show_metrics(ax0, met.get(key))
            else:
                ax0.text(0.5, 0.5, "n/a", transform=ax0.transAxes, ha="center", va="center", fontsize=7)
            if r == 0:
                hdr = f"({letters[ci]}) {h1}" + (f"\n{h2}" if h2 else "\n")
                ax0.set_title(hdr, fontsize=6.6, pad=2.5, linespacing=1.05)
        r += 2

    if args.gain_slice >= 0:
        gt, bm, rec, meta, met = load_slice(args.gain_slice)
        gmax = max(float(gt.max()), 1e-8)
        for ci, (key, h1, h2) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, ci])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            src = gt if key == "gt" else rec.get(key)
            if src is None:
                continue
            ax.imshow(np.clip(args.gain * src / gmax, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if key == "gt":
                ax.text(-0.06, 0.5, f"×{args.gain:g} gain\n({meta['contrast']} s{meta['slice']})",
                        transform=ax.transAxes, rotation=90, fontsize=6.0, ha="right", va="center")

    if im_err is not None:
        cax = fig.add_subplot(gs[1:2 * len(slices), n_cols])
        cb = fig.colorbar(im_err, cax=cax, ticks=[0, 0.05, 0.10])
        cb.ax.tick_params(labelsize=5.4, length=2, pad=1.5)
        cb.set_label("|recon − GT| / max(GT)", fontsize=5.6, labelpad=1.5)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{args.out_name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"saved paper/figs/{args.out_name}.png/.pdf  ({fig_w:.2f}×{fig_h:.2f} in, {n_rows}×{n_cols} panels; slices {slices}, gain row {args.gain_slice})")


if __name__ == "__main__":
    main()
