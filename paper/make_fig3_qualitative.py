#!/usr/bin/env python
"""Fig. 3 — multi-model qualitative comparison (professor's ETER-Net paper layout):
   Ground truth | Zero-filled | U-Net† | E2E-VarNet† | PromptMR+ | bi-GRU (original) | SS2D (controlled) | SS2D (enhanced)
   for several validation slices of different contrasts; per slice two rows — reconstruction (RSS magnitude,
   per-slice least-squares intensity-scaled inside the brain mask) and brain-masked |error| on a shared 0–0.10 colour scale
   of the [0,1]-normalised GT — plus one ×GAIN display-gain row of a single slice that reveals the background
   ringing of the bi-GRU model outside the skull (not penalised by the brain-masked metrics).

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
from matplotlib import font_manager as fm, rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/vis/multimodel_compare")
OUT = os.path.join(ROOT, "paper/figs")

# 09-09: 학술지판 page 폭(6.69 in)에 실제 크기로 삽입되므로 그 폭으로 직접 조판(축소 없음). 패널 폭은 page 폭에서
# 역산하고, dpi 는 384 px 슬라이스가 패널에 1:1 로 들어가도록 잡는다(리샘플 없음). 글꼴은 그림 1·2·4 와 같은 Liberation Sans.
PAGE_W = 6.69
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    _p = os.path.join(ROOT, "paper/fonts", _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist):
    rcParams["font.family"] = "Liberation Sans"
FS = dict(metric=6.5, errlabel=6.5, rowlabel=6.5, header=6.5, cbar_tick=6.0, cbar_label=6.5)   # 10-01: 글자 6 pt 이상

COLS = [  # (npz key, header line 1, header line 2)
    ("gt",       "Ground truth", ""),
    ("zf",       "Zero-filled", ""),
    ("unet",     "U-Net†", ""),
    ("varnet",   "E2E-VarNet†", ""),
    ("promptmr", "PromptMR+", ""),
    ("gru",      "ETER-net", "(bi-GRU)"),
    ("ss2d",     "SS2D", "(controlled)"),
    ("v9",       "SS2D", "(enhanced)"),
]
NARROW_HDR = {"gt": ("Ground", "truth"), "zf": ("Zero-", "filled"), "unet": ("U-Net†", ""),
              "varnet": ("E2E-", "VarNet†"), "promptmr": ("Prompt", "MR+")}
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
    ap.add_argument("--panel-in", type=float, default=None,
                    help="panel width in inches (default: fit 8 panels + margins into the 6.69 in page width)")
    ap.add_argument("--out-name", default="fig3_qualitative")
    # 09-09 통합판(학술지 4쪽 = 학술대회판 동일 내용): 단 폭 3.15 in 에 열을 골라 넣는 축약형 — 예) --methods gt,zf,gru,ss2d --fig-w 3.15
    ap.add_argument("--methods", default=None, help="comma-separated npz keys to show, in order (default: all 8 columns)")
    ap.add_argument("--fig-w", type=float, default=PAGE_W, help="total figure width in inches (6.69 page / 3.15 column)")
    args = ap.parse_args()
    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    os.makedirs(OUT, exist_ok=True)
    cols = COLS if not args.methods else [c for k in args.methods.split(",") for c in COLS if c[0] == k.strip()]
    if args.fig_w < 4.0:   # 단 폭: 한 줄 헤더가 패널(≈0.62 in)보다 길어지므로 두 줄로 나눈다
        cols = [(k, *NARROW_HDR.get(k, (h1, h2))) for k, h1, h2 in cols]

    n_rows = 2 * len(slices) + (1 if args.gain_slice >= 0 else 0)
    n_cols = len(cols)
    L_IN, R_IN = 0.30, 0.38              # row-label margin / colourbar tick+label margin (inches, same at every width)
    CB = 0.10                            # colourbar column = 0.10 panel width (inside the grid)
    W = args.panel_in if args.panel_in else (args.fig_w - L_IN - R_IN) / (n_cols + CB)   # 0.749 in at page width (8 cols)
    dpi = int(round(384 / W))            # one slice pixel = one image pixel (512 dpi at page width)
    fig_w = (n_cols + CB) * W + L_IN + R_IN
    fig_h = n_rows * W + 0.34            # + column headers
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [CB],
                          wspace=0.03, hspace=0.03,
                          left=L_IN / fig_w, right=1 - R_IN / fig_w, top=1 - 0.32 / fig_h, bottom=0.01)
    hot = plt.get_cmap("hot").copy(); hot.set_bad("black")
    txt_fx = [pe.withStroke(linewidth=1.6, foreground="black")]
    im_err = None
    letters = "abcdefgh"

    def show_metrics(ax, m):
        if m is None:
            return
        ax.text(0.03, 0.03, f"{m['psnr']:.2f} / {m['ssim']:.3f}", transform=ax.transAxes,
                fontsize=FS["metric"], color="white", ha="left", va="bottom", path_effects=txt_fx)

    r = 0
    for si, idx in enumerate(slices):
        gt, bm, rec, meta, met = load_slice(idx)
        gmax = max(float(gt.max()), 1e-8)
        mb = bm > 0.5
        for ci, (key, h1, h2) in enumerate(cols):
            ax0 = fig.add_subplot(gs[r, ci]); ax1 = fig.add_subplot(gs[r + 1, ci])
            for ax in (ax0, ax1):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if key == "gt":
                ax0.imshow(gt / gmax, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax1.imshow(np.where(mb, 0.0, np.nan), cmap=hot, vmin=0, vmax=ERR_VMAX, interpolation="nearest")
                ax1.text(0.5, 0.5, "|error|\n(brain mask)", transform=ax1.transAxes, fontsize=FS["errlabel"],
                         color="white", ha="center", va="center")
                # row labels (contrast) on the left
                ax0.text(-0.06, 0.5, f"{meta['contrast']}", transform=ax0.transAxes,
                         rotation=90, fontsize=FS["rowlabel"], ha="right", va="center")
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
                ax0.set_title(hdr, fontsize=FS["header"], pad=2.5, linespacing=1.05)
        r += 2

    if args.gain_slice >= 0:
        gt, bm, rec, meta, met = load_slice(args.gain_slice)
        gmax = max(float(gt.max()), 1e-8)
        for ci, (key, h1, h2) in enumerate(cols):
            ax = fig.add_subplot(gs[r, ci])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            src = gt if key == "gt" else rec.get(key)
            if src is None:
                continue
            ax.imshow(np.clip(args.gain * src / gmax, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if key == "gt":
                ax.text(-0.06, 0.5, f"×{args.gain:g} gain",
                        transform=ax.transAxes, rotation=90, fontsize=FS["rowlabel"], ha="right", va="center")

    if im_err is not None:
        cax = fig.add_subplot(gs[1:2 * len(slices), n_cols])
        cb = fig.colorbar(im_err, cax=cax, ticks=[0, 0.05, 0.10])
        cb.ax.tick_params(labelsize=FS["cbar_tick"], length=2, width=0.5, pad=1.5)
        cb.set_label(r"|$\hat{x}$ − $x^{*}$| / max($x^{*}$)", fontsize=FS["cbar_label"], labelpad=1.5)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{args.out_name}.{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"saved paper/figs/{args.out_name}.png/.pdf  ({fig_w:.2f}×{fig_h:.2f} in @ {dpi} dpi, {n_rows}×{n_cols} panels; slices {slices}, gain row {args.gain_slice})")


if __name__ == "__main__":
    main()
