#!/usr/bin/env python
"""Fig. 2 — validation learning curves (SSIM / PSNR) of the three arms, read from logs/*/log.txt.

Standard metrics only (brain-masked SSIM and PSNR). The internal composite scalar is NOT plotted.
Values are the trainer's batch-pooled validation numbers (per-epoch log), so PSNR is on a different
scale from the slice-level Table 1 — the caption must say so. Output: paper/figs/fig2_learning_curves.{png,pdf}
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/figs")
LOGS = {
    "GRU":            os.path.join(ROOT, "logs/PureETER_GRU_noDC_R4_brain384_v8/log.txt"),
    "SS2D":           os.path.join(ROOT, "logs/PureETER_SS2D_noDC_R4_brain384_v8/log.txt"),
    "enhanced SS2D":  os.path.join(ROOT, "logs/PureETER_SS2D_V9_unleashed_R4_brain384/log.txt"),
}
BEST_EP = {"GRU": 50, "SS2D": 48, "enhanced SS2D": 78}     # checkpoints reported in the tables
STYLE = {
    "GRU":           dict(color="#e34948", ls="-",  marker="o", ms=3, label="bi-GRU (ETER-Net original, 668M)"),
    "SS2D":          dict(color="#2a78d6", ls="-",  marker="s", ms=3, label="SS2D, controlled (31M)"),
    "enhanced SS2D": dict(color="#2e9e6b", ls="--", marker="^", ms=3, label="SS2D, enhanced (34M, 80 ep)"),
}
LINE_RE = re.compile(r'Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)'
                     r'(?:\s+val_composite=([\d.]+)\s+val_ssim_m=([\d.]+)\s+val_psnr=([\d.]+)'
                     r'\s+val_nmse=([\d.]+)\s+val_l1=([\d.]+))?')


def parse_log(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m and m.group(3) is not None:          # last occurrence wins (resumed runs re-log)
                out[int(m.group(1))] = dict(ssim=float(m.group(4)), psnr=float(m.group(5)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    traj = {k: parse_log(p) for k, p in LOGS.items()}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for key in ("ssim", "psnr"):
        ax = axes[0] if key == "ssim" else axes[1]
        for arm, t in traj.items():
            eps = sorted(t)
            ax.plot(eps, [t[e][key] for e in eps], lw=1.2, **STYLE[arm])
            be = BEST_EP[arm]
            if be in t:
                ax.plot([be], [t[be][key]], marker="*", ms=10, color=STYLE[arm]["color"], zorder=5)
        ax.axvline(50, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation SSIM (brain-masked)")
    axes[0].set_ylim(0.895, 0.916)
    axes[0].set_title("(a) SSIM", fontsize=10)
    axes[1].set_ylabel("validation PSNR (dB, batch-pooled)")
    axes[1].set_ylim(33.5, 35.4)
    axes[1].set_title("(b) PSNR", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="lower right", frameon=False)

    # (c) matched-epoch SSIM difference, SS2D - GRU, at the shared validation epochs (2..50)
    g, s = traj["GRU"], traj["SS2D"]
    eps = sorted(set(g) & set(s))
    d = [s[e]["ssim"] - g[e]["ssim"] for e in eps]
    ax = axes[2]
    ax.bar(eps, d, width=1.4, color=["#2a78d6" if v > 0 else "#e34948" for v in d])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ΔSSIM  (SS2D − GRU)")
    ax.set_title("(c) matched-epoch difference", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    n_pos = sum(v > 0 for v in d); n_tie = sum(v == 0 for v in d)
    ax.set_ylim(0, max(d) * 1.3)
    ax.text(0.97, 0.95, f"SS2D ≥ GRU at {n_pos + n_tie}/{len(d)} validation points\n(tie: {n_tie})",
            transform=ax.transAxes, va="top", ha="right", fontsize=8)
    fig.tight_layout()
    png = os.path.join(OUT, "fig2_learning_curves.png"); pdf = os.path.join(OUT, "fig2_learning_curves.pdf")
    fig.savefig(png, dpi=300); fig.savefig(pdf)
    print("saved:", png, pdf)
    for arm, t in traj.items():
        be = BEST_EP[arm]
        print(f"{arm:14s} n_val={len(t):2d} best-ep {be}: ssim={t[be]['ssim']:.4f} psnr={t[be]['psnr']:.2f}"
              f"  ep50: ssim={t[50]['ssim']:.4f}")
    print("matched-epoch: SS2D>GRU", n_pos, "tie", n_tie, "of", len(d), " min/max Δ", f"{min(d):+.4f}/{max(d):+.4f}")


if __name__ == "__main__":
    main()
