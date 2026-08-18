"""Fig.4 — per-slice paired-difference distributions (draft_ko_v2 부록 A).

입력:  results/eval/v9_unleashed/per_slice_paired_v9.csv  (7,334 슬라이스,
       GRU/SS2D(통제판)/강화 SS2D 3모델 × SSIM/PSNR/NMSE/L1 per-slice 값)
출력:  paper/figs/fig4_per_slice_distribution.{png,pdf}

행 (a): 통제 비교  Δ = SS2D − GRU        (NMSE/L1 은 부호 반전 — 항상 양수 = SS2D 우위)
행 (b): 강화 비교  Δ = 강화판 − 통제판   (동일 규약 — 양수 = 강화판 우위)
x 축은 |Δ| 의 99.5 백분위로 대칭 클리핑(범위 밖 ≤0.5% 미표시, 그림 각주에 명시).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results/eval/v9_unleashed/per_slice_paired_v9.csv")
OUT_DIR = os.path.join(ROOT, "paper/figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- palette (dataviz skill 검증 완료: blue/red diverging, CVD ΔE 21.6) ----
C_POS = "#2a78d6"   # Δ>0 : 치환/강화 모델 우위
C_NEG = "#e34948"   # Δ<0 : 기존 모델 우위
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

cols = [
    "gru_ssim", "ss2d_ssim", "v9_ssim",
    "gru_psnr", "ss2d_psnr", "v9_psnr",
    "gru_nmse", "ss2d_nmse", "v9_nmse",
    "gru_l1", "ss2d_l1", "v9_l1",
]
d = np.genfromtxt(CSV, delimiter=",", names=True, usecols=cols, encoding="utf-8")
n = d.shape[0]

# metric: (표시 제목, x 배율, 배율 표기)
METRICS = [
    ("ssim", "SSIM", 1e3, r"$\Delta$ SSIM ($\times 10^{-3}$)"),
    ("psnr", "PSNR", 1.0, r"$\Delta$ PSNR (dB)"),
    ("nmse", "NMSE", 1e3, r"$\Delta$ NMSE ($\times 10^{-3}$)"),
    ("l1", "L1", 1.0, r"$\Delta$ L1"),
]
LOWER_BETTER = {"nmse", "l1"}

ROWS = [
    # (행 제목, baseline 접두, 치환 접두, 승자 표기)
    ("(a)  Controlled substitution:  SS2D  vs.  GRU", "gru", "ss2d", "SS2D"),
    ("(b)  Enhanced SS2D  vs.  controlled SS2D", "ss2d", "v9", "enhanced"),
]

fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6), facecolor="white")
plt.subplots_adjust(left=0.055, right=0.985, top=0.845, bottom=0.135,
                    hspace=0.62, wspace=0.24)

report = []
for r, (row_title, base_p, new_p, winner) in enumerate(ROWS):
    for c, (key, title, scale, xlabel) in enumerate(METRICS):
        ax = axes[r, c]
        a = d[f"{base_p}_{key}"]
        b = d[f"{new_p}_{key}"]
        delta = (a - b) if key in LOWER_BETTER else (b - a)   # 양수 = new 우위
        delta = delta * scale
        win = 100.0 * np.mean(delta > 0)
        report.append((row_title[:3], title, win, np.median(delta)))

        q = np.percentile(np.abs(delta), 99.5)
        bins = np.linspace(-q, q, 61)                          # 짝수 60칸 → 0 이 경계
        cnt, edges = np.histogram(delta, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        colors = [C_POS if x > 0 else C_NEG for x in centers]
        ax.bar(centers, cnt, width=(edges[1] - edges[0]),
               color=colors, edgecolor="white", linewidth=0.4, zorder=2)
        ax.axvline(0, ymax=0.72, color=INK2, linewidth=0.9, zorder=3)  # 주석 띠 아래까지만

        ax.set_xlim(-q, q)
        ax.set_ylim(0, cnt.max() * 1.34)   # 상단 여백 — 주석이 막대와 겹치지 않게
        ax.set_title(title, fontsize=9.5, color=INK, fontweight="bold", pad=4)
        ax.set_xlabel(xlabel, fontsize=8, color=INK2, labelpad=1.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=4)
        if c == 0:
            ax.set_ylabel("slices", fontsize=8, color=INK2)
        ax.annotate(f"{win:.1f}% favor {winner}",
                    xy=(0.03, 0.965), xycoords="axes fraction",
                    fontsize=8, color=INK, fontweight="bold",
                    ha="left", va="top")

    # 행 제목 (각 행 위)
    y = 0.925 if r == 0 else 0.455
    fig.text(0.055, y, row_title, fontsize=10, color=INK, fontweight="bold",
             ha="left", va="bottom")

fig.suptitle(
    f"Per-slice paired differences on the full validation set (n = {n:,} slices)",
    fontsize=11, color=INK, x=0.055, y=0.985, ha="left", fontweight="bold")
fig.legend(
    handles=[Patch(facecolor=C_POS, label=r"$\Delta>0$: slice favors the replacement"),
             Patch(facecolor=C_NEG, label=r"$\Delta<0$: slice favors the baseline")],
    loc="upper right", bbox_to_anchor=(0.985, 1.002), ncol=2, frameon=False,
    fontsize=8, handlelength=1.2, handleheight=0.9, labelcolor=INK2)
fig.text(0.055, 0.022,
         "Signs are oriented so that positive always favors the replacement (NMSE/L1 differences are negated). "
         "x-axes span the 99.5th percentile of |Δ| per panel; the ≤0.5% of slices beyond this range are not shown.",
         fontsize=7.2, color=MUTED, ha="left")

png = os.path.join(OUT_DIR, "fig4_per_slice_distribution.png")
pdf = os.path.join(OUT_DIR, "fig4_per_slice_distribution.pdf")
fig.savefig(png, dpi=300)
fig.savefig(pdf)
print(f"saved: {png}\nsaved: {pdf}\n")
print(f"{'row':4s} {'metric':6s} {'win%':>6s} {'median Δ(scaled)':>18s}")
for row, m, w, md in report:
    print(f"{row:4s} {m:6s} {w:6.1f} {md:18.4f}")
