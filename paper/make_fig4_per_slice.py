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
from matplotlib import font_manager as fm, rcParams
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results/eval/v9_unleashed/per_slice_paired_v9.csv")
OUT_DIR = os.path.join(ROOT, "paper/figs")
os.makedirs(OUT_DIR, exist_ok=True)

# 09-09: 학술지판 page 폭(6.69 in)에 실제 크기로 삽입되므로 그 폭으로 직접 조판한다(축소 없음 → 최소 글자 6.5 pt).
# 글꼴은 그림 1·2·학술대회판과 동일한 Liberation Sans(paper/fonts/, OFL); 수식(Δ·×10⁻³)도 같은 글꼴로.
PAGE_W = 6.69
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    _p = os.path.join(ROOT, "paper/fonts", _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist):
    rcParams["font.family"] = "Liberation Sans"
    rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
                     "mathtext.cal": "Liberation Sans"})
FS = dict(suptitle=8.5, row=8, title=8, label=7, tick=6.5, note=7, foot=6.5, legend=6.5)

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
    ("nmse", "nMSE", 1e2, r"$\Delta$ nMSE (%)"),   # 표와 같은 백분율 단위 (09-09)
    ("l1", "L1", 1.0, r"$\Delta$ L1"),
]
LOWER_BETTER = {"nmse", "l1"}

ROWS = [
    # (행 제목, baseline 접두, 치환 접두, 승자 표기)
    ("(a)  Controlled substitution:  SS2D  vs.  GRU", "gru", "ss2d", "SS2D"),
    ("(b)  Enhanced SS2D  vs.  controlled SS2D", "ss2d", "v9", "enhanced"),
]

fig, axes = plt.subplots(2, 4, figsize=(PAGE_W, 3.7), facecolor="white")
LEFT, RIGHT = 0.072, 0.988
plt.subplots_adjust(left=LEFT, right=RIGHT, top=0.85, bottom=0.165,
                    hspace=0.75, wspace=0.30)   # 행 사이 = 행 (a) x 라벨 + 행 (b) 제목 + 행 (b) 패널 제목

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
        ax.set_title(title, fontsize=FS["title"], color=INK, fontweight="bold", pad=3)
        ax.set_xlabel(xlabel, fontsize=FS["label"], color=INK2, labelpad=1.5)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=FS["tick"], length=2, width=0.5, pad=2)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=4)
        if c == 0:
            ax.set_ylabel("slices", fontsize=FS["label"], color=INK2, labelpad=2)
        ax.annotate(f"{win:.1f}% favor {winner}",
                    xy=(0.03, 0.965), xycoords="axes fraction",
                    fontsize=FS["note"], color=INK, fontweight="bold",
                    ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))   # 격자선 가림

    # 행 제목 (각 행 위)
    y = 0.90 if r == 0 else 0.463
    fig.text(LEFT, y, row_title, fontsize=FS["row"], color=INK, fontweight="bold",
             ha="left", va="bottom")

fig.suptitle(
    f"Per-slice paired differences on the full validation set (n = {n:,} slices)",
    fontsize=FS["suptitle"], color=INK, x=LEFT, y=0.992, ha="left", fontweight="bold")
# 범례는 page 폭에서 제목과 한 줄에 못 들어가므로 우상단 2행 세로 배치(행 (a) 제목과 좌우로 분리)
fig.legend(
    handles=[Patch(facecolor=C_POS, label=r"$\Delta>0$: slice favors the replacement"),
             Patch(facecolor=C_NEG, label=r"$\Delta<0$: slice favors the baseline")],
    loc="upper right", bbox_to_anchor=(RIGHT, 1.005), ncol=1, frameon=False,
    fontsize=FS["legend"], handlelength=1.2, handleheight=0.8, labelcolor=INK2,
    labelspacing=0.25, borderpad=0.2)
fig.text(LEFT, 0.012,
         "Signs are oriented so that positive always favors the replacement (nMSE/L1 differences are negated).\n"
         "x-axes span the 99.5th percentile of |Δ| per panel; the ≤0.5% of slices beyond this range are not shown.",
         fontsize=FS["foot"], color=MUTED, ha="left", va="bottom", linespacing=1.35)

png = os.path.join(OUT_DIR, "fig4_per_slice_distribution.png")
pdf = os.path.join(OUT_DIR, "fig4_per_slice_distribution.pdf")
fig.savefig(png, dpi=600)
fig.savefig(pdf)
print(f"saved: {png}\nsaved: {pdf}\n")
print(f"{'row':4s} {'metric':6s} {'win%':>6s} {'median Δ(scaled)':>18s}")
for row, m, w, md in report:
    print(f"{row:4s} {m:6s} {w:6.1f} {md:18.4f}")
