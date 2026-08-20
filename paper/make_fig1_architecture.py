"""Fig.1 — 아키텍처 다이어그램 (draft_ko_v2 §3.2/§3.3).

(a) 순수 ETER-Net 공통 골격 (v8 통제비교): 유일 변수 = 시퀀스 모델 슬롯(bi-GRU vs SS2D).
(b) 강화 SS2D (v9): stem → ds=3 coarse scan → 잔차 SS2D 블록 ×3(게이팅) → 업샘플 → head.

색 규약은 Fig.4 와 동일: blue = SS2D(치환/강화), red 계열 = GRU(기존), 회색 = 공유 구성요소.
출력: paper/figs/fig1_architecture.{png,pdf}
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/figs")
os.makedirs(OUT, exist_ok=True)

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
BLUE = "#2a78d6"; RED = "#e34948"
FILL_N = "#f0efec"; EDGE_N = "#c3c2b7"          # 공유(중립)
FILL_B = "#e3eefb"; FILL_R = "#fbe7e7"          # SS2D / GRU 연한 배경


def box(ax, x, y, w, h, lines, fc=FILL_N, ec=EDGE_N, lw=1.1, fs=8.2,
        title=None, tfs=8.6, tc=INK, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    cy = y + h / 2
    txt = ([] if title is None else [(title, tfs, tc, "bold")]) + \
          [(t, fs, INK2, "normal") for t in lines]
    n = len(txt)
    for i, (t, s, c, wgt) in enumerate(txt):
        yy = cy + (n - 1 - 2 * i) * (h / (2 * n)) * 0.82
        ax.text(x + w / 2, yy, t, ha="center", va="center",
                fontsize=s, color=c, fontweight=wgt, zorder=3)


def arrow(ax, p1, p2, cs="arc3,rad=0", color=INK2, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=cs, zorder=1,
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, shrinkA=1, shrinkB=1))


fig = plt.figure(figsize=(12.5, 8.2), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0],
                      left=0.01, right=0.99, top=0.965, bottom=0.03, hspace=0.16)

# ============================ (a) 공통 골격 ============================
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 34)
ax.axis("off"); ax.set_facecolor("white")

ax.text(1, 33.2, "(a)  Pure ETER-Net backbone (controlled comparison)  —  no ViT, no data-consistency block",
        fontsize=10.5, color=INK, fontweight="bold", va="top")

# 입력
box(ax, 2, 19.5, 18, 7, [r"$\tilde{y}_c = M \odot y_c$", "32 ch (16 coils × Re/Im), 384²"],
    title="Undersampled k-space")
box(ax, 2, 3.5, 18, 7, [r"$F^{-1}\tilde{y}_c$  (zero-filled)", "32 ch, 384²"],
    title="Aliased coil images")

# 시퀀스 모델 슬롯 (유일 변수) — 라벨은 프레임 안 상단
ax.add_patch(FancyBboxPatch((26, 8.0), 28, 21, boxstyle="round,pad=0.35",
                            fc="none", ec=INK, lw=1.4, linestyle=(0, (4, 2)), zorder=1))
ax.text(40, 27.8, r"sequence model  $f_\theta$  —  the only variable",
        ha="center", fontsize=9.2, color=INK, fontweight="bold")
box(ax, 28, 19.6, 24, 6.6, ["horizontal + vertical bidirectional", "flatten–reshape  ·  arm total 668M"],
    title="bi-GRU  (ETER-Net original)", fc=FILL_R, ec=RED)
ax.text(40, 18.3, "or", ha="center", fontsize=9, color=MUTED, style="italic")
box(ax, 28, 10.8, 24, 6.6, ["4-direction selective scan", "single block, 20 ch out  ·  arm total 31M"],
    title="SS2D  (selective state space)", fc=FILL_B, ec=BLUE)

# concat
ax.add_patch(plt.Circle((60.5, 12.5), 2.1, fc="white", ec=INK2, lw=1.3, zorder=2))
ax.text(60.5, 12.5, "C", ha="center", va="center", fontsize=10, color=INK,
        fontweight="bold", zorder=3)
ax.text(58.2, 9.5, "channel\nconcat", ha="right", va="center", fontsize=7.6, color=INK2)

# U-Net + 출력
box(ax, 66, 8.5, 20, 8, ["skip connections, depth 5, wf 6", "~30M  ·  identical in both arms"],
    title=r"De-aliasing U-Net  $g_\phi$")
box(ax, 90, 9.5, 9, 6, ["1 × 384²"], title="Magnitude\nimage", fs=8.0, tfs=8.2)

# 화살표
arrow(ax, (20, 23), (25.5, 23))                                   # ksp → slot
arrow(ax, (54.9, 16.5), (59.1, 13.9), cs="arc3,rad=-0.2")         # slot → concat
arrow(ax, (20, 7), (60.5, 10.15),                                 # aliased → (프레임 아래 엘보) → concat
      cs="angle,angleA=0,angleB=90,rad=4")
arrow(ax, (62.7, 12.5), (65.6, 12.5))                             # concat → U-Net
arrow(ax, (86, 12.5), (89.6, 12.5))                               # U-Net → out

ax.text(2, 0.9, "All components except the sequence module are identical across the two arms: "
                "data loader, undersampling masks, loss, optimizer, schedule, epochs, and seed.",
        fontsize=8, color=MUTED)

# ============================ (b) 강화 SS2D ============================
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 26)
ax.axis("off"); ax.set_facecolor("white")

ax.text(1, 25.4, r"(b)  Enhanced SS2D  —  drop-in replacement for the  $f_\theta$  slot in (a)",
        fontsize=10.5, color=INK, fontweight="bold", va="top")

box(ax, 2, 11, 12, 7, ["32 ch, 384²"], title="k-space\ninput", fs=8.0)
box(ax, 18, 11, 14, 7, ["full resolution", "384²"], title="Stem")
box(ax, 36, 11, 14, 7, ["coarse-scan grid", "384² → 128²"], title="Downsample\nds = 3")

# 잔차 SS2D 블록 ×3 (뒤에 그림자 사각형 2개로 스택 표현)
for off in (1.6, 0.8):
    ax.add_patch(FancyBboxPatch((54 + off, 8.5 + off), 26, 12, boxstyle="round,pad=0.25",
                                fc="white", ec=BLUE, lw=1.0, zorder=1))
box(ax, 54, 8.5, 26, 12,
    [r"in_proj → (x, z)   ·   4-direction scan",
     "fp16 selective scan,  d_inner 256,  d_state 32",
     r"gating  y ⊙ SiLU(z)   ·   + residual"],
    title="Residual SS2D block  × 3", fc=FILL_B, ec=BLUE)

box(ax, 84, 13.5, 15, 6.5, ["bilinear, 128² → 384²"], title="Upsample", fs=8.0)
box(ax, 84, 4.5, 15, 6.5, ["output 64 ch", "→ concat & U-Net as in (a)"], title="Head", fs=8.0)

arrow(ax, (14, 14.5), (17.6, 14.5))
arrow(ax, (32, 14.5), (35.6, 14.5))
arrow(ax, (50, 14.5), (53.6, 14.5))
arrow(ax, (80.8, 15.5), (83.6, 16.2))
arrow(ax, (91.5, 13.1), (91.5, 11.4))

ax.text(2, 1.2, "Enhancements over the controlled SS2D: gating restored · 3 residual blocks · bottleneck lifted "
                "(20 → 64 ch, d_inner 128 → 256, d_state 16 → 32) · SSM stack ~2M.\n"
                "fp16 + coarse scan make it faster per epoch than the controlled arm "
                "(2.51 vs 2.78 h/epoch) — enabling 80 instead of 50 epochs in a comparable budget.",
        fontsize=8, color=MUTED, va="bottom")

png = os.path.join(OUT, "fig1_architecture.png")
pdf = os.path.join(OUT, "fig1_architecture.pdf")
fig.savefig(png, dpi=300); fig.savefig(pdf)
print("saved:", png, "\nsaved:", pdf)
