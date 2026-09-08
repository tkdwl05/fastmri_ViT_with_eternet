"""학술대회판(IEIE 2단, 단 폭 3.15 in) 전용 아키텍처 그림 3장 — 옛 그림 1(2패널 광폭) 을 3장으로 분할.

  conf_fig1_pipeline.png : 데이터 → 마스크 → f_θ(유일 변수) / zero-filled → concat → U-Net → 출력 → 손실·GT
  conf_fig2_arms.png     : 두 팔의 내부 — (a) bi-GRU flatten→2단 bi-GRU→reshape, (b) SS2D 4방향 selective scan
  conf_fig3_enhanced.png : 강화 SS2D — stem → ds=3 → 잔차 게이팅 블록 ×3 → 업샘플 → head + 블록 내부

물리 크기(figsize)를 최종 인쇄 폭(3.15 in)으로 잡아 글자 크기(pt)가 그대로 지면 크기가 되게 했고,
박스 안 글자는 렌더러로 실제 폭을 재서 박스를 넘치면 자동 축소한다(최소 4.6 pt; 축소 시 stderr 경고).
수치 출처: models/pure_eternet/u_pure_eternet_{gru,ss2d}.py, models/mamba_eternet/ss2d{,_v9}.py,
           paper/make_tables.py(params), CLAUDE.md(34.2M 분해·h/ep). 색 규약 = Fig.2/Fig.4 와 동일
(red = bi-GRU, blue = SS2D, green = enhanced, gray = 공유).
출력: paper/figs/conf_fig{1,2,3}_*.{png,pdf}
"""
import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper/figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

INK, INK2, MUTED = "#0b0b0b", "#3f3e3b", "#7d7b75"
BLUE, RED, GREEN = "#2a78d6", "#e34948", "#2e9e6b"
FILL_N, EDGE_N = "#f1f0ed", "#b9b8ae"
FILL_B, FILL_R, FILL_G = "#e3eefb", "#fbe7e7", "#e2f3ea"
W = 3.15  # column width (in)
MIN_FS = 4.6

_FIG = None


def _fit(t, max_w):
    """shrink a Text until its rendered width (in) <= max_w."""
    r = _FIG.canvas.get_renderer()
    for _ in range(40):
        w = t.get_window_extent(renderer=r).width / _FIG.dpi
        if w <= max_w or t.get_fontsize() <= MIN_FS:
            if w > max_w:
                print(f"  ! overflow ({w:.2f} > {max_w:.2f} in): {t.get_text()[:50]!r}", file=sys.stderr)
            return
        t.set_fontsize(t.get_fontsize() - 0.2)


def text(ax, x, y, s, fs=6.0, color=INK2, ha="center", va="center", max_w=None, z=3, **kw):
    t = ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, **kw)
    if max_w is not None:
        _fit(t, max_w)
    return t


def box(ax, x, y, w, h, title=None, lines=(), fc=FILL_N, ec=EDGE_N, lw=0.7,
        fs=5.6, tfs=6.2, tc=INK, ls="-", pad=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    rows = ([] if title is None else [(title, tfs, tc, "bold")]) + \
           [(t, fs, INK2, "normal") for t in lines]
    n = len(rows)
    if n == 0:
        return
    lh = h / n
    for i, (s, size, c, wgt) in enumerate(rows):
        yy = y + h - lh * (i + 0.5)
        text(ax, x + w / 2, yy, s, fs=size, color=c, fontweight=wgt, max_w=w - 0.05, z=z + 1)


def arrow(ax, p1, p2, color=INK2, lw=0.8, ms=6, z=1, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, zorder=z, ls=ls, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0.3, shrinkB=0.3))


def line(ax, p1, p2, color=INK2, lw=0.8, z=1, ls="-"):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, zorder=z, ls=ls,
            solid_capstyle="round")


def elbow(ax, pts, color=INK2, lw=0.8, z=1, ls="-"):
    """polyline through pts, arrowhead on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        line(ax, a, b, color=color, lw=lw, z=z, ls=ls)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z, ls=ls)


def circ(ax, x, y, sym, r=0.075, fs=7):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK2, lw=0.8, zorder=2))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, color=INK, zorder=3,
            fontweight="bold" if sym.isalpha() else "normal")


def header(ax, y, text_, color, sub=None):
    ax.add_patch(Rectangle((0.04, y - 0.02), 0.05, 0.16, fc=color, ec="none", zorder=2))
    text(ax, 0.13, y + 0.06, text_, fs=6.6, color=INK, ha="left", fontweight="bold", max_w=2.95)
    if sub:
        text(ax, 0.13, y - 0.085, sub, fs=5.4, color=MUTED, ha="left", max_w=2.95)


def canvas(h):
    global _FIG
    fig = plt.figure(figsize=(W, h), facecolor="white", dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, h)
    ax.axis("off")
    _FIG = fig
    return fig, ax


def save(fig, name):
    png = os.path.join(OUT, name + ".png"); pdf = os.path.join(OUT, name + ".pdf")
    fig.savefig(png, dpi=600, facecolor="white"); fig.savefig(pdf, facecolor="white")
    print("saved:", os.path.relpath(png, ROOT))


# ══════════════════════════════════════════════════════════════════════════
# 그림 1 — 통제 비교 파이프라인 (f_θ 만 변수)
# ══════════════════════════════════════════════════════════════════════════
H1 = 3.90
fig, ax = canvas(H1)
T = lambda v: H1 - v          # top-down coordinate helper
cx = 1.22                      # main-flow center x
RM = 3.02                      # right edge of boxes (GT rail runs at 3.09)

# fully-sampled k-space + ground truth
box(ax, cx - 0.80, T(0.42), 1.60, 0.36, title=r"Fully-sampled multicoil k-space  $y_c$",
    lines=["16 coils, 384² (crop/pad)"])
box(ax, 2.14, T(0.42), RM - 2.14, 0.36, title="Ground truth", lines=[r"RSS($F^{-1}y_c$), 384²"])
arrow(ax, (cx + 0.80, T(0.24)), (2.14, T(0.24)))

# mask ⊙
my = T(0.70)
circ(ax, cx, my, "⊙")
arrow(ax, (cx, T(0.42)), (cx, my + 0.08))
box(ax, 0.06, my - 0.18, 0.96, 0.36, title="Undersampling mask  $M$",
    lines=["R = 4 equispaced, 8 % ACS"])
arrow(ax, (1.02, my), (cx - 0.08, my))

# undersampled k-space
box(ax, cx - 0.80, T(1.30), 1.60, 0.36, title=r"Undersampled k-space  $\tilde{y}=M\odot y_c$",
    lines=["32 ch = 16 coils × (Re, Im), 384²"])
arrow(ax, (cx, my - 0.08), (cx, T(0.94)))

# split: left → f_θ, right → zero-filled
fx, fy, fw, fh = 0.06, T(2.36), 1.80, 0.90        # f_θ frame
zx, zy, zw, zh = 2.02, T(2.02), RM - 2.02, 0.46   # zero-filled box
line(ax, (cx, T(1.30)), (cx, T(1.38)))
line(ax, (fx + fw / 2, T(1.38)), (zx + zw / 2, T(1.38)))
arrow(ax, (fx + fw / 2, T(1.38)), (fx + fw / 2, fy + fh + 0.02))
arrow(ax, (zx + zw / 2, T(1.38)), (zx + zw / 2, zy + zh + 0.02))
text(ax, zx + zw / 2 + 0.04, T(1.46), r"$F^{-1}$ per coil", fs=5.4, color=MUTED, ha="left")

# f_θ frame (dashed = the only variable)
ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0.02", fc="white",
                            ec=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=2))
text(ax, fx + fw / 2, fy + fh - 0.08, r"Sequence model  $f_\theta$", fs=6.5, color=INK,
     fontweight="bold", max_w=fw - 0.1)
text(ax, fx + fw / 2, fy + fh - 0.19, "the only variable between the arms", fs=5.4, color=INK2,
     style="italic", max_w=fw - 0.1)
bw_ = 0.74
box(ax, fx + 0.07, fy + 0.20, bw_, 0.36, title="bi-GRU", lines=["ETER-Net original", "arm 668.2M"],
    fc=FILL_R, ec=RED, fs=5.4, tfs=6.2)
box(ax, fx + fw - 0.07 - bw_, fy + 0.20, bw_, 0.36, title="SS2D", lines=["selective state space", "arm 31.2M"],
    fc=FILL_B, ec=BLUE, fs=5.4, tfs=6.2)
text(ax, fx + fw / 2, fy + 0.38, "or", fs=6, color=MUTED, style="italic")
text(ax, fx + fw / 2, fy + 0.09, "→ image-domain features, 20 ch, 384²  (Fig. 2)", fs=5.2,
     color=INK2, max_w=fw - 0.08)

# zero-filled
box(ax, zx, zy, zw, zh, title="Zero-filled coil images", lines=[r"$F^{-1}\tilde{y}$,  32 ch, 384²"])

# concat
ccx, ccy = 1.94, T(2.66)
circ(ax, ccx, ccy, "C", r=0.085)
line(ax, (fx + fw / 2, fy - 0.02), (fx + fw / 2, ccy))
arrow(ax, (fx + fw / 2, ccy), (ccx - 0.09, ccy))
line(ax, (zx + zw / 2, zy - 0.02), (zx + zw / 2, ccy))
arrow(ax, (zx + zw / 2, ccy), (ccx + 0.09, ccy))
text(ax, ccx + 0.12, ccy - 0.13, "channel concat → 52 ch", fs=5.2, color=MUTED, ha="left")

# U-Net
ux, uy, uw, uh = 0.24, T(3.24), 2.62, 0.40
box(ax, ux, uy, uw, uh, title=r"De-aliasing U-Net  $g_\phi$  (dual-frame skip)",
    lines=["depth 5, 64 → 1024 ch, 31.1M  ·  identical in both arms"])
arrow(ax, (ccx, ccy - 0.09), (ccx, uy + uh + 0.02))

# output + loss
ox, oy, ow, oh = 0.24, T(3.70), 1.20, 0.34
lx, ly, lw_, lh_ = 1.58, T(3.70), RM - 1.58, 0.34
box(ax, ox, oy, ow, oh, title=r"Reconstruction  $\hat{x}$", lines=["magnitude, 1 × 384²"])
box(ax, lx, ly, lw_, lh_, title="Training loss  /  metrics",
    lines=["L1 + (1 − SSIM) in brain mask", "SSIM · PSNR · nMSE  vs. GT"], fs=5.3, tfs=6.0)
arrow(ax, (ox + 0.60, uy - 0.02), (ox + 0.60, oy + oh + 0.02))
arrow(ax, (ox + ow + 0.02, oy + oh / 2), (lx - 0.02, oy + oh / 2))
# GT rail down the right margin
elbow(ax, [(RM + 0.02, T(0.24)), (3.09, T(0.24)), (3.09, ly + lh_ / 2), (RM + 0.02, ly + lh_ / 2)],
      color=MUTED, ls=(0, (2, 1.5)))
text(ax, 3.09, T(2.5), "GT", fs=5.0, color=MUTED, rotation=90,
     bbox=dict(fc="white", ec="none", pad=0.6))

text(ax, W / 2, T(3.83), "gray = shared by both arms   ·   dashed = the only difference between the arms",
     fs=5.2, color=MUTED, max_w=3.05)
save(fig, "conf_fig1_pipeline")

# ══════════════════════════════════════════════════════════════════════════
# 그림 2 — 두 팔의 내부 구조
# ══════════════════════════════════════════════════════════════════════════
H2 = 4.05
fig, ax = canvas(H2)
T = lambda v: H2 - v
G = 0.12  # horizontal gap between boxes


def row(ax, y, specs, h=0.34, x0=0.06, gap=G):
    """place boxes left→right with arrows between; specs = [(w, kwargs), ...]; returns centers."""
    xs, x = [], x0
    for i, (w, kw) in enumerate(specs):
        box(ax, x, y, w, h, **kw)
        xs.append((x, x + w))
        if i:
            arrow(ax, (xs[i - 1][1] + 0.02, y + h / 2), (x - 0.02, y + h / 2))
        x += w + gap
    return xs


# ---------- (a) bi-GRU ----------
header(ax, T(0.20), "(a)  bi-GRU  —  original ETER-Net arm", RED,
       sub="arm 668.2M parameters, of which GRU stack 637.1M")
r1 = T(0.76)
xa = row(ax, r1, [(0.62, dict(title="k-space", lines=["32 × 384 × 384"])),
                  (1.06, dict(title="flatten → sequence", lines=["384 steps × 12,288-dim"])),
                  (1.10, dict(title="bi-GRU ①   ⇄", lines=["hidden 3,840 × 2 dir."], fc=FILL_R, ec=RED))])
r2 = T(1.26)
xb = row(ax, r2, [(1.00, dict(title="transpose", lines=["384 steps × 7,680-dim"])),
                  (1.10, dict(title="bi-GRU ②   ⇅", lines=["hidden 3,840 × 2 dir."], fc=FILL_R, ec=RED)),
                  (0.66, dict(title="reshape", lines=["20 × 384 × 384"]))])
cx1 = (xa[2][0] + xa[2][1]) / 2; cx2 = (xb[0][0] + xb[0][1]) / 2
elbow(ax, [(cx1, r1 - 0.02), (cx1, r1 - 0.08), (cx2, r1 - 0.08), (cx2, r2 + 0.34 + 0.02)])
text(ax, 0.06, T(1.34),
     "Each step consumes a 12,288-dim (7,680-dim) slice of the flattened\n"
     "k-space tensor, so the input-to-hidden matrices alone are 12,288 × 11,520\n"
     "and 7,680 × 11,520 per direction (→ 637M), and the 384 steps run strictly\n"
     "sequentially.",
     fs=5.2, color=MUTED, ha="left", va="top", max_w=3.03, linespacing=1.15)

# ---------- (b) SS2D ----------
header(ax, T(1.92), "(b)  SS2D  —  controlled substitution", BLUE,
       sub="arm 31.2M parameters, of which SSM stack 0.117M (same 20-ch output as bi-GRU)")
s1 = T(2.50)
xs_ = row(ax, s1, [(0.62, dict(title="k-space", lines=["32 × 384 × 384"])),
                   (0.98, dict(title="Linear 32 → 128", lines=["LN · SiLU, per pixel"])),
                   (1.18, dict(title="Depthwise conv 3×3", lines=["local context, 128 ch"]))])

# four-scan pictogram: 2×2 tiny maps, one scan direction each (VMamba-style cross scan)
gx, gy, gs, gg = 0.16, T(3.34), 0.27, 0.06
ts = gs
tiles = {"→": (gx, gy + ts + gg), "←": (gx + ts + gg, gy + ts + gg), "↓": (gx, gy), "↑": (gx + ts + gg, gy)}
for sym, (tx, ty) in tiles.items():
    ax.add_patch(Rectangle((tx, ty), ts, ts, fc="white", ec=EDGE_N, lw=0.5, zorder=2))
    for k in (1, 2):
        line(ax, (tx + ts * k / 3, ty), (tx + ts * k / 3, ty + ts), color="#e2e1db", lw=0.35, z=2)
        line(ax, (tx, ty + ts * k / 3), (tx + ts, ty + ts * k / 3), color="#e2e1db", lw=0.35, z=2)
    for k in range(3):
        c = ts * (k + 0.5) / 3
        if sym == "→":
            arrow(ax, (tx + 0.02, ty + c), (tx + ts - 0.02, ty + c), color=BLUE, lw=0.7, ms=4, z=3)
        elif sym == "←":
            arrow(ax, (tx + ts - 0.02, ty + c), (tx + 0.02, ty + c), color=BLUE, lw=0.7, ms=4, z=3)
        elif sym == "↓":
            arrow(ax, (tx + c, ty + ts - 0.02), (tx + c, ty + 0.02), color=BLUE, lw=0.7, ms=4, z=3)
        else:
            arrow(ax, (tx + c, ty + 0.02), (tx + c, ty + ts - 0.02), color=BLUE, lw=0.7, ms=4, z=3)
GS = 2 * ts + gg                                   # pictogram side
text(ax, gx + GS / 2, gy - 0.05, "4 scans of the map:\nrows →, ←  ·  cols ↓, ↑", fs=5.0, color=INK2,
     va="top", linespacing=1.15, max_w=0.90)
cxd = (xs_[2][0] + xs_[2][1]) / 2
elbow(ax, [(cxd, s1 - 0.02), (cxd, s1 - 0.08), (gx + GS / 2, s1 - 0.08), (gx + GS / 2, gy + GS + 0.02)])

bx, by, bw, bh = 0.98, T(3.46), 2.10, 0.84
box(ax, bx, by, bw, bh, title="Selective scan (S6), one per direction",
    lines=[r"$h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t, \quad y_t = C_t\, h_t + D\, x_t$",
           r"$(\Delta_t, B_t, C_t) = \mathrm{Linear}(x_t)$ : input-dependent",
           "d_inner 128, N = 16 · weights shared by all lines",
           "parallel scan, linear in the length L = 384",
           "merge: concat 4 × 128 → LN · Linear → 128",
           "1×1 conv → 20 ch (= bi-GRU output)"],
    fc=FILL_B, ec=BLUE, fs=5.3, tfs=6.0)
arrow(ax, (gx + GS + 0.02, gy + GS / 2), (bx - 0.02, gy + GS / 2))
fbx = 2.46
box(ax, fbx, T(3.90), 3.08 - fbx, 0.32, title="features", lines=["20 × 384 × 384"])
arrow(ax, (fbx + 0.31, by - 0.02), (fbx + 0.31, T(3.90) + 0.32 + 0.02))
text(ax, 0.06, T(3.66),
     "Cost does not grow with the line length: every pixel\n"
     "is projected to 128 dims and the recurrence state is\n"
     "128 × 16, hence 0.117M for the whole stack.",
     fs=5.2, color=MUTED, ha="left", va="top", max_w=2.32, linespacing=1.15)
save(fig, "conf_fig2_arms")

# ══════════════════════════════════════════════════════════════════════════
# 그림 3 — 강화 SS2D
# ══════════════════════════════════════════════════════════════════════════
H3 = 3.22
fig, ax = canvas(H3)
T = lambda v: H3 - v
header(ax, T(0.20), r"Enhanced SS2D  —  drop-in replacement for  $f_\theta$", GREEN,
       sub="arm 34.2M parameters, of which SSM stack 3.1M  ·  2.84 h/epoch, 80 epochs")
r1 = T(0.76)
xr = row(ax, r1, [(0.50, dict(title="k-space", lines=["32 ch, 384²"])),
                  (1.14, dict(title="Stem", lines=["LN · Linear 32 → 256 · SiLU"])),
                  (1.14, dict(title="Downsample ×3", lines=["3×3 conv, stride 3 → 128²"]))], h=0.34)
r2 = T(1.34)
for off in (0.05, 0.025):
    ax.add_patch(FancyBboxPatch((0.06 + off, r2 + off), 1.16, 0.42, boxstyle="round,pad=0.02",
                                fc="white", ec=GREEN, lw=0.6, zorder=1))
xq = row(ax, r2, [(1.16, dict(title="Gated SS2D block  ×3", lines=["residual, 256 ch", "128² grid, fp16 scan"],
                              fc=FILL_G, ec=GREEN)),
                  (1.04, dict(title="LN · Upsample", lines=["bilinear 128² → 384²", "3×3 conv · SiLU"])),
                  (0.60, dict(title="Head", lines=["1×1 conv", "→ 64 ch"]))], h=0.42, gap=0.115)
c1 = (xr[2][0] + xr[2][1]) / 2; c2 = (xq[0][0] + xq[0][1]) / 2
elbow(ax, [(c1, r1 - 0.02), (c1, r1 - 0.08), (c2, r1 - 0.08), (c2, r2 + 0.42 + 0.07)])
text(ax, 3.08, r2 - 0.07, "→ concat with zero-filled images & U-Net (Fig. 1)", fs=5.2, color=MUTED,
     ha="right", max_w=2.4)

# block detail frame
dx, dy, dw, dh = 0.06, T(2.80), 3.02, 1.27
ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle="round,pad=0.02", fc="white",
                            ec=GREEN, lw=0.8, ls=(0, (3, 1.5)), zorder=1))
text(ax, dx + 0.08, dy + dh - 0.09, "Inside one block  (256 → 256 ch)", fs=6.2, color=INK, ha="left",
     fontweight="bold")
b1 = dy + dh - 0.50
xk = row(ax, b1, [(0.98, dict(title="LN · Linear 256 → 512", lines=["split → x_ssm | z"], fs=5.2, tfs=5.8)),
                  (0.86, dict(title="DWConv 3×3 · SiLU", lines=["on x_ssm"], fs=5.2, tfs=5.8)),
                  (0.86, dict(title="4-dir scan (S6)", lines=["d_inner 256, N = 32"], fc=FILL_B, ec=BLUE,
                              fs=5.2, tfs=5.8))], h=0.30, x0=dx + 0.08, gap=0.08)
yB = b1 - 0.22                    # gate row
yC = yB - 0.30                    # output row
mx = (xk[2][0] + xk[2][1]) / 2
circ(ax, mx, yB, "⊗", r=0.07)
arrow(ax, (mx, b1 - 0.02), (mx, yB + 0.075))                                     # scan → ⊗
zx0 = xk[0][0] + 0.84
elbow(ax, [(zx0, b1 - 0.02), (zx0, yB), (mx - 0.075, yB)])                      # z → ⊗ (from the left)
text(ax, (zx0 + mx) / 2, yB + 0.035, "gate:  z → SiLU(z)", fs=5.2, color=INK2, va="bottom")
lbx, lbw = dx + 1.16, 0.66
box(ax, lbx, yC - 0.15, lbw, 0.30, title="Linear 256 → 256", lines=["dropout 0.05"], fs=5.2, tfs=5.8)
elbow(ax, [(mx, yB - 0.075), (mx, yC), (lbx + lbw + 0.02, yC)])                  # ⊗ → Linear
px = dx + 0.66
circ(ax, px, yC, "⊕", r=0.07)
arrow(ax, (lbx - 0.02, yC), (px + 0.075, yC))
arrow(ax, (px - 0.075, yC), (dx + 0.12, yC))
text(ax, dx + 0.12, yC + 0.10, "out", fs=5.2, color=INK2, ha="left")
rx = xk[0][0] + 0.14                                                             # residual tap (block input)
elbow(ax, [(rx, b1 - 0.02), (rx, yC + 0.19), (px, yC + 0.19), (px, yC + 0.075)], color=MUTED,
      ls=(0, (2, 1.5)))
text(ax, rx + 0.04, yC + 0.21, "residual", fs=5.0, color=MUTED, ha="left", va="bottom")

text(ax, W / 2, T(3.04),
     "vs. controlled SS2D (Fig. 2b): gating y ⊙ SiLU(z) restored · 3 residual blocks\n"
     "· bottleneck lifted (20 → 64 ch, d_inner 128 → 256, d_state 16 → 32)\n"
     "· coarse 128² scan keeps the epoch time (2.84 h vs. 3.07 h)",
     fs=5.2, color=MUTED, max_w=3.05, linespacing=1.15)
save(fig, "conf_fig3_enhanced")
