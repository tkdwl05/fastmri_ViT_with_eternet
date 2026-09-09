#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEIE 학술대회판 전용 세부 아키텍처 그림 3장 — 고전적 블록 다이어그램 양식(v2, 2026-09-08).

  paper/figs/conf_fig1_pipeline.{png,pdf}   그림 1  두 팔이 공유하는 통제 파이프라인(데이터 노드는 실제 썸네일)
  paper/figs/conf_fig2_arms.{png,pdf}       그림 2  (a) bi-GRU 팔 = 펼친(unrolled) 양방향 순환 체인, (b) SS2D 팔 = 4방향 cross-scan → S6 → merge
  paper/figs/conf_fig3_enhanced.{png,pdf}   그림 3  강화 SS2D: 위 = 블록 체인(화살표 위 텐서 크기), 아래 = Mamba 블록 내부(게이트·잔차)

양식 규칙(v1 의 "슬라이드" 인상을 고친 지점):
  - 블록 = 균일한 크기의 둥근 사각형에 한 줄 라벨만. 텐서 크기·채널 수는 블록 안이 아니라 화살표 위 작은 회색 글자.
  - 설명 문장은 그림 안에 넣지 않는다(캡션·본문 몫). 굵은 제목·색 막대·각주 없음.
  - 데이터 노드(k-space·마스크·zero-filled·복원·GT)는 정본 슬라이스의 실제 영상 썸네일.
  - RNN 은 펼친 셀 체인(→/← 두 줄), SS2D 는 cross-scan(4 격자) → S6 ×4 → merge, Mamba 블록은 원 논문의 게이트 분기 형태로 그린다.
  - 글꼴 Liberation Sans(Arial metric 호환, paper/fonts/) 6.5 pt 본문 / 5.5 pt 주석; 없으면 DejaVu Sans 폴백.
  - 단 폭 3.15 in, 600 dpi PNG + PDF. 색은 팔 식별용 3색(bi-GRU 빨강·SS2D 파랑·강화 초록)만 옅게.

데이터(로컬 전용): 정본 슬라이스 index 4689 = fastMRI_data/multicoil_val/file_brain_AXT2_203_2030309.h5 slice 7
(visualize_slices_canonical.json) 의 k-space 를 dataloader_h5_v5 와 같은 전처리(코일 영상 384² crop/pad → re-FFT →
R=4 equispaced 마스크, offset 3)로 만들고, zero-filled·SS2D 복원·GT 는 results/vis/multimodel_compare/recon_4689.npz 에서 읽는다.
둘 중 하나라도 없으면 합성 자리표시 썸네일로 대체하고 경고를 출력한다.

실행: CUDA_VISIBLE_DEVICES="" python paper/make_figs_conf_arch.py
"""
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ───────────────────────── fonts ─────────────────────────
_FONT_DIR = os.path.join(HERE, "fonts")
_found = False
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
           "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    _p = os.path.join(_FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        _found = True
if _found:
    rcParams["font.family"] = "Liberation Sans"
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "Liberation Sans"
    rcParams["mathtext.it"] = "Liberation Sans:italic"
    rcParams["mathtext.bf"] = "Liberation Sans:bold"
    rcParams["mathtext.fallback"] = "stixsans"
else:
    warnings.warn("paper/fonts/LiberationSans-*.ttf 없음 → DejaVu Sans 폴백")
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["mathtext.fontset"] = "dejavusans"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# ───────────────────────── style ─────────────────────────
W = 3.15                      # column width (in)
DPI = 600
INK = "#000000"
INK2 = "#444444"
MUTED = "#6b6b6b"
FILL = "#f2f2f2"              # shared / identical blocks
EDGE = "#3a3a3a"
FILL_R, EDGE_R = "#fbe9e8", "#c0392b"   # bi-GRU
FILL_B, EDGE_B = "#e4eefb", "#2a6fc9"   # SS2D
FILL_G, EDGE_G = "#e3f3ea", "#2a8f5f"   # enhanced SS2D
FS = 6.5                      # block label
FS_S = 5.5                    # annotations (tensor sizes, sub-labels)
FS_P = 7.5                    # panel letters
LW_BOX = 0.6
LW_ARR = 0.7

_OVERFLOW = []


def canvas(h, y_lo=0.0, w=None):
    """폭 w(기본 W=단 폭) 캔버스(단위 inch). y_lo > 0 이면 아래쪽 y_lo 만큼 잘라낸다(여백 정리).
    학술지판 page 폭 그림(make_fig1_architecture.py)은 w 를 넘겨 같은 도우미를 재사용한다."""
    w = W if w is None else w
    fig = plt.figure(figsize=(w, h - y_lo))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(y_lo, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, fs=FS, color=INK, ha="center", va="center", max_w=None, z=5, **kw):
    """텍스트 배치. max_w(in) 를 넘으면 4.6 pt 까지 자동 축소(렌더러 실측)."""
    t = ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, **kw)
    if max_w is not None:
        fig = ax.figure
        r = fig.canvas.get_renderer()
        while True:
            w_in = t.get_window_extent(renderer=r).width / fig.dpi
            if w_in <= max_w or t.get_fontsize() <= 4.6:
                break
            t.set_fontsize(t.get_fontsize() - 0.25)
        if w_in > max_w + 1e-3:
            _OVERFLOW.append((s, w_in, max_w))
    return t


def box(ax, x, y, w, h, label, fc=FILL, ec=EDGE, lw=LW_BOX, ls="-", fs=FS, tc=INK,
        sub=None, sub_fs=FS_S, sub_color=INK2, weight="normal", z=3, r=0.035):
    """균일 둥근 사각형 + 가운데 한 줄 라벨(+ 선택적 작은 부라벨)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if sub is None:
        text(ax, x + w / 2, y + h / 2, label, fs=fs, color=tc, max_w=w - 0.04, weight=weight, z=z + 1)
    else:
        text(ax, x + w / 2, y + h * 0.66, label, fs=fs, color=tc, max_w=w - 0.04, weight=weight, z=z + 1)
        text(ax, x + w / 2, y + h * 0.30, sub, fs=sub_fs, color=sub_color, max_w=w - 0.04, z=z + 1)
    return (x, y, w, h)


def thumb(ax, x, y, s, img, label=None, label_fs=FS_S, label_dy=0.045, vmin=None, vmax=None,
          cmap="gray", frame=EDGE, z=3):
    """실제 영상 썸네일(정사각 s in) + 아래 라벨. 작은 배열(마스크 축소판)은 nearest 로 계단 유지."""
    interp = "nearest" if max(img.shape) <= 64 else "antialiased"
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + s, y, y + s),
              interpolation=interp, zorder=z, aspect="auto")
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=frame, lw=0.5, zorder=z + 1))
    if label:
        text(ax, x + s / 2, y - label_dy, label, fs=label_fs, color=INK, va="top", z=z + 1)
    return (x, y, s, s)


def arrow(ax, p1, p2, color=INK, lw=LW_ARR, ls="-", head=True, z=4, mutation=6):
    style = "-|>" if head else "-"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mutation, lw=lw,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0, zorder=z,
                                 capstyle="butt", joinstyle="miter"))


def polyline(ax, pts, color=INK, lw=LW_ARR, ls="-", head=True, z=4):
    """꺾은선 경로, 마지막 구간에만 화살촉."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z,
                           solid_capstyle="projecting"))
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, head=head, z=z)


def dot(ax, x, y, r=0.017, color=INK, z=5):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=z))


def op(ax, x, y, sym, r=0.055, fs=6.5, fc="white", ec=INK, z=5):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=LW_BOX, zorder=z))
    text(ax, x, y - 0.002, sym, fs=fs, color=INK, z=z + 1)


def lab(ax, x, y, s, fs=FS_S, color=MUTED, ha="center", va="bottom", max_w=None, z=5, **kw):
    """화살표 위 텐서 크기 등 작은 회색 주석."""
    return text(ax, x, y, s, fs=fs, color=color, ha=ha, va=va, max_w=max_w, z=z, **kw)


def save(fig, name):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"saved {name}.png/.pdf")


# ───────────────────────── data thumbnails ─────────────────────────
SLICE_FILE = os.path.join(ROOT, "fastMRI_data", "multicoil_val", "file_brain_AXT2_203_2030309.h5")
SLICE_NO = 7
NPZ = os.path.join(ROOT, "results", "vis", "multimodel_compare", "recon_4689.npz")


def load_thumbs():
    """정본 슬라이스 4689 의 실제 영상. dict(ksp_full, mask2d, ksp_und, zf, recon, gt)."""
    out = {}
    try:
        sys.path.insert(0, ROOT)
        import h5py
        from dataloaders.dataloader_h5_v5 import build_r4_mask, crop_or_pad_to, fft2c, ifft2c
        with h5py.File(SLICE_FILE, "r") as f:
            ksp = np.asarray(f["kspace"][SLICE_NO])                    # (coil, H, W) complex
        img = crop_or_pad_to(ifft2c(ksp), (384, 384))
        ksp = fft2c(img)
        m1 = build_r4_mask(384)                                          # 결정적 offset(=3), ACS 8 %
        ms = build_r4_mask(40)                                           # 같은 규칙의 40열 축소판(썸네일 가독용)
        m2 = np.repeat(ms[None, :], 40, axis=0)
        rss = lambda k: np.sqrt(np.sum(np.abs(k) ** 2, axis=0))
        kf = np.log1p(rss(ksp) / rss(ksp).max() * 2e3)
        ku = np.log1p(rss(ksp * m1[None, None, :]) / rss(ksp).max() * 2e3)
        out.update(ksp_full=(kf / kf.max()) ** 0.8, ksp_und=(ku / kf.max()) ** 0.8, mask2d=m2)
    except Exception as e:                                             # pragma: no cover
        warnings.warn(f"k-space 썸네일 생성 실패({e!r}) → 합성 자리표시")
        yy, xx = np.mgrid[0:384, 0:384]
        rr = np.hypot(yy - 192, xx - 192) + 1
        kf = np.clip(1 - np.log(rr) / np.log(280), 0, 1)
        m1 = np.zeros(384, np.float32); m1[3::4] = 1; m1[177:207] = 1
        ms = np.zeros(40, np.float32); ms[3::4] = 1; ms[18:21] = 1
        out.update(ksp_full=kf, ksp_und=kf * m1[None, :], mask2d=np.repeat(ms[None, :], 40, 0))
    try:
        d = np.load(NPZ)
        out.update(zf=d["rec_zf"], recon=d["rec_ss2d"], gt=d["gt"])
    except Exception as e:                                             # pragma: no cover
        warnings.warn(f"복원 썸네일 로드 실패({e!r}) → 합성 자리표시")
        yy, xx = np.mgrid[0:384, 0:384]
        disk = (np.hypot(yy - 192, xx - 192) < 140).astype(np.float32)
        out.update(zf=disk * 0.7, recon=disk, gt=disk)
    return out


# ═════════════════════════ Fig. 1 — shared pipeline ═════════════════════════
def fig1(th):
    H, Y_LO = 2.30, 0.16                      # 아래 0.16 in 은 잘라냄 → 그림 높이 2.14 in
    fig, ax = canvas(H, y_lo=Y_LO)
    T = 0.38                      # thumbnail size (main row)
    vmax = float(th["gt"].max())
    y1 = 1.08                     # bottom of main-row thumbs
    yc = y1 + T / 2               # main-row centre line

    # ── data preparation: y_c ⊙ M → ỹ (vertical chain, top-left) ──
    thumb(ax, 0.07, 1.90, 0.32, th["ksp_full"])
    text(ax, 0.44, 2.11, "fully-sampled k-space $y_c$", fs=6.0, ha="left")
    lab(ax, 0.44, 1.98, "16 coils, 384$^2$ (crop/pad), Re/Im", ha="left", va="center", fs=5.0)
    arrow(ax, (0.23, 1.90), (0.23, 1.775))
    op(ax, 0.23, 1.72, r"$\odot$")
    thumb(ax, 0.46, 1.595, 0.25, th["mask2d"], vmin=0, vmax=1)
    arrow(ax, (0.46, 1.72), (0.285, 1.72))
    text(ax, 0.75, 1.77, "mask $M$", fs=6.0, ha="left")
    lab(ax, 0.75, 1.66, "R = 4 equispaced, 8 % ACS", ha="left", va="center", fs=5.0)
    arrow(ax, (0.23, 1.665), (0.23, y1 + T))

    # ── main row ──
    thumb(ax, 0.04, y1, T, th["ksp_und"])
    text(ax, 0.23, y1 - 0.045, r"$\tilde{y} = M \odot y_c$", fs=6.0, va="top")
    lab(ax, 0.23, y1 - 0.15, "undersampled", va="top", fs=4.9)
    dot(ax, 0.46, yc)
    arrow(ax, (0.42, yc), (0.46, yc), head=False)
    # sequence-model slot (the only variable)
    sx, sw, sh = 0.56, 0.72, 0.50
    sy = yc - sh / 2
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.035",
                                fc="white", ec=INK, lw=0.7, ls=(0, (2.2, 1.4)), zorder=3))
    text(ax, sx + sw / 2, sy + sh - 0.08, "sequence model $f_\\theta$", fs=FS, max_w=sw - 0.04, z=4)
    text(ax, sx + sw / 2, sy + sh - 0.165, "(the only variable)", fs=4.9, color=MUTED, max_w=sw - 0.04, z=4)
    bw, bh = 0.28, 0.15
    box(ax, sx + 0.03, sy + 0.05, bw, bh, "bi-GRU", fc=FILL_R, ec=EDGE_R, fs=5.4)
    text(ax, sx + sw / 2, sy + 0.05 + bh / 2, "or", fs=5.0, color=MUTED, z=4)
    box(ax, sx + sw - 0.03 - bw, sy + 0.05, bw, bh, "SS2D", fc=FILL_B, ec=EDGE_B, fs=5.4)
    arrow(ax, (0.46, yc), (sx, yc))
    lab(ax, 0.505, yc + 0.03, "32", fs=4.8)
    # concat
    cx = 1.42
    arrow(ax, (sx + sw, yc), (cx - 0.055, yc))
    lab(ax, (sx + sw + cx - 0.055) / 2, yc + 0.03, "20", fs=5.0)
    op(ax, cx, yc, "C", fs=6.0)
    # U-Net
    ux, uw = 1.55, 0.48
    box(ax, ux, sy + 0.05, uw, sh - 0.10, "U-Net $g_\\phi$", sub="31.1M, shared")
    arrow(ax, (cx + 0.055, yc), (ux, yc))
    lab(ax, (cx + 0.055 + ux) / 2, yc + 0.03, "52", fs=5.0)
    # reconstruction / ground truth
    rx, gx = 2.12, 2.68
    arrow(ax, (ux + uw, yc), (rx, yc))
    thumb(ax, rx, y1, T, th["recon"], vmin=0, vmax=vmax)
    text(ax, rx + T / 2, y1 - 0.045, r"reconstruction $\hat{x}$", fs=5.6, va="top")
    lab(ax, rx + T / 2, y1 - 0.15, "magnitude", va="top", fs=4.9)
    thumb(ax, gx, y1, T, th["gt"], vmin=0, vmax=vmax)
    text(ax, gx + T / 2, y1 - 0.045, "ground truth $x$", fs=5.6, va="top")
    lab(ax, gx + T / 2, y1 - 0.15, "RSS($F^{-1} y_c$)", va="top", fs=4.9)
    # loss bracket above the two images
    yl = y1 + T + 0.10
    for xx in (rx + T / 2, gx + T / 2):
        ax.add_line(Line2D([xx, xx], [y1 + T, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    ax.add_line(Line2D([rx + T / 2, gx + T / 2], [yl, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    text(ax, (rx + gx + T) / 2, yl + 0.03, "loss: $L_1 + (1-\\mathrm{SSIM})$, brain mask", fs=5.0, va="bottom",
         max_w=0.96)

    # ── zero-filled branch (row below) ──
    zs, zx, zy = 0.36, 0.70, 0.54
    zc = zy + zs / 2
    polyline(ax, [(0.46, yc), (0.46, zc), (zx, zc)])
    lab(ax, 0.59, zc + 0.03, "$F^{-1}$", fs=6.0, color=INK)
    thumb(ax, zx, zy, zs, th["zf"], vmin=0, vmax=vmax)
    text(ax, zx + zs / 2, zy - 0.045, "zero-filled coil images", fs=5.8, va="top")
    polyline(ax, [(zx + zs, zc), (cx, zc), (cx, yc - 0.055)])
    lab(ax, (zx + zs + cx) / 2, zc + 0.03, "32", fs=5.0)
    lab(ax, W - 0.05, Y_LO + 0.05, "numbers on arrows: channels (384$^2$)", fs=4.9, ha="right", va="bottom")
    save(fig, "conf_fig1_pipeline")


# ═════════════════════════ Fig. 2 — the two arms ═════════════════════════
def unrolled_birnn(ax, x0, y0, n=4, cw=0.19, ch=0.13, gap=0.10, fc=FILL_R, ec=EDGE_R, ell_after=2):
    """펼친 양방향 순환 체인: 입력 x_t(아래) → 역방향 셀 · 순방향 셀 → 출력 y_t(위).
    한 수직선이 두 셀을 모두 관통(입력은 두 방향에 공급, 출력은 두 방향을 결합)."""
    xs = [x0 + i * (cw + gap) for i in range(n)]
    yb, yf = y0 + 0.16, y0 + 0.34                 # backward / forward rows (cell bottoms)
    idx = [str(i + 1) for i in range(n - 1)] + ["L"]
    for i, x in enumerate(xs):
        xc = x + cw / 2
        if i == ell_after:
            text(ax, xc, yf + ch / 2, "…", fs=7, color=INK2)
            text(ax, xc, yb + ch / 2, "…", fs=7, color=INK2)
            continue
        ax.add_line(Line2D([xc, xc], [y0 + 0.10, yf + ch + 0.04], color=INK, lw=LW_ARR, zorder=2))
        for yy in (yb, yf):
            arrow(ax, (xc, yy - 0.03), (xc, yy), mutation=4, z=2)
        arrow(ax, (xc, yf + ch), (xc, yf + ch + 0.05), mutation=4, z=2)
        box(ax, x, yf, cw, ch, r"$\vec{h}$", fc=fc, ec=ec, fs=6.0, r=0.02)
        box(ax, x, yb, cw, ch, r"$\overleftarrow{h}$", fc=fc, ec=ec, fs=6.0, r=0.02)
        text(ax, xc, y0 + 0.02, f"$x_{{{idx[i]}}}$", fs=5.5, va="bottom")
        text(ax, xc, yf + ch + 0.065, f"$y_{{{idx[i]}}}$", fs=5.5, va="bottom")
    for i in range(n - 1):
        xa, xb = xs[i] + cw, xs[i + 1]
        arrow(ax, (xa, yf + ch / 2), (xb, yf + ch / 2), mutation=4)      # forward →
        arrow(ax, (xb, yb + ch / 2), (xa, yb + ch / 2), mutation=4)      # backward ←
    return xs[-1] + cw, yf + ch / 2, yf + ch + 0.14


def scan_grid(ax, x, y, s, direction, n=4, color=EDGE_B):
    """n×n 격자 + 각 행/열의 독립 스캔 방향."""
    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=EDGE, lw=0.4, zorder=3))
    for i in range(1, n):
        ax.add_line(Line2D([x, x + s], [y + i * s / n] * 2, color="#c8c8c8", lw=0.3, zorder=3))
        ax.add_line(Line2D([x + i * s / n] * 2, [y, y + s], color="#c8c8c8", lw=0.3, zorder=3))
    c = [(i + 0.5) * s / n for i in range(n)]
    for v in c:
        if direction == "r":
            arrow(ax, (x + 0.02, y + v), (x + s - 0.02, y + v), color=color, lw=0.6, mutation=3.5)
        elif direction == "l":
            arrow(ax, (x + s - 0.02, y + v), (x + 0.02, y + v), color=color, lw=0.6, mutation=3.5)
        elif direction == "d":
            arrow(ax, (x + v, y + s - 0.02), (x + v, y + 0.02), color=color, lw=0.6, mutation=3.5)
        else:
            arrow(ax, (x + v, y + 0.02), (x + v, y + s - 0.02), color=color, lw=0.6, mutation=3.5)


def fig2(th):
    H = 2.30
    fig, ax = canvas(H)
    y0 = 1.40                                  # (a) chain baseline
    yr = 0.66                                  # (b) main-row box bottom

    # ─────────── (a) bi-GRU arm ───────────
    text(ax, 0.04, H - 0.05, "(a)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, H - 0.05, "bi-GRU (original ETER-Net)", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, H - 0.05, "668.2M (GRU stack 637.1M)", fs=FS_S, color=MUTED, ha="right", va="top")
    ts, tx, ty = 0.34, 0.06, y0 + 0.20
    thumb(ax, tx, ty, ts, th["ksp_und"])
    ax.add_line(Line2D([tx, tx + ts], [ty + ts * 0.5] * 2, color=EDGE_R, lw=0.9, zorder=5))
    lab(ax, 0.04, ty - 0.03, "$x_t$ = k-space row $t$", ha="left", va="top", fs=5.0, color=INK, max_w=0.56)
    lab(ax, 0.04, ty - 0.12, "12,288-d (32×384)", ha="left", va="top", fs=5.0, max_w=0.54)
    arrow(ax, (tx + ts + 0.01, ty + ts * 0.5), (0.58, ty + ts * 0.5))
    x_end, y_mid, y_top = unrolled_birnn(ax, 0.60, y0, cw=0.16, gap=0.085)
    y_mid = y0 + 0.315                         # centre of the two-row block (= layer output)
    text(ax, 1.05, y_top, "pass 1: rows — 384 steps, hidden 3,840 per direction", fs=5.0, color=INK2,
         va="bottom", max_w=1.6)
    bx, bw2 = 1.84, 0.54
    arrow(ax, (x_end + 0.01, y_mid), (bx, y_mid))
    lab(ax, (x_end + bx) / 2, y_mid + 0.03, "transpose", fs=4.8, max_w=bx - x_end - 0.03)
    box(ax, bx, y_mid - 0.125, bw2, 0.25, "pass 2: bi-GRU", sub="columns, 7,680-d", fc=FILL_R, ec=EDGE_R,
        fs=5.4, sub_fs=4.7)
    arrow(ax, (bx + bw2, y_mid), (bx + bw2 + 0.28, y_mid))
    lab(ax, bx + bw2 + 0.14, y_mid + 0.03, "reshape", fs=4.8, max_w=0.25)
    text(ax, bx + bw2 + 0.30, y_mid, "20×384$^2$", fs=5.2, ha="left")
    lab(ax, 0.04, y0 - 0.03, "recurrence sequential in $t$; input–hidden matrices 12,288×11,520 and 7,680×11,520 per direction",
        ha="left", va="top", fs=4.8, max_w=W - 0.08)

    # ─────────── (b) SS2D arm ───────────
    ptop = y0 - 0.18
    text(ax, 0.04, ptop, "(b)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, ptop, "SS2D (controlled substitution)", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, ptop, "31.2M (SSM stack 0.117M)", fs=FS_S, color=MUTED, ha="right", va="top")
    bh = 0.22
    ym = yr + bh / 2
    box(ax, 0.04, yr, 0.52, bh, "LN·Linear·SiLU", sub="32 → 128", fs=5.6, sub_fs=4.8)
    lab(ax, 0.04, yr + bh + 0.03, "k-space 32×384$^2$", ha="left", fs=5.0)
    arrow(ax, (0.56, ym), (0.62, ym))
    box(ax, 0.62, yr, 0.44, bh, "DWConv 3×3", sub="SiLU, 128", fs=5.6, sub_fs=4.8)
    # cross-scan: four grids
    gs, gg = 0.20, 0.05
    gx0, gy0 = 1.18, yr - 0.14
    for d, i, j in (("r", 0, 1), ("l", 1, 1), ("d", 0, 0), ("u", 1, 0)):
        scan_grid(ax, gx0 + i * (gs + gg), gy0 + j * (gs + gg), gs, d)
    gxc = gx0 + gs + gg / 2
    text(ax, gxc, gy0 + 2 * gs + gg + 0.04, "cross-scan", fs=5.2, va="bottom")
    lab(ax, gxc, gy0 - 0.03, "rows →←, cols ↓↑, $L$=384", va="top", fs=4.8)
    arrow(ax, (1.06, ym), (gx0 - 0.005, ym))
    # four S6 scans — independent weights per direction (ss2d.py: ssm_h_fwd/h_bwd/v_fwd/v_bwd), each shared by all its rows/columns
    s6x, s6w, s6h, pitch = 1.74, 0.36, 0.10, 0.115
    ys6 = [gy0 + 0.005 + k * pitch for k in range(4)]
    for yy in ys6:
        box(ax, s6x, yy, s6w, s6h, "S6", fc=FILL_B, ec=EDGE_B, fs=5.8, r=0.02)
        arrow(ax, (gx0 + 2 * gs + gg + 0.005, yy + s6h / 2), (s6x, yy + s6h / 2), mutation=4)
    text(ax, s6x + s6w / 2, gy0 + 2 * gs + gg + 0.04, "S6 ×4 (parallel)", fs=5.2, va="bottom")
    # concat merge
    mx = 2.20
    for yy in ys6:
        polyline(ax, [(s6x + s6w, yy + s6h / 2), (mx, yy + s6h / 2), (mx, ym)], head=False)
    op(ax, mx, ym, "C", fs=6.0)
    box(ax, 2.28, yr, 0.40, bh, "LN·Linear", sub="512 → 128", fs=5.6, sub_fs=4.8)
    arrow(ax, (mx + 0.055, ym), (2.28, ym))
    arrow(ax, (2.68, ym), (2.74, ym))
    box(ax, 2.74, yr, 0.36, bh, "1×1 conv", sub="128 → 20", fs=5.6, sub_fs=4.8)
    lab(ax, 2.92, yr - 0.03, "20×384$^2$", va="top", fs=5.0, color=INK)
    # S6 recurrence + hyper-parameters
    text(ax, 0.04, 0.30, "S6:  $h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t$,   $y_t = C_t h_t + D x_t$,   "
         "$(\\Delta_t, B_t, C_t)$ from $x_t$", fs=5.6, ha="left", va="center", max_w=W - 0.08)
    lab(ax, 0.04, 0.15, "d_inner 128, d_state 16; one S6 weight set per direction, shared by all its rows (columns)",
        ha="left", va="center", fs=4.9, max_w=W - 0.08)
    save(fig, "conf_fig2_arms")


# ═════════════════════════ Fig. 3 — enhanced SS2D ═════════════════════════
def fig3(th):
    H = 1.88
    fig, ax = canvas(H)
    text(ax, 0.04, H - 0.05, "enhanced SS2D (replaces $f_\\theta$ in Fig. 1)", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, H - 0.05, "34.2M (SSM stack 3.1M), fp16 scan", fs=FS_S, color=MUTED, ha="right", va="top")

    # ── top chain ──
    bh, yt = 0.24, H - 0.62
    specs = [("stem", "32 → 256", FILL, EDGE),
             ("conv ↓3", "256, stride 3", FILL, EDGE),
             ("SS2D block", "×3, 256", FILL_G, EDGE_G),
             ("upsample ↑3", "LN · bilinear", FILL, EDGE),
             ("head", "256 → 64", FILL, EDGE)]
    dims = ["32×384$^2$", "256×384$^2$", "256×128$^2$", "256×128$^2$", "256×384$^2$", "64×384$^2$"]
    bw, gap, x = 0.42, 0.20, 0.04
    xs = []
    for k, (lbl, sub, fc, ec) in enumerate(specs):
        if k == 2:
            for off in (0.05, 0.025):
                ax.add_patch(FancyBboxPatch((x + off, yt + off), bw, bh, boxstyle="round,pad=0,rounding_size=0.035",
                                            fc=fc, ec=ec, lw=LW_BOX, zorder=2))
        box(ax, x, yt, bw, bh, lbl, sub=sub, fc=fc, ec=ec, fs=5.8, sub_fs=4.5)
        xs.append(x)
        x += bw + gap
    for k in range(len(specs) - 1):
        xa, xb = xs[k] + bw + (0.05 if k == 2 else 0), xs[k + 1]
        arrow(ax, (xa, yt + bh / 2), (xb, yt + bh / 2))
        lab(ax, (xa + xb) / 2, yt + bh / 2 + 0.03, dims[k + 1].split("×")[1], fs=4.7)
    lab(ax, xs[0], yt + bh + 0.03, dims[0], ha="left", fs=4.7)
    x_out = xs[-1] + bw
    arrow(ax, (x_out, yt + bh / 2), (W - 0.04, yt + bh / 2))
    lab(ax, W - 0.04, yt + bh + 0.03, dims[-1], ha="right", fs=4.7)                   # output tensor
    lab(ax, W - 0.04, yt - 0.03, "to concat · U-Net (Fig. 1)", ha="right", va="top", fs=4.7)
    lab(ax, 0.04, yt - 0.03, "stem = LN · Linear · SiLU;   head = 3×3 conv · SiLU · 1×1 conv", ha="left", va="top",
        fs=4.5, max_w=W - 0.08 - 0.90)

    # ── one block (bottom panel) ──
    py = 0.06
    ph = yt - 0.20 - py
    ax.add_patch(FancyBboxPatch((0.04, py), W - 0.08, ph, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=EDGE_G, lw=0.7, ls=(0, (2.2, 1.4)), zorder=1))
    text(ax, 0.10, py + ph - 0.045, "one SS2D block: 256 channels on the 128$^2$ grid", fs=FS, ha="left", va="top")
    hb = 0.20
    c = py + 0.50                       # main line
    cu, cl = c + 0.24, c - 0.24         # upper (SSM) / lower (gate) branches
    text(ax, 0.08, c, "$x$", fs=6.5, ha="left")
    dot(ax, 0.19, c)
    ax.add_line(Line2D([0.15, 0.19], [c, c], color=INK, lw=LW_ARR, zorder=4))
    box(ax, 0.26, c - hb / 2, 0.38, hb, "LN · Linear", sub="256 → 512", fs=5.6, sub_fs=4.8)
    arrow(ax, (0.19, c), (0.26, c))
    dot(ax, 0.72, c)
    arrow(ax, (0.64, c), (0.72, c), head=False)
    polyline(ax, [(0.72, c), (0.72, cu), (0.90, cu)])
    lab(ax, 0.81, cu + 0.03, "$x_{\\mathrm{ssm}}$", fs=5.0, color=INK)
    polyline(ax, [(0.72, c), (0.72, cl), (0.90, cl)])
    lab(ax, 0.81, cl + 0.03, "$z$", fs=5.0, color=INK)
    box(ax, 0.90, cu - hb / 2, 0.46, hb, "DWConv 3×3", sub="SiLU, 256", fs=5.4, sub_fs=4.8)
    arrow(ax, (1.36, cu), (1.42, cu))
    box(ax, 1.42, cu - hb / 2, 0.56, hb, "4-dir scan · merge", sub="d_inner 256, N 32", fc=FILL_B, ec=EDGE_B,
        fs=5.4, sub_fs=4.6)
    box(ax, 0.90, cl - hb / 2, 0.46, hb, "SiLU", sub="gate", fs=5.6, sub_fs=4.8)
    gx = 2.10
    polyline(ax, [(1.98, cu), (gx, cu), (gx, c + 0.055)])
    polyline(ax, [(1.36, cl), (gx, cl), (gx, c - 0.055)])
    op(ax, gx, c, r"$\otimes$")
    box(ax, 2.20, c - hb / 2, 0.42, hb, "Linear", sub="drop 0.05", fs=5.6, sub_fs=4.8)
    arrow(ax, (gx + 0.055, c), (2.20, c))
    ox = 2.74
    arrow(ax, (2.62, c), (ox - 0.055, c))
    op(ax, ox, c, r"$\oplus$")
    arrow(ax, (ox + 0.055, c), (ox + 0.14, c))
    text(ax, ox + 0.15, c, "out", fs=5.8, ha="left")
    yres = py + 0.09
    polyline(ax, [(0.19, c), (0.19, yres), (ox, yres), (ox, c - 0.055)], ls=(0, (1.6, 1.2)), color=INK2)
    lab(ax, 1.95, yres + 0.02, "residual", fs=4.8)
    save(fig, "conf_fig3_enhanced")


if __name__ == "__main__":
    th = load_thumbs()
    fig1(th)
    fig2(th)
    fig3(th)
    if _OVERFLOW:
        for s, w_in, mw in _OVERFLOW:
            print(f"  ! overflow ({w_in:.2f} > {mw:.2f} in): {s!r}")
    else:
        print("no text overflow")
