#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학술지판 그림 1 — 아키텍처 (page 폭, 고전적 논문 블록 다이어그램 문법 v2, 2026-09-09).

  (a) 두 모델에 공통인 ETER-Net 파이프라인(가중치 공유 없음) — 데이터 노드는 고정 대표 슬라이스 실제 썸네일, 화살표 위 숫자 = 채널 수
  (b) 시퀀스 모듈 두 구성 — 왼쪽 bi-GRU(펼친 양방향 순환 체인 → 전치 → 2단), 오른쪽 SS2D(행·열 4방향 스캔 → S6×4 → Ⓒ → LN·Linear → 1×1 conv)
  (c) 강화 SS2D — 왼쪽 블록 체인(화살표 위 공간 크기), 오른쪽 게이트 잔차 블록 내부

양식 규칙(균일 블록·화살표 위 크기·실데이터 썸네일·펼친 bi-RNN·4방향(행·열) 스캔→S6×4→Ⓒ·게이트 잔차 블록·Liberation Sans)과
도우미·썸네일 로더는 학술대회판 스크립트 paper/make_figs_conf_arch.py 를 그대로 import 한다(단일 출처 — 규칙을 바꾸면 그쪽을 고친다).
(a) 는 page 폭에 맞춰 다시 배치했고, (b)·(c) 는 학술대회판 그림 2·3 과 같은 축척(각 반폭 ≈ 단 폭)이다.

폭 = IEIE 학술지 본문 폭 9637 twips = 6.69 in (build_ieie_docx.py PAGE_W), 600 dpi PNG + PDF.
출력: paper/figs/fig1_architecture.{png,pdf}
실행: CUDA_VISIBLE_DEVICES="" python paper/make_fig1_architecture.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figs_conf_arch as C                                   # noqa: E402  (글꼴 등록·스타일 상수 포함)
from make_figs_conf_arch import (                                 # noqa: E402
    FancyBboxPatch, Line2D, arrow, box, canvas, dot, lab, load_thumbs, op, polyline, save, scan_grid,
    text, thumb, unrolled_birnn,
    EDGE, EDGE_B, EDGE_G, EDGE_R, FILL, FILL_B, FILL_G, FILL_R, FS, FS_P, FS_S, INK, INK2, LW_ARR, LW_BOX, MUTED,
)

PW = 9637 / 1440.0            # page width (in) — IEIE 학술지 본문 폭
H = 4.15                      # figure height (in)
HALF = 3.36                   # (b)·(c) 오른쪽 반폭의 x 오프셋 (왼쪽 반폭 0.04~3.25, 오른쪽 3.40~6.65)


# ═════════════════════════ (a) common pipeline (no weight sharing) ═════════════════════════
def panel_a(ax, th, top):
    vmax = float(th["gt"].max())
    text(ax, 0.04, top - 0.03, "(a)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.30, top - 0.03, "common ETER-Net pipeline (no weight sharing)", fs=FS, ha="left", va="top")
    T = 0.46                                  # thumbnail size (main row)
    yc = top - 0.60                           # main-row centre line
    y1 = yc - T / 2

    # fully-sampled k-space ⊙ mask → undersampled k-space   (row starts at x = 0.38 so the labels stay inside the page)
    x_yc = 0.38
    thumb(ax, x_yc, y1, T, th["ksp_full"])
    text(ax, x_yc + T / 2, y1 - 0.045, "fully-sampled k-space $y_c$", fs=6.0, va="top")
    lab(ax, x_yc + T / 2, y1 - 0.15, "16 coils, 384$^2$ (crop/pad), Re/Im", va="top", fs=5.2)
    mx_ = x_yc + 0.70                         # ⊙ centre
    arrow(ax, (x_yc + T, yc), (mx_ - 0.055, yc))
    op(ax, mx_, yc, r"$\odot$")
    thumb(ax, mx_ - 0.13, yc + 0.18, 0.26, th["mask2d"], vmin=0, vmax=1)
    arrow(ax, (mx_, yc + 0.18), (mx_, yc + 0.055))
    text(ax, mx_ + 0.20, yc + 0.40, "mask $M$", fs=6.0, ha="left")
    lab(ax, mx_ + 0.20, yc + 0.30, "R = 4 equispaced, 8 % ACS", ha="left", va="center", fs=5.2)
    x_yu = mx_ + 0.32
    arrow(ax, (mx_ + 0.055, yc), (x_yu, yc))
    thumb(ax, x_yu, y1, T, th["ksp_und"])
    text(ax, x_yu + T / 2, y1 - 0.045, r"$\tilde{y}_c = M \odot y_c$", fs=6.0, va="top")
    lab(ax, x_yu + T / 2, y1 - 0.15, "undersampled", va="top", fs=5.2)
    jx = x_yu + T + 0.20                      # junction: k-space feeds f_θ and the zero-filled branch
    arrow(ax, (x_yu + T, yc), (jx, yc), head=False)
    dot(ax, jx, yc)

    # sequence-module box (the only part replaced)
    sx, sw, sh = jx + 0.18, 1.14, 0.66
    sy = yc - sh / 2
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.035",
                                fc="white", ec=INK, lw=0.7, ls=(0, (2.2, 1.4)), zorder=3))
    text(ax, sx + sw / 2, sy + sh - 0.09, "sequence module $f_\\theta$", fs=7.0, max_w=sw - 0.04, z=4)
    text(ax, sx + sw / 2, sy + sh - 0.20, "only part replaced — see (b)", fs=5.4, color=MUTED, max_w=sw - 0.04, z=4)
    bw, bh = 0.42, 0.19
    box(ax, sx + 0.05, sy + 0.08, bw, bh, "bi-GRU", fc=FILL_R, ec=EDGE_R, fs=6.0)
    text(ax, sx + sw / 2, sy + 0.08 + bh / 2, "or", fs=5.4, color=MUTED, z=4)
    box(ax, sx + sw - 0.05 - bw, sy + 0.08, bw, bh, "SS2D", fc=FILL_B, ec=EDGE_B, fs=6.0)
    arrow(ax, (jx, yc), (sx, yc))
    lab(ax, (jx + sx) / 2, yc + 0.03, "32", fs=5.2)

    # concat → U-Net → reconstruction
    cx = sx + sw + 0.32
    arrow(ax, (sx + sw, yc), (cx - 0.055, yc))
    lab(ax, (sx + sw + cx - 0.055) / 2, yc + 0.03, "20", fs=5.2)
    op(ax, cx, yc, "C", fs=6.0)
    ux, uw = cx + 0.18, 0.90
    box(ax, ux, sy + 0.07, uw, sh - 0.14, "U-Net $g_\\phi$", sub="depth 5, 64 base channels\n31.1M, same architecture", fs=7.0, sub_fs=5.0)
    arrow(ax, (cx + 0.055, yc), (ux, yc))
    lab(ax, (cx + 0.055 + ux) / 2, yc + 0.03, "52", fs=5.2)
    rx = ux + uw + 0.30
    gx = rx + T + 0.44
    arrow(ax, (ux + uw, yc), (rx, yc))
    lab(ax, (ux + uw + rx) / 2, yc + 0.03, "1", fs=5.2)
    thumb(ax, rx, y1, T, th["recon"], vmin=0, vmax=vmax)
    text(ax, rx + T / 2, y1 - 0.045, r"reconstruction $\hat{x}$", fs=6.0, va="top")
    lab(ax, rx + T / 2, y1 - 0.15, "magnitude", va="top", fs=5.2)
    thumb(ax, gx, y1, T, th["gt"], vmin=0, vmax=vmax)
    text(ax, gx + T / 2, y1 - 0.045, "ground truth $x^{*}$", fs=6.0, va="top")
    lab(ax, gx + T / 2, y1 - 0.15, "dataset RSS", va="top", fs=5.2, max_w=T + 0.14)   # full-coil RSS shipped with fastMRI, crop/pad (not from the 16-coil y_c)
    yl = y1 + T + 0.10                        # loss bracket above the two images
    for xx in (rx + T / 2, gx + T / 2):
        ax.add_line(Line2D([xx, xx], [y1 + T, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    ax.add_line(Line2D([rx + T / 2, gx + T / 2], [yl, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    text(ax, (rx + gx + T) / 2, yl + 0.03, "loss: $L_1 + (1-\\mathrm{SSIM})$, brain mask", fs=5.6, va="bottom",
         max_w=1.30)

    # zero-filled branch (row below)
    zs = 0.40
    zc = y1 - 0.42
    zx, zy = sx + 0.40, zc - zs / 2
    polyline(ax, [(jx, yc), (jx, zc), (zx, zc)])
    lab(ax, (jx + zx) / 2, zc + 0.03, "$F^{-1}$", fs=6.0, color=INK)
    thumb(ax, zx, zy, zs, th["zf"], vmin=0, vmax=vmax)
    text(ax, zx + zs / 2, zy - 0.045, "zero-filled coil images", fs=5.8, va="top")
    polyline(ax, [(zx + zs, zc), (cx, zc), (cx, yc - 0.055)])
    lab(ax, (zx + zs + cx) / 2, zc + 0.03, "32", fs=5.2)
    lab(ax, PW - 0.05, zy - 0.045, "numbers on arrows: channels (spatial 384$^2$)", fs=5.2, ha="right", va="top")
    print(f"  (a) right edge: GT thumb ends at x = {gx + T:.2f} in (page {PW:.2f})")
    return zy - 0.17                          # panel bottom


# ═════════════════════════ (b) the two sequence modules ═════════════════════════
def panel_b(ax, th, top):
    text(ax, 0.04, top - 0.03, "(b)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.30, top - 0.03, "bi-GRU (original ETER-Net)", fs=FS, ha="left", va="top")
    text(ax, 3.25, top - 0.03, "668.2M (bi-GRU module 637.1M)", fs=FS_S, color=MUTED, ha="right", va="top")
    text(ax, HALF + 0.04, top - 0.03, "SS2D (controlled)", fs=FS, ha="left", va="top")
    text(ax, PW - 0.04, top - 0.03, "31.2M (SS2D module 0.12M)", fs=FS_S, color=MUTED, ha="right", va="top")

    # ── left: unrolled bi-GRU, two passes ──
    y0 = top - 0.84                           # chain baseline
    ts, tx, ty = 0.34, 0.06, y0 + 0.20
    thumb(ax, tx, ty, ts, th["ksp_und"])
    ax.add_line(Line2D([tx, tx + ts], [ty + ts * 0.5] * 2, color=EDGE_R, lw=0.9, zorder=5))
    lab(ax, 0.04, ty - 0.03, "$x_t$ = k-space row $t$", ha="left", va="top", fs=5.0, color=INK, max_w=0.56)
    lab(ax, 0.04, ty - 0.12, "12,288-d (32×384)", ha="left", va="top", fs=5.0, max_w=0.54)
    arrow(ax, (tx + ts + 0.01, ty + ts * 0.5), (0.58, ty + ts * 0.5))
    x_end, _, y_top = unrolled_birnn(ax, 0.60, y0, cw=0.16, gap=0.085)
    y_mid = y0 + 0.315                        # centre of the two-row block (= layer output)
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
        ha="left", va="top", fs=4.8, max_w=3.21)

    # ── right: SS2D = stem → 4-way row/column scan → S6 ×4 → concat → merge ──
    dx = HALF
    bh = 0.22
    yr = top - 0.58                           # main-row box bottom
    ym = yr + bh / 2
    box(ax, dx + 0.04, yr, 0.52, bh, "LN·Linear·SiLU", sub="32 → 128", fs=5.6, sub_fs=4.8)
    lab(ax, dx + 0.04, yr + bh + 0.03, "k-space 32×384$^2$", ha="left", fs=5.0)
    arrow(ax, (dx + 0.56, ym), (dx + 0.62, ym))
    box(ax, dx + 0.62, yr, 0.44, bh, "DWConv 3×3", sub="SiLU, 128", fs=5.6, sub_fs=4.8)
    gs, gg = 0.20, 0.05
    gx0, gy0 = dx + 1.18, yr - 0.14
    for d, i, j in (("r", 0, 1), ("l", 1, 1), ("d", 0, 0), ("u", 1, 0)):
        scan_grid(ax, gx0 + i * (gs + gg), gy0 + j * (gs + gg), gs, d)
    gxc = gx0 + gs + gg / 2
    text(ax, gxc, gy0 + 2 * gs + gg + 0.04, "4-way scan", fs=5.2, va="bottom")
    lab(ax, gxc, gy0 - 0.03, "rows →←, columns ↓↑, $L$=384", va="top", fs=4.8)
    arrow(ax, (dx + 1.06, ym), (gx0 - 0.005, ym))
    # four S6 scans — independent weights per direction, each shared by all its rows/columns
    s6x, s6w, s6h, pitch = dx + 1.74, 0.36, 0.10, 0.115
    ys6 = [gy0 + 0.005 + k * pitch for k in range(4)]
    for yy in ys6:
        box(ax, s6x, yy, s6w, s6h, "S6", fc=FILL_B, ec=EDGE_B, fs=5.8, r=0.02)
        arrow(ax, (gx0 + 2 * gs + gg + 0.005, yy + s6h / 2), (s6x, yy + s6h / 2), mutation=4)
    text(ax, s6x + s6w / 2, gy0 + 2 * gs + gg + 0.04, "S6 ×4 (parallel)", fs=5.2, va="bottom")
    mx = dx + 2.20
    for yy in ys6:
        polyline(ax, [(s6x + s6w, yy + s6h / 2), (mx, yy + s6h / 2), (mx, ym)], head=False)
    op(ax, mx, ym, "C", fs=6.0)
    box(ax, dx + 2.28, yr, 0.40, bh, "LN·Linear", sub="512 → 128", fs=5.6, sub_fs=4.8)
    arrow(ax, (mx + 0.055, ym), (dx + 2.28, ym))
    arrow(ax, (dx + 2.68, ym), (dx + 2.74, ym))
    box(ax, dx + 2.74, yr, 0.36, bh, "1×1 conv", sub="128 → 20", fs=5.6, sub_fs=4.8)
    lab(ax, dx + 2.92, yr - 0.03, "20×384$^2$", va="top", fs=5.0, color=INK)
    text(ax, dx + 0.04, yr - 0.36, "S6:  $h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t$,   $y_t = C_t h_t + D x_t$,   "
         "$(\\Delta_t, B_t, C_t)$ from $x_t$", fs=5.6, ha="left", va="center", max_w=PW - dx - 0.08)
    lab(ax, dx + 0.04, yr - 0.50, "128 inner channels, $N$ = 16; one S6 weight set per direction, shared by all its rows (columns)",
        ha="left", va="center", fs=4.9, max_w=PW - dx - 0.08)
    return yr - 0.55                          # panel bottom


# ═════════════════════════ (c) enhanced SS2D ═════════════════════════
def panel_c(ax, top):
    text(ax, 0.04, top - 0.03, "(c)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.30, top - 0.03, "SS2D (enhanced) — replaces $f_\\theta$ in (a)", fs=FS, ha="left", va="top")
    text(ax, PW - 0.04, top - 0.03, "34.2M (SS2D module 3.1M), fp16 scan", fs=FS_S, color=MUTED, ha="right", va="top")

    # ── left: block chain ──
    py, ph = 0.06, top - 0.20 - 0.06          # bottom panel extent (shared with the block detail on the right)
    bh = 0.24
    yt = py + ph / 2 - bh / 2 - 0.02
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
    arrow(ax, (x_out, yt + bh / 2), (3.24, yt + bh / 2))
    lab(ax, 3.24, yt + bh + 0.03, dims[-1], ha="right", fs=4.7)
    lab(ax, 3.24, yt - 0.03, "to concat · U-Net in (a)", ha="right", va="top", fs=4.7)
    lab(ax, 0.04, yt - 0.03, "stem = LN · Linear · SiLU;   head = 3×3 conv · SiLU · 1×1 conv", ha="left", va="top",
        fs=4.5, max_w=3.21 - 0.90)
    lab(ax, 0.04, yt - 0.13, "arrows: spatial size", ha="left", va="top", fs=4.5)

    # ── right: one block (gated residual, Mamba-style) ──
    dx = HALF
    ax.add_patch(FancyBboxPatch((dx + 0.04, py), PW - dx - 0.08, ph, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=EDGE_G, lw=0.7, ls=(0, (2.2, 1.4)), zorder=1))
    text(ax, dx + 0.10, py + ph - 0.045, "one SS2D block: 256 channels on the 128$^2$ grid", fs=FS, ha="left", va="top")
    hb = 0.20
    c = py + 0.50                             # main line
    cu, cl = c + 0.24, c - 0.24               # upper (SSM) / lower (gate) branches
    text(ax, dx + 0.08, c, "$x$", fs=6.5, ha="left")
    dot(ax, dx + 0.19, c)
    ax.add_line(Line2D([dx + 0.15, dx + 0.19], [c, c], color=INK, lw=LW_ARR, zorder=4))
    box(ax, dx + 0.26, c - hb / 2, 0.38, hb, "LN · Linear", sub="256 → 512", fs=5.6, sub_fs=4.8)
    arrow(ax, (dx + 0.19, c), (dx + 0.26, c))
    dot(ax, dx + 0.72, c)
    arrow(ax, (dx + 0.64, c), (dx + 0.72, c), head=False)
    polyline(ax, [(dx + 0.72, c), (dx + 0.72, cu), (dx + 0.90, cu)])
    lab(ax, dx + 0.81, cu + 0.03, "$x_{\\mathrm{ssm}}$", fs=5.0, color=INK)
    polyline(ax, [(dx + 0.72, c), (dx + 0.72, cl), (dx + 0.90, cl)])
    lab(ax, dx + 0.81, cl + 0.03, "$z$", fs=5.0, color=INK)
    box(ax, dx + 0.90, cu - hb / 2, 0.46, hb, "DWConv 3×3", sub="SiLU, 256", fs=5.4, sub_fs=4.8)
    arrow(ax, (dx + 1.36, cu), (dx + 1.42, cu))
    box(ax, dx + 1.42, cu - hb / 2, 0.56, hb, "4-dir scan · merge", sub="256 ch., $N$ = 32", fc=FILL_B, ec=EDGE_B,
        fs=5.4, sub_fs=4.6)
    box(ax, dx + 0.90, cl - hb / 2, 0.46, hb, "SiLU", sub="gate", fs=5.6, sub_fs=4.8)
    gx = dx + 2.10
    polyline(ax, [(dx + 1.98, cu), (gx, cu), (gx, c + 0.055)])
    polyline(ax, [(dx + 1.36, cl), (gx, cl), (gx, c - 0.055)])
    op(ax, gx, c, r"$\otimes$")
    box(ax, dx + 2.20, c - hb / 2, 0.42, hb, "Linear", sub="dropout 0.05", fs=5.6, sub_fs=4.8)
    arrow(ax, (gx + 0.055, c), (dx + 2.20, c))
    ox = dx + 2.74
    arrow(ax, (dx + 2.62, c), (ox - 0.055, c))
    op(ax, ox, c, r"$\oplus$")
    arrow(ax, (ox + 0.055, c), (ox + 0.14, c))
    text(ax, ox + 0.15, c, "out", fs=5.8, ha="left")
    yres = py + 0.09
    polyline(ax, [(dx + 0.19, c), (dx + 0.19, yres), (ox, yres), (ox, c - 0.055)], ls=(0, (1.6, 1.2)), color=INK2)
    lab(ax, dx + 1.95, yres + 0.02, "residual", fs=4.8)


def main():
    th = load_thumbs()
    fig, ax = canvas(H, w=PW)
    bottom_a = panel_a(ax, th, H)
    bottom_b = panel_b(ax, th, bottom_a - 0.02)
    panel_c(ax, bottom_b - 0.02)
    save(fig, "fig1_architecture")
    if C._OVERFLOW:
        for s, w_in, mw in C._OVERFLOW:
            print(f"  ! overflow ({w_in:.2f} > {mw:.2f} in): {s!r}")
    else:
        print("no text overflow")


if __name__ == "__main__":
    main()
