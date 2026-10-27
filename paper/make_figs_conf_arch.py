#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IEIE 학술대회판 전용 세부 아키텍처 그림 3장 — 고전적 블록 다이어그램 양식(v2, 2026-09-08).

  paper/figs/conf_fig1_pipeline.{png,pdf}   그림 1  두 모델에 공통인 파이프라인(가중치 공유 없음)(데이터 노드는 실제 썸네일)
  paper/figs/conf_fig2_modules.{png,pdf}       그림 2  (a) bi-GRU 모듈 = 펼친(unrolled) 양방향 순환 체인, (b) SS2D 모듈 = 행·열 4방향 스캔 → S6 → merge
  paper/figs/conf_fig3_enhanced.{png,pdf}   그림 3  강화 SS2D: 위 = 블록 체인(화살표 위 텐서 크기), 아래 = Mamba 블록 내부(게이트·잔차)

양식 규칙(v1 의 "슬라이드" 인상을 고친 지점):
  - 블록 = 균일한 크기의 둥근 사각형에 한 줄 라벨만. 텐서 크기·채널 수는 블록 안이 아니라 화살표 위 작은 회색 글자.
  - 설명 문장은 그림 안에 넣지 않는다(캡션·본문 몫). 굵은 제목·색 막대·각주 없음.
  - 데이터 노드(k-space·마스크·zero-filled·복원·GT)는 고정 대표 슬라이스의 실제 영상 썸네일.
  - RNN 은 펼친 셀 체인(→/← 두 줄), SS2D 는 행·열 4방향 스캔(4 격자) → S6 ×4 → merge, Mamba 블록은 원 논문의 게이트 분기 형태로 그린다.
  - 글꼴 Liberation Sans(Arial metric 호환, paper/fonts/) 6.5 pt 본문 / 5.5 pt 주석; 없으면 DejaVu Sans 폴백.
  - 단 폭 3.15 in, 600 dpi PNG + PDF. 색은 모델 식별용 3색(bi-GRU 빨강·SS2D 파랑·강화 초록)만 옅게.

데이터(로컬 전용): 고정 대표 슬라이스 index 4689 = fastMRI_data/multicoil_val/file_brain_AXT2_203_2030309.h5 slice 7
(visualize_slices_canonical.json) 의 k-space 를 dataloader_h5_v5 와 같은 전처리(코일 영상 384×384 크롭 및 패딩 → 다시 푸리에 변환 →
R=4 equispaced 마스크, offset 3)로 만들고, zero-filled·SS2D 복원·GT 는 results/vis/multimodel_compare/recon_4689.npz 에서 읽는다.
둘 중 하나라도 없으면 합성 자리표시 썸네일로 대체하고 경고를 출력한다.

실행: CUDA_VISIBLE_DEVICES="" python paper/make_figs_conf_arch.py          (v8 판 4장 — 기존과 바이트 동일 PNG)
v9 판(2026-10-17, IEIE v9 초안 — 모든 글자 ≥ 6.5 pt 재배치, 손실 L_1 + λ(1−SSIM), 복원 썸네일 = v8fix SS2D(controlled)):
      CUDA_VISIBLE_DEVICES="" python paper/make_figs_conf_arch.py --suffix _v9 \
          --npz results/vis/multimodel_compare_v8fix/recon_4689.npz
      → conf_fig1_pipeline_v9 · conf_fig2_modules_v9 · conf_fig2_modules_impl_v9 (.png/.pdf). conf_fig3_enhanced 는 v9 에서 만들지 않음.
      v9 판은 그리는 동안만 mathtext 첨자 축소를 끈다(SHRINK_FACTOR 1.0, 아래첨자 내림 0.6 x-height) — 첨자도 6.5 pt. 384² 는 384×384.
      그린 뒤 자체 점검(audit_text, 저장과 같은 600 dpi 래스터): Text 공칭 크기와 mathtext 가 실제로 그리는 글리프(첨자 포함) ≥ 6.5 pt ·
      유니코드 첨자 문자 금지 · 그림 밖 잘림 · 글자 겹침 · 같은 줄 글자 간격 ≥ 0.06 in · 글자 잉크 ↔ 상자·원·화살표·선·영상 간격 ≥ 0.02 in
      (상자 안 글자는 테두리까지) · 상자 넘침 → 위반 시 exit 1. 저장된 PNG 의 실제 픽셀 크기를 출력. STIX 대체 글리프(⊙·ℱ·h 위 화살표)는
      목록으로 보고(PDF 에 STIXGeneral-Regular 가 함께 들어감 — v8 PDF 와 같음).
      v8 판(인자 없음)은 고정 이름으로 저장하므로 --suffix·--npz 와 함께 쓰면 거부한다(커밋된 v8 그림 보호).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
import matplotlib.text

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
    # \mathcal·\mathsf 기본값(cursive·sans)은 글꼴을 찾지 못해 findfont 경고를 낸다 — 이 그림들은 쓰지 않지만 경고를 없애려고
    # Liberation Sans 로 지정(출력 무변화: v8 PNG 바이트 동일 확인). 유니코드 수학 기호(⊙·ℱ 등)는 위 fallback(STIX)으로 그려진다.
    rcParams["mathtext.cal"] = "Liberation Sans:italic"
    rcParams["mathtext.sf"] = "Liberation Sans"
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
_MIN_FS = 4.6                 # text() 자동 축소 하한(pt). v9 레이아웃은 6.5 로 올려 축소를 금지한다(넘치면 _OVERFLOW 기록).


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
    """텍스트 배치. max_w(in) 를 넘으면 _MIN_FS(기본 4.6 pt) 까지 자동 축소(렌더러 실측)."""
    t = ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, **kw)
    if max_w is not None:
        fig = ax.figure
        r = fig.canvas.get_renderer()
        while True:
            w_in = t.get_window_extent(renderer=r).width / fig.dpi
            if w_in <= max_w or t.get_fontsize() <= _MIN_FS:
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


def load_thumbs(npz=NPZ):
    """고정 대표 슬라이스 4689 의 실제 영상. dict(ksp_full, mask2d, ksp_und, zf, recon, gt).
    npz: zero-filled·복원·GT 출처(기본 = v8 붕괴 런의 recon_4689.npz, v9 는 --npz 로 v8fix 판). 복원 = rec_ss2d."""
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
        d = np.load(npz)
        out.update(zf=d["rec_zf"], recon=d["rec_ss2d"], gt=d["gt"])
    except Exception as e:                                             # pragma: no cover
        warnings.warn(f"복원 썸네일 로드 실패({e!r}) → 합성 자리표시")
        yy, xx = np.mgrid[0:384, 0:384]
        disk = (np.hypot(yy - 192, xx - 192) < 140).astype(np.float32)
        out.update(zf=disk * 0.7, recon=disk, gt=disk)
    return out


# ═════════════════════════ Fig. 1 — common pipeline (no weight sharing) ═════════════════════════
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
    text(ax, 0.23, y1 - 0.045, r"$\tilde{y}_c = M \odot y_c$", fs=6.0, va="top")
    lab(ax, 0.23, y1 - 0.15, "undersampled", va="top", fs=4.9)
    dot(ax, 0.46, yc)
    arrow(ax, (0.42, yc), (0.46, yc), head=False)
    # sequence-module box (the only part replaced)
    sx, sw, sh = 0.56, 0.72, 0.50
    sy = yc - sh / 2
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.035",
                                fc="white", ec=INK, lw=0.7, ls=(0, (2.2, 1.4)), zorder=3))
    text(ax, sx + sw / 2, sy + sh - 0.08, "sequence module $f_\\theta$", fs=FS, max_w=sw - 0.04, z=4)
    text(ax, sx + sw / 2, sy + sh - 0.165, "(only part replaced)", fs=4.9, color=MUTED, max_w=sw - 0.04, z=4)
    bw, bh = 0.28, 0.15
    box(ax, sx + 0.03, sy + 0.05, bw, bh, "bi-GRU", fc=FILL_R, ec=EDGE_R, fs=5.4)
    text(ax, sx + sw / 2, sy + 0.05 + bh / 2, "or", fs=5.0, color=MUTED, z=4)
    box(ax, sx + sw - 0.03 - bw, sy + 0.05, bw, bh, "SS2D", fc=FILL_B, ec=EDGE_B, fs=5.4)
    arrow(ax, (0.46, yc), (sx, yc))
    lab(ax, 0.505, yc + 0.03, "32", fs=4.8)
    # concat
    cx = 1.42
    arrow(ax, (sx + sw, yc), (cx - 0.055, yc))
    lab(ax, (sx + sw + cx - 0.055) / 2 + 0.012, yc + 0.03, "20", fs=4.8)   # 10-07: 점선 테두리에서 띄움
    op(ax, cx, yc, "C", fs=6.0)
    # U-Net
    ux, uw = 1.55, 0.48
    box(ax, ux, sy + 0.05, uw, sh - 0.10, "U-Net $g_\\varphi$", sub="31.1M, same\narchitecture")
    arrow(ax, (cx + 0.055, yc), (ux, yc))
    lab(ax, (cx + 0.055 + ux) / 2, yc + 0.03, "52", fs=5.0)
    # reconstruction / ground truth
    rx, gx = 2.12, 2.68
    arrow(ax, (ux + uw, yc), (rx, yc))
    thumb(ax, rx, y1, T, th["recon"], vmin=0, vmax=vmax)
    text(ax, rx + T / 2, y1 - 0.045, r"reconstruction $\hat{x}$", fs=5.6, va="top")
    lab(ax, rx + T / 2, y1 - 0.15, "magnitude", va="top", fs=4.9)
    thumb(ax, gx, y1, T, th["gt"], vmin=0, vmax=vmax)
    text(ax, gx + T / 2, y1 - 0.045, "ground truth $x^{*}$", fs=5.6, va="top")
    lab(ax, gx + T / 2, y1 - 0.15, "dataset RSS", va="top", fs=4.9, max_w=T + 0.14)   # full-coil RSS shipped with fastMRI, crop/pad to 384² (not derived from the 16-coil y_c)
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
    lab(ax, 0.59, zc + 0.03, "$ℱ^{-1}$", fs=6.0, color=INK)
    thumb(ax, zx, zy, zs, th["zf"], vmin=0, vmax=vmax)
    text(ax, zx + zs / 2, zy - 0.045, "zero-filled coil images", fs=5.8, va="top")
    polyline(ax, [(zx + zs, zc), (cx, zc), (cx, yc - 0.055)])
    lab(ax, (zx + zs + cx) / 2, zc + 0.03, "32", fs=5.0)
    lab(ax, W - 0.05, Y_LO + 0.05, "numbers on arrows: channels (384$^2$)", fs=4.9, ha="right", va="bottom")
    save(fig, "conf_fig1_pipeline")


# ═════════════════════════ Fig. 2 — the two sequence modules ═════════════════════════
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


def fig2(th, variant="rows"):
    """variant="rows": 기존 판(한 스텝 = k-space 한 행으로 표현).
    variant="impl" (10-07): 원 구현대로 — (32, 384, 384) 입력을 순열 없이 (384, 12,288) 로 펼치므로
    스텝 t = 채널 ⌊t/12⌋ 의 연속된 32개 행(32×384). 교수님 확인 결과에 따라 둘 중 하나를 쓴다."""
    impl = variant == "impl"
    H = 2.30
    fig, ax = canvas(H, y_lo=0.50)             # 10-01: 하단 설명 두 줄 삭제 → 아래 0.50 in 잘라냄(높이 1.80 in)
    y0 = 1.40                                  # (a) chain baseline
    yr = 0.78                                  # (b) main-row box bottom

    # ─────────── (a) bi-GRU module ───────────
    text(ax, 0.04, H - 0.05, "(a)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, H - 0.05, "bi-GRU module (ETER-net)", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, H - 0.05, "637.1M (model 668.2M)", fs=FS_S, color=MUTED, ha="right", va="top")
    ts, tx, ty = 0.34, 0.06, y0 + 0.20
    if impl:
        for k in (2, 1):                       # 뒤에 겹친 사각형 = 채널 축(32채널)
            ax.add_patch(Rectangle((tx + 0.025 * k, ty + 0.025 * k), ts, ts, fc="white", ec=EDGE, lw=0.4, zorder=2))
    thumb(ax, tx, ty, ts, th["ksp_und"])
    if impl:
        bh_ = ts / 12                          # 32/384 = 1/12 높이의 띠
        ax.add_patch(Rectangle((tx, ty + ts * 0.5 - bh_ / 2), ts, bh_, fc=EDGE_R, ec=EDGE_R, lw=0.6,
                               alpha=0.45, zorder=5))
        lab(ax, 0.04, ty - 0.03, "$x_t$: 32 rows × 384", ha="left", va="top", fs=5.0, color=INK, max_w=0.56)
        lab(ax, 0.04, ty - 0.12, "of one channel", ha="left", va="top", fs=5.0, max_w=0.54)
    else:
        ax.add_line(Line2D([tx, tx + ts], [ty + ts * 0.5] * 2, color=EDGE_R, lw=0.9, zorder=5))
        lab(ax, 0.04, ty - 0.03, "$x_t$ = k-space row $t$", ha="left", va="top", fs=5.0, color=INK, max_w=0.56)
        lab(ax, 0.04, ty - 0.12, "12,288-d (32×384)", ha="left", va="top", fs=5.0, max_w=0.54)
    arrow(ax, (tx + ts + 0.01, ty + ts * 0.5), (0.58, ty + ts * 0.5))
    x_end, y_mid, y_top = unrolled_birnn(ax, 0.60, y0, cw=0.16, gap=0.085)
    y_mid = y0 + 0.315                         # centre of the two-row block (= layer output)
    text(ax, 1.05, y_top, "stage 1 — 384 steps (12,288-d each)" if impl else "stage 1: rows — 384 steps",
         fs=5.0, color=INK2, va="bottom", max_w=1.6)
    bx, bw2 = 1.84, 0.62
    arrow(ax, (x_end + 0.01, y_mid), (bx, y_mid))
    lab(ax, (x_end + bx) / 2, y_mid + 0.03, "transpose", fs=4.8, max_w=bx - x_end - 0.03)
    box(ax, bx, y_mid - 0.125, bw2, 0.25, "stage 2" if impl else "stage 2: columns", sub="384 steps",
        fc=FILL_R, ec=EDGE_R, fs=5.4, sub_fs=4.7)
    arrow(ax, (bx + bw2, y_mid), (bx + bw2 + 0.28, y_mid))
    lab(ax, bx + bw2 + 0.14, y_mid + 0.03, "reshape", fs=4.8, max_w=0.25)
    text(ax, bx + bw2 + 0.30, y_mid, "20×384$^2$", fs=5.2, ha="left")

    # ─────────── (b) SS2D module ───────────
    ptop = y0 - 0.06
    text(ax, 0.04, ptop, "(b)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, ptop, "SS2D module", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, ptop, "0.12M (model 31.2M)", fs=FS_S, color=MUTED, ha="right", va="top")
    bh = 0.22
    ym = yr + bh / 2
    box(ax, 0.04, yr, 0.52, bh, "LN·Linear·SiLU", sub="32 → 128", fs=5.6, sub_fs=5.2)
    lab(ax, 0.04, yr + bh + 0.03, "k-space 32×384$^2$", ha="left", fs=5.0)
    arrow(ax, (0.56, ym), (0.62, ym))
    box(ax, 0.62, yr, 0.44, bh, "DWConv 3×3", sub="SiLU, 128", fs=5.6, sub_fs=5.2)
    # 4-way row/column scan: four grids
    gs, gg = 0.20, 0.05
    gx0, gy0 = 1.18, yr - 0.14
    for d, i, j in (("r", 0, 1), ("l", 1, 1), ("d", 0, 0), ("u", 1, 0)):
        scan_grid(ax, gx0 + i * (gs + gg), gy0 + j * (gs + gg), gs, d)
    gxc = gx0 + gs + gg / 2
    text(ax, gxc, gy0 + 2 * gs + gg + 0.04, "4-way scan", fs=5.2, va="bottom")
    lab(ax, gxc, gy0 - 0.03, "rows →←, columns ↓↑", va="top", fs=5.0)
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
    box(ax, 2.28, yr, 0.40, bh, "LN·Linear", sub="512 → 128", fs=5.6, sub_fs=5.2)
    arrow(ax, (mx + 0.055, ym), (2.28, ym))
    arrow(ax, (2.68, ym), (2.74, ym))
    box(ax, 2.74, yr, 0.36, bh, "1×1 conv", sub="128 → 20", fs=5.6, sub_fs=5.2)
    lab(ax, 2.92, yr - 0.03, "20×384$^2$", va="top", fs=5.0, color=INK)
    # 10-01 교수님 검토: 식은 본문 식 (2)에 있으므로 그림에서 빼고, 작은 변수 설명 줄도 뺀다.
    save(fig, "conf_fig2_modules_impl" if impl else "conf_fig2_modules")


# ═════════════════════════ Fig. 3 — enhanced SS2D ═════════════════════════
def fig3(th):
    H = 1.88
    fig, ax = canvas(H)
    text(ax, 0.04, H - 0.05, "SS2D (enhanced) — replaces $f_\\theta$ in Fig. 1", fs=FS, ha="left", va="top")
    text(ax, W - 0.04, H - 0.05, "34.2M (SS2D module 3.1M), fp16 scan", fs=FS_S, color=MUTED, ha="right", va="top")

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
    box(ax, 1.42, cu - hb / 2, 0.56, hb, "4-dir scan · merge", sub="256 ch., $N$ = 32", fc=FILL_B, ec=EDGE_B,
        fs=5.4, sub_fs=4.6)
    box(ax, 0.90, cl - hb / 2, 0.46, hb, "SiLU", sub="gate", fs=5.6, sub_fs=4.8)
    gx = 2.10
    polyline(ax, [(1.98, cu), (gx, cu), (gx, c + 0.055)])
    polyline(ax, [(1.36, cl), (gx, cl), (gx, c - 0.055)])
    op(ax, gx, c, r"$\otimes$")
    box(ax, 2.20, c - hb / 2, 0.42, hb, "Linear", sub="dropout 0.05", fs=5.6, sub_fs=4.8)
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


# ═════════════════════════ v9 판 (2026-10-17) — 모든 글자 ≥ 6.5 pt ═════════════════════════
# IEIE v9 초안용. 위 v8 판 함수(fig1·fig2·fig3)는 그대로 두어 v8 그림을 재현할 수 있게 하고,
# 같은 단 폭 3.15 in 에서 글자를 6.5 pt 로 올린 재배치 판을 새 함수로 둔다(출력 이름 = <name><suffix>).
# 재배치(글자 폭 때문에 한 줄에 다 들어가지 않음 — 6.5 pt 실측 폭 합 ≈ 4.0 in > 3.1 in):
#   그림 1  주 행 = ỹ_c → f_θ → Ⓒ → U-Net, 아래 행 = zero-filled(왼쪽) · 정답(GT)·복원(오른쪽, U-Net 출력이 꺾여 내려옴).
#           GT·복원 라벨은 두 줄(이름 / 기호, 설명) — 기호를 둘째 줄로 내려 두 라벨 사이를 띄움. 손실 괄호는 라벨 아래,
#           손실 식은 v9 식 (3)과 같게 L_1 + λ(1−SSIM). 화살표 숫자 각주는 비어 있던 오른쪽 위로.
#   그림 2  (a) 체인을 오른쪽으로 옮겨 입력 라벨과 띄우고, stage 2 출력(reshape)은 상자 아래로.
#           (b) 앞 두 블록은 세로로 쌓고, 오른쪽 블록(스캔·S6·merge·출력)을 쌓은 블록과 위아래를 맞춰 빈 띠를 없앰.
#           S6 출력은 세로 모음선(bus)에 들어간 뒤 한 줄로 Ⓒ 에 들어간다(Ⓒ 위아래에 붙어 들어가지 않게).
# 아래첨자·위첨자(2026-10-17 검증 지적): matplotlib mathtext 는 첨자를 SHRINK_FACTOR(0.7) 배로 그려 6.5 pt 라벨의
#   첨자가 4.55 pt 가 된다. v9 그림은 '모든 글자 ≥ 6.5 pt' 규칙을 문자 그대로 지키도록 그리는 동안만 SHRINK_FACTOR 를
#   MIN_PT / FS9 = 1.0 으로 바꾼다(첨자 = 본 글자 크기, 위치만 올리고 내림). 384² 는 '384×384' 로 풀어 쓴다.
#   자체 점검은 Text 의 공칭 크기만이 아니라 mathtext 가 실제로 그리는 글리프 크기(첨자 포함)까지 잰다.
import contextlib

import matplotlib._mathtext as _mtx
import matplotlib.mathtext as _mtt

FS9 = 6.5                     # v9 기본 글자 크기(pt)
MIN_PT = 6.5                  # 자체 점검 기준: 첨자 포함 모든 글리프
SCRIPT_SHRINK_V9 = MIN_PT / FS9   # = 1.0 — 첨자도 6.5 pt
SCRIPT_SUB1_V9 = 0.6          # 아래첨자 내림(x-height 배, 기본 0.3) — 같은 크기 첨자가 본 글자와 한 단어처럼 붙어 보이지 않게
CLEAR_IN = 0.02               # 글자(잉크) ↔ 상자·선·영상 최소 간격(in). 상자 안 글자는 상자 테두리까지의 안쪽 간격
TEXT_GAP_IN = 0.06            # 같은 줄(세로 범위가 겹치는) 두 글자 사이 최소 가로 간격(in) — 두 라벨이 한 구절처럼 붙어 읽히지 않게
_UNI_SCRIPTS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")   # 유니코드 첨자 문자(공칭 크기만 6.5 pt 이고 실제 글자는 작음) → 금지


def _clear_text_caches():
    _mtt.MathTextParser._parse_cached.cache_clear()
    f = getattr(matplotlib.text, "_get_text_metrics_with_cache_impl", None)
    if f is not None:
        f.cache_clear()


@contextlib.contextmanager
def _v9_typesetting():
    """v9 그림을 그리는 동안만: 첨자 축소 해제(SHRINK_FACTOR = 1.0)·아래첨자 내림 0.6 x-height + text() 자동 축소 하한 6.5 pt.
    파싱 결과 캐시를 앞뒤로 비워 v8 판(같은 프로세스에서 그려도)에 영향이 없게 한다."""
    global _MIN_FS
    fc = _mtx.FontConstantsBase
    old = (_mtx.SHRINK_FACTOR, _MIN_FS, fc.sub1)
    _mtx.SHRINK_FACTOR, _MIN_FS, fc.sub1 = SCRIPT_SHRINK_V9, MIN_PT, SCRIPT_SUB1_V9
    _clear_text_caches()
    try:
        yield
    finally:
        _mtx.SHRINK_FACTOR, _MIN_FS, fc.sub1 = old
        _clear_text_caches()


def box9(ax, x, y, w, h, label, sub=None, fc=FILL, ec=EDGE, lw=LW_BOX, r=0.035, z=3, dy=0.056):
    """v9 블록: 한 줄 라벨(+ 선택 부라벨) 모두 6.5 pt. 두 줄일 때 중심 ±dy."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    yc = y + h / 2
    if sub is None:
        text(ax, x + w / 2, yc, label, fs=FS9, max_w=w - 0.03, z=z + 1)
    else:
        text(ax, x + w / 2, yc + dy, label, fs=FS9, max_w=w - 0.03, z=z + 1)
        text(ax, x + w / 2, yc - dy, sub, fs=FS9, color=INK2, max_w=w - 0.03, z=z + 1)
    return (x, y, w, h)


# ───────── self-check ─────────
def _math_glyphs(t):
    """mathtext 라벨이 실제로 그리는 글리프의 (크기 pt, 글꼴 파일 이름, 글리프 이름) 목록. 수식이 아니면 []."""
    s, ismath = t._preprocess_math(t.get_text())
    if not ismath:
        return []
    vp = _mtt.MathTextParser("path").parse(s, dpi=72, prop=t.get_fontproperties())
    out = []
    for g in vp.glyphs:
        font, size, num = g[0], g[1], g[2]
        gname = f"{chr(num)} U+{num:04X}" if isinstance(num, int) else str(num)
        out.append((float(size), os.path.basename(font.fname), gname))
    return out


def _ink_bboxes(fig, texts):
    """각 글자의 실제 잉크 영역(표시 좌표 px, 현재 fig.dpi). 그 글자 하나만 보이게 Agg 로 그려 불투명 픽셀(α > 25 %)의
    범위를 잰다 — mathtext 첨자·악센트·대체 글리프까지 저장 결과와 같은 래스터로 잰다."""
    ax = fig.axes[0]
    others = [a for a in fig.findobj() if a is not fig and a is not ax]
    vis = {a: a.get_visible() for a in others}
    out = []
    try:
        for a in others:
            a.set_visible(False)
        hpx = None
        for t in texts:
            t.set_visible(True)
            fig.canvas.draw()
            alpha = np.asarray(fig.canvas.buffer_rgba())[..., 3]
            hpx = alpha.shape[0]
            rows, cols = np.nonzero(alpha > 64)
            t.set_visible(False)
            if rows.size == 0:
                out.append(None)
                continue
            out.append(matplotlib.transforms.Bbox([[cols.min(), hpx - (rows.max() + 1)], [cols.max() + 1, hpx - rows.min()]]))
    finally:
        for a, v in vis.items():
            a.set_visible(v)
    return out


def _graphics(ax, r, dpi):
    """그림 요소(상자·원·화살표·선·영상)의 표시 좌표 범위(선 두께 절반만큼 넓힘)."""
    out = []
    for p in ax.patches:
        if not p.get_visible():
            continue
        bb = p.get_window_extent(renderer=r)
        pad = p.get_linewidth() * dpi / 72.0 / 2
        filled = bool(p.get_fill()) and p.get_facecolor()[3] > 0 and not isinstance(p, FancyArrowPatch)
        out.append(dict(bb=bb.padded(pad), z=p.get_zorder(), filled=filled,
                        kind=type(p).__name__, container=not isinstance(p, FancyArrowPatch)))
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        bb = ln.get_window_extent(renderer=r)
        out.append(dict(bb=bb.padded(ln.get_linewidth() * dpi / 72.0 / 2), z=ln.get_zorder(), filled=False,
                        kind="Line2D", container=False))
    for im in ax.images:
        out.append(dict(bb=im.get_window_extent(renderer=r), z=im.get_zorder(), filled=True, kind="image",
                        container=False))
    return out


def _inside(a, b, tol=0.0):
    return a.x0 >= b.x0 - tol and a.x1 <= b.x1 + tol and a.y0 >= b.y0 - tol and a.y1 <= b.y1 + tol


def _gap(a, b):
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0.0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0.0)
    return dx, dy


def audit_text(fig, name, enforce=True):
    """그린 뒤 자체 점검(위반 목록 반환; enforce=False 면 보고만).
      1. 보이는 모든 Text 의 공칭 크기 ≥ MIN_PT, 그리고 mathtext 가 실제로 그리는 글리프(첨자 포함) 크기 ≥ MIN_PT.
         유니코드 첨자 문자(²·₁ 등) 사용 금지.
      2. 글자 잉크가 그림 밖으로 잘리지 않음.
      3. 글자끼리 겹치지 않음, 같은 줄의 두 글자 사이 가로 간격 ≥ TEXT_GAP_IN.
      4. 글자 잉크 ↔ 상자·원·화살표·선·영상 간격 ≥ CLEAR_IN (상자 안 글자는 테두리까지 안쪽 간격 ≥ CLEAR_IN;
         글자가 놓인 불투명 상자 뒤로 지나가는 선은 가려지므로 제외).
    수식 글리프 중 Liberation Sans 가 아닌 글꼴(STIX 수학 기호 대체 글리프)도 목록으로 보고한다."""
    dpi0 = fig.dpi
    fig.set_dpi(DPI)                                              # 점검은 저장과 같은 600 dpi 래스터에서
    try:
        return _audit_text(fig, name, enforce)
    finally:
        fig.set_dpi(dpi0)


def _audit_text(fig, name, enforce):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    ax = fig.axes[0]
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible()]
    shown = [t for t in texts if t.get_text().strip()]
    bad = []
    # 1. sizes
    fs_nominal = min(t.get_fontsize() for t in texts)
    glyph_min, n_math, fallback = float("inf"), 0, set()
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-9:
            bad.append(f"font {t.get_fontsize():.2f} pt < {MIN_PT}: {t.get_text()!r}")
    for t in shown:
        if _UNI_SCRIPTS & set(t.get_text()):
            bad.append(f"unicode super/subscript character (renders small): {t.get_text()!r}")
        gl = _math_glyphs(t)
        if gl:
            n_math += 1
            m = min(g[0] for g in gl)
            glyph_min = min(glyph_min, m)
            if m < MIN_PT - 1e-6:
                bad.append(f"rendered math glyph {m:.2f} pt < {MIN_PT} (sub/superscript): {t.get_text()!r}")
            fallback |= {f"{g[2]} ({g[1]})" for g in gl if not g[1].startswith("LiberationSans")}
    # 2./3. clipping and text-text
    ink = [(t, b) for t, b in zip(shown, _ink_bboxes(fig, shown)) if b is not None]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    tol = 0.5                                                     # px
    for t, b in ink:
        if b.x0 < fb.x0 - tol or b.y0 < fb.y0 - tol or b.x1 > fb.x1 + tol or b.y1 > fb.y1 + tol:
            bad.append(f"clipped by figure edge: {t.get_text()!r} "
                       f"({b.x0 / dpi:.3f},{b.y0 / dpi:.3f})-({b.x1 / dpi:.3f},{b.y1 / dpi:.3f}) in")
    min_tt = (float("inf"), "")
    for i in range(len(ink)):
        for j in range(i + 1, len(ink)):
            (ta, a), (tb, b) = ink[i], ink[j]
            dx, dy = _gap(a, b)
            if dx == 0 and dy == 0:
                bad.append(f"text overlap: {ta.get_text()!r} × {tb.get_text()!r}")
            elif dy == 0:                                         # same line (vertical ranges overlap)
                g = dx / dpi
                if g < min_tt[0]:
                    min_tt = (g, f"{ta.get_text()!r} | {tb.get_text()!r}")
                if g < TEXT_GAP_IN - 1e-9:
                    bad.append(f"same-line text gap {g:.3f} in < {TEXT_GAP_IN}: {ta.get_text()!r} | {tb.get_text()!r}")
    # 4. text vs graphics
    gr = _graphics(ax, r, dpi)
    min_tg = (float("inf"), "")
    for t, b in ink:
        tz = t.get_zorder()
        holders = [g for g in gr if g["filled"] and g["container"] and g["z"] < tz and _inside(b, g["bb"])]
        for g in gr:
            gb = g["bb"]
            dx, dy = _gap(b, gb)
            if dx == 0 and dy == 0:
                if g["container"] and g["z"] < tz and _inside(b, gb):
                    c = min(b.x0 - gb.x0, gb.x1 - b.x1, b.y0 - gb.y0, gb.y1 - b.y1) / dpi
                    what = f"inside {g['kind']}"
                elif any(h["z"] > g["z"] for h in holders):
                    continue                                      # hidden behind the opaque box the text sits in
                else:
                    bad.append(f"text touches/crosses {g['kind']}: {t.get_text()!r}")
                    continue
            else:
                c = float(np.hypot(dx, dy)) / dpi
                what = f"to {g['kind']}"
            if c < min_tg[0]:
                min_tg = (c, f"{t.get_text()!r} {what}")
            if c < CLEAR_IN - 1e-9:
                bad.append(f"clearance {c:.3f} in < {CLEAR_IN} ({what}): {t.get_text()!r}")
    w_in, h_in = fig.get_size_inches()
    print(f"[audit] {name}: {len(texts)} visible Text ({len(shown)} non-empty, {n_math} with mathtext); "
          f"min nominal font {fs_nominal:.2f} pt; min rendered math glyph (incl. sub/superscripts) "
          f"{glyph_min:.2f} pt; size {w_in:.3f}×{h_in:.3f} in")
    print(f"[audit]   min text–graphic clearance {min_tg[0]:.3f} in ({min_tg[1]}); "
          f"min same-line text gap {min_tt[0]:.3f} in ({min_tt[1]})")
    if fallback:
        print(f"[audit]   math glyphs not in Liberation Sans (mathtext.fallback = stixsans): {', '.join(sorted(fallback))}")
    for m in bad:
        print(f"  ! {m}")
    return bad if enforce else []


def save9(fig, name):
    """save() 와 같고, 저장된 PNG 의 실제 픽셀 크기를 읽어 출력한다."""
    save(fig, name)
    try:
        from PIL import Image
        with Image.open(os.path.join(FIG_DIR, f"{name}.png")) as im:
            print(f"[audit]   {name}.png = {im.size[0]}×{im.size[1]} px @ {DPI} dpi "
                  f"({im.size[0] / DPI:.3f}×{im.size[1] / DPI:.3f} in)")
    except Exception as e:                                         # pragma: no cover
        print(f"[audit]   (PNG size not read: {e!r})")


# ───────── Fig. 1 (v9) ─────────
def fig1_v9(th, suffix="_v9"):
    with _v9_typesetting():
        return _fig1_v9(th, suffix)


def _fig1_v9(th, suffix):
    T, T2 = 0.32, 0.30            # ỹ thumbnail / zero-filled·GT·reconstruction thumbnails
    R_OP = 0.055                  # Ⓒ radius
    R_DOT = 0.064                 # ⊙ radius (STIX ⊙ glyph is nearly as large as the circle — keep ≥ 0.02 in to the arrows)
    # 세로 기준선(아래→위): 손실 식 → 괄호 → 아래 행 라벨 2줄 → 아래 행 영상(중심 zc) → 주 행(중심 yc) → ⊙ → y_c
    y_loss = 0.170                # loss 줄 (va=top)
    yl = 0.205                    # 손실 괄호 가로선
    zc = 0.735                    # 아래 행 영상 중심
    l1 = zc - T2 / 2 - 0.115      # 아래 행 라벨 1 줄 기준선(va=baseline)
    l2 = l1 - 0.150               # 2 줄 기준선
    yc = zc + 0.50                # 주 행 중심
    y1 = yc - T / 2
    yo = yc + 0.40                # ⊙ 중심
    yk = yo + R_DOT + 0.10        # y_c 썸네일 아래
    H = yk + T + 0.035
    fig, ax = canvas(H)
    vmax = float(th["gt"].max())

    # ── data preparation: y_c ⊙ M → ỹ (left column) ──
    xL, xc0 = 0.04, 0.04 + T / 2
    thumb(ax, xL, yk, T, th["ksp_full"])
    text(ax, 0.42, yk + T * 0.70, "fully-sampled k-space $y_c$", fs=FS9, ha="left")
    lab(ax, 0.42, yk + T * 0.28, "16 coils, 384×384 (crop/pad), Re/Im", ha="left", va="center", fs=FS9)
    arrow(ax, (xc0, yk), (xc0, yo + R_DOT))
    op(ax, xc0, yo, r"$\odot$", r=R_DOT, fs=FS9)
    ms = 0.25
    thumb(ax, 0.42, yo - ms / 2, ms, th["mask2d"], vmin=0, vmax=1)
    arrow(ax, (0.42, yo), (xc0 + R_DOT, yo))
    text(ax, 0.71, yo + 0.062, "mask $M$", fs=FS9, ha="left")
    lab(ax, 0.71, yo - 0.062, "R = 4 equispaced, 8 % ACS", ha="left", va="center", fs=FS9)
    arrow(ax, (xc0, yo - R_DOT), (xc0, y1 + T))
    # footnote in the free top-right corner
    lab(ax, W - 0.04, yk + T * 0.70, "numbers on arrows:", ha="right", va="center", fs=FS9)
    lab(ax, W - 0.04, yk + T * 0.28, "channels (384×384)", ha="right", va="center", fs=FS9)

    # ── main row ──
    thumb(ax, xL, y1, T, th["ksp_und"])
    text(ax, 0.03, y1 - 0.035, r"$\tilde{y}_c = M \odot y_c$", fs=FS9, ha="left", va="top")
    lab(ax, 0.03, y1 - 0.185, "undersampled", ha="left", va="top", fs=FS9)
    xb = 0.65                                       # branch point (right of the ỹ labels)
    dot(ax, xb, yc)
    arrow(ax, (xL + T, yc), (xb, yc), head=False)
    g_lab = 0.165                                   # arrow length that carries a 2-digit channel label
    sx, sw, sh = xb + g_lab, 0.92, 0.50
    sy = yc - sh / 2
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.035",
                                fc="white", ec=INK, lw=0.7, ls=(0, (2.2, 1.4)), zorder=3))
    text(ax, sx + sw / 2, sy + sh - 0.085, "sequence module $f_\\theta$", fs=FS9, max_w=sw - 0.02, z=4)
    text(ax, sx + sw / 2, sy + sh - 0.195, "(only part replaced)", fs=FS9, color=MUTED, max_w=sw - 0.02, z=4)
    bw, bh, bm = 0.35, 0.15, 0.04
    box9(ax, sx + bm, sy + 0.045, bw, bh, "bi-GRU", fc=FILL_R, ec=EDGE_R)
    text(ax, sx + sw / 2, sy + 0.045 + bh / 2, "or", fs=FS9, color=MUTED, z=4)
    box9(ax, sx + sw - bm - bw, sy + 0.045, bw, bh, "SS2D", fc=FILL_B, ec=EDGE_B)
    arrow(ax, (xb, yc), (sx, yc))
    lab(ax, (xb + sx) / 2, yc + 0.03, "32", fs=FS9)
    cx = sx + sw + g_lab + R_OP
    arrow(ax, (sx + sw, yc), (cx - R_OP, yc))
    lab(ax, (sx + sw + cx - R_OP) / 2, yc + 0.03, "20", fs=FS9)
    op(ax, cx, yc, "C", fs=FS9)
    ux, uw, uh = cx + R_OP + g_lab, 0.58, 0.40
    ax.add_patch(FancyBboxPatch((ux, yc - uh / 2), uw, uh, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=FILL, ec=EDGE, lw=LW_BOX, zorder=3))
    text(ax, ux + uw / 2, yc + 0.105, "U-Net $g_\\varphi$", fs=FS9, max_w=uw - 0.03, z=4)
    text(ax, ux + uw / 2, yc - 0.020, "31.1M, same", fs=FS9, color=INK2, max_w=uw - 0.03, z=4)
    text(ax, ux + uw / 2, yc - 0.125, "architecture", fs=FS9, color=INK2, max_w=uw - 0.03, z=4)
    arrow(ax, (cx + R_OP, yc), (ux, yc))
    lab(ax, (cx + R_OP + ux) / 2, yc + 0.03, "52", fs=FS9)

    # ── bottom row: zero-filled (left), ground truth · reconstruction (right) ──
    zx = 0.92
    polyline(ax, [(xb, yc), (xb, zc), (zx, zc)])
    lab(ax, (xb + zx) / 2 + 0.005, zc + 0.03, "$ℱ^{-1}$", fs=FS9, color=INK)
    thumb(ax, zx, zc - T2 / 2, T2, th["zf"], vmin=0, vmax=vmax)
    text(ax, zx + T2 / 2, l1, "zero-filled coil images", fs=FS9, va="baseline")
    polyline(ax, [(zx + T2, zc), (cx, zc), (cx, yc - R_OP)])
    lab(ax, (zx + T2 + cx) / 2, zc + 0.03, "32", fs=FS9)
    gxc = cx + 0.035 + T2 / 2                       # GT thumbnail just right of the concat line
    rxc = W - 0.04 - 0.29                           # reconstruction centre: its widest label ends at the right margin
    thumb(ax, gxc - T2 / 2, zc - T2 / 2, T2, th["gt"], vmin=0, vmax=vmax)
    text(ax, gxc, l1, "ground truth", fs=FS9, va="baseline")
    lab(ax, gxc, l2, "$x^{*}$, dataset RSS", va="baseline", fs=FS9)
    thumb(ax, rxc - T2 / 2, zc - T2 / 2, T2, th["recon"], vmin=0, vmax=vmax)
    text(ax, rxc, l1, "reconstruction", fs=FS9, va="baseline")
    lab(ax, rxc, l2, r"$\hat{x}$, magnitude", va="baseline", fs=FS9)
    polyline(ax, [(ux + uw, yc), (rxc, yc), (rxc, zc + T2 / 2)])          # U-Net output → x̂
    # loss bracket below the two labels
    yt = l2 - 0.065
    for xx in (gxc, rxc):
        ax.add_line(Line2D([xx, xx], [yt, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    ax.add_line(Line2D([gxc, rxc], [yl, yl], color=INK2, lw=LW_ARR, ls=(0, (1.2, 1.2)), zorder=4))
    text(ax, W - 0.04, y_loss, "loss: $L_1 + \\lambda(1-\\mathrm{SSIM})$, brain mask", fs=FS9, ha="right", va="top")
    bad = audit_text(fig, f"conf_fig1_pipeline{suffix}")
    save9(fig, f"conf_fig1_pipeline{suffix}")
    return bad


# ───────── Fig. 2 (v9) ─────────
def unrolled_birnn_v9(ax, x0, y0, n=4, cw=0.15, ch=0.14, gap=0.07, fc=FILL_R, ec=EDGE_R, ell_after=2):
    """unrolled_birnn 과 같은 그림, 글자 6.5 pt 와 그에 맞춘 세로 간격. (끝 x, 출력 y 라벨 위 끝) 반환."""
    xs = [x0 + i * (cw + gap) for i in range(n)]
    yb = y0 + 0.165
    yf = yb + ch + 0.04
    idx = [str(i + 1) for i in range(n - 1)] + ["L"]
    for i, x in enumerate(xs):
        xc = x + cw / 2
        if i == ell_after:
            text(ax, xc, yf + ch / 2, "…", fs=7, color=INK2)
            text(ax, xc, yb + ch / 2, "…", fs=7, color=INK2)
            continue
        ax.add_line(Line2D([xc, xc], [y0 + 0.125, yf + ch + 0.035], color=INK, lw=LW_ARR, zorder=2))
        for yy in (yb, yf):
            arrow(ax, (xc, yy - 0.03), (xc, yy), mutation=4, z=2)
        arrow(ax, (xc, yf + ch), (xc, yf + ch + 0.045), mutation=4, z=2)
        for yy, s in ((yf, r"$\vec{h}$"), (yb, r"$\overleftarrow{h}$")):
            box(ax, x, yy, cw, ch, "", fc=fc, ec=ec, fs=FS9, r=0.02)
            text(ax, xc, yy + ch / 2 - 0.012, s, fs=FS9, z=4)     # 화살표 악센트가 위로 올라가므로 조금 내려 상자 안 간격을 맞춤
        text(ax, xc, y0, f"$x_{{{idx[i]}}}$", fs=FS9, va="bottom")
        text(ax, xc, yf + ch + 0.065, f"$y_{{{idx[i]}}}$", fs=FS9, va="bottom")
    for i in range(n - 1):
        xa, xb = xs[i] + cw, xs[i + 1]
        arrow(ax, (xa, yf + ch / 2), (xb, yf + ch / 2), mutation=4)
        arrow(ax, (xb, yb + ch / 2), (xa, yb + ch / 2), mutation=4)
    return xs[-1] + cw, (yb + yf + ch) / 2, yf + ch + 0.065 + 0.125


def fig2_v9(th, variant="rows", suffix="_v9"):
    with _v9_typesetting():
        return _fig2_v9(th, variant, suffix)


def _fig2_v9(th, variant, suffix):
    """fig2 와 같은 내용(두 판), 글자 ≥ 6.5 pt 재배치."""
    impl = variant == "impl"
    # ── (b) vertical plan (bottom → top): 2 label lines under the grids → 4 grids (centre ym) → S6 label = top line ──
    #    left column: k-space label (top line) → LN·Linear·SiLU → DWConv (its output bends up to ym)
    gs, gg = 0.21, 0.05
    gy0 = 0.025 + 0.11 + 0.12 + 0.03           # bottom of the lower grids ("4-way scan" + scan-direction lines below)
    ym = gy0 + gs + gg / 2                     # (b) main line = centre of the 2×2 grids
    gtop = gy0 + 2 * gs + gg
    t0 = gtop + 0.13                           # top of the (b) content (k-space / S6 ×4 labels)
    ptop_b = t0 + 0.15                         # (b) title top
    # ── (a) ──
    y0 = ptop_b + 0.15                         # (a) chain baseline (x_t labels bottom)
    yb = y0 + 0.165
    ch = 0.155
    yf = yb + ch + 0.04
    y_mid = (yb + yf + ch) / 2
    y_lab_top = yf + ch + 0.065 + 0.105
    y_stage = y_lab_top + 0.015                # stage-1 label (va=bottom)
    H = y_stage + 0.105 + 0.02 + 0.12         # + stage label + gap + title line
    fig, ax = canvas(H)

    # ─────────── (a) bi-GRU module ───────────
    text(ax, 0.04, H - 0.035, "(a)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, H - 0.035, "bi-GRU module (ETER-net)", fs=FS9, ha="left", va="top")
    text(ax, W - 0.04, H - 0.035, "637.1M (model 668.2M)", fs=FS9, color=MUTED, ha="right", va="top")
    ts, tx = 0.34, 0.06
    ty = y_mid - ts / 2
    if impl:
        for k in (2, 1):
            ax.add_patch(Rectangle((tx + 0.025 * k, ty + 0.025 * k), ts, ts, fc="white", ec=EDGE, lw=0.4, zorder=2))
    thumb(ax, tx, ty, ts, th["ksp_und"])
    if impl:
        bh_ = ts / 12
        ax.add_patch(Rectangle((tx, ty + ts * 0.5 - bh_ / 2), ts, bh_, fc=EDGE_R, ec=EDGE_R, lw=0.6,
                               alpha=0.45, zorder=5))
        lab(ax, 0.04, ty - 0.035, "$x_t$: 32 rows × 384", ha="left", va="top", fs=FS9, color=INK)
        lab(ax, 0.04, ty - 0.170, "of one channel", ha="left", va="top", fs=FS9)
    else:
        ax.add_line(Line2D([tx, tx + ts], [ty + ts * 0.5] * 2, color=EDGE_R, lw=0.9, zorder=5))
        lab(ax, 0.04, ty - 0.035, "$x_t$ = k-space row $t$", ha="left", va="top", fs=FS9, color=INK)
        lab(ax, 0.04, ty - 0.170, "12,288-d (32×384)", ha="left", va="top", fs=FS9)
    x0c = 0.82
    arrow(ax, (tx + ts + 0.01 + (0.05 if impl else 0), y_mid), (x0c - 0.02, y_mid))
    x_end, _, _ = unrolled_birnn_v9(ax, x0c, y0, ch=ch)
    text(ax, (x0c + x_end) / 2, y_stage, "stage 1 — 384 steps (12,288-d each)" if impl else "stage 1: rows — 384 steps",
         fs=FS9, color=INK2, va="bottom")
    bx, bw2, bh2 = 2.13, 0.76, 0.26
    arrow(ax, (x_end + 0.01, y_mid), (bx, y_mid))
    lab(ax, (x_end + bx) / 2, y_mid + 0.045, "transpose", fs=FS9)
    box9(ax, bx, y_mid - bh2 / 2, bw2, bh2, "stage 2" if impl else "stage 2: columns", sub="384 steps",
         fc=FILL_R, ec=EDGE_R)
    ox = bx + bw2 / 2
    arrow(ax, (ox, y_mid - bh2 / 2), (ox, y0 + 0.02))
    lab(ax, ox - 0.045, (y_mid - bh2 / 2 + y0 + 0.02) / 2, "reshape", fs=FS9, ha="right", va="center")
    text(ax, ox, y0 - 0.005, "20×384×384", fs=FS9, va="top")

    # ─────────── (b) SS2D module ───────────
    text(ax, 0.04, ptop_b, "(b)", fs=FS_P, weight="bold", ha="left", va="top")
    text(ax, 0.28, ptop_b, "SS2D module", fs=FS9, ha="left", va="top")
    text(ax, W - 0.04, ptop_b, "0.12M (model 31.2M)", fs=FS9, color=MUTED, ha="right", va="top")
    bh, lw_, bx0 = 0.25, 0.68, 0.04
    yA = t0 - 0.12 - bh                        # LN·Linear·SiLU box bottom (input label above it, top at t0)
    lab(ax, bx0, yA + bh + 0.03, "k-space 32×384×384", ha="left", fs=FS9)
    box9(ax, bx0, yA, lw_, bh, "LN·Linear·SiLU", sub="32 → 128")
    yD = yA - 0.07 - bh                        # DWConv box bottom (stacked below)
    arrow(ax, (bx0 + lw_ / 2, yA), (bx0 + lw_ / 2, yD + bh))
    box9(ax, bx0, yD, lw_, bh, "DWConv 3×3", sub="SiLU, 128")
    gx0 = 0.96                                 # grids right of the DWConv box (their labels below clear its right edge)
    for d, i, j in (("r", 0, 1), ("l", 1, 1), ("d", 0, 0), ("u", 1, 0)):
        scan_grid(ax, gx0 + i * (gs + gg), gy0 + j * (gs + gg), gs, d)
    gxc = gx0 + gs + gg / 2
    gright = gx0 + 2 * gs + gg
    text(ax, gxc, gy0 - 0.03, "4-way scan", fs=FS9, va="top")
    lab(ax, gxc, gy0 - 0.14, "rows →←, columns ↓↑", va="top", fs=FS9)
    xk = bx0 + lw_ + 0.08                      # DWConv output bends up to the grid centre line
    polyline(ax, [(bx0 + lw_, yD + bh / 2), (xk, yD + bh / 2), (xk, ym), (gx0 - 0.005, ym)])
    s6x, s6w, s6h, pitch = gright + 0.13, 0.25, 0.115, 0.13
    ys6 = [ym + (k - 1.5) * pitch - s6h / 2 for k in range(4)]   # S6 stack centred on ym
    for yy in ys6:
        box9(ax, s6x, yy, s6w, s6h, "S6", fc=FILL_B, ec=EDGE_B, r=0.02)
        arrow(ax, (gright + 0.005, yy + s6h / 2), (s6x, yy + s6h / 2), mutation=4)
    text(ax, s6x + s6w / 2, t0, "S6 ×4 (parallel)", fs=FS9, va="top")
    mx = s6x + s6w + 0.045                     # vertical merge bus: the four outputs join here, then one line into Ⓒ
    for yy in ys6:
        ax.add_line(Line2D([s6x + s6w, mx], [yy + s6h / 2] * 2, color=INK, lw=LW_ARR, zorder=4,
                           solid_capstyle="projecting"))
    ax.add_line(Line2D([mx, mx], [ys6[0] + s6h / 2, ys6[-1] + s6h / 2], color=INK, lw=LW_ARR, zorder=4,
                       solid_capstyle="projecting"))
    ccx = mx + 0.045 + 0.055                   # Ⓒ right of the bus
    ax.add_line(Line2D([mx, ccx - 0.055], [ym, ym], color=INK, lw=LW_ARR, zorder=4))
    op(ax, ccx, ym, "C", fs=FS9)
    lx, lw2 = ccx + 0.055 + 0.06, 0.50
    box9(ax, lx, ym - bh / 2, lw2, bh, "LN·Linear", sub="512 → 128")
    arrow(ax, (ccx + 0.055, ym), (lx, ym))
    cx2, cw2 = lx + lw2 + 0.06, 0.46
    arrow(ax, (lx + lw2, ym), (cx2, ym))
    box9(ax, cx2, ym - bh / 2, cw2, bh, "1×1 conv", sub="128 → 20")
    lab(ax, cx2 + cw2 / 2, ym - bh / 2 - 0.03, "20×384×384", va="top", fs=FS9, color=INK)
    name = ("conf_fig2_modules_impl" if impl else "conf_fig2_modules") + suffix
    bad = audit_text(fig, name)
    save9(fig, name)
    return bad


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    ap.add_argument("--suffix", default="", help="출력 이름 접미사(예: _v9). 주면 기본 레이아웃 = v9")
    ap.add_argument("--npz", default=None, help="zero-filled·복원(rec_ss2d)·GT 썸네일 npz (기본 = v8 recon_4689.npz)")
    ap.add_argument("--layout", choices=("v8", "v9"), default=None,
                    help="v8 = 기존 그림(글자 4.7~6.5 pt, 고정 이름 — --suffix/--npz 불가), "
                         "v9 = 모든 글자(첨자 포함) ≥ 6.5 pt 재배치판. 기본: --suffix 있으면 v9")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    layout = args.layout or ("v9" if args.suffix else "v8")
    if layout == "v8" and (args.suffix or args.npz):
        # v8 판은 출력 이름이 고정(conf_fig1_pipeline 등)이라 --suffix 를 쓸 수 없고, --npz 를 주면 커밋된 v8 그림의
        # 썸네일이 바뀐 채로 덮어써진다 → 둘 다 거부.
        sys.exit("error: v8 layout writes the fixed v8 file names and ignores --suffix; refusing --suffix/--npz with "
                 "--layout v8 so the committed v8 figures are not overwritten (use --suffix _v9 for the v9 layout)")
    if layout == "v9" and not args.suffix:
        sys.exit("error: v9 layout needs --suffix (so the v8 figure files are not overwritten)")
    npz = args.npz if args.npz else NPZ
    if not os.path.isabs(npz):
        npz = os.path.join(ROOT, npz) if not os.path.exists(npz) else os.path.abspath(npz)
    th = load_thumbs(npz)
    print(f"layout {layout}, thumbnails zf/recon(rec_ss2d)/gt from {os.path.relpath(npz, ROOT)}")
    if layout == "v8":
        fig1(th)
        fig2(th)
        fig2(th, variant="impl")
        fig3(th)
        if _OVERFLOW:
            for s, w_in, mw in _OVERFLOW:
                print(f"  ! overflow ({w_in:.2f} > {mw:.2f} in): {s!r}")
        else:
            print("no text overflow")
    else:
        bad = []
        bad += fig1_v9(th, args.suffix)
        bad += fig2_v9(th, "rows", args.suffix)
        bad += fig2_v9(th, "impl", args.suffix)
        for s, w_in, mw in _OVERFLOW:
            bad.append(f"overflow ({w_in:.2f} > {mw:.2f} in): {s!r}")
        if bad:
            print(f"FAIL: {len(bad)} self-check violation(s)")
            for m in bad:
                print(f"  ! {m}")
            sys.exit(1)
        print(f"self-check OK: every Text and every rendered math glyph (incl. sub/superscripts) >= {MIN_PT} pt; "
              f"no clipping; no text overlap; same-line text gaps >= {TEXT_GAP_IN} in; "
              f"text-graphic clearance >= {CLEAR_IN} in; no box overflow")
