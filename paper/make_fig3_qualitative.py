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
    # 10-07: 2회차(시드 1) 모델과 U-Net 단독(시드 1) — --src 로 그 npz 폴더를 줄 때만 사용(기본 8열에는 없음)
    ("unetonly", "U-Net only", ""),
    ("gru2",     "ETER-net", "(bi-GRU)"),
    ("ss2d2",    "SS2D", "(controlled)"),
]
DEFAULT_KEYS = ["gt", "zf", "unet", "varnet", "promptmr", "gru", "ss2d", "v9"]   # --methods 미지정 시(기존 8열 그대로)
NARROW_HDR = {"gt": ("Ground", "truth"), "zf": ("Zero-", "filled"), "unet": ("U-Net†", ""),
              "varnet": ("E2E-", "VarNet†"), "promptmr": ("Prompt", "MR+"), "unetonly": ("U-Net", "only")}
ERR_VMAX = 0.10


def load_slice(idx, src=None):
    src = src or SRC
    z = np.load(os.path.join(src, f"recon_{idx:04d}.npz"))
    meta = json.load(open(os.path.join(src, f"metrics_{idx:04d}.json")))
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
    ap.add_argument("--src", default=SRC, help="recon_<idx>.npz / metrics_<idx>.json 폴더 (기본 results/vis/multimodel_compare)")
    # 10-17: 색 막대 눈금 글자 크기(기본 = 이전과 같은 FS['cbar_tick'] 6.0 pt → 이전 빌드 그대로). 6.5 pt 규칙판은 --cbar-tick-pt 6.5
    ap.add_argument("--cbar-tick-pt", type=float, default=None, help="colourbar tick label size in pt (default: 6.0, unchanged)")
    # 10-17 (v8fix 판 검수): 아래 세 옵션은 모두 기본값이면 이전 빌드와 바이트 단위로 같다.
    #   --dpi N        : 저장 dpi 를 N 으로 고정(기본 = 384/패널 폭 → 슬라이스 1 px = 이미지 1 px). 단 폭 4열에서 1:1 은 648 dpi 라
    #                    "3.15 in @ 600 dpi = 1890 px" 규칙을 맞추려면 --dpi 600 (패널 384 → 약 356 px 로 리샘플).
    #   --interp NAME  : imshow 보간(기본 = 1:1 일 때 nearest, --dpi 를 주면 antialiased — 축소 시 nearest 는 행·열을 건너뛰어 줄무늬가 생김)
    #   --liberation-cbar-label : 색 막대 라벨을 Liberation Sans 만으로 — mathtext 글꼴을 custom(Liberation Sans/Italic, fallback 없음)으로
    #                    두고 라벨을 '|$\hat{x}$ − $x$*| / max($x$*)' 로 쓴다('*' 는 수식 밖 본 크기 6.5 pt). 기본 mathtext 라벨은
    #                    DejaVu Sans 로 그려지고 윗첨자 '*' 가 0.7×6.5 = 4.55 pt 가 된다. (결합 곡절 U+0302 평문은 matplotlib 가
    #                    음의 left bearing 을 반영하지 않아 곡절이 x 오른쪽으로 밀려 쓰지 않음.)
    #   --header KEY=LINE1|LINE2 (반복 가능): 열 머리글 두 줄을 직접 지정(단 폭 NARROW_HDR 적용 뒤에 덮어씀). 예) --header 'gru=ETER-net|'
    ap.add_argument("--dpi", type=int, default=None, help="force output dpi (default: 384 / panel width, 1:1 pixels)")
    ap.add_argument("--interp", default=None, help="imshow interpolation (default: nearest at 1:1, antialiased with --dpi)")
    ap.add_argument("--liberation-cbar-label", action="store_true",
                    help="colourbar label in Liberation Sans only (custom mathtext fontset, no superscript)")
    ap.add_argument("--header", action="append", default=[], metavar="KEY=LINE1|LINE2",
                    help="override a column header (two lines separated by '|'); may be repeated")
    args = ap.parse_args()
    if args.cbar_tick_pt is not None:
        FS["cbar_tick"] = args.cbar_tick_pt
    if args.liberation_cbar_label:   # 이 그림의 mathtext 는 색 막대 라벨뿐 — 그 글꼴을 Liberation Sans 로, 없는 글리프는 오류(대체 글꼴 금지)
        rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
                         "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans:italic",
                         "mathtext.fallback": None})
    interp = args.interp or ("antialiased" if args.dpi else "nearest")
    hdr_over = {}
    for h in args.header:
        k, _, v = h.partition("=")
        l1, _, l2 = v.partition("|")
        hdr_over[k.strip()] = (l1, l2)
    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    os.makedirs(OUT, exist_ok=True)
    cols = ([c for c in COLS if c[0] in DEFAULT_KEYS] if not args.methods
            else [c for k in args.methods.split(",") for c in COLS if c[0] == k.strip()])
    if args.fig_w < 4.0:   # 단 폭: 한 줄 헤더가 패널(≈0.62 in)보다 길어지므로 두 줄로 나눈다
        cols = [(k, *NARROW_HDR.get(k, (h1, h2))) for k, h1, h2 in cols]
    cols = [(k, *hdr_over.get(k, (h1, h2))) for k, h1, h2 in cols]

    n_rows = 2 * len(slices) + (1 if args.gain_slice >= 0 else 0)
    n_cols = len(cols)
    L_IN, R_IN = 0.30, 0.42              # row-label margin / colourbar tick+label margin (inches, same at every width; 10-07 0.38→0.42)
    CB = 0.10                            # colourbar column = 0.10 panel width (inside the grid)
    W = args.panel_in if args.panel_in else (args.fig_w - L_IN - R_IN) / (n_cols + CB)   # 0.749 in at page width (8 cols)
    dpi = args.dpi if args.dpi else int(round(384 / W))   # default: one slice pixel = one image pixel (512 dpi at page width)
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
        gt, bm, rec, meta, met = load_slice(idx, args.src)
        gmax = max(float(gt.max()), 1e-8)
        mb = bm > 0.5
        for ci, (key, h1, h2) in enumerate(cols):
            ax0 = fig.add_subplot(gs[r, ci]); ax1 = fig.add_subplot(gs[r + 1, ci])
            for ax in (ax0, ax1):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if key == "gt":
                ax0.imshow(gt / gmax, cmap="gray", vmin=0, vmax=1, interpolation=interp)
                ax1.imshow(np.where(mb, 0.0, np.nan), cmap=hot, vmin=0, vmax=ERR_VMAX, interpolation=interp)
                ax1.text(0.5, 0.5, "|error|\n(brain mask)", transform=ax1.transAxes, fontsize=FS["errlabel"],
                         color="white", ha="center", va="center")
                # row labels (contrast) on the left
                ax0.text(-0.06, 0.5, f"{meta['contrast']}", transform=ax0.transAxes,
                         rotation=90, fontsize=FS["rowlabel"], ha="right", va="center")
            elif key in rec:
                x = rec[key] / gmax
                ax0.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation=interp)
                im_err = ax1.imshow(np.where(mb, np.abs(x - gt / gmax), np.nan), cmap=hot,
                                    vmin=0, vmax=ERR_VMAX, interpolation=interp)
                show_metrics(ax0, met.get(key))
            else:
                ax0.text(0.5, 0.5, "n/a", transform=ax0.transAxes, ha="center", va="center", fontsize=7)
            if r == 0:
                hdr = f"({letters[ci]}) {h1}" + (f"\n{h2}" if h2 else "\n")
                ax0.set_title(hdr, fontsize=FS["header"], pad=2.5, linespacing=1.05)
        r += 2

    if args.gain_slice >= 0:
        gt, bm, rec, meta, met = load_slice(args.gain_slice, args.src)
        gmax = max(float(gt.max()), 1e-8)
        for ci, (key, h1, h2) in enumerate(cols):
            ax = fig.add_subplot(gs[r, ci])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            src = gt if key == "gt" else rec.get(key)
            if src is None:
                continue
            ax.imshow(np.clip(args.gain * src / gmax, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation=interp)
            if key == "gt":
                ax.text(-0.06, 0.5, f"×{args.gain:g} gain",
                        transform=ax.transAxes, rotation=90, fontsize=FS["rowlabel"], ha="right", va="center")

    if im_err is not None:
        cax = fig.add_subplot(gs[1:2 * len(slices), n_cols])
        cb = fig.colorbar(im_err, cax=cax, ticks=[0, 0.05, 0.10])
        cb.ax.tick_params(labelsize=FS["cbar_tick"], length=2, width=0.5, pad=1.5)
        if args.liberation_cbar_label:   # 10-17: Liberation Sans 만, '*' 는 수식 밖 본 크기(윗첨자 축소 없음)
            cb.set_label(r"|$\hat{x}$ − $x$*| / max($x$*)", fontsize=FS["cbar_label"], labelpad=4.0)
        else:
            cb.set_label(r"|$\hat{x}$ − $x^{*}$| / max($x^{*}$)", fontsize=FS["cbar_label"], labelpad=4.0)   # 10-07: 1.5→4.0 (윗첨자가 눈금 숫자와 겹침)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{args.out_name}.{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"saved paper/figs/{args.out_name}.png/.pdf  ({fig_w:.2f}×{fig_h:.2f} in @ {dpi} dpi, {n_rows}×{n_cols} panels; slices {slices}, gain row {args.gain_slice})")


if __name__ == "__main__":
    main()
