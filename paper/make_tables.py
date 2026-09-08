"""논문 표 자동 생성기 — per-slice CSV 단일 원천에서 Table 1·2·3·S1 을 재현 가능하게 뽑는다.

입력:  results/eval/v9_unleashed/per_slice_paired_v9.csv (7,334 슬라이스 × 3모델 × 4지표)
출력:  paper/tables/table{1,2,2b,3,S1}.{md,tex}            — draft_ko_v2 표기(역사 유지)
       paper/tables/tableC{1_main_volume,1_main_slice,2_paired,3_efficiency,S1_contrast}.{md,tex}
       paper/tables/ieie_table1_block.md                     — IEIE .src.md 에 붙여넣는 @table 블록
       paper/tables/tableC4_reference.{md,tex} · ieie_table_ref_block.md — 공개 모델 참고선(전체 val 우리 프로토콜, 09-08; 미완주 방법 [TBD])
       - .md  : 한국어 헤더 (현행 draft_ko_v2 와 동일 표기) / C 계열은 관례형 영문
       - .tex : 영문 헤더 + booktabs
       관례형(C 계열, 2026-09-03): 표 내용 영문·L1 제외·nMSE %·볼륨 단위 mean±SD·Params (M)·Zero-filled 행·
       최고값 굵게/차선 밑줄·↑↓ — 근거: 최근 Mamba 재구성 논문·Oh 2025·fastMRI 공식 관례 (docs/paper_table_conventions.md)

통계 (draft_ko_v2 §3.8 과 동일 정의):
  - Δ 부호 규약: 항상 양수 = 치환/강화 모델 우위 (NMSE/L1 은 부호 반전)
  - 우위 슬라이스 비율 + 볼륨 클러스터 부트스트랩 95% CI (2,000회, seed 고정 = 초안 수치 재현)
  - Δ 중앙값 (IQR), 볼륨 단위 우위 비율, 볼륨 단위 Wilcoxon signed-rank (n=464)
자가 검증: 산출값이 draft_ko_v2 의 대표 수치(0.9140 / 78.2% / CI [76.8,79.7])와 일치해야 통과.
"""
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results/eval/v9_unleashed/per_slice_paired_v9.csv")
OUT = os.path.join(ROOT, "paper/tables")
os.makedirs(OUT, exist_ok=True)

METRICS = ["ssim", "psnr", "nmse", "l1"]
LOWER = {"nmse", "l1"}
BOOT_N, BOOT_SEED = 2000, 0

# CSV 에 없는 상수 (config/문서 ground-truth)
INFO = {
    "gru":  {"ko": "GRU",           "en": "GRU",              "ep": "50/50", "params": "668M"},
    "ss2d": {"ko": "SS2D (통제판)", "en": "SS2D (controlled)", "ep": "48/50", "params": "31M"},
    "v9":   {"ko": "강화 SS2D",     "en": "Enhanced SS2D",     "ep": "78/80", "params": "~33M"},
}

# ---------------------------------------------------------------- 데이터 적재
rows = list(csv.DictReader(open(CSV)))
n = len(rows)
files = np.array([r["file"] for r in rows])
contrast = np.array([f.split("_")[2] for f in files])
M = {f"{p}_{m}": np.array([float(r[f"{p}_{m}"]) for r in rows])
     for p in INFO for m in METRICS}
uf = np.unique(files)
V = len(uf)
vol_idx = [np.where(files == f)[0] for f in uf]


def delta(base, new, m):
    d = M[f"{base}_{m}"] - M[f"{new}_{m}"] if m in LOWER else M[f"{new}_{m}"] - M[f"{base}_{m}"]
    return d


def paired_stats(base, new, rng):
    """비교당 4지표 — rng 소비 순서를 고정해 초안 CI 를 그대로 재현."""
    out = {}
    for m in METRICS:
        d = delta(base, new, m)
        vols = [d[ix] for ix in vol_idx]
        wins, means = [], []
        for _ in range(BOOT_N):
            pick = rng.integers(0, V, V)
            s = np.concatenate([vols[i] for i in pick])
            wins.append(100 * np.mean(s > 0))
            means.append(s.mean())
        dv = np.array([x.mean() for x in vols])
        out[m] = dict(
            win=100 * np.mean(d > 0),
            win_ci=np.percentile(wins, [2.5, 97.5]),
            mean=d.mean(), mean_ci=np.percentile(means, [2.5, 97.5]),
            med=np.median(d), iqr=np.percentile(d, [25, 75]),
            vol_win=100 * np.mean(dv > 0), p_vol=wilcoxon(dv).pvalue,
            p_slice=wilcoxon(d).pvalue,
        )
    return out


rng = np.random.default_rng(BOOT_SEED)          # 초안과 동일: v8 비교 먼저, 그다음 v9 비교
S_V8 = paired_stats("gru", "ss2d", rng)
S_V9 = paired_stats("ss2d", "v9", rng)

# ---------------------------------------------------------------- 포맷터
F_MEAN = {"ssim": "{:.4f}", "psnr": "{:.2f}", "nmse": "{:.5f}", "l1": "{:.3f}"}
NAME_KO = {"ssim": "SSIM", "psnr": "PSNR (dB)", "nmse": "NMSE", "l1": "L1 (×10⁻⁶)"}
NAME_EN = {"ssim": "SSIM", "psnr": "PSNR (dB)", "nmse": "NMSE",
           "l1": r"L1 ($\times 10^{-6}$)"}


def f_mean(p, m):
    return F_MEAN[m].format(M[f"{p}_{m}"].mean())


def f_ms(p, m):
    return f"{M[f'{p}_{m}'].mean():.4f}±{M[f'{p}_{m}'].std():.4f}" if m == "ssim" else \
           f"{M[f'{p}_{m}'].mean():.2f}±{M[f'{p}_{m}'].std():.2f}" if m == "psnr" else \
           f"{M[f'{p}_{m}'].mean():.5f}±{M[f'{p}_{m}'].std():.5f}" if m == "nmse" else \
           f"{M[f'{p}_{m}'].mean():.2f}±{M[f'{p}_{m}'].std():.2f}"


def f_delta(m, v):
    if m == "ssim":
        return f"{v:+.4f}"
    if m == "psnr":
        return f"{v:+.2f}"
    if m == "nmse":
        return f"{v * 1e5:+.1f}"       # ×10⁻⁵ 단위 열
    return f"{v:+.3f}"


def f_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def d_head(m, ko=True):
    if m == "nmse":
        return "Δ (×10⁻⁵)" if ko else r"$\Delta$ ($\times 10^{-5}$)"
    return "Δ" if ko else r"$\Delta$"


def tex_escape(s):
    return (s.replace("×10⁻⁶", r"$\times 10^{-6}$").replace("×10⁻⁵", r"$\times 10^{-5}$")
             .replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("%", r"\%")
             .replace("~", r"$\sim$"))


def write_pair(name, md_lines, tex_lines):
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md_lines) + "\n")
    open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(tex_lines) + "\n")
    print(f"  wrote {name}.md / .tex")


def tex_table(caption, label, colspec, header, body_rows):
    L = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    L += [r + r" \\" for r in body_rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


# ---------------------------------------------------------------- Table 1 (v8 best)
md = ["**Table 1. Best checkpoint 기준 (val 전체 7,334 슬라이스, brain-masked, 슬라이스 단위 평균).**", "",
      "| | best epoch | SSIM | PSNR (dB) | NMSE | L1 (×10⁻⁶) | params |",
      "|---|---:|---:|---:|---:|---:|---:|"]
for p, bold in [("ss2d", True), ("gru", False)]:
    c = [f_mean(p, m) for m in METRICS]
    if bold:
        c = [f"**{x}**" for x in c]
        row = f"| **{INFO[p]['ko']}** | {INFO[p]['ep']} | " + " | ".join(c) + f" | **{INFO[p]['params']}** |"
    else:
        row = f"| {INFO[p]['ko']} | {INFO[p]['ep']} | " + " | ".join(c) + f" | {INFO[p]['params']} |"
    md.append(row)
tex_rows = []
for p in ["ss2d", "gru"]:
    b = r"\textbf" if p == "ss2d" else None
    cells = [INFO[p]["en"], INFO[p]["ep"]] + [f_mean(p, m) for m in METRICS] + [tex_escape(INFO[p]["params"])]
    if b:
        cells = [rf"\textbf{{{c}}}" for c in cells]
    tex_rows.append(" & ".join(cells))
tex = tex_table("Best-checkpoint results on the full validation set (7{,}334 slices, brain-masked, per-slice means).",
                "tab:best", "lrrrrrr",
                " & best epoch & SSIM & PSNR (dB) & NMSE & " + NAME_EN["l1"] + " & params", tex_rows)
write_pair("table1_best", md, tex)


# ---------------------------------------------------------------- Table 2 / 2b (paired)
def paired_table(name, base, new, S, cap_ko, cap_en, label):
    md = [cap_ko, "",
          f"| 지표 | {INFO[base]['ko']} mean±std | {INFO[new]['ko']} mean±std | Δ 중앙값 (IQR) | 우위 슬라이스 [95% CI] | 우위 볼륨 | p(볼륨, n={V}) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    tex_rows = []
    for m in METRICS:
        s = S[m]
        head = NAME_KO[m] + (" — Δ×10⁻⁵" if m == "nmse" else "")
        med = f"{f_delta(m, s['med'])} ({f_delta(m, s['iqr'][0])}, {f_delta(m, s['iqr'][1])})"
        win = f"**{s['win']:.1f}%** [{s['win_ci'][0]:.1f}, {s['win_ci'][1]:.1f}]"
        md.append(f"| {head} | {f_ms(base, m)} | {f_ms(new, m)} | {med} | {win} | **{s['vol_win']:.1f}%** | {f_p(s['p_vol'])} |")
        tex_rows.append(" & ".join([
            NAME_EN[m] + (r" --- $\Delta\times 10^{-5}$" if m == "nmse" else ""),
            tex_escape(f_ms(base, m)), tex_escape(f_ms(new, m)), tex_escape(med),
            rf"\textbf{{{s['win']:.1f}\%}} [{s['win_ci'][0]:.1f}, {s['win_ci'][1]:.1f}]",
            rf"\textbf{{{s['vol_win']:.1f}\%}}", f_p(s["p_vol"]).replace("<", r"$<$"),
        ]))
    md += ["", "(Δ 는 항상 양수 = " + INFO[new]["ko"] + " 우위 방향, NMSE/L1 부호 반전. CI 는 볼륨 클러스터 부트스트랩 "
           f"{BOOT_N:,}회. p 는 볼륨 단위 Wilcoxon signed-rank 양측.)"]
    tex = tex_table(cap_en, label, "lrrrrrr",
                    "Metric & " + INFO[base]["en"] + r" (mean$\pm$std) & " + INFO[new]["en"]
                    + r" (mean$\pm$std) & $\Delta$ median (IQR) & Slices favoring [95\% CI] & Volumes favoring & $p$ (volume, $n{=}" + str(V) + "$)",
                    tex_rows)
    write_pair(name, md, tex)


paired_table("table2_paired_v8", "gru", "ss2d", S_V8,
             "**Table 2. Per-slice paired 비교 (SS2D vs GRU).**",
             r"Per-slice paired comparison, SS2D vs.\ GRU. $\Delta$ is oriented so that positive favors SS2D "
             r"(signs of NMSE/L1 are negated). CIs: volume-cluster bootstrap (2{,}000 resamples); "
             r"$p$: two-sided volume-level Wilcoxon signed-rank.",
             "tab:paired-v8")
paired_table("table2b_paired_v9", "ss2d", "v9", S_V9,
             "**Table 2b. Per-slice paired 비교 (강화 SS2D vs 통제판) — §4.3 서술 근거.**",
             r"Per-slice paired comparison, enhanced vs.\ controlled SS2D (same conventions as Table 2).",
             "tab:paired-v9")

# ---------------------------------------------------------------- Table 3 (3모델)
md = ["**Table 3. 강화 SS2D vs 통제판 (val 전체 7,334 슬라이스, brain-masked, 슬라이스 단위 평균).**", "",
      "| | SSIM | PSNR (dB) | NMSE | L1 (×10⁻⁶) |", "|---|---:|---:|---:|---:|"]
best = {m: max(["v9", "ss2d", "gru"], key=lambda p: (-M[f"{p}_{m}"].mean() if m in LOWER else M[f"{p}_{m}"].mean()))
        for m in METRICS}
tex_rows = []
for p in ["v9", "ss2d", "gru"]:
    cells_md, cells_tex = [], []
    for m in METRICS:
        v = f_mean(p, m)
        cells_md.append(f"**{v}**" if best[m] == p else v)
        cells_tex.append(rf"\textbf{{{v}}}" if best[m] == p else v)
    name_md = f"**{INFO[p]['ko']} (best ep{INFO[p]['ep']})**" if p == "v9" else f"{INFO[p]['ko']} (best ep{INFO[p]['ep']})"
    md.append("| " + name_md + " | " + " | ".join(cells_md) + " |")
    tex_rows.append(" & ".join([INFO[p]["en"] + f" (best ep{INFO[p]['ep']})"] + cells_tex))
tex = tex_table("Enhanced vs.\\ controlled SS2D (full validation set, brain-masked, per-slice means). "
                "Bold = best per metric.", "tab:enhanced", "lrrrr",
                " & SSIM & PSNR (dB) & NMSE & " + NAME_EN["l1"], tex_rows)
write_pair("table3_enhanced", md, tex)

# ---------------------------------------------------------------- Table S1 (contrast)
md = ["**Table S1. Contrast 서브그룹별 우위 슬라이스 비율** (각 칸 = SSIM 기준 (4지표 범위)).", "",
      "| Contrast | n (슬라이스) | SS2D vs GRU | 강화판 vs 통제판 |", "|---|---:|---:|---:|"]
tex_rows = []
for c in sorted(set(contrast)):
    sel = contrast == c
    cell = {}
    for tag, (a, b) in [("v8", ("gru", "ss2d")), ("v9", ("ss2d", "v9"))]:
        ws = [100 * np.mean(delta(a, b, m)[sel] > 0) for m in METRICS]
        cell[tag] = (ws[0], min(ws), max(ws))
    s8 = f"{cell['v8'][0]:.1f}% ({cell['v8'][1]:.1f}~{cell['v8'][2]:.1f})"
    s9 = f"{cell['v9'][0]:.1f}% ({cell['v9'][1]:.1f}~{cell['v9'][2]:.1f})"
    if cell["v9"][0] < 50:
        s9 = f"**{s9}**"
    md.append(f"| {c} | {sel.sum():,} | {s8} | {s9} |")
    tex_rows.append(" & ".join([c, f"{sel.sum():,}".replace(",", r"{,}"),
                                tex_escape(s8.replace("**", "")),
                                (r"\textbf{" + tex_escape(s9.replace('**', '')) + "}") if "**" in s9 else tex_escape(s9)]))
tex = tex_table(r"Slice win-rates by contrast subgroup. Each cell: SSIM win-rate (min$\sim$max over the four metrics). "
                r"Bold marks the reversed subgroup.", "tab:contrast", "lrrr",
                r"Contrast & $n$ (slices) & SS2D vs.\ GRU & Enhanced vs.\ controlled", tex_rows)
write_pair("tableS1_contrast", md, tex)

# ================================================================ 관례형 표 (2026-09-03, IEIE 투고용)
# 최근 MRI 재구성 논문(MMR-Mamba·DM-Mamba·HiFi-Mamba·MambaRecon·SO-Mamba)·교수님 논문(Oh 2025)·fastMRI 공식 관례에 맞춘 표:
#   - 표 내용 영문, L1 열 제외, nMSE 는 % 단위, 볼륨 단위 mean±SD(ddof=1, n=464; 슬라이스 단위 변형도 생성),
#   - Params (M) 열, Zero-filled 기준선 행(results/eval/zero_filled/, CPU 계산; 없으면 [TBD]),
#   - 최고값 **굵게**·차선 __밑줄__ (IEIE 빌더가 두 마크업을 해석), 머리글 ↑↓.
# 출력: tableC1_main{,_slice}.{md,tex}, tableC2_paired, tableC3_efficiency, tableCS1_contrast, ieie_table1_block.md
ZF_CSV = os.path.join(ROOT, "results/eval/zero_filled/per_slice_zero_filled.csv")
CONV = {  # 표 안 영문 명칭 · Params (M)
    "zf":   {"en": "Zero-filled",        "params": "–"},
    "gru":  {"en": "bi-GRU (original)",  "params": "668"},
    "ss2d": {"en": "SS2D (controlled)",  "params": "31"},
    "v9":   {"en": "Enhanced SS2D",      "params": "33"},
}
CM = ["ssim", "psnr", "nmse"]                       # 관례형 표의 지표 3종 (L1 제외)
CM_HEAD_MD = {"ssim": "SSIM ↑", "psnr": "PSNR (dB) ↑", "nmse": "nMSE (%) ↓"}
CM_HEAD_TEX = {"ssim": r"SSIM $\uparrow$", "psnr": r"PSNR (dB) $\uparrow$", "nmse": r"nMSE (\%) $\downarrow$"}
CM_SCALE = {"ssim": 1.0, "psnr": 1.0, "nmse": 100.0}   # NMSE → %
CM_FMT = {"ssim": ("{:.4f}", "{:.4f}"), "psnr": ("{:.2f}", "{:.2f}"), "nmse": ("{:.3f}", "{:.3f}")}

# zero-filled per-slice 결과를 (file, slice_idx) 키로 조인 — 없으면 None
ZF = None
if os.path.exists(ZF_CSV):
    zf_rows = {(r["file"], int(r["slice_idx"])): r for r in csv.DictReader(open(ZF_CSV))}
    keys = [(r["file"], int(r["slice_idx"])) for r in rows]
    if all(k in zf_rows for k in keys):
        ZF = {m: np.array([float(zf_rows[k][f"raw_{m}"]) for k in keys]) for m in METRICS}
        print(f"  zero-filled 기준선 조인 완료 ({len(zf_rows):,} 슬라이스, raw 변형)")
    else:
        print("  !! zero-filled CSV 가 불완전 — Zero-filled 행은 [TBD]")
else:
    print("  zero-filled CSV 없음 — Zero-filled 행은 [TBD] (v8_eter_pure/eval_zero_filled_v8.py 실행 필요)")


def arm_values(p, m):
    return ZF[m] if p == "zf" else M[f"{p}_{m}"]


def vol_mean(a):
    return np.array([a[ix].mean() for ix in vol_idx])


def ms(p, m, unit):
    """mean±SD 문자열. unit='volume' 이면 볼륨 단위 평균의 평균±SD(n=464), 'slice' 면 슬라이스 단위(n=7,334)."""
    a = arm_values(p, m) * CM_SCALE[m]
    if unit == "volume":
        a = vol_mean(a)
    fm, fs = CM_FMT[m]
    return fm.format(a.mean()) + "±" + fs.format(a.std(ddof=1))


def mean_only(p, m, unit):
    a = arm_values(p, m) * CM_SCALE[m]
    if unit == "volume":
        a = vol_mean(a)
    return CM_FMT[m][0].format(a.mean())


def rank_marks(arms, m, unit):
    """열 안에서 최고 → 'b', 차선 → 'u' (나머지 '')."""
    vals = {}
    for p in arms:
        if p == "zf" and ZF is None:
            continue
        a = arm_values(p, m) * CM_SCALE[m]
        vals[p] = (vol_mean(a) if unit == "volume" else a).mean()
    order = sorted(vals, key=lambda p: vals[p] if m in LOWER else -vals[p])
    return {p: ("b" if i == 0 else "u" if i == 1 else "") for i, p in enumerate(order)}


def mark_md(text, k):
    return f"**{text}**" if k == "b" else f"__{text}__" if k == "u" else text


def mark_tex(text, k):
    return rf"\textbf{{{text}}}" if k == "b" else rf"\underline{{{text}}}" if k == "u" else text


def conv_main_table(unit):
    arms = ["zf", "gru", "ss2d", "v9"]
    n_txt = f"{V} volumes" if unit == "volume" else f"{n:,} slices"
    cap_en = (f"Quantitative comparison on the fastMRI brain multi-coil validation set ({n_txt}, R = 4, brain-masked); "
              f"mean±SD over {'volumes' if unit == 'volume' else 'slices'}, best in bold, second best underlined")
    cap_ko = (f"fastMRI brain multi-coil 검증 집합({V} 볼륨, R=4, brain-masked)의 정량 비교"
              f"({'볼륨' if unit == 'volume' else '슬라이스'} 단위 평균±표준편차, 최고값 굵게·차선 밑줄)")
    marks = {m: rank_marks(arms, m, unit) for m in CM}
    md = [f"**Table 1 ({unit}-level). {cap_en}.**", "",
          "| Method | Params (M) | " + " | ".join(CM_HEAD_MD[m] for m in CM) + " |",
          "|---|---:|" + "---:|" * len(CM)]
    tex_rows, blk_rows = [], []
    for p in arms:
        if p == "zf" and ZF is None:
            cells_md = ["[TBD]"] * len(CM)
            cells_tex = ["[TBD]"] * len(CM)
        else:
            cells_md = [mark_md(ms(p, m, unit), marks[m].get(p, "")) for m in CM]
            cells_tex = [mark_tex(tex_escape(ms(p, m, unit)), marks[m].get(p, "")) for m in CM]
        md.append(f"| {CONV[p]['en']} | {CONV[p]['params']} | " + " | ".join(cells_md) + " |")
        blk_rows.append(f"| {CONV[p]['en']} | {CONV[p]['params']} | " + " | ".join(cells_md) + " |")
        tex_rows.append(" & ".join([CONV[p]["en"], CONV[p]["params"].replace("–", "--")] + cells_tex))
    md += ["", "(Zero-filled = RSS of the inverse FFT of the undersampled k-space, no intensity rescaling; "
           "SD = sample standard deviation (ddof = 1). Public leaderboard U-Net/E2E-VarNet checkpoints were trained on "
           "train+val and are excluded from the ranking; see text.)"]
    tex = tex_table(cap_en + ".", f"tab:main-{unit}", "lr" + "r" * len(CM),
                    "Method & Params (M) & " + " & ".join(CM_HEAD_TEX[m] for m in CM), tex_rows)
    write_pair(f"tableC1_main_{unit}", md, tex)
    # IEIE .src.md 붙여넣기용 블록 (단 폭 col, 열 폭 합 4560 ≤ 4563 twips; 8 pt Times 기준 숫자 셀 줄바꿈 없음)
    blk = ["@table: col | 1120,560,1040,880,960",
           f"@cap_ko: {cap_ko}",
           f"@cap_en: {cap_en}",
           "| Method | Params (M) | " + " | ".join(CM_HEAD_MD[m] for m in CM) + " |",
           "|---|---|" + "---|" * len(CM)] + blk_rows + ["@end"]
    return blk


blk_vol = conv_main_table("volume")
blk_slice = conv_main_table("slice")
open(os.path.join(OUT, "ieie_table1_block.md"), "w").write(
    "%% IEIE .src.md 붙여넣기용 — 볼륨 단위 (권장, fastMRI 관례)\n" + "\n".join(blk_vol)
    + "\n\n%% 슬라이스 단위 변형 (현행 문서·초안의 대표 수치 0.9126/0.9140/0.9145 와 일치)\n" + "\n".join(blk_slice) + "\n")
print("  wrote ieie_table1_block.md")

# ---- Table C2: 쌍대 통계 보조표 (두 비교 × 3지표, nMSE Δ 는 10⁻³ % 단위)
def f_delta_conv(m, v):
    return f"{v:+.4f}" if m == "ssim" else f"{v:+.2f}" if m == "psnr" else f"{v * 1e5:+.1f}"


md = ["**Table 2. Paired per-slice analysis (7,334 slices). Δ = row method − comparator, oriented so that positive favors "
      "the row method (nMSE sign negated). 95% CI: volume-clustered bootstrap (2,000 resamples); p: two-sided Wilcoxon "
      "signed-rank over 464 volumes.**", "",
      "| Comparison | Metric | Δ median (IQR) | Slices favoring (%) [95% CI] | Volumes favoring (%) | p |",
      "|---|---|---:|---:|---:|---:|"]
tex_rows = []
for label, S in [("SS2D vs. bi-GRU", S_V8), ("Enhanced vs. SS2D", S_V9)]:
    for i, m in enumerate(CM):
        s = S[m]
        mname = {"ssim": "SSIM", "psnr": "PSNR (dB)", "nmse": "nMSE (10⁻³ %)"}[m]
        med = f"{f_delta_conv(m, s['med'])} ({f_delta_conv(m, s['iqr'][0])}, {f_delta_conv(m, s['iqr'][1])})"
        md.append(f"| {label if i == 0 else ''} | {mname} | {med} | {s['win']:.1f} [{s['win_ci'][0]:.1f}, {s['win_ci'][1]:.1f}] "
                  f"| {s['vol_win']:.1f} | {f_p(s['p_vol'])} |")
        tex_rows.append(" & ".join([
            (rf"\multirow{{3}}{{*}}{{{label}}}" if i == 0 else ""),
            tex_escape(mname).replace("10⁻³", r"$10^{-3}$"), tex_escape(med),
            f"{s['win']:.1f} [{s['win_ci'][0]:.1f}, {s['win_ci'][1]:.1f}]", f"{s['vol_win']:.1f}",
            f_p(s["p_vol"]).replace("<", r"$<$")]))
    tex_rows.append(r"\midrule")
tex_rows.pop()
tex = tex_table(r"Paired per-slice analysis (7{,}334 slices). $\Delta$ = row method $-$ comparator, oriented so that positive "
                r"favors the row method (nMSE sign negated). 95\% CI: volume-clustered bootstrap (2{,}000 resamples); "
                r"$p$: two-sided Wilcoxon signed-rank over 464 volumes.", "tab:paired", "llrrrr",
                r"Comparison & Metric & $\Delta$ median (IQR) & Slices favoring (\%) [95\% CI] & Volumes favoring (\%) & $p$",
                tex_rows)
write_pair("tableC2_paired", md, tex)

# ---- Table C3: 효율 (상수 — draft_ko_v2 Table 5 의 h/ep 실측; 추론 ms/slice·VRAM 은 GPU 확보 후)
EFF = [("bi-GRU (original)", "668", "2.41", "[TBD]", "[TBD]"),
       ("SS2D (controlled)", "31", "3.07", "[TBD]", "[TBD]"),
       ("Enhanced SS2D", "33", "2.84‡", "[TBD]", "[TBD]")]
md = ["**Table 3. Parameter and time efficiency (TITAN RTX 24 GB, batch 8, AMP, 384×384).** "
      "† Median wall-clock between 5-epoch checkpoints, validation included. ‡ Trained after a container/dataloader "
      "upgrade — not directly comparable with the two v8 rows. Inference time and VRAM to be measured once GPU0 is free.", "",
      "| Method | Params (M) | Train (h/epoch)† | Inference (ms/slice) | Peak VRAM (GB) |", "|---|---:|---:|---:|---:|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in EFF]
tex = tex_table(r"Parameter and time efficiency (TITAN RTX 24\,GB, batch 8, AMP, $384\times384$). "
                r"$\dagger$ median wall-clock between 5-epoch checkpoints, validation included; "
                r"$\ddagger$ trained after a container/dataloader upgrade.", "tab:efficiency", "lrrrr",
                r"Method & Params (M) & Train (h/epoch)$^\dagger$ & Inference (ms/slice) & Peak VRAM (GB)",
                [" & ".join([r[0], r[1], r[2].replace("‡", r"$^\ddagger$"), r[3], r[4]]) for r in EFF])
write_pair("tableC3_efficiency", md, tex)

# ---- Table CS1: contrast 별 SSIM (볼륨/슬라이스 평균) + 우위 슬라이스 비율(SSIM 기준, 괄호 = 3지표 범위)
#      volume: 볼륨 단위 평균의 contrast 내 평균 (n = 볼륨 수) — IEIE 초안 표 4 (09-04 결정: 볼륨 단위)
#      slice : 슬라이스 평균 (n = 슬라이스 수)
vol_contrast = np.array([contrast[ix[0]] for ix in vol_idx])
blk4 = ["%% IEIE .src.md 붙여넣기용 표 4 — 볼륨 단위 SSIM + 우위 슬라이스 비율(SSIM 기준, 괄호 = SSIM·PSNR·nMSE 범위)"]
for unit in ["volume", "slice"]:
    n_lab = "n (volumes)" if unit == "volume" else "n (slices)"
    md = [f"**Table S1 ({unit}-level). Per-contrast SSIM ({unit}-level mean; best in bold) and fraction of favoring slices "
          "(SSIM-based; parentheses: range over SSIM, PSNR and nMSE).**", "",
          f"| Contrast | {n_lab} | SSIM bi-GRU | SSIM SS2D | SSIM Enhanced | SS2D vs. bi-GRU (%) | Enhanced vs. SS2D (%) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    tex_rows, src_rows = [], []
    for c in sorted(set(contrast)):
        sel = contrast == c
        if unit == "volume":
            vsel = vol_contrast == c
            means = [vol_mean(M[f"{p}_ssim"])[vsel].mean() for p in ["gru", "ss2d", "v9"]]
            n_c = int(vsel.sum())
        else:
            means = [M[f"{p}_ssim"][sel].mean() for p in ["gru", "ss2d", "v9"]]
            n_c = int(sel.sum())
        win = {}
        for tag, (a, b) in [("v8", ("gru", "ss2d")), ("v9", ("ss2d", "v9"))]:
            ws = [100 * np.mean(delta(a, b, m)[sel] > 0) for m in CM]
            win[tag] = f"{ws[0]:.1f} ({min(ws):.1f}–{max(ws):.1f})"
        best_i = int(np.argmax(means))
        cells = [f"{v:.4f}" for v in means]
        cells_md = [f"**{x}**" if i == best_i else x for i, x in enumerate(cells)]
        cells_tex = [rf"\textbf{{{x}}}" if i == best_i else x for i, x in enumerate(cells)]
        row_md = f"| {c} | {n_c:,} | " + " | ".join(cells_md) + f" | {win['v8']} | {win['v9']} |"
        md.append(row_md)
        src_rows.append(row_md)
        tex_rows.append(" & ".join([c, f"{n_c:,}".replace(",", r"{,}")] + cells_tex +
                                   [tex_escape(win["v8"]), tex_escape(win["v9"])]))
    tex = tex_table(rf"Per-contrast SSIM ({unit}-level mean; best in bold) and fraction of favoring slices "
                    r"(SSIM-based; parentheses: range over SSIM, PSNR and nMSE).", f"tab:contrast-{unit}",
                    "lrrrrrr", rf"Contrast & $n$ ({'volumes' if unit == 'volume' else 'slices'}) & SSIM bi-GRU & SSIM SS2D & "
                    r"SSIM Enhanced & SS2D vs.\ bi-GRU (\%) & Enhanced vs.\ SS2D (\%)", tex_rows)
    write_pair(f"tableCS1_contrast_{unit}", md, tex)
    if unit == "volume":
        blk4 += ["@table: page | 1100,800,1100,1100,1100,2150,2150",
                 "@cap_ko: Contrast 서브그룹별 SSIM(볼륨 단위 평균, 최고값 굵게)과 우위 슬라이스 비율. 우위 비율 칸은 SSIM 기준이며 괄호는 세 지표(SSIM·PSNR·nMSE)에 걸친 범위",
                 "@cap_en: Per-contrast SSIM (volume-level mean, best in bold) and fraction of favoring slices. The fraction cells are SSIM-based; parentheses give the range over the three metrics (SSIM, PSNR, nMSE)",
                 md[2], "|---|---|---|---|---|---|---|"] + src_rows + ["@end"]
open(os.path.join(OUT, "ieie_table4_block.md"), "w", encoding="utf-8").write("\n".join(blk4) + "\n")
print("  wrote ieie_table4_block.md")

# ---- Table C4 / IEIE 표 블록: 공개 모델 참고선 (전체 검증셋 · 우리 프로토콜 · CPU 추론 — results/eval/baselines_384_full/)
#      행 = U-Net†·E2E-VarNet†·PromptMR+ (v8_eter_pure/eval_baselines_full.py 의 per-slice CSV) + 우리 3모델(좌표용, 표 2 와 동일 값).
#      순위 표시(굵게/밑줄) 없음 — 참고선: † 는 train+val 누수, 세 공개 가중치 모두 원 코일 구성으로 학습된 것을 본 프로토콜에 적용(domain shift),
#      PromptMR+ 는 인접 5슬라이스 입력. 공개 모델 지표는 brain mask 내부 per-slice LS 강도 정합 후(정합은 공개 모델에만 유리할 수 있음), 우리 행은 정합 없음.
#      미완주(7,334 슬라이스 미만) 방법은 [TBD] 셀 — CSV 가 채워지면 재실행만으로 확정된다(09-08: U-Net†·E2E-VarNet† 완주, PromptMR+ 진행 중).
PUB_DIR = os.path.join(ROOT, "results/eval/baselines_384_full")
PUB = [("unet", "U-Net†", "train+val", "496"),        # fastMRI leaderboard U-Net (chans 256, 4 pools) 496.4M
       ("varnet", "E2E-VarNet†", "train+val", "30"),  # 12 cascades, chans 18, sens 8 → 29.9M
       ("promptmr", "PromptMR+", "train", "93")]      # fm-brain 공개 가중치(train 구획만) 92.9M
keys_ours = [(r["file"], int(r["slice_idx"])) for r in rows]


def load_pub(method):
    """per_slice_<method>.csv → 우리 CSV 행 순서로 정렬된 {metric: array}. 미완주면 (None, 확보 수)."""
    path = os.path.join(PUB_DIR, f"per_slice_{method}.csv")
    if not os.path.exists(path):
        return None, 0
    got = {}
    for r in csv.DictReader(open(path)):
        try:
            v = {m: float(r[m]) for m in METRICS}
        except (KeyError, ValueError):
            continue                       # 동시 append 중 잘린 행
        if all(np.isfinite(x) for x in v.values()):
            got[(r["file"], int(r["slice_idx"]))] = v
    if not all(k in got for k in keys_ours):
        return None, len(got)
    return {m: np.array([got[k][m] for k in keys_ours]) for m in METRICS}, len(got)


def ms_arr(a, m, unit):
    a = a * CM_SCALE[m]
    if unit == "volume":
        a = vol_mean(a)
    fm, fs = CM_FMT[m]
    return fm.format(a.mean()) + "±" + fs.format(a.std(ddof=1))


def fav_vs_ss2d(a_ssim, a_psnr):
    """통제 SS2D 보다 나은 슬라이스 비율 'SSIM / PSNR' (%)."""
    return f"{100 * np.mean(a_ssim > M['ss2d_ssim']):.1f} / {100 * np.mean(a_psnr > M['ss2d_psnr']):.1f}"


ref_rows, PUB_DATA, pending = [], {}, []
for meth, name, split, params in PUB:
    arr, n_got = load_pub(meth)
    PUB_DATA[meth] = arr
    if arr is None:
        ref_rows.append((name, split, params, ["[TBD]"] * len(CM), "[TBD]"))
        pending.append(name)
        print(f"  참고선 {name}: {n_got:,}/{n:,} 슬라이스 — 미완주 → [TBD]")
    else:
        ref_rows.append((name, split, params, [ms_arr(arr[m], m, "volume") for m in CM],
                         fav_vs_ss2d(arr["ssim"], arr["psnr"])))
        print(f"  참고선 {name}: 완주 ({n_got:,} 슬라이스)")
for p in ["gru", "ss2d", "v9"]:
    fav = "–" if p == "ss2d" else fav_vs_ss2d(M[f"{p}_ssim"], M[f"{p}_psnr"])
    ref_rows.append((CONV[p]["en"], "train", CONV[p]["params"], [ms(p, m, "volume") for m in CM], fav))

REF_CAP_KO = ("공개 모델 참고선 — 전체 검증 집합(464 볼륨/7,334 슬라이스)을 본 논문과 동일한 프로토콜(384² 재-FFT·16코일·R=4·brain-masked)로 "
              "추론한 볼륨 단위 평균±표준편차. 참고선이므로 순위 표시(굵게·밑줄)는 두지 않는다. 마지막 열은 통제 SS2D보다 나은 슬라이스의 비율(SSIM / PSNR, %)")
REF_CAP_EN = ("Public-model reference lines — full validation set (464 volumes/7,334 slices) inferred under the protocol of this paper "
              "(384² re-FFT, 16 coils, R = 4, brain-masked); mean±SD over volumes. Reference only, hence no ranking marks. "
              "Last column: fraction of slices on which the method beats the controlled SS2D (SSIM / PSNR, %)")
REF_NOTE = ("†: public fastMRI leaderboard weights trained on the train+val split, so this validation set is part of their training data. "
            "PromptMR+: public weights trained on the train split only, but a 12-cascade unrolled model that takes five adjacent slices as input. "
            "All public weights were trained with their native coil configuration and are applied here to the 384² re-FFT/16-coil protocol "
            "(domain shift). Public-model metrics are computed after per-slice least-squares intensity alignment inside the brain mask "
            "(their output scales differ; the alignment can only favor them); the three rows of this paper are the unaligned values of Table 2. "
            "CPU fp32 inference."
            + (" [TBD] = full-validation inference still running." if pending else ""))
REF_HEAD = ["Method", "Training split", "Params (M)"] + [CM_HEAD_MD[m] for m in CM] + ["Favoring vs. SS2D (%) SSIM / PSNR"]
md = [f"**Table 4. {REF_CAP_EN}.** {REF_NOTE}", "",
      "| " + " | ".join(REF_HEAD) + " |", "|---|---|---:|" + "---:|" * len(CM) + "---:|"]
tex_rows = []
for name, split, params, cells, fav in ref_rows:
    md.append(f"| {name} | {split} | {params} | " + " | ".join(cells) + f" | {fav} |")
    tex_rows.append(" & ".join([name.replace("†", r"$^\dagger$"), split, params.replace("–", "--")]
                               + [tex_escape(c) for c in cells] + [tex_escape(fav).replace("–", "--")]))
tex = tex_table(tex_escape(REF_CAP_EN) + ". " + tex_escape(REF_NOTE).replace("†", r"$^\dagger$"), "tab:reference",
                "llr" + "r" * len(CM) + "r",
                "Method & Training split & Params (M) & " + " & ".join(CM_HEAD_TEX[m] for m in CM)
                + r" & Favoring vs.\ SS2D (\%) SSIM / PSNR", tex_rows)
write_pair("tableC4_reference", md, tex)
blk_ref = ["%% IEIE .src.md 붙여넣기용 — 공개 모델 참고선(볼륨 단위, 순위 표시 없음). 열 폭 합 9400 twips(page). 미완주 방법은 [TBD] 셀.",
           "@table: page | 1900,1000,800,1400,1300,1300,1700",
           f"@cap_ko: {REF_CAP_KO}",
           f"@cap_en: {REF_CAP_EN}",
           "| " + " | ".join(REF_HEAD) + " |", "|---|---|---|" + "---|" * len(CM) + "---|"]
blk_ref += [f"| {name} | {split} | {params} | " + " | ".join(cells) + f" | {fav} |" for name, split, params, cells, fav in ref_rows]
blk_ref += [f"@note: {REF_NOTE}", "@end"]
open(os.path.join(OUT, "ieie_table_ref_block.md"), "w", encoding="utf-8").write("\n".join(blk_ref) + "\n")
print("  wrote ieie_table_ref_block.md")

# ---------------------------------------------------------------- 자가 검증
checks = [
    ("Table1 SS2D SSIM", f_mean("ss2d", "ssim"), "0.9140"),
    ("Table1 SS2D PSNR", f_mean("ss2d", "psnr"), "33.90"),
    ("Table2 SSIM win", f"{S_V8['ssim']['win']:.1f}", "78.2"),
    ("Table2 SSIM winCI-lo", f"{S_V8['ssim']['win_ci'][0]:.1f}", "76.8"),
    ("Table2 SSIM winCI-hi", f"{S_V8['ssim']['win_ci'][1]:.1f}", "79.7"),
    ("Table2b SSIM win", f"{S_V9['ssim']['win']:.1f}", "55.8"),
    ("Table3 v9 SSIM", f_mean("v9", "ssim"), "0.9145"),
    ("TableC1 volume GRU SSIM", mean_only("gru", "ssim", "volume"), "0.9127"),
    ("TableC1 volume SS2D SSIM", mean_only("ss2d", "ssim", "volume"), "0.9141"),
    ("TableC1 volume v9 SSIM", mean_only("v9", "ssim", "volume"), "0.9146"),
    ("TableC1 volume SS2D nMSE%", mean_only("ss2d", "nmse", "volume"), "0.438"),
]
for meth, want in [("varnet", "0.9181"), ("unet", "0.8971")]:     # baseline_summary_full.md (09-06 / 09-08)
    if PUB_DATA.get(meth) is not None:
        checks.append((f"TableC4 volume {meth} SSIM", ms_arr(PUB_DATA[meth]["ssim"], "ssim", "volume").split("±")[0], want))
print("\n[자가 검증 — draft_ko_v2 수치 재현]")
ok = True
for name, got, want in checks:
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}: {got} (기대 {want})")
print("ALL PASS" if ok else "!! MISMATCH — 초안과 대조 필요")
