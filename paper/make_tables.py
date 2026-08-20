"""논문 표 자동 생성기 — per-slice CSV 단일 원천에서 Table 1·2·3·S1 을 재현 가능하게 뽑는다.

입력:  results/eval/v9_unleashed/per_slice_paired_v9.csv (7,334 슬라이스 × 3모델 × 4지표)
출력:  paper/tables/table{1,2,2b,3,S1}.{md,tex}
       - .md  : 한국어 헤더 (현행 draft_ko_v2 와 동일 표기)
       - .tex : 영문 헤더 + booktabs (영어 전환용; MDPI 는 Word 도 허용되므로 선택 사용)

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

# ---------------------------------------------------------------- 자가 검증
checks = [
    ("Table1 SS2D SSIM", f_mean("ss2d", "ssim"), "0.9140"),
    ("Table1 SS2D PSNR", f_mean("ss2d", "psnr"), "33.90"),
    ("Table2 SSIM win", f"{S_V8['ssim']['win']:.1f}", "78.2"),
    ("Table2 SSIM winCI-lo", f"{S_V8['ssim']['win_ci'][0]:.1f}", "76.8"),
    ("Table2 SSIM winCI-hi", f"{S_V8['ssim']['win_ci'][1]:.1f}", "79.7"),
    ("Table2b SSIM win", f"{S_V9['ssim']['win']:.1f}", "55.8"),
    ("Table3 v9 SSIM", f_mean("v9", "ssim"), "0.9145"),
]
print("\n[자가 검증 — draft_ko_v2 수치 재현]")
ok = True
for name, got, want in checks:
    good = got == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}: {got} (기대 {want})")
print("ALL PASS" if ok else "!! MISMATCH — 초안과 대조 필요")
