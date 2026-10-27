"""IEIE v9 표·짝 비교·본문 토큰 생성기 — v8fix(수정 레시피) 50 epoch 세 모델의 test_full 결과 (2026-10-17).

세 모델(각 SEED 1 로 한 번 학습, RunPod): U-Net only(시퀀스 모듈 제거, 출력 20채널 = 0) · ETER-net(교수님 원본 bi-GRU,
기준 모델) · SS2D(controlled). 최종 평가 = fastMRI brain multicoil test_full 558 볼륨 / 8,852 슬라이스, 최적 체크포인트는
공식 검증 분할 전체(= 검증 배치 0~2, 1,378 볼륨 / 21,842 슬라이스)에서 선택.
2026-10-17 개정(outline_v9 C판 생성기 요청 4건): (a) 표 1 캡션 = "the official validation split (1,378 volumes)",
(b) 주 블록 주석 = 반올림 문장 한 줄(U-Net only 정의·마스크·한 번 학습 문장은 본문·캡션과 중복이라 삭제),
(c) 주석의 공식 정의 SSIM 문장은 NOTE_OFFICIAL_IN_MAIN = False(Word 실측 여유가 있을 때만 True), (d) 본문 토큰 추가
(모듈 파라미터 수·SS2D 기능 시험·마지막 epoch 학습 손실·L1/SSIM 항 비·학습 분할 크기·AXT2 비율·λ_SSIM) — 아래 '추가 입력'.

입력 (git 무시 — 숫자의 유일한 출처):
  <src-dir>/per_volume_{unet,gru,ss2d}.csv   (v8_eter_pure/eval_v8fix_official.py 출력, 558 행)
  <src-dir>/per_slice_{unet,gru,ss2d}.csv    (같은 스크립트, 8,852 행)
  <src-dir>/{unet,gru,ss2d}.log              (첫 줄들의 "(N.NM params)")
  logs/runpod_mirror/logs/PureETER_{UNET,GRU,SS2D}_noDC_R4_brain384_v8fix_s1_fullval/DONE   (best epoch·검증 SSIM)
  검증 per-slice CSV 3 개(선택 집합 크기 N_VAL_VOL·N_VAL_SLICE 계산)
교차 검증 전용 (출력 숫자의 출처로 쓰지 않음): <ref-dir>/three_model_table.md · compare_*.md · summary_*.json,
  DONE 사본(pod3_final_extra_20261017), GRU 학습 로그의 "Loaded … slices total" 줄.
추가 입력 (d — 출처 / 교차 검증):
  - 모듈 파라미터 수: logs/runpod_mirror/logs/PureETER_*_v8fix_s1_fullval/pure_{unet,gru,ss2d}_best.pt 의 state_dict 원소 수
    (최상위 이름 gru_h·gru_v = bi-GRU 모듈, ss2d = SS2D 모듈, unet = U-Net; torch.load mmap, CPU)
    / 점검: GRU 식 3·(입력·은닉+은닉²+2·은닉)·2 (C=32, N=384, H=10), 합계 = eval 로그·log.txt 머리 params, U-Net 세 모델 동일.
  - SS2D 기능 시험: logs/runpod_mirror_pod3/results/v8fix_s1_fullval_test_full/functional_ss2d.json (seq_out_rms_mean,
    tests.shuffle_cross) / 점검: per_slice_ssim 목록으로 평균·SE·하락 수 재계산, 정상 SSIM = <src-dir>/per_slice_ss2d.csv 의
    같은 인덱스(같은 체크포인트·같은 test_full 슬라이스), summary_ss2d.md 표 줄, 사본 md5.
  - 마지막 epoch 학습 손실·L1/SSIM 항 비: 세 런 log.txt 의 'Epoch 50/50' 줄(train_loss·train_l1·train_ssim_loss·
    l1_over_ssim_term = epoch 50 전체 step 평균) / 점검: train_loss ≈ train_l1 + λ·train_ssim_loss, 비 = train_l1/(λ·train_ssim_loss),
    GRU·SS2D wandb-summary.json(전체 정밀도) 반올림 일치. U-Net only 는 wandb 요약이 없음(pod2 종료).
  - 학습 분할 크기: GRU·SS2D 학습 로그 첫 'Loaded … slices total' 줄 / 점검: pod3_setup_logs/verify.log:27, 세 log.txt 의
    steps=⌈65,060/8⌉=8,133 (U-Net only 는 이 step 수로만 확인).
  - AXT2 비율: <src-dir>/per_volume_unet.csv 파일 이름(file_brain_<대조도>_…) / 점검: pod 쪽 평가 per_slice_ss2d.csv 의 파일 목록.
  - λ_SSIM: 세 런 recipe.json 의 lambda_ssim(= lambda_base 1.0 / s̄, s̄ = ×1e6 세기 단위의 s 평균 → 원 단위 λ = 10⁻⁶/s̄_원)
    / 점검: 세 recipe 동일, 나눗셈 항등식, log.txt 머리의 lambda_ssim·s_bar.

계산 규약 (eval_v8fix_official.py / eval_v8fix_official_compare.py 를 그대로 재현):
  - 볼륨 값: ssim_mask·psnr_mask·nmse_mask·ssim_off 는 슬라이스 값의 볼륨 평균, psnr_off·nmse_off·ssim_mask_volrange 는
    볼륨 전체에서 한 번에 계산(volume-pooled), ssim_off_in/out·frac_in 은 슬라이스별 map 평균의 볼륨 평균(per-slice CSV 에 없음).
  - 표: 볼륨 사이 평균 ± SD(ddof=1). nMSE 는 분수 → % (×100).
  - 짝 비교: Δ = A − B (볼륨별), 나은 볼륨 수 = sgn·Δ > 0 (SSIM/PSNR sgn=+1, nMSE/NMSE sgn=−1; 동점 제외),
    p = scipy.stats.wilcoxon(Δ).pvalue (기본값 = 양측) — Δ 가 모두 0 이면 nan.
  - 공식 SSIM Δ 근사 분해: f = A 의 frac_in 평균, 안 = f × mean(Δ ssim_off_in), 밖 = (1 − f) × mean(Δ ssim_off_out).

출력 (paper/tables/): ieie_table1_block_v9.md · ieie_table1_block_v9_official.md · ieie_v9_paired.md · ieie_v9_tokens.{md,json}
자가 점검 (불일치 시 exit 1, 아무것도 쓰지 않음): (a) 볼륨 CSV = 슬라이스 CSV 의 파일별 평균 (b) 표 칸 = three_model_table.md
  (c) 짝 비교 = compare_*.md·three_model_table.md (d) 반올림 전 평균·SD = summary_*.json (1e-9) (e) 블록 형식·금지 어휘·옛 수치.
  --check-only: 점검만 하고 쓰지 않음 + 디스크의 출력이 재생성 결과와 같은지(손편집 감지)까지 확인.
paper/make_tables.py 는 import 하지 않는다(가드 없음 — import 하면 모든 표를 다시 씀).

실행: python paper/make_tables_v8fix.py [--check-only] [--src-dir DIR] [--ref-dir DIR] [--out-dir DIR]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results/eval/v8fix_official_test_full")
OUT_DIR = os.path.join(ROOT, "paper/tables")
MIRROR = os.path.join(ROOT, "logs/runpod_mirror")
EXTRA = os.path.join(MIRROR, "pod3_final_extra_20261017/fastmri_ViT_with_eternet")

# (CSV 접미, 토큰 이름, 원고 표기)
MODELS = [("unet", "UNET", "U-Net only"), ("gru", "ETER", "ETER-net"), ("ss2d", "SS2D", "SS2D(controlled)")]
NAME = {s: n for s, _, n in MODELS}
TOK = {s: t for s, t, _ in MODELS}
RUN = {"unet": "UNET", "gru": "GRU", "ss2d": "SS2D"}
DONE = {s: os.path.join(MIRROR, f"logs/PureETER_{RUN[s]}_noDC_R4_brain384_v8fix_s1_fullval/DONE") for s in RUN}
DONE_COPY = {"gru": os.path.join(EXTRA, "logs/PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval/DONE"),
             "ss2d": os.path.join(ROOT, "logs/runpod_mirror_pod3/logs/PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval/DONE")}
VAL_CSV = {"unet": os.path.join(MIRROR, "results/v8fix_s1_fullval_val/per_slice_unet.csv"),
           "gru": os.path.join(EXTRA, "results/eval/v8fix_s1_fullval_val/per_slice_gru.csv"),
           "ss2d": os.path.join(EXTRA, "results/eval/v8fix_s1_fullval_val/per_slice_ss2d.csv")}
GRU_TRAIN_LOG = os.path.join(EXTRA, "v8_eter_pure/runs/runpod/run_PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval.log")
# (d) 추가 입력
RUNDIR = {s: os.path.dirname(DONE[s]) for s in RUN}
LOGTXT = {s: os.path.join(RUNDIR[s], "log.txt") for s in RUN}
RECIPE = {s: os.path.join(RUNDIR[s], "recipe.json") for s in RUN}
CKPT_BEST = {s: os.path.join(RUNDIR[s], f"pure_{s}_best.pt") for s in RUN}
TRAIN_LOG = {"gru": GRU_TRAIN_LOG,
             "ss2d": os.path.join(EXTRA, "v8_eter_pure/runs/runpod/run_PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval.log")}
VERIFY_LOG = os.path.join(MIRROR, "pod3_setup_logs/verify.log")
WANDB_SUMMARY = {
    "gru": os.path.join(EXTRA, "wandb/run-20261015_083843-PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval_pod3f/files/wandb-summary.json"),
    "ss2d": os.path.join(MIRROR, "pod3_final_extra/fastmri_ViT_with_eternet/wandb/"
                         "run-20261010_095353-PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval_pod3/files/wandb-summary.json")}
FUNC_DIR = os.path.join(ROOT, "logs/runpod_mirror_pod3/results/v8fix_s1_fullval_test_full")
FUNC_JSON = os.path.join(FUNC_DIR, "functional_ss2d.json")
FUNC_SUMMARY_MD = os.path.join(FUNC_DIR, "summary_ss2d.md")
FUNC_JSON_COPIES = [os.path.join(MIRROR, "results_pod3/v8fix_s1_fullval_test_full/functional_ss2d.json"),
                    os.path.join(EXTRA, "results/eval/v8fix_s1_fullval_test_full/functional_ss2d.json")]
POD_SLICE_CSV_SS2D = os.path.join(FUNC_DIR, "per_slice_ss2d.csv")   # pod 쪽 학습기 평가 출력(AXT2 교차 검증)
FINAL_EPOCH = 50
# GRU 모듈 파라미터 식의 상수 (구조 정의 — 점검 전용): 입력 채널 C, 영상 한 변 N, 은닉 배수 H
GRU_C, GRU_N, GRU_H = 32, 384, 10

# 점검 전용 기댓값 (과제 명세) — 출력 숫자의 출처가 아님
EXPECT = dict(n_vol=558, n_slice=8852, n_val_vol=1378, n_val_slice=21842, n_train_vol=4110, n_train_slice=65060,
              params={"unet": "31.1", "gru": "668.2", "ss2d": "31.2"}, bestep={"unet": 48, "gru": 46, "ss2d": 50})

# 지표: (토큰, per-volume 열, per-slice 열 또는 None, sgn(+1 = 클수록 좋음), 종류, 설명)
METRICS = [
    ("SSIM", "ssim_mask", "ssim", +1, "ssim", "SSIM inside the brain mask"),
    ("PSNR", "psnr_mask", "psnr", +1, "psnr", "PSNR (dB) inside the brain mask"),
    ("NMSE", "nmse_mask", "nmse", -1, "nmse", "nMSE (%) inside the brain mask"),
    ("OFFSSIM", "ssim_off", "ssim_official", +1, "ssim", "SSIM, fastMRI official definition"),
    ("OFFPSNR", "psnr_off", None, +1, "psnr", "PSNR (dB), fastMRI official definition (volume-pooled)"),
    ("OFFNMSE", "nmse_off", None, -1, "nmse", "nMSE (%), fastMRI official definition (volume-pooled)"),
    ("OFFSSIMIN", "ssim_off_in", None, +1, "ssim", "official SSIM map averaged inside the brain mask"),
    ("OFFSSIMOUT", "ssim_off_out", None, +1, "ssim", "official SSIM map averaged outside the brain mask"),
]
PAIRS = [("SS2D_ETER", "ss2d", "gru", "primary"), ("UNET_ETER", "unet", "gru", "primary"),
         ("SS2D_UNET", "ss2d", "unet", "secondary")]
NDEC = {"ssim": 4, "psnr": 2, "nmse": 3}          # 표 정밀도 (nMSE 는 % 단위)
SCALE = {"ssim": 1.0, "psnr": 1.0, "nmse": 100.0}
WIDTHS = [1180, 540, 1015, 870, 930]
# 2026-10-18 학술대회판 5쪽 감축(사용자 승인, ARS run ledger seq 45): 주 표 1 은 Params 열을 빼고(파라미터 수는 본문
# 2.1·2.2절) Word 실측 글자 폭(HY신명조 8 pt: 숫자 5.0 pt, ± 8.0 pt)에 맞춘 4열 폭을 쓴다. 대안 블록(block_official)은 그대로.
WIDTHS_MAIN = [1025, 1340, 1035, 1135]
COL_W_CONF = 4535
FORBIDDEN_OLD = ["0.9126", "0.9140", "0.9127", "0.9141", "0.9146", "0.9136", "0.9133", "0.7523", "0.9145"]
OUT_FILES = ["ieie_table1_block_v9.md", "ieie_table1_block_v9_official.md", "ieie_v9_paired.md",
             "ieie_v9_tokens.md", "ieie_v9_tokens.json"]
MINUS = "−"


class CheckError(Exception):
    pass


FAILS = []
NPASS = [0]


def check(cond, msg):
    if cond:
        NPASS[0] += 1
    else:
        FAILS.append(msg)


# ------------------------------------------------------------------ 서식
def signed(x, nd):
    return f"{x:+.{nd}f}".replace("-", MINUS)


def mean_sd(v, kind):
    nd, sc = NDEC[kind], SCALE[kind]
    return f"{sc * v.mean():.{nd}f}±{sc * v.std(ddof=1):.{nd}f}"


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def thousands(n):
    return f"{n:,}"


# ------------------------------------------------------------------ 적재
def load(src):
    V, S, params = {}, {}, {}
    for s, _, _ in MODELS:
        V[s] = pd.read_csv(os.path.join(src, f"per_volume_{s}.csv"))
        S[s] = pd.read_csv(os.path.join(src, f"per_slice_{s}.csv"))
        with open(os.path.join(src, f"{s}.log"), encoding="utf-8", errors="replace") as f:
            head = f.read(20000)
        m = re.search(r"\(([\d.]+)M params\)", head)
        params[s] = m.group(1) if m else None
    done = {}
    for s in RUN:
        with open(DONE[s]) as f:
            done[s] = json.load(f)
    val = {}
    for s, p in VAL_CSV.items():
        d = pd.read_csv(p, usecols=["file", "ssim"])
        val[s] = dict(n_slice=len(d), n_vol=int(d["file"].nunique()), ssim_mean=float(d["ssim"].mean()))
    return V, S, params, done, val


def paired(A, B, key, sgn):
    """eval_v8fix_official_compare.py 와 같은 계산 (Δ = A − B)."""
    x, y = A[key].to_numpy(float), B[key].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    dd = x[ok] - y[ok]
    better = int(((sgn * dd) > 0).sum())
    p = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else float("nan")
    return dict(dd=dd, mean=float(dd.mean()), better=better, n=int(ok.sum()), ties=int((dd == 0).sum()), p=p,
                a_mean=float(x[ok].mean()), a_sd=float(x[ok].std(ddof=1)),
                b_mean=float(y[ok].mean()), b_sd=float(y[ok].std(ddof=1)))


def compute(V, S, params, done, val):
    R = dict(V=V, S=S, params=params, done=done, val=val)
    R["n_vol"] = len(V["unet"])
    R["n_slice"] = len(S["unet"])
    R["pairs"] = {}
    for pname, a, b, _ in PAIRS + [("ETER_UNET", "gru", "unet", "check")]:
        res = {}
        for tok, vk, sk, sgn, kind, _ in METRICS:
            r = paired(V[a], V[b], vk, sgn)
            if sk is not None:
                ds = S[a][sk].to_numpy(float) - S[b][sk].to_numpy(float)
                r["slice_better"] = int(((sgn * ds) > 0).sum())
                r["slice_n"] = int(np.isfinite(ds).sum())
            else:
                r["slice_better"] = None
                r["slice_n"] = None
            res[tok] = r
        fi = float(V[a]["frac_in"].mean())
        din = float((V[a]["ssim_off_in"] - V[b]["ssim_off_in"]).mean())
        dout = float((V[a]["ssim_off_out"] - V[b]["ssim_off_out"]).mean())
        res["_decomp"] = dict(frac_in=fi, din=din, dout=dout, t_in=fi * din, t_out=(1 - fi) * dout,
                              total=res["OFFSSIM"]["mean"])
        R["pairs"][pname] = res
    R["frac_in"] = float(V["unet"]["frac_in"].mean())
    return R


# ------------------------------------------------------------------ 블록
def model_row(s, R, cols):
    V = R["V"][s]
    return (f"| {NAME[s]} | {R['params'][s]} | {mean_sd(V[cols[0]], 'ssim')} | "
            f"{mean_sd(V[cols[1]], 'psnr')} | {mean_sd(V[cols[2]], 'nmse')} |")


LABEL_MAIN = {"unet": "U-Net only", "gru": "ETER-net", "ss2d": "SS2D<br>(controlled)"}   # 주 표 1 표시 라벨(<br> = 칸 안 줄바꿈)


def model_row_main(s, R, cols):
    V = R["V"][s]
    return (f"| {LABEL_MAIN[s]} | {mean_sd(V[cols[0]], 'ssim')} | {mean_sd(V[cols[1]], 'psnr')} | "
            f"{mean_sd(V[cols[2]], 'nmse')} |")


def caption_common(R):
    # 2026-10-17 1차 수정(outline_v9 §6 분량 후보 4, UC-009): 슬라이스 수(8,852)는 본문 2.3에 있으므로 캡션에서 뺀다.
    return (f"on the fastMRI brain multicoil test_full set ({R['n_vol']} volumes, R=4); "
            f"each model was trained once (seed 1), and its checkpoint was selected on the official validation split "
            f"({thousands(R['val']['unet']['n_vol'])} volumes)")


NOTE_UNET = ("U-Net only는 시퀀스 모듈의 출력 20채널을 0으로 대체하고, 다른 모델과 같은 구조의 U-Net을 처음부터 학습한 모델이다.")
NOTE_ONCE = ("각 모델은 난수 시드를 1로 고정하여 50 epochs 동안 한 번만 학습하였으므로, 표의 수치는 모델별로 한 번의 학습 결과를 나타낸다.")
# 2026-10-17 check 응답: '표의' 를 빼서 본문에만 있는 평균(공식 정의 SSIM 등)끼리 뺀 값에도 이 주석이 적용되게 한다.
NOTE_ROUND = ("본문에 인용한 차이는 같은 볼륨끼리 짝지은 차이의 평균(반올림 전 값)이므로, 반올림한 평균끼리 뺀 값과 "
              "마지막 자리에서 다를 수 있다.")
# (c) 주 블록 주석의 공식 정의 SSIM 한 문장 — outline_v9 §4 (c): 선택, Word 실측 여유가 있을 때만 True 로 바꾼다(약 +10 pt).
NOTE_OFFICIAL_IN_MAIN = False
# 2026-10-18 감축: 주 표 1 주석(짧은 판). '표의' 는 넣지 않는다 — 본문에만 있는 평균(공식 정의 SSIM 등)에도 적용.
NOTE_ROUND_MAIN = "본문의 차이는 짝지은 차이의 평균(반올림 전)이므로, 반올림한 평균끼리 뺀 값과 마지막 자리가 다를 수 있다."


def note_official(R):
    v = [f"{R['V'][s]['ssim_off'].mean():.4f}" for s, _, _ in MODELS]
    return (f"fastMRI 공식 정의로 계산한 SSIM은 {NAME['unet']}, {NAME['gru']}, {NAME['ss2d']}가 각각 "
            f"{v[0]}, {v[1]}, {v[2]}이다(3.1절).")


# 2026-10-17 2차 수정(내부 심사 B-08(a)): 본문은 세 모델을 '모델'로 부르고 SS2D 는 제안 방법이 아니므로 열 머리를 Model 로.
HEADER = "| Model | Params (M) | SSIM | PSNR (dB) | nMSE (%) |\n|---|---|---|---|---|"
HEADER_MAIN = "| Model | SSIM | PSNR (dB) | nMSE (%) |\n|---|---|---|---|"
MASK_COLS = ("ssim_mask", "psnr_mask", "nmse_mask")
OFF_COLS = ("ssim_off", "psnr_off", "nmse_off")
GROUP_MASK = "**Brain mask**"
GROUP_OFF = "**fastMRI official**"


def block_main(R):
    lines = ["%% IEIE .src.md 붙여넣기용 — v9 표 1 (v8fix 50 epochs, test_full, 볼륨 단위, brain mask 내부, 순위 표시 없음, "
             "Zero-filled 행 없음 — 사용자 결정 D2). paper/make_tables_v8fix.py 가 생성 — 손편집 금지",
             "@table: col | " + ",".join(map(str, WIDTHS_MAIN)),
             f"@cap_en: Volume-level results (mean ± standard deviation) on the fastMRI brain test_full split "
             f"({R['n_vol']} volumes, R=4, brain-masked)",
             HEADER_MAIN]
    lines += [model_row_main(s, R, MASK_COLS) for s, _, _ in MODELS]
    note = NOTE_ROUND_MAIN + (" " + note_official(R) if NOTE_OFFICIAL_IN_MAIN else "")
    lines.append("@note: Note. " + note)
    lines.append("@end")
    return "\n".join(lines) + "\n"


def block_official(R):
    lines = ["%% IEIE .src.md 붙여넣기용 — v9 표 1 대안 (brain mask 행 묶음 + fastMRI 공식 정의 행 묶음, 같은 세 모델·같은 5열). "
             "paper/make_tables_v8fix.py 가 생성 — 손편집 금지",
             "@table: col | " + ",".join(map(str, WIDTHS)),
             "@cap_en: Volume-level results (mean ± standard deviation) computed inside the brain mask and with the "
             "fastMRI official definition " + caption_common(R),
             HEADER, f"| {GROUP_MASK} | | | | |"]
    lines += [model_row(s, R, MASK_COLS) for s, _, _ in MODELS]
    lines.append(f"| {GROUP_OFF} | | | | |")
    lines += [model_row(s, R, OFF_COLS) for s, _, _ in MODELS]
    lines.append(
        "@note: Note. Brain mask 행의 지표는 brain mask 내부에서 계산하였다. fastMRI official 행의 지표는 fastMRI 공식 평가 정의에 따라 "
        "마스크를 쓰지 않고, 참조 영상과 복원 영상을 참조 영상의 가로 크기를 한 변으로 하는 정사각형으로 가운데를 잘라 계산하였으며, "
        "동적 범위(data range)는 참조 볼륨의 최댓값으로 두었다. 이 정의에서 SSIM은 슬라이스별 값을 볼륨마다 평균한 값이고, "
        # 2026-10-17 2차 수정(내부 심사 A-m2): brain mask 는 두피를 포함한 머리 영역이므로 '뇌 바깥의 배경' 대신 '머리 밖 배경'.
        "PSNR과 nMSE는 볼륨 전체에서 한 번에 계산한 값이다. 공식 정의는 머리 밖 배경을 포함하는데, 학습 손실은 brain mask "
        "내부에서만 계산하였으므로 어느 모델도 이 배경을 학습하지 않았다. " + NOTE_UNET + " " + NOTE_ONCE + " " + NOTE_ROUND)
    lines.append("@end")
    return "\n".join(lines) + "\n"


def pair_label(a, b):
    return f"{NAME[a]} {MINUS} {NAME[b]}"


def paired_md(R, inputs_md5):
    o = ["# IEIE v9 — paired comparisons, v8fix test_full (generated by paper/make_tables_v8fix.py; do not edit by hand)", "",
         f"- Data: fastMRI brain multicoil test_full, {R['n_vol']} volumes / {thousands(R['n_slice'])} slices, R=4. "
         "Each model trained once (seed 1); best checkpoint selected on the official validation split (validation batches 0–2, "
         f"{thousands(R['val']['unet']['n_vol'])} volumes / {thousands(R['val']['unet']['n_slice'])} slices).",
         "- Reference model = ETER-net. Primary pairs: SS2D(controlled) − ETER-net, U-Net only − ETER-net. Secondary: "
         "SS2D(controlled) − U-Net only.",
         "- Δ = A − B per volume (mean of paired per-volume differences). \"A better\" = A higher for SSIM/PSNR, lower for "
         "nMSE; ties are not counted. p = volume-level two-sided Wilcoxon signed-rank test "
         "(`scipy.stats.wilcoxon(Δ)`, same call as `v8_eter_pure/eval_v8fix_official_compare.py`).",
         "- Slice-level better counts use the per-slice CSV (8,852 slices) for metrics that exist per slice; official "
         "PSNR/nMSE are volume-pooled and the official SSIM map inside/outside the brain exists only per volume (n/a).",
         "- nMSE differences are in % (fraction × 100). Each model was trained once, so p reflects variation across test "
         "volumes only, not across training runs.", ""]
    for pname, a, b, role in PAIRS:
        P = R["pairs"][pname]
        o += [f"## {pair_label(a, b)} ({role})", "",
              f"| Metric | {NAME[a]} | {NAME[b]} | Δ mean (unrounded) | Δ mean (rounded) | {NAME[a]} better, volumes "
              f"| {NAME[a]} better, slices | Wilcoxon p (volume) |",
              "|---|---|---|---|---|---|---|---|"]
        for tok, vk, sk, sgn, kind, desc in METRICS:
            r = P[tok]
            sc, nd = SCALE[kind], NDEC[kind]
            sl = (f"{thousands(r['slice_better'])}/{thousands(r['slice_n'])} ({100 * r['slice_better'] / r['slice_n']:.1f}%)"
                  if r["slice_better"] is not None else "n/a")
            o.append(f"| {desc} | {sc * r['a_mean']:.{nd}f}±{sc * r['a_sd']:.{nd}f} | {sc * r['b_mean']:.{nd}f}±{sc * r['b_sd']:.{nd}f} "
                     f"| {signed(sc * r['mean'], nd + 4)} | {signed(sc * r['mean'], nd)} "
                     f"| {r['better']}/{r['n']} ({100 * r['better'] / r['n']:.1f}%) | {sl} | {r['p']:.2g} ({fmt_p(r['p'])}) |")
        d = P["_decomp"]
        o += ["", f"Approximate decomposition of Δ official SSIM (frac_in = mean fraction of brain-mask pixels in the official "
              f"crop, 3-px border excluded = {d['frac_in']:.3f}): inside {d['frac_in']:.3f} × {signed(d['din'], 5)} = "
              f"{signed(d['t_in'], 5)}, outside {1 - d['frac_in']:.3f} × {signed(d['dout'], 5)} = {signed(d['t_out'], 5)}; "
              f"sum {signed(d['t_in'] + d['t_out'], 5)} vs. Δ official SSIM {signed(d['total'], 5)} "
              f"(residual {signed(d['total'] - d['t_in'] - d['t_out'], 5)}: the exact identity holds per slice, but frac_in and the inside/outside means are averaged separately over slices and volumes).", ""]
    o += ["## Inputs (md5)", ""] + [f"- `{k}` {v}" for k, v in inputs_md5.items()] + [""]
    return "\n".join(o)


def tokens(R):
    T = {}

    def put(name, value, raw, desc):
        assert name not in T, name
        T[name] = dict(value=value, raw=raw, desc=desc)

    put("N_TEST_VOL", str(R["n_vol"]), R["n_vol"], "test_full volumes")
    put("N_TEST_SLICE", thousands(R["n_slice"]), R["n_slice"], "test_full slices")
    put("N_VAL_VOL", thousands(R["val"]["unet"]["n_vol"]), R["val"]["unet"]["n_vol"], "selection set: validation batches 0–2, volumes")
    put("N_VAL_SLICE", thousands(R["val"]["unet"]["n_slice"]), R["val"]["unet"]["n_slice"], "selection set slices")
    for s, t, n in MODELS:
        put(f"PARAMS_{t}", R["params"][s], float(R["params"][s]), f"{n} parameters (M), from src-dir {s}.log")
        bv = R["done"][s]["best_val"]
        put(f"BESTEP_{t}", str(bv["epoch"]), int(bv["epoch"]), f"{n} best epoch (DONE)")
        put(f"VALSSIM_{t}", f"{bv['ssim']:.4f}", float(bv["ssim"]),
            f"{n} best validation SSIM — SELECTION SET (val batches 0–2, slice mean as logged by the trainer), not a test value")
    for s, t, n in MODELS:
        V = R["V"][s]
        for tok, vk, sk, sgn, kind, desc in METRICS:
            sc, nd = SCALE[kind], NDEC[kind]
            m, sd = float(V[vk].mean()), float(V[vk].std(ddof=1))
            put(f"{tok}_{t}", f"{sc * m:.{nd}f}", sc * m, f"{n}: {desc}, volume-level mean")
            put(f"{tok}_{t}_SD", f"{sc * sd:.{nd}f}", sc * sd, f"{n}: {desc}, volume-level SD (ddof=1)")
    put("FRAC_IN", f"{R['frac_in']:.3f}", R["frac_in"], "mean fraction of brain-mask pixels in the official crop (3-px border excluded)")
    for pname, a, b, role in PAIRS:
        P = R["pairs"][pname]
        lab = pair_label(a, b)
        for tok, vk, sk, sgn, kind, desc in METRICS:
            r = P[tok]
            sc, nd = SCALE[kind], NDEC[kind]
            put(f"D_{pname}_{tok}", signed(sc * r["mean"], nd), sc * r["mean"], f"{lab}: mean paired difference, {desc}")
            put(f"D_{pname}_{tok}_ABS", f"{abs(sc * r['mean']):.{nd}f}", abs(sc * r["mean"]), f"absolute value of ({lab}), {desc}")
            put(f"WIN_{pname}_{tok}_VOL", f"{r['better']}/{r['n']} ({100 * r['better'] / r['n']:.1f}%)", r["better"],
                f"volumes where {NAME[a]} is better, {desc}")
            put(f"WIN_{pname}_{tok}_VOLPCT", f"{100 * r['better'] / r['n']:.1f}%", 100 * r["better"] / r["n"],
                f"% of volumes where {NAME[a]} is better, {desc}")
            if r["slice_better"] is not None:
                put(f"WIN_{pname}_{tok}_SLICE",
                    f"{thousands(r['slice_better'])}/{thousands(r['slice_n'])} ({100 * r['slice_better'] / r['slice_n']:.1f}%)",
                    r["slice_better"], f"slices where {NAME[a]} is better, {desc}")
                put(f"WIN_{pname}_{tok}_SLICEPCT", f"{100 * r['slice_better'] / r['slice_n']:.1f}%",
                    100 * r["slice_better"] / r["slice_n"], f"% of slices where {NAME[a]} is better, {desc}")
            put(f"P_{pname}_{tok}", fmt_p(r["p"]), r["p"], f"{lab}: volume-level two-sided Wilcoxon p, {desc}")
        r = P["SSIM"]
        put(f"WIN_{pname}_VOL", f"{r['better']}/{r['n']} ({100 * r['better'] / r['n']:.1f}%)", r["better"],
            f"alias of WIN_{pname}_SSIM_VOL (mask SSIM)")
        put(f"WIN_{pname}_SLICE", f"{thousands(r['slice_better'])}/{thousands(r['slice_n'])} "
            f"({100 * r['slice_better'] / r['slice_n']:.1f}%)", r["slice_better"], f"alias of WIN_{pname}_SSIM_SLICE (mask SSIM)")
        d = P["_decomp"]
        put(f"DECOMP_{pname}_IN", signed(d["t_in"], 4), d["t_in"], f"{lab}: approximate decomposition — FRAC_IN × Δ official SSIM map inside the brain")
        put(f"DECOMP_{pname}_OUT", signed(d["t_out"], 4), d["t_out"], f"{lab}: approximate decomposition — (1 − FRAC_IN) × Δ official SSIM map outside the brain")
        put(f"DECOMP_{pname}_OUTSHARE", f"{100 * d['t_out'] / d['total']:.0f}%", d["t_out"] / d["total"],
            f"{lab}: approximate share — outside term / Δ official SSIM")
    return T


def tokens_md(T):
    o = ["# IEIE v9 — prose tokens (generated by paper/make_tables_v8fix.py; do not edit by hand)", "",
         "Formatted values for the v9 draft. Raw values are in `ieie_v9_tokens.json`. Differences use U+2212 for minus; "
         "nMSE in %. VALSSIM_* are selection-set (validation) values, not test values.", "",
         "| Token | Value | Description |", "|---|---|---|"]
    o += [f"| `{k}` | {v['value']} | {v['desc']} |" for k, v in T.items()]
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------ 자가 점검
def check_a(V, S):
    """(a) per-volume = per-slice 의 파일별 평균 (슬라이스 평균으로 만든 열만)."""
    comparable = {"ssim_mask": "ssim", "psnr_mask": "psnr", "nmse_mask": "nmse", "ssim_off": "ssim_official"}
    for s, _, _ in MODELS:
        v, sl = V[s], S[s]
        check(len(v) == EXPECT["n_vol"], f"(a) {s}: per-volume 행 {len(v)} ≠ {EXPECT['n_vol']}")
        check(len(sl) == EXPECT["n_slice"], f"(a) {s}: per-slice 행 {len(sl)} ≠ {EXPECT['n_slice']}")
        order = list(dict.fromkeys(sl["file"]))
        check(order == list(v["file"]), f"(a) {s}: 볼륨 순서가 per-slice 의 파일 순서와 다름")
        g = sl.groupby("file", sort=False)
        cnt = g.size().reindex(v["file"]).to_numpy()
        check(np.array_equal(cnt, v["n"].to_numpy()), f"(a) {s}: 볼륨별 슬라이스 수 n 불일치")
        for vk, sk in comparable.items():
            mean = g[sk].mean().reindex(v["file"]).to_numpy(float)
            x = v[vk].to_numpy(float)
            err = np.abs(mean - x) / np.maximum(1.0, np.abs(x))
            check(float(err.max()) <= 1e-9, f"(a) {s}: {vk} ≠ 파일별 mean({sk}) (최대 상대 오차 {err.max():.3g}, "
                                            f"볼륨 {v['file'].iloc[int(err.argmax())]})")
    # 모델 사이 정렬
    for s in ("gru", "ss2d"):
        check(list(V[s]["file"]) == list(V["unet"]["file"]), f"(a) 볼륨 순서 {s} ≠ unet")
        check(S[s][["file", "slice_idx"]].equals(S["unet"][["file", "slice_idx"]]), f"(a) 슬라이스 순서 {s} ≠ unet")
        check(np.array_equal(V[s]["frac_in"].to_numpy(), V["unet"]["frac_in"].to_numpy()), f"(a) frac_in {s} ≠ unet")


def parse_three_model(path):
    txt = open(path, encoding="utf-8").read()
    label_key = {"마스크 SSIM (현행 주 지표)": ("ssim_mask", "ssim"), "마스크 SSIM, data_range 볼륨 단위": ("ssim_mask_volrange", "ssim"),
                 "공식 SSIM (fastMRI evaluate.py 정의)": ("ssim_off", "ssim"), "공식 SSIM map — 뇌 안": ("ssim_off_in", "ssim"),
                 "공식 SSIM map — 뇌 밖": ("ssim_off_out", "ssim"), "마스크 PSNR (dB)": ("psnr_mask", "psnr"),
                 "공식 PSNR (dB)": ("psnr_off", "psnr"), "마스크 nMSE (%)": ("nmse_mask", "nmse"), "공식 NMSE (%)": ("nmse_off", "nmse")}
    pair_key = {"ETER-net − U-Net only": ("gru", "unet"), "SS2D − U-Net only": ("ss2d", "unet"), "SS2D − ETER-net": ("ss2d", "gru")}
    cells, pairs = {}, {}
    for ln in txt.splitlines():
        if not ln.startswith("| "):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if c[0] in label_key:
            cells[label_key[c[0]]] = c[1:4]
        elif c[0] in pair_key:
            pairs[pair_key[c[0]]] = c[1:5]
    dec = {}
    for lab, a, b in (("ETER-net − U-Net", "gru", "unet"), ("SS2D − U-Net", "ss2d", "unet"), ("SS2D − ETER-net", "ss2d", "gru")):
        m = re.search(re.escape(lab) + r" 안 ([+−]\d+\.\d+) / 밖 ([+−]\d+\.\d+)", txt)
        dec[(a, b)] = m.groups() if m else None
    m = re.search(r"뇌 안 화소 비율 (\d+\.\d+)", txt)
    return cells, pairs, dec, (m.group(1) if m else None)


def check_b_c_three(R, ref):
    cells, pairs, dec, frac = parse_three_model(os.path.join(ref, "three_model_table.md"))
    check(len(cells) == 9, f"(b) three_model_table.md 지표 행 {len(cells)}/9 만 읽힘")
    for (vk, kind), row in cells.items():
        for s, got in zip(("unet", "gru", "ss2d"), row):
            mine = mean_sd(R["V"][s][vk], kind)
            check(mine == got, f"(b) {s} {vk}: 생성 {mine} ≠ three_model_table {got}")
    # 표 블록 칸이 위 값과 같은지(블록 문자열 기준)
    check(len(pairs) == 3, f"(c) three_model_table.md 짝 비교 행 {len(pairs)}/3")
    cols = ["ssim_mask", "ssim_off", "ssim_off_in", "ssim_off_out"]
    for (a, b), row in pairs.items():
        for vk, cell in zip(cols, row):
            m = re.fullmatch(r"([+−-]\d+\.\d+)(?: \((\d+)(?:, (\S+))?\))?", cell)
            check(m is not None, f"(c) three_model_table 짝 칸 해석 실패: {cell!r}")
            if not m:
                continue
            r = paired(R["V"][a], R["V"][b], vk, +1)
            check(signed(r["mean"], 4) == m.group(1).replace("-", MINUS), f"(c) {a}−{b} {vk}: Δ {signed(r['mean'], 4)} ≠ {m.group(1)}")
            if m.group(2) is not None:
                check(r["better"] == int(m.group(2)), f"(c) {a}−{b} {vk}: 나은 볼륨 {r['better']} ≠ {m.group(2)}")
            if m.group(3) is not None:
                check(f"{r['p']:.0e}" == m.group(3), f"(c) {a}−{b} {vk}: p {r['p']:.0e} ≠ {m.group(3)}")
    for (a, b), g in dec.items():
        check(g is not None, f"(c) three_model_table 분해 줄 없음 {a}−{b}")
        if g is None:
            continue
        d = paired_decomp(R, a, b)
        check((signed(d["t_in"], 4), signed(d["t_out"], 4)) == g, f"(c) 분해 {a}−{b}: {signed(d['t_in'], 4)}/{signed(d['t_out'], 4)} ≠ {g}")
    check(frac == f"{R['frac_in']:.3f}", f"(c) frac_in {R['frac_in']:.3f} ≠ three_model_table {frac}")


def paired_decomp(R, a, b):
    for pname, pa, pb, _ in PAIRS + [("ETER_UNET", "gru", "unet", "check")]:
        if (pa, pb) == (a, b):
            return R["pairs"][pname]["_decomp"]
    raise KeyError((a, b))


def check_c_compare(R, ref):
    names = {"마스크 SSIM (현행)": ("ssim_mask", 1), "마스크 SSIM, data_range 볼륨 단위": ("ssim_mask_volrange", 1),
             "공식 SSIM": ("ssim_off", 1), "공식 SSIM map — 뇌 마스크 안": ("ssim_off_in", 1),
             "공식 SSIM map — 뇌 마스크 밖": ("ssim_off_out", 1), "마스크 PSNR (dB)": ("psnr_mask", 1),
             "공식 PSNR (dB)": ("psnr_off", 1), "마스크 nMSE": ("nmse_mask", -1), "공식 NMSE": ("nmse_off", -1)}
    for fn, a, b in (("compare_gru_vs_unet.md", "gru", "unet"), ("compare_ss2d_vs_gru.md", "ss2d", "gru"),
                     ("compare_ss2d_vs_unet.md", "ss2d", "unet")):
        txt = open(os.path.join(ref, fn), encoding="utf-8").read()
        check(f"Δ = {a} − {b}" in txt, f"(c) {fn}: 방향 'Δ = {a} − {b}' 없음")
        seen = 0
        for ln in txt.splitlines():
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if not ln.startswith("| ") or c[0] not in names:
                continue
            seen += 1
            vk, sgn = names[c[0]]
            r = paired(R["V"][a], R["V"][b], vk, sgn)
            mine = [f"{r['b_mean']:.4f}±{r['b_sd']:.4f}", f"{r['a_mean']:.4f}±{r['a_sd']:.4f}", f"{r['mean']:+.5f}",
                    f"{r['better']}/{r['n']} ({100 * r['better'] / r['n']:.1f}%)", f"{r['p']:.2g}"]
            check(mine == c[1:6], f"(c) {fn} {vk}: 생성 {mine} ≠ {c[1:6]}")
        check(seen == 9, f"(c) {fn}: 지표 행 {seen}/9")
        d = paired_decomp(R, a, b)
        m = re.search(r"분해: 안 (\S+)×(\S+) = (\S+), 밖 (\S+)×(\S+) = (\S+)$", txt, re.M)
        check(m is not None, f"(c) {fn}: 분해 줄 없음")
        if m:
            mine = (f"{d['frac_in']:.3f}", f"{d['din']:+.5f}", f"{d['t_in']:+.5f}", f"{1 - d['frac_in']:.3f}",
                    f"{d['dout']:+.5f}", f"{d['t_out']:+.5f}")
            check(mine == m.groups(), f"(c) {fn} 분해: 생성 {mine} ≠ {m.groups()}")
    # 주 짝(U-Net only − ETER-net)은 compare_gru_vs_unet.md 의 반대 방향 — 부호 반전·같은 p·나은 수 합
    F, B = R["pairs"]["UNET_ETER"], R["pairs"]["ETER_UNET"]
    for tok, *_ in METRICS:
        f, b = F[tok], B[tok]
        check(abs(f["mean"] + b["mean"]) <= 1e-15 * max(1.0, abs(f["mean"])), f"(c) UNET_ETER {tok}: Δ 가 ETER_UNET 의 부호 반전이 아님")
        check(f["better"] + b["better"] + f["ties"] == f["n"], f"(c) UNET_ETER {tok}: 나은 수 합 ≠ n")
        check(abs(f["p"] - b["p"]) <= 1e-12 * max(abs(b["p"]), 1e-300), f"(c) UNET_ETER {tok}: p {f['p']} ≠ 반대 방향 p {b['p']}")


def check_d(R, ref):
    for s, _, _ in MODELS:
        J = json.load(open(os.path.join(ref, f"summary_{s}.json")))
        check(J["n_vol"] == R["n_vol"] and J["n_slice"] == R["n_slice"], f"(d) {s}: n_vol/n_slice ≠ summary json")
        for k, ms in J["metrics"].items():
            v = R["V"][s][k].to_numpy(float)
            m, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
            check(abs(m - ms["mean"]) <= 1e-9, f"(d) {s} {k}: 평균 {m!r} ≠ summary {ms['mean']!r}")
            check(abs(sd - ms["sd"]) <= 1e-9, f"(d) {s} {k}: SD {sd!r} ≠ summary {ms['sd']!r}")


def check_meta(R, src):
    for s in RUN:
        check(R["params"][s] == EXPECT["params"][s], f"(meta) {s} params {R['params'][s]} ≠ 기댓값 {EXPECT['params'][s]}")
        bv = R["done"][s]["best_val"]
        check(bv["epoch"] == EXPECT["bestep"][s], f"(meta) {s} best epoch {bv['epoch']} ≠ {EXPECT['bestep'][s]}")
        check(R["done"][s]["epochs"] == 50, f"(meta) {s} DONE epochs ≠ 50")
        val = R["val"][s]
        check((val["n_vol"], val["n_slice"]) == (EXPECT["n_val_vol"], EXPECT["n_val_slice"]),
              f"(meta) {s} 검증 CSV {val['n_vol']}/{val['n_slice']} ≠ 1378/21842")
        check(f"{val['ssim_mean']:.4f}" == f"{bv['ssim']:.4f}",
              f"(meta) {s} 검증 CSV 슬라이스 평균 SSIM {val['ssim_mean']:.4f} ≠ DONE {bv['ssim']:.4f}")
    for s, p in DONE_COPY.items():
        check(os.path.exists(p) and md5(p) == md5(DONE[s]), f"(meta) DONE 사본 불일치: {p}")
    with open(GRU_TRAIN_LOG, encoding="utf-8", errors="replace") as f:
        head = "".join(next(f) for _ in range(80))
    m = re.findall(r"Loaded (\d+) \.h5 files\s+→\s+(\d+) slices total", head)
    check(len(m) >= 2 and m[1] == (str(EXPECT["n_val_vol"]), str(EXPECT["n_val_slice"])),
          f"(meta) GRU 학습 로그의 검증 데이터 줄 {m}")
    for s in RUN:
        J = json.load(open(os.path.join(src, f"summary_{s}.json"))) if os.path.exists(os.path.join(src, f"summary_{s}.json")) else None
        if J is not None:
            check(J["ckpt"].endswith(f"pure_{s}_best.pt"), f"(meta) {s} summary ckpt 가 best 가 아님: {J['ckpt']}")


def check_e(texts):
    from importlib import import_module
    sys.path.insert(0, os.path.join(ROOT, "paper", "ieie"))
    B = import_module("build_ieie_docx")
    expected_first = {"ieie_table1_block_v9.md": [LABEL_MAIN[s] for s, _, _ in MODELS],
                      "ieie_table1_block_v9_official.md": [GROUP_MASK] + [NAME[s] for s, _, _ in MODELS]
                      + [GROUP_OFF] + [NAME[s] for s, _, _ in MODELS]}
    for fn, first in expected_first.items():
        t = texts[fn]
        lines = t.rstrip("\n").split("\n")
        tl = [ln for ln in lines if ln.startswith("@table:")]
        check(len(tl) == 1, f"(e) {fn}: @table 줄 {len(tl)}")
        w = [int(x) for x in tl[0].split("|")[1].split(",")] if tl else []
        check(sum(w) <= COL_W_CONF, f"(e) {fn}: 열 폭 합 {sum(w)} > {COL_W_CONF}")
        rows = [ln for ln in lines if ln.startswith("|")]
        ncols = [len(ln.strip().strip("|").split("|")) for ln in rows]
        check(len(set(ncols)) == 1 and ncols[0] == len(w), f"(e) {fn}: 열 수 {set(ncols)} ≠ 폭 {len(w)}")
        check(sum(ln == "@end" for ln in lines) == 1 and lines[-1] == "@end", f"(e) {fn}: @end 가 하나·마지막이 아님")
        notes = [ln for ln in lines if ln.startswith("@note:")]
        check(len(notes) == 1 and notes[0].startswith("@note: Note. "), f"(e) {fn}: @note 줄 {len(notes)}")
        check(sum(ln.startswith("@cap_en:") for ln in lines) == 1, f"(e) {fn}: @cap_en 이 한 줄이 아님")
        check(lines[0].startswith("%%"), f"(e) {fn}: 첫 줄이 %% 주석이 아님")
        body = [ln.strip().strip("|").split("|") for ln in rows[2:]]
        check([c[0].strip() for c in body] == first, f"(e) {fn}: 행 이름 {[c[0].strip() for c in body]} ≠ {first}")
        for c in body:
            for cell in c[1:]:
                check("**" not in cell and "__" not in cell, f"(e) {fn}: 수치 칸에 순위 표시 {cell!r}")
        check(not any("Zero-filled" in ln for ln in rows), f"(e) {fn}: Zero-filled 행이 있음")
        # 실제 빌더 파서로 읽기
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "probe.src.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write("@title_ko: t\n@body:\n" + t)
            doc = B.parse_src(p)
        tb = [b for b in doc.blocks if b["kind"] == "table"]
        check(len(tb) == 1, f"(e) {fn}: 빌더 파서가 읽은 표 {len(tb)}")
        if tb:
            tb = tb[0]
            check(tb["width"] == "col" and tb["colw"] == w, f"(e) {fn}: 빌더 폭 {tb['width']} {tb['colw']}")
            check(len(tb["rows"]) == len(rows) - 1 and all(len(r) == len(w) for r in tb["rows"]),
                  f"(e) {fn}: 빌더 행 {len(tb['rows'])}")
            check(tb["note"].startswith("Note. ") and tb["cap_en"], f"(e) {fn}: 빌더 note/cap_en 비어 있음")
            for r in tb["rows"]:
                if r[0].startswith("**"):
                    check(B.MARK_RE.fullmatch(r[0]) is not None, f"(e) {fn}: 묶음 라벨 굵게 마크업 해석 실패 {r[0]}")
    for fn, t in texts.items():
        check("[TBD" not in t, f"(e) {fn}: [TBD 잔존")
        for old in FORBIDDEN_OLD:
            hit = re.search(r"(?<![\d.])" + re.escape(old) + r"(?!\d)", t)
            check(hit is None, f"(e) {fn}: 옛(붕괴 런) 수치 {old} 포함")
        check(re.search(r"\barms?\b|팔", t, re.I) is None, f"(e) {fn}: 금지 용어 arm/팔")
        check(re.search(r"composite", t, re.I) is None, f"(e) {fn}: composite 언급")
        check(re.search(r"SS2D \(controlled\)|U-Net-only|U-Net단독|ETER-Net|ETERnet", t) is None, f"(e) {fn}: 모델 이름 변형")
        check("proposed" not in t.lower() and "제안" not in t, f"(e) {fn}: '제안 방법' 표현")


# ------------------------------------------------------------------ (d) 추가 토큰 — 적재·토큰·점검
EPOCH_LINE_RE = re.compile(r"^Epoch (\d+)/(\d+)\s+(.*)$", re.M)
LOADED_RE = re.compile(r"Loaded (\d+) \.h5 files\s+→\s+(\d+) slices total")


def parse_kv(s):
    out = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def ckpt_numel(path):
    """state_dict 의 최상위 이름별 원소 수 (mmap — 텐서 값을 읽지 않음)."""
    import torch
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    cnt = {}
    for k, v in sd.items():
        cnt[k.split(".")[0]] = cnt.get(k.split(".")[0], 0) + int(v.numel())
    return cnt


def head_lines(path, n):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, ln in enumerate(f):
            if i >= n:
                break
            out.append(ln)
    return "".join(out)


def load_extra(V):
    X = {"numel": {s: ckpt_numel(CKPT_BEST[s]) for s in RUN}}
    X["recipe"] = {s: json.load(open(RECIPE[s])) for s in RUN}
    X["epoch"], X["epoch_lines"], X["header"] = {}, {}, {}
    for s in RUN:
        txt = open(LOGTXT[s], encoding="utf-8").read()
        lines = [m for m in EPOCH_LINE_RE.finditer(txt) if int(m.group(1)) == FINAL_EPOCH]
        X["epoch_lines"][s] = [m.group(3).strip() for m in lines]
        X["epoch"][s] = parse_kv(lines[-1].group(3)) if lines else {}
        hdr = [ln for ln in txt.splitlines() if ln.startswith("SCRATCH START")]
        X["header"][s] = parse_kv(hdr[0]) if hdr else {}
    X["func"] = json.load(open(FUNC_JSON))
    X["train_loaded"] = {s: LOADED_RE.findall(head_lines(p, 200)) for s, p in TRAIN_LOG.items()}
    X["contrast"] = [f.split("_")[2] for f in V["unet"]["file"]]
    return X


def gru_numel(n_in, n_hid):
    """torch.nn.GRU(bidirectional=True, 1층) 원소 수: 게이트 3개 × (W_ih + W_hh + b_ih + b_hh) × 방향 2."""
    return 3 * (n_in * n_hid + n_hid * n_hid + 2 * n_hid) * 2


def tokens_extra(T, R, X):
    def put(name, value, raw, desc):
        assert name not in T, name
        T[name] = dict(value=value, raw=raw, desc=desc)

    ck = "state_dict element count of logs/runpod_mirror/logs/PureETER_{RUN}_noDC_R4_brain384_v8fix_s1_fullval/pure_{s}_best.pt"
    g = X["numel"]["gru"]
    nmod = g["gru_h"] + g["gru_v"]
    put("PARAMS_GRUMOD", f"{nmod / 1e6:.1f}", nmod, "bi-GRU module of ETER-net (gru_h + gru_v), parameters (M) — "
        + ck.format(RUN="GRU", s="gru"))
    put("PARAMS_GRUMOD_EXACT", thousands(nmod), nmod, f"bi-GRU module, exact parameter count (gru_h {thousands(g['gru_h'])} + "
        f"gru_v {thousands(g['gru_v'])})")
    nss = X["numel"]["ss2d"]["ss2d"]
    put("PARAMS_SS2DMOD", f"{nss / 1e6:.2f}", nss, "SS2D module (ss2d.*), parameters (M) — " + ck.format(RUN="SS2D", s="ss2d"))
    put("PARAMS_SS2DMOD_EXACT", thousands(nss), nss, "SS2D module, exact parameter count")
    nu = X["numel"]["unet"]["unet"]
    put("PARAMS_UNETBODY", f"{nu / 1e6:.1f}", nu, f"U-Net (unet.*), parameters (M); identical in the three models ({thousands(nu)})")
    # 2026-10-17 check 응답: 초록의 '약 1/21' 을 토큰으로 추적(로그 params(M) 비, 정수로 반올림)
    pe = float(R["params"]["gru"])
    for s, t, n in MODELS:
        if s == "gru":
            continue
        r = pe / float(R["params"][s])
        put(f"PARAMS_RATIO_{t}", f"1/{round(r)}", r, f"{n} parameters relative to ETER-net, 1/(PARAMS_ETER/PARAMS_{t}) "
            f"rounded to an integer ({R['params']['gru']}/{R['params'][s]} = {r:.2f})")

    F = X["func"]
    sc = F["tests"]["shuffle_cross"]
    fsrc = "logs/runpod_mirror_pod3/results/v8fix_s1_fullval_test_full/functional_ss2d.json"
    put("FUNC_N", str(F["n"]), F["n"], f"SS2D functional test: number of test_full slices (evenly spaced; {fsrc} n)")
    # 2026-10-17 최종 검증 응답: 모듈 출력은 LayerNorm·선형·1×1 합성곱을 거친 학습된 특징이라 세기 단위가 아니다(ss2d.py norm_in·out_proj)
    put("FUNC_RMS", f"{F['seq_out_rms_mean']:.4f}", F["seq_out_rms_mean"],
        f"SS2D functional test: RMS of the SS2D module output as fed to the U-Net (20 channels; learned feature after "
        f"LayerNorm/Linear/1x1 conv, no intensity unit), mean over slices ({fsrc} seq_out_rms_mean)")
    put("FUNC_SHUFFLE_DROP", f"{sc['mean']:.5f}", sc["mean"], "SS2D functional test: mean SSIM drop (normal − test) when the SS2D "
        f"module output is replaced with that of a slice from another volume ({fsrc} tests.shuffle_cross.mean)")
    put("FUNC_SHUFFLE_DROP_SE", f"{sc['se']:.5f}", sc["se"], "standard error of FUNC_SHUFFLE_DROP (tests.shuffle_cross.se)")
    put("FUNC_SHUFFLE_NDROP", f"{sc['n_pos']}/{sc['n']} ({100 * sc['n_pos'] / sc['n']:.1f}%)", sc["n_pos"],
        "slices whose SSIM dropped in the shuffle_cross test (tests.shuffle_cross.n_pos / n)")
    put("FUNC_SHUFFLE_P", fmt_p(sc["p_sign"]), sc["p_sign"], "shuffle_cross sign-test p (tests.shuffle_cross.p_sign)")
    put("FUNC_SHUFFLE_VERDICT", F["verdict"]["shuffle_cross"], F["verdict"]["shuffle_cross"],
        f"shuffle_cross verdict under the pre-fixed criterion '{F['criteria']}'")

    E = X["epoch"]
    lsrc = "logs/runpod_mirror/logs/PureETER_{}_noDC_R4_brain384_v8fix_s1_fullval/log.txt 'Epoch 50/50' line"
    base = float(E["unet"]["train_loss"])
    for s, t, n in MODELS:
        put(f"TRAINLOSS_{t}", E[s]["train_loss"], float(E[s]["train_loss"]),
            f"{n}: training loss, mean over epoch 50 (as logged, 5 decimals; {lsrc.format(RUN[s])} train_loss)")
    for s, t, n in MODELS[1:]:
        v = float(E[s]["train_loss"])
        gap = 100 * (v - base) / base
        lo = 100 * ((v - 5e-6) - (base + 5e-6)) / (base + 5e-6)
        hi = 100 * ((v + 5e-6) - (base - 5e-6)) / (base - 5e-6)
        put(f"TRAINLOSS_GAP_{t}", f"{gap:+.1f}%".replace("-", MINUS), gap,
            f"{n} − U-Net only: epoch-50 training loss gap relative to U-Net only (from the logged 5-decimal values; "
            f"log rounding allows {lo:.2f}–{hi:.2f}%)")
    for s, t, n in MODELS:
        put(f"L1_OVER_SSIM_{t}", E[s]["l1_over_ssim_term"], float(E[s]["l1_over_ssim_term"]),
            f"{n}: first loss term / (λ·SSIM term), epoch-50 means ({lsrc.format(RUN[s])} l1_over_ssim_term)")
    r = [float(E[s]["l1_over_ssim_term"]) for s in RUN]
    put("L1_OVER_SSIM", f"{min(r):.1f}–{max(r):.1f}", [min(r), max(r)], "range of L1_OVER_SSIM_* over the three models (epoch 50)")
    put("L1_OVER_SSIM_APPROX", f"{np.mean(r):.0f}", float(np.mean(r)), "L1_OVER_SSIM rounded for '약 N배' (mean of the three, epoch 50)")

    tl = X["train_loaded"]["gru"][0]
    put("N_TRAIN_VOL", thousands(int(tl[0])), int(tl[0]), "training volumes (official train split as loaded on the pod; first "
        "'Loaded … slices total' line of the GRU and SS2D training logs)")
    put("N_TRAIN_SLICE", thousands(int(tl[1])), int(tl[1]), "training slices (same line)")

    nax, nv = sum(c == "AXT2" for c in X["contrast"]), len(X["contrast"])
    put("TEST_AXT2", f"{nax}/{nv} ({100 * nax / nv:.1f}%)", nax, "test_full volumes that are AXT2 (contrast in the file name "
        "file_brain_<contrast>_…; src-dir per_volume_unet.csv)")
    put("TEST_AXT2_PCT", f"{100 * nax / nv:.1f}%", 100 * nax / nv, "% of test_full volumes that are AXT2")
    put("TEST_AXT2_PCT0", f"{100 * nax / nv:.0f}%", 100 * nax / nv, "TEST_AXT2_PCT rounded to an integer ('약 N%')")

    rc = X["recipe"]["unet"]
    lam, sbar = float(rc["lambda_ssim"]), float(rc["s_bar"]["mean"])
    sbar_o = sbar * 1e-6
    put("LAMBDA_SSIM", f"{lam:.7f}", lam, "λ_SSIM = lambda_base / s̄ (recipe.json lambda_ssim; identical in the three runs)")
    e = int(np.floor(np.log10(lam)))
    put("LAMBDA_SSIM_SCI", f"{lam / 10 ** e:.1f}\\times10^{{{e}}}", lam, "LAMBDA_SSIM in inline-math form ($…$)")
    put("SBAR", f"{sbar:.2f}", sbar, "s̄ in the loader's ×1e6 intensity units (recipe.json s_bar.mean)")
    eo = int(np.floor(np.log10(sbar_o)))
    put("SBAR_ORIG_SCI", f"{sbar_o / 10 ** eo:.2f}\\times10^{{{eo}}}", sbar_o,
        "s̄ in original intensity units (= SBAR × 1e-6), inline-math form; λ = 10^-6 / s̄ in these units")
    put("SBAR_N", str(rc["s_bar"]["n"]), int(rc["s_bar"]["n"]), "number of training slices used to compute s̄ (recipe.json s_bar.n)")


def check_extra(R, X, src):
    nm = X["numel"]
    check(set(nm["unet"]) == {"unet"}, f"(x) U-Net only 체크포인트 최상위 이름 {sorted(nm['unet'])}")
    check(set(nm["gru"]) == {"gru_h", "gru_v", "unet"}, f"(x) ETER-net 체크포인트 최상위 이름 {sorted(nm['gru'])}")
    check(set(nm["ss2d"]) == {"ss2d", "unet"}, f"(x) SS2D 체크포인트 최상위 이름 {sorted(nm['ss2d'])}")
    check(len({nm[s].get("unet") for s in RUN}) == 1, f"(x) U-Net 원소 수가 세 모델에서 다름 {[nm[s].get('unet') for s in RUN]}")
    hid = GRU_H * GRU_N
    check(nm["gru"].get("gru_h") == gru_numel(GRU_C * GRU_N, hid), f"(x) gru_h {nm['gru'].get('gru_h')} ≠ 식 {gru_numel(GRU_C * GRU_N, hid)}")
    check(nm["gru"].get("gru_v") == gru_numel(2 * hid, hid), f"(x) gru_v {nm['gru'].get('gru_v')} ≠ 식 {gru_numel(2 * hid, hid)}")
    for s in RUN:
        tot = sum(nm[s].values())
        check(f"{tot / 1e6:.1f}" == R["params"][s], f"(x) {s} 체크포인트 합계 {tot / 1e6:.1f}M ≠ eval 로그 {R['params'][s]}M")
        check(X["header"][s].get("params") == f"{tot / 1e6:.1f}M", f"(x) {s} log.txt 머리 params {X['header'][s].get('params')}")
    # 기능 시험
    F = X["func"]
    sc = F["tests"]["shuffle_cross"]
    nrm = np.asarray(F["per_slice_ssim"]["normal"], float)
    shf = np.asarray(F["per_slice_ssim"]["shuffle_cross"], float)
    idx = np.asarray(F["indices"], int)
    check(F["n"] == len(idx) == len(set(idx.tolist())) == len(nrm) == len(shf) == sc["n"], "(x) 기능 시험 n·인덱스 길이 불일치")
    d = nrm - shf
    check(abs(d.mean() - sc["mean"]) <= 1e-12, f"(x) shuffle_cross 평균 재계산 {d.mean()} ≠ {sc['mean']}")
    check(abs(d.std(ddof=1) / np.sqrt(len(d)) - sc["se"]) <= 1e-12, "(x) shuffle_cross SE 재계산 불일치")
    check(int((d > 0).sum()) == sc["n_pos"] and int((d < 0).sum()) == sc["n_neg"], "(x) shuffle_cross 하락 수 재계산 불일치")
    check(abs(nrm.mean() - F["normal"]["ssim"]) <= 1e-12, "(x) 기능 시험 정상 SSIM 평균 불일치")
    check(F["ckpt"].endswith("pure_ss2d_best.pt"), f"(x) 기능 시험 체크포인트가 best 가 아님: {F['ckpt']}")
    check(F["verdict"]["shuffle_cross"] == "used", "(x) shuffle_cross 판정 ≠ used")
    S = R["S"]["ss2d"]
    check(int(idx.max()) < len(S), "(x) 기능 시험 인덱스가 test_full 범위 밖")
    if int(idx.max()) < len(S):
        err = float(np.abs(S["ssim"].to_numpy(float)[idx] - nrm).max())
        check(err <= 1e-9, f"(x) 기능 시험 정상 SSIM ≠ src per_slice_ss2d.csv 같은 인덱스 (최대 차 {err:.3g}) — 다른 체크포인트·다른 슬라이스")
    md = open(FUNC_SUMMARY_MD, encoding="utf-8").read()
    row = f"| shuffle_cross | 다른 볼륨 슬라이스의 시퀀스 출력으로 교체 | {sc['mean']:+.5f} ± {sc['se']:.5f} | {sc['n_pos']}/{sc['n']} |"
    check(row in md, f"(x) summary_ss2d.md 에 shuffle_cross 줄 {row!r} 없음")
    for p in FUNC_JSON_COPIES:
        check(os.path.exists(p) and md5(p) == md5(FUNC_JSON), f"(x) 기능 시험 json 사본 불일치: {p}")
    # 학습 손실·L1/SSIM 항 비
    lam = float(X["recipe"]["unet"]["lambda_ssim"])
    for s in RUN:
        L = X["epoch_lines"][s]
        check(len(L) >= 1 and len(set(L)) == 1, f"(x) {s} log.txt 'Epoch {FINAL_EPOCH}/' 줄 {len(L)}개(서로 다름)")
        E = X["epoch"][s]
        if not E:
            continue
        tl, l1, ss, ratio = (float(E[k]) for k in ("train_loss", "train_l1", "train_ssim_loss", "l1_over_ssim_term"))
        check(abs(tl - (l1 + lam * ss)) <= 1.5e-5, f"(x) {s} train_loss {tl} ≠ train_l1 + λ·ssim {l1 + lam * ss:.6f}")
        check(abs(ratio - l1 / (lam * ss)) <= 0.1, f"(x) {s} l1_over_ssim_term {ratio} ≠ train_l1/(λ·ssim) {l1 / (lam * ss):.2f}")
        check(E.get("steps") == str(int(np.ceil(EXPECT["n_train_slice"] / X["recipe"][s]["batch_size"]))) and
              str(int(np.ceil(int(X["train_loaded"]["gru"][0][1]) / X["recipe"][s]["batch_size"]))) == E.get("steps"),
              f"(x) {s} epoch step 수 {E.get('steps')} ≠ ⌈학습 슬라이스/배치⌉")
        if s in WANDB_SUMMARY:
            W = json.load(open(WANDB_SUMMARY[s]))
            check(W.get("epoch") == FINAL_EPOCH, f"(x) {s} wandb 요약 epoch {W.get('epoch')}")
            for k in ("train_loss", "train_l1", "train_ssim_loss"):
                check(f"{W['epoch/' + k]:.5f}" == E[k], f"(x) {s} wandb epoch/{k} {W['epoch/' + k]:.5f} ≠ log.txt {E[k]}")
    # 학습 분할 크기
    for s, m in X["train_loaded"].items():
        check(len(m) >= 2 and m[0] == (str(EXPECT["n_train_vol"]), str(EXPECT["n_train_slice"])) and m[1] == (str(EXPECT["n_val_vol"]), str(EXPECT["n_val_slice"])),
              f"(x) {s} 학습 로그 'Loaded' 줄 {m[:2]}")
        check(re.search(rf"train 파일 수: {EXPECT['n_train_vol']}\b", head_lines(TRAIN_LOG[s], 200)) is not None, f"(x) {s} 학습 로그 'train 파일 수' 줄")
    check(X["train_loaded"]["gru"][0] == X["train_loaded"]["ss2d"][0], "(x) GRU·SS2D 학습 데이터 크기 다름")
    vl = open(VERIFY_LOG, encoding="utf-8", errors="replace").read().splitlines()
    tl = X["train_loaded"]["gru"][0]
    check(len(vl) >= 27 and LOADED_RE.search(vl[26]) is not None and LOADED_RE.search(vl[26]).groups() == tl,
          f"(x) verify.log:27 학습 분할 줄 ≠ {tl}")
    # AXT2
    P = pd.read_csv(POD_SLICE_CSV_SS2D, usecols=["file"])
    pf = list(dict.fromkeys(P["file"]))
    check(len(pf) == R["n_vol"] and sorted(pf) == sorted(R["V"]["unet"]["file"]), "(x) pod 평가 파일 목록 ≠ src per_volume 파일 목록")
    check(sum(f.split("_")[2] == "AXT2" for f in pf) == sum(c == "AXT2" for c in X["contrast"]), "(x) AXT2 수 교차 검증 불일치")
    check(all(f.startswith("file_brain_") for f in R["V"]["unet"]["file"]), "(x) 파일 이름 형식이 file_brain_<대조도>_… 가 아님")
    # λ_SSIM
    rc = X["recipe"]
    for s in RUN:
        check(rc[s]["lambda_ssim"] == rc["unet"]["lambda_ssim"] and rc[s]["s_bar"] == rc["unet"]["s_bar"]
              and rc[s]["lambda_base"] == 1.0 and rc[s]["seed"] == 1, f"(x) {s} recipe.json 이 다른 런과 다름")
        h = X["header"][s]
        check(h.get("lambda_ssim") == f"{lam:.6f}" and h.get("s_bar") == f"{rc[s]['s_bar']['mean']:.2f}",
              f"(x) {s} log.txt 머리 lambda_ssim/s_bar {h.get('lambda_ssim')}/{h.get('s_bar')}")
    sbar = float(rc["unet"]["s_bar"]["mean"])
    check(abs(lam - 1.0 / sbar) <= 1e-12 * lam, f"(x) lambda_ssim {lam} ≠ 1/s̄ {1.0 / sbar}")
    check(abs(lam - 1e-6 / (sbar * 1e-6)) <= 1e-12 * lam, "(x) λ = 10^-6/s̄_원 항등식 불일치")
    check(rc["unet"]["s_bar"]["n"] == 64, f"(x) s̄ 표본 수 {rc['unet']['s_bar']['n']} ≠ 64")


def extra_inputs_md5():
    paths = [FUNC_JSON, FUNC_SUMMARY_MD, VERIFY_LOG, POD_SLICE_CSV_SS2D] + [LOGTXT[s] for s in RUN] + [RECIPE[s] for s in RUN] \
        + list(TRAIN_LOG.values()) + list(WANDB_SUMMARY.values())
    out = {os.path.relpath(p, ROOT): md5(p) for p in paths}
    out.update({os.path.relpath(CKPT_BEST[s], ROOT): f"size {os.path.getsize(CKPT_BEST[s])} B (md5 not computed)" for s in RUN})
    return out


# ------------------------------------------------------------------ 본체
def main(argv=None):
    ap = argparse.ArgumentParser(description="IEIE v9 표·짝 비교·토큰 생성 (v8fix test_full)")
    ap.add_argument("--check-only", action="store_true", help="점검만 하고 쓰지 않음 + 디스크 출력이 최신인지 확인")
    ap.add_argument("--src-dir", default=EVAL_DIR, help="per_volume/per_slice CSV·*.log 폴더 (기본 results/eval/v8fix_official_test_full)")
    ap.add_argument("--ref-dir", default=EVAL_DIR, help="교차 검증 원천 폴더 (three_model_table.md·compare_*.md·summary_*.json)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)

    V, S, params, done, val = load(a.src_dir)
    R = compute(V, S, params, done, val)
    inputs = {os.path.relpath(os.path.join(a.src_dir, f"{k}_{s}.csv"), ROOT): md5(os.path.join(a.src_dir, f"{k}_{s}.csv"))
              for k in ("per_volume", "per_slice") for s, _, _ in MODELS}
    inputs.update({os.path.relpath(DONE[s], ROOT): md5(DONE[s]) for s in RUN})
    X = load_extra(V)
    T = tokens(R)
    tokens_extra(T, R, X)
    texts = {"ieie_table1_block_v9.md": block_main(R), "ieie_table1_block_v9_official.md": block_official(R),
             "ieie_v9_paired.md": paired_md(R, inputs), "ieie_v9_tokens.md": tokens_md(T),
             "ieie_v9_tokens.json": json.dumps(dict(generator="paper/make_tables_v8fix.py", inputs=inputs,
                                                    inputs_extra=extra_inputs_md5(), tokens=T),
                                               ensure_ascii=False, indent=1) + "\n"}

    for name, fn in (("(a) per-volume vs per-slice", lambda: check_a(V, S)),
                     ("(b,c) three_model_table.md", lambda: check_b_c_three(R, a.ref_dir)),
                     ("(c) compare_*.md", lambda: check_c_compare(R, a.ref_dir)),
                     ("(d) summary_*.json", lambda: check_d(R, a.ref_dir)),
                     ("(meta) params·DONE·val", lambda: check_meta(R, a.ref_dir)),
                     ("(x) added tokens (d)", lambda: check_extra(R, X, a.src_dir)),
                     ("(e) blocks", lambda: check_e(texts))):
        before = len(FAILS)
        try:
            fn()
        except Exception as e:     # 점검 자체가 깨져도 실패로 처리
            FAILS.append(f"{name}: 예외 {type(e).__name__}: {e}")
        print(f"  {name}: {'OK' if len(FAILS) == before else f'FAIL ({len(FAILS) - before})'}")
    # 표 칸 ↔ 토큰 일관성 (같은 계산에서 나온 두 출력)
    for s, t, _ in MODELS:
        row = [ln for ln in texts["ieie_table1_block_v9.md"].splitlines() if ln.startswith(f"| {LABEL_MAIN[s]} |")][0]
        check(f"{T[f'SSIM_{t}']['value']}±{T[f'SSIM_{t}_SD']['value']}" in row, f"(e) 표 칸 ≠ 토큰 SSIM_{t}")
    if a.check_only:
        for fn, t in texts.items():
            p = os.path.join(a.out_dir, fn)
            ok = os.path.exists(p) and open(p, encoding="utf-8").read() == t
            check(ok, f"(check-only) {p} 가 없거나 재생성 결과와 다름(손편집 또는 낡음)")

    if FAILS:
        print(f"\nSELF-CHECK FAIL — {len(FAILS)} 건 (통과 {NPASS[0]}). 아무것도 쓰지 않았다.")
        for m in FAILS[:40]:
            print("  ✗", m)
        if len(FAILS) > 40:
            print(f"  … 외 {len(FAILS) - 40} 건")
        return 1
    print(f"\nSELF-CHECK PASS — {NPASS[0]} 항목")
    if a.check_only:
        print("--check-only: 쓰지 않음 (디스크 출력 = 재생성 결과)")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    for fn, t in texts.items():
        with open(os.path.join(a.out_dir, fn), "w", encoding="utf-8") as f:
            f.write(t)
        print("  wrote", os.path.relpath(os.path.join(a.out_dir, fn), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
