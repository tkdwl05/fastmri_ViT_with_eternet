"""
v8 Pure ETER-Net — U-Net 단독 대조 모델(시퀀스 모듈 제거, f_θ ≡ 0) per-slice 평가 + 기준 CSV 대응 비교 (2026-10-01).

체크포인트 1개(PureETER_UNET, models/pure_eternet/u_pure_eternet_unet.py)를 전체 val(464 볼륨 / 7,334 슬라이스)에서
`eval_paired_v8_nodc.py` 와 **같은 슬라이스 단위 프로토콜**로 평가한다:
  - 지표 공식(masked SSIM/PSNR/NMSE/L1)·체크포인트 로드 = eval_paired_v8_nodc 의 slice_metrics / load_ckpt 를
    그대로 import (복사·수정 없음). slice_metrics 가 함께 돌려주는 체크포인트 선택용 내부 점수는 버린다(보고 제외).
  - 데이터셋 = 같은 FastMRI_H5_Dataloader(같은 경로, R=4·center 0.08 고정 마스크, random_mask=False, augment=False)
    → 같은 슬라이스 순서(idx). DataLoader(batch 1, shuffle=False) 는 num_workers 와 무관하게 순서를 보존한다.
  - 추론 = batch 1, torch.amp.autocast('cuda') (eval_paired_v8_nodc 와 동일).
출력 (--out-dir, 기본 results/eval/v8_unet_only/):
  - per_slice_unet_only.csv : idx,file,slice_idx,ssim,psnr,nmse,l1  (이미 있으면 --force 없이는 덮어쓰지 않음)
  - summary_unet_only.md    : (1) 볼륨 단위 평균±SD(ddof=1, n=464) — paper/make_tables.py 표 1 과 같은 계산
                              (2) --ref name=path:column 로 준 기준 per-slice CSV 와의 대응(paired) 비교
                                  (idx 조인 + file/slice_idx 일치 검사, 우위 슬라이스·볼륨 비율, 볼륨 단위 Wilcoxon,
                                   볼륨 단위 군집 부트스트랩 95% CI — 2,000회, seed 0)
집계 식은 paper/make_tables.py 를 따른다. 그 파일은 import 하는 순간 paper/tables/ 를 다시 쓰는 부작용이 있어
import 하지 않고 같은 식을 옮겨 적었다(각 함수 docstring 에 원 줄 번호). --summary-only 는 기존 CSV 로 요약만 다시 만든다
(예: 1b 대응 평가 CSV 가 나온 뒤 --ref 를 추가할 때 — 재추론 불필요).

--seq (2026-10-05): 같은 프로토콜로 단일 비교 모델(pixel-GRU·Transformer)도 평가한다. 기본값 unet 은 출력 파일 이름·
요약 문구가 이전과 같다. 모델 생성 인자는 main_train_pure_v8.build_model 과 같다(공유 config, use_dc=False).
"""

import os
import sys
import re
import csv
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))

# 지표·로드·config·dataset 경로 설정은 per-slice paired 평가 스크립트의 것을 그대로 쓴다
# (import 부작용 = sys.path 추가·PYTORCH_CUDA_ALLOC_CONF 설정뿐, main() 은 실행되지 않음)
from eval_paired_v8_nodc import C, slice_metrics, load_ckpt
from dataloader_h5_v5 import FastMRI_H5_Dataloader

METRICS = ['ssim', 'psnr', 'nmse', 'l1']        # CSV 열 순서 = make_tables.METRICS (부트스트랩 rng 소비 순서도 이 순서)
LOWER = {'nmse', 'l1'}
BOOT_N, BOOT_SEED = 2000, 0                      # make_tables.py:32
CM = ['ssim', 'psnr', 'nmse']                    # 표 1 의 지표 3종 (make_tables.py:263)
CM_SCALE = {'ssim': 1.0, 'psnr': 1.0, 'nmse': 100.0, 'l1': 1.0}     # nMSE → % (make_tables.py:266)
CM_FMT = {'ssim': '{:.4f}', 'psnr': '{:.2f}', 'nmse': '{:.3f}', 'l1': '{:.3f}'}   # make_tables.py:267 (+L1)
HEAD = {'ssim': 'SSIM ↑', 'psnr': 'PSNR (dB) ↑', 'nmse': 'nMSE (%) ↓', 'l1': 'L1 ↓'}
NAME_D = {'ssim': 'SSIM', 'psnr': 'PSNR (dB)', 'nmse': 'nMSE (10⁻³ %)', 'l1': 'L1'}   # Δ 표의 지표 이름
# --seq 별 이름·기본 경로 (unet = 이전과 동일). main() 에서 아래 전역을 선택한 모델의 값으로 바꾼다.
SEQ_SPECS = {
    'unet': dict(name='U-Net only', title='U-Net 단독 대조 모델 (시퀀스 모듈 제거, $f_\\theta \\equiv 0$)',
                 csv='per_slice_unet_only.csv', summary='summary_unet_only.md',
                 ckpt='logs/PureETER_UNET_noDC_R4_brain384_v8_s1_50ep/pure_unet_best.pt',
                 out_dir='results/eval/v8_unet_only'),
    'pixelgru': dict(name='pixel-GRU', title='pixel-GRU 비교 모델 (화소 단위로 행과 열을 스캔하는 가중치 공유 bi-GRU 시퀀스 모듈)',
                     csv='per_slice_pixelgru.csv', summary='summary_pixelgru.md',
                     ckpt='logs/PureETER_PIXELGRU_noDC_R4_brain384_v8_s1_50ep/pure_pixelgru_best.pt',
                     out_dir='results/eval/v8_pixelgru'),
    'transformer': dict(name='Transformer', title='Transformer 비교 모델 (axial attention 시퀀스 모듈)',
                        csv='per_slice_transformer.csv', summary='summary_transformer.md',
                        ckpt='logs/PureETER_TRANSFORMER_noDC_R4_brain384_v8_s1_50ep/pure_transformer_best.pt',
                        out_dir='results/eval/v8_transformer'),
}
SEQ = 'unet'
CSV_NAME = SEQ_SPECS[SEQ]['csv']
SUMMARY_NAME = SEQ_SPECS[SEQ]['summary']
MODEL_NAME = SEQ_SPECS[SEQ]['name']
TITLE = SEQ_SPECS[SEQ]['title']


# ------------------------------------------------------------------ 모델
def build_model(device):
    """main_train_pure_v8.build_model 과 같은 공유 config 인자 (use_dc=False). SEQ 전역으로 모델 선택."""
    if SEQ == 'unet':
        from u_pure_eternet_unet import PureETER_UNET
        model = PureETER_UNET(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, use_dc=False,
        )
    elif SEQ == 'pixelgru':
        from u_pure_eternet_pixelgru import PureETER_PIXELGRU
        model = PureETER_PIXELGRU(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            pixelgru_hidden=C.PIXELGRU_HIDDEN, use_dc=False,
        )
    elif SEQ == 'transformer':
        from u_pure_eternet_transformer import PureETER_TRANSFORMER
        model = PureETER_TRANSFORMER(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            axial_d_model=C.TRANSFORMER_D_MODEL, axial_n_pairs=C.TRANSFORMER_N_PAIRS,
            axial_n_heads=C.TRANSFORMER_N_HEADS, use_dc=False,
        )
    else:
        raise ValueError(SEQ)
    return model.to(device)


def build_val_dataset(data_path):
    """eval_paired_v8_nodc.main() 과 같은 인자 — idx 조인 정합의 전제."""
    return FastMRI_H5_Dataloader(data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                                 acceleration=4, center_fraction=0.08,
                                 random_mask=False, augment=False)


def run_inference(model, ds, total, device, num_workers):
    """eval_paired_v8_nodc.main() 의 슬라이스 루프와 같은 순서·같은 forward·같은 지표. 행 리스트 반환."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    rows = []
    it = iter(loader)
    with torch.no_grad():
        for idx in tqdm(range(total), desc=f'{MODEL_NAME} eval', unit='slice'):
            sample = next(it)
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask, sens)

            sm = slice_metrics(out, data_ref, brain_mask)       # 내부 점수 키는 아래에서 버림

            file_path, slice_idx, _ = ds.samples[idx]
            row = {'idx': idx, 'file': os.path.basename(file_path), 'slice_idx': slice_idx}
            for k in METRICS:
                row[k] = sm[k]
            rows.append(row)
    return rows


# ------------------------------------------------------------------ CSV 입출력
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'file', 'slice_idx'] + METRICS)
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path, prefix=''):
    """per-slice CSV → [(idx, file, slice_idx, {metric: float})]. prefix='gru' 이면 gru_ssim … 열을 읽는다."""
    col = {m: (f'{prefix}_{m}' if prefix else m) for m in METRICS}
    out = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in col.values() if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f'[ERROR] {path}: 열 없음 {missing} (있는 열: {reader.fieldnames})')
        for r in reader:
            out.append((int(r['idx']), r['file'], int(r['slice_idx']),
                        {m: float(r[col[m]]) for m in METRICS}))
    return out


def parse_ref(spec):
    """'name=path:column' → (name, path, column 접두). column 은 'gru' 또는 'gru_ssim' 꼴 모두 허용, 빈 값 = 접두 없음."""
    if '=' not in spec or ':' not in spec.split('=', 1)[1]:
        raise SystemExit(f'[ERROR] --ref 형식은 name=path:column (받은 값: {spec!r})')
    name, rest = spec.split('=', 1)
    path, column = rest.rsplit(':', 1)
    for m in METRICS:
        if column.endswith(f'_{m}'):
            column = column[:-len(m) - 1]
            break
    return name.strip(), path, column


def join_ref(model_rows, ref_rows, ref_label):
    """idx 로 조인하고 file·slice_idx 가 같은지 검사. 하나라도 어긋나면 중단 (다른 검증 집합/순서와의 비교 방지)."""
    ref_by_idx = {r[0]: r for r in ref_rows}
    missing, mismatch = [], []
    vals = {m: [] for m in METRICS}
    for idx, fname, sidx, _ in model_rows:
        r = ref_by_idx.get(idx)
        if r is None:
            missing.append(idx)
            continue
        if r[1] != fname or r[2] != sidx:
            mismatch.append((idx, (fname, sidx), (r[1], r[2])))
            continue
        for m in METRICS:
            vals[m].append(r[3][m])
    if missing or mismatch:
        raise SystemExit(f'[ERROR] {ref_label}: idx 미매칭 {len(missing)}, file/slice_idx 불일치 {len(mismatch)} '
                         f'(예: {(missing[:3], mismatch[:3])}) — 같은 검증 집합·같은 순서의 CSV 가 아님')
    return {m: np.array(v) for m, v in vals.items()}, len(ref_rows)


# ------------------------------------------------------------------ 집계 (paper/make_tables.py 와 같은 식)
def volume_groups(files):
    """make_tables.py:44,48-50 — 볼륨 = file 이름, np.unique 정렬 순서, 각 볼륨의 슬라이스 위치 배열."""
    files = np.asarray(files)
    uf = np.unique(files)
    return uf, [np.where(files == f)[0] for f in uf]


def vol_mean(a, vol_idx):
    """make_tables.py:287-288."""
    return np.array([a[ix].mean() for ix in vol_idx])


def mean_sd(a, m, vol_idx, unit='volume'):
    """make_tables.py:291-297 ms() 의 수치 부분 — nMSE 는 % (×100), unit='volume' 이면 볼륨 평균들의 평균±SD(ddof=1)."""
    a = np.asarray(a) * CM_SCALE[m]
    if unit == 'volume':
        a = vol_mean(a, vol_idx)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float('nan')


def fmt_ms(m, mean, sd):
    return CM_FMT[m].format(mean) + '±' + CM_FMT[m].format(sd)


def paired_vs_ref(model_m, ref_m, vol_idx, seed=BOOT_SEED, n_boot=BOOT_N):
    """make_tables.py:53-79 paired_stats 와 같은 계산. Δ 부호 = 양수일 때 **기준(ref) 우위** (NMSE/L1 은 부호 반전).

    make_tables 의 paired_stats(base, new) 에서 base = U-Net 단독, new = 기준 으로 둔 것과 같다.
    rng 는 비교마다 새로 default_rng(seed) — make_tables 의 첫 비교(SS2D vs GRU)와 같은 소비 순서(METRICS 순)라
    paired_vs_ref(gru, ss2d) 는 make_tables 의 S_V8 을 그대로 재현한다(--ref 순서와 무관하게 재현 가능).
    추가 항목 vmean/vmean_ci: 같은 재표본(pick)에서 계산한 볼륨 평균 Δ 의 평균 (표 1 의 볼륨 단위와 같은 가중).
    """
    rng = np.random.default_rng(seed)
    V = len(vol_idx)
    out = {}
    for m in METRICS:
        d = (model_m[m] - ref_m[m]) if m in LOWER else (ref_m[m] - model_m[m])
        vols = [d[ix] for ix in vol_idx]
        dv = np.array([x.mean() for x in vols])
        wins, means, vmeans = [], [], []
        for _ in range(n_boot):
            pick = rng.integers(0, V, V)
            s = np.concatenate([vols[i] for i in pick])
            wins.append(100 * np.mean(s > 0))
            means.append(s.mean())
            vmeans.append(dv[pick].mean())
        try:
            p_vol = wilcoxon(dv).pvalue
        except ValueError:                        # 볼륨 1개·Δ 전부 0 등 (부분 평가)
            p_vol = float('nan')
        try:
            p_slice = wilcoxon(d).pvalue
        except ValueError:
            p_slice = float('nan')
        out[m] = dict(
            win=100 * np.mean(d > 0), win_ci=np.percentile(wins, [2.5, 97.5]),
            mean=d.mean(), mean_ci=np.percentile(means, [2.5, 97.5]),
            vmean=dv.mean(), vmean_ci=np.percentile(vmeans, [2.5, 97.5]),
            med=np.median(d), iqr=np.percentile(d, [25, 75]),
            vol_win=100 * np.mean(dv > 0), p_vol=p_vol, p_slice=p_slice,
        )
    return out


def f_delta(m, v):
    """Δ 표기 — make_tables.py:372-373 f_delta_conv (nMSE 는 10⁻³ % 단위) + L1."""
    if m == 'ssim':
        return f'{v:+.4f}'
    if m == 'psnr':
        return f'{v:+.2f}'
    if m == 'nmse':
        return f'{v * 1e5:+.1f}'
    return f'{v:+.3f}'


def f_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return f'<0.001 ({p:.1e})' if p < 1e-3 else f'{p:.3f}'


# ------------------------------------------------------------------ 체크포인트 에폭 확인 (sanity 앵커)
def ckpt_epoch_info(ckpt_path):
    """ckpt 폴더 log.txt 에서 이 ckpt 의 에폭과 그 에폭 val_ssim_m 을 찾는다(없으면 None).

    *_best.pt = 트레이너의 체크포인트 선택 규칙(log.txt 의 선택 점수 최댓값, 최초 등장)으로 추정 — 그 점수 값 자체는 출력하지 않는다.
    *_epoch_N.pt = 에폭 N. 학습 val 의 SSIM 도 슬라이스 단위 계산이라 per-slice SSIM 평균과 소수 넷째 자리까지 일치해야 정상
    (v8 주 실험 실측: SS2D ep48 0.9140 / GRU ep50 0.9126 = per-slice CSV 평균).
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'log.txt')
    if not os.path.exists(log_path):
        return None
    pat = re.compile(r'^Epoch (\d+)/(\d+).*?val_composite=([\d.]+)\s+val_ssim_m=([\d.]+)')
    ep_rows = {}
    n_total = None
    with open(log_path) as f:
        for line in f:
            mt = pat.match(line)
            if mt:
                ep_rows[int(mt.group(1))] = (float(mt.group(3)), float(mt.group(4)))   # 재개로 중복되면 마지막 줄
                n_total = int(mt.group(2))
    if not ep_rows:
        return None
    base = os.path.basename(ckpt_path)
    mt = re.search(r'_epoch_(\d+)\.pt$', base)
    if mt:
        ep = int(mt.group(1))
        if ep not in ep_rows:
            return None
        return dict(epoch=ep, n_total=n_total, val_ssim_m=ep_rows[ep][1], how='파일 이름')
    if base.endswith('_best.pt'):
        best_ep, best_sel = None, -1.0
        for ep in sorted(ep_rows):
            if ep_rows[ep][0] > best_sel:
                best_ep, best_sel = ep, ep_rows[ep][0]
        ties = [ep for ep in ep_rows if ep_rows[ep][0] == best_sel and ep != best_ep]
        return dict(epoch=best_ep, n_total=n_total, val_ssim_m=ep_rows[best_ep][1],
                    how='log.txt 체크포인트 선택 규칙으로 추정' + (f', 소수 넷째 자리 동률 에폭 {ties}' if ties else ''))
    return None


# ------------------------------------------------------------------ 요약
def build_summary(model_rows, refs, ckpt, n_val_total, ep_info):
    files = [r[1] for r in model_rows]
    uf, vol_idx = volume_groups(files)
    V, n = len(uf), len(model_rows)
    model_m = {m: np.array([r[3][m] for r in model_rows]) for m in METRICS}

    joined = []                                   # (name, path, column, ref_m, n_ref_rows)
    for name, path, column in refs:
        ref_m, n_ref = join_ref(model_rows, read_csv_rows(path, column), f'{name} ({path})')
        joined.append((name, path, column, ref_m, n_ref))

    L = [f'# v8 Pure ETER-Net — {TITLE} per-slice 평가', '']
    L.append(f'- 체크포인트: `{ckpt}`' + (f' — ep {ep_info["epoch"]}/{ep_info["n_total"]} ({ep_info["how"]})'
                                        if ep_info else ''))
    partial = n < n_val_total if n_val_total else False
    L.append(f'- 평가 슬라이스: {n:,} / {n_val_total:,} (볼륨 {V})' if n_val_total else
             f'- 평가 슬라이스: {n:,} (볼륨 {V})')
    if partial:
        L.append('- **부분 평가(--max-samples) — 배관 점검용, 논문·문서 인용 불가**')
    L.append('- 프로토콜: `eval_paired_v8_nodc.py` 와 동일 (같은 val 순서, R=4 고정 마스크, brain-masked, batch 1, AMP, '
             '지표 = `slice_metrics` import)')
    ssim_slice = float(model_m['ssim'].mean())
    if ep_info:
        L.append(f'- sanity: 슬라이스 단위 SSIM 평균 = **{ssim_slice:.4f}** — log.txt ep {ep_info["epoch"]} 의 '
                 f'val_ssim_m = {ep_info["val_ssim_m"]:.4f} 와 같아야 정상')
    else:
        L.append(f'- sanity: 슬라이스 단위 SSIM 평균 = **{ssim_slice:.4f}** (ckpt 폴더 log.txt 의 해당 에폭 val_ssim_m 과 비교)')
    L.append('')

    # (1) 볼륨 단위 표 — make_tables 표 1 과 같은 식 (+ 슬라이스 단위 변형)
    for unit in ['volume', 'slice']:
        if unit == 'volume':
            L += [f'## 1. 볼륨 단위 평균±표준편차 (n = {V} 볼륨, SD ddof=1 — `paper/make_tables.py` 표 1 과 같은 계산)', '']
        else:
            L += [f'### (참고) 슬라이스 단위 평균±표준편차 (n = {n:,} 슬라이스, SD ddof=1 — 표 1 의 슬라이스 단위 변형)', '']
        L += ['| Method | ' + ' | '.join(HEAD[m] for m in CM) + ' |', '|---|' + '---:|' * len(CM)]
        L.append(f'| {MODEL_NAME} | ' + ' | '.join(fmt_ms(m, *mean_sd(model_m[m], m, vol_idx, unit)) for m in CM) + ' |')
        for name, _path, _col, ref_m, _n in joined:
            L.append(f'| {name} (같은 슬라이스) | '
                     + ' | '.join(fmt_ms(m, *mean_sd(ref_m[m], m, vol_idx, unit)) for m in CM) + ' |')
        L.append('')
    L += ['(nMSE (%) = 슬라이스 nMSE ×100. 볼륨 단위 = 볼륨별 슬라이스 평균을 낸 뒤 볼륨 사이 평균±SD.)', '']

    # (2) 기준 CSV 대응 비교
    if not joined:
        L += ['## 2. 대응(paired) 비교', '', '(--ref 미지정 — 생략)', '']
    for i, (name, path, column, ref_m, n_ref) in enumerate(joined):
        S = paired_vs_ref(model_m, ref_m, vol_idx)
        L += [f'## 2.{i + 1} 대응(paired) 비교: {name} vs {MODEL_NAME}', '',
              f'- 기준 CSV: `{path}` (열 접두 `{column or "(없음)"}`), 조인 = idx 기준 {n:,}/{n_ref:,} 행, '
              f'file·slice_idx 불일치 0',
              f'- Δ = 양수일 때 **{name} 우위** ({name} − {MODEL_NAME}; nMSE·L1 은 부호 반전). nMSE Δ 단위 = 10⁻³ %.',
              f'- 95% CI = 볼륨 단위 군집(cluster) 부트스트랩 {BOOT_N:,}회 (seed {BOOT_SEED}, `make_tables.paired_stats` 와 같은 규약). '
              f'p = 볼륨 단위 Wilcoxon signed-rank 양측 (n={V}).', '',
              f'| 지표 | Δ 슬라이스 평균 [95% CI] | Δ 볼륨 평균 [95% CI] | Δ 중앙값 (IQR) | {name} 우위 슬라이스 (%) [95% CI] '
              f'| {name} 우위 볼륨 (%) | p (볼륨) |',
              '|---|---:|---:|---:|---:|---:|---:|']
        for m in METRICS:
            s = S[m]
            L.append(f'| {NAME_D[m]} '
                     f'| {f_delta(m, s["mean"])} [{f_delta(m, s["mean_ci"][0])}, {f_delta(m, s["mean_ci"][1])}] '
                     f'| {f_delta(m, s["vmean"])} [{f_delta(m, s["vmean_ci"][0])}, {f_delta(m, s["vmean_ci"][1])}] '
                     f'| {f_delta(m, s["med"])} ({f_delta(m, s["iqr"][0])}, {f_delta(m, s["iqr"][1])}) '
                     f'| {s["win"]:.1f} [{s["win_ci"][0]:.1f}, {s["win_ci"][1]:.1f}] '
                     f'| {s["vol_win"]:.1f} | {f_p(s["p_vol"])} |')
        L.append('')
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ main
def main():
    global SEQ, CSV_NAME, SUMMARY_NAME, MODEL_NAME, TITLE
    p = argparse.ArgumentParser(description='v8 단일 모델(U-Net 단독·pixel-GRU·Transformer) per-slice 평가 + 기준 CSV 대응 비교')
    p.add_argument('--seq', choices=sorted(SEQ_SPECS), default='unet',
                   help='평가할 모델 (unet = U-Net 단독 대조 모델, 기본). --ckpt·--out-dir 기본값도 이에 따라 바뀜')
    p.add_argument('--ckpt', default=None,
                   help='state_dict 체크포인트(*_best.pt / *_epoch_N.pt). 전체 상태 *_last.pt 는 load_ckpt 계약 밖. '
                        '--summary-only 때도 추론에 쓴 경로를 그대로 줄 것 (요약의 ckpt·에폭 표기용)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 검증 집합 전체 (양수 = 배관 점검용 부분 평가)')
    p.add_argument('--num-workers', type=int, default=4,
                   help='순서 보존(shuffle=False). eval_paired_v8_nodc 의 0 고정은 /dev/shm 64MB 시절 — 현 컨테이너 128g')
    p.add_argument('--ref', action='append', default=[], metavar='NAME=PATH:COLUMN',
                   help='대응 비교 기준 per-slice CSV (반복 가능). 예: '
                        '"bi-GRU (original)=results/eval/v8_nodc/per_slice_paired.csv:gru"')
    p.add_argument('--force', action='store_true', help='기존 per-slice CSV 덮어쓰기 허용')
    p.add_argument('--summary-only', action='store_true',
                   help='추론 없이 --out-dir 의 기존 CSV 로 요약만 다시 생성 (--ref 추가용)')
    args = p.parse_args()
    spec = SEQ_SPECS[args.seq]
    SEQ, CSV_NAME, SUMMARY_NAME, MODEL_NAME, TITLE = (args.seq, spec['csv'], spec['summary'],
                                                     spec['name'], spec['title'])
    args.ckpt = args.ckpt or spec['ckpt']
    args.out_dir = args.out_dir or spec['out_dir']

    refs = [parse_ref(s) for s in args.ref]
    for name, path, _ in refs:
        if not os.path.exists(path):
            raise SystemExit(f'[ERROR] --ref {name}: 파일 없음 {path}')

    csv_path = os.path.join(args.out_dir, CSV_NAME)
    summary_path = os.path.join(args.out_dir, SUMMARY_NAME)

    if args.summary_only:
        if not os.path.exists(csv_path):
            raise SystemExit(f'[ERROR] --summary-only: CSV 없음 {csv_path}')
        model_rows = read_csv_rows(csv_path)
        n_val_total = len(build_val_dataset(args.data_path)) if os.path.isdir(args.data_path) else None
    else:
        if os.path.exists(csv_path) and not args.force:
            raise SystemExit(f'[ERROR] {csv_path} 가 이미 있음 — 덮어쓰려면 --force (요약만 갱신은 --summary-only)')
        if not os.path.exists(args.ckpt):
            raise SystemExit(f'[ERROR] 체크포인트 없음: {args.ckpt}')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('=' * 64)
        print(f' v8 Pure ETER-Net — {TITLE} per-slice 평가')
        print(f'  device={device}')
        print('=' * 64)
        if not torch.cuda.is_available():
            print('  [WARN] CUDA 없음 — CPU 로 진행 (매우 느릴 수 있음)')

        model = load_ckpt(build_model(device), args.ckpt, device)
        n_params = sum(q.numel() for q in model.parameters())
        print(f'  ckpt: {args.ckpt}  (params {n_params / 1e6:.1f}M)')

        ds = build_val_dataset(args.data_path)
        n_val_total = len(ds)
        total = min(n_val_total, args.max_samples) if args.max_samples > 0 else n_val_total
        print(f'\n평가 대상: {total} / {n_val_total} 슬라이스\n')

        t0 = time.time()
        rows = run_inference(model, ds, total, device, args.num_workers)
        dt = time.time() - t0
        print(f'\n추론+지표 {dt / 60:.1f} 분 ({total / max(dt, 1e-9):.2f} slice/s)')

        os.makedirs(args.out_dir, exist_ok=True)
        write_csv(csv_path, rows)
        model_rows = read_csv_rows(csv_path)      # 요약은 항상 저장된 CSV 에서 (--summary-only 와 같은 경로)

    ep_info = ckpt_epoch_info(args.ckpt)
    msg = build_summary(model_rows, refs, args.ckpt, n_val_total, ep_info)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(msg)
    print('\n' + msg)
    print(f'CSV : {csv_path}')
    print(f'요약: {summary_path}')


if __name__ == '__main__':
    main()
