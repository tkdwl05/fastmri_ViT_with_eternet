"""
공개 모델 기준선 — 전체 검증셋(7,334 슬라이스 / 464 볼륨) 추론-only 평가 (CPU 가능 · 재개 가능).

U-Net† / E2E-VarNet†(fastMRI leaderboard 가중치, train+val 학습 → 누수 참고선) / PromptMR+(train 구획만
학습된 공개 가중치, 인접 5슬라이스 입력)를 **우리 프로토콜 행**(384² 재-FFT · 16코일 절단 · 동일 R4 마스크 ·
동일 GT · brain-masked 지표 · per-slice LS 강도 정합)으로 평가한다 — 즉 `visualize_multimodel_compare.py` 의
정본 12 슬라이스 추론을 전체 val 로 확장한 것(러너·지표 공식을 그 스크립트에서 import, 수치 동일성 확인 완료).

설계:
  - 방법별(outer) → 슬라이스(inner) 루프, 모델 1개씩만 메모리에. 입력 준비(재-FFT·코일 절단·이웃 스택)는
    DataLoader 워커가, forward 는 메인 프로세스가 맡는다(PromptMR+ 는 워커가 z±2 이웃 5장을 함께 로드).
  - 결과는 `per_slice_<method>.csv` 에 슬라이스마다 append+flush → 중단 후 같은 명령으로 재개(완료 idx skip).
  - `--summary` : 방법별 CSV + 우리 3모델 CSV(v9 paired) + zero-filled CSV 를 (file, slice) 로 조인해
    슬라이스/볼륨 단위 mean±SD, non-finite 수, SS2D·v9 대비 우위 비율(슬라이스·볼륨)·Wilcoxon(볼륨) 을 md 로.
  - native 프로토콜 행(전체 코일·native 해상도)은 여기서 다루지 않는다 — `eval_paired_baselines.py`(GPU) 몫.

실행 (CPU, GPU0 은 학습 점유 — nice 19 · 스레드 제한). 방법별 CSV 가 독립이므로 **방법마다 별도 프로세스**로 띄우면
호스트 유휴 코어를 더 쓴다(09-06 실측: 호스트 외부 부하 ~10코어 상황에서 프로세스당 4~6 스레드가 12 스레드보다 빠름):
  # 유휴 코어 ≈8 기준 배분: PromptMR+(long pole) 5 스레드 + {varnet→unet} 순차 3 스레드, 워커 1씩
  CUDA_VISIBLE_DEVICES="" nice -n 19 setsid nohup python v8_eter_pure/eval_baselines_full.py \
    --methods promptmr --threads 5 --num-workers 1 > results/eval/baselines_384_full/run_promptmr.log 2>&1 < /dev/null & disown
  CUDA_VISIBLE_DEVICES="" nice -n 19 setsid nohup python v8_eter_pure/eval_baselines_full.py \
    --methods varnet,unet --threads 3 --num-workers 1 > results/eval/baselines_384_full/run_varnet_unet.log 2>&1 < /dev/null & disown
  (처리 순서는 기본 interleaved — 중단 시점의 완료 prefix 가 전 볼륨에 고른 층화 표본이라 `--summary` 중간 집계가 편향 없음)
요약만 재생성:  CUDA_VISIBLE_DEVICES="" python v8_eter_pure/eval_baselines_full.py --summary
스모크:         CUDA_VISIBLE_DEVICES="" python v8_eter_pure/eval_baselines_full.py --methods varnet --max-samples 3 --out-dir <scratch>
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import visualize_multimodel_compare as VM        # noqa: E402  (import 시 cwd 를 저장소 루트로 옮김)

METRICS = ['ssim', 'psnr', 'nmse', 'l1']
LOWER_IS_BETTER = {'nmse', 'l1'}
METHOD_NAMES = {'unet': 'U-Net†', 'varnet': 'E2E-VarNet†', 'promptmr': 'PromptMR+'}
OURS = {'gru': 'bi-GRU (original)', 'ss2d': 'SS2D (controlled)', 'v9': 'Enhanced SS2D'}


# ──────────────────────────────────────────────
#  워커 측 입력 준비
# ──────────────────────────────────────────────

class ItemDataset(Dataset):
    """indices[k] 슬라이스의 (모델별 입력, GT, brain mask) — 재-FFT·코일 절단·이웃 스택을 워커에서 수행."""

    def __init__(self, h5, indices, method, num_adj=5):
        self.h5, self.indices, self.method, self.num_adj = h5, list(indices), method, num_adj

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        s = self.h5[idx]
        if self.method == 'unet':
            inp = VM.prep_unet(s)
            inp['coils_kept'] = int(VM._kspace_keep_coils(s['data'])[1].sum())
        elif self.method == 'varnet':
            inp = VM.prep_varnet(s)
        elif self.method == 'promptmr':
            nb_idx = VM.neighbor_indices(self.h5, idx, self.num_adj)
            nbs = [s if i == idx else self.h5[i] for i in nb_idx]
            inp = VM.prep_promptmr(s, nbs)
        else:
            raise ValueError(self.method)
        inp['idx'] = idx
        inp['gt'] = s['label'][0].astype(np.float32)
        inp['brain'] = s['brain_mask'][0].astype(np.float32)
        return inp


def _worker_init(_):
    torch.set_num_threads(1)


# ──────────────────────────────────────────────
#  메타 / CSV 유틸
# ──────────────────────────────────────────────

_META = {}


def file_meta(fp):
    """(acquisition, 파일의 실제 코일 수) — h5 attrs, 파일당 1회."""
    if fp not in _META:
        import h5py
        with h5py.File(fp, 'r') as f:
            acq = f.attrs.get('acquisition', '')
            acq = acq.decode() if isinstance(acq, bytes) else str(acq)
            _META[fp] = (acq, int(f['kspace'].shape[1]))
    return _META[fp]


def interleaved_order(indices, period=64):
    """idx % period 의 비트반전 순으로 정렬 — 어느 시점에 중단해도 완료 prefix 가 전 볼륨에 고르게 퍼진 층화 표본이
    되도록(중간 요약을 편향 없이 쓰기 위함). period=64 ≈ 볼륨 4개마다 1 슬라이스씩 먼저 훑는다."""
    bits = period.bit_length() - 1
    def rev(r):
        return int(format(r, f'0{bits}b')[::-1], 2)
    return sorted(indices, key=lambda i: (rev(i % period), i))


def csv_path(out_dir, method):
    return os.path.join(out_dir, f'per_slice_{method}.csv')


def read_done(path):
    """완료 행 {idx: row}. 다른 프로세스가 append 중인 미완성 마지막 행은 무시(방법별 병렬 실행 대비)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                if r.get('sec') is None or r['sec'] == '':
                    continue
                float(r['ssim']); float(r['sec'])
                out[int(r['idx'])] = r
            except (TypeError, ValueError):
                continue
    return out


FIELDS = ['idx', 'file', 'slice_idx', 'acquisition', 'coils_file', 'coils_kept', 'finite', 'alpha',
          'ssim', 'psnr', 'nmse', 'l1', 'sec']


# ──────────────────────────────────────────────
#  방법 하나 평가
# ──────────────────────────────────────────────

def eval_method(method, args, h5, device):
    path = csv_path(args.out_dir, method)
    done = read_done(path)
    if args.indices:
        universe = [int(x) for x in args.indices.split(',') if x.strip()]
    else:
        universe = list(range(len(h5) if args.max_samples <= 0 else min(len(h5), args.max_samples)))
        if args.order == 'interleaved':
            universe = interleaved_order(universe)
    n_total = len(universe)
    todo = [i for i in universe if i not in done]
    print(f'\n[{method}] 완료 {len(done)} / 대상 {n_total} → 남은 {len(todo)}', flush=True)
    if not todo:
        return
    ckpt = {'unet': args.unet_ckpt, 'varnet': args.varnet_ckpt, 'promptmr': args.promptmr_ckpt}[method]
    loader_fn = {'unet': VM.load_unet, 'varnet': VM.load_varnet, 'promptmr': VM.load_promptmr}[method]
    fwd = {'unet': VM.forward_unet, 'varnet': VM.forward_varnet, 'promptmr': VM.forward_promptmr}[method]
    t0 = time.time()
    model = loader_fn(ckpt, device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    num_adj = getattr(model, 'num_adj_slices_cfg', 5)
    print(f'  ckpt {ckpt}  로드 {time.time()-t0:.0f}s  params {n_par:.1f}M', flush=True)

    ds = ItemDataset(h5, todo, method, num_adj)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.num_workers,
                        worker_init_fn=_worker_init, prefetch_factor=(2 if args.num_workers > 0 else None),
                        persistent_workers=False)
    new_file = not os.path.exists(path)
    f = open(path, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    t_start = time.time()
    n_done_here = 0
    n_nonfinite = 0
    for item in loader:
        idx = int(item['idx'])
        t1 = time.time()
        rec = fwd(model, item, device)
        gt = np.asarray(item['gt'], dtype=np.float32)
        bm = np.asarray(item['brain'], dtype=np.float32)
        finite = bool(np.isfinite(rec).all())
        if not finite:
            n_nonfinite += 1
        rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        m = bm > 0.5
        r, g = rec[m], gt[m]
        denom = float((r * r).sum())
        alpha = float((r * g).sum()) / denom if (m.any() and denom >= 1e-12) else 1.0
        rec = VM.ls_scale(rec, gt, bm)
        met = VM.slice_metrics_np(rec, gt, bm)                       # ssim/psnr/nmse (eval_paired 공식)
        met['l1'] = float((np.abs(rec - gt) * bm).sum() / max(float(bm.sum()), 1.0))   # eval_paired_v8_nodc 의 masked L1
        fp, si, _ = h5.samples[idx]
        acq, coils_file = file_meta(fp)
        row = {'idx': idx, 'file': os.path.basename(fp), 'slice_idx': int(si), 'acquisition': acq,
               'coils_file': coils_file, 'coils_kept': int(item.get('coils_kept', -1)), 'finite': int(finite),
               'alpha': f'{alpha:.6g}', 'sec': f'{time.time()-t1:.2f}',
               **{k: f'{met[k]:.6g}' for k in METRICS}}
        w.writerow(row)
        f.flush()
        n_done_here += 1
        if n_done_here % args.log_every == 0 or n_done_here == len(todo):
            el = time.time() - t_start
            rate = el / n_done_here
            eta = rate * (len(todo) - n_done_here)
            print(f'  [{method}] {len(done)+n_done_here}/{n_total}  {rate:.1f}s/slice  경과 {el/3600:.2f}h  '
                  f'ETA {eta/3600:.2f}h  non-finite {n_nonfinite}  (last idx {idx}: PSNR {met["psnr"]:.2f} SSIM {met["ssim"]:.4f})',
                  flush=True)
    f.close()
    del model
    print(f'[{method}] 완료 — {path}  (non-finite {n_nonfinite})', flush=True)


# ──────────────────────────────────────────────
#  요약
# ──────────────────────────────────────────────

def _read_ours(v9_csv):
    d = {}
    with open(v9_csv, newline='') as f:
        for r in csv.DictReader(f):
            d[(r['file'], int(r['slice_idx']))] = {f'{a}_{k}': float(r[f'{a}_{k}'])
                                                    for a in OURS for k in METRICS}
    return d


def _read_zf(zf_csv):
    d = {}
    if not os.path.exists(zf_csv):
        return d
    with open(zf_csv, newline='') as f:
        for r in csv.DictReader(f):
            d[(r['file'], int(r['slice_idx']))] = {f'zf_{k}': float(r[f'raw_{k}']) for k in METRICS}
    return d


def _fmt(k, v):
    return {'ssim': f'{v:.4f}', 'psnr': f'{v:.2f}', 'nmse': f'{100*v:.3f}', 'l1': f'{v:.2f}'}[k]


def _ms(k, a):
    a = np.asarray(a, dtype=np.float64)
    return f'{_fmt(k, a.mean())}±{_fmt(k, a.std(ddof=1))}' if len(a) > 1 else _fmt(k, a.mean())


def write_summary(args, h5):
    from scipy.stats import wilcoxon
    ours = _read_ours(args.v9_csv)
    zf = _read_zf(args.zf_csv)
    per = {m: read_done(csv_path(args.out_dir, m)) for m in ('unet', 'varnet', 'promptmr')}
    per = {m: d for m, d in per.items() if d}
    lines = ['# 공개 모델 기준선 — 전체 검증셋 (우리 프로토콜 행: 384² 재-FFT · 16코일 절단 · R4 · brain-masked · per-slice LS 정합)', '',
             f'- 생성: `v8_eter_pure/eval_baselines_full.py --summary` ({time.strftime("%Y-%m-%d %H:%M")})',
             f'- 우리 3모델: `{args.v9_csv}` (GPU fp16 평가, LS 정합 없음 α≈1) · zero-filled: `{args.zf_csv}` (raw)',
             '- 공개 모델은 CPU fp32 추론(정본 12 슬라이스에서 GPU 와 SSIM ≤0.003 · PSNR ≤0.3 dB 차이 확인) · non-finite 출력은 0 으로 치환 후 집계(개수 별도 표기)',
             '- † = fastMRI leaderboard 공개 가중치(train+val 합본 학습 → 본 검증셋이 학습 데이터에 포함, 누수 참고선). PromptMR+ = train 구획만 학습(누수 없음)·인접 5슬라이스 입력(정보량 다름)',
             '']
    # 진행 상황
    lines += ['## 진행 상황', '', '| 방법 | 완료 슬라이스 | non-finite | 평균 s/slice |', '|---|---|---|---|']
    for m, d in per.items():
        nf = sum(1 for r in d.values() if r['finite'] == '0')
        sec = np.mean([float(r['sec']) for r in d.values()])
        lines.append(f'| {METHOD_NAMES[m]} | {len(d)} / {len(h5)} | {nf} | {sec:.1f} |')
    lines.append('')

    # 공통 슬라이스 집합 (모든 완료 방법 + 우리 3모델)
    keys_by_m = {m: {(r['file'], int(r['slice_idx'])) for r in d.values()} for m, d in per.items()}
    common = set(ours)
    for ks in keys_by_m.values():
        common &= ks
    common = sorted(common, key=lambda k: (k[0], k[1]))
    files = sorted({k[0] for k in common})
    lines += [f'## 공통 슬라이스 {len(common)} (볼륨 {len(files)}) 에서의 평균', '']

    def arm_arr(arm, k):
        if arm in per:
            d = {(r['file'], int(r['slice_idx'])): float(r[k]) for r in per[arm].values()}
            return np.array([d[key] for key in common])
        if arm == 'zf':
            return np.array([zf[key][f'zf_{k}'] for key in common]) if zf else None
        return np.array([ours[key][f'{arm}_{k}'] for key in common])

    vol_of = np.array([files.index(k[0]) for k in common])

    def vol_mean(a):
        return np.array([a[vol_of == v].mean() for v in range(len(files))])

    arms = (['zf'] if zf and all(k in zf for k in common) else []) + list(per) + list(OURS)
    names = {'zf': 'Zero-filled (raw)', **METHOD_NAMES, **OURS}
    for unit in ('volume', 'slice'):
        n = len(files) if unit == 'volume' else len(common)
        lines += [f'### {unit} 단위 mean±SD (n={n}, SD ddof=1)', '',
                  '| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |', '|---|---|---|---|---|']
        for arm in arms:
            cells = []
            for k in METRICS:
                a = arm_arr(arm, k)
                if a is None:
                    cells.append('–'); continue
                if unit == 'volume':
                    a = vol_mean(a)
                cells.append(_ms(k, a))
            lines.append(f'| {names[arm]} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    # 우위 비율 (공개 모델 vs 우리 SS2D / v9)
    for ref in ('ss2d', 'v9'):
        lines += [f'## 공개 모델 vs {OURS[ref]} — 우위 비율 (슬라이스 / 볼륨) · Wilcoxon(볼륨 단위 paired)', '',
                  '| 공개 모델 | 지표 | 공개 모델 우위 슬라이스 % | 공개 모델 우위 볼륨 % | Δ 볼륨 평균 (공개−우리) | p (볼륨) |',
                  '|---|---|---|---|---|---|']
        for m in per:
            for k in METRICS:
                a = arm_arr(m, k); b = arm_arr(ref, k)
                better = (a < b) if k in LOWER_IS_BETTER else (a > b)
                va, vb = vol_mean(a), vol_mean(b)
                vbetter = (va < vb) if k in LOWER_IS_BETTER else (va > vb)
                try:
                    pval = wilcoxon(va, vb).pvalue
                except ValueError:
                    pval = float('nan')
                dv = (va - vb).mean() * (100 if k == 'nmse' else 1)
                lines.append(f'| {METHOD_NAMES[m]} | {k} | {100*better.mean():.1f} | {100*vbetter.mean():.1f} | '
                             f'{dv:+.4f} | {pval:.2e} |')
        lines.append('')
    # contrast 별 SSIM (볼륨 단위)
    acqs = sorted({per[list(per)[0]][str(0)]['acquisition'] for _ in [0]} if False else
                  {r['acquisition'] for d in per.values() for r in d.values()})
    if acqs:
        lines += ['## contrast 별 SSIM (볼륨 단위 평균, 공통 슬라이스)', '',
                  '| Contrast | 볼륨 수 | ' + ' | '.join(names[a] for a in arms) + ' |',
                  '|---|---|' + '---|' * len(arms)]
        acq_of_file = {}
        for d in per.values():
            for r in d.values():
                acq_of_file[r['file']] = r['acquisition']
        for acq in acqs:
            vsel = np.array([acq_of_file.get(fn) == acq for fn in files])
            if not vsel.any():
                continue
            cells = []
            for arm in arms:
                a = arm_arr(arm, 'ssim')
                cells.append('–' if a is None else f'{vol_mean(a)[vsel].mean():.4f}')
            lines.append(f'| {acq} | {int(vsel.sum())} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['(우위 비율 = proportion favoring the public model; nMSE·L1 은 낮을수록 우위. 이 표는 참고선이며 순위 판정에 쓰지 않는다 — '
              '†는 누수, PromptMR+ 는 다중 슬라이스 입력·물리 모델 계열.)']
    msg = '\n'.join(lines)
    out = os.path.join(args.out_dir, 'baseline_summary_full.md')
    with open(out, 'w') as f:
        f.write(msg + '\n')
    print(msg)
    print(f'\n요약: {out}')


# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='공개 모델 기준선 전체 검증셋 평가 (CPU 가능·재개 가능)')
    p.add_argument('--methods', default='varnet,unet,promptmr')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--varnet-ckpt', default='models/pretrained/varnet_brain_leaderboard_state_dict.pt')
    p.add_argument('--promptmr-ckpt', default=VM.PMR_CKPT)
    p.add_argument('--v9-csv', default='results/eval/v9_unleashed/per_slice_paired_v9.csv')
    p.add_argument('--zf-csv', default='results/eval/zero_filled/per_slice_zero_filled.csv')
    p.add_argument('--out-dir', default='results/eval/baselines_384_full')
    p.add_argument('--max-samples', type=int, default=-1)
    p.add_argument('--indices', default='', help='스모크용: 평가할 idx 목록(쉼표) — 정본 슬라이스 대조 등')
    p.add_argument('--order', choices=['interleaved', 'sequential'], default='interleaved',
                   help='interleaved(기본): 완료 prefix 가 항상 층화 표본이 되는 순서 / sequential: idx 순')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=3)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--summary', action='store_true', help='추론 없이 요약만 재생성')
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print('=' * 72)
    print(' 공개 모델 기준선 전체 검증셋 평가 — ' + ', '.join(METHOD_NAMES[m] for m in args.methods.split(',') if m))
    print(f'  device={device}  threads={torch.get_num_threads()}  workers={args.num_workers}  out={args.out_dir}')
    print('=' * 72, flush=True)

    h5 = VM.FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=384, acceleration=VM.ACCEL,
                                  center_fraction=VM.CENTER_FRACTION, random_mask=False, augment=False)
    print(f'  val 슬라이스 {len(h5)}', flush=True)
    if not args.summary:
        for m in [x for x in args.methods.split(',') if x]:
            eval_method(m, args, h5, device)
    try:
        write_summary(args, h5)
    except Exception as e:                       # 요약은 부가 산출물 — 추론 CSV 완료 후의 실패로 런을 실패로 보이게 하지 않는다
        print(f'[summary] 실패(추론 CSV 는 완료됨): {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
