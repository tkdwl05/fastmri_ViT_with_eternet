"""
v8fix 체크포인트를 fastMRI 공식 정의 지표로 평가한다 (2026-10-16, 사용자 "직접 확인을 진행해보자").

배경: 우리 지표는 뇌 마스크 안 평균·data_range = 슬라이스별 마스크 안 GT max−min 이라 fastMRI 공식 `evaluate.py`
(마스크 없음, 볼륨 최댓값, W×W 가운데 자르기)와 정의가 다르다(docs/ssim_protocol_survey_2026-10-16.md (b)·(c)).
이 스크립트는 한 번의 추론에서 두 정의를 함께 계산한다 — 절차는 위 문서 (c) "384 출력에서 공식 정의로 계산하는 절차".

  - 추론: eval_unet_only_v8.run_inference 와 같은 데이터셋 인자·같은 순서·batch 1·AMP(cuda), 입력 정규화는
    eval_v8fix.Normalized(학습과 같은 단일 출처). 기존 두 평가 파일은 pod3 런처가 쓰므로 고치지 않고 import 만 한다.
  - 마스크 지표: eval_paired_v8_nodc.slice_metrics 그대로 — 기존 per-slice CSV 와 같아야 한다(--check-csv 로 대조).
  - 공식 지표 (fastmri.evaluate 의 ssim/psnr/nmse 를 옮겨 적음 — runstats 미설치라 import 불가):
      출력(라벨 단위 = RSS × 1e6)을 /1e6 하고 그 파일 reconstruction_rss 의 (H, W) 로 가운데 자름(패딩 오프셋
      (384−H)//2 = fastmri center_crop 오프셋) → 정답 = h5 의 reconstruction_rss 그대로 → 둘 다 (W, W) 로 가운데 자름 →
      maxval = 잘라낸 정답 볼륨 최댓값 → SSIM = 슬라이스별 skimage structural_similarity(data_range=maxval) 의 볼륨 평균,
      PSNR = 볼륨 전체 peak_signal_noise_ratio(data_range=maxval), NMSE = ‖gt−pred‖²/‖gt‖² (볼륨). 최소제곱 보정 없음.
  - 분해(기술용): 같은 SSIM map(full=True)을 skimage 와 같이 테두리 3 px 를 뺀 영역에서 뇌 마스크 안/밖으로 나눈 평균과
    안쪽 화소 비율. 공식 SSIM = 그 영역 전체 평균(분해 합과 같아야 함 — 자가 점검).
  - 민감도(기술용): 마스크 SSIM 의 data_range 만 볼륨 단위(볼륨 안 마스크 안 GT max − min)로 바꾼 값.
출력 (--out-dir): per_volume_<seq>.csv, per_slice_<seq>.csv(마스크 지표 + 공식 정의 슬라이스 SSIM), summary_<seq>.json/.md
자가 점검: --selftest (fastmri.evaluate 함수와 수치 일치·center_crop 역변환·분해 합; GPU·데이터 불필요)

실행 예:
  CUDA_VISIBLE_DEVICES=1 python v8_eter_pure/eval_v8fix_official.py --seq ss2d \
      --ckpt logs/PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval/pure_ss2d_best.pt \
      --data-path ./fastMRI_data/multicoil_test_full --out-dir results/v8fix_official_test_full \
      --check-csv results/v8fix_s1_fullval_test_full/per_slice_ss2d.csv
  ETER-net 은 --seq gru (학습과 같이 gru_h 를 fp32 로 — gru_precision.install_class_patch, eval_v8fix_grufp32.py 와 같음)
  --tf32 off = 트레이너와 같은 정밀도(TF32 끔). 기본값은 torch 기본(cudnn TF32 켜짐)이라 기존 평가 CSV 와 같은 조건 —
  ETER-net 은 off 로 돌리고 런처 CSV(켜짐)와의 마스크 SSIM 차로 TF32 영향을 잰다 (10-16).
"""

import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(os.path.dirname(_HERE), 'models', 'pure_eternet'))

from skimage.metrics import structural_similarity, peak_signal_noise_ratio  # noqa: E402

VAL_AMP_Y = 1e6        # dataloader_h5_v5.val_amp_Y — label = reconstruction_rss × 1e6
BORDER = 3             # skimage structural_similarity 기본 win_size 7 → (7−1)//2 픽셀 테두리를 평균에서 뺌


# ------------------------------------------------------------------ fastmri.evaluate 정의 (옮겨 적음)
def center_crop_np(x, shape):
    """fastmri.data.transforms.center_crop 과 같은 오프셋 (마지막 두 축)."""
    h, w = shape
    if not (0 < h <= x.shape[-2] and 0 < w <= x.shape[-1]):
        raise ValueError(f'Invalid shapes {x.shape} → {shape}')
    a = (x.shape[-2] - h) // 2
    b = (x.shape[-1] - w) // 2
    return x[..., a:a + h, b:b + w]


def official_volume(gt, pred, bmask=None):
    """gt, pred: (S, H, W) float — reconstruction_rss 크기. bmask: (S, H, W) bool 또는 None.
    fastmri.evaluate: target/recons 를 (W, W) 로 자른 뒤 ssim(maxval=gt.max())·psnr·nmse."""
    W = gt.shape[-1]
    gt = center_crop_np(gt, (W, W)).astype(np.float64)
    pred = center_crop_np(pred, (W, W)).astype(np.float64)
    maxval = gt.max()
    ssim_s, s_in, s_out, f_in, s_all = [], [], [], [], []
    for s in range(gt.shape[0]):
        val, smap = structural_similarity(gt[s], pred[s], data_range=maxval, full=True)
        ssim_s.append(val)
        inner = smap[BORDER:-BORDER, BORDER:-BORDER]
        s_all.append(inner.mean())
        if bmask is not None:
            m = center_crop_np(bmask[s], (W, W))[BORDER:-BORDER, BORDER:-BORDER]
            f_in.append(m.mean())
            s_in.append(inner[m].mean() if m.any() else np.nan)
            s_out.append(inner[~m].mean() if (~m).any() else np.nan)
    out = dict(ssim=float(np.mean(ssim_s)), psnr=float(peak_signal_noise_ratio(gt, pred, data_range=maxval)),
               nmse=float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2), maxval=float(maxval),
               ssim_slices=[float(v) for v in ssim_s], decomp_maxabs=float(np.max(np.abs(np.array(ssim_s) - np.array(s_all)))))
    if bmask is not None:
        out.update(ssim_in=float(np.nanmean(s_in)), ssim_out=float(np.nanmean(s_out)), frac_in=float(np.mean(f_in)))
    return out


def masked_ssim_volrange(gt384, pred384, bmask384):
    """마스크 SSIM(eval_paired_v8_nodc.skimage_ssim_masked 와 같은 map·같은 마스크 평균)에서 data_range 만 볼륨 단위로."""
    vals = gt384[bmask384]
    L = float(vals.max() - vals.min()) if vals.size else 1.0
    out = []
    for s in range(gt384.shape[0]):
        m = bmask384[s]
        if not m.any():
            continue
        _, smap = structural_similarity(gt384[s].astype(np.float64), pred384[s].astype(np.float64), data_range=L, full=True)
        out.append(smap[m].mean())
    return float(np.mean(out)) if out else float('nan')


# ------------------------------------------------------------------ 자가 점검
def selftest():
    import types
    sys.modules.setdefault('runstats', types.SimpleNamespace(Statistics=object))   # fastmri.evaluate import 용 대역
    from fastmri import evaluate as FE
    from fastmri.data import transforms as T
    rng = np.random.default_rng(0)
    for H, W in ((320, 320), (320, 260), (213, 213), (384, 312)):
        gt = rng.random((5, H, W)).astype(np.float32) * 3e-4
        pred = (gt + rng.normal(0, 3e-5, gt.shape)).astype(np.float32)
        # 패딩 → 가운데 자르기 역변환 (dataloader crop_or_pad_to 와 같은 오프셋)
        sys.path.append(os.path.join(os.path.dirname(_HERE), 'dataloaders'))
        from dataloader_h5_v5 import crop_or_pad_to
        pad = crop_or_pad_to(gt, (384, 384))
        assert np.array_equal(center_crop_np(pad, (H, W)), gt), 'pad/crop 역변환 불일치'
        tg = T.center_crop(torch.from_numpy(gt), (W, W)).numpy()
        rc = T.center_crop(torch.from_numpy(pred), (W, W)).numpy()
        ref = dict(ssim=FE.ssim(tg, rc), psnr=FE.psnr(tg, rc), nmse=FE.nmse(tg, rc))
        bm = rng.random(gt.shape) > 0.5
        ours = official_volume(gt, pred, bm)
        for k in ref:
            assert abs(ours[k] - float(ref[k])) <= 1e-6 * max(1.0, abs(float(ref[k]))), (k, ours[k], ref[k])
        assert ours['decomp_maxabs'] < 1e-9, ours['decomp_maxabs']
        recomb = ours['frac_in'] * ours['ssim_in'] + (1 - ours['frac_in']) * ours['ssim_out']
        assert abs(recomb - ours['ssim']) < 1e-3, (recomb, ours['ssim'])   # 슬라이스별 비율이 달라 근사
        print(f'  ({H},{W}) ok: ssim {ours["ssim"]:.6f} = fastmri {float(ref["ssim"]):.6f}, psnr {ours["psnr"]:.4f}, nmse {ours["nmse"]:.3e}')
    print('selftest PASS')


# ------------------------------------------------------------------ 본체
def build(seq, device):
    import eval_v8fix as V                 # Normalized (학습과 같은 정규화)
    import smoke_recipe_fix as R           # 모델 생성 (학습과 같은 단일 출처)
    if seq == 'gru':
        import gru_precision
        names = tuple(n for n in os.environ.get('GRU_FP32_MODULES', 'gru_h').split(',') if n)
        gru_precision.install_class_patch(names)
        print(f'[gru] {names} → fp32 (eval_v8fix_grufp32.py 와 같음)')
    base = R.build_model(seq, device)
    return base, V.Normalized


def main():
    p = argparse.ArgumentParser(description='v8fix 공식 정의(fastMRI evaluate.py) + 마스크 지표 동시 평가')
    p.add_argument('--seq', choices=['unet', 'gru', 'ss2d'])
    p.add_argument('--ckpt')
    p.add_argument('--data-path')
    p.add_argument('--out-dir')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-files', type=int, default=-1, help='-1 = 전체; 양수 = 앞 N 볼륨만 (배관 점검용, 볼륨 단위라 슬라이스를 자르지 않음)')
    p.add_argument('--check-csv', default=None, help='기존 per-slice CSV (idx,file,slice_idx,ssim,...) — 마스크 지표 대조')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tf32', choices=['default', 'off'], default='default',
                   help='default = torch 기본값 그대로(cudnn TF32 켜짐 — eval_v8fix(_grufp32).py 와 같음, 10-16 U-Net·SS2D 실행), '
                        'off = 트레이너와 같이 matmul·cudnn TF32 끔 (docs/v8fix_setup_audit_2026-10-14.md §1-6)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    for k in ('seq', 'ckpt', 'data_path', 'out_dir'):
        if getattr(a, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} 필요')

    import eval_unet_only_v8 as E
    from eval_paired_v8_nodc import slice_metrics
    os.makedirs(a.out_dir, exist_ok=True)
    pv_path = os.path.join(a.out_dir, f'per_volume_{a.seq}.csv')
    ps_path = os.path.join(a.out_dir, f'per_slice_{a.seq}.csv')
    sm_path = os.path.join(a.out_dir, f'summary_{a.seq}.json')
    if os.path.exists(pv_path) and not a.force:
        raise SystemExit(f'{pv_path} 가 이미 있음 (--force)')

    if a.tf32 == 'off':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tf32_state = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
    print(f'TF32 {a.tf32}: {tf32_state}')
    device = torch.device(a.device)
    base, Normalized = build(a.seq, device)
    base.load_state_dict(torch.load(a.ckpt, map_location=device))
    base.eval()
    model = Normalized(base).eval()
    n_par = sum(q.numel() for q in base.parameters())
    print(f'[{a.seq}] ckpt {a.ckpt} ({n_par / 1e6:.1f}M params), device {device}')

    ds = E.build_val_dataset(a.data_path)
    files = []
    for fp, s, _ in ds.samples:
        if not files or files[-1][0] != fp:
            files.append([fp, 0])
        files[-1][1] += 1
    if a.max_files > 0:
        files = files[:a.max_files]
    total = sum(n for _, n in files)
    print(f'{len(files)} 볼륨 / {total} 슬라이스 (데이터셋 전체 {len(ds)})')

    import h5py
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.num_workers)
    it = iter(loader)
    vol_rows, sl_rows = [], []
    t0 = time.time()
    idx = 0
    with torch.no_grad():
        for vi, (fp, n) in enumerate(files):
            with h5py.File(fp, 'r') as f:
                gt = f['reconstruction_rss'][()].astype(np.float32)          # (S, H, W) 원래 크기 그대로
            S, H, W = gt.shape
            assert S == n, (fp, S, n)
            preds, gts384, bms, sl = [], [], [], []
            for s in range(n):
                sample = next(it)
                f_s, s_idx, _ = ds.samples[idx]
                assert f_s == fp and s_idx == s, (idx, f_s, s_idx, fp, s)
                x_ksp = sample['data'].float().to(device)
                x_img = sample['data_img'].float().to(device)
                ref = sample['label'].float().to(device)
                bm = sample['brain_mask'].float().to(device)
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model(x_img, x_ksp, sample['mask'].float().to(device), sample['sens'].float().to(device))
                else:
                    out = model(x_img, x_ksp, sample['mask'].float().to(device), sample['sens'].float().to(device))
                m = slice_metrics(out, ref, bm)
                o = out.float()[0, 0].cpu().numpy()
                preds.append(center_crop_np(o, (H, W)) / VAL_AMP_Y)
                gts384.append(ref[0, 0].cpu().numpy())
                bms.append(bm[0, 0].cpu().numpy() > 0.5)
                sl.append(dict(idx=idx, file=os.path.basename(fp), slice_idx=s, **{k: m[k] for k in ('ssim', 'psnr', 'nmse', 'l1')}))
                idx += 1
            pred = np.stack(preds)
            bm384 = np.stack(bms)
            bm_hw = center_crop_np(bm384, (H, W))
            lab_check = float(np.max(np.abs(center_crop_np(np.stack(gts384), (H, W)) / VAL_AMP_Y - gt)) / max(gt.max(), 1e-30))
            off = official_volume(gt, pred, bm_hw)
            # 민감도: 384 틀의 라벨·출력으로 계산 (마스크 지표와 같은 틀)
            pred384 = np.zeros_like(np.stack(gts384))
            a0, b0 = (384 - H) // 2, (384 - W) // 2
            pred384[:, a0:a0 + H, b0:b0 + W] = pred * VAL_AMP_Y
            vr = masked_ssim_volrange(np.stack(gts384), pred384, bm384)
            for r, v in zip(sl, off['ssim_slices']):
                r['ssim_official'] = v
            sl_rows += sl
            vol_rows.append(dict(file=os.path.basename(fp), n=n, H=H, W=W,
                                 ssim_mask=float(np.mean([r['ssim'] for r in sl])),
                                 psnr_mask=float(np.mean([r['psnr'] for r in sl])),
                                 nmse_mask=float(np.mean([r['nmse'] for r in sl])),
                                 ssim_mask_volrange=vr,
                                 ssim_off=off['ssim'], psnr_off=off['psnr'], nmse_off=off['nmse'], maxval=off['maxval'],
                                 ssim_off_in=off['ssim_in'], ssim_off_out=off['ssim_out'], frac_in=off['frac_in'],
                                 decomp_maxabs=off['decomp_maxabs'], label_vs_rss_relmax=lab_check))
            if (vi + 1) % 25 == 0 or vi + 1 == len(files):
                el = time.time() - t0
                print(f'  {vi + 1}/{len(files)} 볼륨, {idx} 슬라이스, {el / 60:.1f} 분 (≈{el / idx:.3f} s/slice)', flush=True)

    keys_v = list(vol_rows[0])
    with open(pv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys_v)
        w.writeheader()
        w.writerows(vol_rows)
    with open(ps_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'file', 'slice_idx', 'ssim', 'psnr', 'nmse', 'l1', 'ssim_official'])
        w.writeheader()
        w.writerows(sl_rows)

    def ms(k):
        v = np.array([r[k] for r in vol_rows], dtype=float)
        return dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else None)
    summ = dict(seq=a.seq, ckpt=a.ckpt, data_path=a.data_path, n_vol=len(vol_rows), n_slice=len(sl_rows),
                device=str(device), tf32=dict(option=a.tf32, **tf32_state), seconds=round(time.time() - t0, 1),
                metrics={k: ms(k) for k in ('ssim_mask', 'psnr_mask', 'nmse_mask', 'ssim_mask_volrange', 'ssim_off', 'psnr_off',
                                            'nmse_off', 'ssim_off_in', 'ssim_off_out', 'frac_in')},
                checks=dict(decomp_maxabs=max(r['decomp_maxabs'] for r in vol_rows),
                            label_vs_rss_relmax=max(r['label_vs_rss_relmax'] for r in vol_rows)))
    if a.check_csv:
        ref = {}
        with open(a.check_csv, newline='') as f:
            for r in csv.DictReader(f):
                ref[(r['file'], int(r['slice_idx']))] = float(r['ssim'])
        d = [abs(r['ssim'] - ref[(r['file'], r['slice_idx'])]) for r in sl_rows if (r['file'], r['slice_idx']) in ref]
        summ['checks']['vs_csv'] = dict(csv=a.check_csv, n_matched=len(d), n_rows=len(sl_rows),
                                        ssim_maxabs=float(max(d)) if d else None, ssim_meanabs=float(np.mean(d)) if d else None)
    with open(sm_path, 'w') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
