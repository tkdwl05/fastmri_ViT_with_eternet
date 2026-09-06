"""
다중 모델 정성 비교 (교수님 ETER-net 논문 그림 형식) — GT / Zero-filled / U-Net† / E2E-VarNet† /
PromptMR+ / bi-GRU (original) / SS2D (controlled) / Enhanced SS2D, 384×384 · R4 · brain-masked.

`visualize_v9_compare.py` 의 **단일 파이프라인 공정 비교** 원칙을 그대로 따른다:
  - 모든 방법이 같은 슬라이스 · 같은 R4/cf0.08 equispaced mask · 같은 GT(384 RSS) · 같은
    brain mask(Otsu×0.4 + largest CC) · 같은 16-coil 절단 측정값을 받는다.
  - 표시·지표 전에 모든 재구성을 brain-mask 안 per-slice LS 강도 정합(α=⟨r,g⟩/⟨r,r⟩)한다
    (leaderboard/PromptMR+ 출력 스케일이 제각각이라 필수; 우리 팔은 α≈1).
  - 지표식은 `v8_eter_pure/eval_paired_v8_nodc.py` / `eval_zero_filled_v8.py` 와 동일.

† U-Net / E2E-VarNet = fastMRI brain leaderboard 공개 가중치(train+val 학습 → 우리 val 이 학습셋에
  포함, `docs/frontier_baselines_plan.md`·논문 §기준선 캐비엇) — 참고선. PromptMR+ (fm-brain, train-only
  학습 → 누수 없음)는 `external/PromptMR-plus` 코드 + `external/weights/…ep44.ckpt` 로 추론하며,
  인접 5슬라이스(z±2, 경계 복제) 스택을 우리 측정값에서 유도해 공급한다.

CPU 전용 실행을 전제로 설계했다(GPU0 은 공정성 스위트 점유). SS2D 계열은 mamba_ssm CUDA 커널
(`selective_scan_fn`)을 같은 패키지의 순수-PyTorch 참조 구현(`selective_scan_ref`)으로 바꿔 치기해
CPU 에서 돌린다(원본 파일 무수정 — 런타임 몽키패치, 수식 동일·fp32).

실행 (저장소 루트에서; 학습 런과 CPU 를 나눠 쓰므로 nice + 스레드 제한):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python visualize_multimodel_compare.py \
      --slice-indices 0,1321,2643,3368
출력: results/vis/multimodel_compare/recon_<idx>.npz (gt/brain_mask/각 방법의 LS 정합 재구성),
      metrics_<idx>.json, compare_<idx>.png (진단용 2행×8열), metrics_summary.txt
논문 그림(Fig.3)은 이 npz 를 `paper/make_fig3_qualitative.py` 가 조판한다.
"""

import os
import sys
import json
import time
import argparse
import importlib

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.append(os.path.join(_HERE, 'dataloaders'))
sys.path.append(os.path.join(_HERE, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_HERE, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_HERE, 'v8_eter_pure', 'configs'))
sys.path.append(os.path.join(_HERE, 'v9_mamba_unleashed', 'configs'))

from dataloader_h5_v5 import FastMRI_H5_Dataloader

CENTER_FRACTION = 0.08
ACCEL = 4
AMP_X_IMG = 1e6      # dataloader val_amp_X_img
PMR_ROOT = os.path.join(_HERE, 'external', 'PromptMR-plus')
PMR_CKPT = os.path.join(_HERE, 'external', 'weights', 'promptmr-plus-fm-brain-ep44.ckpt')

# (key, 표시 이름, 각주) — 열 순서 = 논문 그림 순서 (기준선 → 원본 → 치환 → 강화)
METHODS = [
    ('zf',       'Zero-filled',        ''),
    ('unet',     'U-Net',              '†'),
    ('varnet',   'E2E-VarNet',         '†'),
    ('promptmr', 'PromptMR+',          ''),
    ('gru',      'bi-GRU (original)',  ''),
    ('ss2d',     'SS2D (controlled)',  ''),
    ('v9',       'Enhanced SS2D',      ''),
]


# ──────────────────────────────────────────────
#  공통 헬퍼 (eval 스크립트와 동일 공식)
# ──────────────────────────────────────────────

def unpack_complex(packed):
    """(2C,H,W) packed real/imag → (C,H,W) complex."""
    return packed[0::2].astype(np.float32) + 1j * packed[1::2].astype(np.float32)


def ls_scale(recon, gt, mask):
    m = mask > 0.5
    if not m.any():
        return recon
    r, g = recon[m], gt[m]
    denom = float((r * r).sum())
    if denom < 1e-12:
        return recon
    return (float((r * g).sum()) / denom) * recon


def slice_metrics_np(out, ref, m):
    """eval_paired_v8_nodc.slice_metrics 와 동일 공식 (composite 제외)."""
    m = m.astype(np.float32)
    m_sum = max(float(m.sum()), 1.0)
    diff_sq_sum = float(((out - ref) ** 2 * m).sum())
    mse = diff_sq_sum / m_sum
    ref_max_in_mask = max(float((ref * m).max()), 1e-10)
    psnr = float(20.0 * np.log10(ref_max_in_mask / np.sqrt(max(mse, 1e-10))))
    ref_sq_sum = max(float((ref ** 2 * m).sum()), 1e-10)
    nmse = diff_sq_sum / ref_sq_sum
    mb = m > 0.5
    ssim = 0.0
    if mb.any():
        t_in = ref[mb]
        dr = float(t_in.max() - t_in.min())
        if dr > 0:
            _, smap = compare_ssim(ref, out, data_range=dr, full=True)
            ssim = float(smap[mb].mean())
    return {'ssim': ssim, 'psnr': psnr, 'nmse': nmse}


def _tensor(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).float().to(device)


# ──────────────────────────────────────────────
#  모델 로더 / 러너
# ──────────────────────────────────────────────

def _load_state(model, ckpt_path, device, weights_only=False):
    obj = torch.load(ckpt_path, map_location=device, weights_only=weights_only)
    state = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


def patch_selective_scan_for_cpu():
    """CPU 에서는 mamba_ssm CUDA 커널 대신 참조 구현(순수 PyTorch, 동일 수식)을 쓴다.
    ss2d.py / ss2d_v9.py 는 `from … import selective_scan_fn` 으로 이름을 바인딩하므로
    두 모듈 네임스페이스의 이름을 교체한다 (원본 파일 무수정)."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    n = 0
    for mod in ('ss2d', 'ss2d_v9'):
        m = importlib.import_module(mod)
        m.selective_scan_fn = selective_scan_ref
        n += 1
    print(f'  [CPU] selective_scan_fn → selective_scan_ref 로 교체 ({n} 모듈)')


def load_pure_gru(ckpt, device):
    import myConfig_pure_eter_v8 as c
    from u_pure_eternet_gru import PureETER_GRU
    m = PureETER_GRU(dim=c.IMAGE_SIZE[0], n_coil=c.N_COIL,
                     n_hidden_1=c.N_HIDDEN_LRNN_1, n_hidden_2=c.N_HIDDEN_LRNN_2,
                     unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF, use_dc=False)
    return _load_state(m, ckpt, device)


def load_pure_ss2d(ckpt, device):
    import myConfig_pure_eter_v8 as c
    from u_pure_eternet_ss2d import PureETER_SS2D
    m = PureETER_SS2D(n_coil=c.N_COIL, n_hidden_2=c.N_HIDDEN_LRNN_2,
                      unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF,
                      ss2d_d_inner=c.SS2D_D_INNER, ss2d_d_state=c.SS2D_D_STATE, use_dc=False)
    return _load_state(m, ckpt, device)


def load_pure_ss2d_v9(ckpt, device):
    import myConfig_ss2d_v9 as c
    from u_pure_eternet_ss2d_v9 import PureETER_SS2D_V9
    m = PureETER_SS2D_V9(n_coil=c.N_COIL, out_ch=c.SS2D_OUT_CH,
                         unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF,
                         ss2d_d_inner=c.SS2D_D_INNER, ss2d_d_state=c.SS2D_D_STATE,
                         ss2d_n_blocks=c.SS2D_N_BLOCKS, ss2d_dropout=c.SS2D_DROPOUT,
                         ss2d_use_checkpoint=False, ss2d_downsample=c.SS2D_DOWNSAMPLE)
    return _load_state(m, ckpt, device)


def load_unet(ckpt, device):
    from fastmri.models import Unet
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    return _load_state(m, ckpt, device, weights_only=True)


def load_varnet(ckpt, device):
    from fastmri.models import VarNet
    m = VarNet(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)
    return _load_state(m, ckpt, device, weights_only=True)


def load_promptmr(ckpt, device):
    """PromptMR+ (fm-brain). Lightning 모듈을 거치지 않고 `models.promptmr_v2.PromptMR` 을 ckpt 의
    hyper_parameters 로 직접 생성 → state_dict 의 `promptmr.` 접두사를 벗겨 로드."""
    if PMR_ROOT not in sys.path:
        sys.path.insert(0, PMR_ROOT)          # 그들의 `models` 패키지가 우리 models/ 보다 먼저
    for k in [k for k in sys.modules if k == 'models' or k.startswith('models.')]:
        del sys.modules[k]
    pmr = importlib.import_module('models.promptmr_v2')
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    hp = obj['hyper_parameters']
    keys = ['num_cascades', 'num_adj_slices', 'n_feat0', 'feature_dim', 'prompt_dim', 'sens_n_feat0',
            'sens_feature_dim', 'sens_prompt_dim', 'len_prompt', 'prompt_size', 'n_enc_cab', 'n_dec_cab',
            'n_skip_cab', 'n_bottleneck_cab', 'no_use_ca', 'learnable_prompt', 'adaptive_input',
            'n_buffer', 'n_history', 'use_sens_adj']
    model = pmr.PromptMR(**{k: hp[k] for k in keys})
    sd = {k[len('promptmr.'):]: v for k, v in obj['state_dict'].items() if k.startswith('promptmr.')}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f'PromptMR+ unexpected keys: {unexpected[:5]}'
    assert not missing, f'PromptMR+ missing keys: {missing[:5]}'
    # 업스트림 버그 우회: PromptMRBlock.forward 가 정의되지 않은 self.n_buffer 를 참조
    # (promptmr_v2.py:249, 의도 = self.model.n_buffer). 외부 clone 은 무수정, 런타임 속성만 부여.
    for blk in model.cascades:
        if not hasattr(blk, 'n_buffer'):
            blk.n_buffer = blk.model.n_buffer
    model.num_adj_slices_cfg = int(hp['num_adj_slices'])
    return model.to(device).eval()


def run_pure(model, s, device):
    """GRU / SS2D / v9 공용 — forward(x_img, x_ksp, mask, sens). CPU 는 fp32 (autocast 없음)."""
    d, di = _tensor(s['data'], device), _tensor(s['data_img'], device)
    mk, se = _tensor(s['mask'], device), _tensor(s['sens'], device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = model(di, d, mk, se)
        else:
            out = model(di, d, mk, se)
    return out.squeeze().float().cpu().numpy()


def zero_filled(s):
    img_c = unpack_complex(s['data_img']) / AMP_X_IMG
    return np.sqrt((np.abs(img_c) ** 2).sum(0)).astype(np.float32)


def prep_unet(s):
    """워커에서 실행 가능한 입력 준비 (모델 불필요): zero-filled RSS (H,W) float32."""
    return {'zf': zero_filled(s)}


def forward_unet(model, inp, device):
    """zero-filled RSS → z-score(clamp ±6, fastmri 방식) → U-Net → unnorm."""
    t = torch.as_tensor(inp['zf']).to(device)
    mean = t.mean()
    std = t.std().clamp(min=1e-8)
    x = ((t - mean) / std).clamp(-6.0, 6.0)[None, None]
    with torch.no_grad():
        out = model(x).squeeze()
    return (out * std + mean).float().cpu().numpy()


def run_unet(model, s, device):
    return forward_unet(model, prep_unet(s), device)


def _kspace_keep_coils(packed_ksp, keep=None):
    ksp_c = unpack_complex(packed_ksp)
    if keep is None:
        keep = np.abs(ksp_c).reshape(ksp_c.shape[0], -1).sum(1) > 0   # zero-pad 코일 제거 (sens NaN 방지)
    return ksp_c[keep], keep


def _mask_1d(s):
    mask_arr = s['mask']
    return mask_arr.reshape(-1, mask_arr.shape[-1])[0]


def _ksp_to_tensor(ksp_c):
    """(C,H,W) complex → (1,C,H,W,2) float (fastmri/PromptMR+ 입력 규약)."""
    return torch.stack([torch.from_numpy(np.ascontiguousarray(ksp_c.real)),
                        torch.from_numpy(np.ascontiguousarray(ksp_c.imag))], dim=-1).unsqueeze(0).float()


def prep_varnet(s):
    """실측 코일만 남긴 masked k-space(unit-max) + 1D mask — 워커에서 실행 가능."""
    ksp_c, keep = _kspace_keep_coils(s['data'])
    ksp_c = ksp_c / (float(np.abs(ksp_c).max()) + 1e-12)
    return {'ksp': _ksp_to_tensor(ksp_c), 'mask1d': torch.from_numpy(_mask_1d(s) > 0.5),
            'coils_kept': int(keep.sum())}


def forward_varnet(model, inp, device):
    mk = inp['ksp'].to(device)
    W = mk.shape[-2]
    mask_vn = inp['mask1d'].view(1, 1, 1, W, 1).to(device)
    n_low = int(round(W * CENTER_FRACTION))
    with torch.no_grad():
        out = model(mk, mask_vn, num_low_frequencies=n_low)
    return out.squeeze().float().cpu().numpy()


def run_varnet(model, s, device):
    """masked k-space(실측 코일만) + 1D mask + n_low → VarNet RSS (eval_paired_baselines.run_varnet 동일)."""
    return forward_varnet(model, prep_varnet(s), device)


def prep_promptmr(s, neighbors):
    """인접 슬라이스 스택 (z-2..z+2, 경계 복제 = 그들 SliceDataset 규약) 을 코일축에 쌓는다.
    neighbors: 중심 포함 num_adj 개 sample dict 리스트 (슬라이스 순). 코일 집합은 중심 슬라이스 기준."""
    _, keep = _kspace_keep_coils(s['data'])
    stack = []
    for nb in neighbors:
        k_c, _ = _kspace_keep_coils(nb['data'], keep)
        stack.append(k_c)
    ksp = np.concatenate(stack, axis=0)                          # (adj*C, H, W) complex
    ksp = ksp / (float(np.abs(ksp).max()) + 1e-12)
    return {'ksp': _ksp_to_tensor(ksp), 'mask1d': torch.from_numpy(_mask_1d(s) > 0.5),
            'coils_kept': int(keep.sum())}


def forward_promptmr(model, inp, device):
    mk = inp['ksp'].to(device)
    W = mk.shape[-2]
    mask_t = inp['mask1d'].view(1, 1, 1, W, 1).to(device)
    n_low = torch.tensor([int(round(W * CENTER_FRACTION))], device=device)
    with torch.no_grad():
        out = model(mk, mask_t, n_low, mask_type=('cartesian',), compute_sens_per_coil=True)
    return out['img_pred'].squeeze().float().cpu().numpy()


def run_promptmr(model, s, neighbors, device):
    return forward_promptmr(model, prep_promptmr(s, neighbors), device)


# ──────────────────────────────────────────────
#  슬라이스 선택 / 이웃
# ──────────────────────────────────────────────

def resolve_slice_spec(spec_path, h5):
    with open(spec_path) as f:
        spec = json.load(f)
    pos = {(os.path.basename(fp), s): i for i, (fp, s, _) in enumerate(h5.samples)}
    idx = [pos[(it['file'], int(it['slice']))] for it in spec['slices'] if (it['file'], int(it['slice'])) in pos]
    print(f"  슬라이스 스펙 '{spec.get('name', spec_path)}' → 인덱스 {idx}")
    return idx


def neighbor_indices(h5, idx, num_adj):
    """같은 파일 안에서 z±(num_adj//2), 경계는 복제 (PromptMR+ SliceDataset._get_frames_indices 규약)."""
    fp, s, _ = h5.samples[idx]
    pos = {(f, k): i for i, (f, k, _) in enumerate(h5.samples) if f == fp}
    n = len(pos)
    half = num_adj // 2
    return [pos[(fp, min(max(s + d, 0), n - 1))] for d in range(-half, half + 1)]


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='다중 모델 정성 비교 (GT/ZF/U-Net†/VarNet†/PromptMR+/bi-GRU/SS2D/Enhanced)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--gru-ckpt', default='logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_best.pt')
    p.add_argument('--ss2d-ckpt', default='logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt')
    p.add_argument('--v9-ckpt', default='logs/PureETER_SS2D_V9_unleashed_R4_brain384/ss2d_v9_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--varnet-ckpt', default='models/pretrained/varnet_brain_leaderboard_state_dict.pt')
    p.add_argument('--promptmr-ckpt', default=PMR_CKPT)
    p.add_argument('--out-dir', default='results/vis/multimodel_compare')
    p.add_argument('--slice-spec', default='visualize_slices_canonical.json')
    p.add_argument('--slice-indices', default=None, help='쉼표구분 dataset 인덱스 (스펙보다 우선)')
    p.add_argument('--methods', default=','.join(k for k, _, _ in METHODS))
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--err-vmax-frac', type=float, default=0.10)
    p.add_argument('--skip-existing', action='store_true', help='npz 에 이미 있는 (슬라이스, 방법) 은 건너뜀')
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wanted = [k for k in args.methods.split(',') if k]
    print('=' * 72)
    print(' 다중 모델 정성 비교 — 단일 파이프라인 (384, R4, brain-masked, per-slice LS 정합)')
    print(f'  device={device}, threads={args.threads}, methods={wanted}')
    print('=' * 72)
    if device.type != 'cuda':
        patch_selective_scan_for_cpu()

    h5 = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=384,
                               acceleration=ACCEL, center_fraction=CENTER_FRACTION,
                               random_mask=False, augment=False)
    if args.slice_indices:
        indices = [int(x) for x in args.slice_indices.split(',') if x.strip()]
    else:
        indices = resolve_slice_spec(args.slice_spec, h5)
    print(f'\n총 val 슬라이스: {len(h5)} → 선택 {len(indices)}개: {indices}')
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 슬라이스 캐시
    samples, meta = {}, {}
    for idx in indices:
        samples[idx] = h5[idx]
        fp, si, _ = h5.samples[idx]
        meta[idx] = {'idx': idx, 'file': os.path.basename(fp), 'slice': int(si),
                     'contrast': os.path.basename(fp).split('_')[2]}
    recon = {idx: {} for idx in indices}
    metrics = {idx: {} for idx in indices}
    timing = {}

    def npz_path(idx):
        return os.path.join(args.out_dir, f'recon_{idx:04d}.npz')

    if args.skip_existing:
        for idx in indices:
            if os.path.exists(npz_path(idx)):
                z = np.load(npz_path(idx))
                for k in z.files:
                    if k.startswith('rec_'):
                        recon[idx][k[4:]] = z[k]
                jp = os.path.join(args.out_dir, f'metrics_{idx:04d}.json')
                if os.path.exists(jp):
                    metrics[idx].update(json.load(open(jp)).get('metrics', {}))

    def finish(idx, key, rec, t0):
        gt = samples[idx]['label'][0].astype(np.float32)
        bm = samples[idx]['brain_mask'][0].astype(np.float32)
        rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rec = ls_scale(rec, gt, bm)
        recon[idx][key] = rec
        metrics[idx][key] = slice_metrics_np(rec, gt, bm)
        dt = time.time() - t0
        timing.setdefault(key, []).append(dt)
        m = metrics[idx][key]
        print(f'    idx {idx:5d} {key:9s} PSNR {m["psnr"]:6.2f} dB  SSIM {m["ssim"]:.4f}  nMSE {100*m["nmse"]:.3f}%  ({dt:.0f}s)',
              flush=True)

    # 2) 방법별 outer loop (모델 1개씩 메모리에)
    loaders = {
        'unet': (args.unet_ckpt, load_unet, run_unet),
        'varnet': (args.varnet_ckpt, load_varnet, run_varnet),
        'promptmr': (args.promptmr_ckpt, load_promptmr, None),
        'gru': (args.gru_ckpt, load_pure_gru, run_pure),
        'ss2d': (args.ss2d_ckpt, load_pure_ss2d, run_pure),
        'v9': (args.v9_ckpt, load_pure_ss2d_v9, run_pure),
    }
    for key in wanted:
        todo = [i for i in indices if not (args.skip_existing and key in recon[i])]
        if not todo:
            print(f'\n[{key}] 전부 캐시됨 — 건너뜀'); continue
        if key == 'zf':
            print('\n[zf] zero-filled RSS')
            for idx in todo:
                finish(idx, 'zf', zero_filled(samples[idx]), time.time())
            continue
        ckpt, loader, runner = loaders[key]
        if not os.path.exists(ckpt):
            print(f'\n[{key}] ckpt 없음 → 건너뜀 ({ckpt})'); continue
        print(f'\n[{key}] 로드: {ckpt}', flush=True)
        t0 = time.time()
        model = loader(ckpt, device)
        print(f'  로드 {time.time()-t0:.0f}s, params {sum(p.numel() for p in model.parameters())/1e6:.1f}M', flush=True)
        for idx in todo:
            t0 = time.time()
            if key == 'promptmr':
                nb_idx = neighbor_indices(h5, idx, model.num_adj_slices_cfg)
                nbs = [samples[i] if i in samples else h5[i] for i in nb_idx]
                rec = run_promptmr(model, samples[idx], nbs, device)
            else:
                rec = runner(model, samples[idx], device)
            finish(idx, key, rec, t0)
        del model
        # 3) 슬라이스별 npz/json 저장 (방법 하나 끝날 때마다 — 중단돼도 결과 보존)
        for idx in indices:
            gt = samples[idx]['label'][0].astype(np.float32)
            bm = samples[idx]['brain_mask'][0].astype(np.float32)
            np.savez_compressed(npz_path(idx), gt=gt, brain_mask=bm,
                                **{f'rec_{k}': v for k, v in recon[idx].items()})
            json.dump({'meta': meta[idx], 'metrics': metrics[idx],
                       'note': 'per-slice LS scale-aligned inside brain mask; metrics = eval_paired formulas'},
                      open(os.path.join(args.out_dir, f'metrics_{idx:04d}.json'), 'w'), indent=1)
    for idx in indices:   # zf-only 실행 등 위 루프에서 저장되지 않은 경우
        if not os.path.exists(npz_path(idx)):
            gt = samples[idx]['label'][0].astype(np.float32)
            bm = samples[idx]['brain_mask'][0].astype(np.float32)
            np.savez_compressed(npz_path(idx), gt=gt, brain_mask=bm,
                                **{f'rec_{k}': v for k, v in recon[idx].items()})
            json.dump({'meta': meta[idx], 'metrics': metrics[idx]},
                      open(os.path.join(args.out_dir, f'metrics_{idx:04d}.json'), 'w'), indent=1)

    # 4) 진단 PNG (슬라이스당 2행 × (1+방법수) 열)
    hot_bad = plt.get_cmap('hot').copy(); hot_bad.set_bad(color='black')
    names = {k: n + f for k, n, f in METHODS}
    for idx in indices:
        gt = samples[idx]['label'][0].astype(np.float32)
        bm = samples[idx]['brain_mask'][0].astype(np.float32)
        gmax = max(float(gt.max()), 1e-8)
        cols = ['GT'] + [k for k, _, _ in METHODS if k in recon[idx]]
        fig, axes = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6))
        for j, k in enumerate(cols):
            ax0, ax1 = axes[0, j], axes[1, j]
            if k == 'GT':
                ax0.imshow(gt / gmax, cmap='gray', vmin=0, vmax=1); ax0.set_title('Ground truth', fontsize=9)
                ax1.set_visible(False)
            else:
                r = recon[idx][k] / gmax
                m = metrics[idx][k]
                ax0.imshow(r, cmap='gray', vmin=0, vmax=1)
                ax0.set_title(f'{names[k]}\n{m["psnr"]:.2f} dB / {m["ssim"]:.4f}', fontsize=8)
                im = ax1.imshow(np.where(bm > 0.5, np.abs(r - gt / gmax), np.nan), cmap=hot_bad,
                                vmin=0, vmax=args.err_vmax_frac)
            ax0.axis('off'); ax1.axis('off')
        fig.colorbar(im, ax=axes[1, -1], fraction=0.046)
        mt = meta[idx]
        fig.suptitle(f"#{idx} {mt['file']} slice {mt['slice']} ({mt['contrast']}) — 384, R4, brain-masked, LS-aligned; "
                     f"† = leaderboard weights trained on train+val", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'compare_{idx:04d}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)

    # 5) 요약
    lines = ['# 다중 모델 정성 비교 — per-slice 지표 (brain-masked, per-slice LS 정합, 384·R4)',
             f'slices: {indices}', f'device: {device}, threads: {args.threads}', '',
             '| idx | file | slice | ' + ' | '.join(names[k] for k, _, _ in METHODS) + ' |',
             '|---|---|---|' + '---:|' * len(METHODS)]
    for idx in indices:
        cells = []
        for k, _, _ in METHODS:
            m = metrics[idx].get(k)
            cells.append(f'{m["psnr"]:.2f} / {m["ssim"]:.4f}' if m else '–')
        lines.append(f'| {idx} | {meta[idx]["file"]} | {meta[idx]["slice"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '(cell = PSNR dB / SSIM)', '', '## 방법별 평균 추론 시간 (s/slice, 이 장비·설정)']
    for k, ts in timing.items():
        lines.append(f'- {k}: {np.mean(ts):.1f}s (n={len(ts)})')
    msg = '\n'.join(lines)
    print('\n' + msg)
    open(os.path.join(args.out_dir, 'metrics_summary.txt'), 'w').write(msg + '\n')
    print(f'\n저장: {args.out_dir}/')


if __name__ == '__main__':
    main()
