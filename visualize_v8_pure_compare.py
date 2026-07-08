"""
v8 Pure ETER-Net 4-way 공정 비교 시각화: GT / U-Net / GRU(no-DC) / SS2D(no-DC) (384×384).

`visualize_v7_titan_compare.py` 클론 — ViT 트랙(ETER v7_titan/SS2D v7_titan) 대신
교수님 순수 ETER-Net(ViT 없음, DC 없음)의 GRU/SS2D no-DC arm 을 비교한다
(docs/v8_eter_pure_rnn_vs_ss2d.md, aggregate composite: SS2D 0.9200 > GRU 0.9182).

핵심 = **단일 파이프라인 공정 비교**:
  - 모든 모델이 같은 슬라이스, 같은 R4/cf0.08 mask, 같은 GT(384 RSS),
    같은 brain_mask(Otsu×0.4+largest CC)로 평가된다.
  - U-Net 도 별도 fastmri SliceDataset 이 아니라, GRU/SS2D 와 동일
    FastMRI_H5_Dataloader 의 masked 측정값에서 입력을 유도한다 → mask/GT 완전 일치.
  - 지표: brain-mask 안에서 masked SSIM/PSNR. recon 은 per-slice LS scale 로 GT 에
    정합(magnitude 단위 bias 제거).

⚠ 베이스라인 캐비엇: U-Net = fastmri brain leaderboard 사전학습 가중치 — 참고용
  (v7_titan 비교 스크립트와 동일 caveat, docs/v8_eter_pure_rnn_vs_ss2d.md 참고).

Layout (슬라이스당 2행×4열):
  Row 0: GT | U-Net | GRU (no-DC) | SS2D (no-DC)   (recon)
  Row 1: (blank) | U-Net err | GRU err | SS2D err   (masked |err|)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

from fastmri.models import Unet

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_HERE, 'dataloaders'))
sys.path.append(os.path.join(_HERE, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_HERE, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_HERE, 'v8_eter_pure', 'configs'))

from dataloader_h5_v5 import FastMRI_H5_Dataloader

CENTER_FRACTION = 0.08
ACCEL = 4
AMP_X_KSP = 1e4   # dataloader val_amp_X_ksp
AMP_X_IMG = 1e6   # dataloader val_amp_X_img


# ──────────────────────────────────────────────
#  공통 헬퍼
# ──────────────────────────────────────────────

def to_tensor_sample(d):
    out = {}
    for k, v in d.items():
        out[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
    return out


def unpack_complex(packed):
    """(2C,H,W) packed real/imag (real=짝수 idx, imag=홀수 idx) → (C,H,W) complex."""
    return packed[0::2].astype(np.float32) + 1j * packed[1::2].astype(np.float32)


def ls_scale(recon, gt, mask):
    """brain-mask 안에서 α = ⟨recon,gt⟩/⟨recon,recon⟩ 최소제곱 scale 정합."""
    m = mask > 0.5
    if not m.any():
        return recon
    r = recon[m]
    g = gt[m]
    denom = float((r * r).sum())
    if denom < 1e-12:
        return recon
    return (float((r * g).sum()) / denom) * recon


def masked_psnr(recon, gt, mask):
    m = mask > 0.5
    if not m.any():
        return 0.0
    mse = float(np.mean((recon[m] - gt[m]) ** 2))
    if mse <= 0:
        return float('inf')
    return 20.0 * np.log10(float(gt[m].max()) / np.sqrt(mse))


def masked_ssim(recon, gt, mask):
    """v8 run_val 과 동일 방식: full SSIM map → mask 안 평균, data_range=mask 안 max-min."""
    m = mask > 0.5
    if not m.any():
        return 0.0
    t_in = gt[m]
    dr = float(t_in.max() - t_in.min())
    if dr <= 0:
        return 0.0
    _, smap = compare_ssim(gt, recon, data_range=dr, full=True)
    return float(smap[m].mean())


# ──────────────────────────────────────────────
#  모델 로더
# ──────────────────────────────────────────────

def _load_state(model, ckpt_path, device, weights_only=False):
    obj = torch.load(ckpt_path, map_location=device, weights_only=weights_only)
    state = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


def load_pure_gru(ckpt, device):
    import myConfig_pure_eter_v8 as c
    from u_pure_eternet_gru import PureETER_GRU
    m = PureETER_GRU(
        dim=c.IMAGE_SIZE[0], n_coil=c.N_COIL,
        n_hidden_1=c.N_HIDDEN_LRNN_1, n_hidden_2=c.N_HIDDEN_LRNN_2,
        unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF, use_dc=False,
    )
    return _load_state(m, ckpt, device)


def load_pure_ss2d(ckpt, device):
    import myConfig_pure_eter_v8 as c
    from u_pure_eternet_ss2d import PureETER_SS2D
    m = PureETER_SS2D(
        n_coil=c.N_COIL, n_hidden_2=c.N_HIDDEN_LRNN_2,
        unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF,
        ss2d_d_inner=c.SS2D_D_INNER, ss2d_d_state=c.SS2D_D_STATE, use_dc=False,
    )
    return _load_state(m, ckpt, device)


def load_unet(ckpt, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    return _load_state(m, ckpt, device, weights_only=True)


# ──────────────────────────────────────────────
#  모델별 추론 (모두 동일 H5 sample 에서 입력 유도)
# ──────────────────────────────────────────────

def run_ss2d(model, sample, device):
    """GRU/SS2D pure arm 공용 — forward(x_img, x_ksp, mask, sens) 시그니처 동일."""
    d  = sample['data'].unsqueeze(0).float().to(device)
    di = sample['data_img'].unsqueeze(0).float().to(device)
    mk = sample['mask'].unsqueeze(0).float().to(device)
    se = sample['sens'].unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(di, d, mk, se)
    return out.squeeze().float().cpu().numpy()


def run_unet(model, sample, device):
    """masked 측정값에서 zero-filled RSS → z-score(clamp ±6, fastmri 방식) → U-Net → unnorm."""
    di = sample['data_img'].numpy()                 # (32,H,W) packed aliased image *1e6
    img_c = unpack_complex(di) / AMP_X_IMG          # (16,H,W) complex aliased image
    zf_rss = np.sqrt((np.abs(img_c) ** 2).sum(0)).astype(np.float32)  # (H,W)
    t = torch.from_numpy(zf_rss).to(device)
    mean = t.mean()
    std = t.std().clamp(min=1e-8)
    x = ((t - mean) / std).clamp(-6.0, 6.0)[None, None]
    with torch.no_grad():
        out = model(x).squeeze()
    return (out * std + mean).float().cpu().numpy()


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='v8 pure ETER-Net 4-way 공정 비교 (GT/U-Net/GRU/SS2D, no-DC)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--gru-ckpt', default='logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_best.pt')
    p.add_argument('--ss2d-ckpt', default='logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--out-dir', default='results/vis/v8_pure_eternet_compare')
    p.add_argument('--num-samples', type=int, default=12)
    p.add_argument('--slice-indices', default=None,
                   help='쉼표구분 인덱스 직접 지정 (예: 0,660,...). 미지정 시 linspace(0,len-1,num_samples)')
    p.add_argument('--err-vmax-frac', type=float, default=0.10,
                   help='masked 에러맵 vmax (정규화 [0,1] 단위). 기본 0.10')
    p.add_argument('--no-scale-match', action='store_true',
                   help='LS scale 정합 비활성화 (기본 활성)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_match = not args.no_scale_match

    print('=' * 64)
    print(' v8 Pure ETER-Net 4-way 비교: GT / U-Net / GRU(no-DC) / SS2D(no-DC)  (384×384)')
    print(f'  device={device}, scale_match={scale_match}')
    print('=' * 64)

    h5 = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=384,
                               acceleration=ACCEL, center_fraction=CENTER_FRACTION,
                               random_mask=False, augment=False)
    total = len(h5)

    if args.slice_indices:
        indices = [int(x) for x in args.slice_indices.split(',') if x.strip() != '']
        indices = [i for i in indices if 0 <= i < total]
    else:
        indices = np.linspace(0, total - 1, args.num_samples, dtype=int).tolist()
    print(f'\n총 val 슬라이스: {total}  →  선택 {len(indices)}개: {indices}')

    n_low = int(round(384 * CENTER_FRACTION))
    print(f'mask: R={ACCEL}, center_fraction={CENTER_FRACTION} → ACS {n_low} lines (동일 mask 를 전 모델에 주입)\n')

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 슬라이스 입력/GT/brain_mask 캐시 (CPU)
    print('슬라이스 캐시 중...')
    samp_cache, gt_cache, brain_cache = {}, {}, {}
    for idx in tqdm(indices, desc='cache', unit='img'):
        s = to_tensor_sample(h5[idx])
        samp_cache[idx] = s
        gt_cache[idx] = s['label'].squeeze().numpy().astype(np.float32)
        brain_cache[idx] = s['brain_mask'].squeeze().numpy().astype(np.float32)

    # 2) 모델 outer-loop (한 번에 GPU 위 모델 1개) → recon/metric 캐시
    model_specs = [
        ('U-Net',           args.unet_ckpt, load_unet,     run_unet),
        ('GRU (no-DC)',     args.gru_ckpt,  load_pure_gru, run_ss2d),
        ('SS2D (no-DC)',    args.ss2d_ckpt, load_pure_ss2d, run_ss2d),
    ]
    recon_cache = {idx: {} for idx in indices}
    metric_cache = {idx: {} for idx in indices}
    failed = {}
    for name, ckpt, loader, runner in model_specs:
        if not os.path.exists(ckpt):
            print(f'  [SKIP] {name}: ckpt 없음 ({ckpt})')
            failed[name] = 'ckpt 없음'
            continue
        print(f'\n[{name}] 로드: {ckpt}')
        try:
            model = loader(ckpt, device)
        except Exception as e:
            print(f'  [ERROR] {name} 로드 실패: {e}')
            failed[name] = f'load: {e}'
            continue
        for idx in tqdm(indices, desc=f'{name} forward', unit='img'):
            try:
                rec = runner(model, samp_cache[idx], device)
            except Exception as e:
                tqdm.write(f'  [ERROR] {name} idx={idx}: {e}')
                rec = np.zeros_like(gt_cache[idx])
            rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            gt = gt_cache[idx]
            brain = brain_cache[idx]
            rec_adj = ls_scale(rec, gt, brain) if scale_match else rec
            recon_cache[idx][name] = rec_adj
            metric_cache[idx][name] = (masked_psnr(rec_adj, gt, brain),
                                       masked_ssim(rec_adj, gt, brain))
        del model
        torch.cuda.empty_cache()

    present = [n for (n, *_rest) in model_specs if n not in failed]

    # 3) 플로팅 (per-slice [0,1] 정규화 + brain-mask 에러맵)
    print('\nPNG 생성 중...')
    hot_bad = plt.get_cmap('hot').copy()
    hot_bad.set_bad(color='black')   # masked(뇌 밖) 영역 = 검정 (recon 행과 배경 통일)
    err_max = args.err_vmax_frac
    title_color = {'U-Net': 'tab:orange', 'GRU (no-DC)': 'tab:red', 'SS2D (no-DC)': 'tab:blue'}

    def masked_err(recon_n, gt_n, mask):
        return np.where(mask > 0.5, np.abs(recon_n - gt_n), np.nan)

    rows = []
    for idx in tqdm(indices, desc='plot', unit='img'):
        gt = gt_cache[idx]
        brain = brain_cache[idx]
        gt_max = max(float(gt.max()), 1e-8)
        gt_n = gt / gt_max

        fig, axes = plt.subplots(2, 4, figsize=(21, 11))
        axes[0, 0].imshow(gt_n, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title('GT (Ground Truth)', fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].set_visible(False)

        for col, name in enumerate(['U-Net', 'GRU (no-DC)', 'SS2D (no-DC)'], start=1):
            if name not in recon_cache[idx]:
                axes[0, col].axis('off'); axes[1, col].axis('off')
                axes[0, col].set_title(f'{name}\n(N/A)', fontsize=11)
                continue
            rec_adj = recon_cache[idx][name]
            ps, ss = metric_cache[idx][name]
            rec_n = rec_adj / gt_max
            axes[0, col].imshow(rec_n, cmap='gray', vmin=0.0, vmax=1.0)
            axes[0, col].set_title(f'{name}\nPSNR {ps:.2f}dB | SSIM {ss:.4f}',
                                   fontsize=11, color=title_color.get(name, 'black'))
            axes[0, col].axis('off')
            im = axes[1, col].imshow(masked_err(rec_n, gt_n, brain), cmap=hot_bad, vmin=0, vmax=err_max)
            axes[1, col].set_title(f'{name} masked |err|', fontsize=11, color=title_color.get(name, 'black'))
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        sm_note = ' · per-slice LS scale-aligned' if scale_match else ''
        fig.suptitle(
            f'Sample #{idx}  —  v8 Pure ETER-Net 4-way (384, R{ACCEL}, no-DC, brain-masked metric{sm_note})\n'
            f'U-Net = fastmri brain leaderboard, 16-coil→384 (domain shift, 참고 베이스라인)',
            fontsize=13, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'compare_{idx:04d}.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)

        rows.append((idx, {n: metric_cache[idx][n] for n in present if n in metric_cache[idx]}))

    # 4) 요약
    lines = ['========== v8 Pure ETER-Net 4-way 비교 요약 ==========',
             f'슬라이스 {len(indices)}개: {indices}',
             f'scale_match={scale_match}, brain-masked metric, 384×384, R{ACCEL}, no-DC',
             '',
             f'{"모델":>16s} | {"masked PSNR(dB)":>18s} | {"masked SSIM":>16s}',
             f'{"-"*16} | {"-"*18} | {"-"*16}']
    for name in present:
        ps = [m[name][0] for _i, m in rows if name in m and np.isfinite(m[name][0])]
        ss = [m[name][1] for _i, m in rows if name in m and np.isfinite(m[name][1])]
        if not ss:
            continue
        lines.append(f'{name:>16s} | {np.mean(ps):>6.2f}±{np.std(ps):<5.2f} (med {np.median(ps):>5.2f}) '
                     f'| {np.mean(ss):>6.4f}±{np.std(ss):<6.4f} (med {np.median(ss):.4f})  [n={len(ss)}]')
    lines += [
        '',
        '[해석 / 주의]',
        '- U-Net = fastmri brain leaderboard 사전학습. 본 파이프라인은 16-coil·image-domain crop→384',
        '  → fastmri 원 학습분포(전 coil·native)와 domain shift → 참고 베이스라인.',
        '- GRU/SS2D = 교수님 순수 ETER-Net(ViT 없음) no-DC arm. aggregate 는 SS2D 완승',
        '  (composite 0.9200 vs 0.9182, docs/v8_eter_pure_rnn_vs_ss2d.md). 각 PNG 패널 제목에',
        '  슬라이스별 실제 지표 표기 — sulci/혈관 detail 은 육안으로 직접 비교할 것.',
    ]
    if failed:
        lines.append('')
        for n, why in failed.items():
            lines.append(f'  [실패] {n}: {why}')
    msg = '\n'.join(lines)
    print('\n' + msg)
    with open(os.path.join(args.out_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(msg + '\n')
    print(f'\n저장 완료: {args.out_dir}/  (PNG {len(indices)}장 + metrics_summary.txt)')


if __name__ == '__main__':
    main()
