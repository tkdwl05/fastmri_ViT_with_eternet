"""그림 3 시드 1판(paper/figs/fig3_qualitative_col_seed1) 입력 생성 (2026-10-07): 슬라이스 3368 을 U-Net 단독·2회차(seed 1) ETER-net·SS2D 로 CPU 추론.
visualize_multimodel_compare.py 와 같은 경로(fp32 CPU, selective_scan_ref, per-slice LS 정합, 같은 지표 공식)를 import 해서 쓴다.
검산: 1회차 SS2D 를 다시 돌려 기존 npz 지표(34.28 / 0.948)와 같은지 확인."""
import sys, os, json, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import visualize_multimodel_compare as V          # import 시 os.chdir(ROOT)
OUT = os.path.join(ROOT, 'results', 'vis', 'multimodel_compare_seed1')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(6)
dev = torch.device('cpu')
V.patch_selective_scan_for_cpu()
h5 = V.FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val', num_files=None, target_size=384,
                             acceleration=V.ACCEL, center_fraction=V.CENTER_FRACTION,
                             random_mask=False, augment=False)
idx = 3368
s = h5[idx]
fp, si, _ = h5.samples[idx]
print('slice', idx, os.path.basename(fp), si, flush=True)
gt = s['label'][0].astype(np.float32); bm = s['brain_mask'][0].astype(np.float32)

def load_unet_only(ckpt, device):
    import myConfig_pure_eter_v8 as c
    from u_pure_eternet_unet import PureETER_UNET
    m = PureETER_UNET(n_coil=c.N_COIL, n_hidden_2=c.N_HIDDEN_LRNN_2,
                      unet_depth=c.UNET_DEPTH, unet_wf=c.UNET_WF, use_dc=False)
    return V._load_state(m, ckpt, device)

rec, met = {}, {}
def finish(key, r):
    r = np.nan_to_num(r.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r = V.ls_scale(r, gt, bm)
    rec[key] = r; met[key] = V.slice_metrics_np(r, gt, bm)
    m = met[key]; print(f'  {key:9s} PSNR {m["psnr"]:6.2f}  SSIM {m["ssim"]:.4f}  nMSE {100*m["nmse"]:.3f}%', flush=True)

finish('zf', V.zero_filled(s))
jobs = [('unetonly', 'logs/PureETER_UNET_noDC_R4_brain384_v8_s1_50ep/pure_unet_best.pt', load_unet_only),
        ('gru2',     'logs/PureETER_GRU_noDC_R4_brain384_v8_s1_50ep/pure_gru_best.pt', V.load_pure_gru),
        ('ss2d2',    'logs/PureETER_SS2D_noDC_R4_brain384_v8_s1_50ep/pure_ss2d_best.pt', V.load_pure_ss2d),
        ('ss2d_r1check', 'logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt', V.load_pure_ss2d)]
for key, ck, loader in jobs:
    t0 = time.time(); model = loader(ck, dev)
    r = V.run_pure(model, s, dev); del model
    finish(key, r); print(f'    ({time.time()-t0:.0f}s)', flush=True)

old = json.load(open(os.path.join(ROOT, 'results/vis/multimodel_compare/metrics_3368.json')))['metrics']
print('check run1 ss2d: old', round(old['ss2d']['psnr'], 4), round(old['ss2d']['ssim'], 5),
      ' new', round(met['ss2d_r1check']['psnr'], 4), round(met['ss2d_r1check']['ssim'], 5), flush=True)
z = np.load(os.path.join(ROOT, 'results/vis/multimodel_compare/recon_3368.npz'))
np.savez_compressed(os.path.join(OUT, 'recon_3368.npz'), gt=gt, brain_mask=bm,
                    rec_zf=rec['zf'], rec_unetonly=rec['unetonly'], rec_gru2=rec['gru2'], rec_ss2d2=rec['ss2d2'],
                    rec_gru=z['rec_gru'], rec_ss2d=z['rec_ss2d'])
mm = {k: met[k] for k in ['zf', 'unetonly', 'gru2', 'ss2d2']}
mm['gru'] = old['gru']; mm['ss2d'] = old['ss2d']
meta = json.load(open(os.path.join(ROOT, 'results/vis/multimodel_compare/metrics_3368.json')))['meta']
json.dump({'meta': meta, 'metrics': mm, 'check_run1_ss2d': met['ss2d_r1check'],
           'note': 'per-slice LS scale-aligned inside brain mask; CPU fp32; run 2 = seed 1 (50 ep best), unetonly = seed 1'},
          open(os.path.join(OUT, 'metrics_3368.json'), 'w'), indent=1)
print('saved', OUT, flush=True)
# 실행(저장소 루트, CPU — 학습 런과 CPU 를 나눠 쓰므로 nice·스레드 제한):
#   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 nice -n 19 python paper/make_fig3_seed1_inputs.py
# 조판:
#   python paper/make_fig3_qualitative.py --src results/vis/multimodel_compare_seed1 --methods gt,unetonly,gru2,ss2d2 \
#       --slices 3368 --gain-slice 3368 --fig-w 3.15 --out-name fig3_qualitative_col_seed1
