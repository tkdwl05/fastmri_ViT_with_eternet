"""그림 3 v8fix 판(paper/figs/fig3_qualitative_col_v8fix) 입력 생성 (2026-10-17).

수정 레시피(v8fix) 50 epoch 학습의 평가 best 체크포인트 세 개(U-Net only · ETER-net · SS2D(controlled), SEED 1, RunPod)로
고정 슬라이스 두 장을 재구성한다. 경로는 v8fix 평가(v8_eter_pure/eval_v8fix.py)와 같은 것을 import 해서 쓴다:
  - 데이터: eval_unet_only_v8.build_val_dataset (= eval_v8fix 의 val/test 데이터 구성 — R=4, center 0.08, random_mask=False
    → 마스크 시작 offset 3 고정, augment=False → flip 없음, 앞 16 코일, 384×384, 로더의 brain mask)
  - 모델: smoke_recipe_fix.build_model(seq) + torch.load → load_state_dict(strict) (eval_v8fix.main 과 같음)
  - 래퍼: eval_v8fix.Normalized (샘플별 s = zero-filled RSS 99 백분위수, 입력 정규화, 출력 ×s 로 원 단위 복원)
  - 지표: eval_paired_v8_nodc.slice_metrics (eval_v8fix → eval_unet_only_v8 가 import 하는 것) — brain mask 안
    SSIM(skimage, data_range = 마스크 안 GT max−min), PSNR(마스크 안 GT max 기준), nMSE, L1.
차이: fp32 추론(autocast 없음, TF32 해제 — 학습과 같게). 평가(pod)는 AMP. 슬라이스별 최소제곱 세기 정합 없음(사용자 결정 D4):
npz 에는 모델 출력을 원 단위(로더 label 과 같은 ×1e6 단위) 그대로 저장한다.
슬라이스는 loader index 가 아니라 파일 이름 + 슬라이스 번호로 고른다. 출력 파일 이름의 3368 / 4689 는
paper/make_fig3_qualitative.py 호환용으로 이전 로컬 val 로더 번호(results/vis/multimodel_compare/metrics_{3368,4689}.json meta)를 쓴다.
출력: results/vis/multimodel_compare_v8fix/recon_{3368,4689}.npz, metrics_{3368,4689}.json
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'v8_eter_pure'))
import eval_v8fix as EV                 # Normalized (+ import: eval_unet_only_v8, smoke_recipe_fix)
E = EV.E                                # eval_unet_only_v8: build_val_dataset, slice_metrics
R = EV.R                                # smoke_recipe_fix: build_model, per_sample_scale

OUT = os.path.join(ROOT, 'results', 'vis', 'multimodel_compare_v8fix')
DATA_VAL = os.path.join(ROOT, 'fastMRI_data', 'multicoil_val')
# (npz/metrics 키, smoke_recipe_fix seq, 표시 이름, 체크포인트, 기대 best epoch, md5 접두(있으면 검사))
MODELS = [
    ('unetonly', 'unet', 'U-Net only',
     'logs/runpod_mirror/logs/PureETER_UNET_noDC_R4_brain384_v8fix_s1_fullval/pure_unet_best.pt', 48, None),
    ('gru', 'gru', 'ETER-net',
     'logs/runpod_mirror_pod3/logs/PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval/pure_gru_best.pt', 46, 'f19c9b64'),
    ('ss2d', 'ss2d', 'SS2D(controlled)',
     'logs/runpod_mirror/logs/PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval/pure_ss2d_best.pt', 50, None),
]
# (출력 번호 = 이전 로컬 val 로더 번호, 파일, 슬라이스, 용도)
SLICES = [
    (3368, 'file_brain_AXT2_200_2000556.h5', 6,
     'Fig. 3 slice; lies in validation batch 0, i.e. inside the checkpoint-selection set (val batches 0~2)'),
    (4689, 'file_brain_AXT2_203_2030309.h5', 7,
     'Fig. 1 pipeline thumbnail slice; pod val idx 13789 of the 21,842 slices of validation batches 0~2, i.e. inside '
     'the checkpoint-selection set'),
]
METRIC_KEYS = ('ssim', 'psnr', 'nmse', 'l1')


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def stored_epoch(state):
    """체크포인트 안의 epoch 기록 (state_dict 만 저장된 경우 None)."""
    if isinstance(state, dict):
        for k in ('epoch', 'best_epoch'):
            if k in state and not torch.is_tensor(state[k]):
                return state[k]
    return None


# 10-17: ETER-net 런 폴더(logs/runpod_mirror_pod3/…)에는 DONE 이 없다 — 학습 종료 후 pod3 에서 회수한 DONE
# (2026-10-16T23:49:15, best_val.epoch 46)은 아래 사본 폴더에 있다. 런 폴더에 없으면 이 경로를 본다.
DONE_FALLBACK = {
    'PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval':
        'logs/runpod_mirror/pod3_final_extra_20261017/fastmri_ViT_with_eternet/logs/'
        'PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval/DONE',
}


def done_best_epoch(ck):
    """트레이너가 끝에 쓴 DONE(json)의 (best_val.epoch, DONE 경로) — 런 폴더 → DONE_FALLBACK 순, 없으면 (None, None)."""
    run = os.path.basename(os.path.dirname(ck))
    for p in (os.path.join(os.path.dirname(ck), 'DONE'), DONE_FALLBACK.get(run)):
        if p and os.path.exists(p):
            return json.load(open(p))['best_val']['epoch'], p
    return None, None


def metrics(out, ref, bm):
    sm = E.slice_metrics(out, ref, bm)
    return {k: float(sm[k]) for k in METRIC_KEYS}


def main():
    ap = argparse.ArgumentParser(description='Fig. 3 v8fix inputs (fp32, no LS scaling)')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--amp-check', action='store_true',
                    help='추가로 pod 평가와 같은 autocast(+ torch 기본 cuDNN TF32) 지표를 계산해 json 에 기록 (npz 에는 넣지 않음)')
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)

    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA 없음 — --device cpu 는 SS2D 에 CPU selective-scan 패치가 필요'
        dev = torch.device('cuda')
        dev_name = torch.cuda.get_device_name(0)
    else:
        R.install_cpu_shims(need_ss2d=True)
        dev = torch.device('cpu')
        dev_name = 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ---- 데이터: eval 과 같은 val 데이터셋에서 (파일, 슬라이스) 로 찾는다
    ds = E.build_val_dataset(DATA_VAL)
    lookup = {(os.path.basename(fp), s): i for i, (fp, s, _) in enumerate(ds.samples)}
    old_meta_dir = os.path.join(ROOT, 'results', 'vis', 'multimodel_compare')
    samples = {}
    for out_idx, fname, sl, purpose in SLICES:
        old = json.load(open(os.path.join(old_meta_dir, f'metrics_{out_idx:04d}.json')))['meta']
        assert (old['file'], old['slice']) == (fname, sl), f'{out_idx}: 이전 meta {old} ≠ {fname} slice {sl}'
        i = lookup[(fname, sl)]
        d = ds[i]
        t = {k: torch.from_numpy(np.asarray(d[k]))[None].float() for k in ('data', 'data_img', 'label', 'brain_mask', 'mask', 'sens')}
        rss = torch.sqrt((t['data_img'] ** 2).sum(dim=1, keepdim=True))       # zero-filled RSS (16 코일, label 과 같은 단위)
        s = float(R.per_sample_scale(t['data_img']))
        samples[out_idx] = dict(fname=fname, slice=sl, purpose=purpose, loader_idx=i, t=t, zf=rss,
                                contrast=old['contrast'], s=s, rec={}, met={}, amp={})
        print(f'slice {out_idx}: {fname} #{sl} → 현 val 로더 idx {i} (이전 번호 {out_idx}), s={s:.3f}, '
              f'brain mask {int(t["brain_mask"].sum())} px', flush=True)
        samples[out_idx]['met']['zf'] = metrics(rss, t['label'], t['brain_mask'])

    # ---- 모델별 추론
    ck_info = {}
    for key, seq, name, ck, exp_ep, md5_prefix in MODELS:
        t0 = time.time()
        h = md5(ck)
        if md5_prefix:
            assert h.startswith(md5_prefix), f'{ck}: md5 {h} 가 {md5_prefix} 로 시작하지 않음 (오래된 사본?)'
        state = torch.load(ck, map_location=dev)
        ep_stored = stored_epoch(state)
        base = R.build_model(seq, dev)
        base.load_state_dict(state)                      # strict=True (eval_v8fix.main 과 같음)
        del state
        base.eval()
        model = EV.Normalized(base).eval()
        ep_log = EV.ckpt_epoch_info(ck)                  # log.txt 검증 SSIM 4자리 최댓값의 첫 epoch (동률이면 이른 epoch 로 기움 — 참고만)
        ep_done, done_path = done_best_epoch(ck)
        assert ep_done == exp_ep, f'{ck}: DONE best epoch {ep_done} ({done_path}) ≠ 기대 {exp_ep}'
        ck_info[key] = dict(name=name, ckpt=ck, md5=h, size_bytes=os.path.getsize(ck),
                            best_epoch=ep_done, best_epoch_source=f'DONE best_val.epoch ({done_path})',
                            epoch_stored_in_ckpt=ep_stored, epoch_from_DONE=ep_done,
                            epoch_from_log_first_max_4dp=ep_log,
                            epoch_from_log_note='first epoch reaching the 4-decimal max val SSIM in log.txt; '
                                                'ties resolve to the earliest epoch, so it can differ from DONE (not authoritative)',
                            expected_best_epoch=exp_ep,
                            params_M=round(sum(q.numel() for q in base.parameters()) / 1e6, 3))
        print(f'[{name}] {ck}\n   md5 {h}  stored epoch {ep_stored}  DONE {ep_done} ({done_path})  log first-max {ep_log}', flush=True)
        for out_idx, S in samples.items():
            t = {k: v.to(dev) for k, v in S['t'].items()}
            with torch.no_grad():
                out = model(t['data_img'], t['data'], t['mask'], t['sens']).float()
                S['met'][key] = metrics(out, t['label'], t['brain_mask'])
                S['rec'][key] = out[0, 0].cpu().numpy().astype(np.float32)
                if args.amp_check and dev.type == 'cuda':
                    torch.backends.cudnn.allow_tf32 = True      # torch 기본값 (pod 평가 프로세스와 같음)
                    with torch.amp.autocast('cuda'):
                        out_a = model(t['data_img'], t['data'], t['mask'], t['sens'])
                    torch.backends.cudnn.allow_tf32 = False
                    S['amp'][key] = metrics(out_a.float(), t['label'], t['brain_mask'])
                    del out_a
            m = S['met'][key]
            print(f'   {out_idx}: SSIM {m["ssim"]:.4f}  PSNR {m["psnr"]:6.2f}  nMSE {100 * m["nmse"]:.3f}%'
                  + (f'   (amp SSIM {S["amp"][key]["ssim"]:.4f})' if key in S['amp'] else ''), flush=True)
            del t, out
        del model, base
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'   ({time.time() - t0:.0f}s)', flush=True)

    # ---- 저장
    precision = ('fp32, no autocast, torch.backends.cuda.matmul.allow_tf32=False, torch.backends.cudnn.allow_tf32=False'
                 if dev.type == 'cuda' else 'fp32 CPU (smoke_recipe_fix.install_cpu_shims selective_scan_cpu for SS2D)')
    for out_idx, S in samples.items():
        np.savez_compressed(os.path.join(OUT, f'recon_{out_idx:04d}.npz'),
                            gt=S['t']['label'][0, 0].numpy().astype(np.float32),
                            brain_mask=S['t']['brain_mask'][0, 0].numpy().astype(np.float32),
                            rec_zf=S['zf'][0, 0].numpy().astype(np.float32),
                            **{f'rec_{k}': v for k, v in S['rec'].items()})
        meta = dict(idx=out_idx, file=S['fname'], slice=S['slice'], contrast=S['contrast'],
                    idx_note='file name for compatibility = old local val-loader index; selection is by file + slice',
                    loader_idx_current=S['loader_idx'], purpose=S['purpose'],
                    split='fastMRI brain multicoil val (local fastMRI_data/multicoil_val); NOT test_full',
                    selection_set_note='this slice is in validation batches 0~2 (1,378 volumes / 21,842 slices), which '
                                       'were used to select the best checkpoints; quantitative results are on test_full',
                    per_sample_scale_s=S['s'], units='loader units (RSS x 1e6), same as label',
                    labels={'gt': 'Ground truth', 'zf': 'Zero-filled (16-coil RSS, reference only)',
                            **{k: v['name'] for k, v in ck_info.items()}},
                    checkpoints=ck_info, device=dev_name, precision=precision)
        doc = dict(meta=meta, metrics=S['met'])
        if S['amp']:
            doc['amp_check'] = dict(desc='same models under torch.amp.autocast("cuda") with default cuDNN TF32 '
                                         '(as in the pod evaluation); metrics only, not stored in the npz', metrics=S['amp'])
        doc['note'] = ('v8fix best checkpoints (SEED 1, 50 epochs; U-Net only / ETER-net / SS2D(controlled)); '
                       'inputs built as in v8_eter_pure/eval_v8fix.py (fixed R4 mask offset 3, no flip, first 16 coils, 384x384) '
                       'and wrapped with eval_v8fix.Normalized; NO per-slice least-squares intensity scaling (raw outputs in '
                       'original units); metrics = eval_paired_v8_nodc.slice_metrics inside the brain mask; '
                       f'device {dev_name}, {precision}. rec_zf is unscaled zero-filled RSS (reference only).')
        with open(os.path.join(OUT, f'metrics_{out_idx:04d}.json'), 'w') as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()

# 실행(저장소 루트, GPU0 에 다른 프로세스가 없을 때 — nvidia-smi 로 먼저 확인; 약 1~2 분):
#   CUDA_VISIBLE_DEVICES=0 python paper/make_fig3_v8fix_inputs.py --amp-check
# (GPU0 사용 중이면 CPU: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 nice -n 19 python paper/make_fig3_v8fix_inputs.py --device cpu)
# 조판:
#   python paper/make_fig3_qualitative.py --src results/vis/multimodel_compare_v8fix --methods gt,unetonly,gru,ss2d \
#       --slices 3368 --gain-slice 3368 --fig-w 3.15 --cbar-tick-pt 6.5 --dpi 600 --liberation-cbar-label \
#       --header 'gru=ETER-net|' --header 'ss2d=|SS2D(controlled)' --out-name fig3_qualitative_col_v8fix
#   (10-17: 3.15 in × 600 dpi = 1890 px, 패널 384 → 약 356 px antialiased 리샘플; 글꼴 Liberation Sans 만; 머리글 = 정확한 모델 이름)
