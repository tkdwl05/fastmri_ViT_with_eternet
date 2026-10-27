"""
ETER-net(gru:DL) 대조 — E-a·Mamba-ETER 와 같은 학습 뒤 분석을 ETER-net 에도 한다 (2026-10-11).

배경 (results/ss2d_diag_2026-10-11/, results/mamba1d_smoke_2026-10-11/summary.md):
  - smoke_recipe_fix 의 gru:DL(val SSIM 0.7973, U-Net 단독 unet:DL 0.7773 대비 +0.020)은 zero·shuffle 시험만 있다.
    shuffle 하락(+0.0148)은 '출력에 반응한다'는 뜻일 뿐, 슬라이스 고유 내용이 이득이라는 증거가 아니다.
  - E-a(SS2D out_proj×0.03)는 출력이 99.7% 전역 상수인데도 U-Net 단독 대비 +0.021(64/64)로 gru:DL 과 같았고,
    Mamba-ETER 두 조건도 슬라이스 평균 평면으로 바꿔도 손실이 없었다(outmatch 는 오히려 +0.004 개선).
  - 따라서 ETER-net 의 이득이 슬라이스 고유 정보에서 오는지 같은 대체 시험(slice_mean_plane·fixed_pattern·
    no_slice_specific_spatial·shuffle_cross)과 에너지 분해로 확인해야 SS2D·Mamba-ETER 결과를 기준 모델과 비교할 수 있다.

구현: R.run_job('gru', 'DL', …) 를 원 R.build_model 그대로 호출하고(데이터·레시피·SEED·검증 = gru:DL 과 같음),
끝 ckpt 로 D.post_analysis 를 프록시(.ss2d = model._seq)로 실행한다. smoke_recipe_fix.py·diag_ss2d_variants.py 무수정.
cuDNN GRU 는 deterministic(cudnn.deterministic=True)이라 val SSIM 이 기준 0.7973 과 거의 같아야 한다(재현 확인 겸 —
num_workers 가 기준 16 과 다르면 augment rng 배분이 달라 거부). 결과 이름 v4_gru_ctrl → D.rebuild_md 표에 같이 나온다.

실행 예:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python -u v8_eter_pure/diag_gru_post.py --out-dir results/ss2d_diag_2026-10-11 --skip-done
  # 스모크: ... --steps 30 --log-every 10 --log-dense-every 0 --val-slices 4 --post-slices 4 --out-dir <scratch>
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import OrderedDict

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import diag_ss2d_variants as D                                         # noqa: E402

R, C = D.R, D.C
RUNNER_VERSION = '2026-10-11a'
VNAME = 'v4_gru_ctrl'
REF_NUM_WORKERS = 16                                                   # results/smoke_recipe_fix/gru_DL_summary.json
REF_GRU_DL = dict(val_ssim=0.7973, shuffle_cross_mean=0.0148, shuffle_cross_npos=62, zero_mean=0.0176, n=64,
                  source='results/smoke_recipe_fix/gru_DL_summary.json')


class SeqProxy:
    """D.post_analysis 가 요구하는 인터페이스(.eval()/.train()/.unet/.ss2d(xk))를 ETER-net 에 연결 (ss2d = model._seq)."""

    def __init__(self, model):
        self.model = model
        self.unet = model.unet

    def ss2d(self, xk):
        return self.model._seq(xk)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


@torch.no_grad()
def weight_norm_ratios_gru(model, model0):
    """시퀀스 모듈(gru_h·gru_v) 전 텐서의 끝/초기 norm 비·cos(Δw, w0)·relΔ + U-Net 첫 conv 의 시퀀스/zero-filled 입력 slice."""
    sd, sd0 = model.state_dict(), model0.state_dict()
    out = OrderedDict()
    for k in sd:
        if not k.startswith(('gru_h.', 'gru_v.')):
            continue
        a, b = sd[k].float(), sd0[k].float()
        out[k] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()),
                      ratio=float(a.norm() / b.norm().clamp(min=1e-30)),
                      cos_delta_w0=float(F.cosine_similarity((a - b).flatten(), b.flatten(), dim=0)),
                      rel_change=float((a - b).norm() / b.norm().clamp(min=1e-30)))
    out.update(D.weight_norm_ratios(model, model0))                    # ss2d 키는 없으므로 U-Net 항목만 더해진다
    return out


def parse_args():
    p = argparse.ArgumentParser(description='ETER-net(gru:DL) 대조 + 학습 뒤 분석 (E-a·Mamba-ETER 와 같은 정의)')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--log-dense-every', type=int, default=20)
    p.add_argument('--log-dense-until', type=int, default=500)
    p.add_argument('--val-slices', type=int, default=64)
    p.add_argument('--post-slices', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--out-dir', default='results/ss2d_diag_2026-10-11')
    p.add_argument('--max-minutes', type=float, default=240)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--val-workers', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--skip-done', action='store_true')
    a = p.parse_args()
    if a.num_workers is None:
        a.num_workers = C.NUM_WORKERS_TRAIN
    if a.val_workers is None:
        a.val_workers = C.NUM_WORKERS_VAL
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(D._PROJECT_ROOT, a.out_dir)
    a.post_slices = min(a.post_slices, a.val_slices)
    return a


def run(args, device):
    vdir = os.path.join(args.out_dir, VNAME)
    os.makedirs(vdir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f'{VNAME}_final.json')
    jsonl_final = os.path.join(vdir, f'{VNAME}.jsonl')
    ckpt_final = os.path.join(vdir, f'{VNAME}_end.pt')

    train_ds, val_ds, val_indices, probe_indices, probe = D.prepare_data(args, device)
    print(f'  val 슬라이스 {len(val_indices)}개: {val_indices[:8]}{" ..." if len(val_indices) > 8 else ""} | probe {probe_indices}')
    t0 = time.time()
    dl_scale = R.estimate_mean_scale(train_ds, args.seed)
    dl_scale['source'] = f'train {dl_scale["n"]} slices'
    print(f'  DL s̄ = {dl_scale["mean"]:.4f} → λ_DL = {C.LAMBDA_SSIM_PER_PIXEL / dl_scale["mean"]:.6g} ({time.time() - t0:.0f}s)')

    diag = OrderedDict(diag_version=D.DIAG_VERSION, runner_version=RUNNER_VERSION, smoke_script_version=R.SCRIPT_VERSION,
                       variant='gru_ctrl', vname=VNAME,
                       description='ETER-net(gru:DL) 재실행 + 학습 뒤 분석 — E-a·Mamba-ETER 와 같은 대체 시험·에너지 분해',
                       started=D.now(), argv=sys.argv, seed=args.seed, steps=args.steps, num_workers=args.num_workers,
                       reference_smoke_gru_DL=REF_GRU_DL)
    ns = argparse.Namespace(
        seq='gru', cond='DL', jobs=None, steps=args.steps, tmax=None, log_every=args.log_every,
        log_dense_every=args.log_dense_every, log_dense_until=args.log_dense_until, val_slices=args.val_slices,
        seed=args.seed, lambda_ssim=C.LAMBDA_SSIM_PER_PIXEL, lambda_dl_scale=None, out_dir=vdir, save_ckpt='end',
        bs=args.bs, device='cuda', force=args.force, max_minutes=args.max_minutes, num_workers=args.num_workers,
        val_workers=args.val_workers, skip_done=False, job_list=[('gru', 'DL')], tmax_eff=args.steps,
        probe_indices=probe_indices, dl_scale=dl_scale)
    status, summ, extra, err = 'failed', None, None, None
    try:
        summ = R.run_job('gru', 'DL', ns, train_ds, val_ds, val_indices, probe, device, vdir)
        status = summ.get('status', 'completed')
        os.replace(os.path.join(vdir, 'gru_DL.jsonl'), jsonl_final)
        os.replace(os.path.join(vdir, 'gru_DL_end.pt'), ckpt_final)
        gc.collect()
        torch.cuda.empty_cache()
        model = R.build_model('gru', device)
        sd = torch.load(ckpt_final, map_location=device)
        model.load_state_dict(sd, strict=True)
        del sd
        R.seed_all(args.seed)
        model0 = R.build_model('gru', device)                         # 같은 시드로 다시 만든 초기 모델
        idx = val_indices[:args.post_slices]
        extra = OrderedDict(weight_norm_ratios=weight_norm_ratios_gru(model, model0))
        extra['end'] = D.post_analysis(SeqProxy(model), val_ds, idx, device, args.val_workers)
        extra['init'] = D.post_analysis(SeqProxy(model0), val_ds, idx, device, args.val_workers)
        del model, model0
        e, i0 = extra['end'], extra['init']
        st = e['substitution_tests']
        print(f'[post] end: ssim {e["ssim_normal_mean"]:.4f} seq_out_rms {e["decomposition"]["seq_out_rms"]:.4g} '
              f'(init {i0["decomposition"]["seq_out_rms"]:.4g}) slice-specific frac '
              f'{e["decomposition"]["frac_slice_specific_spatial"]:.3f} | zero {st["zero"]["mean"]:+.5f} '
              f'slice_mean {st["slice_mean_plane"]["mean"]:+.5f} ({st["slice_mean_plane"]["n_pos"]}/{st["slice_mean_plane"]["n"]}) '
              f'fixed {st["fixed_pattern"]["mean"]:+.5f} shuffle {st["shuffle_cross"]["mean"]:+.5f} '
              f'| grid R² zf {e["grid"]["r2_zf_rss"]:.3f} logk {e["grid"]["r2_logk_sampled"]:.3f}', flush=True)
    except KeyboardInterrupt:
        raise
    except Exception as ex:                                             # noqa: BLE001
        err = f'{type(ex).__name__}: {ex}\n{traceback.format_exc()[-4000:]}'
        print(f'[FAIL] {VNAME}: {err}', flush=True)
        status = 'failed' if summ is None else f'{status}+post_failed'
    diag['finished'] = D.now()
    out = OrderedDict(diag=diag, status=status, error=err, run_job_summary=summ, post=extra)
    D.write_json(final_path, out)
    D.rebuild_md(args.out_dir)
    print(f'[{VNAME}] 끝 status={status} → {final_path}', flush=True)
    return 0 if summ is not None and err is None else R.EXIT_JOB_FAILED


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print('=' * 72)
    print(f' ETER-net 대조 + 학습 뒤 분석 ({RUNNER_VERSION}; diag {D.DIAG_VERSION}, smoke {R.SCRIPT_VERSION}) '
          f'steps={args.steps} bs={args.bs} seed={args.seed} val={args.val_slices} out={args.out_dir}  {D.now()}')
    print('=' * 72, flush=True)
    if args.skip_done:
        fp = os.path.join(args.out_dir, f'{VNAME}_final.json')
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    old = json.load(f)
                d = old.get('diag', {})
                if (old.get('status') in ('completed', 'truncated_time') and old.get('error') is None
                        and d.get('diag_version') == D.DIAG_VERSION and d.get('steps') == args.steps):
                    print(f'[skip-done] {VNAME} 이미 끝남 — 건너뜀')
                    return 0
            except Exception as ex:                                     # noqa: BLE001
                print(f'[skip-done] 이전 결과 읽기 실패 ({ex}) — 다시 실행')
    if args.steps == 4000 and args.num_workers != REF_NUM_WORKERS:
        R.refuse(f'num_workers {args.num_workers} ≠ 기준 {REF_NUM_WORKERS} — augment rng 배분이 달라 gru:DL 과 비교 불가')
    R.gpu0_precheck(args.force)
    return run(args, torch.device('cuda'))


if __name__ == '__main__':
    sys.exit(main())
