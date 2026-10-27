"""
원래 U-Net(OU) 판 SS2D 모델의 계약 점검 (2026-10-26) — 계획 보충
paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan_amend1_ss2d.md.
GPU 하나(forward·기능 시험 경로)와 CPU(구조) 로 돈다:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 nice -n 19 python -B v8_eter_pure/sanity_ou_ss2d.py [--out x.json]
도우미(check·nparams·sd_equal·unet_checksum·emulate_same_unet_init)는 sanity_ou.py 의 것을 그대로 쓴다.

점검 (하나라도 실패하면 종료 코드 1):
  1. 등록: recipe_ou.MODELS 에 SS2D 이름 넷, 기존 이름·recipe_ou.OU_NAMES 튜플은 그대로, 런 폴더 이름 = 런처 pod_run_ou_ss2d.sh 규칙.
  2. 파라미터 수: SS2D 모듈 = SS2D(controlled)(PureETER_SS2D.ss2d) 와 같음, FourierSS2D = 그것 + 640(res 32·20), FourierLinear = 640,
     U-Net = OriginalUNet(52)(d-) / OriginalUNet(20)(o-).
  3. SS2D 초기값: seed_all(1) 직후 만든 d-SS2D (OU)·o-SS2D (OU) 의 SS2D = 같은 시드의 SS2D(controlled) 의 SS2D (비트 일치).
     o-SS2D-F (OU) 는 out_proj 를 뺀 SS2D 텐서 전부 일치, out_proj weight·bias = 0, res = 코일 0–9 선택 행렬.
  4. U-Net 초기값 (트레이너 SAME_UNET_INIT 절차 — sanity_ou.emulate_same_unet_init): d-SS2D (OU) 는 U-Net only (OU) 에서 전부 복사
     (U-Net 체크섬 일치), o- 모델은 모양이 같은 텐서 전부 복사·자기 초기값 = 첫 합성곱 weight 하나, 그리고 그 복사분이 o-ETER (OU) 의
     복사분과 비트 일치.
  5. 푸리에 헤드 규약: 로컬 val 실제 슬라이스 2 개를 정규화(R.per_sample_scale · R.normalize_inputs)한 뒤, 초기 FourierSS2D·FourierLinear
     출력 = x_img[:, :20] (로더의 ifft2c 규약과 같음 — fp32·autocast 모두). 출력 dtype = float32.
  6. forward: 네 모델 (1, 32, 384, 384) ×2 → (1, 1, 384, 384), 유한 (GPU, autocast).
  5b. 국소 k-space 크기 E(kspace_envelope): 모양 (1, 1, 384, 384), 측정 줄에 신호가 있는 행에서 빠진 줄(마스크 0)의 E > 0 이
     99% 넘음, 중심 열의 E > 가장자리 열의 E.
  7. 기울기 (트레이너와 같은 GradScaler): o-SS2D-F (OU) 첫 역전파에서 out_proj·res·U-Net 기울기 ≠ 0 (SS2D 앞쪽은 out_proj = 0 이라
     정확히 0), 한 걸음 뒤 두 번째 역전파에서 SS2D 앞쪽(in_proj)·스캔 기울기 ≠ 0 (경로가 살아 있음).
  8. 기능 시험 경로 (GPU): recipe_ou.run_validation 으로 실제 val 슬라이스 4 개 — 네 모델 모두 normal·deep4·deep34·zero·shuffle
     결과, 분해 점검 decomp_equiv_maxabs ≤ 1e-4 (autocast fp16 — cuDNN 알고리즘 선택에 따른 차이를 허용).
  8b. 대리 객체 배선 (CPU, fp32, 마지막에 — CPU shim 이 전역을 바꿈): 분해 경로 R.unet_input → (대리) U-Net = model.forward,
     zero 경로 = U-Net(0 입력) 이 정확히 같음 (64×64 난수 입력).
  9. 건강 감시 그룹: SS2D 본체가 있는 모델은 seq · seq.ss2d_in · seq.ss2d_ssm · seq.ss2d_out 이 있고 그 원소 수가 SS2D(controlled) 와
     같음(FourierSS2D 의 seq 는 +640). U-Net 그룹 원소 수 = 같은 입력 채널의 OU 기준 모델.
 10. 원본 무수정: git diff 7d4e4e0 -- models/hybrid_eternet/myUNet_DF.py 비어 있음, git diff HEAD -- models/mamba_eternet/ss2d.py ·
     dataloaders/dataloader_h5_v5.py 비어 있음, recipe_ou.py·u_pure_eternet_ou.py·OU 트레이너 sha256 = 볼륨 판(아래 상수).
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)

import torch                       # noqa: E402

import sanity_ou as SO             # noqa: E402  (recipe_ou·smoke_recipe_fix 를 import)
import recipe_ou_ss2d as ROS       # noqa: E402  (SS2D 이름을 recipe_ou 등록부에 더함)

RO, R, C = SO.RO, SO.R, SO.C
check = SO.check

OU_BASE_SHA256 = {   # 이 점검을 쓸 때의 로컬 = 볼륨 판 (실행 중인 OU 런이 쓰는 파일 — 이 계획은 고치지 않는다)
    'v8_eter_pure/recipe_ou.py': 'fb3353185658351480aeb7731582fb28565515db3eb7f5b0a96db997ac60707b',
    'models/pure_eternet/u_pure_eternet_ou.py': 'bcce8f2ea10917a0c6caeae9f129f06b14dabe70b28e5a486c6627904ef92874',
    'v8_eter_pure/main_train_pure_v8fix_ou.py': 'e40e7e2affd74a073f917b31bcb1c0918e6a3726d2e9683e7f901a4847b950f0',
    'models/pure_eternet/unet_original.py': 'a5965f896737e1591821057cc132b9af8fc2176b4c5f7fb82b6ca66e81605d63',
    'v8_eter_pure/smoke_recipe_fix.py': 'b18938364539181016d9839f6d2ba7086e96f3146443c1dc8d6742a34ff0319a',
}


def _sha(path):
    with open(os.path.join(_ROOT, path), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _val_samples(n=2):
    ds = R.build_val_dataset()
    idx = R.even_indices(len(ds), n)
    out = []
    for i in idx:
        b = ds[i]
        out.append({k: torch.as_tensor(v).float().unsqueeze(0) for k, v in b.items() if k in ('data', 'data_img', 'mask')})
    return ds, idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cpu, dev = torch.device('cpu'), torch.device('cuda')
    if not torch.cuda.is_available():
        raise SystemExit('GPU 가 필요하다 (SS2D 의 selective scan CUDA 커널, 기능 시험 경로)')

    # ---- 1. 등록
    names = ROS.SS2D_OU_NAMES
    reg_ok = all(n in RO.MODELS for n in names) and tuple(RO.OU_NAMES) == ('unet_ou', 'gru_ou', 'gruo_ou')
    run_names = {n: RO.run_name(n, '_s1_fullval') for n in names}
    sh = os.path.join(_ROOT, 'infra', 'runpod', 'pod_run_ou_ss2d.sh')
    rule_line = "TAG=$(echo \"$MODEL\" | tr -d '_' | tr '[:lower:]' '[:upper:]')"
    has_rule = os.path.exists(sh) and rule_line in open(sh).read()
    check('1_registration', reg_ok and has_rule and all(
        v == f"PureETER_{n.replace('_', '').upper()}_noDC_R4_brain384_v8fix_s1_fullval" for n, v in run_names.items()),
          run_names=run_names, ou_names=list(RO.OU_NAMES), launcher_rule_found=has_rule,
          bases={n: RO.base_seq(n) for n in names}, partial={n: RO.partial_unet_copy(n) for n in names})

    # ---- 2·3. 파라미터 수, SS2D 초기값
    R.seed_all(1)
    ref_ss2d = R.build_model('ss2d', cpu)                       # SS2D(controlled) (DFU), 같은 시드
    ref_sd = ref_ss2d.ss2d.state_dict()
    n_ref_seq = SO.nparams(ref_ss2d.ss2d)
    models = {}
    for n in names:
        R.seed_all(1)
        models[n] = RO.build_model(n, cpu)
    counts = {n: dict(total=SO.nparams(m), seq=SO.nparams(m.ss2d), unet=SO.nparams(m.unet)) for n, m in models.items()}
    unet52 = SO.nparams(RO.build_model('unet_ou', cpu).unet)
    R.seed_all(1)
    unet20 = SO.nparams(RO.build_model('gruo_ou', cpu).unet)
    ok2 = (counts['ss2d_ou']['seq'] == n_ref_seq and counts['ss2do_ou']['seq'] == n_ref_seq
           and counts['ss2dfo_ou']['seq'] == n_ref_seq + 640 and counts['linfo_ou']['seq'] == 640
           and counts['ss2d_ou']['unet'] == unet52 and all(counts[n]['unet'] == unet20 for n in ('ss2do_ou', 'ss2dfo_ou', 'linfo_ou')))
    check('2_param_counts', ok2, counts=counts, ss2d_controlled_seq=n_ref_seq, unet52=unet52, unet20=unet20)

    eq_d, diff_d = SO.sd_equal(ref_sd, models['ss2d_ou'].ss2d.state_dict())
    eq_o, diff_o = SO.sd_equal(ref_sd, models['ss2do_ou'].ss2d.state_dict())
    f_sd = models['ss2dfo_ou'].ss2d.state_dict()
    eq_f, diff_f = SO.sd_equal(ref_sd, f_sd, skip=('out_proj.weight', 'out_proj.bias'))
    sel = torch.zeros(20, 32, 1, 1)
    for j in range(20):
        sel[j, j, 0, 0] = 1.0
    out0 = bool((f_sd['out_proj.weight'] == 0).all() and (f_sd['out_proj.bias'] == 0).all())
    res_ok = torch.equal(f_sd['res.weight'], sel) and torch.equal(models['linfo_ou'].ss2d.res.weight.detach(), sel)
    check('3_ss2d_init', eq_d and eq_o and eq_f and out0 and res_ok,
          ss2d_ou_equal=eq_d, ss2do_ou_equal=eq_o, ss2dfo_ou_equal_except_out_proj=eq_f, out_proj_zero=out0, res_select=res_ok,
          diffs=dict(ss2d_ou=diff_d, ss2do_ou=diff_o, ss2dfo_ou=diff_f))
    del ref_ss2d, models

    # ---- 4. U-Net 초기값 (트레이너 절차)
    m_u, _ = SO.emulate_same_unet_init('unet_ou', 1)
    m_d, rec_d = SO.emulate_same_unet_init('ss2d_ou', 1)
    m_go, rec_go = SO.emulate_same_unet_init('gruo_ou', 1)
    first_w = 'unet.down_path.0.block.0.weight'
    rec_o = {}
    o_equal_go = {}
    go_sd = m_go.unet.state_dict()
    for n in ('ss2do_ou', 'ss2dfo_ou', 'linfo_ou'):
        m_o, rec_o[n] = SO.emulate_same_unet_init(n, 1)
        o_equal_go[n] = SO.sd_equal(go_sd, m_o.unet.state_dict(), skip=('down_path.0.block.0.weight',))[0]
        del m_o
    ok4 = (not rec_d['own_init'] and rec_d['n_copied'] == rec_d['n_total']
           and SO.unet_checksum(m_d) == SO.unet_checksum(m_u)
           and all([o['name'] for o in rec_o[n]['own_init']] == [first_w] for n in rec_o) and all(o_equal_go.values()))
    check('4_unet_init', ok4, d_rec=rec_d, o_recs=rec_o, o_equal_o_eter_except_first_conv=o_equal_go,
          checksum_unet_ou=SO.unet_checksum(m_u), checksum_ss2d_ou=SO.unet_checksum(m_d))
    del m_u, m_d, m_go

    # ---- 5. 푸리에 헤드 규약 (실제 val 슬라이스)
    import u_pure_eternet_ou_ss2d as M
    ds, vidx, samples = _val_samples(2)
    R.seed_all(1)
    head_f = M.FourierSS2D(c_in=32, d_inner=C.SS2D_D_INNER, d_state=C.SS2D_D_STATE, out_ch=20).to(dev).eval()
    head_l = M.FourierLinear(c_in=32, out_ch=20).to(dev).eval()
    errs = []
    dtypes = set()
    with torch.no_grad():
        for b in samples:
            x_ksp, x_img = b['data'].to(dev), b['data_img'].to(dev)
            s = R.per_sample_scale(x_img)
            x_ksp, x_img = R.normalize_inputs(x_ksp, x_img, s)
            ref = x_img[:, :20]
            scale = float(ref.abs().max())
            for head in (head_f, head_l):
                for ac in (False, True):
                    with torch.amp.autocast('cuda', enabled=ac):
                        y = head(x_ksp)
                    dtypes.add(str(y.dtype))
                    errs.append(float((y - ref).abs().max()) / max(scale, 1e-12))
    check('5_fourier_head_convention', max(errs) < 1e-5 and dtypes == {'torch.float32'},
          max_rel_err=max(errs), val_indices=vidx, out_dtypes=sorted(dtypes))
    del head_f, head_l

    # ---- 5b. 국소 k-space 크기 E: 빠진 줄(마스크 0)에서도 양수, 측정 줄의 크기를 따름
    env = {}
    for i, b in zip(vidx, samples):
        x_ksp, x_img = b['data'].to(dev), b['data_img'].to(dev)
        s = R.per_sample_scale(x_img)
        x_ksp, _ = R.normalize_inputs(x_ksp, x_img, s)
        E = M.kspace_envelope(x_ksp)
        mcols = b['mask'].reshape(-1, E.shape[-1]).to(dev).amax(dim=0) > 0   # 마지막 축의 측정 줄 (마스크는 (…, H, W) 꼴)
        a = torch.sqrt((x_ksp.float() ** 2).sum(dim=1, keepdim=True) / 16)
        rows = a[..., mcols].mean(dim=-1).reshape(-1) > 0                # 측정 줄에 신호가 있는 행
        miss = E[0, 0][rows][:, ~mcols]
        env[i] = dict(shape=list(E.shape), missing_cols=int((~mcols).sum()), frac_pos_missing=float((miss > 0).float().mean()),
                      center_over_edge=float(E[0, 0, :, E.shape[-1] // 2].mean() / E[0, 0, :, :8].mean().clamp(min=1e-30)))
    check('5b_kspace_envelope', all(v['shape'] == [1, 1, 384, 384] and v['frac_pos_missing'] > 0.99 and v['center_over_edge'] > 1
                                    for v in env.values()), envelope=env, width=M.ENVELOPE_WIDTH)

    # ---- 6. forward
    fwd = {}
    for n in names:
        R.seed_all(1)
        m = RO.build_model(n, dev).eval()
        x = torch.randn(1, 32, 384, 384, device=dev)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            y = m(x, x)
        fwd[n] = dict(shape=list(y.shape), finite=bool(torch.isfinite(y).all()))
        del m
    check('6_forward', all(v['shape'] == [1, 1, 384, 384] and v['finite'] for v in fwd.values()), forward=fwd)

    # ---- 7. 기울기 (o-SS2D-F (OU))
    R.seed_all(1)
    m = RO.build_model('ss2dfo_ou', dev).train()
    b = samples[0]
    x_ksp, x_img = b['data'].to(dev), b['data_img'].to(dev)
    s = R.per_sample_scale(x_img)
    x_ksp, x_img = R.normalize_inputs(x_ksp, x_img, s)
    tgt = torch.sqrt((x_img ** 2).sum(dim=1, keepdim=True))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')                               # 트레이너와 같음 (main_train_pure_v8fix_ou.py)

    def gnorm(prefix):
        g = [p.grad for n_, p in m.named_parameters() if n_.startswith(prefix) and p.grad is not None]
        return float(sum(float(t.norm()) for t in g)) if g else 0.0

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = m(x_img, x_ksp)
        loss = (out.float() - tgt).abs().mean()
        scaler.scale(loss).backward()                                   # 기울기는 배율이 곱해진 값 (0 여부만 본다)
        return float(loss)

    keys = ('ss2d.out_proj.', 'ss2d.res.', 'ss2d.in_proj.', 'ss2d.ssm_h_fwd.', 'unet.')
    step()
    g1 = {k: gnorm(k) for k in keys}
    skipped = 0
    for _ in range(30):                     # GradScaler 가 넘친 걸음을 건너뛰는 동안 (학습 초반과 같음) — 한 걸음이 반영될 때까지
        before = float(m.ss2d.out_proj.weight.detach().abs().sum())
        scaler.step(opt)
        scaler.update()
        if float(m.ss2d.out_proj.weight.detach().abs().sum()) != before:
            break
        skipped += 1
        step()
    step()
    g2 = {k: gnorm(k) for k in keys}
    ok7 = (g1['ss2d.out_proj.'] > 0 and g1['ss2d.res.'] > 0 and g1['ss2d.in_proj.'] == 0
           and float(m.ss2d.out_proj.weight.detach().abs().sum()) > 0
           and 0 < g2['ss2d.in_proj.'] < float('inf') and 0 < g2['ss2d.ssm_h_fwd.'] < float('inf'))
    check('7_gradients_fourier', ok7, step1=g1, step_after_update=g2, skipped_overflow_steps=skipped, scale=scaler.get_scale())
    del m, opt, scaler

    # ---- 8. 기능 시험 경로
    fres = {}
    idx4 = R.even_indices(len(ds), 4)
    for n in names:
        R.seed_all(1)
        m = RO.build_model(n, dev).eval()
        res = RO.run_validation(m, n, True, ds, idx4, dev, 0, False, True)
        fres[n] = dict(tests=sorted(res['tests']), decomp_equiv_maxabs=res.get('decomp_equiv_maxabs'), n=res['n'])
        del m
    ok8 = all(v['decomp_equiv_maxabs'] <= 1e-4 and {'zero', 'shuffle_cross', 'shuffle_adj', 'deep4', 'deep34'} <= set(v['tests'])
              for v in fres.values())
    check('8_functional_path_gpu', ok8, results=fres, indices=idx4,
          note='GPU autocast: decomp_equiv_maxabs <= 1e-4 (fp16; 정확한 일치는 8b 에서 CPU 로)')

    # ---- 9. 건강 감시 그룹
    R.seed_all(1)
    g_ref = R.build_groups(R.build_model('ss2d', cpu), 'ss2d')
    g_u52 = RO.build_groups(RO.build_model('unet_ou', cpu), 'unet_ou')
    g_u20 = RO.build_groups(RO.build_model('gruo_ou', cpu), 'gruo_ou')
    grp = {}
    ok9 = True
    for n in names:
        R.seed_all(1)
        mm = RO.build_model(n, cpu)
        g = RO.build_groups(mm, n)
        sizes = {k: int(sum(dict(mm.named_parameters())[p].numel() for p in v)) for k, v in g.items()}
        grp[n] = sizes
        g_un = g_u52 if n == 'ss2d_ou' else g_u20
        for k in [k for k in g_un if k.startswith('unet')]:
            if k not in g or len(g[k]) != len(g_un[k]):
                ok9 = False
        if n != 'linfo_ou':
            for k in ('seq.ss2d_in', 'seq.ss2d_ssm', 'seq.ss2d_out'):
                if k not in g or len(g[k]) != len(g_ref[k]):
                    ok9 = False
        del mm
    check('9_groups', ok9, group_param_counts=grp)

    # ---- 10. 원본 무수정·기준 파일 판
    def git_quiet(*a):
        return subprocess.run(['git', '-C', _ROOT, 'diff', '--quiet', *a], capture_output=True).returncode == 0
    orig_ok = (git_quiet('7d4e4e0', '--', 'models/hybrid_eternet/myUNet_DF.py')
               and git_quiet('HEAD', '--', 'models/mamba_eternet/ss2d.py', 'dataloaders/dataloader_h5_v5.py'))
    shas = {p: _sha(p) for p in OU_BASE_SHA256}
    check('10_originals_and_base_files', orig_ok and all(shas[p] == OU_BASE_SHA256[p] for p in shas),
          originals_unchanged=orig_ok, base_sha256_equal={p: shas[p] == OU_BASE_SHA256[p] for p in shas})

    # ---- 8b. 대리 객체 배선 (CPU, fp32, 정확히): 분해 경로 = model.forward, zero 경로 = U-Net(0) — 마지막에 (CPU shim 이 전역을 바꿈)
    R.install_cpu_shims(need_ss2d=True)
    exact = {}
    for n in names:
        R.seed_all(1)
        m = RO.build_model(n, cpu).eval()
        g = torch.Generator().manual_seed(0)
        x_ksp, x_img = torch.randn(1, 32, 64, 64, generator=g), torch.randn(1, 32, 64, 64, generator=g)
        obj, base = RO.functional_target(m, n)
        with torch.no_grad():
            unet_in, so = R.unet_input(obj, base, x_ksp, x_img)
            full = m(x_img, x_ksp)
            dec = obj.unet(unet_in)
            z = obj.unet(torch.cat((torch.zeros_like(so), x_img), dim=1))
            z0 = torch.zeros(so.shape, dtype=so.dtype)                  # NCHW (zeros_like 는 SS2D 출력의 channels-last 배치를 물려받음)
            z_ref = m.unet(z0) if RO.MODELS[n]['seq_only_input'] else m.unet(torch.cat((z0, x_img), 1))
        exact[n] = dict(decomp_equal=bool(torch.equal(full, dec)), zero_equal=bool(torch.equal(z, z_ref)), seq_shape=list(so.shape))
        del m
    check('8b_functional_wiring_cpu_exact', all(v['decomp_equal'] and v['zero_equal'] for v in exact.values()), results=exact)

    print(f'\n{len(SO.RESULTS) - len(SO.FAILS)}/{len(SO.RESULTS)} 점검 통과' + (f' — 실패 {SO.FAILS}' if SO.FAILS else ''))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(ok=not SO.FAILS, fails=SO.FAILS, results=SO.RESULTS), f, ensure_ascii=False, indent=1, default=str)
        print(f'→ {args.out}')
    return 1 if SO.FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
