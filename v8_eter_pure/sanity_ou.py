"""
원래 U-Net(OU) 판 세 모델의 계약 점검 (2026-10-26) — 계획 paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md.
CPU 로 돈다(GPU 를 건드리지 않음): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python -B v8_eter_pure/sanity_ou.py [--out x.json]

점검 (하나라도 실패하면 종료 코드 1):
  1. 파라미터 수: OriginalUNet(52)·OriginalUNet(20)·DFU(UNet_choh_skip, 입력 52) 와 세 모델 전체·시퀀스 모듈·U-Net.
  2. forward 모양: 세 모델 (1, 32, 384, 384) ×2 → (1, 1, 384, 384), 유한.
  3. 같은 U-Net 구조: d-ETER (OU) 의 U-Net state_dict 키·모양 = U-Net only (OU). o-ETER (OU) 는 키가 같고 모양은 첫 합성곱 weight 만 다름.
  4. 같은 초기 U-Net 가중치 (한 시드, 트레이너 SAME_UNET_INIT 와 같은 절차 — recipe_ou.copy_unet_init):
       seed_all(1) 직후 만든 U-Net only (OU) 의 U-Net = d-ETER (OU) 에 복사한 U-Net (텐서 전부 비트 일치, 트레이너의 unet_init_checksum 식 일치).
       o-ETER (OU) 는 모양이 같은 텐서 전부 일치, 자기 초기값 = 첫 합성곱 weight 하나뿐.
     bi-GRU: d-ETER (OU)·o-ETER (OU) 의 gru_h·gru_v 초기값 = seed_all(1) 직후 만든 ETER-net(DFU) 과 비트 일치, _seq 는 상속(재정의 없음).
  5. U-Net only (OU) 의 0 채널: 첫 합성곱 weight[:, :20] 의 기울기 = 정확히 0 (나머지는 0 아님), 그 slice 를 바꿔도 출력이 비트까지 같음.
  6. 원래 U-Net 에 dual-frame 잔차 빼기·입력 재연결이 없음 (DFU 로 검출기 자체도 확인):
       - 올림 블록 호출 인자 = (x, skip) 둘뿐 (DFU 는 두 번째 올림부터 residual 셋째 인자)
       - 첫 합성곱 weight·bias 를 0 으로 하면 출력이 입력과 무관 (입력은 첫 합성곱으로만 들어감; DFU 는 마지막 1×1 이 입력을 다시 읽어 달라짐)
       - 마지막 1×1 합성곱 입력 채널 = 64 (DFU = 52 + 64)
       - pooling = max_pool2d 4 번, avg_pool2d 0 번 (DFU 는 avg 4 번)
  7. 건강 감시 그룹(smoke_recipe_fix.build_groups)·깊은 층 그룹의 원소 수가 DFU 와 같은 단계를 가리킴.
  8. o-ETER (OU) 기능 시험 대리 객체: 분해 경로(R.unet_input → 대리 U-Net)·zero 경로가 model.forward·U-Net(0) 과 비트 일치.
  9. 런 폴더 이름 규칙(recipe_ou.run_name) = 런처 infra/runpod/pod_run_ou.sh 의 규칙(이름에서 '_' 를 빼고 대문자).
"""

import os
import sys
import json
import random
import argparse
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)

import numpy as np                 # noqa: E402
import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

import recipe_ou as RO             # noqa: E402  (smoke_recipe_fix 를 import — sys.path 에 models/* 추가)

R, C = RO.R, RO.C
RESULTS = OrderedDict()
FAILS = []


def check(name, ok, **info):
    RESULTS[name] = dict(ok=bool(ok), **info)
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  {json.dumps(info, ensure_ascii=False, default=str)}' if info else ''),
          flush=True)
    if not ok:
        FAILS.append(name)


def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))


def sd_equal(a, b, skip=()):
    """두 state_dict 의 텐서 비트 일치 (skip 키 제외). 반환 (전부 같음, 다른 키 목록)."""
    diff = [k for k in a if k not in skip and not torch.equal(a[k], b[k])]
    return not diff, diff


def unet_checksum(model):
    """트레이너 SCRATCH START 의 unet_init_checksum 과 같은 식 (main_train_pure_v8fix.py)."""
    with torch.no_grad():
        return float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(model.unet.parameters())))


def emulate_same_unet_init(name, seed):
    """트레이너의 U-Net 초기값 통일 절차: seed_all(SEED) → 모델, fork_rng 안에서 seed_all(SEED) → 기준 모델 → 복사."""
    R.seed_all(seed)
    model = RO.build_model(name, torch.device('cpu'))
    rec = None
    ref_name = RO.unet_ref_name(name)
    if name != ref_name:
        np_state, py_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=[]):
            R.seed_all(seed)
            ref = RO.build_model(ref_name, torch.device('cpu'))
        np.random.set_state(np_state)
        random.setstate(py_state)
        rec = RO.copy_unet_init(model, ref, partial=RO.partial_unet_copy(name))
        del ref
    return model, rec


class CallCounter:
    """F.max_pool2d / F.avg_pool2d 호출 수 (원 함수는 그대로 부른다; 끝나면 되돌림)."""

    def __enter__(self):
        self.n = dict(max_pool2d=0, avg_pool2d=0)
        self._orig = {k: getattr(F, k) for k in self.n}
        for k, f in self._orig.items():
            def w(*a, _k=k, _f=f, **kw):
                self.n[_k] += 1
                return _f(*a, **kw)
            setattr(F, k, w)
        return self

    def __exit__(self, *exc):
        for k, f in self._orig.items():
            setattr(F, k, f)


def up_call_args(unet, x):
    """올림 블록마다 forward 에 들어온 위치 인자 수와 키워드 인자 이름."""
    seen = []
    hs = [up.register_forward_pre_hook(lambda m, a, kw, i=i: seen.append((i, len(a), sorted(kw))), with_kwargs=True)
          for i, up in enumerate(unet.up_path)]
    try:
        with torch.no_grad():
            unet(x)
    finally:
        for h in hs:
            h.remove()
    return seen


def input_only_via_first_conv(unet, in_ch, gen):
    """첫 합성곱 weight·bias 를 0 으로 한 사본에서 서로 다른 두 입력의 출력이 같은가 (입력 경로가 첫 합성곱 하나뿐인가)."""
    import copy
    u = copy.deepcopy(unet).eval()
    with torch.no_grad():
        u.down_path[0].block[0].weight.zero_()
        u.down_path[0].block[0].bias.zero_()
        a = torch.randn(1, in_ch, 64, 64, generator=gen)
        b = torch.randn(1, in_ch, 64, 64, generator=gen) * 3 + 1
        return bool(torch.equal(u(a), u(b)))


def main():
    ap = argparse.ArgumentParser(description='원래 U-Net(OU) 세 모델 계약 점검 (CPU)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None, help='결과 JSON 경로 (선택)')
    args = ap.parse_args()
    if torch.cuda.is_available():
        print('[note] CUDA 가 보이지만 이 점검은 CPU 로만 계산한다 (CUDA_VISIBLE_DEVICES="" 권장)')
    torch.set_grad_enabled(True)
    cpu = torch.device('cpu')
    seed = args.seed
    from myUNet_DF import UNet_choh_skip     # 교수님 DFU (비교용으로만 만듦)
    from unet_original import OriginalUNet
    import u_pure_eternet_ou as M
    from u_pure_eternet_gru import PureETER_GRU

    in52 = 2 * C.N_HIDDEN_LRNN_2 + 2 * C.N_COIL
    in20 = 2 * C.N_HIDDEN_LRNN_2
    # ---- 1. 파라미터 수
    torch.manual_seed(0)
    ou52, ou20 = OriginalUNet(in52, 1, C.UNET_DEPTH, C.UNET_WF), OriginalUNet(in20, 1, C.UNET_DEPTH, C.UNET_WF)
    dfu52 = UNet_choh_skip(in_channels=in52, n_classes=1, depth=C.UNET_DEPTH, wf=C.UNET_WF, padding=True,
                           batch_norm=False, up_mode='upconv', n_hidden=C.N_HIDDEN_LRNN_2 + C.N_COIL)
    counts = OrderedDict(original_unet_in52=nparams(ou52), original_unet_in20=nparams(ou20), dfu_in52=nparams(dfu52))
    models = OrderedDict()
    for name in RO.OU_NAMES:
        m, rec = emulate_same_unet_init(name, seed)
        models[name] = (m, rec)
    R.seed_all(seed)
    dfu_gru = R.build_model('gru', cpu)
    R.seed_all(seed)
    dfu_unet = R.build_model('unet', cpu)
    for name, (m, _) in list(models.items()) + [('gru (DFU ETER-net)', (dfu_gru, None)), ('unet (DFU U-Net only)', (dfu_unet, None))]:
        seq_n = sum(p.numel() for n_, p in m.named_parameters() if n_.startswith(('gru_h.', 'gru_v.')))
        counts[name] = dict(total=nparams(m), seq_module=int(seq_n), unet=nparams(m.unet))
    expect_last = dict(original_unet_in52=2 ** C.UNET_WF + 1, dfu_in52=(2 ** C.UNET_WF + in52) + 1)
    check('1_param_counts', counts['original_unet_in52'] - counts['dfu_in52'] == (expect_last['original_unet_in52']
                                                                                  - expect_last['dfu_in52'])
          and counts['original_unet_in52'] - counts['original_unet_in20'] == (in52 - in20) * 9 * 2 ** C.UNET_WF
          and counts['gru_ou']['seq_module'] == counts['gru (DFU ETER-net)']['seq_module']
          and counts['gruo_ou']['seq_module'] == counts['gru (DFU ETER-net)']['seq_module'],
          counts=counts, note='DFU 와의 차 = 마지막 1×1 합성곱 입력 채널(116 → 64)뿐; 52 vs 20 = 첫 합성곱 입력 채널 32 개 × 9 × 64')

    # ---- 2. forward 모양
    g = torch.Generator().manual_seed(123)
    x_img = torch.randn(1, 2 * C.N_COIL, C.IMAGE_SIZE[0], C.IMAGE_SIZE[1], generator=g)
    x_ksp = torch.randn(1, 2 * C.N_COIL, C.IMAGE_SIZE[0], C.IMAGE_SIZE[1], generator=g) * 0.05
    shapes = OrderedDict()
    for name, (m, _) in models.items():
        m.eval()
        with torch.no_grad():
            out = m(x_img, x_ksp)
        shapes[name] = dict(out=list(out.shape), finite=bool(torch.isfinite(out).all()))
    check('2_forward_shapes', all(v['out'] == [1, 1, C.IMAGE_SIZE[0], C.IMAGE_SIZE[1]] and v['finite'] for v in shapes.values()),
          shapes=shapes)

    # ---- 3. 같은 U-Net 구조
    su = models['unet_ou'][0].unet.state_dict()
    sd_ = models['gru_ou'][0].unet.state_dict()
    so_ = models['gruo_ou'][0].unet.state_dict()
    same_d = list(su) == list(sd_) and all(su[k].shape == sd_[k].shape for k in su)
    diff_o = [k for k in su if su[k].shape != so_[k].shape]
    check('3_same_unet_structure', same_d and list(su) == list(so_) and diff_o == ['down_path.0.block.0.weight']
          and type(models['unet_ou'][0].unet) is OriginalUNet and type(models['gru_ou'][0].unet) is OriginalUNet
          and type(models['gruo_ou'][0].unet) is OriginalUNet,
          n_tensors=len(su), gru_ou_same_keys_shapes=same_d, gruo_ou_shape_differs=diff_o,
          gruo_ou_first_conv=list(so_['down_path.0.block.0.weight'].shape), ref_first_conv=list(su['down_path.0.block.0.weight'].shape))

    # ---- 4. 같은 초기 U-Net 가중치 (한 시드)
    eq_d, dk = sd_equal(su, sd_)
    eq_o, ok_ = sd_equal(su, so_, skip=('down_path.0.block.0.weight',))
    rec_d, rec_o = models['gru_ou'][1], models['gruo_ou'][1]
    cks = {n: unet_checksum(models[n][0]) for n in RO.OU_NAMES}
    cks['unet (DFU U-Net only)'] = unet_checksum(dfu_unet)
    check('4a_same_initial_unet_weights', eq_d and rec_d['own_init'] == [] and rec_d['n_copied'] == len(su)
          and cks['unet_ou'] == cks['gru_ou'],
          gru_ou_copy=rec_d, unet_init_checksum=cks, note_gru_ou=RO.unet_init_note('gru_ou', rec_d))
    check('4b_gruo_ou_partial_copy', eq_o and [o['name'] for o in rec_o['own_init']] == ['unet.down_path.0.block.0.weight']
          and rec_o['n_copied'] == len(su) - 1,
          gruo_ou_copy=rec_o, note_gruo_ou=RO.unet_init_note('gruo_ou', rec_o),
          caveat='첫 합성곱 bias (64,) 는 모양이 같아 기준(입력 52 채널의 기본 초기 분포)에서 복사된다')
    sg = dfu_gru.state_dict()
    gru_eq = {}
    for name in RO.OU_GRU_NAMES:
        s = models[name][0].state_dict()
        keys = [k for k in sg if k.startswith(('gru_h.', 'gru_v.'))]
        gru_eq[name] = bool(keys) and all(torch.equal(s[k], sg[k]) for k in keys) and \
            [k for k in s if k.startswith(('gru_h.', 'gru_v.'))] == keys
    inh = {c.__name__: ('_seq' not in vars(c)) and issubclass(c, PureETER_GRU) for c in (M.PureETER_GRUOU, M.PureETER_GRUOOU)}
    check('4c_bigru_unchanged_same_init', all(gru_eq.values()) and all(inh.values()),
          gru_init_equal_to_dfu_eternet=gru_eq, seq_inherited_from_PureETER_GRU=inh,
          gru_h=str(models['gru_ou'][0].gru_h), gru_v=str(models['gru_ou'][0].gru_v))

    # ---- 5. U-Net only (OU) 0 채널
    mu = models['unet_ou'][0]
    mu.train()
    mu.zero_grad(set_to_none=True)
    tgt = torch.rand(1, 1, C.IMAGE_SIZE[0], C.IMAGE_SIZE[1], generator=g)
    loss = (mu(x_img, x_ksp) - tgt).abs().mean()
    loss.backward()
    gw = mu.unet.down_path[0].block[0].weight.grad
    zero_slice_all0 = bool((gw[:, :in20] == 0).all())
    rest_nonzero = float(gw[:, in20:].abs().max())
    with torch.no_grad():
        mu.eval()
        o1 = mu(x_img, x_ksp)
        w = mu.unet.down_path[0].block[0].weight
        keep = w[:, :in20].clone()
        w[:, :in20] = torch.randn(keep.shape, generator=g)
        o2 = mu(x_img, x_ksp)
        w[:, :in20] = keep
    check('5_zero_channels_zero_grad', zero_slice_all0 and rest_nonzero > 0 and torch.equal(o1, o2),
          grad_zero_slice_maxabs=float(gw[:, :in20].abs().max()), grad_rest_maxabs=rest_nonzero,
          output_unchanged_when_zero_slice_weights_changed=bool(torch.equal(o1, o2)))
    mu.zero_grad(set_to_none=True)

    # ---- 6. 잔차 빼기·입력 재연결 없음 (DFU 로 검출기 확인)
    gx = torch.Generator().manual_seed(7)
    xs = torch.randn(1, in52, 64, 64, generator=gx)
    up_ou, up_dfu = up_call_args(ou52, xs), up_call_args(dfu52, xs)
    with CallCounter() as cc_ou:
        with torch.no_grad():
            ou52(xs)
    with CallCounter() as cc_dfu:
        with torch.no_grad():
            dfu52(xs)
    only_first_ou = input_only_via_first_conv(ou52, in52, gx)
    only_first_dfu = input_only_via_first_conv(dfu52, in52, gx)
    ok6 = (all(n == 2 and not kw for _, n, kw in up_ou) and any(n == 3 for _, n, _ in up_dfu)
           and only_first_ou and not only_first_dfu
           and ou52.last.in_channels == 2 ** C.UNET_WF and dfu52.last.in_channels == 2 ** C.UNET_WF + in52
           and cc_ou.n == dict(max_pool2d=C.UNET_DEPTH - 1, avg_pool2d=0)
           and cc_dfu.n == dict(max_pool2d=0, avg_pool2d=C.UNET_DEPTH - 1))
    check('6_no_residual_no_input_reconcat', ok6,
          up_block_args_original=[(i, n) for i, n, _ in up_ou], up_block_args_dfu=[(i, n) for i, n, _ in up_dfu],
          input_only_via_first_conv=dict(original=only_first_ou, dfu=only_first_dfu),
          last_in_channels=dict(original=ou52.last.in_channels, dfu=dfu52.last.in_channels),
          pooling_calls=dict(original=cc_ou.n, dfu=cc_dfu.n))

    # ---- 7. 건강 감시 그룹
    gstat = OrderedDict()
    for name, m in [(n, models[n][0]) for n in RO.OU_NAMES] + [('gru', dfu_gru), ('unet', dfu_unet)]:
        grp = RO.build_groups(m, name)
        numel = {k: int(sum(dict(m.named_parameters())[p].numel() for p in v)) for k, v in grp.items()}
        gstat[name] = dict(groups=list(grp), unet_deep=numel.get('unet_deep'), seq=numel.get('seq'),
                           unet_all=numel.get('unet_all'))
    ok7 = (gstat['unet_ou']['unet_deep'] == gstat['unet']['unet_deep'] == gstat['gru_ou']['unet_deep']
           == gstat['gruo_ou']['unet_deep'] and gstat['gru_ou']['seq'] == gstat['gruo_ou']['seq'] == gstat['gru']['seq']
           and gstat['unet_ou']['seq'] is None)
    check('7_health_groups', ok7, groups=gstat)

    # ---- 8. o-ETER (OU) 기능 시험 대리 객체
    mo = models['gruo_ou'][0].eval()
    view, base = RO.functional_target(mo, 'gruo_ou')
    with torch.no_grad():
        unet_in, so = R.unet_input(view, base, x_ksp, x_img)
        out_view = view.unet(unet_in)
        out_fwd = mo(x_img, x_ksp)
        out_full_view = view(x_img, x_ksp, None, None)
        out_zero_view = view.unet(torch.cat((torch.zeros_like(so), x_img), dim=1))
        out_zero = mo.unet(torch.zeros_like(so))
    check('8_gruo_ou_functional_view', base == 'gru' and unet_in.shape[1] == in52 and torch.equal(out_view, out_fwd)
          and torch.equal(out_full_view, out_fwd) and torch.equal(out_zero_view, out_zero)
          and view.unet.down_path is mo.unet.down_path,
          base_seq=base, unet_input_channels=int(unet_in.shape[1]),
          decomp_maxabs=float((out_view - out_fwd).abs().max()), zero_maxabs=float((out_zero_view - out_zero).abs().max()))

    # ---- 9. 런 폴더 이름 규칙 (런처와 같은가)
    sh = os.path.join(_ROOT, 'infra', 'runpod', 'pod_run_ou.sh')
    rule_line = "TAG=$(echo \"$MODEL\" | tr -d '_' | tr '[:lower:]' '[:upper:]')"
    has_rule = os.path.exists(sh) and rule_line in open(sh).read()
    names = {n: RO.run_name(n, '_s1_fullval') for n in RO.OU_NAMES}
    check('9_run_name_rule', has_rule and all(v == f"PureETER_{n.replace('_', '').upper()}_noDC_R4_brain384_v8fix_s1_fullval"
                                              for n, v in names.items()),
          run_names=names, launcher=os.path.relpath(sh, _ROOT), launcher_rule_found=has_rule)

    print(f'\n{len(RESULTS) - len(FAILS)}/{len(RESULTS)} 점검 통과' + (f' — 실패 {FAILS}' if FAILS else ''))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(ok=not FAILS, fails=FAILS, results=RESULTS), f, ensure_ascii=False, indent=1, default=str)
        print(f'→ {args.out}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
