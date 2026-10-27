"""CPU 점검(GPU 안 씀): 파라미터 수(교수님 코드 식·논문), 원문 클래스와 사본의 키·초기값, 작은 크기 순전파·역전파,
설정 출처, optimizer 대체, 감시 그룹.

  python -B repro_prof/selftest_cpu.py [--json OUT.json]
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_prof import config_repro as CR  # noqa: E402
from repro_prof import models_repro as M  # noqa: E402
from repro_prof import monitor as MON  # noqa: E402
from repro_prof import myoptimizer_shim as OPT  # noqa: E402
from repro_prof import prof_paths as P  # noqa: E402


def count(spec, refine='dfu'):
    with torch.device('meta'):
        m = M.make_for_count(spec, refine)
    return m, M.param_breakdown(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = dict(ok=True, checks=[])

    def check(name, cond, detail=''):
        res['checks'].append(dict(name=name, ok=bool(cond), detail=detail))
        if not cond:
            res['ok'] = False
        print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))

    # 1) 파라미터 수
    for desc, spec, exp in M.EXPECTED_PARAMS:
        _, pb = count(spec)
        check(f'params {desc}', pb['total'] == exp, f"{pb['total']:,} vs {exp:,}; {pb}")
    for desc, spec, mods, exp in M.EXPECTED_SUBMODULE:
        _, pb = count(spec)
        got = sum(pb[m] for m in mods)
        check(f'params {desc}', got == exp, f'{got:,} vs {exp:,}')
    other = {}
    for name, kw in [('unet2015 rss1 depth5 wf6', dict(input_form='rss1', refine='unet2015', depth=5)),
                     ('unet2015 coil32 depth5 wf6', dict(input_form='coil32', refine='unet2015', depth=5)),
                     ('dfu_plain rss1 depth5', dict(input_form='rss1', refine='dfu_plain', depth=5)),
                     ('fastmri rss1 chans64 pools4', dict(input_form='rss1', refine='fastmri', depth=5)),
                     ('fastmri rss1 chans32 pools4', dict(input_form='rss1', refine='fastmri', depth=5, fastmri_chans=32)),
                     ('fastmri coil32 chans64 pools4', dict(input_form='coil32', refine='fastmri', depth=5)),
                     ('dfu coil32 depth5', dict(input_form='coil32', refine='dfu', depth=5))]:
        with torch.device('meta'):
            m = M.UNetOnlyRepro(**kw)
        other[name] = M.param_breakdown(m)['total']
    print('other baselines:', json.dumps(other, indent=1))
    res['other_params'] = other
    check('fastMRI U-Net chans32 pools4 in1 ≈ 7.76M (교수님 2026 공유 실험 로그의 fastMRI U-Net chans 32)',
          abs(other['fastmri rss1 chans32 pools4'] - 7.76e6) < 0.01e6, f"{other['fastmri rss1 chans32 pools4']:,}")

    # 2) 원문 클래스(AST 추출)와 사본: 키·모양·같은 시드 초기값 (작은 크기, CPU — 원문 forward 는 .cuda() 라 GPU 스모크에서)
    tiny = dict(N_INPUT_VERTICAL=32, N_FREQ_ENCODING=32, N_RFCOIL=2, N_OUT_X=32, N_OUT_Y=32, N_HIDDEN_LRNN_1=2,
                N_HIDDEN_LRNN_2=2, N_UNET_DEPTH=3)
    for variant, script, cls in (('d', P.PROF_DETER_SCRIPT, 'ETER_hybrid_GRU_DFU'), ('o', P.PROF_OETER_SCRIPT, 'oETER_GRU_DFU')):
        Orig = P.exec_prof_class(script, cls, tiny)
        torch.manual_seed(7)
        ma = Orig()
        torch.manual_seed(7)
        mb = M.ETERNetRepro(variant=variant, n_in_x=32, n_freq=32, n_coil=2, n_out_x=32, n_out_y=32, h1=2, h2=2, depth=3)
        sa, sb = ma.state_dict(), mb.state_dict()
        check(f'{cls} 원문 vs 사본: state_dict 키 순서', list(sa) == list(sb), f'{len(sa)} keys')
        check(f'{cls} 원문 vs 사본: 같은 시드 초기값 비트 동일', all(torch.equal(sa[k], sb[k]) for k in sa))

    # 3) 작은 크기 순전파·역전파 (모든 정제망 × d/o, U-Net 단독 coil32/rss1)
    B = 2
    x = torch.randn(B, 4, 32, 32)
    xi = torch.randn(B, 4, 32, 32)
    y = torch.rand(B, 1, 32, 32)
    for refine in M.REFINERS:
        for variant in ('d', 'o'):
            torch.manual_seed(0)
            m = M.ETERNetRepro(variant=variant, n_in_x=32, n_freq=32, n_coil=2, n_out_x=32, n_out_y=32, h1=2, h2=2,
                               refine=refine, depth=3)
            out = m(x, xi)
            loss = torch.nn.functional.l1_loss(out, y)
            loss.backward()
            g_gru = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith('gru'))
            g_unet = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith('unet'))
            check(f'tiny {variant}-ETER refine={refine}: out {tuple(out.shape)}, grad gru>0 unet>0',
                  out.shape == (B, 1, 32, 32) and g_gru > 0 and g_unet > 0 and torch.isfinite(loss),
                  f'loss {float(loss):.4f}')
        for form in ('coil32', 'rss1'):
            if refine == 'dfu' and form == 'rss1':
                try:
                    M.UNetOnlyRepro(input_form='rss1', refine='dfu', depth=3, n_coil=2)
                    check('dfu + rss1 은 거부되어야 함', False)
                except ValueError:
                    check('dfu + rss1 은 거부됨(UNet_choh_skip 입력 = 2·n_hidden)', True)
                continue
            torch.manual_seed(0)
            m = M.UNetOnlyRepro(input_form=form, refine=refine, depth=3, n_coil=2)
            out = m(x, xi)
            loss = torch.nn.functional.l1_loss(out, y)
            loss.backward()
            check(f'tiny U-Net 단독 {form} refine={refine}', out.shape == (B, 1, 32, 32) and torch.isfinite(loss),
                  f'in_ch {m.unet_in_ch}')
    # rss1 입력이 코일 결합 크기인지
    m = M.UNetOnlyRepro(input_form='rss1', refine='unet2015', depth=3, n_coil=2)
    r = m.make_input(xi)
    check('rss1 = sqrt(sum_c x_img²)', torch.allclose(r, xi.pow(2).sum(1, keepdim=True).sqrt()))

    # 4) 설정: 모든 값에 출처·상태, 프리셋 3개, 메타 파라미터 수
    for preset in CR.PRESETS:
        cfg = CR.Config(preset)
        bad = [s.name for s in cfg.settings.values() if not s.src or s.status not in CR.STATUS_TAG]
        check(f'config {preset}: 모든 설정에 출처·상태', not bad, str(bad))
        buf = io.StringIO()
        cfg.print_all(write=lambda s, b=buf: b.write(s + '\n'))
        with torch.device('meta'):
            mm = M.build_model(cfg)
        pb = M.param_breakdown(mm)
        print(f'preset {preset}: run {cfg.run_name}, params {pb}, assumed {len(cfg.assumed())}')
        res[f'preset_{preset}'] = dict(run_name=cfg.run_name, params=pb, assumed=[s.name for s in cfg.assumed()])
    check('dETER preset params = 851,350,585', res['preset_dETER']['params']['total'] == 851_350_585)
    check('oETER preset params = 851,332,121', res['preset_oETER']['params']['total'] == 851_332_121)
    check('unet_only preset params = 1,879,585', res['preset_unet_only']['params']['total'] == 1_879_585)
    cfg = CR.Config('dETER', CR.parse_overrides(['N_UNET_DEPTH=5', 'BATCH_SIZE=1']))
    check('override 표시', cfg.settings['N_UNET_DEPTH'].status == 'override' and cfg['BATCH_SIZE'] == 1)

    # 5) optimizer 대체
    tm = torch.nn.Linear(3, 3)
    o = OPT.myOptimizer(tm, 'else', 1e-4)
    check("shim lr_passthrough 'else' → Adam lr 1e-4 WD 0", isinstance(o, torch.optim.Adam)
          and o.param_groups[0]['lr'] == 1e-4 and o.param_groups[0]['weight_decay'] == 0)
    o = OPT.myOptimizer(tm, 'else', 3e-4, mode='legacy_2arg')
    check("shim legacy_2arg 'else' → 저장소 판 그대로 lr 1e-4", o.param_groups[0]['lr'] == 1e-4)
    o = OPT.myOptimizer(tm, 'adam', 3e-4, mode='legacy_2arg')
    check("shim legacy_2arg 'adam' → 저장소 판 lr 1e-7", o.param_groups[0]['lr'] == 1e-7)
    o = OPT.myOptimizer(tm, 'else', 1e-4, mode='coupled_wd', weight_decay=1e-7)
    check('shim coupled_wd → Adam lr 1e-4 WD 1e-7', o.param_groups[0]['weight_decay'] == 1e-7)
    try:
        P.prof_utils_module().myOptimizer(tm, 'else', 1e-4)
        check('저장소 myOptimizer 에 인자 3개 → TypeError', False)
    except TypeError:
        check('저장소 myOptimizer 에 인자 3개 → TypeError (231030:191 이 그대로는 안 도는 이유)', True)

    # 6) 감시 그룹
    for refine, depth in (('dfu', 3), ('dfu', 5), ('unet2015', 5), ('fastmri', 5), ('dfu_plain', 5)):
        with torch.device('meta'):
            mm = M.ETERNetRepro(variant='d', refine=refine, depth=depth, h1=1, h2=1)
        g = MON.build_groups(mm, refine, depth)
        check(f'monitor groups {refine} depth{depth}', {'seq', 'unet_all', 'unet_deep'} <= set(g),
              ', '.join(f'{k}:{len(v)}' for k, v in g.items()))
    m = M.ETERNetRepro(variant='d', n_in_x=32, n_freq=32, n_coil=2, n_out_x=32, n_out_y=32, h1=2, h2=2, depth=3)
    st = MON.param_stats(m, MON.build_groups(m, 'dfu', 3))
    check('monitor stats (CPU)', all(0 <= v['near'] <= 1 for v in st.values()), MON.format_health(st, 0, 0, 'cpu'))

    print('\nALL OK' if res['ok'] else '\nSOME CHECKS FAILED')
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
