"""가중치 소멸 감시 (우리 v8fix 트레이너의 생각을 옮김 — 교수님 레시피에는 없다, 기본은 기록만).

v8fix(main_train_pure_v8fix.py:34-36, smoke_recipe_fix.py:161·165·474-499): HEALTH_EVERY_STEPS step 마다와 epoch 끝에
그룹별 |w| < 1e-6 비율을 기록하고, seq 또는 unet_deep 이 0.5 이상이면 멈췄다. 여기서는 같은 문턱·같은 그룹 이름을 쓰되,
COLLAPSE_ACTION='log'(기본)이면 학습에 개입하지 않는다(교수님 레시피에 없는 중단 장치를 넣지 않기 위해; 중단 여부는 사용자가 정함).
'stop' 이면 ALERT_COLLAPSE 파일을 쓰고 종료 코드 3 으로 끝낸다.
"""
from collections import OrderedDict

import torch

NEAR_THR = 1e-6           # v8fix 와 같음 (smoke_recipe_fix.py:161)
DEAD_THR = 1e-30          # v8fix 와 같음 (smoke_recipe_fix.py:161)
TH_COLLAPSED_NEAR = 0.5   # v8fix 와 같음 (smoke_recipe_fix.py:170)
CHUNK = 1 << 25           # 큰 텐서는 3,200만 원소씩 나눠 센다(GPU 임시 메모리 상한)


def _unet_prefix(model):
    names = [n for n, _ in model.named_parameters()]
    return 'unet.net.' if any(n.startswith('unet.net.') for n in names) else 'unet.'


def deep_prefixes(model, refine, depth):
    """U-Net 의 가장 깊은 두 수준과 첫 업샘플 블록(v8fix DEEP_PREFIXES 를 깊이에 맞춰 일반화)."""
    u = _unet_prefix(model)
    if refine in ('dfu', 'dfu_plain'):
        return (f'{u}down_path.{depth - 2}.', f'{u}down_path.{depth - 1}.', f'{u}up_path.0.')
    if refine == 'unet2015':
        return (f'{u}down.{depth - 2}.', f'{u}down.{depth - 1}.', f'{u}upconv.0.', f'{u}up.0.')
    if refine == 'fastmri':
        n_down = sum(1 for n, _ in model.named_parameters() if n.startswith(f'{u}down_sample_layers.') and n.endswith('layers.0.weight'))
        return (f'{u}down_sample_layers.{n_down - 1}.', f'{u}conv.', f'{u}up_transpose_conv.0.', f'{u}up_conv.0.')
    return ()


def build_groups(model, refine, depth):
    names = [n for n, _ in model.named_parameters()]
    g = OrderedDict()

    def add(key, pred):
        lst = [n for n in names if pred(n)]
        if lst:
            g[key] = lst

    add('seq', lambda n: n.startswith(('gru_h.', 'gru_v.')))
    add('seq.gru_h', lambda n: n.startswith('gru_h.'))
    add('seq.gru_v', lambda n: n.startswith('gru_v.'))
    dp = deep_prefixes(model, refine, depth)
    if dp:
        add('unet_deep', lambda n: n.startswith(dp))
    add('unet_all', lambda n: n.startswith('unet.'))
    return g


@torch.no_grad()
def param_stats(model, groups):
    """그룹별 near-zero(|w|<1e-6) 비율, dead(|w|<1e-30) 비율, RMS."""
    params = dict(model.named_parameters())
    per = {}
    for n, p in params.items():
        flat = p.detach().reshape(-1)
        near = dead = 0
        sq = 0.0
        for i in range(0, flat.numel(), CHUNK):
            c = flat[i:i + CHUNK]
            a = c.abs()
            near += int((a < NEAR_THR).sum())
            dead += int((a < DEAD_THR).sum())
            sq += float((c.double() ** 2).sum())
        per[n] = (near, dead, sq, flat.numel())
    out = OrderedDict()
    for k, lst in groups.items():
        near = sum(per[n][0] for n in lst)
        dead = sum(per[n][1] for n in lst)
        sq = sum(per[n][2] for n in lst)
        cnt = sum(per[n][3] for n in lst)
        out[k] = dict(near=near / cnt, dead=dead / cnt, rms=(sq / cnt) ** 0.5, n=cnt)
    return out


def format_health(stats, epoch, step, where):
    return (f'HEALTH ep{epoch + 1} step{step} ({where}) near_zero(|w|<{NEAR_THR:g}) '
            + ' '.join(f'{k}={100 * v["near"]:.2f}%' for k, v in stats.items())
            + ' | rms ' + ' '.join(f'{k}={v["rms"]:.3e}' for k, v in stats.items()))


def collapsed_groups(stats):
    return [k for k in ('seq', 'unet_deep') if k in stats and stats[k]['near'] >= TH_COLLAPSED_NEAR]
