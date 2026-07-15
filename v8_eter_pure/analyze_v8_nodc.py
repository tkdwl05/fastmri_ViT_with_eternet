#!/usr/bin/env python
"""
v8 no-DC 정량 분석 (로그 파싱, 재계산 0): GRU vs SS2D matched-epoch 비교 + 궤적 곡선.

per-epoch masked composite 는 이미 각 arm 의 log.txt 에 있으므로 GPU 재계산이 필요 없다.
학습 중에도 안전(부분 로그 OK) — 완주 후 다시 실행하면 최종본이 된다.

입력: logs/PureETER_{GRU,SS2D}_noDC_R4_brain384_v8/log.txt
      (라인 형식: 'Epoch N/50  train_loss=..  val_composite=..  val_ssim_m=..  val_psnr=..  val_nmse=..  val_l1=..')
출력: results/eval/v8_nodc/matched_epoch_table.md + curves_composite_ssim_psnr.png
"""
import os
import re
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# v7_titan baseline (384, masked composite) — 참고선 (docs/summary_2026-06-11.md)
V7_BASELINE = {
    'ETER': dict(composite=0.9127, ssim_m=0.9084, psnr=34.59),
    'SS2D': dict(composite=0.9127, ssim_m=0.9083, psnr=34.60),
}

# val 이 있는 epoch 라인만 매칭 (train-only epoch 는 val_* 그룹이 None)
LINE_RE = re.compile(
    r'Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)'
    r'(?:\s+val_composite=([\d.]+)\s+val_ssim_m=([\d.]+)'
    r'\s+val_psnr=([\d.]+)\s+val_nmse=([\d.]+)\s+val_l1=([\d.]+))?'
)


def parse_log(path):
    """→ {epoch: {composite, ssim_m, psnr, nmse, l1}} (val 있는 epoch 만)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m or m.group(3) is None:   # 매칭 실패 or train-only(val 없음)
                continue
            out[int(m.group(1))] = dict(
                composite=float(m.group(3)), ssim_m=float(m.group(4)),
                psnr=float(m.group(5)), nmse=float(m.group(6)), l1=float(m.group(7)),
            )
    return out


def best(traj):
    if not traj:
        return None, None
    ep = max(traj, key=lambda e: traj[e]['composite'])
    return ep, traj[ep]


def fmt(x, nd=4):
    return f'{x:.{nd}f}' if x is not None else '—'


def main():
    p = argparse.ArgumentParser(description='v8 no-DC GRU vs SS2D 로그기반 정량 분석')
    p.add_argument('--gru-log',  default=os.path.join(PROJECT_ROOT, 'logs/PureETER_GRU_noDC_R4_brain384_v8/log.txt'))
    p.add_argument('--ss2d-log', default=os.path.join(PROJECT_ROOT, 'logs/PureETER_SS2D_noDC_R4_brain384_v8/log.txt'))
    p.add_argument('--out-dir',  default=os.path.join(PROJECT_ROOT, 'results/eval/v8_nodc'))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gru = parse_log(args.gru_log)
    ss2d = parse_log(args.ss2d_log)
    common = sorted(set(gru) & set(ss2d))

    L = []
    L.append('# v8 no-DC — GRU vs SS2D (masked composite, 384/R4)')
    L.append('')
    L.append(f'- GRU  val epochs: {min(gru) if gru else "—"}..{max(gru) if gru else "—"}  ({len(gru)} pts)')
    L.append(f'- SS2D val epochs: {min(ss2d) if ss2d else "—"}..{max(ss2d) if ss2d else "—"}  ({len(ss2d)} pts)')
    if len(gru) != 50 // 2 or len(ss2d) != 50 // 2:
        L.append('- ⚠ **부분 로그(학습 진행 중)** — 완주(ep50) 후 재실행 시 최종본.')
    L.append('')

    gb_ep, gb = best(gru)
    sb_ep, sb = best(ss2d)
    L.append('## best (지금까지) / v7_titan 대비')
    if gb:
        L.append(f'- **GRU best** ep{gb_ep}: composite {fmt(gb["composite"])} / ssim_m {fmt(gb["ssim_m"])} / psnr {fmt(gb["psnr"],2)}')
    if sb:
        L.append(f'- **SS2D best** ep{sb_ep}: composite {fmt(sb["composite"])} / ssim_m {fmt(sb["ssim_m"])} / psnr {fmt(sb["psnr"],2)}')
    L.append(f'- v7_titan(384): ETER {V7_BASELINE["ETER"]["composite"]}/{V7_BASELINE["ETER"]["ssim_m"]}, '
             f'SS2D {V7_BASELINE["SS2D"]["composite"]}/{V7_BASELINE["SS2D"]["ssim_m"]}')
    L.append('')

    L.append('## matched-epoch (Δ = SS2D − GRU)')
    L.append('')
    L.append('| epoch | GRU ssim_m | SS2D ssim_m | Δ ssim_m | GRU comp | SS2D comp | Δ comp | GRU psnr | SS2D psnr |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for ep in common:
        g, s = gru[ep], ss2d[ep]
        L.append(f'| {ep} | {fmt(g["ssim_m"])} | {fmt(s["ssim_m"])} | {s["ssim_m"]-g["ssim_m"]:+.4f} '
                 f'| {fmt(g["composite"])} | {fmt(s["composite"])} | {s["composite"]-g["composite"]:+.4f} '
                 f'| {fmt(g["psnr"],2)} | {fmt(s["psnr"],2)} |')
    if not common:
        L.append('| (공통 val epoch 없음 — SS2D 아직 재개 전) | | | | | | | | |')
    L.append('')

    table_md = '\n'.join(L)
    out_md = os.path.join(args.out_dir, 'matched_epoch_table.md')
    with open(out_md, 'w') as f:
        f.write(table_md + '\n')
    print(table_md)
    print(f'\n[저장] {out_md}')

    # ── 궤적 곡선 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [('composite', 'masked composite', 0.9127, 'v7_titan (0.9127)'),
             ('ssim_m',    'masked SSIM',      0.9084, 'v7_titan ETER (0.9084)'),
             ('psnr',      'masked PSNR (dB)',  34.59, 'v7_titan (34.59)')]
    for ax, (key, ttl, base, blab) in zip(axes, specs):
        if gru:
            e = sorted(gru);  ax.plot(e, [gru[x][key] for x in e],  '-o', ms=3, color='tab:red',  label='GRU no-DC')
        if ss2d:
            e = sorted(ss2d); ax.plot(e, [ss2d[x][key] for x in e], '-s', ms=3, color='tab:blue', label='SS2D no-DC')
        ax.axhline(base, ls='--', c='gray', lw=1, label=blab)
        ax.set_xlabel('epoch'); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('v8 no-DC trajectory: GRU vs SS2D (masked, 384/R4)', fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'curves_composite_ssim_psnr.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[저장] {out_png}')


if __name__ == '__main__':
    main()
