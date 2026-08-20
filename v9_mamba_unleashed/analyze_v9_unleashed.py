#!/usr/bin/env python
"""
v9 unleashed 정량 분석 (로그 파싱, 재계산 0): v9 vs v8-SS2D vs v8-GRU matched-epoch + 궤적 곡선.

`v8_eter_pure/analyze_v8_nodc.py` 의 v9 확장 — per-epoch masked composite 는 이미 각 run 의
log.txt 에 있으므로 GPU 재계산이 필요 없다. v9 는 80ep(v8 은 50ep)라 ep2~50 은 3-way
matched-epoch, ep52~80 은 v9 단독 연장 구간으로 표를 나눈다. v8-SS2D best(0.9200) 수평
참고선 + v9 의 돌파 epoch(수직선)를 곡선에 표시한다.

정직 주석: v9 의 v8 초과는 80ep 연장 구간(ep70+)에서 나온다 — matched-ep50 시점 v9 는
v8@50 미달. ep당 시간은 v9 가 빠르지만(2.51 vs 2.78 h/ep) best 도달 wall-clock 은 더 길다.

입력: logs/PureETER_SS2D_V9_unleashed_R4_brain384/log.txt (+ v8 GRU/SS2D no-DC log.txt)
출력: results/eval/v9_unleashed/matched_epoch_table_v9.md + curves_v9_vs_v8.png
"""
import os
import re
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 참고선 (docs/summary_2026-06-11.md)
V7_BASELINE_COMP = 0.9127
HOURS_PER_EP = {'v9': 2.51, 'v8': 2.78}   # 실측 (스모크/런 로그)

LINE_RE = re.compile(
    r'Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)'
    r'(?:\s+val_composite=([\d.]+)\s+val_ssim_m=([\d.]+)'
    r'\s+val_psnr=([\d.]+)\s+val_nmse=([\d.]+)\s+val_l1=([\d.]+))?'
)


def parse_log(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m or m.group(3) is None:
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
    p = argparse.ArgumentParser(description='v9 unleashed vs v8 no-DC 로그기반 정량 분석')
    p.add_argument('--v9-log',   default=os.path.join(PROJECT_ROOT, 'logs/PureETER_SS2D_V9_unleashed_R4_brain384/log.txt'))
    p.add_argument('--ss2d-log', default=os.path.join(PROJECT_ROOT, 'logs/PureETER_SS2D_noDC_R4_brain384_v8/log.txt'))
    p.add_argument('--gru-log',  default=os.path.join(PROJECT_ROOT, 'logs/PureETER_GRU_noDC_R4_brain384_v8/log.txt'))
    p.add_argument('--out-dir',  default=os.path.join(PROJECT_ROOT, 'results/eval/v9_unleashed'))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    v9 = parse_log(args.v9_log)
    s8 = parse_log(args.ss2d_log)
    g8 = parse_log(args.gru_log)

    v9b_ep, v9b = best(v9)
    s8b_ep, s8b = best(s8)
    g8b_ep, g8b = best(g8)

    # v8-SS2D best composite 를 v9 가 처음 넘는(≥) epoch
    cross_ep = None
    if v9 and s8b:
        for ep in sorted(v9):
            if v9[ep]['composite'] >= s8b['composite']:
                cross_ep = ep
                break

    L = []
    L.append('# v9 unleashed vs v8 no-DC — 로그기반 비교 (masked composite, 384/R4, 동일 val)')
    L.append('')
    L.append(f'- v9 val epochs: {min(v9) if v9 else "—"}..{max(v9) if v9 else "—"} ({len(v9)} pts, 80ep 완주)')
    L.append(f'- v8-SS2D/GRU val epochs: ..{max(s8) if s8 else "—"} / ..{max(g8) if g8 else "—"} (50ep)')
    L.append('')
    L.append('## best 요약')
    if v9b:
        L.append(f'- **v9 best** ep{v9b_ep}: composite **{fmt(v9b["composite"])}** / ssim_m {fmt(v9b["ssim_m"])} '
                 f'/ psnr {fmt(v9b["psnr"],2)} / nmse {fmt(v9b["nmse"])} / l1 {fmt(v9b["l1"],2)}')
    if s8b:
        L.append(f'- v8-SS2D best ep{s8b_ep}: composite {fmt(s8b["composite"])} / ssim_m {fmt(s8b["ssim_m"])} / psnr {fmt(s8b["psnr"],2)}')
    if g8b:
        L.append(f'- v8-GRU  best ep{g8b_ep}: composite {fmt(g8b["composite"])} / ssim_m {fmt(g8b["ssim_m"])} / psnr {fmt(g8b["psnr"],2)}')
    if v9b and s8b:
        L.append(f'- Δ(v9 − v8-SS2D) = **{v9b["composite"]-s8b["composite"]:+.4f}** composite '
                 f'/ {v9b["ssim_m"]-s8b["ssim_m"]:+.4f} ssim_m — 근소 우위')
    if cross_ep:
        L.append(f'- v9 가 v8-SS2D best({fmt(s8b["composite"])})에 도달한 최초 epoch: **ep{cross_ep}**')
    L.append('')
    if v9b and s8b:
        v9_h = v9b_ep * HOURS_PER_EP['v9']
        s8_h = s8b_ep * HOURS_PER_EP['v8']
        L.append(f'## wall-clock 정직 주석')
        L.append(f'- matched-ep50 시점 v9 composite {fmt(v9.get(50, {}).get("composite"))} — v8-SS2D@ep48~50 미달.')
        L.append(f'  v9 의 우위는 **80ep 연장 구간(ep{cross_ep}+)** 에서 나온다.')
        L.append(f'- best 도달 wall-clock: v9 ≈ {v9_h:.0f}h (ep{v9b_ep}×2.51) vs v8-SS2D ≈ {s8_h:.0f}h (ep{s8b_ep}×2.78) '
                 f'— ep당은 v9 가 빠르지만 총 시간은 더 소요.')
        L.append('')

    L.append('## matched-epoch 3-way (ep2~50, Δ = v9 − v8-SS2D)')
    L.append('')
    L.append('| epoch | v8-GRU comp | v8-SS2D comp | v9 comp | Δ comp | v8-SS2D ssim | v9 ssim |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|')
    common = sorted(set(v9) & set(s8) & set(g8))
    for ep in common:
        v, s, g = v9[ep], s8[ep], g8[ep]
        L.append(f'| {ep} | {fmt(g["composite"])} | {fmt(s["composite"])} | {fmt(v["composite"])} '
                 f'| {v["composite"]-s["composite"]:+.4f} | {fmt(s["ssim_m"])} | {fmt(v["ssim_m"])} |')
    L.append('')

    L.append('## v9 연장 구간 (ep52~80, v8 은 50ep 종료)')
    L.append('')
    L.append('| epoch | v9 comp | v9 ssim_m | v9 psnr | vs v8-SS2D best |')
    L.append('|---:|---:|---:|---:|---:|')
    for ep in sorted(e for e in v9 if e > (max(common) if common else 0)):
        v = v9[ep]
        mark = '**돌파**' if (s8b and v['composite'] >= s8b['composite']) else ''
        L.append(f'| {ep} | {fmt(v["composite"])} | {fmt(v["ssim_m"])} | {fmt(v["psnr"],2)} '
                 f'| {v["composite"]-s8b["composite"]:+.4f} {mark} |' if s8b else
                 f'| {ep} | {fmt(v["composite"])} | {fmt(v["ssim_m"])} | {fmt(v["psnr"],2)} | — |')
    L.append('')

    table_md = '\n'.join(L)
    out_md = os.path.join(args.out_dir, 'matched_epoch_table_v9.md')
    with open(out_md, 'w') as f:
        f.write(table_md + '\n')
    print(table_md)
    print(f'\n[저장] {out_md}')

    # ── 궤적 곡선 (3-way + v8-SS2D best 참고선 + 돌파 수직선) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [('composite', 'masked composite'), ('ssim_m', 'masked SSIM'), ('psnr', 'masked PSNR (dB)')]
    for ax, (key, ttl) in zip(axes, specs):
        if g8:
            e = sorted(g8); ax.plot(e, [g8[x][key] for x in e], '-o', ms=3, color='tab:red',   label='v8 GRU no-DC')
        if s8:
            e = sorted(s8); ax.plot(e, [s8[x][key] for x in e], '-s', ms=3, color='tab:blue',  label='v8 SS2D no-DC')
        if v9:
            e = sorted(v9); ax.plot(e, [v9[x][key] for x in e], '-^', ms=3, color='tab:green', label='v9 unleashed')
        if s8b:
            ax.axhline(s8b[key], ls='--', c='tab:blue', lw=1, alpha=0.7,
                       label=f'v8 SS2D best ({s8b[key]:.4f})' if key != 'psnr' else f'v8 SS2D best ({s8b[key]:.2f})')
        if key == 'composite':
            ax.axhline(V7_BASELINE_COMP, ls=':', c='gray', lw=1, label=f'v7_titan ({V7_BASELINE_COMP})')
        if cross_ep and key == 'composite':
            # (곡선 라벨은 ASCII — 컨테이너 matplotlib 에 한글 폰트 없음)
            ax.axvline(cross_ep, ls=':', c='tab:green', lw=1, alpha=0.7, label=f'v9 reaches v8 best (ep{cross_ep})')
        ax.set_xlabel('epoch'); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('v9 unleashed vs v8 no-DC trajectory (masked, 384/R4)', fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'curves_v9_vs_v8.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[저장] {out_png}')


if __name__ == '__main__':
    main()
