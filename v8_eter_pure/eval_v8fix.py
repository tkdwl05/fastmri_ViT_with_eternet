"""
v8fix 평가 (2026-10-08) — 레시피 DL(main_train_pure_v8fix.py)로 학습한 체크포인트를 기존 per-slice 프로토콜로 평가한다.

  - 추론·지표·요약 = eval_unet_only_v8 의 run_inference / build_summary 를 그대로 쓴다(전체 val 7,334 슬라이스, 같은 순서·
    R=4 고정 마스크·batch 1·AMP, 지표 = eval_paired_v8_nodc.slice_metrics, 볼륨 단위 표·--ref 대응 비교·Wilcoxon·부트스트랩).
    다른 점은 모델 입력뿐: 학습과 같게 샘플별 정규화(영상 /s, k-space ×100/s — smoke_recipe_fix.per_sample_scale/
    normalize_inputs)를 하고 출력×s 로 원 단위에 복원한다(Normalized 래퍼). 기존 런 CSV 와 같은 단위·같은 공식이다.
  - 기능 시험 = smoke_recipe_fix.run_validation(normalize=True) 를 val 에 고르게 퍼진 --func-slices(기본 512) 슬라이스에서:
    shuffle_cross(다른 볼륨 슬라이스의 시퀀스 출력으로 교체)·shuffle_adj(같은 볼륨 이웃)·zero(0 대체, 분포 밖 — 참고)·
    deep4/deep34(U-Net 깊은 층 출력 0 대체). 사전 기준(스모크와 같음): 평균 > 0 & 평균 > 2·SE & 부호 검정 p < 0.01 → 'used'.
출력 (--out-dir, 기본 results/eval/v8fix_ep5/): per_slice_<seq>.csv, summary_<seq>.md, functional_<seq>.json

실행 예 (GPU0 에 다른 런이 없을 때):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix.py --seq gru
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix.py --seq unet \\
      --ref "ETER-net (bi-GRU)=results/eval/v8fix_ep5/per_slice_gru.csv:" --ref "SS2D=results/eval/v8fix_ep5/per_slice_ss2d.csv:"
  python v8_eter_pure/eval_v8fix.py --seq unet --summary-only --ref ...   # 추론 없이 요약만 다시
"""

import os
import re
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)

import eval_unet_only_v8 as E          # 추론 루프·CSV·요약 (import 부작용 = sys.path·PYTORCH_CUDA_ALLOC_CONF 뿐)
import smoke_recipe_fix as R           # 레시피 정규화·모델 생성·기능 시험 (학습과 같은 단일 출처)

SPECS = {
    'unet': dict(name='U-Net only', title='U-Net 단독 대조 모델 (시퀀스 모듈 제거) — 레시피 DL'),
    'gru':  dict(name='ETER-net (bi-GRU)', title='ETER-net 기준 모델 (교수님 원본 bi-GRU 시퀀스 모듈) — 레시피 DL'),
    'ss2d': dict(name='SS2D', title='SS2D 비교 모델 (통제판 시퀀스 모듈) — 레시피 DL'),
}
FUNC_TESTS = [('shuffle_cross', '다른 볼륨 슬라이스의 시퀀스 출력으로 교체'),
              ('shuffle_adj', '같은 볼륨 이웃 슬라이스의 시퀀스 출력으로 교체'),
              ('zero', '시퀀스 출력 0 대체 (학습 분포 밖 — 참고만)'),
              ('deep4', 'U-Net down_path.4 출력 0 대체'),
              ('deep34', 'U-Net down_path.3 출력 0 대체')]


class Normalized(nn.Module):
    """학습과 같은 입력 정규화 + 출력 원 단위 복원 (state_dict 는 base 그대로 로드)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x_img, x_ksp, mask, sens):
        s = R.per_sample_scale(x_img)
        k, i = R.normalize_inputs(x_ksp, x_img, s)
        return self.base(i, k, mask, sens).float() * s


def ckpt_epoch_info(ckpt_path):
    """v8fix log.txt ('Epoch N/M ... val_ssim_m=') 에서 이 체크포인트의 에폭과 학습 중 검증 SSIM 을 찾는다."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'log.txt')
    if not os.path.exists(log_path):
        return None
    pat = re.compile(r'^Epoch (\d+)/(\d+)\s.*?val_ssim_m=([\d.]+)')
    rows, n_total = {}, None
    with open(log_path) as f:
        for line in f:
            mt = pat.match(line)
            if mt:
                rows[int(mt.group(1))] = float(mt.group(3))
                n_total = int(mt.group(2))
    if not rows:
        return None
    base = os.path.basename(ckpt_path)
    mt = re.search(r'_epoch_(\d+)\.pt$', base)
    if mt:
        ep = int(mt.group(1))
        return dict(epoch=ep, n_total=n_total, val_ssim_m=rows[ep], how='파일 이름') if ep in rows else None
    if base.endswith('_best.pt'):
        ep = max(sorted(rows), key=lambda e: rows[e])          # 최댓값의 최초 에폭 (트레이너 규칙: 엄격히 클 때만 갱신)
        return dict(epoch=ep, n_total=n_total, val_ssim_m=rows[ep], how='log.txt 검증 SSIM 최댓값으로 추정')
    return None


def func_section(fres):
    if not fres:
        return ''
    L = [f'## 3. 기능 시험 (val 에 고르게 퍼진 {fres["n"]} 슬라이스 — `smoke_recipe_fix.run_validation`, normalize=True)', '',
         '- SSIM 하락 = 정상 출력 SSIM − 시험 출력 SSIM (슬라이스별, 원 단위). 판정 기준(스모크와 같음, 실행 전 고정): '
         f'{fres["criteria"]}.',
         f'- 이 슬라이스들의 정상 SSIM 평균 {fres["normal"]["ssim"]:.4f} (전체 val 표와 표본이 달라 값이 다를 수 있음).', '',
         '| 시험 | 내용 | SSIM 하락 평균 ± SE | 하락 슬라이스 | 부호 검정 p | 판정 |', '|---|---|---:|---:|---:|---|']
    for t, desc in FUNC_TESTS:
        st = fres['tests'].get(t)
        if st is None:
            continue
        se = f'{st["se"]:.5f}' if st['se'] is not None else '-'
        p = f'{st["p_sign"]:.2g}' if st['p_sign'] is not None else '-'
        L.append(f'| {t} | {desc} | {st["mean"]:+.5f} ± {se} | {st["n_pos"]}/{st["n"]} | {p} | {R.used_verdict(st)} |')
    for k, desc in (('seq_out_pair_rel_diff_mean', '시퀀스 출력의 슬라이스 간 상대 차이 (shuffle_cross 짝; 0 이면 입력 무관)'),
                    ('seq_out_spatial_std_mean', '시퀀스 출력 채널별 공간 std 평균 (0 이면 공간 상수)')):
        if fres.get(k) is not None:
            L.append(f'\n- {desc}: {fres[k]:.4g}')
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser(description='v8fix(레시피 DL) per-slice 평가 + 기능 시험')
    p.add_argument('--seq', choices=sorted(SPECS), required=True)
    p.add_argument('--ckpt', default=None, help='기본 logs/PureETER_{SEQ}_noDC_R4_brain384_v8fix_s1/pure_{seq}_epoch_5.pt')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/v8fix_ep5')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 전체 val (양수 = 배관 점검용)')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--func-slices', type=int, default=512, help='기능 시험 슬라이스 수 (0 = 생략)')
    p.add_argument('--ref', action='append', default=[], metavar='NAME=PATH:COLUMN',
                   help='대응 비교 기준 per-slice CSV (eval_unet_only_v8 와 같은 형식; v8fix CSV 는 열 접두 없음 → "NAME=PATH:")')
    p.add_argument('--force', action='store_true', help='기존 CSV·기능 시험 결과 덮어쓰기')
    p.add_argument('--summary-only', action='store_true', help='추론 없이 기존 CSV·functional json 으로 요약만 다시')
    p.add_argument('--skip-precheck', action='store_true', help='GPU0 사용 중 거부 검사 생략')
    args = p.parse_args()

    seq = args.seq
    spec = SPECS[seq]
    E.SEQ, E.MODEL_NAME, E.TITLE = seq, spec['name'], spec['title']
    E.CSV_NAME, E.SUMMARY_NAME = f'per_slice_{seq}.csv', f'summary_{seq}.md'
    args.ckpt = args.ckpt or f'logs/PureETER_{seq.upper()}_noDC_R4_brain384_v8fix_s1/pure_{seq}_epoch_5.pt'
    refs = [E.parse_ref(s) for s in args.ref]
    for name, path, _ in refs:
        if not os.path.exists(path):
            raise SystemExit(f'[ERROR] --ref {name}: 파일 없음 {path}')
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, E.CSV_NAME)
    sum_path = os.path.join(args.out_dir, E.SUMMARY_NAME)
    func_path = os.path.join(args.out_dir, f'functional_{seq}.json')

    if args.summary_only:
        if not os.path.exists(csv_path):
            raise SystemExit(f'[ERROR] --summary-only: CSV 없음 {csv_path}')
        n_val_total = len(E.build_val_dataset(args.data_path))
    else:
        for path in (csv_path, func_path):
            if os.path.exists(path) and not args.force:
                raise SystemExit(f'[ERROR] {path} 가 이미 있음 — 덮어쓰려면 --force (요약만은 --summary-only)')
        if not os.path.exists(args.ckpt):
            raise SystemExit(f'[ERROR] 체크포인트 없음: {args.ckpt}')
        if not args.skip_precheck:
            R.gpu0_precheck(force=False)
        device = torch.device('cuda')
        base = R.build_model(seq, device)
        base.load_state_dict(torch.load(args.ckpt, map_location=device))
        base.eval()
        model = Normalized(base).eval()
        print(f'[{seq}] ckpt {args.ckpt} (params {sum(q.numel() for q in base.parameters()) / 1e6:.1f}M)')

        ds = E.build_val_dataset(args.data_path)
        n_val_total = len(ds)
        total = min(n_val_total, args.max_samples) if args.max_samples > 0 else n_val_total
        t0 = time.time()
        rows = E.run_inference(model, ds, total, device, args.num_workers)
        print(f'추론+지표 {(time.time() - t0) / 60:.1f} 분 ({total} 슬라이스)')
        E.write_csv(csv_path, rows)

        if args.func_slices > 0:
            idx = R.even_indices(n_val_total, args.func_slices)
            t0 = time.time()
            res = R.run_validation(base, seq, True, ds, idx, device, args.num_workers, True, True)
            fres = dict(seq=seq, ckpt=args.ckpt, n=res['n'], indices=res['indices'], normal=res['normal'],
                        tests=res['tests'], criteria=f'mean>0 & mean>2SE & sign-test p<{R.TH_USED_P}',
                        verdict={t: R.used_verdict(res['tests'].get(t)) for t, _ in FUNC_TESTS if t in res['tests']},
                        per_slice_ssim=res['per_slice_ssim'], seconds=round(time.time() - t0, 1))
            for k in ('seq_out_spatial_std_mean', 'seq_out_rms_mean', 'seq_out_pair_rel_diff_mean', 'decomp_equiv_maxabs'):
                if k in res:
                    fres[k] = res[k]
            with open(func_path + '.tmp', 'w') as f:
                json.dump(fres, f, ensure_ascii=False, indent=1)
            os.replace(func_path + '.tmp', func_path)
            print(f'기능 시험 {fres["seconds"] / 60:.1f} 분 — 판정 {fres["verdict"]}')

    model_rows = E.read_csv_rows(csv_path)
    msg = E.build_summary(model_rows, refs, args.ckpt, n_val_total, ckpt_epoch_info(args.ckpt))
    msg = msg.replace('# v8 Pure ETER-Net —', '# v8fix (레시피 DL) —', 1)
    msg = msg.replace('- 프로토콜: `eval_paired_v8_nodc.py` 와 동일',
                      '- 프로토콜: `eval_paired_v8_nodc.py` 와 동일하되 입력을 학습과 같이 샘플별 정규화하고 출력×s 로 원 단위 복원', 1)
    fres = None
    if os.path.exists(func_path):
        with open(func_path) as f:
            fres = json.load(f)
    msg += '\n' + func_section(fres)
    with open(sum_path + '.tmp', 'w') as f:
        f.write(msg)
    os.replace(sum_path + '.tmp', sum_path)
    print('\n' + msg)
    print(f'CSV : {csv_path}\n요약: {sum_path}')


if __name__ == '__main__':
    main()
