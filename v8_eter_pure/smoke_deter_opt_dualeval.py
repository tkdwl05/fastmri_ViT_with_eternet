"""d-ETER 스모크 사후 평가 — 두 SSIM 규약을 따로 (2026-10-15, smoke_deter_opt.py 설계 결정 16). 결과 폴더는 읽기만 한다.

각 조건의 {cond}/end_state.pt (step 4000 의 model.state_dict(); E·EC 는 seq_gate 추가 — 하네스 _run_inner 의 저장 코드)를
하네스와 같은 E256 슬라이스에서 두 규약으로 평가한다.
  OURS      = 하네스 최종 평가 그대로 — smoke_deter_opt.Job.eval_set 를 그대로 호출(같은 슬라이스·같은 배치 8·같은
              입력 정규화·같은 seq_transform·같은 metrics = eval_paired_v8_nodc.slice_metrics: 뇌 마스크 안 SSIM 지도 평균,
              data_range = 마스크 안 GT max−min; PSNR·nMSE·L1 도 마스크). {cond}/eval_per_slice.csv 를 재현해야 한다.
  PROF_TEST = 교수님 테스트 스크립트 호출 (scripts_legacy/myTEST_ViT_ETER_regular_251008.py:403-411):
              skimage.metrics.structural_similarity(img_truth, img_pred, data_range=max(img_truth.max(), img_pred.max())),
              기본값(균일 7×7, 표본 공분산, 가장자리 3 px 제외 평균), 384×384 전체, 마스크 없음, float64. 같은 곳의 nMSE(전체 영상)도.
  PROF_TEST 지지 영역 판(ssim_prof_crop, 기술) = 같은 호출을 GT ≠ 0 의 외접 사각형으로 자른 영상에 — 0 채움 테두리가 있는 볼륨에서
              교수님 영상(테두리 없음)의 '전체 영상'에 가장 가까운 판. 테두리가 없는 슬라이스는 ssim_prof 와 같다.
  zero-filled(하네스 zf = 코일 RSS)도 두 규약으로 같이 낸다(기준점).
  --tests (GRU 조건만): 하네스 최종 평가와 같은 full=True 경로(시퀀스 출력 치환 시험 zero·chanmean·adj·adjprev·cross)를 돌려
              각 치환 출력도 PROF_TEST 와 전체 영상 L1(384² 평균, 원 단위)로 잰다 — F·P 모델이 배경 쪽에 시퀀스 출력을 쓰는지(마스크 시험이
              놓치는 부분) 보려는 기술 통계. CPU 에서 GRU 조건당 ≈2~3 배 오래 걸린다.
PROF_TEST 는 metrics 호출에 넘어가는 바로 그 출력·GT 텐서로 계산한다: eval_set 실행 동안만 smoke_deter_opt.metrics 를 감싸
(OURS 값은 원래 함수가 낸 그대로 돌려줌) 텐서 사본을 받아 두고, 호출 순서(_eval_body 의 결정적 순서: 슬라이스마다 zf → 정상 →
[zero → chanmean], 그다음 8 개 묶음마다 adj → adjprev → cross)를 행의 OURS 값과 비트 대조해 확인한다.
정밀도: --device cuda = 하네스와 같은 경로(autocast fp16, gru_h fp32 — 같은 비트 기대), --device cpu = autocast 가 꺼져 fp32
(입력 xi16 의 fp16 반올림만 하네스와 같음) → OURS 와 CSV 의 슬라이스별 차 ≈ 1e-4 (AMP 대 fp32; E256 실측 max ref 1.2e-4·A 2.0e-4,
평균 차 < 1e-5). 재현 대조 max·p99·mean|Δ| 를 summary 에 기록하고 TOL_OURS 를 넘으면 flag 만 한다.
GPU 는 기본으로 쓰지 않는다 (cpu: CUDA_VISIBLE_DEVICES="" 필수). cuda 는 런처 잠금(/tmp/smoke_deter_opt.lock)이 잡혀 있으면 거부,
하네스 GPU 잠금(/tmp/smoke_deter_opt.gpu.lock)을 실행 내내 잡고(하네스가 동시에 뜨면 그쪽이 exit 2 로 거부), GPU0 가 비어 있을 때만.
--skip-done 은 같은 config_hash·같은 end_state.pt(크기·시각)·같은 부분 표본·(--tests 면) tests 포함일 때만 건너뛴다 — 조건을
다시 돌리면 낡은 dualeval 결과를 새 OURS 와 짝짓지 않게(loss_compare 도 같은 대조를 한다).
출력(기본 <out>/dualeval/, 조건 폴더는 건드리지 않음): {cond}_per_slice.csv · {cond}_summary.json · dualeval_index.json
사용:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 nice -n 19 python v8_eter_pure/smoke_deter_opt_dualeval.py --conds ref A refP AP \\
      [--tests] [--n 8] [--dest <dir>] [--skip-done]
"""

import os
import sys
import csv
import json
import time
import argparse
import datetime
import warnings
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True                         # 하네스 import 로 __pycache__ 를 만들지 않음
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
import torch.nn as nn                                  # noqa: E402
from skimage.metrics import structural_similarity as sk_ssim      # noqa: E402

import smoke_deter_opt as H                            # 하네스 (CONDS·Job·metrics·R·gru_precision 재사용, 읽기 전용)
import smoke_deter_opt_report as RP                    # eval_per_slice.csv 읽기 (_csv_rows)

R = H.R
OUT_DEFAULT = H.OUT_DEFAULT
DUALEVAL_VERSION = '2026-10-15b'                       # 출력 열이 바뀌면 올림 (--skip-done 이 이전 판 결과를 다시 계산)
LAUNCHER_LOCK = '/tmp/smoke_deter_opt.lock'            # run_smoke_deter_opt.sh 의 LOCK 기본값
# OURS 재현 허용 (슬라이스별 max|Δ SSIM|) — 넘으면 flag (중단 아님). cpu = fp32 대 하네스 AMP, 10-15 실측(E256 전체):
#   ref max 1.24e-4·p99 6.3e-5·mean|Δ| 1.3e-5 / A(GRU) max 1.98e-4·p99 1.3e-4·mean|Δ| 2.5e-5 (평균 Δ 는 둘 다 |·| < 1e-5).
#   (이 값은 실측을 본 뒤 정한 것 — 새 조건이 넘으면 flag 일 뿐 판정에 쓰지 않는다.) cuda = 하네스와 같은 경로라 거의 같은 비트를 기대.
TOL_OURS = dict(cpu=2e-4, cuda=1e-5)
TESTS = ('zero', 'chanmean', 'adj', 'adjprev', 'cross')
PROTOCOLS = OrderedDict(
    OURS='smoke_deter_opt.metrics -> eval_paired_v8_nodc.slice_metrics: skimage SSIM map (uniform 7, sample cov.) averaged '
         'inside the brain mask, data_range = in-mask GT max-min (per slice); PSNR/nMSE/L1 in mask',
    PROF_TEST='myTEST_ViT_ETER_regular_251008.py:403-411: skimage structural_similarity(img_truth, img_pred, '
              'data_range=max(img_truth.max(), img_pred.max())) on the full 384x384 image, defaults, no mask, float64; '
              'nMSE = sum((gt-pred)^2)/sum(gt^2) on the full image',
    PROF_TEST_crop='same call on the bounding box of GT != 0 (drops the zero-padded frame; descriptive)',
    L1_full='mean |pred - gt| over 384x384 (raw units; replacement tests only, --tests)')


def prof_test(gt, pred):
    """교수님 테스트 호출 그대로 (myTEST_ViT_ETER_regular_251008.py:403-411). gt·pred: (384, 384)."""
    img_truth = np.asarray(gt, dtype=np.float64)
    img_pred = np.asarray(pred, dtype=np.float64)
    val_maximum = np.amax((img_truth.max(), img_pred.max()))
    ssim_between = sk_ssim(img_truth, img_pred, data_range=val_maximum)
    numera = np.sum(np.sum(np.square(np.subtract(img_truth, img_pred))))
    denomi = np.sum(np.sum(np.square(img_truth)))
    return dict(ssim=float(ssim_between), nmse=float(numera / denomi), data_range=float(val_maximum))


def support_box(gt):
    """GT ≠ 0 의 외접 사각형 (r0, r1, c0, c1) — 없거나 7 px 미만이면 None."""
    nz = np.asarray(gt) != 0
    rows, cols = np.where(nz.any(1))[0], np.where(nz.any(0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    if r1 - r0 < 7 or c1 - c0 < 7:
        return None
    return r0, r1, c0, c1


def prof_test_crop(gt, pred):
    b = support_box(gt)
    if b is None:
        return None, None
    r0, r1, c0, c1 = b
    p = prof_test(np.asarray(gt)[r0:r1, c0:c1], np.asarray(pred)[r0:r1, c0:c1])
    return p['ssim'], float((r1 - r0) * (c1 - c0)) / float(np.asarray(gt).size)


def l1_full(gt, pred):
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(gt, np.float64))))


def load_summary(out, cond):
    """실행된 summary.json (completed 만) — 하네스 CONDS 에 같은 이름이 있으면 preset 이 같은지 대조."""
    sp = os.path.join(out, cond, 'summary.json')
    if not os.path.exists(sp):
        return None, None, f'{cond}/summary.json 없음'
    with open(sp) as f:
        s = json.load(f)
    if s.get('status') != 'completed':
        return None, s, f"{cond} status={s.get('status')}"
    P = s.get('preset')
    note = None
    if cond in H.CONDS and json.loads(H.jdumps(H.CONDS[cond])) != P:
        note = f'summary preset != 현재 하네스 CONDS[{cond}] (summary 값을 씀)'
    elif cond not in H.CONDS:
        note = f'현재 하네스 CONDS 에 {cond} 없음 (summary preset 을 씀)'
    return P, s, note


def end_state_info(out, cond):
    sd_path = os.path.join(out, cond, 'end_state.pt')
    if not os.path.exists(sd_path):
        return sd_path, None
    st = os.stat(sd_path)
    return sd_path, dict(path=os.path.relpath(sd_path, H.REPO), bytes=st.st_size,
                         mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'))


def done_ok(dest, tag, out, cond, n, want_tests):
    """--skip-done: 저장된 dualeval 이 지금 조건 폴더(config_hash·end_state)·부분 표본·tests 요구와 맞는가."""
    sp = os.path.join(dest, f'{tag}_summary.json')
    if not (os.path.exists(sp) and os.path.exists(os.path.join(dest, f'{tag}_per_slice.csv'))):
        return False, 'dualeval 없음'
    with open(sp) as f:
        old = json.load(f)
    _, s, _ = load_summary(out, cond)
    _, es = end_state_info(out, cond)
    why = []
    if old.get('dualeval_version') != DUALEVAL_VERSION:
        why.append(f"dualeval 판 {old.get('dualeval_version')} != {DUALEVAL_VERSION}")
    if not s or old.get('config_hash') != s.get('config_hash'):
        why.append('config_hash 다름')
    if es is None or (old.get('end_state') or {}).get('bytes') != es['bytes'] or (old.get('end_state') or {}).get('mtime') != es['mtime']:
        why.append('end_state.pt 다름')
    if old.get('subset') != (None if n is None else f'first {n} of E256'):
        why.append('부분 표본 다름')
    if want_tests and (old.get('preset') or {}).get('model') == 'gru' and not old.get('tests_on'):
        why.append('tests 없음')
    return not why, '; '.join(why)


def build_net(P, sd_path, dev):
    """end_state.pt → 하네스와 같은 모델 (R.build_model + gru_h fp32 래퍼, gate 는 nn.Parameter)."""
    kind = P['model']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sd = torch.load(sd_path, map_location='cpu', weights_only=True)
    gate_t = sd.pop('seq_gate', None)
    torch.manual_seed(0)                                          # 초기값은 곧 덮어씀 (결정성만)
    model = R.build_model(kind, torch.device('cpu'))
    keys_model = list(model.state_dict().keys())
    info = dict(n_tensors=len(sd), keys_equal=list(sd.keys()) == keys_model, has_seq_gate=gate_t is not None)
    model.load_state_dict(sd, strict=True)
    del sd
    if kind == 'gru':
        H.gru_precision.apply(model, ('gru_h',))                  # 하네스 build_model 과 같음 (state_dict 키 불변)
    model = model.to(dev).eval()
    gate = None
    if P.get('seq') == 'gate':
        assert gate_t is not None, 'gate 조건인데 end_state.pt 에 seq_gate 없음'
        gate = nn.Parameter(gate_t.to(dev), requires_grad=False)
        info['gate_absmean'] = float(gate_t.abs().mean())
    else:
        assert gate_t is None, 'gate 가 아닌 조건에 seq_gate 가 있음'
    info['n_params'] = sum(p.numel() for p in model.parameters())
    return model, gate, info


def stub_job(cond, P, dev, n):
    """하네스 Job 의 eval 경로만 쓰는 객체 (Job.__init__·startup_guards 는 거치지 않음 — invariants.json 등에 쓰지 않음)."""
    J = H.Job.__new__(H.Job)
    J.cond, J.P, J.dev = cond, P, dev
    J.has_seq = P['model'] == 'gru'
    J.has_gate = P.get('seq') == 'gate'
    J.eval_n, J.probe_n = 256, 32                                 # 하네스 본 실행 값 (E256·P32)
    H.Job.build_val(J)                                            # 같은 val 데이터셋·같은 E256 선택 코드 (E_nxt·E_prv·E_cross 포함)
    J.E_full = list(J.E)
    if n is not None:
        J.E = J.E[:n]                                             # 앞쪽 n 개 (배치 8 경계가 하네스와 같음)
        J.E_nxt, J.E_prv = J.E_nxt[:n], J.E_prv[:n]
        J.E_cross = [J.E[(k + n // 2) % n] for k in range(n)]     # 부분 표본 안에서 다시 (점검용 — cross 는 CSV 와 대조하지 않음)
    return J


def expected_order(J, rows_by_idx, tests_on):
    """_eval_body 의 metrics 호출 순서 (idx, 열) — 대조용."""
    ids = J.E
    seq = []
    for i in ids:
        seq += [(i, 'zf_ssim'), (i, 'ssim')]
        if tests_on:
            seq += [(i, 'ssim_zero'), (i, 'ssim_chanmean')]
    if tests_on:
        pos_of = {i: k for k, i in enumerate(ids)}
        for c0 in range(0, len(ids), 8):
            cs = ids[c0:c0 + 8]
            for t, plist in (('adj', [J.E_nxt[pos_of[i]] for i in cs]), ('adjprev', [J.E_prv[pos_of[i]] for i in cs]),
                             ('cross', [J.E_cross[pos_of[i]] for i in cs])):
                seq += [(cs[b], f'ssim_{t}') for b, j in enumerate(plist) if j is not None]
    return seq


def run_cond(out, dest, cond, dev, n, want_tests, workers_note):
    t0 = time.time()
    P, summ, note = load_summary(out, cond)
    if P is None:
        print(f'[dualeval] {cond}: 건너뜀 — {note}', flush=True)
        return dict(cond=cond, status='skipped', why=note)
    sd_path, es = end_state_info(out, cond)
    if es is None:
        return dict(cond=cond, status='skipped', why='end_state.pt 없음')
    J = stub_job(cond, P, dev, n)
    tests_on = bool(want_tests and J.has_seq)
    inv_p = os.path.join(out, 'invariants.json')
    e_ok = None
    if os.path.exists(inv_p):
        with open(inv_p) as f:
            e_ok = json.load(f).get('E256') == J.E_full
        assert e_ok, 'E256 이 invariants.json 과 다름'
    model, gate, minfo = build_net(P, sd_path, dev)
    J.model = J.net = model
    J.base = model if J.has_seq else None
    J.gate = gate
    cap = []
    orig_metrics = H.metrics

    def metrics_capture(o, ref, bm):                              # OURS 는 원래 함수 값 그대로 돌려주고, 같은 텐서로 PROF_TEST 등을 잰다
        m = orig_metrics(o, ref, bm)
        on, gn = o.detach().float().cpu()[0, 0].numpy(), ref.detach().float().cpu()[0, 0].numpy()
        crop, crop_frac = prof_test_crop(gn, on)
        cap.append(dict(ssim=m['ssim'], prof=prof_test(gn, on), crop=crop, crop_frac=crop_frac, l1f=l1_full(gn, on),
                        ref_sum=float(np.sum(gn, dtype=np.float64)), gt_max=float(gn.max()), pred_max=float(on.max()),
                        pred_min=float(on.min()), frame_frac=float(np.mean(gn == 0))))
        return m
    H.metrics = metrics_capture
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*CUDA is not available.*')      # cpu: autocast('cuda') → 꺼짐
            warnings.filterwarnings('ignore', message='.*device_type of \'cuda\'.*')
            res = H.Job.eval_set(J, full=tests_on, tag=f'dualeval_{dev.type}{"_tests" if tests_on else ""}')
    finally:
        H.metrics = orig_metrics
    rows = res['rows']
    rix = {r['idx']: r for r in rows}
    order = expected_order(J, rix, tests_on)
    assert len(cap) == len(order), (len(cap), len(order))
    got = OrderedDict()                                           # (idx, 열) → 받아 둔 텐서
    for (i, col), c in zip(order, cap):
        assert c['ssim'] == rix[i][col], f'호출 순서 대응 실패 ({i}, {col})'
        got[(i, col)] = c
    csv_rows = RP._csv_rows(os.path.join(out, cond, 'eval_per_slice.csv')) or {}
    outrows, d_ours, d_zf, d_tests = [], [], [], []
    for r in rows:
        i = r['idx']
        cz, cm = got[(i, 'zf_ssim')], got[(i, 'ssim')]
        assert cz['ref_sum'] == cm['ref_sum'], f'zf·정상 호출의 GT 가 다름 ({i})'
        pz, pm = cz['prof'], cm['prof']
        c = csv_rows.get(i)
        row = OrderedDict(idx=i, file=r['file'], slice=r['slice'],
                          ssim_ours=r['ssim'], psnr_ours=r['psnr'], nmse_ours=r['nmse'], l1_ours=r['l1'],
                          ssim_prof=pm['ssim'], nmse_prof=pm['nmse'], dr_prof=pm['data_range'],
                          ssim_prof_crop=cm['crop'], crop_frac=cm['crop_frac'], frame_frac=cm['frame_frac'],
                          l1_full=cm['l1f'], gt_max=cm['gt_max'], pred_max=cm['pred_max'], pred_min=cm['pred_min'],
                          zf_ssim_ours=r['zf_ssim'], zf_ssim_prof=pz['ssim'], zf_nmse_prof=pz['nmse'], zf_ssim_prof_crop=cz['crop'],
                          ssim_ours_csv=None if c is None else c.get('ssim'),
                          zf_ssim_ours_csv=None if c is None else c.get('zf_ssim'))
        if tests_on:
            for t in TESTS:
                ct = got.get((i, f'ssim_{t}'))
                row[f'ssim_ours_{t}'] = r.get(f'ssim_{t}')
                if ct is not None:
                    assert ct['ref_sum'] == cm['ref_sum'], f'치환 시험 {t} 의 GT 가 다름 ({i})'
                row[f'ssim_prof_{t}'] = None if ct is None else ct['prof']['ssim']
                row[f'l1full_{t}'] = None if ct is None else ct['l1f']
                if ct is not None and c is not None and c.get(f'ssim_{t}') is not None and not (t == 'cross' and n is not None):
                    d_tests.append(r[f'ssim_{t}'] - c[f'ssim_{t}'])
        if c is not None:
            d_ours.append(r['ssim'] - c['ssim'])
            d_zf.append(r['zf_ssim'] - c['zf_ssim'])
        outrows.append(row)
    cols = list(outrows[0].keys())
    os.makedirs(dest, exist_ok=True)
    tag = cond if n is None else f'{cond}_n{n}'
    pth = os.path.join(dest, f'{tag}_per_slice.csv')
    with open(pth + '.tmp', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in outrows:
            w.writerow(['' if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])
    os.replace(pth + '.tmp', pth)
    mean = lambda k: (float(np.mean([r[k] for r in outrows if r[k] is not None]))       # noqa: E731
                      if any(r[k] is not None for r in outrows) else None)
    ad = np.abs(np.asarray(d_ours, float))
    adt = np.abs(np.asarray(d_tests, float))
    rep = OrderedDict(n=len(d_ours), max_abs_ssim=float(ad.max()) if d_ours else None,
                      p99_abs_ssim=float(np.percentile(ad, 99)) if d_ours else None,
                      mean_abs_ssim=float(ad.mean()) if d_ours else None, n_over_1e4=int((ad > 1e-4).sum()),
                      mean_ssim=float(np.mean(d_ours)) if d_ours else None,
                      max_abs_zf_ssim=float(np.max(np.abs(d_zf))) if d_zf else None,
                      tests_n=len(d_tests), tests_max_abs_ssim=float(adt.max()) if d_tests else None,
                      tol=TOL_OURS[dev.type])
    rep['ok'] = bool(d_ours) and rep['max_abs_ssim'] <= rep['tol'] and (not d_tests or rep['tests_max_abs_ssim'] <= rep['tol'])
    tests_prof = None
    if tests_on:
        tests_prof = OrderedDict()
        for t in TESTS:
            ds = [r['ssim_prof'] - r[f'ssim_prof_{t}'] for r in outrows if r.get(f'ssim_prof_{t}') is not None]
            dl = [r[f'l1full_{t}'] - r['l1_full'] for r in outrows if r.get(f'l1full_{t}') is not None]
            tests_prof[t] = OrderedDict(n=len(ds), ssim_prof_drop_mean=float(np.mean(ds)) if ds else None,
                                        ssim_prof_n_pos=int(sum(x > 0 for x in ds)), ssim_prof_n_neg=int(sum(x < 0 for x in ds)),
                                        l1_full_rise_mean=float(np.mean(dl)) if dl else None)
    out_s = OrderedDict(
        cond=cond, dualeval_version=DUALEVAL_VERSION, time=datetime.datetime.now().isoformat(timespec='seconds'), device=dev.type,
        precision=('harness path (autocast fp16, gru_h fp32)' if dev.type == 'cuda'
                   else 'cpu fp32 (autocast(cuda) disabled; xi16 input rounding as in harness)'),
        n=len(outrows), subset=None if n is None else f'first {n} of E256', tests_on=tests_on, preset=P, preset_note=note,
        config_hash=summ.get('config_hash'), loss=summ.get('loss'), e256_matches_invariants=e_ok, model=minfo,
        end_state=es, protocols=PROTOCOLS,
        mean=OrderedDict(ssim_ours=mean('ssim_ours'), ssim_prof=mean('ssim_prof'), ssim_prof_crop=mean('ssim_prof_crop'),
                         nmse_ours=mean('nmse_ours'), nmse_prof=mean('nmse_prof'), psnr_ours=mean('psnr_ours'),
                         l1_full=mean('l1_full'), zf_ssim_ours=mean('zf_ssim_ours'), zf_ssim_prof=mean('zf_ssim_prof'),
                         zf_ssim_prof_crop=mean('zf_ssim_prof_crop'), frame_frac=mean('frame_frac')),
        frac_zf_beats_model=OrderedDict(ours=float(np.mean([r['zf_ssim_ours'] > r['ssim_ours'] for r in outrows])),
                                        prof=float(np.mean([r['zf_ssim_prof'] > r['ssim_prof'] for r in outrows]))),
        tests_prof=tests_prof, reproduce_eval_csv=rep, workers=workers_note, sec=time.time() - t0)
    with open(os.path.join(dest, f'{tag}_summary.json'), 'w') as f:
        json.dump(out_s, f, ensure_ascii=False, indent=1)
    print(f'[dualeval] {cond} n={len(outrows)} OURS {out_s["mean"]["ssim_ours"]:.4f} PROF_TEST {out_s["mean"]["ssim_prof"]:.4f} '
          f'(crop {out_s["mean"]["ssim_prof_crop"]:.4f}; zf {out_s["mean"]["zf_ssim_ours"]:.4f} / {out_s["mean"]["zf_ssim_prof"]:.4f})'
          f'{" tests" if tests_on else ""} | 재현 max|Δ| {rep["max_abs_ssim"]}'
          f'{" tests " + format(rep["tests_max_abs_ssim"], ".2e") if rep["tests_max_abs_ssim"] is not None else ""} '
          f'{"OK" if rep["ok"] else "FLAG"} ({out_s["sec"]:.0f}s)', flush=True)
    del model, J, cap, got
    return dict(cond=cond, status='done', summary=os.path.relpath(os.path.join(dest, f'{tag}_summary.json'), H.REPO),
                reproduce_ok=rep['ok'], max_abs_ssim=rep['max_abs_ssim'], tests_on=tests_on)


def lock_held(path):
    """비차단 flock 시험 — 다른 프로세스가 잡고 있으면 (True, 내용). 잡히지 않았으면 곧바로 풀고 (False, None)."""
    import fcntl
    if not os.path.exists(path):
        return False, None
    with open(path, 'a') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                holder = open(path).read().strip()
            except OSError:
                holder = '?'
            return True, holder
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return False, None


def main():
    ap = argparse.ArgumentParser(description='d-ETER 스모크 사후 평가: OURS · PROF_TEST 두 SSIM 규약')
    ap.add_argument('--out-dir', default=OUT_DEFAULT, help='하네스 결과 폴더 (읽기만)')
    ap.add_argument('--dest', default=None, help='출력 폴더 (기본 <out-dir>/dualeval)')
    ap.add_argument('--conds', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=None, help='E256 의 앞 n 슬라이스만 (점검용)')
    ap.add_argument('--tests', action='store_true', help='GRU 조건: 시퀀스 출력 치환 시험도 PROF_TEST·전체 영상 L1 로 (느림)')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '6')))
    ap.add_argument('--skip-done', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    dest = os.path.abspath(a.dest or os.path.join(out, 'dualeval'))
    gpu_lock = None
    if a.device == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            print('[refuse] cpu 평가는 CUDA_VISIBLE_DEVICES="" 로 (GPU 를 건드리지 않게)', flush=True)
            return 2
        torch.set_num_threads(a.threads)
        dev = torch.device('cpu')
    else:
        held, who = lock_held(LAUNCHER_LOCK)
        if held:
            print(f'[refuse] 스모크 런처가 실행 중 ({LAUNCHER_LOCK}: {who or "?"}) — cuda 평가 거부', flush=True)
            return 2
        gpu_lock = H.take_gpu_lock()                              # 하네스 GPU 잠금을 실행 내내 잡는다
        if gpu_lock is None:
            print(f'[refuse] 하네스 GPU job 이 실행 중 ({H.GPU_LOCK})', flush=True)
            return 2
        R.gpu0_precheck(force=False)                              # GPU0 사용 중이면 거부 (exit 2/4)
        dev = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = False             # 하네스 startup_guards 와 같음
        torch.backends.cudnn.allow_tf32 = False
    assert torch.backends.cudnn.deterministic is True             # u_choh_SSIM import 부작용 (하네스와 같음)
    idx = []
    for c in a.conds:
        tag = c if a.n is None else f'{c}_n{a.n}'
        if a.skip_done:
            ok, why = done_ok(dest, tag, out, c, a.n, a.tests)
            if ok:
                print(f'[dualeval] {c}: skip-done (config_hash·end_state·부분 표본 일치)', flush=True)
                idx.append(dict(cond=c, status='skip-done'))
                continue
            print(f'[dualeval] {c}: 다시 계산 — {why}', flush=True)
        idx.append(run_cond(out, dest, c, dev, a.n, a.tests, dict(loader_workers=8, torch_threads=torch.get_num_threads())))
    os.makedirs(dest, exist_ok=True)
    ip = os.path.join(dest, 'dualeval_index.json')
    prev = []
    if os.path.exists(ip):
        with open(ip) as f:
            prev = json.load(f).get('runs', [])
    with open(ip, 'w') as f:
        json.dump(dict(updated=datetime.datetime.now().isoformat(timespec='seconds'), source=out, protocols=PROTOCOLS,
                       runs=prev + idx), f, ensure_ascii=False, indent=1)
    bad = [r['cond'] for r in idx if r.get('status') == 'done' and not r.get('reproduce_ok')]
    if bad:
        print(f'[dualeval] FLAG OURS 재현 > {TOL_OURS[dev.type]:g}: {bad}', flush=True)
    del gpu_lock
    return 0


if __name__ == '__main__':
    sys.exit(main())
