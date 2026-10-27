"""GPU0 짧은 스모크 (몇십 step, fp32): 원문 클래스 동등성 + 모델별 GPU 메모리·step 시간·손실 감소·파라미터 수 + 시간 추정.

정책: GPU0 이 쉬고 있을 때만(nvidia-smi 사용률 ≤ 10 % 가 3번 연속, 여유 메모리 ≥ 필요량 + 1 GB) 시험마다 시작한다.
조건이 안 되면 기다렸다 다시 본다(--max-wait-min). 학습 데이터는 캐시(data_cache.py)의 목록 0·1번 볼륨(32 슬라이스),
검증은 300번. 체크포인트·로그는 --out 아래(스크래치)에만 쓴다.

  CUDA_VISIBLE_DEVICES=0 python -B repro_prof/smoke_gpu.py --cache DIR --out DIR [--steps 40] [--only 이름 ...]
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from repro_prof import config_repro as CR  # noqa: E402
from repro_prof import data_cache as DC  # noqa: E402
from repro_prof import models_repro as M  # noqa: E402
from repro_prof import prof_paths as P  # noqa: E402
from repro_prof import train_repro as T  # noqa: E402

SKIP_FIRST = 3          # 처음 3 step 은 시간 통계에서 뺀다(cuDNN·할당기 초기 비용)
STEPS_PER_EPOCH_SLICES = 300 * 16   # 교수님 학습 300볼륨 × 16칸 = 4,800 슬라이스/epoch
PROF_LOADER_S_PER_VOL = 10.3        # 원 로더 실측(10-23, 볼륨 0·1·43·300 평균 약 10.3 s)
# RTX 3090 추정 비율(TITAN RTX 대비). GRU 재귀는 대역폭, U-Net·입력 행렬곱은 fp32 처리량에 비례한다고 본다.
R_BW = (936 / 672, 940 / 626)       # 공칭 대역폭비 1.39 ~ verify_design 보정 실효 대역폭비 1.50
R_FLOP = (1.5, 25 / 13)             # 보수 1.5 ~ verify_design 가정 fp32 처리량비 1.92 (TF32 끔)

TESTS = [
    # 이름, 프리셋, 설정 덮어쓰기, 필요 GPU 메모리(GB, 대략)
    ('unet_dfu_coil32_u3_bs1', 'unet_only', dict(N_UNET_DEPTH=3, BATCH_SIZE=1), 3),
    ('unet_dfu_coil32_u3_bs2', 'unet_only', dict(N_UNET_DEPTH=3, BATCH_SIZE=2), 3),
    ('unet_dfu_coil32_u5_bs1', 'unet_only', dict(N_UNET_DEPTH=5, BATCH_SIZE=1), 4),
    ('unet_dfu_coil32_u5_bs2', 'unet_only', dict(N_UNET_DEPTH=5, BATCH_SIZE=2), 5),
    ('unet_dfuplain_rss1_u5_bs2', 'unet_only', dict(REFINE='dfu_plain', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=2), 5),
    ('unet_u2015_rss1_u5_bs1', 'unet_only', dict(REFINE='unet2015', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=1), 4),
    ('unet_u2015_rss1_u5_bs2', 'unet_only', dict(REFINE='unet2015', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=2), 5),
    ('unet_u2015_coil32_u5_bs2', 'unet_only', dict(REFINE='unet2015', BASELINE_INPUT='coil32', N_UNET_DEPTH=5, BATCH_SIZE=2), 5),
    ('unet_fastmri_rss1_c64p4_bs1', 'unet_only', dict(REFINE='fastmri', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=1), 4),
    ('unet_fastmri_rss1_c64p4_bs2', 'unet_only', dict(REFINE='fastmri', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=2), 6),
    ('unet_fastmri_coil32_c64p4_bs2', 'unet_only', dict(REFINE='fastmri', BASELINE_INPUT='coil32', N_UNET_DEPTH=5, BATCH_SIZE=2), 6),
    ('unet_fastmri_rss1_c32p4_bs2', 'unet_only', dict(REFINE='fastmri', BASELINE_INPUT='rss1', N_UNET_DEPTH=5, BATCH_SIZE=2, FASTMRI_CHANS=32), 4),
    ('equiv_prof_classes_H12_u3', None, None, 9),
    ('oETER_H12_u3_bs1', 'oETER', dict(N_UNET_DEPTH=3, BATCH_SIZE=1), 17),
    ('oETER_H12_u3_bs2', 'oETER', dict(N_UNET_DEPTH=3, BATCH_SIZE=2), 18),
    ('dETER_H12_u3_bs1', 'dETER', dict(N_UNET_DEPTH=3, BATCH_SIZE=1), 17),
    ('dETER_H12_u3_bs2', 'dETER', dict(N_UNET_DEPTH=3, BATCH_SIZE=2), 18),
    ('dETER_H12_u5_bs1', 'dETER', dict(N_UNET_DEPTH=5, BATCH_SIZE=1), 18),
    ('dETER_H12_u5_bs2', 'dETER', dict(N_UNET_DEPTH=5, BATCH_SIZE=2), 19),
    ('oETER_H12_u5_bs2', 'oETER', dict(N_UNET_DEPTH=5, BATCH_SIZE=2), 19),
]
SAVE_TEST = 'dETER_H12_u3_bs2'      # 이 시험에서만 체크포인트 저장 시간을 잰다(파일은 바로 지운다)
EXPECTED_BY_TEST = {
    'unet_dfu_coil32_u3': 1_879_585, 'unet_dfu_coil32_u5': 31_048_481,
    'oETER_H12_u3': 851_332_121, 'oETER_H12_u5': 880_501_017,
    'dETER_H12_u3': 851_350_585, 'dETER_H12_u5': 880_519_481,
}


def gpu_state():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits',
                            '-i', '0'], capture_output=True, text=True, timeout=30)
        util, used, total = [int(x.strip()) for x in r.stdout.strip().split(',')]
        a = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        return dict(util=util, used_MiB=used, total_MiB=total, apps=a.replace('\n', '; '))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e))


def wait_for_gpu(need_gb, max_wait_s, log):
    t0 = time.time()
    ok = 0
    last = None
    while True:
        s = gpu_state()
        if 'error' in s:
            log(f'nvidia-smi 실패 {s}')
            time.sleep(30)
            continue
        free_gb = (s['total_MiB'] - s['used_MiB']) / 1024
        good = s['util'] <= 10 and free_gb >= need_gb + 1.0
        ok = ok + 1 if good else 0
        if ok >= 3:
            return True, s, time.time() - t0
        if time.time() - t0 > max_wait_s:
            return False, s, time.time() - t0
        if not good and (last is None or time.time() - last > 300):
            log(f'  GPU0 대기: util {s["util"]}% used {s["used_MiB"]} MiB (필요 {need_gb} GB + 1) apps [{s["apps"]}]')
            last = time.time()
        time.sleep(4 if good else 30)


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def equivalence_test(cache_dir, depth=3):
    """원문 클래스(231030·231022 에서 AST 로 잘라 실행) 대 사본: 같은 시드 초기값, 같은 입력의 순전파 출력."""
    P.prof_ssim_module()     # 학습과 같은 cuDNN 설정(deterministic=True)
    consts = dict(N_INPUT_VERTICAL=384, N_FREQ_ENCODING=384, N_RFCOIL=16, N_OUT_X=384, N_OUT_Y=384,
                  N_HIDDEN_LRNN_1=12, N_HIDDEN_LRNN_2=12, N_UNET_DEPTH=depth)
    data, data_img, _ = DC.read_volume(cache_dir, 0)
    x = torch.from_numpy(np.ascontiguousarray(data[4:6])).cuda()
    xi = torch.from_numpy(np.ascontiguousarray(data_img[4:6])).cuda()
    out = {}
    for variant, script, cls in (('d', P.PROF_DETER_SCRIPT, 'ETER_hybrid_GRU_DFU'), ('o', P.PROF_OETER_SCRIPT, 'oETER_GRU_DFU')):
        Orig = P.exec_prof_class(script, cls, consts)
        torch.manual_seed(1234)
        ma = Orig().cuda()
        torch.manual_seed(1234)
        mb = M.ETERNetRepro(variant=variant, h1=12, h2=12, depth=depth).cuda()
        sa, sb = ma.state_dict(), mb.state_dict()
        keys = list(sa) == list(sb)
        init = keys and all(torch.equal(sa[k], sb[k]) for k in sa)
        ya = ma(x, xi) if variant == 'd' else ma(x)
        yb = mb(x, xi) if variant == 'd' else mb(x)
        same = torch.equal(ya, yb)
        out[cls] = dict(keys_equal=keys, init_bit_identical=init, forward_bit_identical=same,
                        max_abs_diff=float((ya - yb).abs().max()), n_params=sum(p.numel() for p in ma.parameters()),
                        out_shape=list(ya.shape))
        del ma, mb, sa, sb, ya, yb
        free_cuda()
    return out


def projections(rec, bs, unet_part_s=None):
    st = [s[4] + (s[5] if len(s) > 5 else 0.0) for s in rec['steps']][SKIP_FIRST:]   # GPU step + 배치 모으기
    med = statistics.median(st)
    steps_per_epoch = STEPS_PER_EPOCH_SLICES / bs
    val_s = rec.get('val_s', 0.0) or 0.0           # 검증 1볼륨(16 슬라이스) 순전파
    save_s = rec.get('save_s_est', 0.0)
    epoch_titan_s = med * steps_per_epoch + val_s + save_s
    out = dict(step_s_median=med, step_s_mean=statistics.mean(st), step_s_p90=float(np.percentile(st, 90)),
               n_timed_steps=len(st), steps_per_epoch=steps_per_epoch,
               titan_h_per_epoch=epoch_titan_s / 3600, titan_h_50ep=50 * epoch_titan_s / 3600,
               prof_loader_extra_h_per_epoch=300 * PROF_LOADER_S_PER_VOL / 3600)
    if unet_part_s is not None:          # ETER: GRU 몫 = 전체 − 같은 입력 채널 U-Net 단독 step
        gru_s = max(med - unet_part_s, 0.0)
        lo = gru_s / R_BW[0] + unet_part_s / R_FLOP[0]
        hi = gru_s / R_BW[1] + unet_part_s / R_FLOP[1]
        out.update(gru_part_s=gru_s, unet_part_s=unet_part_s)
    else:
        lo, hi = med / R_FLOP[0], med / R_FLOP[1]
    out['rtx3090_step_s_range'] = (hi, lo)
    out['rtx3090_h_per_epoch_range'] = ((hi * steps_per_epoch + val_s / R_FLOP[1]) / 3600,
                                        (lo * steps_per_epoch + val_s / R_FLOP[0]) / 3600)
    out['rtx3090_h_50ep_range'] = tuple(50 * v for v in out['rtx3090_h_per_epoch_range'])
    return out


def refiner_step_time(in_ch, n_hidden, depth, bs, n=15):
    """ETER 의 U-Net 과 같은 입력 채널의 UNet_choh_skip 단독 fwd+bwd+Adam step 시간(초, 중앙값)."""
    unet = P.prof_unet_module().UNet_choh_skip(in_channels=in_ch, n_classes=1, depth=depth, wf=6, batch_norm=False,
                                               up_mode='upconv', n_hidden=n_hidden).cuda()
    opt = torch.optim.Adam(unet.parameters(), lr=1e-4)
    x = torch.randn(bs, in_ch, 384, 384, device='cuda')
    y = torch.rand(bs, 1, 384, 384, device='cuda')
    ts = []
    for i in range(n + SKIP_FIRST):
        torch.cuda.synchronize()
        t = time.perf_counter()
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(unet(x), y)
        loss.backward()
        opt.step()
        loss.item()
        ts.append(time.perf_counter() - t)
    del unet, opt, x, y
    free_cuda()
    return statistics.median(ts[SKIP_FIRST:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--max-wait-min', type=float, default=90)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / 'smoke_results.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    logf = open(out / 'smoke_runner.log', 'a')

    def log(s):
        line = f'[{time.strftime("%H:%M:%S")}] {s}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'start; GPU {gpu_state()}')
    wait_budget = a.max_wait_min * 60          # 기다린 시간의 합 상한(시험 실행 시간은 세지 않음)
    for name, preset, ov, need in TESTS:
        if a.only and name not in a.only:
            continue
        if name in results and results[name].get('status') == 'ok':
            log(f'{name}: 이미 있음, 건너뜀')
            continue
        fallback = False
        is_eter = preset in ('dETER', 'oETER')
        if is_eter:
            ok, st, waited = wait_for_gpu(need, 60.0, log)
            if not ok:
                # 다른 프로세스가 메모리를 쥐고 있으면 Adam foreach 임시 목록(파라미터 1벌 ≈ 3.4 GB)을 끈 판으로 잰다
                ok, st, w2 = wait_for_gpu(need - 3.5, max(30.0, wait_budget), log)
                waited += w2
                fallback = ok
        else:
            ok, st, waited = wait_for_gpu(need, max(30.0, wait_budget), log)
        wait_budget -= waited
        if not ok:
            results[name] = dict(status='not_run_gpu_busy', gpu=st, waited_s=waited)
            log(f'{name}: GPU0 조건 미충족(대기 {waited:.0f}s) → 건너뜀 {st}')
            res_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
            continue
        log(f'{name}: start (waited {waited:.0f}s, GPU {st})')
        rec = dict(status='error', gpu_before=st)
        try:
            if preset is None:
                rec.update(equivalence_test(a.cache))
                rec['status'] = 'ok'
            else:
                ovv = dict(DATA_SOURCE='cache', CACHE_DIR=a.cache, NUM_TRAIN_SET=1, NUM_VOL_PER_SET=2, VAL_IDX=300,
                           HEALTH_EVERY_STEPS=20, SAVE_EVERY_EPOCH=False, SAVE_RESUME=False, PREFETCH_THREADS=4,
                           RUN_NAME='smoke_' + name, LOG_ROOT=str(out / 'logs'), CKPT_ROOT=str(out / 'ckpt'))
                ovv.update(ov)
                if fallback:
                    ovv['ADAM_FOREACH'] = False
                    rec['adam_foreach_off_fallback'] = True
                cfg = CR.Config(preset, ovv)
                free_cuda()
                torch.cuda.reset_peak_memory_stats()
                s = T.run(cfg, max_steps=a.steps, smoke_save=(name == SAVE_TEST), smoke_val=True, quiet_steps=True)
                st_gpu_after = gpu_state()
                steps = s.pop('steps')
                rec.update({k: v for k, v in s.items() if k not in ('health_start', 'health_end')})
                rec['health_start'] = {k: round(v['near'], 6) for k, v in s['health_start'].items()}
                rec['health_end'] = {k: round(v['near'], 6) for k, v in s.get('health_end', {}).items()}
                rec['steps'] = steps
                rec['gpu_after'] = st_gpu_after
                L = [x[1] for x in steps]
                k = min(8, len(L) // 3)
                rec['loss_first_mean'] = float(np.mean(L[:k]))
                rec['loss_last_mean'] = float(np.mean(L[-k:]))
                rec['loss_ratio_last_first'] = rec['loss_last_mean'] / rec['loss_first_mean']
                rec['loss_decreased'] = rec['loss_last_mean'] < rec['loss_first_mean']
                key = name.rsplit('_bs', 1)[0]
                if key in EXPECTED_BY_TEST:
                    rec['params_expected'] = EXPECTED_BY_TEST[key]
                    rec['params_match'] = s['params']['total'] == EXPECTED_BY_TEST[key]
                unet_part = None
                free_cuda()
                if cfg['MODEL'] == 'eter':
                    m_in = (2 * cfg['N_HIDDEN_LRNN_2'] + (32 if cfg['ETER_VARIANT'] == 'd' else 0))
                    unet_part = refiner_step_time(m_in, m_in // 2, cfg['N_UNET_DEPTH'], cfg['BATCH_SIZE'])
                if 'save_s' in rec:
                    rec['save_s_est'] = rec['save_s']
                    for f in (out / 'ckpt' / cfg.run_name).glob('tensors_*.pt'):
                        f.unlink()
                rec['projection'] = projections(rec, cfg['BATCH_SIZE'], unet_part)
                rec['status'] = 'ok'
                pj = rec['projection']
                log(f"{name}: ok params {s['params']['total']:,} peak {s['peak_alloc_GB']:.2f}/{s['peak_reserved_GB']:.2f} GB "
                    f"step {pj['step_s_median']:.3f}s → TITAN {pj['titan_h_per_epoch']:.2f} h/ep, "
                    f"{pj['titan_h_50ep']:.1f} h/50ep; loss {rec['loss_first_mean']:.4f}→{rec['loss_last_mean']:.4f}")
        except Exception as e:  # noqa: BLE001
            rec['error'] = repr(e)
            rec['traceback'] = traceback.format_exc()
            log(f'{name}: ERROR {e!r}')
        finally:
            free_cuda()
        results[name] = rec
        res_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    log(f'end; GPU {gpu_state()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
