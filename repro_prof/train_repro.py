"""교수님 순수 ETER-net 학습 루프 재현 트레이너 (231030 d-ETER / 231022 o-ETER) + U-Net 단독 기준선.

교수님 루프에서 그대로 옮긴 것 (231030:164-305)
- 묶음 루프: epoch 마다 n_set = 0..NUM_TRAIN_SET-1 순서로 목록 [5n, 5n+5) 볼륨 묶음을 만들고(:205-214),
  묶음 안에서만 DataLoader(batch_size=BATCH_SIZE, shuffle=True) 로 섞는다(:212).
- step: zero_grad → forward → loss = criterion(out, ref) + λ·(1 − mySSIM(out, ref)) → backward → step (:223-239).
  fp32, AMP·clip·스케줄 없음. 매 step 교수님과 같은 형식의 줄을 출력한다(:247-248).
- training_loss[epoch, n_set·total_step + i_batch] = loss_pixel (:245, 폭 4,800 :202).
- epoch 끝: 'loss_ssim improved' 출력 검사(:263-266, 원래처럼 저장하지 않음), 검증 볼륨을 no_grad 로 돌고
  마지막 배치 criterion 만 기록(:268-277), 매 epoch 모델 저장(:281), 끝에 tensors_entire.pt·loss 배열(:286-293).
- 원 코드처럼 model.eval() 은 부르지 않는다(DFU 에는 BN·dropout 이 없다).

우리 추가(config 에 'ours' 로 표시, 계산식은 바꾸지 않음)
- 캐시 데이터(data_cache.py, 원 로더와 비트 동일)와 다음 묶음 미리 읽기. DATA_SOURCE='prof' 면 교수님 클래스를 그대로 쓴다.
- SEED 와 묶음별 전용 생성기(DATA_ORDER='independent'), epoch 단위 재개(resume.pt), 가중치 소멸 감시(기록만),
  검증 볼륨 전체 평균 기록, 체크포인트 위치(HDD)·여유 공간 확인, step 시간·GPU 메모리 기록.

사용 (저장소 루트에서)
  python -B repro_prof/train_repro.py --preset dETER --dry-run            # 설정·파라미터 수·디스크·데이터 점검(GPU 안 씀)
  CUDA_VISIBLE_DEVICES=0 python -B repro_prof/train_repro.py --preset dETER [--set N_UNET_DEPTH=5 --set BATCH_SIZE=1]
  CUDA_VISIBLE_DEVICES=0 python -B repro_prof/train_repro.py --preset dETER --resume      # 같은 RUN_NAME 으로 재개
  프리셋: dETER(런 #22 재구성) · oETER · unet_only. 값 바꾸기: --set NAME=VALUE (config_repro.py 의 이름).
"""
import argparse
import contextlib
import datetime
import json
import os
import random
import shutil
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_prof import config_repro as CR  # noqa: E402
from repro_prof import data_cache as DC  # noqa: E402
from repro_prof import models_repro as M  # noqa: E402
from repro_prof import monitor as MON  # noqa: E402
from repro_prof import prof_paths as P  # noqa: E402

EXIT_ALERT = 3


# ------------------------------------------------------------------------------------------------------ 데이터
class ArraySet:
    """교수님 로더 클래스와 같은 모양의 묶음(torch Dataset 규약: __len__, __getitem__ → dict)."""

    def __init__(self, data, data_img, label, idx_list, source):
        self.data, self.data_img, self.label = data, data_img, label
        self.idx_list, self.source = list(idx_list), source

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, i):
        return {'data': self.data[i], 'data_img': self.data_img[i], 'label': self.label[i]}

    def __repr__(self):
        return f'<ArraySet volumes {self.idx_list[0]}..{self.idx_list[-1]} ({len(self.idx_list)}) source={self.source}>'


class SetSource:
    def __init__(self, cfg, with_img=True):
        self.mode = cfg['DATA_SOURCE']
        self.cache_dir = cfg['CACHE_DIR']
        self.threads = int(cfg['PREFETCH_THREADS'])
        self.with_img = with_img
        if self.mode not in ('cache', 'prof'):
            raise ValueError(self.mode)
        self._ex = ThreadPoolExecutor(1) if (self.mode == 'cache' and self.threads > 0) else None
        self._pending = None
        self.wait_s = 0.0
        self.load_s = []

    def _load(self, idx_list):
        t = time.time()
        if self.mode == 'cache':
            d, di, lab = DC.load_set(self.cache_dir, idx_list, n_threads=max(1, self.threads), with_img=self.with_img)
        else:
            if list(idx_list) != list(range(idx_list[0], idx_list[-1] + 1)):
                raise ValueError('prof 원천은 연속 볼륨 묶음만')
            ds = P.prof_dataset('test_v2', idx_start=idx_list[0], idx_end=idx_list[-1] + 1)
            d, di, lab = ds.data, ds.data_img, ds.label
        self.load_s.append(time.time() - t)
        return ArraySet(d, di, lab, idx_list, self.mode)

    def prefetch(self, idx_list):
        if self._ex is not None:
            self._pending = (tuple(idx_list), self._ex.submit(self._load, list(idx_list)))

    def get(self, idx_list):
        t = time.time()
        if self._pending is not None and self._pending[0] == tuple(idx_list):
            ds = self._pending[1].result()
            self._pending = None
        else:
            ds = self._load(list(idx_list))
        self.wait_s += time.time() - t
        return ds

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True)


def check_data(cfg, idx_list):
    """필요한 볼륨이 캐시(또는 .mat)에 있는지."""
    missing = []
    lines = P.read_list()
    for i in idx_list:
        if cfg['DATA_SOURCE'] == 'cache':
            if not (DC.vol_dir(cfg['CACHE_DIR'], i) / 'meta.json').exists():
                missing.append(i)
        elif not os.path.exists(P.mat_path(i, lines)):
            missing.append(i)
    return missing


# ------------------------------------------------------------------------------------------------------ 보조
class Tee:
    """교수님 Logger(231030:35-46)처럼 화면과 log.txt 에 같이 쓴다."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@contextlib.contextmanager
def tee_stdout(path):
    old = sys.stdout
    t = Tee(path)
    sys.stdout = t
    try:
        yield t
    finally:
        sys.stdout = old
        t.log.close()


def order_generator(cfg, kind, epoch, n_set=0):
    import torch
    if cfg['DATA_ORDER'] == 'global' or cfg['SEED'] is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(cfg['SEED']) * 100_000_000 + epoch * 100_000 + n_set * 10 + (1 if kind == 'val' else 0))
    return g


def save_atomic(obj, path):
    import torch
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seoul_now():
    try:
        import pytz
        return datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    except Exception:  # noqa: BLE001
        return datetime.datetime.now()


def needed_ckpt_bytes(cfg, n_params):
    per = n_params * 4
    n_files = (cfg['NUM_EPOCHS'] + 1) if cfg['SAVE_EVERY_EPOCH'] else 1
    return per * n_files + (n_params * 12 if cfg['SAVE_RESUME'] else 0)


def mssim_L(t):
    """mySSIM 이 고르는 동적 범위 L(mySSIM.py:24-35, 첫 인자 기준)."""
    mx = 255 if float(t.max()) > 128 else 1
    mn = -1 if float(t.min()) < -0.5 else 0
    return mx - mn


# ------------------------------------------------------------------------------------------------------ 학습
def run(cfg, *, max_steps=0, resume=False, log_dir=None, ckpt_dir=None, smoke_save=False, smoke_val=True,
        quiet_steps=False, allow_low_disk=False):
    """학습(또는 max_steps step 스모크). 요약 dict 를 돌려준다."""
    import torch
    from torch.utils.data import DataLoader

    run_dir = Path(log_dir) if log_dir else Path(cfg['LOG_ROOT']) / cfg.run_name
    ck_dir = Path(ckpt_dir) if ckpt_dir else Path(cfg['CKPT_ROOT']) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(run_name=cfg.run_name, preset=cfg.preset, log_dir=str(run_dir), ckpt_dir=str(ck_dir))
    with tee_stdout(run_dir / 'log.txt'):
        print(' ')
        print(run_dir / 'log.txt')
        cfg.print_all()
        (run_dir / 'settings.json').write_text(json.dumps(
            {k: dict(value=repr(s.value), src=s.src, status=s.status, assumed=s.assumed, note=s.note)
             for k, s in cfg.settings.items()}, indent=1, ensure_ascii=False))
        print(socket.gethostname())
        print(seoul_now(), '| UTC', datetime.datetime.utcnow().isoformat(timespec='seconds'))
        print(f'torch {torch.__version__} numpy {np.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()}')
        if not torch.cuda.is_available():
            raise SystemExit('CUDA 없음 — 학습은 GPU 에서만(원 코드도 .cuda() 고정). 점검은 --dry-run')
        dev_name = torch.cuda.get_device_name(0)
        print('device', dev_name)
        summary['device'] = dev_name

        if cfg['SEED'] is not None:
            random.seed(cfg['SEED'])
            np.random.seed(cfg['SEED'])
            torch.manual_seed(cfg['SEED'])
            torch.cuda.manual_seed_all(cfg['SEED'])
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg['TF32'])
        torch.backends.cudnn.allow_tf32 = bool(cfg['TF32'])
        mySSIM = P.prof_ssim_module()                      # import 시 cudnn.deterministic = True (mySSIM.py:6)
        print(f'cudnn.deterministic={torch.backends.cudnn.deterministic} benchmark={torch.backends.cudnn.benchmark} '
              f'tf32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}')
        criterion = P.prof_utils_module().myCriterion(cfg['CRITERION'])   # 231030:190

        need_img = not (cfg['MODEL'] == 'eter' and cfg['ETER_VARIANT'] == 'o')
        need_ksp = cfg['MODEL'] == 'eter'
        source = SetSource(cfg, with_img=need_img)
        miss = check_data(cfg, sorted(set(sum(cfg.sets, [])) | {cfg['VAL_IDX']}))
        if miss:
            raise SystemExit(f'데이터 없음({cfg["DATA_SOURCE"]}): 목록 번호 {miss[:10]}{"..." if len(miss) > 10 else ""}')

        torch.cuda.reset_peak_memory_stats()
        val_ds = source._load([cfg['VAL_IDX']])            # 231030:179 (_val num_total_set=1 → 목록 300번)
        model = M.build_model(cfg).cuda()                  # 231030:186
        pb = M.param_breakdown(model)
        print('params', pb)
        summary['params'] = pb
        mem_model = torch.cuda.memory_allocated()

        need = needed_ckpt_bytes(cfg, pb['total'])
        free = shutil.disk_usage(ck_dir).free
        print(f'checkpoint dir {ck_dir}: need ≈{need / 1e9:.1f} GB for the full schedule, free {free / 1e9:.1f} GB')
        if max_steps == 0 and free < need * 1.05 and not allow_low_disk:
            raise SystemExit('체크포인트 공간 부족 (--allow-low-disk 로 무시)')

        from repro_prof import myoptimizer_shim as OPT
        optimizer = OPT.myOptimizer(model, cfg['OPTIMIZER'], cfg['LEARNING_RATE_ADAM'], mode=cfg['MYOPT_MODE'],
                                    weight_decay=cfg['LAMBDA_REGULAR_PER_PIXEL'], foreach=cfg['ADAM_FOREACH'])  # 231030:191
        crit_ssim = mySSIM.SSIM().cuda()                   # 231030:193
        lambda_ssim_per_pixel = cfg['LAMBDA_SSIM_PER_PIXEL']
        num_epochs = cfg['NUM_EPOCHS']
        BS = cfg['BATCH_SIZE']
        steps_per_set = [int(np.ceil(len(s) * 16 / BS)) for s in cfg.sets]
        validation_loss = np.zeros((num_epochs, 1))
        training_loss = np.zeros((num_epochs, max(4800, sum(steps_per_set))))   # 231030:202 (폭 4,800)
        ssim_prev = 0
        groups = MON.build_groups(model, cfg['REFINE'], cfg['N_UNET_DEPTH'])
        start_epoch, global_step = 0, 0
        resume_path = ck_dir / 'resume.pt'
        if resume and resume_path.exists():
            st = torch.load(resume_path, map_location='cuda')
            model.load_state_dict(st['model'])
            optimizer.load_state_dict(st['optimizer'])
            start_epoch, global_step = st['epoch'] + 1, st['global_step']
            training_loss[:st['training_loss'].shape[0], :st['training_loss'].shape[1]] = st['training_loss']
            validation_loss[:] = st['validation_loss']
            ssim_prev = st['ssim_prev']
            torch.set_rng_state(st['rng_torch'])
            torch.cuda.set_rng_state_all(st['rng_cuda'])
            np.random.set_state(st['rng_numpy'])
            random.setstate(st['rng_python'])
            print(f'resumed from {resume_path}: epoch {start_epoch + 1} global_step {global_step}')
        elif resume:
            print(f'resume 요청했으나 {resume_path} 없음 → 처음부터')

        def health(epoch, where):
            stt = MON.param_stats(model, groups)
            print('  ' + MON.format_health(stt, epoch, global_step, where))
            bad = MON.collapsed_groups(stt)
            if bad and cfg['COLLAPSE_ACTION'] == 'stop':
                (run_dir / 'ALERT_COLLAPSE').write_text(json.dumps(dict(epoch=epoch + 1, step=global_step, groups=bad,
                                                                        stats=stt), indent=1))
                print(f'ALERT_COLLAPSE {bad} → exit {EXIT_ALERT}')
                raise SystemExit(EXIT_ALERT)
            return stt

        summary['health_start'] = health(start_epoch, 'start')
        steps = []
        csv = open(run_dir / 'train_steps.csv', 'a')
        if csv.tell() == 0:
            csv.write('epoch,n_set,i_batch,global_step,loss,pix,one_minus_ssim,step_s,wall\n')
        stop = False
        loss_ssim = None
        print('\n\n\n  start training iterations')
        source.prefetch(cfg.sets[0])
        t_train0 = time.time()
        epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):
            for n_set in range(cfg['NUM_TRAIN_SET']):
                choh_data_train = source.get(cfg.sets[n_set])
                if n_set + 1 < cfg['NUM_TRAIN_SET']:
                    source.prefetch(cfg.sets[n_set + 1])
                elif epoch + 1 < num_epochs and not stop:
                    source.prefetch(cfg.sets[0])
                print(choh_data_train)
                print(' len(choh_data_train) : %d' % len(choh_data_train))
                trainloader = DataLoader(choh_data_train, batch_size=BS, shuffle=True,
                                         generator=order_generator(cfg, 'train', epoch, n_set))   # 231030:212
                print(trainloader)
                total_step = len(trainloader)
                t_prev = time.perf_counter()
                for i_batch, sample_batched in enumerate(trainloader):
                    t0 = time.perf_counter()
                    fetch_s = t0 - t_prev                    # DataLoader 가 배치를 모으는 시간(CPU)
                    data_in = sample_batched['data'].type(torch.cuda.FloatTensor) if need_ksp else None
                    data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor) if need_img else None
                    data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
                    optimizer.zero_grad()
                    out = model(data_in, data_in_img)
                    loss_pixel = criterion(out, data_ref)                       # 231030:234-236
                    loss_ssim = 1 - crit_ssim(out, data_ref)
                    loss = loss_pixel + lambda_ssim_per_pixel * loss_ssim
                    loss.backward()
                    optimizer.step()
                    lp, ls, lt = loss_pixel.item(), loss_ssim.item(), loss.item()
                    training_loss[epoch, n_set * total_step + i_batch] = lp    # 231030:245
                    dt = time.perf_counter() - t0
                    global_step += 1
                    if not quiet_steps:
                        print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format(
                            epoch + 1, num_epochs, n_set + 1, cfg['NUM_TRAIN_SET'], i_batch + 1, total_step, lt, lp, ls))
                    csv.write(f'{epoch + 1},{n_set + 1},{i_batch + 1},{global_step},{lt:.8g},{lp:.8g},{ls:.8g},{dt:.4f},{time.time():.1f}\n')
                    steps.append((global_step, lt, lp, ls, dt, fetch_s))
                    if global_step == 1:
                        summary['L_mySSIM_first_output'] = mssim_L(out.detach())
                        summary['L_mySSIM_label'] = mssim_L(data_ref)
                        summary['label_max_first_batch'] = float(data_ref.max())
                    if cfg['HEALTH_EVERY_STEPS'] and global_step % cfg['HEALTH_EVERY_STEPS'] == 0:
                        health(epoch, 'step')
                    if max_steps and global_step >= max_steps:
                        stop = True
                        break
                    t_prev = time.perf_counter()
                del trainloader, choh_data_train
                if stop:
                    break
            csv.flush()
            print('\n    choh, cwd : %s %s \n' % (os.getcwd(), run_dir))
            print('    socket.gethostname() {} '.format(socket.gethostname()))
            print(seoul_now())
            if stop and not smoke_val:
                break
            if loss_ssim is not None and ssim_prev < loss_ssim.item():               # 231030:263-266 (원문 그대로, 저장 안 함)
                ssim_prev = loss_ssim.item()
                print(' loss_ssim improved, \n')
            # ---- 검증 (231030:268-277)
            tv = time.time()
            vloader = DataLoader(val_ds, batch_size=BS, shuffle=True, generator=order_generator(cfg, 'val', epoch))
            vp, vs = [], []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(vloader):
                    data_in = sample_batched['data'].type(torch.cuda.FloatTensor) if need_ksp else None
                    data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor) if need_img else None
                    data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
                    out = model(data_in, data_in_img)
                    loss_pixel = criterion(out, data_ref)
                    if cfg['EXTRA_VAL_METRICS']:
                        vp.append(loss_pixel.item())
                        vs.append(crit_ssim(out, data_ref).item())
                print('  epoch {}\t validation_loss: {:.6f}'.format(epoch + 1, loss_pixel))
                validation_loss[epoch] = loss_pixel.item()
            if vp:
                print('  epoch {}\t [repro] val mean criterion {:.6f}  mean mySSIM {:.6f}  ({} batches, 선택에 쓰지 않음)'.format(
                    epoch + 1, float(np.mean(vp)), float(np.mean(vs)), len(vp)))
            summary['val_s'] = time.time() - tv
            summary['val_last_batch_criterion'] = float(validation_loss[epoch, 0])
            summary['val_mean_criterion'] = float(np.mean(vp)) if vp else None
            summary['val_mean_mySSIM'] = float(np.mean(vs)) if vs else None
            # ---- 저장 (231030:281)
            if (cfg['SAVE_EVERY_EPOCH'] and not stop) or (stop and smoke_save):
                ts = time.time()
                p = ck_dir / ('tensors_%d.pt' % (epoch + 1))
                save_atomic(model.state_dict() if cfg['CKPT_FORMAT'] == 'state_dict' else model, p)
                summary['save_s'] = time.time() - ts
                summary['save_bytes'] = os.path.getsize(p)
                print('saving done...\n')
            if cfg['SAVE_RESUME'] and not stop:
                save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                 global_step=global_step, training_loss=training_loss, validation_loss=validation_loss,
                                 ssim_prev=ssim_prev, rng_torch=torch.get_rng_state(),
                                 rng_cuda=torch.cuda.get_rng_state_all(), rng_numpy=np.random.get_state(),
                                 rng_python=random.getstate(), settings=cfg.as_dict(), run_name=cfg.run_name),
                            resume_path)
            np.save(run_dir / 'validation_loss', validation_loss)
            np.save(run_dir / 'training_loss', training_loss)
            summary['health_end'] = health(epoch, 'epoch_end')
            if stop:
                break
        if not stop and not max_steps:
            save_atomic(model.state_dict() if cfg['CKPT_FORMAT'] == 'state_dict' else model, ck_dir / 'tensors_entire.pt')
            print('saving done...\n')
            (run_dir / 'DONE').write_text(seoul_now().isoformat())
        csv.close()
        source.close()
        summary['train_wall_s'] = time.time() - t_train0
        summary['data_wait_s'] = source.wait_s
        summary['data_load_s'] = source.load_s
        summary['mem_model_GB'] = mem_model / 2 ** 30
        summary['peak_alloc_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
        summary['peak_reserved_GB'] = torch.cuda.max_memory_reserved() / 2 ** 30
        summary['steps'] = steps
        print('total Time = ', summary['train_wall_s'])
        print(seoul_now())
        print('finished !')
    return summary


# ------------------------------------------------------------------------------------------------------ 점검(GPU 없음)
def dry_run(cfg):
    import torch
    cfg.print_all()
    with torch.device('meta'):
        m = M.build_model(cfg)
    pb = M.param_breakdown(m)
    print('params (meta)', pb)
    need = needed_ckpt_bytes(cfg, pb['total'])
    ck_root = Path(cfg['CKPT_ROOT'])
    probe = ck_root if ck_root.exists() else ck_root.parent
    while not probe.exists():
        probe = probe.parent
    print(f'checkpoint need ≈{need / 1e9:.1f} GB, free at {probe}: {shutil.disk_usage(probe).free / 1e9:.1f} GB')
    miss = check_data(cfg, sorted(set(sum(cfg.sets, [])) | {cfg['VAL_IDX']}))
    print(f'data source {cfg["DATA_SOURCE"]}: missing {len(miss)} volumes' + (f' (e.g. {miss[:5]})' if miss else ''))
    steps = len(cfg.train_idx) * 16 / cfg['BATCH_SIZE']
    print(f'steps/epoch {steps:.0f}, total {steps * cfg["NUM_EPOCHS"]:.0f}')
    return pb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default='dETER', choices=list(CR.PRESETS))
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--allow-low-disk', action='store_true')
    a = ap.parse_args(argv)
    cfg = CR.Config(a.preset, CR.parse_overrides(a.set))
    if a.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''          # 점검은 CPU 만(CUDA 는 처음 쓸 때 초기화되므로 여기서 막힘)
        dry_run(cfg)
        return 0
    run(cfg, max_steps=a.max_steps, resume=a.resume, allow_low_disk=a.allow_low_disk)
    return 0


if __name__ == '__main__':
    sys.exit(main())
