"""교수님 로더(230425) 출력을 비트 단위로 재현하는 압축 캐시 (float32, CPU 전용).

배경
- 교수님 d-ETER 학습 루프(231030:205-216)는 epoch 마다 학습 .mat 300개(173.17 GB, MAT 5.0 압축)를 다시 읽고
  전처리한다. 원 로더는 볼륨당 약 10.8 초가 걸린다(loadmat 3.4~3.8 s + 전처리, verify_design §1.4).
- 원 로더에는 무작위가 없다(고정 마스크 230425:525-526). 그래서 볼륨별 출력을 미리 만들어 두어도 입력이 같다.

캐시 형식 (볼륨당 폴더 vol_{목록번호:03d}/)
- ksp_cols.npy : float32 (S, 32, 384, 120). 원 로더의 마스크 k-space(X_ksp_volume, 230425:524-526)를 x_ksp_zf 배치
                 (230425:536: 채널 = 2·코일 + 0 실수/1 허수, 축 2 = kx 읽기 방향, 축 3 = ky)로 바꾼 뒤 획득 120열만 남긴 값.
                 배율(×1e3) 전 값이다. 획득 열 = 3:384:4 ∪ 176:208(오름차순 120개). 나머지 열은 원 로더에서 정확히 +0.0 이다
                 (빌드 때마다 확인).
- label.npy    : float32 (S, 1, 384, 384). 원 로더의 label(16코일 RSS ×1e5, 230425:516-522·542) 그대로.
- meta.json    : 목록 번호, 파일 이름, 원본 크기·수정 시각, 열 번호, 배율, numpy/scipy 판, sha1.
- S = .mat 의 칸 수(목록 379개 모두 16).

읽기 (read_volume)
- data     = 1e3 · zero_fill(ksp_cols)
- data_img = 1e5 · [코일마다 교수님 choh_ifft2c_single] (함수는 교수님 로더 모듈에서 그대로 import)
- label    = label.npy
- 원 로더와 같은 dtype·같은 식으로 계산하므로 출력이 비트 단위로 같다(verify 명령으로 확인).

크기: 볼륨당 94,371,840 B(ksp_cols) + 9,437,184 B(label) = 103.8 MB → 379볼륨 39.34 GB, 학습 300볼륨 31.14 GB.

CLI (저장소 루트에서, CPU 전용)
  python -B repro_prof/data_cache.py build  --out DIR [--start 0 --end 379 | --idx 0 1 300] [--workers 4] [--verify-every 0]
  python -B repro_prof/data_cache.py verify --out DIR --idx 0 43 62 300
  python -B repro_prof/data_cache.py info   --out DIR
"""
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU 전용(교수님 로더가 torch 를 import 한다)

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_prof import prof_paths as P  # noqa: E402

N_COIL = 16                    # 230425:475
N_CH = 2 * N_COIL              # 230425:477 n_in_ch
N_DIM = 384                    # 230425:474
VAL_AMP_X_IMG = 1e5            # 230425:477
VAL_AMP_X_KSP = 1e3            # 230425:478
VAL_AMP_Y = 1e5                # 230425:479
KY_COLS = np.array(sorted(set(range(3, 384, 4)) | set(range(192 - 16, 192 + 16))), dtype=np.int64)   # 230425:525-526
assert KY_COLS.size == 120
FORMAT_VERSION = 1
DEFAULT_CACHE_DIR = '/home/snorlax/shared/cache_repro_prof/prof379_f32_v1'   # 계획(NVMe, 39.34 GB) — 이번 단계에서는 만들지 않음


def _sha1(a):
    return hashlib.sha1(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def compute_from_mat(idx, lines=None, timing=None):
    """목록 idx 번 볼륨의 .mat 를 읽어 (ksp_cols, label) 을 만든다. 식과 dtype 은 230425:510-536 과 같다."""
    import scipy.io as sio
    L = P.prof_loader_module()
    path = P.mat_path(idx, lines)
    t0 = time.time()
    XX_ksp = sio.loadmat(path)['ksp_16ch']                      # (S, 384, 384, 16, 2) float64
    t1 = time.time()
    num_slices = XX_ksp.shape[0]
    n_in_ch, n_dim = N_CH, N_DIM
    # 정답: 230425:516-522 (코일별 ifft 결과를 float32 배열에 넣은 뒤 float32 로 RSS)
    img_full_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
    for cc in range(XX_ksp.shape[3]):
        ksp_1ch = np.transpose(np.squeeze(XX_ksp[:, :, :, cc, :]), (0, 3, 1, 2))
        img_full_16ch[:, 2 * cc:2 * cc + 2, :, :] = L.choh_ifft2c_single(ksp_1ch)
    Y_volume = np.reshape(np.sqrt(np.sum(np.square(img_full_16ch), axis=1)), [num_slices, 1, n_dim, n_dim])
    label = VAL_AMP_Y * Y_volume                                  # 230425:542 (float32)
    del img_full_16ch
    # 마스크 k-space: 230425:524-526, 배치 230425:535-536
    X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL, 2), dtype=np.float32)
    X_ksp_volume[:, :, 3:384:4, :, :] = XX_ksp[:, :, 3:384:4, :, :]
    X_ksp_volume[:, :, 192 - 16:192 + 16, :, :] = XX_ksp[:, :, 192 - 16:192 + 16, :, :]
    x_ksp_zf = np.reshape(np.transpose(X_ksp_volume, (0, 3, 4, 1, 2)), [num_slices, n_in_ch, n_dim, n_dim])
    rest = np.ones(n_dim, dtype=bool)
    rest[KY_COLS] = False
    rest_vals = x_ksp_zf[..., rest]
    if np.count_nonzero(rest_vals) or np.signbit(rest_vals).any():
        raise RuntimeError(f'idx {idx}: 획득하지 않은 열에 +0.0 이 아닌 값이 있다 — 캐시 형식 가정이 깨짐')
    ksp_cols = np.ascontiguousarray(x_ksp_zf[..., KY_COLS])
    t2 = time.time()
    if timing is not None:
        timing.update(loadmat_s=t1 - t0, compute_s=t2 - t1)
    st = os.stat(path)
    meta = dict(format_version=FORMAT_VERSION, idx=int(idx), file=os.path.basename(path), src_size=st.st_size,
                src_mtime=st.st_mtime, num_slices=int(num_slices), ky_cols=KY_COLS.tolist(),
                amp=dict(x_ksp=VAL_AMP_X_KSP, x_img=VAL_AMP_X_IMG, y=VAL_AMP_Y),
                numpy=np.__version__, python=platform.python_version())
    try:
        import scipy
        meta['scipy'] = scipy.__version__
    except Exception:  # noqa: BLE001
        pass
    return ksp_cols, label, meta


def vol_dir(cache_dir, idx):
    return Path(cache_dir) / f'vol_{int(idx):03d}'


def write_volume(cache_dir, idx, ksp_cols, label, meta):
    final = vol_dir(cache_dir, idx)
    tmp = final.with_name(final.name + f'.tmp{os.getpid()}')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / 'ksp_cols.npy', ksp_cols)
    np.save(tmp / 'label.npy', label)
    meta = dict(meta, sha1_ksp_cols=_sha1(ksp_cols), sha1_label=_sha1(label),
                created=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=1))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def read_meta(cache_dir, idx):
    return json.loads((vol_dir(cache_dir, idx) / 'meta.json').read_text())


def read_volume(cache_dir, idx, out=None, check_sha1=False, with_img=True):
    """캐시 → (data, data_img, label). out=(data, data_img, label) 를 주면 그 배열(슬라이스)에 바로 쓴다.

    with_img=False 면 data_img 를 계산하지 않고 0 으로 둔다(o-ETER 처럼 영상 입력을 쓰지 않는 모델의 CPU 시간 절약).
    """
    L = P.prof_loader_module()
    d = vol_dir(cache_dir, idx)
    ksp_cols = np.load(d / 'ksp_cols.npy')
    label = np.load(d / 'label.npy')
    if check_sha1:
        meta = read_meta(cache_dir, idx)
        if _sha1(ksp_cols) != meta['sha1_ksp_cols'] or _sha1(label) != meta['sha1_label']:
            raise RuntimeError(f'{d}: sha1 불일치')
    S = ksp_cols.shape[0]
    if out is None:
        out = (np.zeros((S, N_CH, N_DIM, N_DIM), np.float32), np.zeros((S, N_CH, N_DIM, N_DIM), np.float32),
               np.zeros((S, 1, N_DIM, N_DIM), np.float32))
    data, data_img, lab = out
    x_ksp_zf = np.zeros((S, N_CH, N_DIM, N_DIM), dtype=np.float32)
    x_ksp_zf[:, :, :, KY_COLS] = ksp_cols
    np.multiply(x_ksp_zf, VAL_AMP_X_KSP, out=data)               # = val_amp_X_ksp*x_ksp_zf (230425:541, float32)
    if with_img:
        for cc in range(N_COIL):                                 # 230425:529-533 (코일별 ifft → float32 배열)
            data_img[:, 2 * cc:2 * cc + 2, :, :] = L.choh_ifft2c_single(x_ksp_zf[:, 2 * cc:2 * cc + 2, :, :])
        np.multiply(data_img, VAL_AMP_X_IMG, out=data_img)       # = val_amp_X_img*img_alias_16ch (230425:540)
    else:
        data_img[...] = 0.0
    lab[...] = label
    return data, data_img, lab


def load_set(cache_dir, idx_list, n_threads=1, with_img=True):
    """여러 볼륨을 원 로더 클래스와 같은 배열(슬라이스 순서 = 목록 순서 × 칸 순서)로 모은다.

    numpy FFT(pocketfft)는 GIL 을 놓으므로 스레드로 병렬화한다. 계산은 볼륨 단위로 독립이라 결과는 스레드 수와 무관하다.
    """
    metas = [read_meta(cache_dir, i) for i in idx_list]
    sizes = [m['num_slices'] for m in metas]
    total = int(sum(sizes))
    data = np.zeros((total, N_CH, N_DIM, N_DIM), np.float32)
    data_img = np.zeros((total, N_CH, N_DIM, N_DIM), np.float32)
    label = np.zeros((total, 1, N_DIM, N_DIM), np.float32)
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def work(k):
        a, b = offs[k], offs[k + 1]
        read_volume(cache_dir, idx_list[k], out=(data[a:b], data_img[a:b], label[a:b]), with_img=with_img)

    if n_threads <= 1:
        for k in range(len(idx_list)):
            work(k)
    else:
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(work, range(len(idx_list))))
    return data, data_img, label


def bitwise_compare(a, b):
    """(같은가, 다른 원소 수, 최대 절대 차). -0.0/+0.0 차이도 다름으로 센다."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1, float('nan')
    ua, ub = np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32)
    ndiff = int(np.count_nonzero(ua != ub))
    mad = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if ndiff else 0.0
    return ndiff == 0, ndiff, mad


def verify_against_prof(cache_dir, idx, kind='test_v2'):
    """교수님 로더 클래스 출력(메모리)과 캐시 읽기 출력을 비트 단위로 비교한다."""
    import contextlib
    import io
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        if kind == 'test_v2':
            ds = P.prof_dataset('test_v2', idx_start=idx, idx_end=idx + 1)
        elif kind == 'val':
            assert idx == 300
            ds = P.prof_dataset('val', num_total_set=1)
        elif kind == 'test59':
            assert idx == 320
            ds = P.prof_dataset('test59', num_total_set=1)
        else:
            raise ValueError(kind)
    t_prof = time.time() - t0
    t1 = time.time()
    data, data_img, label = read_volume(cache_dir, idx, check_sha1=True)
    t_read = time.time() - t1
    res = dict(idx=int(idx), kind=kind, prof_s=round(t_prof, 2), cache_read_s=round(t_read, 2), n_slices=int(len(ds)))
    for name, mine, theirs in (('data', data, ds.data), ('data_img', data_img, ds.data_img), ('label', label, ds.label)):
        same, ndiff, mad = bitwise_compare(mine, theirs)
        res[name] = dict(bit_identical=same, n_diff=ndiff, max_abs_diff=mad, shape=list(theirs.shape), dtype=str(theirs.dtype))
    res['all_bit_identical'] = all(res[k]['bit_identical'] for k in ('data', 'data_img', 'label'))
    return res


def _build_one(args):
    cache_dir, idx, overwrite = args
    if not overwrite and (vol_dir(cache_dir, idx) / 'meta.json').exists():
        return dict(idx=idx, skipped=True)
    tm = {}
    t0 = time.time()
    ksp_cols, label, meta = compute_from_mat(idx, timing=tm)
    t1 = time.time()
    write_volume(cache_dir, idx, ksp_cols, label, meta)
    tm.update(idx=idx, write_s=time.time() - t1, total_s=time.time() - t0)
    return tm


def build(cache_dir, idx_list, workers=1, overwrite=False, verify_every=0, log=print):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    if workers <= 1:
        for i in idx_list:
            results.append(_build_one((cache_dir, i, overwrite)))
            log(json.dumps(results[-1]))
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as ex:
            for r in ex.map(_build_one, [(cache_dir, i, overwrite) for i in idx_list]):
                results.append(r)
                log(json.dumps(r))
    wall = time.time() - t0
    ver = []
    if verify_every and verify_every > 0:
        for i in idx_list[::verify_every]:
            ver.append(verify_against_prof(cache_dir, i))
            log(json.dumps(ver[-1]))
    return dict(wall_s=wall, n=len(idx_list), results=results, verify=ver)


def info(cache_dir):
    vols = sorted(p for p in Path(cache_dir).glob('vol_*') if p.is_dir() and '.tmp' not in p.name)
    nbytes = sum(f.stat().st_size for v in vols for f in v.iterdir())
    return dict(n_volumes=len(vols), bytes=nbytes, first=vols[0].name if vols else None, last=vols[-1].name if vols else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['build', 'verify', 'info'])
    ap.add_argument('--out', default=DEFAULT_CACHE_DIR)
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--idx', type=int, nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verify-every', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'info':
        print(json.dumps(info(a.out), indent=1))
        return
    idx_list = a.idx if a.idx else list(range(a.start or 0, a.end if a.end is not None else 379))
    if a.cmd == 'build':
        probe = Path(a.out)
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        need = len(idx_list) * 103_809_024
        print(f'build {len(idx_list)} volumes → {a.out} (need ≈{need / 1e9:.2f} GB, free {free / 1e9:.1f} GB)')
        if free < need * 1.05:
            raise SystemExit('여유 공간 부족')
        r = build(a.out, idx_list, workers=a.workers, overwrite=a.overwrite, verify_every=a.verify_every)
        print(json.dumps(dict(wall_s=r['wall_s'], n=r['n']), indent=1))
    else:
        for i in idx_list:
            kind = 'val' if i == 300 else ('test59' if i == 320 else 'test_v2')
            print(json.dumps(verify_against_prof(a.out, i, kind)))


if __name__ == '__main__':
    main()
