"""
RunPod Pod 데이터 검증 (2026-10-09) — 공식 링크로 받은 fastMRI brain multicoil 을 로컬 서버 데이터와 같은 구성으로 맞춘다.

기준 = infra/runpod/dataset_manifest_nvme.json (로컬 서버 /home/snorlax/shared/fastmri_data_nvme 의 파일 이름·크기·부분 해시
[앞 1 MiB + 뒤 1 MiB 의 sha256 앞 16자]). 로컬 사본은 train 2 파일이 잘려 있어 데이터 로더가 열지 못하고 건너뛴다
(로컬 학습 로그: 'Loaded 4108 .h5 files → 65028 slices (skipped 2 files)'). 공식본은 그 2 파일이 온전하므로, 같은 학습 데이터를
만들려면 Pod 에서 그 파일들을 지워야 한다.

  1) 파일 대조: 없음 / 남는 파일 / 크기 다름 / 부분 해시 다름.
  2) --fix: 남는 파일과 크기가 다른 파일(= 로컬에서 잘린 파일)을 지운다. 해시만 다른 파일은 지우지 않고 보고만 한다(조사 대상).
  3) 데이터 로더(dataloader_h5_v5.FastMRI_H5_Dataloader)로 파일·슬라이스 수를 세어 로컬과 같은지 확인한다
     (train 4108 파일 / 65028 슬라이스, val 464 / 7334). 로더는 파일을 이름순으로 정렬하므로 수가 같으면 슬라이스 순서도 같다.

실행 (Pod): /opt/mri310/bin/python /root/repo/infra/runpod/pod_verify_data.py --root /root/fastmri_data [--fix]
"""
import os
import sys
import json
import hashlib
import argparse

EXPECT = {'multicoil_train': (4108, 65028), 'multicoil_val': (464, 7334)}


def part_hash(p, sz):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read(1 << 20))
        if sz > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/fastmri_data')
    ap.add_argument('--manifest', default='/root/repo/infra/runpod/dataset_manifest_nvme.json')
    ap.add_argument('--fix', action='store_true')
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    ok_all = True
    for split, rows in man.items():
        d = os.path.join(a.root, split)
        have = {f for f in os.listdir(d) if f.endswith('.h5')} if os.path.isdir(d) else set()
        want = {r[0]: (r[1], r[2]) for r in rows}
        missing = sorted(set(want) - have)
        extra = sorted(have - set(want))
        size_diff, hash_diff = [], []
        for fn in sorted(have & set(want)):
            p = os.path.join(d, fn)
            sz = os.path.getsize(p)
            if sz != want[fn][0]:
                size_diff.append((fn, want[fn][0], sz))
            elif part_hash(p, sz) != want[fn][1]:
                hash_diff.append(fn)
        print(f'[{split}] 기준 {len(want)} / 있음 {len(have)} | 없음 {len(missing)} | 남음 {len(extra)} | '
              f'크기 다름 {len(size_diff)} | 해시 다름 {len(hash_diff)}')
        for fn, w, h in size_diff:
            print(f'    크기 다름: {fn} 로컬 {w:,} B vs 여기 {h:,} B')
        for fn in missing[:10]:
            print(f'    없음: {fn}')
        for fn in hash_diff[:10]:
            print(f'    해시 다름: {fn}')
        if missing or hash_diff:
            ok_all = False
        if a.fix:
            for fn in extra + [x[0] for x in size_diff]:
                os.remove(os.path.join(d, fn))
            if extra or size_diff:
                print(f'    --fix: 지움 {len(extra)} 남는 파일 + {len(size_diff)} 크기 다른 파일')

    sys.path.insert(0, '/root/repo/dataloaders')
    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    for split, (nf, ns) in EXPECT.items():
        ds = FastMRI_H5_Dataloader(os.path.join(a.root, split), num_files=None, target_size=384,
                                   random_mask=False, augment=False)
        good = (len(ds.files) == nf and len(ds.samples) == ns)
        ok_all &= good
        print(f'[{split}] 로더: {len(ds.files)} 파일 / {len(ds.samples)} 슬라이스 — 로컬 {nf} / {ns} → {"일치" if good else "불일치"}')
    print('VERIFY', 'OK' if ok_all else 'MISMATCH')


if __name__ == '__main__':
    main()
