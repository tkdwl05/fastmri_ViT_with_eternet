#!/bin/bash
# RunPod Pod: 공식 fastMRI 링크 → 로컬 디스크로 직접 다운로드·압축 해제, 묶음별 병렬 (2026-10-09 사용자 결정: val 0~2 전체 + test_full 추가).
#   train = 대조표(dataset_manifest_nvme.json) 파일만(로컬 서버와 같은 학습 데이터), val·test_full = 묶음 전체.
# 입력: $S/urls.txt (git 에 두지 않는 개인 서명 URL), $S/dataset_manifest_nvme.json. 재실행 시 .done 묶음은 건너뜀.
set -u
S=/root/_setup; OUT=/root/fastmri_data
mkdir -p $OUT $S/logs
python3 - <<'PY'
import json
m=json.load(open('/root/_setup/dataset_manifest_nvme.json'))
open('/root/_setup/list_train.txt','w').write(''.join(f'multicoil_train/{r[0]}\n' for r in m['multicoil_train']))
open('/root/_setup/list_val.txt','w').write(''.join(f'multicoil_val/{r[0]}\n' for r in m['multicoil_val']))
PY
run_one() {
  local url="$1" name list
  name=$(basename "${url%%\?*}" .tar.xz)
  [ -e $S/logs/$name.done ] && return 0
  case $name in *train*) filt="--files-from=$S/list_train.txt";; *) filt="";; esac
  for attempt in 1 2 3 4 5; do
    echo "$(date -u +%T) $name attempt $attempt" >> $S/logs/$name.log
    set -o pipefail
    curl -sS --fail --retry 10 --retry-delay 10 "$url" | xz -dc | tar -x -C $OUT $filt 2>> $S/logs/$name.tarerr
    rcs=("${PIPESTATUS[@]}")
    echo "$(date -u +%T) $name rc curl=${rcs[0]} xz=${rcs[1]} tar=${rcs[2]}" >> $S/logs/$name.log
    if [ "${rcs[0]}" = 0 ] && [ "${rcs[1]}" = 0 ]; then touch $S/logs/$name.done; return 0; fi
    sleep 30
  done
  echo "$name FAILED" >> $S/logs/$name.log
}
while read -r url; do [ -n "$url" ] && run_one "$url" & sleep 2; done < $S/urls.txt
wait
echo "ALL FINISHED $(date -u +%T)" >> $S/logs/_all.log
