#!/bin/bash
# ============================================================================
#  [3/6] 호스트에서 실행 — 새 이미지 빌드 (기존 컨테이너 건드리지 않음)
#
#  이 단계는 완전히 안전하다: 이미지를 새로 만들 뿐, 돌고 있는 컨테이너에
#  아무 영향이 없다. 빌드가 실패하면 그냥 다시 하면 된다.
#
#  소요: 인터넷 속도에 따라 15~40분 (베이스 이미지 ~9GB 다운로드 포함)
#  실행:  bash 20_host_build.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-mri:v1}"

echo "── 이미지: ${IMAGE}"
echo "── 컨텍스트: ${HERE}"
echo

docker build \
  --pull \
  -t "${IMAGE}" \
  -f "${HERE}/Dockerfile" \
  "${HERE}"

echo
echo "══ 빌드 완료"
docker images "${IMAGE%%:*}" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}'

echo
echo "── 이미지 단독 스모크 (GPU 없이 import 만 확인)"
docker run --rm "${IMAGE}" python -c "
import torch, einops, h5py, scipy, skimage, matplotlib, pandas, wandb, fastmri
print('torch      ', torch.__version__, '/ cuda build', torch.version.cuda)
print('numpy       ', __import__('numpy').__version__)
print('einops/h5py ', einops.__version__, h5py.__version__)
print('skimage/scipy', skimage.__version__, scipy.__version__)
import mamba_ssm; print('mamba_ssm  ', mamba_ssm.__version__)
print('OK — import 전부 통과 (CUDA 커널 실행 검증은 50_verify_env.sh 에서)')
"

echo
echo "  다음 단계: bash 30_host_run.sh"
