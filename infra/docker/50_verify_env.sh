#!/bin/bash
# ============================================================================
#  [6/6] 새 컨테이너 안에서 실행 — 학습 재개 전 최종 검증
#
#  단순 import 가 아니라 "실제로 학습이 돌 수 있는가"를 본다:
#  GPU 연산 · SS2D CUDA 커널 실행 · 데이터 읽기 · ckpt 로드 · shm.
# ============================================================================
set -uo pipefail
REPO="${REPO:-/home/snorlax/shared/fastmri_ViT_with_eternet}"
FAIL=0
ok(){ echo "  ✓ $1"; }; ng(){ echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "════════ [1] GPU 접근 ════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | sed 's/^/  /' && ok "nvidia-smi" || ng "nvidia-smi 실패"
python - <<'PY'
import os
for d in ["/dev/nvidiactl","/dev/nvidia0","/dev/nvidia-uvm"]:
    try:
        fd=os.open(d,os.O_RDWR); os.close(fd); print(f"  ✓ {d} open OK")
    except OSError as e:
        print(f"  ✗ {d} open 실패: {e.strerror}  ← device cgroup allowlist 문제")
PY

echo
echo "════════ [2] torch / CUDA 실연산 ════════"
python - <<'PY'
import torch, sys
print("  torch", torch.__version__, "| cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    print("  ✗ torch.cuda.is_available() == False"); sys.exit(1)
print("  device:", torch.cuda.get_device_name(0))
a=torch.randn(2048,2048,device='cuda'); b=a@a; torch.cuda.synchronize()
print(f"  ✓ matmul OK (sum={b.sum().item():.1f})")
x=torch.randn(1,32,384,384,device='cuda')
f=torch.fft.fftshift(torch.fft.fft2(x)); torch.cuda.synchronize()
print("  ✓ FFT(384²) OK")
PY
[ $? -eq 0 ] && ok "torch CUDA 실연산" || ng "torch CUDA 실연산 실패"

echo
echo "════════ [3] SS2D CUDA 커널 (mamba_ssm) ════════"
python - <<'PY'
import torch, mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
B,D,L,N=2,64,256,16
u=torch.randn(B,D,L,device='cuda',dtype=torch.float16)
delta=torch.rand(B,D,L,device='cuda',dtype=torch.float16)
A=-torch.rand(D,N,device='cuda'); Bm=torch.randn(B,1,N,L,device='cuda',dtype=torch.float16)
C=torch.randn(B,1,N,L,device='cuda',dtype=torch.float16); Dm=torch.randn(D,device='cuda')
y=selective_scan_fn(u,delta,A,Bm,C,Dm); torch.cuda.synchronize()
print(f"  ✓ selective_scan_fn OK  out={tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
PY
[ $? -eq 0 ] && ok "mamba_ssm CUDA 커널" || ng "mamba_ssm CUDA 커널 실패 (wheel/ABI 불일치 의심)"

echo
echo "════════ [4] 데이터 · 체크포인트 접근 ════════"
[ -d "${REPO}" ] && ok "repo 마운트" || ng "repo 없음"
[ -d "${REPO}/logs" ] && ok "logs/ ($(du -sh ${REPO}/logs 2>/dev/null | cut -f1))" || ng "logs/ 없음"
python - <<PY
import glob, os
ck = sorted(glob.glob("${REPO}/logs/*/ss2d_v9_last.pt"))
print("  v9 last ckpt:", len(ck), "개")
for c in ck: print("   ", c, f"{os.path.getsize(c)/1e6:.0f}MB")
if ck:
    import torch
    s = torch.load(ck[-1], map_location='cpu')
    print("  ✓ ckpt 로드 OK — epoch:", s.get('epoch'), "| keys:", len(s))
PY
H5=$(ls ${REPO}/../fastmri_data_nvme/**/*.h5 2>/dev/null | head -1)
[ -z "$H5" ] && H5=$(find /home/snorlax/shared -maxdepth 4 -name '*.h5' 2>/dev/null | head -1)
if [ -n "$H5" ]; then
  python -c "
import h5py,sys
with h5py.File('$H5','r') as f: print('  ✓ h5 읽기 OK:', list(f.keys())[:4])
" || ng "h5 읽기 실패"
else echo "  ⚠ h5 샘플을 못 찾음 (경로 확인 필요)"; fi

echo
echo "════════ [5] 자원 설정 ════════"
SHM=$(df -h /dev/shm | tail -1 | awk '{print $2}'); echo "  /dev/shm: $SHM"
[ "${SHM%G}" -ge 64 ] 2>/dev/null && ok "shm 충분 (DataLoader num_workers>0 안전)" || ng "shm 부족 — bus error 위험"
echo "  memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
echo "  CPU: $(nproc) cores"

echo
echo "════════ [6] 실험 추적 / git ════════"
[ -f /root/.netrc ] && ok "wandb 인증(.netrc)" || ng "wandb .netrc 없음"
[ -f /root/.ssh/id_ed25519 ] && ok "GitHub SSH 키" || ng "SSH 키 없음"
git -C "${REPO}" status --short >/dev/null 2>&1 && ok "git 저장소 정상" || ng "git 오류"

echo
if [ $FAIL -eq 0 ]; then
  echo "══════ 전부 통과. 학습 재개 가능 ══════"
  echo "  radapt 재개:  bash v9_mamba_radapt/runs/post_reboot_rearm.sh"
else
  echo "══════ 실패 ${FAIL}건 — 위 ✗ 항목 해결 후 재실행 ══════"
fi
exit $FAIL
