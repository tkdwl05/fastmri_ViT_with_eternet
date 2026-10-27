#!/bin/bash
# RunPod Pod 학습 환경 — 로컬 Docker mri:v1(infra/docker/Dockerfile)과 같은 핵심 버전: python 3.10 / torch 2.3.1+cu121 / numpy 1.26.4 /
# mamba_ssm 2.2.2·causal_conv1d 1.4.0 prebuilt wheel / setuptools 69.5.1 (triton import 에 필요). venv = /opt/mri310 (컨테이너 디스크).
set -eu
export PIP_CACHE_DIR=/root/.cache/pip UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
cd /root
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
rm -rf /opt/mri310; uv venv -p 3.10 /opt/mri310
. /opt/mri310/bin/activate
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy==1.26.4 einops==0.8.1 h5py==3.16.0 scipy==1.15.3 pandas==2.3.3 scikit-image==0.25.2 matplotlib==3.10.7 \
  wandb==0.27.0 transformers==4.35.2 ninja==1.13.0 psutil==5.9.0 tqdm==4.66.4 pytz==2024.1 packaging setuptools==69.5.1
uv pip install --no-deps fastmri==0.3.0
uv pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python - <<'PY'
import torch, numpy, mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'numpy', numpy.__version__, 'mamba', mamba_ssm.__version__, torch.cuda.get_device_name(0))
u=torch.randn(2,64,384,device='cuda'); y=selective_scan_fn(u,torch.rand_like(u),-torch.rand(64,16,device='cuda'),torch.randn(2,16,384,device='cuda'),torch.randn(2,16,384,device='cuda'))
print('selective_scan ok', tuple(y.shape))
PY
echo SETUP_DONE
