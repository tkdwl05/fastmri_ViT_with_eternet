"""ETER-net(GRU) 의 k-space 쪽 GRU(gru_h)를 autocast 밖에서 fp32 로 계산한다 (2026-10-10).

배경: RunPod pod3 의 v8fix 50 epoch GRU 런이 epoch 1 batch ≈3,535 에서 두 번 연속(같은 시드·같은 데이터 순서) 무너졌다.
그 직전 상태에서 같은 batch 를 정밀도별로 forward/backward 해 보니(진단 결과는 docs·log 참고)
  - gru_h(입력 = k-space 12,288 차원) 기울기: fp32 기준 노름 17.6~3,450 인데 fp16 은 4 batch 중 2 개가 NaN(배율 1024),
    나머지는 최대 40 배 틀렸다. bf16 은 최대 25 배, fp32+TF32 는 최대 2 배 틀렸다.
  - gru_h 만 fp32 로 계산하면 기준과 1% 안에서 같았다. gru_v·U-Net 의 fp16 기울기는 정확했다.
  - 로컬 5 epoch 점검의 정상 GRU(epoch 5)에서는 fp16 도 fp32 와 2% 안에서 같았다 — 기울기가 커지는 구간에서만 fp16 이 틀린다.
교수님 원본 ETER-net 학습 코드는 AMP 를 쓰지 않는다(fp32). 그래서 gru_h 만 원본과 같은 fp32 로 계산하고 나머지(gru_v·U-Net)는
다른 모델과 같은 fp16 autocast 를 유지한다. state_dict 키는 바뀌지 않는다(모듈 인스턴스의 forward 만 감싼다).
"""
import torch


def force_fp32(module):
    """nn.RNN 계열 모듈의 forward 를 autocast 밖 fp32 로 감싼다 (입력·초기 상태를 float 로 올림)."""
    orig = type(module).forward

    def forward(input, hx=None):
        with torch.autocast('cuda', enabled=False):
            return orig(module, input.float(), None if hx is None else hx.float())

    module.forward = forward
    return module


def apply(model, names=('gru_h',)):
    for n in names:
        if n:
            force_fp32(getattr(model, n))
    return model


def install_class_patch(names=('gru_h',)):
    """모델을 직접 만드는 스크립트(eval_v8fix.py 등)용: PureETER_GRU 를 만들 때마다 apply 한다."""
    import u_pure_eternet_gru as M
    if getattr(M.PureETER_GRU, '_fp32_patched', None) is not None:
        return
    init = M.PureETER_GRU.__init__

    def __init__(self, *a, **k):
        init(self, *a, **k)
        apply(self, names)

    M.PureETER_GRU.__init__ = __init__
    M.PureETER_GRU._fp32_patched = tuple(names)
