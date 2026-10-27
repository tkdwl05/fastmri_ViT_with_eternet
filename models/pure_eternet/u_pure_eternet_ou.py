"""
원래 U-Net(OU) 판 세 모델 (2026-10-26) — 계획 paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md §1.

| 모델            | 클래스          | 정제망 입력                                          | 정제망           |
|-----------------|-----------------|------------------------------------------------------|------------------|
| U-Net only (OU) | PureETER_UNETOU | 0 20 채널 + zero-filled 코일별 복소 영상 32 채널 = 52 | OriginalUNet(52) |
| d-ETER (OU)     | PureETER_GRUOU  | bi-GRU 출력 20 채널 + zero-filled 32 채널 = 52        | OriginalUNet(52) |
| o-ETER (OU)     | PureETER_GRUOOU | bi-GRU 출력 20 채널만                                 | OriginalUNet(20) |

bi-GRU = 지금 ETER-net 의 시퀀스 모듈 그대로(H = 10, flatten-reshape 스캔). u_pure_eternet_gru.PureETER_GRU(무수정)를 상속해
gru_h·gru_v 를 같은 인자·같은 순서로 만들고 _seq 를 그대로 물려받는다(재정의하지 않음). 상속한 __init__ 이 만드는 DFU(self.unet)는
곧바로 OriginalUNet 으로 바꾼다(버린 DFU 는 state_dict 에 남지 않는다; 그 생성도 난수를 쓰지만 결정적이다). 그래서 seed_all 직후
만들면 gru_h·gru_v 초기값이 같은 시드의 ETER-net(DFU) 과 비트까지 같다. gru_h 의 fp32 계산은 트레이너
(main_train_pure_v8fix_ampguard_ou.py → gru_precision.apply)와 평가 진입점(gru_precision.install_class_patch — 이 하위 클래스의
__init__ 도 PureETER_GRU.__init__ 을 부르므로 클래스 패치가 그대로 걸린다)이 붙인다.
  - d-ETER (OU): forward = unet(cat(_seq(x_ksp), x_img)) — 지금 ETER-net(PureETER_GRU.forward, use_dc=False)과 같은 배선.
  - o-ETER (OU): forward = unet(_seq(x_ksp)) — 2020 원 ETER-net 배선(k-space 만 입력). x_img 는 읽지 않는다.
  - U-Net only (OU): 시퀀스 모듈이 없어 nn.Module 에서 바로 만든다. 0 채널은 출력에 영향이 없고(원래 U-Net 은 입력을 첫 합성곱
    down_path.0.block.0 에서만 읽는다 — 입력 재연결 없음), 그 가중치 slice [:, :20] 의 기울기는 정확히 0 이다.
    52 채널 형태를 쓰는 이유: 같은 시드의 d-ETER (OU) 와 U-Net 모양·초기값을 같게 하려고(계획 §1, 트레이너 SAME_UNET_INIT).
세 클래스 모두 forward 를 자기 클래스에 정의한다(평가 래퍼가 클래스 속성 forward 를 감싸므로 — eval_v8fix_official_fb.py _wrap_method).

forward 시그니처 = 다른 v8 모델과 같다: forward(x_img, x_ksp, mask=None, sens=None). mask·sens 는 쓰지 않는다(DC 없음 — 계획은
no-DC 만이라 use_dc 인자는 두지 않는다). 출력 (B, 1, H, W). 교수님·프로젝트 원본 파일은 고치지 않는다.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))

from unet_original import OriginalUNet          # noqa: E402
from u_pure_eternet_gru import PureETER_GRU     # noqa: E402  (교수님 원본 ETER-net 의 bi-GRU 재현 — 무수정)


class PureETER_UNETOU(nn.Module):
    """U-Net only (OU): OriginalUNet(cat(0 채널 2·H2, x_img)) — 시퀀스 모듈 없음."""

    def __init__(self, *, n_coil: int = 16, n_hidden_2: int = 10, unet_depth: int = 5, unet_wf: int = 6):
        super().__init__()
        self.seq_out_ch = 2 * n_hidden_2            # 시퀀스 출력 자리 (= 20) — 0 으로 채움
        in_ch = self.seq_out_ch + 2 * n_coil        # 52
        self.unet = OriginalUNet(in_channels=in_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        print(f"   'PureETER_UNETOU' (U-Net only (OU): zero_ch={self.seq_out_ch} + x_img {2 * n_coil} → "
              f"OriginalUNet(in={in_ch}, depth={unet_depth}, wf={unet_wf}))")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        B, _, H, W = x_img.shape
        zeros = torch.zeros((B, self.seq_out_ch, H, W), dtype=x_img.dtype, device=x_img.device)
        return self.unet(torch.cat((zeros, x_img), dim=1))   # 채널 순서 = 다른 모델과 같음 (시퀀스 자리, zero-filled)


class PureETER_GRUOU(PureETER_GRU):
    """d-ETER (OU): PureETER_GRU 의 bi-GRU(_seq) + OriginalUNet(2·H2 + 2·coil)."""

    def __init__(self, *, dim: int = 384, n_coil: int = 16, n_hidden_1: int = 10, n_hidden_2: int = 10,
                 unet_depth: int = 5, unet_wf: int = 6):
        super().__init__(dim=dim, n_coil=n_coil, n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
                         unet_depth=unet_depth, unet_wf=unet_wf, use_dc=False)
        self.seq_out_ch = 2 * n_hidden_2            # 20
        in_ch = self.seq_out_ch + 2 * n_coil        # 52
        self.unet = OriginalUNet(in_channels=in_ch, n_classes=1, depth=unet_depth, wf=unet_wf)   # DFU 교체
        print(f"   'PureETER_GRUOU' (d-ETER (OU): bi-GRU {self.seq_out_ch} + x_img {2 * n_coil} → "
              f"OriginalUNet(in={in_ch}, depth={unet_depth}, wf={unet_wf}); 위 줄의 DFU 는 교체됨)")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        out_v = self._seq(x_ksp)                    # (B, 2·H2, H, W) — PureETER_GRU._seq 그대로
        return self.unet(torch.cat((out_v, x_img), dim=1))


class PureETER_GRUOOU(PureETER_GRU):
    """o-ETER (OU): OriginalUNet(2·H2)(bi-GRU 출력) — x_img 는 읽지 않는다."""

    def __init__(self, *, dim: int = 384, n_coil: int = 16, n_hidden_1: int = 10, n_hidden_2: int = 10,
                 unet_depth: int = 5, unet_wf: int = 6):
        super().__init__(dim=dim, n_coil=n_coil, n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
                         unet_depth=unet_depth, unet_wf=unet_wf, use_dc=False)
        self.seq_out_ch = 2 * n_hidden_2            # 20
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)   # DFU 교체
        print(f"   'PureETER_GRUOOU' (o-ETER (OU): bi-GRU {self.seq_out_ch} 만 → "
              f"OriginalUNet(in={self.seq_out_ch}, depth={unet_depth}, wf={unet_wf}); 위 줄의 DFU 는 교체됨)")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self._seq(x_ksp))
