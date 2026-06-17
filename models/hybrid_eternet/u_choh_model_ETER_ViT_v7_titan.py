"""
ETER-ViT 모델 v7_titan: v5 디코더 클래스를 상속해 최종 합성을 U-Net 으로 교체.

v5 [u_choh_model_ETER_ViT_v5.py] 의 choh_Decoder3_ETER_v5 는 `self.last` 가
RefinementBlock(3×ResBlock) 이다. v7_titan 은 동일 architecture 의 입력 채널 수를
유지한 채 `self.last` 만 UNet_choh_skip(depth=3, wf=6) 으로 교체한다. 원본
ETER-Net (ETER_hybrid_GRU_DFU) 의 U-Net 후처리 복원이 목적.

원본 UNet_choh_skip 의 forward 에서 `self.last = Conv2d(prev_channels + n_hidden*2, ...)`
이 `out = cat([blocks[0], x_up], 1)` 후에 적용되므로:
    n_hidden = in_channels // 2  (= num_ch_last // 2)
를 만족시켜야 마지막 conv 의 입력 차원이 일치한다.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5
from myUNet_DF import UNet_choh_skip


class choh_Decoder3_ETER_v7_titan(choh_Decoder3_ETER_v5):
    """ETER decoder + dropout-enabled Transformer + U-Net 후처리.

    추가 인자:
      unet_depth: U-Net depth (원본 ETER-Net 3~5)
      unet_wf:    U-Net 첫 레이어 채널 지수 (2**wf)
    """

    def __init__(
        self,
        *,
        unet_depth: int = 3,
        unet_wf: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # super().__init__ 후 self.last (RefinementBlock) 가 존재.
        # in_ch 정보를 RefinementBlock 의 head Conv 에서 회수.
        in_ch = self.last.head.in_channels
        assert in_ch % 2 == 0, (
            f"U-Net 후처리 입력 채널 ({in_ch}) 이 홀수여서 n_hidden 산정 불가. "
            "eter_n_vert_hidden / decoder_out_ch_up_tail 조합을 짝수로 맞출 것."
        )
        n_hidden = in_ch // 2

        # RefinementBlock 자리에 UNet_choh_skip 교체
        self.last = UNet_choh_skip(
            in_channels=in_ch,
            n_classes=1,
            depth=unet_depth,
            wf=unet_wf,
            padding=True,
            batch_norm=False,
            up_mode='upconv',
            n_hidden=n_hidden,
        )
        print(f"   'choh_Decoder3_ETER_v7_titan  @u_choh_model_ETER_ViT_v7_titan'  "
              f"(self.last = UNet_choh_skip(in_ch={in_ch}, depth={unet_depth}, "
              f"wf={unet_wf}, n_hidden={n_hidden}))")
