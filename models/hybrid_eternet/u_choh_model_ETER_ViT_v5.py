"""
ETER-ViT 모델 v5: v4 디코더 클래스를 상속해 decoder Transformer 에 dropout 을 주입.

v4 [u_choh_model_ETER_ViT.py:86] 의 `Transformer(...)` 호출이 dropout 인자를 넘기지
않아 사실상 dropout=0 이었다. v5 는 동일 architecture 의 Transformer 를 dropout 적용 버전으로
교체하기만 한다. 파라미터 수, 입출력 shape, GRU 구조 모두 v4 와 완전히 동일.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from u_choh_model_ETER_ViT import (
    choh_Decoder3_ETER_skip_up_tail,
    Transformer,
)


class choh_Decoder3_ETER_v5(choh_Decoder3_ETER_skip_up_tail):
    """ETER decoder + dropout-enabled Transformer.

    추가 인자:
      dropout: decoder Transformer 의 attention/MLP/FF dropout 비율
    """

    def __init__(
        self,
        *,
        dropout: float = 0.0,
        decoder_dim,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        decoder_dim_mlp_hidden=3072,
        **kwargs,
    ):
        super().__init__(
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            decoder_dim_head=decoder_dim_head,
            decoder_dim_mlp_hidden=decoder_dim_mlp_hidden,
            **kwargs,
        )
        # decoder 를 dropout 적용 버전으로 교체 (architecture 동일)
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim_mlp_hidden,
            dropout=dropout,
        )
        print(f"   'choh_Decoder3_ETER_v5  @u_choh_model_ETER_ViT_v5'  (decoder dropout={dropout})")
