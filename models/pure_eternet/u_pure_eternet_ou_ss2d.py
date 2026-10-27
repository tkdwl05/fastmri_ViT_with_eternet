"""
원래 U-Net(OU) 판 SS2D 모델 (2026-10-26) — 계획 paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan_amend1_ss2d.md
(사용자 결정 ARS ledger seq 257–258: SS2D 도 원래 U-Net 실험에 넣고, 출력이 영상 정보를 담도록 푸리에 출력 헤드를 단다).

| 모델                | 클래스            | 시퀀스 모듈                         | 정제망 입력                          | 정제망           |
|---------------------|-------------------|-------------------------------------|--------------------------------------|------------------|
| d-SS2D (OU)         | PureETER_SS2DOU   | SS2D (SS2D(controlled) 와 같음)     | SS2D 출력 20 + zero-filled 32 = 52    | OriginalUNet(52) |
| o-SS2D (OU)         | PureETER_SS2DOOU  | SS2D (같음)                          | SS2D 출력 20 만                       | OriginalUNet(20) |
| o-SS2D-F (OU)       | PureETER_SS2DFOOU | FourierSS2D (SS2D + 푸리에 출력 헤드) | 헤드 출력 20 만 (영상 좌표)          | OriginalUNet(20) |
| o-LIN-F (OU) (대조) | PureETER_LINFOOU  | FourierLinear (헤드의 선형 경로만)   | 헤드 출력 20 만 (영상 좌표)          | OriginalUNet(20) |

- d-SS2D (OU) 는 d-ETER (OU)(u_pure_eternet_ou.PureETER_GRUOU)와, o-SS2D-F (OU) 는 o-ETER (OU)(PureETER_GRUOOU)와 같은 배선이다.
  o-SS2D (OU) 와 o-LIN-F (OU) 는 짧은 비교 시험(계획 보충 1 §3)에만 쓴다.
- SS2D 는 models/mamba_eternet/ss2d.py 의 SS2D(프로젝트 공유 원본, 무수정)를 PureETER_SS2D(u_pure_eternet_ss2d.py)와 같은 인자
  (c_in = 2·n_coil = 32, d_inner, d_state, out_ch = 2·H2 = 20)로 만든다. 만드는 순서도 같다(SS2D 먼저, 그다음 정제망) — 그래서
  seed_all 직후 만들면 SS2D 의 초기값이 같은 시드의 SS2D(controlled) 와 비트까지 같다(FourierSS2D 는 아래 '초기값').
- 푸리에 출력 헤드 (FourierSS2D, 이 파일에서 SS2D 를 상속 — ss2d.py 는 고치지 않는다):
    k = res(x_ksp) + E(x_ksp) ⊙ SS2D(x_ksp)           # (B, 20, H, W), k-space 좌표 — 20 채널 = 복소 10 채널 (real/imag 교번)
    y = ifft2c(k)                                     # 채널 쌍 (2j, 2j+1) = (real, imag) 을 복소수로 보고 2D 역푸리에 변환
    반환 y (B, 20, H, W), 영상 좌표 (real/imag 교번)
  - ifft2c 는 로더(dataloaders/dataloader_h5_v5.py:30 ifft2c)와 같은 규약: ifftshift → ifft2(norm='ortho') → fftshift, 마지막 두 축.
    정규화 뒤(smoke_recipe_fix.normalize_inputs: k-space ×K_RATIO/s, 영상 /s)에도 x_img = ifft2c(x_ksp) 이므로 같은 단위가 된다.
  - res = 1×1 합성곱 32 → 20 (bias 없음) — 측정한 k-space 를 섞어 더하는 세기 보존 경로. 초기값 = 코일 0–9 의 (real, imag) 를
    그대로 고르는 행렬 → 처음에는 y = 코일 0–9 의 zero-filled 영상(x_img[:, :20])과 같다.
  - E = 국소 k-space 크기 (kspace_envelope): 측정 k-space 의 코일 RMS 크기 |k|(u, v) 를 마스크 축(마지막 축 — 로더의 R4 마스크는
    마지막 축의 줄을 고른다)으로 폭 7 이동 평균한 것(B, 1, H, W). 등간격 R4 에서 빠진 줄은 ±2 안에 측정 줄이 있으므로 빠진 줄도 이웃
    측정 줄의 크기를 받는다. SS2D 는 입력 단계 norm_in(위치마다 LayerNorm)으로 위치별 세기를 잃으므로, SS2D 경로의 출력은 그 위치의
    국소 크기에 대한 상대값으로 쓴다(세기 보존의 둘째 부분). E 는 파라미터가 없고 입력만의 함수다.
  - 초기값: SS2D 부분은 SS2D(controlled) 와 같고(같은 생성 순서), out_proj(마지막 1×1 합성곱)만 0 (weight·bias) — SS2D 경로는 0 에서
    시작해 학습으로 자란다(10-11 진단: 큰 초기 출력이 U-Net 학습을 해쳤다). res 는 위 선택 행렬. res 를 만들 때 쓰는 난수는 SS2D 다음,
    정제망 앞이다.
  - SS2D 본체는 autocast 그대로 돌고(SS2D(controlled) 와 같음), 선형 경로 res 와 변환은 autocast 를 끄고 fp32 로 한다(384 는 2 의
    거듭제곱이 아니라 cuFFT 반정밀도를 쓸 수 없고, 선형 경로의 fp16 넘침도 막는다). 반환은 float32 — 정제망의 합성곱이 autocast 에
    따라 다시 바꾼다. 평가 래퍼(eval_v8fix_fb.py)의 시퀀스 단위는 SS2D.forward(본체)다 — 호출 시험 이름은 run_validation 의 현재
    줄로 정하므로 헤드가 한 겹 감싸도 그대로 'seq' 로 잡힌다.
  - 파라미터: SS2D(controlled) 의 SS2D 와 같고 res 640 개(32·20)가 더해진다. 변환에는 파라미터가 없다.
- 기능 시험(smoke_recipe_fix.run_validation, seq 'ss2d')은 model.ss2d(x_ksp) 를 시퀀스 출력으로 쓴다. 그래서 네 클래스 모두 시퀀스
  모듈을 self.ss2d 에 둔다(FourierSS2D·FourierLinear 는 헤드까지 포함한 출력을 낸다). o- 모델은 recipe_ou_ss2d 의 대리 객체가 정제망에
  앞 20 채널만 넘긴다.
- 네 클래스 모두 forward 를 자기 클래스에 정의한다(평가 래퍼가 클래스 속성 forward 를 감싸므로). forward 시그니처 =
  forward(x_img, x_ksp, mask=None, sens=None) (mask·sens 는 쓰지 않음, DC 없음). 출력 (B, 1, H, W).
교수님·프로젝트 원본 파일은 고치지 않는다.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'mamba_eternet'))

from unet_original import OriginalUNet          # noqa: E402
from ss2d import SS2D                           # noqa: E402  (프로젝트 공유 원본 — 무수정)


def ifft2c_ri(k):
    """k (B, 2·n, H, W) real/imag 교번 → ifft2c (로더와 같은 규약, fp32) → (B, 2·n, H, W) real/imag 교번, float32."""
    k = k.float()
    z = torch.complex(k[:, 0::2], k[:, 1::2])
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    z = torch.fft.ifft2(z, dim=(-2, -1), norm='ortho')
    z = torch.fft.fftshift(z, dim=(-2, -1))
    return torch.stack((z.real, z.imag), dim=2).reshape(k.shape)


ENVELOPE_WIDTH = 7


def kspace_envelope(x, width=ENVELOPE_WIDTH):
    """x (B, 2·n_coil, H, W) real/imag 교번 k-space → 국소 크기 E (B, 1, H, W), float32: 코일 RMS 크기
    sqrt(mean_c |z_c|²) 를 마지막 축(마스크 축)으로 폭 width 이동 평균 (가장자리는 안쪽 값만 평균)."""
    x = x.float()
    a = torch.sqrt((x * x).sum(dim=1, keepdim=True) / (x.shape[1] // 2))
    return F.avg_pool2d(a, kernel_size=(1, width), stride=1, padding=(0, width // 2), count_include_pad=False)


def _coil_select_init(conv, out_ch):
    """1×1 합성곱 (out_ch, c_in, 1, 1) 을 앞 out_ch 입력 채널을 그대로 고르는 행렬로 (코일 0..out_ch/2−1 의 real·imag)."""
    with torch.no_grad():
        conv.weight.zero_()
        for j in range(out_ch):
            conv.weight[j, j, 0, 0] = 1.0


class FourierSS2D(SS2D):
    """SS2D + 푸리에 출력 헤드: ifft2c(res(x) + E(x)·SS2D(x)). SS2D 의 하위 모듈 이름(norm_in·in_proj·…·out_proj)은 그대로다."""

    def __init__(self, c_in: int, d_inner: int, d_state: int, out_ch: int):
        super().__init__(c_in=c_in, d_inner=d_inner, d_state=d_state, out_ch=out_ch)
        if out_ch % 2 or out_ch > c_in:
            raise ValueError(f'FourierSS2D: out_ch={out_ch} 는 짝수이고 c_in={c_in} 이하여야 함')
        with torch.no_grad():
            self.out_proj.weight.zero_()
            self.out_proj.bias.zero_()
        self.res = nn.Conv2d(c_in, out_ch, kernel_size=1, bias=False)
        _coil_select_init(self.res, out_ch)

    def forward(self, x):
        k_ss2d = super().forward(x)                  # SS2D 본체 — autocast 그대로 (SS2D(controlled) 와 같음)
        with torch.autocast(device_type=x.device.type, enabled=False):   # 선형 경로·국소 크기·변환은 fp32
            return ifft2c_ri(self.res(x.float()) + kspace_envelope(x) * k_ss2d.float())


class FourierLinear(nn.Module):
    """대조 모델의 시퀀스 자리: FourierSS2D 에서 SS2D 경로를 뺀 것 — ifft2c(res(x)). res 초기값은 FourierSS2D 와 같다."""

    def __init__(self, c_in: int, out_ch: int):
        super().__init__()
        if out_ch % 2 or out_ch > c_in:
            raise ValueError(f'FourierLinear: out_ch={out_ch} 는 짝수이고 c_in={c_in} 이하여야 함')
        self.res = nn.Conv2d(c_in, out_ch, kernel_size=1, bias=False)
        _coil_select_init(self.res, out_ch)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return ifft2c_ri(self.res(x.float()))


class PureETER_SS2DOU(nn.Module):
    """d-SS2D (OU): OriginalUNet(cat(SS2D(x_ksp), x_img)) — SS2D(controlled) 와 같은 배선, 정제망만 원래 U-Net."""

    def __init__(self, *, n_coil: int = 16, n_hidden_2: int = 10, unet_depth: int = 5, unet_wf: int = 6,
                 ss2d_d_inner: int = 128, ss2d_d_state: int = 16):
        super().__init__()
        c_in = 2 * n_coil                            # 32
        self.seq_out_ch = 2 * n_hidden_2             # 20
        self.ss2d = SS2D(c_in=c_in, d_inner=ss2d_d_inner, d_state=ss2d_d_state, out_ch=self.seq_out_ch)
        in_ch = self.seq_out_ch + c_in               # 52
        self.unet = OriginalUNet(in_channels=in_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        print(f"   'PureETER_SS2DOU' (d-SS2D (OU): SS2D {self.seq_out_ch} (d_inner={ss2d_d_inner}, d_state={ss2d_d_state}) "
              f"+ x_img {c_in} → OriginalUNet(in={in_ch}, depth={unet_depth}, wf={unet_wf}))")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(torch.cat((self.ss2d(x_ksp), x_img), dim=1))


class PureETER_SS2DOOU(nn.Module):
    """o-SS2D (OU): OriginalUNet(SS2D(x_ksp)) — x_img 는 읽지 않는다 (짧은 비교 시험 전용).
    SS2D 출력은 channels-last 배치(마지막 rearrange 뒤 1×1 합성곱)라 .contiguous() 로 NCHW 로 바꿔 넘긴다 — 값은 같고, 기능 시험의
    분해 경로(cat 뒤 앞 20 채널, NCHW)와 같은 합성곱 구현을 타게 한다(d-SS2D (OU) 는 cat 이 같은 일을 한다)."""

    def __init__(self, *, n_coil: int = 16, n_hidden_2: int = 10, unet_depth: int = 5, unet_wf: int = 6,
                 ss2d_d_inner: int = 128, ss2d_d_state: int = 16):
        super().__init__()
        c_in = 2 * n_coil
        self.seq_out_ch = 2 * n_hidden_2
        self.ss2d = SS2D(c_in=c_in, d_inner=ss2d_d_inner, d_state=ss2d_d_state, out_ch=self.seq_out_ch)
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        print(f"   'PureETER_SS2DOOU' (o-SS2D (OU): SS2D {self.seq_out_ch} 만 → "
              f"OriginalUNet(in={self.seq_out_ch}, depth={unet_depth}, wf={unet_wf}))")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp).contiguous())


class PureETER_SS2DFOOU(nn.Module):
    """o-SS2D-F (OU): OriginalUNet(FourierSS2D(x_ksp)) — SS2D + 푸리에 출력 헤드, x_img 는 읽지 않는다 (o-ETER (OU) 와 같은 배선)."""

    def __init__(self, *, n_coil: int = 16, n_hidden_2: int = 10, unet_depth: int = 5, unet_wf: int = 6,
                 ss2d_d_inner: int = 128, ss2d_d_state: int = 16):
        super().__init__()
        c_in = 2 * n_coil
        self.seq_out_ch = 2 * n_hidden_2
        self.ss2d = FourierSS2D(c_in=c_in, d_inner=ss2d_d_inner, d_state=ss2d_d_state, out_ch=self.seq_out_ch)
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        print(f"   'PureETER_SS2DFOOU' (o-SS2D-F (OU): ifft2c(SS2D + res) {self.seq_out_ch} 만 → "
              f"OriginalUNet(in={self.seq_out_ch}, depth={unet_depth}, wf={unet_wf}); out_proj 0 초기화, res = 코일 0–"
              f"{self.seq_out_ch // 2 - 1} 선택)")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp))


class PureETER_LINFOOU(nn.Module):
    """o-LIN-F (OU) (대조): OriginalUNet(FourierLinear(x_ksp)) — o-SS2D-F (OU) 에서 SS2D 경로만 뺀 것 (짧은 비교 시험 전용)."""

    def __init__(self, *, n_coil: int = 16, n_hidden_2: int = 10, unet_depth: int = 5, unet_wf: int = 6):
        super().__init__()
        c_in = 2 * n_coil
        self.seq_out_ch = 2 * n_hidden_2
        self.ss2d = FourierLinear(c_in=c_in, out_ch=self.seq_out_ch)
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        print(f"   'PureETER_LINFOOU' (o-LIN-F (OU): ifft2c(res) {self.seq_out_ch} 만 → "
              f"OriginalUNet(in={self.seq_out_ch}, depth={unet_depth}, wf={unet_wf}))")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp))
