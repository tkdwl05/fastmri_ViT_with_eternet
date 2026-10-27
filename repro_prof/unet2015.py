"""원래 U-Net (Ronneberger, Fischer, Brox, MICCAI 2015, arXiv:1505.04597) — 재구성 회귀용 판.

논문 구조 그대로인 것
- 수준마다 3×3 conv + ReLU 를 두 번 한다.
- 2×2 max pooling(stride 2)으로 내려가고, 내려갈 때마다 채널을 2배로 한다(논문 64-128-256-512-1024).
- 올라갈 때 2×2 up-convolution(ConvTranspose2d, 채널 절반) 뒤 같은 수준의 인코더 특징과 이어 붙이고(skip concatenation),
  3×3 conv + ReLU 를 두 번 한다.
- 마지막은 1×1 conv. BatchNorm 없음.
- 교수님 DFU(UNet_choh_skip)와 다른 점: 인코더 특징을 빼는 dual-frame 잔차가 없고, 마지막에 원입력을 이어 붙이지 않으며,
  avg pooling 대신 max pooling 이다(myUNet_DF.py:61·125-128·193 과 대조).

논문과 다르게 둔 것(옵션, 모두 '추정/설계'로 config 에 표시)
- padding=True: 3×3 conv 에 zero padding 1 → 출력이 입력과 같은 384×384. 논문은 unpadded conv + crop(출력이 작아짐).
  padding=False 도 지원한다(그때는 skip 특징을 가운데 crop 하고 출력이 작아진다 — 학습 루프에는 쓰지 않는다).
- out_ch=1(회귀). 논문은 분할(2클래스 softmax).
- init='pytorch_default'(교수님 DFU 와 같은 PyTorch 기본 초기화) 또는 'he_normal'(논문: 표준편차 sqrt(2/N), N = 들어오는 노드 수).
- dropout: 논문은 수축 경로 끝에 drop-out 을 두었으나 확률을 적지 않았다. 기본 0(끔), 켜면 가장 깊은 수준 뒤 Dropout2d.
- depth = 해상도 수준 수(논문 5), wf = log2(첫 층 채널)(논문 64 → 6).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super().__init__()
        p = int(bool(padding))
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=p), nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=p), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.block(x)


def _center_crop(t, hw):
    h, w = t.shape[-2:]
    dy, dx = (h - hw[0]) // 2, (w - hw[1]) // 2
    return t[..., dy:dy + hw[0], dx:dx + hw[1]]


class UNet2015(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, depth=5, wf=6, padding=True, dropout=0.0, init='pytorch_default'):
        super().__init__()
        assert depth >= 2
        self.depth, self.wf, self.padding = depth, wf, padding
        self.down = nn.ModuleList()
        prev = in_ch
        for i in range(depth):
            self.down.append(_DoubleConv(prev, 2 ** (wf + i), padding))
            prev = 2 ** (wf + i)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.upconv = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.upconv.append(nn.ConvTranspose2d(2 ** (wf + i + 1), 2 ** (wf + i), kernel_size=2, stride=2))
            self.up.append(_DoubleConv(2 ** (wf + i + 1), 2 ** (wf + i), padding))
        self.last = nn.Conv2d(2 ** wf, out_ch, kernel_size=1)
        if init == 'he_normal':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        elif init != 'pytorch_default':
            raise ValueError(init)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < self.depth - 1:
                skips.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop(x)
        for upc, blk in zip(self.upconv, self.up):
            x = upc(x)
            s = _center_crop(skips.pop(), x.shape[-2:])
            x = blk(torch.cat([s, x], dim=1))
        return self.last(x)
