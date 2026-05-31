"""VGG16 perceptual feature loss for grayscale MRI 320x320.

사용:
  perc = VGGPerceptualLoss(device=device, layers=('relu1_2', 'relu2_2', 'relu3_3'))
  loss = perc(pred_1ch_amp, gt_1ch_amp)  # 둘 다 (B,1,H,W) float, amplitude

내부적으로:
  1) per-slice max-normalize → [0,1]
  2) 1ch → 3ch replicate
  3) ImageNet 정규화 (torchvision 표준 mean/std)
  4) VGG16 features (ImageNet pretrained) 추출 후 L1 거리 평균

VGG 는 freeze, eval mode 로 고정.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


_LAYER_INDEX = {
    'relu1_2': 3,     # conv1_2 + ReLU
    'relu2_2': 8,     # conv2_2 + ReLU
    'relu3_3': 15,    # conv3_3 + ReLU
    'relu4_3': 22,    # conv4_3 + ReLU
}


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layers=('relu1_2', 'relu2_2', 'relu3_3')):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.cuts = sorted(_LAYER_INDEX[l] for l in layers)

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def _prep(self, x):
        """(B,1,H,W) amplitude → (B,3,H,W) ImageNet-normalized."""
        # per-slice max-normalize (≥1e-8 stabilization)
        B = x.size(0)
        m = x.view(B, -1).amax(dim=1).clamp_min(1e-8).view(B, 1, 1, 1)
        x = (x / m).clamp(0.0, 1.0)
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        p = self._prep(pred)
        t = self._prep(target)
        loss = 0.0
        last_cut = 0
        x_p, x_t = p, t
        for cut in self.cuts:
            x_p = self.vgg[last_cut:cut + 1](x_p)
            x_t = self.vgg[last_cut:cut + 1](x_t)
            loss = loss + (x_p - x_t).abs().mean()
            last_cut = cut + 1
        return loss / len(self.cuts)
