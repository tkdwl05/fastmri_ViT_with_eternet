"""
v7_titan VRAM smoke test — BS auto-fallback (4 → 2 → 1)

**풀 train-step** 측정: forward + (masked L1 + (1-SSIM)) loss + backward
+ unscale + clip_grad_norm + optimizer.step + peak VRAM. OOM 시 BS 를 줄여 재시도.
capacity (ViT depth, SS2D dim, grad ckpt 등) 는 일체 건드리지 않음.

⚠️ 2026-05-31: loss 를 학습 스크립트와 동일한 masked L1 + SSIM-loss 로 교체.
   이전엔 단순 l1_loss 만 써서 SSIM-loss 의 backward 그래프(+conv 활성화 ~2GiB)를
   누락 → BS8 을 OK 로 오판(실제 학습은 BS8 backward 에서 OOM)했었음 (E4 거짓양성 수정).

성공한 BS 를 다음 두 파일에 기록 (chain 스크립트가 읽어서 SMOKE_BS env 로 주입):
  v7_titan/runs/smoke_ss2d_bs.txt
  v7_titan/runs/smoke_eter_bs.txt
"""

import os
import sys
import gc
import time
import traceback

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'tools'))


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_ss2d_model():
    from myConfig_choh_SS2D_model_v7_titan import (
        IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
        NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
        NUM_VIT_ENCODER_HEAD,   NUM_VIT_ENCODER_MLP_SIZE,
        NUM_SS2D_D_INNER, NUM_SS2D_D_STATE, NUM_SS2D_OUT_CH,
        NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        NUM_VIT_DECODER_DIM,  NUM_VIT_DECODER_DIM_HEAD,
        NUM_VIT_DECODER_DEPTH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        DROPOUT, DC_K_SCALE_RATIO, DC_INIT_ALPHA,
    )
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT

    device = torch.device('cuda')
    vit = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    ).to(device)
    model = choh_Decoder_SS2D_ViT(
        encoder=vit,
        ss2d_d_inner=NUM_SS2D_D_INNER, ss2d_d_state=NUM_SS2D_D_STATE,
        ss2d_out_ch=NUM_SS2D_OUT_CH,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
    ).to(device)
    return model, IMAGE_SIZE, INPUT_CHANNELS, 'ss2d'


def build_eter_model():
    from myConfig_choh_ETER_model_v7_titan import (
        IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
        NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
        NUM_VIT_ENCODER_HEAD,   NUM_VIT_ENCODER_MLP_SIZE,
        NUM_ETER_HORI_HIDDEN, NUM_ETER_VERT_HIDDEN,
        ETER_UNET_DEPTH, ETER_UNET_WF,
        NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        NUM_VIT_DECODER_DIM,  NUM_VIT_DECODER_DIM_HEAD,
        NUM_VIT_DECODER_DEPTH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        DROPOUT,
    )
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_ETER_ViT_v7_titan import choh_Decoder3_ETER_v7_titan

    device = torch.device('cuda')
    vit = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    ).to(device)
    model = choh_Decoder3_ETER_v7_titan(
        encoder=vit,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        unet_depth=ETER_UNET_DEPTH, unet_wf=ETER_UNET_WF,
    ).to(device)
    return model, IMAGE_SIZE, INPUT_CHANNELS, 'eter'


def try_one_bs(build_fn, bs: int):
    """주어진 BS 로 1회 forward + backward. 성공 시 peak VRAM 반환, OOM 시 None."""
    _free_cuda()
    try:
        model, image_size, in_ch, kind = build_fn()
        device = next(model.parameters()).device
        model.train()
        H, W = image_size

        from u_choh_SSIM import SSIM
        criterion_ssim_loss = SSIM().to(device)
        LAMBDA_SSIM = 1.0   # = LAMBDA_SSIM_PER_PIXEL (양 config 동일)

        data_in     = torch.randn(bs, in_ch, H, W, device=device, dtype=torch.float32)
        data_in_img = torch.randn(bs, in_ch, H, W, device=device, dtype=torch.float32)
        data_ref    = torch.randn(bs, 1,     H, W, device=device, dtype=torch.float32).abs()
        mask        = torch.zeros(bs, 1, H, W, device=device, dtype=torch.float32)
        mask[..., ::4] = 1.0
        sens        = torch.randn(bs, in_ch, H, W, device=device, dtype=torch.float32)
        # brain_mask: 학습 loss 와 동일하게 마스킹 경로를 태우기 위한 합성 마스크 (중앙 영역)
        brain_mask  = torch.zeros(bs, 1, H, W, device=device, dtype=torch.float32)
        brain_mask[..., H//5:4*H//5, W//5:4*W//5] = 1.0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        with torch.amp.autocast('cuda'):
            if kind == 'ss2d':
                out = model(data_in_img, data_in, mask, sens)
            else:
                out = model(data_in_img, data_in)
            # 학습 스크립트와 동일한 loss 경로 (masked L1 + (1 - masked SSIM))
            out_fp    = out.float()
            m_sum     = brain_mask.sum().clamp(min=1.0)
            loss_l1   = ((out_fp - data_ref).abs() * brain_mask).sum() / m_sum
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref, mask=brain_mask)
            loss      = loss_l1 + LAMBDA_SSIM * loss_ssim

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if 'out of memory' in msg:
            _free_cuda()
            return None
        traceback.print_exc()
        raise
    _free_cuda()
    return peak_gb


def smoke_for(build_fn, label: str, bs_candidates=(4, 2, 1)):
    print(f'\n========== smoke: {label} ==========', flush=True)
    for bs in bs_candidates:
        print(f'  [try] BS={bs} ...', flush=True)
        t0 = time.time()
        try:
            peak = try_one_bs(build_fn, bs)
        except Exception as e:
            print(f'  [error] {type(e).__name__}: {e}')
            return None
        dt = time.time() - t0
        if peak is None:
            print(f'  [OOM] BS={bs} 실패 ({dt:.1f}s)')
            continue
        print(f'  [OK]  BS={bs}  peak VRAM = {peak:.2f} GB  ({dt:.1f}s)')
        return bs, peak
    print(f'  [FAIL] {label}: BS=1 까지도 OOM. capacity 변경 필요 (사용자 확인 요함)')
    return None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU 필수')

    runs_dir = os.path.join(_HERE, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')

    bs_env = os.environ.get('SMOKE_BS_LIST')
    if bs_env:
        bs_list = tuple(int(b) for b in bs_env.split(','))
    else:
        bs_list = (4, 2, 1)

    ss2d_res = smoke_for(build_ss2d_model, 'SS2D-ViT v7_titan', bs_list)
    if ss2d_res is None:
        sys.exit(2)
    ss2d_bs, ss2d_peak = ss2d_res
    with open(os.path.join(runs_dir, 'smoke_ss2d_bs.txt'), 'w') as f:
        f.write(f'{ss2d_bs}\n')

    eter_res = smoke_for(build_eter_model, 'ETER-ViT v7_titan', bs_list)
    if eter_res is None:
        sys.exit(3)
    eter_bs, eter_peak = eter_res
    with open(os.path.join(runs_dir, 'smoke_eter_bs.txt'), 'w') as f:
        f.write(f'{eter_bs}\n')

    print('\n========== smoke summary ==========')
    print(f'  SS2D BS={ss2d_bs}  peak={ss2d_peak:.2f} GB')
    print(f'  ETER BS={eter_bs}  peak={eter_peak:.2f} GB')
    print(f'  → SMOKE_BS_SS2D={ss2d_bs}  SMOKE_BS_ETER={eter_bs}')


if __name__ == '__main__':
    main()
