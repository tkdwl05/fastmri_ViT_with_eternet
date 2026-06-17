"""
v7 smoke test — 학습 전 GPU 단독 검증.
  - SS2D / ETER v7 모델 인스턴스화 (config import 확인)
  - dataloader 가 첫 샘플 읽기
  - 합성 입력으로 forward 1회 (VRAM 사용량 측정)
  - backward / optimizer step 도 1회

torchrun 없이 single-GPU 로 실행하면 됨. extraction 과 병행해도 충돌 없음 (GPU 단독).
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))


def smoke_ss2d(device):
    import myConfig_choh_SS2D_model_v7 as C
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT

    print('  [SS2D] config OK | BS', C.BATCH_SIZE, '| ENC_HIDDEN', C.NUM_VIT_ENCODER_HIDDEN)
    enc = choh_ViT(
        image_size=C.IMAGE_SIZE, patch_size=C.PATCH_SIZE, num_classes=1000,
        dim=C.NUM_VIT_ENCODER_HIDDEN, depth=C.NUM_VIT_ENCODER_LAYER,
        heads=C.NUM_VIT_ENCODER_HEAD, mlp_dim=C.NUM_VIT_ENCODER_MLP_SIZE,
        channels=C.INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    ).to(device)
    model = choh_Decoder_SS2D_ViT(
        encoder=enc,
        ss2d_d_inner=C.NUM_SS2D_D_INNER, ss2d_d_state=C.NUM_SS2D_D_STATE,
        ss2d_out_ch=C.NUM_SS2D_OUT_CH,
        decoder_dim=C.NUM_VIT_DECODER_DIM, decoder_depth=C.NUM_VIT_DECODER_DEPTH,
        decoder_heads=C.NUM_VIT_DECODER_HEAD, decoder_dim_head=C.NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=C.NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=C.NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=C.NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=C.DROPOUT,
        dc_k_scale_ratio=C.DC_K_SCALE_RATIO,
        dc_init_alpha=C.DC_INIT_ALPHA,
    ).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    print(f'  [SS2D] params {nparam/1e6:.1f}M')

    BS = C.BATCH_SIZE
    H, W = C.IMAGE_SIZE
    CH   = C.INPUT_CHANNELS
    # Real dataloader (dataloader_h5_v5.py) shapes:
    #   in_imgs (B, CH, H, W), in_ksp (B, CH, H, W) — CH = 2*N_COIL (RI interleaved)
    #   mask    (B, 1,  H, W) — broadcast 1D phase-encoding mask
    #   sens    (B, CH, H, W) — RI interleaved sens map
    data_img = torch.randn(BS, CH, H, W, device=device)
    data     = torch.randn(BS, CH, H, W, device=device)
    mask     = torch.ones(BS, 1,  H, W, device=device)
    sens     = torch.randn(BS, CH, H, W, device=device)
    label    = torch.randn(BS, 1,  H, W, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE_ADAM)
    crit = nn.L1Loss()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model.train()
    with torch.amp.autocast('cuda'):
        out = model(data_img, data, mask, sens)
    print(f'  [SS2D] forward OK   out.shape={tuple(out.shape)}   elapsed={time.time()-t0:.2f}s')

    loss = crit(out.float(), label)
    scaler = torch.amp.GradScaler('cuda')
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f'  [SS2D] backward+step OK  loss={loss.item():.4f}  peak VRAM={peak:.2f}GB')

    del model, enc, opt; torch.cuda.empty_cache()


def smoke_eter(device):
    import myConfig_choh_ETER_model_v7 as C
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5

    print('  [ETER] config OK | BS', C.BATCH_SIZE,
          '| GRU h/v', C.NUM_ETER_HORI_HIDDEN, C.NUM_ETER_VERT_HIDDEN)
    enc = choh_ViT(
        image_size=C.IMAGE_SIZE, patch_size=C.PATCH_SIZE, num_classes=1000,
        dim=C.NUM_VIT_ENCODER_HIDDEN, depth=C.NUM_VIT_ENCODER_LAYER,
        heads=C.NUM_VIT_ENCODER_HEAD, mlp_dim=C.NUM_VIT_ENCODER_MLP_SIZE,
        channels=C.INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    ).to(device)
    model = choh_Decoder3_ETER_v5(
        encoder=enc,
        eter_n_hori_hidden=C.NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=C.NUM_ETER_VERT_HIDDEN,
        decoder_dim=C.NUM_VIT_DECODER_DIM, decoder_depth=C.NUM_VIT_DECODER_DEPTH,
        decoder_heads=C.NUM_VIT_DECODER_HEAD, decoder_dim_head=C.NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=C.NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=C.NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=C.NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=C.DROPOUT,
    ).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    print(f'  [ETER] params {nparam/1e6:.1f}M')

    BS = C.BATCH_SIZE
    H, W = C.IMAGE_SIZE
    CH   = C.INPUT_CHANNELS
    data_img = torch.randn(BS, CH, H, W, device=device)
    data     = torch.randn(BS, CH, H, W, device=device)
    label    = torch.randn(BS, 1,  H, W, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE_ADAM)
    crit = nn.L1Loss()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model.train()
    with torch.amp.autocast('cuda'):
        out = model(data_img, data)
    print(f'  [ETER] forward OK   out.shape={tuple(out.shape)}   elapsed={time.time()-t0:.2f}s')

    loss = crit(out.float(), label)
    scaler = torch.amp.GradScaler('cuda')
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f'  [ETER] backward+step OK  loss={loss.item():.4f}  peak VRAM={peak:.2f}GB')

    del model, enc, opt; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--skip-ss2d', action='store_true')
    ap.add_argument('--skip-eter', action='store_true')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 필수')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    print('========== v7 smoke test ==========')
    print(f'Device: {device}  ({torch.cuda.get_device_name(args.gpu)})')
    print(f'Torch: {torch.__version__}  CUDA: {torch.version.cuda}')
    print(f'VRAM total: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f}GB')

    if not args.skip_ss2d:
        print('\n--- SS2D ---')
        smoke_ss2d(device)
    if not args.skip_eter:
        print('\n--- ETER ---')
        smoke_eter(device)
    print('\n========== smoke test 완료 ==========')


if __name__ == '__main__':
    main()
