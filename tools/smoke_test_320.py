"""
통합 smoke test (fastMRI 320×320 표준 전환 검증)

확인 항목:
  1. 새 dataloader가 올바른 shape/range로 반환하는가
  2. reconstruction_rss가 GT로 제대로 쓰이는가
  3. SS2D 모델 forward가 오류 없이 통과하는가
  4. ETER 모델 forward가 오류 없이 통과하는가
  5. 한 슬라이스를 PNG로 덤프하여 육안 확인

산출물: results/smoke_test_320/*.png
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'configs'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'dataloaders'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'models', 'hybrid_eternet'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'models', 'mamba_eternet'))

os.chdir(CURRENT_DIR)

from dataloader_h5 import FastMRI_H5_Dataloader


def aliased_to_sos(data_img_np):
    real = data_img_np[0::2]
    imag = data_img_np[1::2]
    return np.sqrt(np.sum(real ** 2 + imag ** 2, axis=0))


def test_dataloader():
    print('\n[1] Dataloader (val, num_files=2) 로드 테스트')
    ds = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val', num_files=2)
    print(f'    total samples: {len(ds)}')

    s = ds[5]
    d, di, lb = s['data'], s['data_img'], s['label']
    print(f'    data     shape={d.shape}  range=[{d.min():.4f}, {d.max():.4f}]')
    print(f'    data_img shape={di.shape} range=[{di.min():.4f}, {di.max():.4f}]')
    print(f'    label    shape={lb.shape} range=[{lb.min():.4f}, {lb.max():.4f}]')

    assert d.shape == (32, 320, 320), f'data shape 불일치: {d.shape}'
    assert di.shape == (32, 320, 320), f'data_img shape 불일치: {di.shape}'
    assert lb.shape == (1, 320, 320), f'label shape 불일치: {lb.shape}'
    print('    OK — shape assertions 통과')

    return ds, s


def dump_png(sample, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gt = sample['label'][0]
    aliased = aliased_to_sos(sample['data_img'])

    vmax = np.percentile(gt, 99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a1a')

    axes[0].imshow(gt, cmap='gray', vmin=0, vmax=vmax)
    axes[0].set_title(f'GT (reconstruction_rss)\nshape={gt.shape}  '
                      f'range=[{gt.min():.2f}, {gt.max():.2f}]',
                      color='white', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(aliased, cmap='gray', vmin=0, vmax=vmax)
    axes[1].set_title(f'Aliased (R=4 masked, SoS)\nshape={aliased.shape}  '
                      f'range=[{aliased.min():.2f}, {aliased.max():.2f}]',
                      color='white', fontsize=11)
    axes[1].axis('off')

    fig.suptitle('New dataloader smoke test — brain AXFLAIR 320×320',
                 color='white', fontsize=12)
    plt.tight_layout()
    fname = os.path.join(out_dir, 'gt_vs_aliased.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'    Saved: {fname}')


def test_model_forward(sample, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[{3 if model_type == "ss2d" else 4}] {model_type.upper()} forward 테스트')

    if model_type == 'ss2d':
        from myConfig_choh_SS2D_model import (
            IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
            NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
            NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
            NUM_SS2D_D_INNER, NUM_SS2D_D_STATE, NUM_SS2D_OUT_CH,
            NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
            NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
            NUM_VIT_DECODER_DIM_MLP_HIDDEN,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        )
        from u_choh_model_ETER_ViT import choh_ViT
        from u_choh_model_SS2D_ViT import choh_Decoder_SS2D_ViT

        encoder = choh_ViT(
            image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
            dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
            heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
            channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
        ).to(device)
        model = choh_Decoder_SS2D_ViT(
            encoder=encoder,
            ss2d_d_inner=NUM_SS2D_D_INNER, ss2d_d_state=NUM_SS2D_D_STATE,
            ss2d_out_ch=NUM_SS2D_OUT_CH,
            decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
            decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
            decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
            decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
            decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        ).to(device)
    else:
        from myConfig_choh_ETER_model import (
            IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
            NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
            NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
            NUM_ETER_HORI_HIDDEN, NUM_ETER_VERT_HIDDEN,
            NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
            NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
            NUM_VIT_DECODER_DIM_MLP_HIDDEN,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        )
        from u_choh_model_ETER_ViT import choh_ViT, choh_Decoder3_ETER_skip_up_tail

        encoder = choh_ViT(
            image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
            dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
            heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
            channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
        ).to(device)
        model = choh_Decoder3_ETER_skip_up_tail(
            encoder=encoder,
            eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
            eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
            decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
            decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
            decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
            decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
            decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'    파라미터: {n_params/1e6:.2f}M')

    data_in     = torch.from_numpy(sample['data']).unsqueeze(0).to(device)
    data_in_img = torch.from_numpy(sample['data_img']).unsqueeze(0).to(device)
    data_ref    = torch.from_numpy(sample['label']).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(data_in_img, data_in)
    print(f'    in_imgs: {tuple(data_in_img.shape)}  in_ksp: {tuple(data_in.shape)}')
    print(f'    output : {tuple(out.shape)}  dtype={out.dtype}'
          f'  range=[{out.float().min().item():.3f}, {out.float().max().item():.3f}]')
    print(f'    label  : {tuple(data_ref.shape)}')
    assert out.shape == data_ref.shape, f'output/label shape 불일치: {out.shape} vs {data_ref.shape}'

    mse = torch.mean((out.float() - data_ref.float()) ** 2).item()
    l1  = torch.mean(torch.abs(out.float() - data_ref.float())).item()
    print(f'    (untrained) MSE={mse:.4f}  L1={l1:.4f}')
    print(f'    OK — {model_type.upper()} forward 통과')

    del model, encoder, out
    torch.cuda.empty_cache()


def main():
    print('=' * 60)
    print(' Smoke test — brain AXFLAIR 320×320 전환')
    print('=' * 60)

    ds, sample = test_dataloader()

    print('\n[2] GT vs Aliased PNG 덤프')
    dump_png(sample, out_dir='results/smoke_test_320')

    test_model_forward(sample, 'ss2d')
    test_model_forward(sample, 'eter')

    print('\n' + '=' * 60)
    print(' ALL TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
