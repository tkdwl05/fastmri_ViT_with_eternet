"""SS2D-ViT 설정 v6_4 — VGG perceptual loss fine-tune.

가설: gradient L1 (v6_1/v6_2) 와 sharpness 완화 (v6_3) 모두 한계가 있다면,
feature-space 의 perceptual loss 가 mean-blurring 을 다른 방식으로 처벌할 수 있다.

loss = L1 + λ_ssim·(1-SSIM) + λ_perc · ||VGG_feat(pred) - VGG_feat(gt)||_1

VGG: torchvision VGG16 (ImageNet pretrained), conv1_2/conv2_2/conv3_3 features.
1ch → 3ch (replicate), ImageNet 정규화는 생략 (MRI raw amplitude → max-normalize 후 [0,1]).

v6 best 부터 50ep fine-tune, LR 5e-5 → 5e-7 cosine.
"""
import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v6_4/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE_ADAM = 5e-5
LR_COSINE_MIN = 5e-7
LAMBDA_REGULAR_PER_PIXEL = 3e-5      # v6 동일
LAMBDA_SSIM_PER_PIXEL = 1.0
LAMBDA_PERCEPTUAL = 0.1              # 보수적 시작; VGG feat L1 은 raw L1 보다 큰 magnitude

DROPOUT = 0.2                        # v6 동일
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

EARLYSTOP_PATIENCE = 5
VAL_EVERY_N_EPOCHS = 5

RESUME_CKPT = './logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt'

DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

NUM_SS2D_D_INNER = 64
NUM_SS2D_D_STATE = 16
NUM_SS2D_OUT_CH  = 20

NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
