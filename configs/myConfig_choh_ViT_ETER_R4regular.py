












### 
# PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_B_ch256_fea16_nh8_ep100_cos_rtx2_23/'	#rtx23
PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_B_ch256_fea16_nh10_ep100_cos_rtx2_24/'
PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_L_ch256_fea16_nh8_ep100_cos_rtx2_25/'
PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_H_ch256_fea16_nh4_ep100_cos_rtx2_26/'
PATH_FOLDER = 'logs/240918_choh_ViT_ETER_skip_up_tail__B_nh10__adam031_ep100_cos_rtx2_27/'
PATH_FOLDER = 'logs/241007_choh_ViT_ETER_skip_up_tail_enc_B_dec_B_ch16_fea32_nh10_adam031_ep100_cos_rtx2_29/'
# PATH_FOLDER = 'logs/temp/'








PATH_SDB = '/mnt/sdb/choh/shared/W_python/myViT/'
if 'PATH_SDB' in globals():
	PATH_FOLDER = PATH_SDB + PATH_FOLDER


NUM_TRAIN_SET = 3 
NUM_TRAIN_VOLUME_PER_SET = 100 #20 #1 #5 #100 #60 #20 		#5 #30


BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 100 #30 #50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000
LEARNING_RATE_ADAM = 1e-3 #1e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5

###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######
# ### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
# NUM_VIT_ENCODER_HIDDEN = 1024
# NUM_VIT_ENCODER_LAYER = 24
# NUM_VIT_ENCODER_MLP_SIZE = 4096
# NUM_VIT_ENCODER_HEAD = 16
### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
NUM_VIT_ENCODER_HIDDEN = 768
NUM_VIT_ENCODER_LAYER = 12
NUM_VIT_ENCODER_MLP_SIZE = 3072
NUM_VIT_ENCODER_HEAD = 12
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_ENCODER_HIDDEN = 1280
# NUM_VIT_ENCODER_LAYER = 32
# NUM_VIT_ENCODER_MLP_SIZE = 5120
# NUM_VIT_ENCODER_HEAD = 16


###### 		BIRNN PARAMETERS		######
NUM_ETER_HORI_HIDDEN = 10 #4
NUM_ETER_VERT_HIDDEN = 10 #4



# ###### 		DECODER PARAMETERS		######
# NUM_VIT_DECODER_DIM_HEAD = 64
# NUM_VIT_DECODER_DIM_FINAL = 128 #32 #16 #4
# NUM_VIT_DECODER_DIM = 1280 #512
# NUM_VIT_DECODER_DEPTH = 12 #6
###### 		DECODER PARAMETERS		######
# ### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
# NUM_VIT_DECODER_DIM = 1024
# NUM_VIT_DECODER_DEPTH = 24
# NUM_VIT_DECODER_DIM_MLP_HIDDEN = 4096
# NUM_VIT_DECODER_HEAD = 16
### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
NUM_VIT_DECODER_HEAD = 12 #16
NUM_VIT_DECODER_DIM_MLP_HIDDEN = 3072
NUM_VIT_DECODER_DIM = 768 #1280 #512
NUM_VIT_DECODER_DEPTH = 12 #6
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_DECODER_DIM = 1280
# NUM_VIT_DECODER_DEPTH = 32
# NUM_VIT_DECODER_DIM_MLP_HIDDEN = 5120
# NUM_VIT_DECODER_HEAD = 16


###### 		TAIL PARAMETERS		######
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 16 
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 32 







# # ###### 		DECODER PARAMETERS		######
# # NUM_VIT_DECODER_DIM_HEAD = 64
# # NUM_VIT_DECODER_DIM_FINAL = 128 #32 #16 #4
# # NUM_VIT_DECODER_DIM = 1280 #512
# # NUM_VIT_DECODER_DEPTH = 12 #6
# ###### 		DECODER PARAMETERS		######
# NUM_VIT_DECODER_DIM_HEAD = 64
# NUM_VIT_DECODER_DIM = 1280 #512
# NUM_VIT_DECODER_DEPTH = 12 #6
# NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 256 #128 #64 #32
# NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16 #8 #16

























import os
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)
