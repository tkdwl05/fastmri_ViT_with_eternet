












# ### 
# # PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_B_ch256_fea16_nh8_ep100_cos_rtx2_23/'	#rtx23
# PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_B_ch256_fea16_nh10_ep100_cos_rtx2_24/'
# PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_L_ch256_fea16_nh8_ep100_cos_rtx2_25/'
# PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_H_ch256_fea16_nh4_ep100_cos_rtx2_26/'
# PATH_FOLDER = 'logs/240918_choh_ViT_ETER_skip_up_tail__B_nh10__adam031_ep100_cos_rtx2_27/'
# PATH_FOLDER = 'logs/241007_choh_ViT_ETER_skip_up_tail_enc_B_dec_B_ch16_fea32_nh10_adam031_ep100_cos_rtx2_29/'
# # PATH_FOLDER = 'logs/temp/'



PATH_FOLDER = 'logs/251008_model3_R4random_enc_B_dec_B_nh10_exp3_5/'
PATH_FOLDER = 'logs/251008_model3_R4random_enc_B_dec_C_nh10_exp3_6/'
PATH_FOLDER = 'logs/251010_model3_R4random_enc_B_dec_C_nh10_WOanneal_exp3_7/'
PATH_FOLDER = 'logs/251011_model3_R8regular_enc_B_dec_C_nh10_WOanneal_exp3_8/'
PATH_FOLDER = 'logs/251011_model3_R8regular_enc_B_dec_C_nh10_WOanneal_exp3_9/'

PATH_FOLDER = 'logs/251011_model3_R4random_enc_B_dec_ismrm_WOanneal_exp3_10/'
PATH_FOLDER = 'logs/251011_model3_R8regular_enc_B_dec_ismrm_WOanneal_exp3_11/'
PATH_FOLDER = 'logs/251011_model3_R8regular_enc_B_dec_ismrm_Anneal_exp3_12/'

PATH_FOLDER = 'logs/251012_model3_R8regular_enc_B_dec_ismrm_adam041_Anneal_exp3_13/'

PATH_FOLDER = 'logs/251013_model3_R4random_enc_B_dec_ismrm_adam041_Anneal_exp3_14/'

PATH_FOLDER = 'logs/251014_model3_R8random_enc_B_dec_ismrm_adam041_Anneal_exp3_15/'

PATH_FOLDER = 'logs/251015_model3_R8regular_enc_B_dec_ismrm_adam041_Anneal_exp3_16/'
PATH_FOLDER = 'logs/251015_model3_R8random_enc_B_dec_ismrm_adam041_Anneal_exp3_17/'

PATH_FOLDER = 'logs/251124_model3_R4random_enc_B_dec_ismrm_adam041_exp3_18/'





PATH_SDA = '/mnt/sda/choh/shared/W_python/myViT/'
if 'PATH_SDA' in globals():
	PATH_FOLDER = './' + PATH_FOLDER
# PATH_SDB = '/mnt/sdb/choh/shared/W_python/myViT/'
# if 'PATH_SDB' in globals():
# 	PATH_FOLDER = PATH_SDB + PATH_FOLDER




NUM_TRAIN_SET = 3 #1 #3 
NUM_TRAIN_VOLUME_PER_SET = 100 #300 #100 #20 #1 #5 #100 #60 #20 		#5 #30
NUM_TRAIN_VOLUME_PER_INNER_LOOP = 10


BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 200 #50
LEARNING_RATE_ADAM = 1e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5

###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######
# ### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
# NUM_VIT_ENCODER_HIDDEN = 1024
# NUM_VIT_ENCODER_LAYER = 24
# NUM_VIT_ENCODER_MLP_SIZE = 4096
# NUM_VIT_ENCODER_HEAD = 16
### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
# NUM_VIT_ENCODER_HIDDEN = 768
# NUM_VIT_ENCODER_LAYER = 12
# NUM_VIT_ENCODER_MLP_SIZE = 3072
# NUM_VIT_ENCODER_HEAD = 12
### ViT-Small (8GB GPU용 축소 버전)
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD = 6
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_ENCODER_HIDDEN = 1280
# NUM_VIT_ENCODER_LAYER = 32
# NUM_VIT_ENCODER_MLP_SIZE = 5120
# NUM_VIT_ENCODER_HEAD = 16


###### 		BIRNN PARAMETERS		######
# NUM_ETER_HORI_HIDDEN = 10  # 원본 (hidden 실제 크기: 384*10=3840, GRU params ~370M)
# NUM_ETER_VERT_HIDDEN = 10
NUM_ETER_HORI_HIDDEN = 2   # 8GB GPU용 (hidden 실제 크기: 384*2=768)
NUM_ETER_VERT_HIDDEN = 2



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
# ### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
# NUM_VIT_DECODER_HEAD = 12 #16
# NUM_VIT_DECODER_DIM_MLP_HIDDEN = 3072
# NUM_VIT_DECODER_DIM = 768 #1280 #512
# NUM_VIT_DECODER_DEPTH = 12 #6
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_DECODER_DIM = 1280
# NUM_VIT_DECODER_DEPTH = 32
# NUM_VIT_DECODER_DIM_MLP_HIDDEN = 5120
# NUM_VIT_DECODER_HEAD = 16
# ### custom ### custom ### custom ### custom ### custom ### custom ### custom ### custom
# NUM_VIT_DECODER_HEAD = 8
# NUM_VIT_DECODER_DIM_MLP_HIDDEN = 5120
# NUM_VIT_DECODER_DIM = 1280
# NUM_VIT_DECODER_DIM_HEAD = 64
# NUM_VIT_DECODER_DEPTH = 12
### 8GB GPU용 축소 버전
NUM_VIT_DECODER_HEAD = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN = 2048
NUM_VIT_DECODER_DIM = 512
NUM_VIT_DECODER_DIM_HEAD = 64
NUM_VIT_DECODER_DEPTH = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 256
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16


###### 		TAIL PARAMETERS		######
# NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 16 
# NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 32 







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
