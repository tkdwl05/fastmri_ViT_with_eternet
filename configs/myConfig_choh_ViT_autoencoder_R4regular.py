












## choh_ViT_autoencoder, R4 regular 
# PATH_FOLDER = 'logs/240821_choh_ViT_recon_ViT_L_size_t300_lssim02_L1reg071_ep30_rtx2_3/'
# PATH_FOLDER = 'logs/240821_choh_ViT_autoencoder_H_size_t300_lssim02_L1reg071_ep30_rtx2_4/'
# PATH_FOLDER = 'logs/240822_choh_ViT_autoencoder_H_size_t300_lssim02_L1reg071_ep30_rtx2_3/'
# PATH_FOLDER = 'logs/240822_choh_ViT_autoencoder_H_size_t300_v4_1_lssim02_L1reg071_ep30_rtx2_4/'
# PATH_FOLDER = 'logs/240822_choh_ViT_autoencoder_H_size_t300_v4_1_lssim02_L1reg071_ep30_rtx2_3/'
# PATH_FOLDER = 'logs/240822_choh_ViT_autoencoder_H_size_deco_1280_12_t300_v4_1_lssim02_L1reg071_ep30_rtx2_4/'
#### with tail 
# PATH_FOLDER = 'logs/240826_choh_ViT_autoencoder_tail_H_size_deco_1280_12_fmlp16_t300_v41_lssim02_L1reg071_ep30_rtx2_11/'
# PATH_FOLDER = 'logs/240828_choh_ViT_AE_tail_inp_ksp_H_size_deco_1280_12_fmlp16_t300_v41_lssim02_L1reg071_ep30_rtx2_14/'
# PATH_FOLDER = 'logs/240828_choh_ViT_AE_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_rtx2_15/'
# PATH_FOLDER = 'logs/240830_choh_ViT_AE_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_retrain24_rtx2_17/'
# PATH_FOLDER = 'logs/240830_choh_ViT_AE_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_retrain24_rtx2_17/'
PATH_FOLDER = 'logs/240906_choh_ViT_AE_tail_H_size_fmlp128_ep100_retrain17_34_rtx2_21/'
PATH_FOLDER = 'logs/240906_choh_ViT_AE_tail_H_size_fmlp128_ep100_cos_retrain17_34_rtx2_22/'
# #### with up tail 
# PATH_FOLDER = 'logs/240827_choh_ViT_AE_up_tail_H_size_deco_1280_12_ch64_fea8_t300_v41_lssim02_L1reg071_ep30_rtx2_12/'
# PATH_FOLDER = 'logs/240827_choh_ViT_AE_up_tail_H_size_deco_1280_12_ch64_fea16_t300_v41_lssim02_L1reg071_ep30_rtx2_13/'
#### with skip tail 
# PATH_FOLDER = 'logs/240829_choh_ViT_AE_skip_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_rtx2_16/'
# PATH_FOLDER = 'logs/240830_choh_ViT_AE_skip_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_retrain14_rtx2_18/'
PATH_FOLDER = '/mnt/sdb/choh/shared/W_python/myViT/logs/240906_choh_ViT_AE_skip_tail_H_size_fmlp128_ep100_retrain16_14_rtx2_18/'
#### with skip up tail 
# PATH_FOLDER = 'logs/240830_choh_ViT_AE_skip_up_tail_H_size_deco_1280_12_ch256_fea16_t300_v41_lssim02_L1reg071_ep100_rtx2_19/'
# PATH_FOLDER = '/mnt/sdb/choh/shared/W_python/myViT/logs/240906_choh_ViT_AE_skip_up_tail_H_ch256_fea16_ep100_retrain19_34_rtx2_20/'
PATH_FOLDER = 'logs/240906_choh_ViT_AE_skip_up_tail_H_ch256_fea16_ep100_cos_retrain19_34_rtx2_20/'
PATH_FOLDER = 'logs/241007_choh_ViT_AE_skip_up_tail_enc_L_dec_L_ch16_fea32_ep100_cos_retrain19_34_rtx2_28/'	#rtx23
PATH_FOLDER = 'logs/241008_choh_ViT_AE_skip_up_tail_enc_L_dec_L_ch16_fea16_ep100_cos_rtx2_30/'	#rtx21
# PATH_FOLDER = 'logs/temp/'





# # #### with tail 
# # PATH_LEGACY = 'logs/240828_choh_ViT_AE_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_rtx2_15/'
# PATH_LEGACY = 'logs/240830_choh_ViT_AE_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_retrain24_rtx2_17/'
# # #### with skip tail 
# # PATH_LEGACY = 'logs/240829_choh_ViT_AE_skip_tail_H_size_deco_1280_12_fmlp128_t300_v41_lssim02_L1reg071_ep100_rtx2_16/'	#rtx22
# # #### with skip up tail 
# PATH_LEGACY = 'logs/240830_choh_ViT_AE_skip_up_tail_H_size_deco_1280_12_ch256_fea16_t300_v41_lssim02_L1reg071_ep100_rtx2_19/'


PATH_SDB = '/mnt/sdb/choh/shared/W_python/myViT/'
if 'PATH_SDB' in globals():
	PATH_FOLDER = PATH_SDB + PATH_FOLDER


NUM_TRAIN_SET = 3 #3 #15 #3 #5 #15		 	#10 #2 #60 #10
NUM_TRAIN_VOLUME_PER_SET = 100 #1 #20 #1 #5 #100 #60 #20 		#5 #30


BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 100 #30 #50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000


LEARNING_RATE_ADAM = 1e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5

###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######	###### 		ENCODER PARAMETERS		######
### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
NUM_VIT_ENCODER_HIDDEN = 1024
NUM_VIT_ENCODER_LAYER = 24
NUM_VIT_ENCODER_MLP_SIZE = 4096
NUM_VIT_ENCODER_HEAD = 16
# ### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
# NUM_VIT_ENCODER_HIDDEN = 768
# NUM_VIT_ENCODER_LAYER = 12
# NUM_VIT_ENCODER_MLP_SIZE = 3072
# NUM_VIT_ENCODER_HEAD = 12
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_ENCODER_HIDDEN = 1280
# NUM_VIT_ENCODER_LAYER = 32
# NUM_VIT_ENCODER_MLP_SIZE = 5120
# NUM_VIT_ENCODER_HEAD = 16


# ###### 		DECODER PARAMETERS		######
# NUM_VIT_DECODER_DIM_HEAD = 64
# NUM_VIT_DECODER_DIM_FINAL = 128 #32 #16 #4
# NUM_VIT_DECODER_DIM = 1280 #512
# NUM_VIT_DECODER_DEPTH = 12 #6
###### 		DECODER PARAMETERS		######
### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
NUM_VIT_DECODER_DIM = 1024
NUM_VIT_DECODER_DEPTH = 24
NUM_VIT_DECODER_DIM_MLP_HIDDEN = 4096
NUM_VIT_DECODER_HEAD = 16
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


###### 		TAIL PARAMETERS		######
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 16 
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16 #32 










# R_GRU_DROPOUT = 0.5 #0.5 #0.0



# OPTIMIZER  =  'else' #'adam'
##torch.optim #'adadelta' 'sgd' 'nesterov' 'asgd' 'rmsprop' 'rprop' 'adagrad'   'adadelta'  'adam'  'sparseadam'



# CRITERION = 'l1'
##   'mse'   'l1'   'smoothl1'
# N_UNET_DEPTH = 3 #4 #5


# N_RFCOIL = 16
# N_INPUT_VERTICAL = 384
# # N_FREQ_ENCODING = 396
# N_FREQ_ENCODING = 384 #32+99 
# N_OUT_X = 384
# N_OUT_Y = 384 #396
# N_OUTPUT = 384
# N_COIL_CH = 16

# N_HIDDEN_LRNN_1 = 12 #4 #15 #12 #8 #4 #15 #16 #22 #24 #32 #16 #8
# N_HIDDEN_LRNN_2 = 12 #4 #15 #12 #8 #4 #15 #16 #8 #16 #8












import os
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)
