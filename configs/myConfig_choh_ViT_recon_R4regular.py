












## choh_ViT_for_image_reconstruction, R4 regular 
# PATH_FOLDER = 'logs/240817_choh_ViT_recon_t300_lssim02_L1reg071_ep30_rtx2_1/'
# PATH_FOLDER = 'logs/240817_choh_ViT_recon_ViT_L_size_t300_lssim02_L1reg071_ep30_rtx2_2/'
PATH_FOLDER = 'logs/240821_choh_ViT_recon_ViT_B_size_t300_lssim02_L1reg071_ep30_rtx2_1/'        #rtx22
PATH_FOLDER = 'logs/240821_choh_ViT_recon_ViT_L_size_t300_lssim02_L1reg071_ep30_rtx2_2/'        #rtx23
PATH_FOLDER = 'logs/240822_choh_ViT_recon_ViT_H_size_t300_v41_lssim02_L1reg071_ep30_rtx2_5/'    #rtx22
PATH_FOLDER = 'logs/240822_choh_ViT_recon_ViT_B_size_t300_v41_lssim02_L1reg071_ep30_rtx2_6/'    #rtx23
## choh_ViT_for_image_reconstruction_with_tail
PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_L_size_fmlp_4_t300_lssim02_L1reg071_ep30_rtx2_7/'    #rtx23
PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_B_size_fmlp_8_t300_lssim02_L1reg071_ep30_rtx2_8/'    #rtx22
PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_H_size_fmlp_2_t300_lssim02_L1reg071_ep30_rtx2_9/'    #rtx21
## dual, 
PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_dual_ViT_L_size_fmlp_4_t300_lssim02_L1reg071_ep30_rtx2_10/'    #rtx20
# PATH_FOLDER = 'logs/temp/'



# PATH_LEGACY = 'logs/231018_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx1_18/'






NUM_TRAIN_SET = 3 #15 #3 #5 #15		 	#10 #2 #60 #10
NUM_TRAIN_VOLUME_PER_SET = 100 #20 #1 #5 #100 #60 #20 		#5 #30


BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 30 #50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000


LEARNING_RATE_ADAM = 1e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5


##### 		ViT PARAMETERS		######
# ### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
# NUM_VIT_HIDDEN = 1024
# NUM_VIT_LAYER = 24
# NUM_VIT_MLP_SIZE = 4096
# NUM_VIT_HEAD = 16
# NUM_VIT_FINAL_MLP_DIM = 4
# ### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
# NUM_VIT_HIDDEN = 768
# NUM_VIT_LAYER = 12
# NUM_VIT_MLP_SIZE = 3072
# NUM_VIT_HEAD = 12
# NUM_VIT_FINAL_MLP_DIM = 8
# ### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
# NUM_VIT_HIDDEN = 1280
# NUM_VIT_LAYER = 32
# NUM_VIT_MLP_SIZE = 5120
# NUM_VIT_HEAD = 16
# NUM_VIT_FINAL_MLP_DIM = 2

### dual ###
### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
NUM_VIT_HIDDEN = 1024
NUM_VIT_LAYER = 24
NUM_VIT_MLP_SIZE = 4096
NUM_VIT_HEAD = 16
NUM_VIT_FINAL_MLP_DIM = 4







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
