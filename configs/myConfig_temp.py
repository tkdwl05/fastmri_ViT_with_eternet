
















## choh_ViT_recon(choh_ViT_for_image_reconstruction in u_choh_model.py), R4 regular 
PATH_FOLDER = 'logs/temp/'



# PATH_LEGACY = 'logs/231018_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx1_18/'






NUM_TRAIN_SET = 3 #5 #15		 	#10 #2 #60 #10
NUM_TRAIN_VOLUME_PER_SET = 1 #5 #100 #60 #20 		#5 #30


BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 30 #50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000


LEARNING_RATE_ADAM = 1e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5


### this params equal to ViT-Large
NUM_VIT_HIDDEN = 1024
NUM_VIT_LAYER = 24
NUM_VIT_MLP_SIZE = 4096
NUM_VIT_HEAD = 16







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













# # ## brain multi 16ch, fullsize
# # PATH_FOLDER = 'logs/191219_fastmri_brain_fullsz16ch_acs32R4_320set_nh15_lssim05_ep50_rtx2_1/'
# # PATH_FOLDER = 'logs/200120_fastmri_brain_fullsz16ch_acs32R4_320set_nh15_lssim05_ep50_rtx2_2/'
# # PATH_FOLDER = 'logs/200121_fastmri_brain_fullsz16ch_acs32R4_fullslice_320set_nh15_lssim05_ep50_rtx2_3/'
# # PATH_FOLDER = 'logs/200129_fastmri_brain_fullsz16ch_acs32R4_320set_nh15_lssim05_ep50_rtx2_4/'
# # PATH_FOLDER = 'logs/200129_fastmri_brain_fullsz16ch_acs32R4_fullslice_320set_nh15_lssim05_ep50_rtx2_5/'
# # PATH_FOLDER = 'logs/200708_fastmri_brain_fullsz16ch_acs32R4_fullslice_320set_nh15_lssim00_ep50_rtx2_6/'     ## train L1 loss only
# # PATH_FOLDER = 'logs/200708_fastmri_brain_fullsz16ch_acs32R4_fullslice_320set_nh15_lssim05_ep50_rtx2_7/'     ## train/val/test 300/20/59 L1+SSIM
# # PATH_FOLDER = 'logs/200708_fastmri_brain_fullsz16ch_acs32R4_fullslice_320set_nh15_lssim00_ep50_rtx2_8/'     ## train/val/test 300/20/59 L1(base)
# # PATH_FOLDER = 'logs/200721_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh15_lssim02_ep50_rtx2_9/'     ## train/val/test 300/20/59 L1(base)
# # PATH_FOLDER = 'logs/200810_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh4_lssim02_ep50_rtx2_10/'     ## train/val/test 300/20/59 ssim
# # PATH_FOLDER = 'logs/200810_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh4_lssim00_ep50_rtx2_11/'     ## train/val/test 300/20/59 l1
# # PATH_FOLDER = 'logs/200814_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh8_lssim00_ep50_rtx2_12/'     ## train/val/test 300/20/59 l1
# # PATH_FOLDER = 'logs/200814_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh8_lssim02_ep50_rtx2_13/'     ## train/val/test 300/20/59 ssim
# # PATH_FOLDER = 'logs/200820_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh12_lssim02_ep50_rtx2_14/'     ## train/val/test 300/20/59 ssim
# # PATH_FOLDER = 'logs/200820_fastmri_brain_fullsz16ch_acs32R4_fullslice_t300v20_nh12_lssim00_ep50_rtx2_15/'     ## train/val/test 300/20/59 ssim
# # PATH_FOLDER = 'logs/200820_fastmriBrain_acs32R4_t300v20_nh15_lssim02_L1reg071_drop05_ep50_rtx2_16/'     ## train/val/test 300/20/59 ssim
# # PATH_FOLDER = 'logs/temp/'
# # PATH_FOLDER = 'logs/210325_fastmriBrain_hybrid_acs32R4_t300v20_nh12_lssim02_L1reg071_drop05_ep50_rtx2_1/'     ## train/val/test 300/20/59 ssim


# ## according to dataset size
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t10_nh12_lssim02_L1reg071_drop05_ep50_rtx2_1/'
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t20_nh12_lssim02_L1reg071_drop05_ep50_rtx2_2/'
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t40_nh12_lssim02_L1reg071_drop05_ep50_rtx1_3/'
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t80_nh12_lssim02_L1reg071_drop05_ep50_rtx2_4/'
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t160_nh12_lssim02_L1reg071_drop05_ep50_rtx2_5/'
# PATH_FOLDER = 'logs/230425_fastmriBrain_hybrid_acs32R4_t240_nh12_lssim02_L1reg071_drop05_ep50_rtx1_6/'
# PATH_FOLDER = 'logs/230430_fastmriBrain_hybrid_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx2_7/'
# PATH_FOLDER = 'logs/230501_fastmriBrain_hybrid_acs32R4_t160_nh12_lssim02_L1reg071_drop05_ep50_rtx1_8/'
# PATH_FOLDER = 'logs/230504_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg071_drop05_ep50_rtx1_9/'
# PATH_FOLDER = 'logs/230504_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg071_drop05_ep50_rtx2_10/'
# PATH_FOLDER = 'logs/230507_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg071_lr031_drop05_ep50_rtx2_11/'	## divergence nan
# PATH_FOLDER = 'logs/230508_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim03_L1reg071_drop05_ep50_rtx2_12/'
# PATH_FOLDER = 'logs/230508_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg061_drop05_ep50_rtx1_13/'
# PATH_FOLDER = 'logs/230514_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg051_drop05_ep50_rtx1_14/'
# PATH_FOLDER = 'logs/230514_fastmriBrain_hybrid_acs32R4_t270_nh12_lssim02_L1reg051_drop05_ep50_rtx2_15/'
# PATH_FOLDER = 'logs/230517_fastmriBrain_hybrid_acs32R4_t30_270_nh12_lssim02_L1reg051_drop05_ep50_rtx2_16/'
# PATH_FOLDER = 'logs/230517_fastmriBrain_hybrid_acs32R4_t30_270_nh12_lssim02_L1reg071_drop05_ep50_rtx1_17/'

# ## o-ETER training
# PATH_FOLDER = 'logs/231018_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx1_18/'
# PATH_FOLDER = 'logs/231021_fastmriBrain_oETER_retrain_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx1_19/'

# PATH_FOLDER = 'logs/231029_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_u3_drop05_ep10_my2_20/'
# PATH_FOLDER = 'logs/231030_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_u3_drop05_ep50_rtx1_21/'


# ## d-ETER training
# PATH_FOLDER = 'logs/231030_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_u3_drop05_ep50_rtx1_22/'
# PATH_FOLDER = 'logs/temp/'


# # PATH_LEGACY = 'logs/231018_fastmriBrain_oETER_acs32R4_t300_nh12_lssim02_L1reg071_drop05_ep50_rtx1_18/'





# # IDX_START = 0 #30
# # NUM_TRAIN_SET = 300 #31 #270 #160 
# # #300 #240 #160 #80 #40 #20 #10 #300 
# NUM_TRAIN_SET = 3 #5 #15		 	#10 #2 #60 #10
# NUM_TRAIN_VOLUME_PER_SET = 1 #5 #100 #60 #20 		#5 #30

# # NUM_TRAIN_SET = 60 #10
# # NUM_VOL_PER_SET = 5 #30


# LEARNING_RATE_ADAM = 1e-4
# LAMBDA_REGULAR_PER_PIXEL = 1e-7 #1e-5 #1e-6 #0.0 #1e-7 #0 #1e-7 #0 #0.0005
# LAMBDA_SSIM_PER_PIXEL = 0.2 #0.3 #0.2 #0.0 # 0.2 #0.5 #0 #0.5

# R_GRU_DROPOUT = 0.5 #0.5 #0.0


# BATCH_SIZE = 2 #1 #2 #8
# NUM_EPOCHS = 50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000
# # OPTIMIZER  =  'else' #'adam'
# ##torch.optim #'adadelta' 'sgd' 'nesterov' 'asgd' 'rmsprop' 'rprop' 'adagrad'   'adadelta'  'adam'  'sparseadam'



# # CRITERION = 'l1'
# ##   'mse'   'l1'   'smoothl1'
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












# import os
# if not os.path.exists(PATH_FOLDER):
#     os.makedirs(PATH_FOLDER)
