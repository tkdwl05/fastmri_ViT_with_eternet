import os
import sys
import torch
import torchvision
import torch.nn as nn
import numpy as np
import time
import datetime
import pytz
import socket
# import h5py
from types import ModuleType
# import mySSIM



# # from u_choh_model import ViT_choh
# # from u_choh_mae import MAE_choh
# # from u_choh_mae import MAE_choh_2
# from u_choh_model import choh_Decoder
# from u_choh_model import choh_Decoder2
# from u_choh_model import choh_ViT_for_image_reconstruction
from u_choh_model_choh_ViT_autoencoder import choh_ViT
from u_choh_model_choh_ViT_autoencoder import choh_Decoder2
from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_tail
from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_upsample_tail
from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_skip_tail
from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_skip_upsample_tail

from u_choh_model_ETER_ViT import choh_Decoder3_ETER_skip_up_tail

from u_choh_SSIM import SSIM









###choh
# # from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_train, choh_fastmri_brain_hybrid_ifft_acs32R4_val
# from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2
# from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2
# # from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2
from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4
from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1
# from myUNet_DF import UNet_choh_skip
# from myUtils import myOptimizer, myCriterion
# from myConfig_ETER_fastmri_brain_acs32R4_5 import *
# from myConfig_temp import *


# from myConfig_choh_ViT_autoencoder_R4regular import *



PATH_FOLDER = 'logs/temp/'
NUM_TRAIN_SET = 3 
NUM_TRAIN_VOLUME_PER_SET = 100 #1 #20 #1 #5 #100 #60 #20 		#5 #30

BATCH_SIZE = 2 #1 #2 #8
NUM_EPOCHS = 100 #30 #50 #3 #10 #50 #25 #25 #2 #11 #2 #100 #100 #1 #100 #1000 #10000 #5000 #1000 #100 #20 #1000
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
NUM_VIT_ENCODER_HIDDEN = 768
NUM_VIT_ENCODER_LAYER = 12
NUM_VIT_ENCODER_MLP_SIZE = 3072
NUM_VIT_ENCODER_HEAD = 12
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
NUM_VIT_DECODER_DIM_HEAD = 64
NUM_VIT_DECODER_DIM = 1280 #512
NUM_VIT_DECODER_DEPTH = 12 #6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 256 #128 #64 #32
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16 #8 #16

###### 		BIRNN PARAMETERS		######
NUM_ETER_HORI_HIDDEN = 8 
NUM_ETER_VERT_HIDDEN = 8 





######choh, logging setting
path_log = PATH_FOLDER + 'log.txt'
print(' ')
print(path_log)

class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open(path_log, "w")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.terminal.flush()
		self.log.flush()

sys.stdout = Logger()
print('\nBATCH_SIZE : %d'%BATCH_SIZE)
for aa in dir()[1:]:
	value_of_var = eval(aa)
	if isinstance(value_of_var, ModuleType) is False:
		print(aa, '\t' ,value_of_var)
print('BATCH_SIZE : %d\n'%BATCH_SIZE)





# path_history = PATH_FOLDER + 'history_loss.txt'
# f_history = open(path_history, "a")
tic1 = time.time()




from torch.utils.data import DataLoader
# import torchvision.datasets as dsets
# from torchvision import transforms
from torch.autograd import Variable


# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(socket.gethostname())
print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))







def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp






def main():
	print('\n  choh, train choh_ViT_autoencoder, fastmri brain_multi 16ch, 384x384, acs32R4, @main')





	vit_choh = choh_ViT(
	# image_size = 384,
	image_size = (384, 384), 
	patch_size = (32, 32),
	num_classes = 1000,
	dim = NUM_VIT_ENCODER_HIDDEN,
	depth = NUM_VIT_ENCODER_LAYER,
	heads = NUM_VIT_ENCODER_HEAD,
	mlp_dim = NUM_VIT_ENCODER_MLP_SIZE,
	channels=32,
	dropout = 0.1,
	emb_dropout = 0.1
	).cuda()

	choh_decoder = choh_Decoder3_ETER_skip_up_tail(
	encoder = vit_choh,
	# masking_ratio = 0.75,   # the paper recommended 75% masked patches
	eter_n_hori_hidden = NUM_ETER_HORI_HIDDEN,
	eter_n_vert_hidden = NUM_ETER_VERT_HIDDEN,
	decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	decoder_depth = NUM_VIT_DECODER_DEPTH,       # anywhere from 1 to 8
	decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	).cuda()


	print('\n number of params : {}\n'.format(get_n_params(choh_decoder)))

	print(' number of params model.encoder: {}'.format(get_n_params(choh_decoder.encoder)))
	print(' number of params model.decoder: {}'.format(get_n_params(choh_decoder.decoder)))
	print(' number of params model.gru_h: {}'.format(get_n_params(choh_decoder.gru_h)))
	print(' number of params model.gru_v: {}'.format(get_n_params(choh_decoder.gru_v)))


	# choh_decoder = choh_Decoder2(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	# decoder_depth = NUM_VIT_DECODER_DEPTH       # anywhere from 1 to 8
	# ).cuda()

	# choh_decoder = choh_Decoder2_with_tail(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	# decoder_depth = NUM_VIT_DECODER_DEPTH,       # anywhere from 1 to 8
	# decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	# decoder_dim_final_linear = NUM_VIT_DECODER_DIM_FINAL
	# ).cuda()

	# choh_decoder = choh_Decoder2_with_skip_tail(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	# decoder_depth = NUM_VIT_DECODER_DEPTH,       # anywhere from 1 to 8
	# decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	# decoder_dim_final_linear = NUM_VIT_DECODER_DIM_FINAL
	# ).cuda()


	# choh_decoder = choh_Decoder2_with_skip_upsample_tail(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	# decoder_depth = NUM_VIT_DECODER_DEPTH,       # anywhere from 1 to 8
	# decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	# decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	# decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	# ).cuda()

	# choh_decoder = choh_Decoder2_with_upsample_tail(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# decoder_dim = NUM_VIT_DECODER_DIM,      # paper showed good results with just 512
	# decoder_depth = NUM_VIT_DECODER_DEPTH,       # anywhere from 1 to 8
	# decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	# decoder_out_ch_final_linear = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	# decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	# ).cuda()





	if 'PATH_LEGACY' in globals():
		# filename_trained_weight = 'tensors_entire.pt'
		filename_idx = 34 #14 #24
		filename_trained_weight = 'tensors_%d.pt'%filename_idx
		# filename_trained_weight = 'tensors_pre%d.pt'%filename_idx

		print('\n retrain from PATH_LEGACY : {}'.format(PATH_LEGACY+filename_trained_weight))
		print(' retrain from PATH_LEGACY : {}'.format(PATH_LEGACY+filename_trained_weight))
		print(' retrain from PATH_LEGACY : {}'.format(PATH_LEGACY+filename_trained_weight))
		print(' retrain from PATH_LEGACY : {}'.format(PATH_LEGACY+filename_trained_weight))
		print(PATH_LEGACY+filename_trained_weight)
		choh_decoder = torch.load(PATH_LEGACY+filename_trained_weight, map_location="cuda")
		print(' loaded ')



	# print(choh_decoder)
	# import torchinfo
	# print('\ntorchinfo')
	# torchinfo.summary(choh_decoder, input_size=(2,32,384,384))
	# import torchsummary
	# print('\ntorchsummary')
	# torchsummary.summary(choh_decoder, (32,384,384))
	


	# criterion = myCriterion(CRITERION)
	# # optimizer = myOptimizer(mae_choh, OPTIMIZER)
	# # optimizer = myOptimizer(choh_decoder, OPTIMIZER)
	# optimizer = myOptimizer(choh_vit_recon, OPTIMIZER)

	# crit_ssim = mySSIM.SSIM().cuda()

	criterion_l1 = nn.L1Loss()
	criterion_ssim = SSIM().cuda()
	optimizer = torch.optim.Adam(choh_decoder.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)
	
	T_0=40
	T_mult=2
	eta_min=0
	print(' CosineAnnealingWarmRestarts : T_0 {}  T_mult {}  eta_min {}'.format( T_0, T_mult, eta_min))
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
	






	lambda_ssim_per_pixel = LAMBDA_SSIM_PER_PIXEL
	num_epochs = NUM_EPOCHS
	loss_prev = 1e8
	ssim_prev = 0
	validation_loss = np.zeros((num_epochs,1))
	training_loss = np.zeros( (num_epochs, 4800) )

	print('\n\n\n  start training iterations')
	for epoch in range(num_epochs):
		for n_set in range(NUM_TRAIN_SET):

			print('\n train set index start : {} end : {}'.format(NUM_TRAIN_VOLUME_PER_SET*n_set, NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET))
			# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			
			print(choh_data_train)
			print(' len(choh_data_train) : %d'%len(choh_data_train))
			trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True)
			print(trainloader)
			total_step = len(trainloader)



			for i_batch, sample_batched in enumerate(trainloader):

				data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
				data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
				data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)


				
				# out = eternet(data_in, data_in_img)
				# out = vit_choh(data_in_img)
				# out = mae_choh(data_in_img)


				# # loss = mae_choh(data_in_img, data_ref)
				# loss = mae_choh_2(data_in_img, data_ref)
				# loss.backward()

				### choh_Decoder1 (2 out version: out1, out2)
				# out = choh_decoder(data_in_img, data_ref)	
				# # print(' out1.shape {} out2.shape {}'.format(out1.shape, out2.shape))


				#### ## choh_Decoder2 (1 out version: out)
				# out = choh_decoder(data_in_img)			
				# out = choh_decoder(data_in) ## input ksp
				# print(' out.shape {}'.format(out.shape))
				if PATH_FOLDER== 'logs/240828_choh_ViT_AE_tail_inp_ksp_H_size_deco_1280_12_fmlp16_t300_v41_lssim02_L1reg071_ep30_rtx2_14/':
					out = choh_decoder(data_in)
				else:
					out = choh_decoder(data_in_img, data_in)
				


				# #### ## choh_vit_recons
				# out = choh_vit_recon(data_in_img)
				# print(' out.shape {}'.format(out.shape))
				# # print(' out1.shape {} out2.shape {}'.format(out1.shape, out2.shape))

				


				# import matplotlib.pyplot as plt
				# plt.figure()

				# print(' data_ref[0].shape {}'.format(data_ref[0].shape))
				# print(' out[0].shape {}'.format(out[0].shape))
				# plt.subplot(2,1,1)

				# img_to_plot_ref = data_ref[0].cpu().detach()
				# img_to_plot_ref = torch.squeeze( img_to_plot_ref )
				# img_to_plot_ref.numpy()

				# # img_to_plot_ref = torch.squeeze( data_ref[0].view(data_ref.shape[2], data_ref.shape[3], data_ref.shape[1]) )
				# print(' img_to_plot_ref.shape {}'.format(img_to_plot_ref.shape))
				
				# plt.imshow(img_to_plot_ref, aspect='equal')
				# plt.title('data_ref ')

				# plt.subplot(2,1,2)


				# img_to_plot_out2 = out[0].cpu().detach()
				# img_to_plot_out2 = torch.squeeze( img_to_plot_out2 )
				# img_to_plot_out2.numpy()


				# # img_to_plot_out2 = torch.squeeze( out2[0].view(out2.shape[2], out2.shape[3], out2.shape[1], out2.shape[0]) )
				
				# plt.imshow(img_to_plot_out2, aspect='equal')
				# plt.title('out ')

				# plt.show()



				
				# out.backward()
				


				### choh_decoder
				loss_pixel = criterion_l1(out, data_ref)
				loss_ssim = 1-criterion_ssim( out, data_ref)
				loss = loss_pixel + lambda_ssim_per_pixel*loss_ssim
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()



				# loss = criterion(out, data_ref)
				# loss.backward()

				# optimizer.step()

				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}'.format( epoch+1, num_epochs, i_batch+1, total_step, loss.item() ))

				# ### apply loss_ssim
				# loss_pixel = criterion(out, data_ref)
				# loss_ssim = 1-crit_ssim( out, data_ref)
				# loss = loss_pixel + lambda_ssim_per_pixel*loss_ssim
				# loss.backward()

				# optimizer.step()

				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}'.format( epoch+1, num_epochs, i_batch+1, total_step, loss.item() ))
				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format( 
				# 	epoch+1, num_epochs, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item()  ))

				# training_loss[epoch,n_set*total_step + i_batch] = loss_pixel.item()
				# training_loss[epoch,n_set*total_step + i_batch] = loss.item()
				
				print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} COS: {}'.format( 
					epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item(),  scheduler.get_lr() ))
				
				# print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format( 
				# 	epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item()  ))

				# print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}  '.format( 
				# 	epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item()  ))

				
			# training_loss[epoch,i_batch] = loss_pixel.item()
		print(' choh_ViT_autoencoder ')
		print('\n    choh, cwd : %s %s \n'%(os.getcwd(), PATH_FOLDER))
		print('    socket.gethostname() {} '.format( socket.gethostname() ) )
		print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
		
		

		### in case of decreasing loss
		# if loss_prev>loss.item():
		# 	loss_prev = loss.item()
		# 	torch.save( eternet, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		# 	print('saving done...\n')
		if ssim_prev<loss_ssim.item():
			ssim_prev = loss_ssim.item()
			# torch.save( eternet, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
			print(' loss_ssim improved, \n')

		# with torch.no_grad():
		# 	for i_batch, sample_batched in enumerate(validloader):
		# 		data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
		# 		data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
		# 		data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
		# 		out = choh_decoder(data_in, data_in_img)
		# 		loss_pixel = criterion(out, data_ref)

		# 	print('  epoch {}\t validation_loss: {:.6f}'.format(epoch+1, loss_pixel ) )
		# 	validation_loss[epoch] = loss_pixel.item()



		torch.save( choh_decoder, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		print('saving done...\n')



	torch.save(choh_decoder, PATH_FOLDER+'tensors_entire.pt')
	print('saving done...\n')

	filename_save_nparry = PATH_FOLDER + 'validation_loss'
	np.save(filename_save_nparry, validation_loss)

	filename_save_nparry = PATH_FOLDER + 'training_loss'
	np.save(filename_save_nparry, training_loss)



	toc1 = time.time()
	print('total Time = ', (toc1 - tic1))
	print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

	print(' ')
	print('    choh, torch.save : %s'%(PATH_FOLDER+'tensors_entire.pt'))
	print('  @ main function, end')
	print('finished !')



if __name__ == '__main__':
	main()
