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
# from u_choh_model_choh_ViT_autoencoder import choh_ViT
# from u_choh_model_choh_ViT_autoencoder import choh_Decoder2
# from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_tail
# from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_upsample_tail
# from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_skip_tail
# from u_choh_model_choh_ViT_autoencoder import choh_Decoder2_with_skip_upsample_tail

from u_choh_model_ETER_ViT import choh_ViT	## same with : from u_choh_model_choh_ViT_autoencoder import choh_ViT
from u_choh_model_ETER_ViT import choh_Decoder3_ETER_skip_up_tail


from u_choh_SSIM import SSIM









###choh
# from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4
# from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1
from myDataloader_fastmri_brain_random_250905 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1_random
from myDataloader_fastmri_brain_R8_250903 import choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_1
from myDataloader_fastmri_brain_R8_251012 import choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_2

# from myConfig_choh_ViT_ETER_R4regular import *
from myConfig_choh_model3 import *









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
	print('\n  choh, train choh_ViT_ETER, fastmri brain_multi 16ch, 384x384, acs32R4, @main')





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

	# choh_decoder = choh_Decoder3_ETER_skip_up_tail(
	# encoder = vit_choh,
	# # masking_ratio = 0.75,   # the paper recommended 75% masked patches
	# eter_n_hori_hidden = NUM_ETER_HORI_HIDDEN,
	# eter_n_vert_hidden = NUM_ETER_VERT_HIDDEN,
	# decoder_dim = NUM_VIT_DECODER_DIM,
	# decoder_depth = NUM_VIT_DECODER_DEPTH,
	# decoder_heads= NUM_VIT_DECODER_HEAD, 
	# # decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	# decoder_dim_mlp_hidden = NUM_VIT_DECODER_DIM_MLP_HIDDEN, 
	# decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	# decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	# ).cuda()
	
	## SAME MODEL, but diff arguments
	choh_decoder = choh_Decoder3_ETER_skip_up_tail( 
	encoder = vit_choh,
	# masking_ratio = 0.75,   # the paper recommended 75% masked patches
	eter_n_hori_hidden = NUM_ETER_HORI_HIDDEN,
	eter_n_vert_hidden = NUM_ETER_VERT_HIDDEN,
	decoder_dim = NUM_VIT_DECODER_DIM,
	decoder_depth = NUM_VIT_DECODER_DEPTH,
	decoder_heads= NUM_VIT_DECODER_HEAD, 
	decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	decoder_dim_mlp_hidden = NUM_VIT_DECODER_DIM_MLP_HIDDEN, 
	decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	).cuda()


	print('\n number of params : {}\n'.format(get_n_params(choh_decoder)))

	print(' number of params model.encoder: {}'.format(get_n_params(choh_decoder.encoder)))
	print(' number of params model.decoder: {}'.format(get_n_params(choh_decoder.decoder)))
	print(' number of params model.gru_h: {}'.format(get_n_params(choh_decoder.gru_h)))
	print(' number of params model.gru_v: {}'.format(get_n_params(choh_decoder.gru_v)))








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





	criterion_l1 = nn.L1Loss()
	criterion_ssim = SSIM().cuda()
	optimizer = torch.optim.Adam(choh_decoder.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)
	
	T_0=40
	T_mult=2
	eta_min=0
	print(' CosineAnnealingWarmRestarts : T_0 {}  T_mult {}  eta_min {}'.format( T_0, T_mult, eta_min))
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
	






	lambda_ssim_per_pixel = LAMBDA_SSIM_PER_PIXEL
	num_epochs = int(NUM_EPOCHS/NUM_TRAIN_VOLUME_PER_INNER_LOOP)
	loss_prev = 1e8
	ssim_prev = 0
	validation_loss = np.zeros((num_epochs,1))
	training_loss = np.zeros( (num_epochs, 4800) )

	n_counter = 1
	print('\n\n\n  start training iterations')
	for epoch in range(num_epochs):
		for n_set in range(NUM_TRAIN_SET):
			print('Epoch [{}/{}], n_SET [{}/{}], n_counter {}'.format( 
				epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, n_counter))
			print('\n    choh, cwd : %s %s \n'%(os.getcwd(), PATH_FOLDER))
			print('    socket.gethostname() {} '.format( socket.gethostname() ) )
			print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
			print('\n train set index start : {} end : {}'.format(NUM_TRAIN_VOLUME_PER_SET*n_set, NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET))
			# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1_random(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_1(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			choh_data_train = choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_2(idx_start= NUM_TRAIN_VOLUME_PER_SET*n_set , idx_end=NUM_TRAIN_VOLUME_PER_SET*n_set+NUM_TRAIN_VOLUME_PER_SET)
			print(choh_data_train)

			print(' len(choh_data_train) : %d'%len(choh_data_train))
			trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True)
			print(trainloader)
			total_step = len(trainloader)

			for n_inner in range(NUM_TRAIN_VOLUME_PER_INNER_LOOP):
				for i_batch, sample_batched in enumerate(trainloader):

					data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
					data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
					data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)

					out = choh_decoder(data_in_img, data_in)

					loss_pixel = criterion_l1(out, data_ref)
					loss_ssim = 1-criterion_ssim( out, data_ref)
					loss = loss_pixel + lambda_ssim_per_pixel*loss_ssim
					loss.backward()

					optimizer.step()
					optimizer.zero_grad()
					scheduler.step()

					print('Epoch [{}/{}], n_SET [{}/{}], n_IN [{}/{}], n_counter {}, Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} COS: {}'.format( 
						epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, n_inner+1, NUM_TRAIN_VOLUME_PER_INNER_LOOP, 
						n_counter, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item(),  scheduler.get_lr()  ))
					

				if (n_inner==0) or (n_inner==5):
					torch.save( choh_decoder, PATH_FOLDER+'tensors_cnt%d.pt'%(n_counter) )
					print(f'saving done...\ttensors_cnt{n_counter}.pt\n')
				
				n_counter = n_counter + 1
				print(' choh_ViT_eter ')
				print('\n    choh, cwd : %s %s \n'%(os.getcwd(), PATH_FOLDER))
				print('    socket.gethostname() {} '.format( socket.gethostname() ) )
				print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
		
		

		### in case of decreasing loss
		# if loss_prev>loss.item():
		# 	loss_prev = loss.item()
		# 	torch.save( eternet, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		# 	print('saving done...\n')
		# if ssim_prev<loss_ssim.item():
		# 	ssim_prev = loss_ssim.item()
		# 	# torch.save( eternet, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		# 	print(' loss_ssim improved, \n')

		# with torch.no_grad():
		# 	for i_batch, sample_batched in enumerate(validloader):
		# 		data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
		# 		data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
		# 		data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
		# 		out = choh_decoder(data_in, data_in_img)
		# 		loss_pixel = criterion(out, data_ref)

		# 	print('  epoch {}\t validation_loss: {:.6f}'.format(epoch+1, loss_pixel ) )
		# 	validation_loss[epoch] = loss_pixel.item()


		# if epoch==1 or epoch == 2 or epoch%10==0:
		# 	torch.save( choh_decoder, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		# 	print('saving done...\n')
		# torch.save( choh_decoder, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		# print('saving done...\n')



	torch.save(choh_decoder, PATH_FOLDER+'tensors_entire.pt')
	print('saving done...\n')

	# filename_save_nparry = PATH_FOLDER + 'validation_loss'
	# np.save(filename_save_nparry, validation_loss)

	# filename_save_nparry = PATH_FOLDER + 'training_loss'
	# np.save(filename_save_nparry, training_loss)



	toc1 = time.time()
	print('total Time = ', (toc1 - tic1))
	print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

	print(' ')
	print('    choh, torch.save : %s'%(PATH_FOLDER+'tensors_entire.pt'))
	print('  @ main function, end')
	print('finished !')



if __name__ == '__main__':
	main()
