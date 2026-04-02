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

from types import ModuleType
# import mySSIM
from u_choh_SSIM import SSIM









###choh
# from u_choh_model import choh_ViT_for_image_reconstruction
from u_choh_model_choh_ViT_for_image_reconstruction import choh_ViT_for_image_reconstruction
from u_choh_model_choh_ViT_for_image_reconstruction import choh_ViT_for_image_reconstruction_with_tail

from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4
from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1
# from myUNet_DF import UNet_choh_skip
# from myUtils import myOptimizer, myCriterion
from myConfig_choh_ViT_recon_R4regular import *


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











def main():
	print('\n  choh, train choh_ViT_for_image_reconstruction, fastmri brain_multi 16ch, 384x384, acs32R4, @main')

	
	# choh_vit_recon = choh_ViT_for_image_reconstruction(
	# image_size = (384, 384), 
	# patch_size = (32, 32),
	# num_classes = 384*384,
	# dim = NUM_VIT_HIDDEN,
	# depth = NUM_VIT_LAYER,
	# heads = NUM_VIT_HEAD,
	# mlp_dim = NUM_VIT_MLP_SIZE,
	# channels=32,
	# dropout = 0.1,
	# emb_dropout = 0.1
	# ).cuda()

	choh_vit_recon = choh_ViT_for_image_reconstruction_with_tail(
	image_size = (384, 384), 
	patch_size = (32, 32),
	num_classes = NUM_VIT_FINAL_MLP_DIM,
	dim = NUM_VIT_HIDDEN,
	depth = NUM_VIT_LAYER,
	heads = NUM_VIT_HEAD,
	mlp_dim = NUM_VIT_MLP_SIZE,
	channels=64, #dual 
	dropout = 0.1,
	emb_dropout = 0.1
	).cuda()





	criterion_l1 = nn.L1Loss()
	criterion_ssim = SSIM().cuda()
	optimizer = torch.optim.Adam(choh_vit_recon.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)




	lambda_ssim_per_pixel = LAMBDA_SSIM_PER_PIXEL
	num_epochs = NUM_EPOCHS
	loss_prev = 1e8
	ssim_prev = 0
	validation_loss = np.zeros((num_epochs,1))
	training_loss = np.zeros( (num_epochs, 4800) )

	print('\n\n\n  start training iterations')
	for epoch in range(num_epochs):
		for n_set in range(NUM_TRAIN_SET):

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

				### dual concat
				# print(' data_in.shape {}'.format( data_in.shape ))
				# print(' data_in_img.shape {}'.format( data_in_img.shape ))
				data_concat = torch.cat((data_in,data_in_img),dim=1)
				# print(' data_concat.shape {}'.format( data_concat.shape ))

				#### ## choh_vit_recons
				out = choh_vit_recon(data_concat)
				# print('  choh_vit_recon out.shape {}'.format(out.shape))


				# ### imshow for debug
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





				### choh_decoder
				loss_pixel = criterion_l1(out, data_ref)
				# print(' data_ref.shape {}'.format(data_ref.shape))
				# print(' out.shape {}'.format(out.shape))
				loss_ssim = 1-criterion_ssim( out, data_ref)
				loss = loss_pixel + lambda_ssim_per_pixel*loss_ssim
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

				
				print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format( 
					epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item()  ))

				# print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}  '.format( 
				# 	epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item()  ))

				
			# training_loss[epoch,i_batch] = loss_pixel.item()
		print(' choh_ViT_for_image_reconstruction ')
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
			print(' loss_ssim improved, \n')

		# with torch.no_grad():
		# 	for i_batch, sample_batched in enumerate(validloader):
		# 		data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
		# 		data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
		# 		data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
		# 		out = eternet(data_in, data_in_img)
		# 		loss_pixel = criterion(out, data_ref)

		# 	print('  epoch {}\t validation_loss: {:.6f}'.format(epoch+1, loss_pixel ) )
		# 	validation_loss[epoch] = loss_pixel.item()



		torch.save( choh_vit_recon, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		print('saving done...\n')



	torch.save(choh_vit_recon, PATH_FOLDER+'tensors_entire.pt')
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
