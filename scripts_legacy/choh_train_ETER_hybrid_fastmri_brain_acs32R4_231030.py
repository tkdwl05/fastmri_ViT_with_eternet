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
import mySSIM

###choh
# from choh_dataloader_fastmri_brain_191218 import choh_fastmri_brain_multi_resize_R2
# from choh_dataloader_fastmri_brain_191219 import choh_fastmri_brain_multi_fullsize_acs32R4
# from choh_dataloader_fastmri_brain_191219 import choh_fastmri_brain_multi_fullsize_acs32R4_fullslice
# from choh_dataloader_fastmri_brain_200708 import choh_fastmri_brain_multi_fullsize_acs32R4_fullslice_train
# from choh_dataloader_fastmri_brain_200708 import choh_fastmri_brain_multi_fullsize_acs32R4_fullslice_valid
# from myDataloader_fastmri_brain_210324 import choh_fastmri_brain_hybrid_acs32R4_train, choh_fastmri_brain_hybrid_acs32R4_val
from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_train, choh_fastmri_brain_hybrid_ifft_acs32R4_val
from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2
from myDataloader_fastmri_brain_230425 import choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2
from myUNet_DF import UNet_choh_skip
from myUtils import myOptimizer, myCriterion
from myConfig_ETER_fastmri_brain_acs32R4_5 import *


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








# ETER hybrid, fastmri multi 16ch, fullsize, 384x384
class ETER_hybrid_GRU_DFU(nn.Module):
	def __init__(self):
		super(ETER_hybrid_GRU_DFU, self).__init__()
		num_in_x = N_INPUT_VERTICAL
		num_in_y = N_FREQ_ENCODING
		n_coil = N_RFCOIL
		num_out_x = N_OUT_X
		num_out_y = N_OUT_Y
		input_size = num_in_y*n_coil*2
		num_layers = 1 #2
		num_out1 = num_out_y*N_HIDDEN_LRNN_1
		num_in2 = num_in_x*N_HIDDEN_LRNN_1
		num_out2 = num_out_x*N_HIDDEN_LRNN_2
		# num_feat_ch = int(num_out2*2/num_out_x)
		num_feat_ch = int(num_out2*2/num_out_x) + N_RFCOIL*2
		n_hidden = N_HIDDEN_LRNN_2 + N_RFCOIL
		# print("num_out1 %d    num_out2 %d"%(num_out1, num_out2))

		self.num_in_x = num_in_x
		self.num_in_y = num_in_y
		self.num_layers = num_layers
		self.num_out1 = num_out1
		self.num_out2 = num_out2
		self.num_out_x = num_out_x
		self.num_out_y = num_out_y


		self.gru_h = nn.GRU(input_size, num_out1, num_layers, batch_first=True, bidirectional=True)
		self.gru_v = nn.GRU(num_in2*2, num_out2, num_layers, batch_first=True, bidirectional=True)

		# self.conv2d = nn.Conv2d(in_channels=num_feat_ch, out_channels=1, kernel_size=1, stride=1)
		# self.unet = UNet(in_channels=num_feat_ch, n_classes=1, depth=3, wf=6, batch_norm=False, up_mode='upconv')
		self.unet = UNet_choh_skip(in_channels=num_feat_ch, n_classes=1, depth=N_UNET_DEPTH, wf=6, batch_norm=False, up_mode='upconv', n_hidden=n_hidden)


	def forward(self, x, x_img):
		h_h0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out1).cuda()
		h_v0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out2).cuda()

		# print(x.shape)
		in_h = x.reshape([x.size(0), self.num_in_x, -1])
		# print('in_h shape {}'.format(in_h.shape))

		out_h, _ = self.gru_h(in_h, h_h0)
		# print('out_h shape {}'.format(out_h.shape))

		out_h = out_h.reshape([x.size(0), self.num_in_x, self.num_out_y,-1])
		# print('out_h shape {}'.format(out_h.shape))
		out_h = out_h.permute(0, 2, 1, 3)
		# print('out_h shape {}'.format(out_h.shape))
		out_h = out_h.reshape([x.size(0), self.num_out_y, -1])
		# print('out_h shape {}'.format(out_h.shape))


		out_v, _ = self.gru_v(out_h, h_v0)
		# print('out_v shape {}'.format(out_v.shape))
		out_v = out_v.reshape([x.size(0), self.num_out_y, self.num_out_x,-1])
		# print('out_v shape {}'.format(out_v.shape))
		out_v = out_v.permute(0, 3, 2, 1)
		# print('out_v shape {}'.format(out_v.shape))

		## merge multi feature
		# out = self.conv2d(out_v)
		# out = self.unet(out_v)
		in_cnn = torch.cat((out_v, x_img), dim=1)
		# print('in_cnn shape {}'.format(in_cnn.shape))
		out = self.unet(in_cnn)
		# print('out shape {}'.format(out.shape))


		return out






def main():
	print('\n  choh, train ETER hybrid, fastmri brain_multi 16ch, fullsize 384x384, acs32R4, @main')


	# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train(num_total_set = NUM_TRAIN_SET)	
	# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2(idx_start=IDX_START, num_total_set = NUM_TRAIN_SET)	
	
	# print(choh_data_train)
	# print(' len(choh_data_train) : %d'%len(choh_data_train))


	# trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True)
	# print(trainloader)
	# total_step = len(trainloader)

	choh_data_valid = choh_fastmri_brain_hybrid_ifft_acs32R4_val(num_total_set = 1)
	validloader = DataLoader(choh_data_valid, batch_size=BATCH_SIZE, shuffle=True)
	total_valid_step = len(validloader)




	eternet = ETER_hybrid_GRU_DFU().cuda()



	criterion = myCriterion(CRITERION)
	optimizer = myOptimizer(eternet, OPTIMIZER, LEARNING_RATE_ADAM)
	# optimizer = torch.optim.Adam(eternet.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)
	crit_ssim = mySSIM.SSIM().cuda()



	lambda_ssim_per_pixel = LAMBDA_SSIM_PER_PIXEL
	num_epochs = NUM_EPOCHS
	loss_prev = 1e8
	ssim_prev = 0
	validation_loss = np.zeros((num_epochs,1))
	training_loss = np.zeros( (num_epochs, 4800) )

	print('\n\n\n  start training iterations')
	for epoch in range(num_epochs):
		for n_set in range(NUM_TRAIN_SET):

			choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2(idx_start= NUM_VOL_PER_SET*n_set , idx_end=NUM_VOL_PER_SET*n_set+NUM_VOL_PER_SET)

			print(choh_data_train)
			print(' len(choh_data_train) : %d'%len(choh_data_train))
			trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True)
			print(trainloader)
			total_step = len(trainloader)

			for i_batch, sample_batched in enumerate(trainloader):

				data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
				data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
				data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)


				optimizer.zero_grad()
				out = eternet(data_in, data_in_img)

				# loss = criterion(out, data_ref)
				# loss.backward()

				# optimizer.step()

				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}'.format( epoch+1, num_epochs, i_batch+1, total_step, loss.item() ))

				### apply loss_ssim
				loss_pixel = criterion(out, data_ref)
				loss_ssim = 1-crit_ssim( out, data_ref)
				loss = loss_pixel + lambda_ssim_per_pixel*loss_ssim
				loss.backward()

				optimizer.step()

				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}'.format( epoch+1, num_epochs, i_batch+1, total_step, loss.item() ))
				# print('Epoch [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format( 
				# 	epoch+1, num_epochs, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item()  ))

				training_loss[epoch,n_set*total_step + i_batch] = loss_pixel.item()
				
				print('Epoch [{}/{}], n_SET [{}/{}], Step [{}/{}], loss: {:.6f}, pix: {:.6f}  1-ssim: {:.6f} '.format( 
					epoch+1, num_epochs, n_set+1, NUM_TRAIN_SET, i_batch+1, total_step, loss.item(), loss_pixel.item() , loss_ssim.item()  ))

				
			# training_loss[epoch,i_batch] = loss_pixel.item()
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

		with torch.no_grad():
			for i_batch, sample_batched in enumerate(validloader):
				data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
				data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
				data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)
				out = eternet(data_in, data_in_img)
				loss_pixel = criterion(out, data_ref)

			print('  epoch {}\t validation_loss: {:.6f}'.format(epoch+1, loss_pixel ) )
			validation_loss[epoch] = loss_pixel.item()



		torch.save( eternet, PATH_FOLDER+'tensors_%d.pt'%(epoch+1) )
		print('saving done...\n')



	torch.save(eternet, PATH_FOLDER+'tensors_entire.pt')
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
