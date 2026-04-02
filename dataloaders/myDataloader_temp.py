


from __future__ import print_function, division
import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import scipy.io as sio
import sys

import psutil

process = psutil.Process()
mem_info = process.memory_info()



def choh_ifft2c_single(ksp):

	# signal = ifftshift(ifftshift(signal,1),2);
	# signal = ifft(signal,[],1);
	# result = ifft(signal,[],2);
	# result = ifftshift(ifftshift(result,1),2);

	C = np.zeros((ksp.shape[0],384, 384),dtype=np.complex_)
	C = np.squeeze( C + ksp[:,0,:,:] + ksp[:,1,:,:]*1j )
	# print(C.shape)

	# signal = np.fft.ifftshift( np.fft.fftshift(ksp, 2), 3)  ### FF (n, 2, kx, ky)
	signal = np.fft.ifftshift( np.fft.fftshift(C, 1), 2)  ### FF (n, 2, kx, ky)
	# result_shift = np.fft.ifft2(signal, axes=(-2, -1), norm="ortho" )
	result_shift = np.fft.ifft2(signal, axes=(-2, -1))
	result_complex = np.fft.ifftshift( np.fft.fftshift(result_shift, 1), 2)  ### FF (n, 2, kx, ky)
	# print(result_complex.shape)

	result = np.zeros(ksp.shape)
	result[:,0,:,:] = result_complex.real
	result[:,1,:,:] = result_complex.imag

	return result

def choh_ifft2c_multi(ksp):

	# signal = ifftshift(ifftshift(signal,1),2);
	# signal = ifft(signal,[],1);
	# result = ifft(signal,[],2);
	# result = ifftshift(ifftshift(result,1),2);

	C = np.zeros((ksp.shape[0],384, 384),dtype=np.complex_)
	C = np.squeeze( C + ksp[:,0,:,:] + ksp[:,1,:,:]*1j )
	# print(C.shape)

	# signal = np.fft.ifftshift( np.fft.fftshift(ksp, 2), 3)  ### FF (n, 2, kx, ky)
	signal = np.fft.ifftshift( np.fft.fftshift(C, 1), 2)  ### FF (n, 2, kx, ky)
	result_shift = np.fft.ifft2(signal, axes=(-2, -1), norm="ortho" )
	result_complex = np.fft.ifftshift( np.fft.fftshift(result_shift, 1), 2)  ### FF (n, 2, kx, ky)
	# print(result_complex.shape)

	result = np.zeros(ksp.shape)
	result[:,0,:,:] = result_complex.real
	result[:,1,:,:] = result_complex.imag

	return result


class choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2(Dataset):
	def __init__(self, idx_start = 0, num_total_set = 300):
		print('\n  Dataset : choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2')

		N_OUTPUT = 384
		N_COIL_CH = 16
		n_dim = N_OUTPUT
		n_in_ch = N_COIL_CH*2		
		val_amp_X_img = 1e5 # 551.4780*1e3
		val_amp_X_ksp = 1e3 # 551.4780
		val_amp_Y = 1e5 #1e3
		num_slices = 16
		num_slices_total = 16*num_total_set
		print("    num_slices {}  num_total_set {}".format(num_slices, num_total_set))
		# print("    num_total_set {}".format( num_total_set))
		print("       val_amp_X_ksp {}   val_amp_X_img {}   val_amp_Y {}\n".format(val_amp_X_ksp, val_amp_X_img, val_amp_Y))

		# path_matfolder_input_img = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_fullslice/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_kspZP/'
		# path_matfolder_label = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/npy_brain_multi_train_16ch_396ky_label/'
		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI/multicoil_train/mat_multicoil_train_ksp_extend_384_384/'

		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)

		file = open('list_brain_train_320.txt', 'r')		
		data = file.readlines()
		idx=0 #1 #0
		idx_segment = 0
		idx_total_slice = 0

		# data_range = data[320:379]
		# data_range = data[:num_total_set]
		data_range = data[idx_start:num_total_set]

		for filename in data_range:
			str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))
			# str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.mat\n', '.mat'))
			# print(str_temp)
			f_matfile = sio.loadmat(str_temp)
			f_imgNET = f_matfile['ksp_16ch']
			XX_ksp = f_imgNET ### (16, 384, 384, 16, 2)
			XX_ksp.astype('float32')
			num_slices = XX_ksp.shape[0]
			# print( XX_ksp.shape)
			img_full_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(XX_ksp[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_full_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )
			Y_volume = np.zeros((num_slices, 1, n_dim, n_dim), dtype=np.float32)
			Y_volume = np.reshape( np.sqrt( np.sum( np.square(img_full_16ch), axis=1 ) ), [num_slices, 1, n_dim, n_dim])

			X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
			X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
			X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
			# X_ksp_volume[:,:,7:384:8,:,:] = XX_ksp[:,:,7:384:8,:,:]
			# X_ksp_volume[:,:,192-8:192+8,:,:] = XX_ksp[:,:,192-8:192+8,:,:]
			img_alias_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(X_ksp_volume[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_alias_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )

			x_ksp_zf = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			x_ksp_zf = np.reshape( np.transpose( X_ksp_volume, (0, 3, 4, 1, 2) ), [num_slices, n_in_ch, n_dim, n_dim])



			X_img_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_img*img_alias_16ch
			X_ksp_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_ksp*x_ksp_zf
			Y_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_Y*Y_volume


			idx+=1
			idx_total_slice = idx_total_slice + num_slices

			mem_info = process.memory_info()
			mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
			print("  loading done : {}\t{} {} {}\t{} GB".format(filename.rstrip('.npy\n'), idx, XX_ksp.shape, idx_total_slice, mem_rss_gb))
			# print("  mem_info.rss : {} GB ".format(mem_rss_gb))
			# if idx==3:
			# 	break

		# self.label = Y_train
		# self.data = X_ksp_train
		# self.data_img = X_img_train
		self.label = Y_train[:idx_total_slice]
		self.data = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]
	def __len__(self):
		return self.label.shape[0]
	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
		return sample


class choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2(Dataset):
	def __init__(self, idx_start = 321, idx_end = 330):
		print('\n  Dataset : choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2')

		N_OUTPUT = 384
		N_COIL_CH = 16
		n_dim = N_OUTPUT
		n_in_ch = N_COIL_CH*2		
		val_amp_X_img = 1e5 # 551.4780*1e3
		val_amp_X_ksp = 1e3 # 551.4780
		val_amp_Y = 1e5 #1e3
		num_slices = 16
		num_slices_total = 16*(idx_end-idx_start)
		print("    num_slices {}  num_total_set {}".format(num_slices, idx_end-idx_start))
		# print("    num_total_set {}".format( num_total_set))
		print("       val_amp_X_ksp {}   val_amp_X_img {}   val_amp_Y {}\n".format(val_amp_X_ksp, val_amp_X_img, val_amp_Y))

		# path_matfolder_input_img = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_fullslice/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_kspZP/'
		# path_matfolder_label = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/npy_brain_multi_train_16ch_396ky_label/'
		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI/multicoil_train/mat_multicoil_train_ksp_extend_384_384/'

		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)

		file = open('list_brain_train_320.txt', 'r')		
		data = file.readlines()
		idx=0 #1 #0
		idx_segment = 0
		idx_total_slice = 0

		# data_range = data[320:379]
		# data_range = data[:num_total_set]
		data_range = data[idx_start:idx_end]

		for filename in data_range:
			str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))
			# str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.mat\n', '.mat'))
			# print(str_temp)
			f_matfile = sio.loadmat(str_temp)
			f_imgNET = f_matfile['ksp_16ch']
			XX_ksp = f_imgNET ### (16, 384, 384, 16, 2)
			XX_ksp.astype('float32')
			num_slices = XX_ksp.shape[0]
			# print( XX_ksp.shape)
			img_full_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(XX_ksp[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_full_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )
			Y_volume = np.zeros((num_slices, 1, n_dim, n_dim), dtype=np.float32)
			Y_volume = np.reshape( np.sqrt( np.sum( np.square(img_full_16ch), axis=1 ) ), [num_slices, 1, n_dim, n_dim])

			X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
			X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
			X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
			# X_ksp_volume[:,:,7:384:8,:,:] = XX_ksp[:,:,7:384:8,:,:]
			# X_ksp_volume[:,:,192-8:192+8,:,:] = XX_ksp[:,:,192-8:192+8,:,:]
			img_alias_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(X_ksp_volume[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_alias_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )

			x_ksp_zf = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			x_ksp_zf = np.reshape( np.transpose( X_ksp_volume, (0, 3, 4, 1, 2) ), [num_slices, n_in_ch, n_dim, n_dim])



			X_img_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_img*img_alias_16ch
			X_ksp_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_ksp*x_ksp_zf
			Y_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_Y*Y_volume


			idx+=1
			idx_total_slice = idx_total_slice + num_slices

			mem_info = process.memory_info()
			mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
			print("  loading done : {}\t{} {} {}\t{} GB".format(filename.rstrip('.npy\n'), idx, XX_ksp.shape, idx_total_slice, mem_rss_gb))
			# print("  mem_info.rss : {} GB ".format(mem_rss_gb))
			# if idx==3:
			# 	break

		# self.label = Y_train
		# self.data = X_ksp_train
		# self.data_img = X_img_train
		self.label = Y_train[:idx_total_slice]
		self.data = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]
	def __len__(self):
		return self.label.shape[0]
	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
		return sample








class choh_fastmri_brain_hybrid_ifft_acs32R4_train_v3(Dataset):
	def __init__(self, idx_start = 0, idx_end = 300):
		print('\n  Dataset : choh_fastmri_brain_hybrid_ifft_acs32R4_train_v3')

		N_OUTPUT = 384
		N_COIL_CH = 16
		n_dim = N_OUTPUT
		n_in_ch = N_COIL_CH*2		
		val_amp_X_img = 1e5 # 551.4780*1e3
		val_amp_X_ksp = 1e3 # 551.4780
		val_amp_Y = 1e5 #1e3
		num_slices = 16
		num_slices_total = 16*(idx_end-idx_start)
		print("    num_slices {}  num_total_set {}".format(num_slices, idx_end-idx_start))
		# print("    num_total_set {}".format( num_total_set))
		print("       val_amp_X_ksp {}   val_amp_X_img {}   val_amp_Y {}\n".format(val_amp_X_ksp, val_amp_X_img, val_amp_Y))

		# path_matfolder_input_img = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_fullslice/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_kspZP/'
		# path_matfolder_label = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/npy_brain_multi_train_16ch_396ky_label/'
		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI/multicoil_train/mat_multicoil_train_ksp_extend_384_384/'

		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)

		file = open('list_brain_train_320.txt', 'r')		
		data = file.readlines()
		idx=0 #1 #0
		idx_segment = 0
		idx_total_slice = 0

		# data_range = data[320:379]
		# data_range = data[:num_total_set]
		data_range = data[idx_start:idx_end]

		for filename in data_range:
			str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))
			# str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.mat\n', '.mat'))
			# print(str_temp)
			f_matfile = sio.loadmat(str_temp)
			f_imgNET = f_matfile['ksp_16ch']
			XX_ksp = f_imgNET ### (16, 384, 384, 16, 2)
			XX_ksp.astype('float32')
			num_slices = XX_ksp.shape[0]
			# print( XX_ksp.shape)
			img_full_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(XX_ksp[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_full_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )
			Y_volume = np.zeros((num_slices, 1, n_dim, n_dim), dtype=np.float32)
			Y_volume = np.reshape( np.sqrt( np.sum( np.square(img_full_16ch), axis=1 ) ), [num_slices, 1, n_dim, n_dim])

			X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
			X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
			X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
			# X_ksp_volume[:,:,7:384:8,:,:] = XX_ksp[:,:,7:384:8,:,:]
			# X_ksp_volume[:,:,192-8:192+8,:,:] = XX_ksp[:,:,192-8:192+8,:,:]
			img_alias_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(X_ksp_volume[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_alias_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )

			x_ksp_zf = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			x_ksp_zf = np.reshape( np.transpose( X_ksp_volume, (0, 3, 4, 1, 2) ), [num_slices, n_in_ch, n_dim, n_dim])



			X_img_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_img*img_alias_16ch
			X_ksp_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_ksp*x_ksp_zf
			Y_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_Y*Y_volume


			idx+=1
			idx_total_slice = idx_total_slice + num_slices

			mem_info = process.memory_info()
			mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
			print("  loading done : {}\t{} {} {}\t{} GB".format(filename.rstrip('.npy\n'), idx, XX_ksp.shape, idx_total_slice, mem_rss_gb))
			# print("  mem_info.rss : {} GB ".format(mem_rss_gb))
			# if idx==3:
			# 	break

		# self.label = Y_train
		# self.data = X_ksp_train
		# self.data_img = X_img_train
		self.label = Y_train[:idx_total_slice]
		self.data = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]
	def __len__(self):
		return self.label.shape[0]
	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
		return sample





class choh_fastmri_kneeee_hybrid_ifft_acs32R4_test(Dataset):
	def __init__(self, num_total_set = 2, idx_start = 193):
		print('\n  Dataset : choh_fastmri_kneeee_hybrid_ifft_acs32R4_test')

		N_OUTPUT = 384
		N_COIL_CH = 16
		n_dim = N_OUTPUT
		n_in_ch = N_COIL_CH*2		
		val_amp_X_img = 1e5 # 551.4780*1e3
		val_amp_X_ksp = 1e3 # 551.4780
		val_amp_Y = 1e5 #1e3
		num_slices = 16
		num_slices_total = 64*num_total_set
		print("    idx_start {}  num_total_set {}".format(idx_start, num_total_set))
		# print("    num_total_set {}".format( num_total_set))
		print("       val_amp_X_ksp {}   val_amp_X_img {}   val_amp_Y {}\n".format(val_amp_X_ksp, val_amp_X_img, val_amp_Y))

		# path_matfolder_input_img = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_fullslice/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_kspZP/'
		# path_matfolder_label = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/npy_brain_multi_train_16ch_396ky_label/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI/multicoil_train/mat_multicoil_train_ksp_extend_384_384/'

		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)

		file = open('list_knee_ksp384.txt', 'r')		
		data = file.readlines()
		idx=0 #1 #0
		idx_segment = 0
		idx_total_slice = 0

		# data_range = data[320:379]
		# data_range = data[:num_total_set]
		data_range = data[idx_start:idx_start+num_total_set]

		for filename in data_range:
			# str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))
			str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.mat\n', '.mat'))
			# print(str_temp)
			f_matfile = sio.loadmat(str_temp)
			f_imgNET = f_matfile['ksp_16ch']
			XX_ksp = f_imgNET ### (16, 384, 384, 16, 2)
			XX_ksp.astype('float32')
			num_slices = XX_ksp.shape[0]
			# print( XX_ksp.shape)
			img_full_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(XX_ksp[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_full_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )
			Y_volume = np.zeros((num_slices, 1, n_dim, n_dim), dtype=np.float32)
			Y_volume = np.reshape( np.sqrt( np.sum( np.square(img_full_16ch), axis=1 ) ), [num_slices, 1, n_dim, n_dim])

			X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
			X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
			X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
			# X_ksp_volume[:,:,7:384:8,:,:] = XX_ksp[:,:,7:384:8,:,:]
			# X_ksp_volume[:,:,192-8:192+8,:,:] = XX_ksp[:,:,192-8:192+8,:,:]
			img_alias_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			for cc in range(XX_ksp.shape[3]):
				ksp_1ch = np.zeros((num_slices, 2, n_dim, n_dim), dtype=np.float32)
				ksp_1ch = np.transpose( np.squeeze(X_ksp_volume[:,:,:,cc,:]), (0, 3, 1, 2) )
				img_alias_16ch[:,2*cc:2*cc+2,:,:] = choh_ifft2c_single( ksp_1ch )

			x_ksp_zf = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			x_ksp_zf = np.reshape( np.transpose( X_ksp_volume, (0, 3, 4, 1, 2) ), [num_slices, n_in_ch, n_dim, n_dim])



			X_img_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_img*img_alias_16ch
			X_ksp_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_ksp*x_ksp_zf
			Y_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_Y*Y_volume


			idx+=1
			idx_total_slice = idx_total_slice + num_slices

			mem_info = process.memory_info()
			mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
			print("  loading done : {}\t{} {} {} \t{}".format(filename.rstrip('.npy\n'), idx, XX_ksp.shape, idx_total_slice, mem_rss_gb))
			# if idx==3:
			# 	break

		# self.label = Y_train
		# self.data = X_ksp_train
		# self.data_img = X_img_train
		self.label = Y_train[:idx_total_slice]
		self.data = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]
	def __len__(self):
		return self.label.shape[0]
	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
		return sample







def main():
	print('  choh')

	mem_info = process.memory_info()
	mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
	print("  mem_info.rss : {} GB ".format(mem_rss_gb))


	choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train(num_total_set = 2)

	print(choh_data_train)
	print(' len(choh_data_train) : %d'%len(choh_data_train))

	trainloader = DataLoader(choh_data_train, batch_size=8, shuffle=False)
	# trainloader = DataLoader(choh_data_train, batch_size=8, shuffle=True)
	print(trainloader)


	mem_info = process.memory_info()
	mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
	print("  mem_info.rss : {} GB ".format(mem_rss_gb))

	# del choh_data_train
	# mem_info = process.memory_info()
	# mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
	# print("  mem_info.rss : {} GB ".format(mem_rss_gb))

	# del trainloader
	# mem_info = process.memory_info()
	# mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
	# print("  mem_info.rss : {} GB ".format(mem_rss_gb))




	for i_batch, sample_batched in enumerate(trainloader):
		print(i_batch, sample_batched['data'].size(), sample_batched['label'].size())

		# if i_batch == 10:
		plt.figure()
		img_batch = sample_batched['label']
		img_disp = np.squeeze(img_batch[0,0,:,:])

		# img_disp = sample_batched['label'][7,:,:]
		plt.subplot(141)
		plt.imshow(img_disp)
		# plt.axis('off')
		plt.colorbar()
		# plt.ioff()
		plt.subplot(142)
		img_disp = np.squeeze(sample_batched['data'][0,0,:,:])
		plt.imshow(img_disp)
		plt.colorbar()

		plt.subplot(143)
		img_disp = np.squeeze(sample_batched['data_img'][0,0,:,:])
		plt.imshow(img_disp)
		plt.colorbar()

		# plt.subplot(144)
		# img_disp = np.squeeze( img_batch[0,0,:,:]-sample_batched['data'][0,0,:,:] )
		# plt.imshow(img_disp)
		# plt.colorbar()

		plt.show()


		# plt.figure()
		# img_disp = np.squeeze(sample_batched['data'][0,0,:,:])
		# plt.imshow(img_disp)
		# plt.colorbar()
		# plt.show()

			# break
		if i_batch == 4:
			break

if __name__ == '__main__':
    main()



















