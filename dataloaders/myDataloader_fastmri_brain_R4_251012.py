


from __future__ import print_function, division
import os
import gc
import time
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import einops

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import scipy.io as sio
import sys

import psutil

process = psutil.Process()
mem_info = process.memory_info()






def choh_ifft2c_multi_ch(ksp):
	######## ifftshift fftshift, ifft2, ifftshift fftshift -> fftshift fftshift, ifft2, ifftshift ifftshift
	####### input ksp shape is: (num_slice, num_ch*2, num_dim, num_dim)

	num_ch_for_C = ksp.shape[1]//2
	C = np.zeros((ksp.shape[0], num_ch_for_C, ksp.shape[2], ksp.shape[3]),dtype=np.complex128)
	C = C + ksp[:,0:ksp.shape[1]:2,:,:] + ksp[:,1:ksp.shape[1]:2,:,:]*1j 

	signal = np.fft.fftshift( np.fft.fftshift(C, 2), 3)
	# result_shift = np.fft.ifft2(signal, axes=(-2, -1), norm="ortho" )
	result_shift = np.fft.ifft2(signal, axes=(-2, -1))
	result_complex = np.fft.ifftshift( np.fft.ifftshift(result_shift, 2), 3)

	result = np.zeros(ksp.shape)
	result[:,0:ksp.shape[1]:2,:,:] = result_complex.real
	result[:,1:ksp.shape[1]:2,:,:] = result_complex.imag

	return result


class choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_2(Dataset):
	def __init__(self, idx_start=0, idx_end=300, disable_gc=True):
		print('\n  @ myDataloader_fastmri_brain_R4_251012.py ')
		print('  Dataset : choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_2')
		print('  choh_ifft2c_multi')

		N_OUTPUT    = 384
		N_COIL_CH   = 16
		n_dim       = N_OUTPUT
		n_in_ch     = N_COIL_CH * 2   # real/imag → 2배
		val_amp_X_img = 1e6
		val_amp_X_ksp = 1e4
		val_amp_Y     = 1e6

		num_slices_per_file = 16  # 파일당 슬라이스 수(코드 상 고정)
		num_files           = idx_end - idx_start
		num_slices_total    = num_slices_per_file * num_files

		print(f"    num_slices {num_slices_per_file}  num_total_set {num_files} "
			  f"idx_start {idx_start} idx_end {idx_end}")
		print(f"       val_amp_X_ksp {int(val_amp_X_ksp)}   "
			  f"val_amp_X_img {int(val_amp_X_img)}   val_amp_Y {int(val_amp_Y)}\n")

		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		if not os.path.exists(path_matfolder_input_ksp):
			path_matfolder_input_ksp = '/mnt/sdc/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# path_matfolder_input_ksp = './'  # 필요 시 변경

		# ── 최종 배열 선할당 ───────────────────────────────────────────────
		X_ksp_train = np.empty((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.empty((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train     = np.empty((num_slices_total, 1,       n_dim, n_dim), dtype=np.float32)

		# ── 워크스페이스(임시) 배열 사전 할당 & 재사용 ─────────────────────
		X_ksp_volume_tmp = np.zeros((num_slices_per_file, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
		x_ksp_zf_tmp     = np.empty((num_slices_per_file, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_volume_tmp     = np.empty((num_slices_per_file, 1, n_dim, n_dim), dtype=np.float32)

		if disable_gc:
			gc.disable()
		process = psutil.Process(os.getpid())

		with open('list_brain_train_320.txt', 'r') as file:
			data = file.readlines()

		idx_total_slice = 0

		for f_idx, filename in enumerate(data[idx_start:idx_end], start=1):
			fname_mat = os.path.join(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))

			t0 = time.perf_counter()
			# loadmat → 바로 float32/contiguous
			f_matfile = sio.loadmat(fname_mat)
			XX_ksp = np.asarray(f_matfile['ksp_16ch'], dtype=np.float32, order='C')  # (16,384,384,16,2)
			num_slices = XX_ksp.shape[0]  # 보통 16

			# ── 1) Ref 이미지(SoS) ────────────────────────────────────────
			# (16,384,384,16,2) → (16,16,2,384,384) → (16,32,384,384)
			ksp_16ch = np.moveaxis(XX_ksp, (3, 4), (1, 2)).reshape(num_slices, n_in_ch, n_dim, n_dim)
			img_16ch = choh_ifft2c_multi_ch(ksp_16ch)  # float32 가정

			# Y = sqrt(sum(img_16ch**2, axis=1))  (임시 최소화)
			np.square(img_16ch, out=img_16ch)  # in-place
			Y_volume_tmp[...] = np.sqrt(np.sum(img_16ch, axis=1, dtype=np.float32, keepdims=True))

			# ── 2) Aliased k-space 구성 (R=4 with ACS 32) ─────────────────
			X_ksp_volume_tmp.fill(0.0)
			X_ksp_volume_tmp[:, :, 3:384:4, :, :]     = XX_ksp[:, :, 3:384:4, :, :]
			X_ksp_volume_tmp[:, :, 192-16:192+16, :, :] = XX_ksp[:, :, 192-16:192+16, :, :]

			# (16,384,384,16,2) → (16,16,2,384,384) → (16,32,384,384)
			x_ksp_zf_view = np.moveaxis(X_ksp_volume_tmp, (3, 4), (1, 2)).reshape(num_slices, n_in_ch, n_dim, n_dim)
			# contiguous 보장(필요 시) 및 재사용 버퍼로 복사
			np.copyto(x_ksp_zf_tmp, x_ksp_zf_view)

			img_alias_16ch = choh_ifft2c_multi_ch(x_ksp_zf_tmp)

			# ── 3) 스케일 적용: out= 로 임시배열 없이 목적지에 바로 기록 ──
			sl = slice(idx_total_slice, idx_total_slice + num_slices)
			np.multiply(img_alias_16ch, val_amp_X_img, out=X_img_train[sl])
			np.multiply(x_ksp_zf_tmp,  val_amp_X_ksp, out=X_ksp_train[sl])
			np.multiply(Y_volume_tmp,  val_amp_Y,     out=Y_train[sl])

			idx_total_slice += num_slices

			# 큰 임시 참조 제거 (GC는 비활성이라도 수동 수거 가능)
			del f_matfile, XX_ksp, ksp_16ch, img_16ch, x_ksp_zf_view, img_alias_16ch

			# 주기적 수거로 파편화/레퍼런스 잔존 완화
			if (f_idx % 8) == 0:
				gc.collect()

			mem_rss_gb = process.memory_info().rss / (1024**3)
			print(f"  loaded: {filename.strip():<40}  +{num_slices:2d}  "
				  f"total={idx_total_slice:<5d}  time={time.perf_counter()-t0:.2f}s  RSS={mem_rss_gb:.1f} GB")

		if disable_gc:
			gc.enable()
			gc.collect()

		# 실제 로드 수에 맞게 슬라이스(예방적)
		self.label    = Y_train[:idx_total_slice]
		self.data     = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]

	def __len__(self):
		return self.label.shape[0]

	def __getitem__(self, idx):
		return {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}

# class choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_2(Dataset):
# 	def __init__(self, idx_start = 0, idx_end = 300):
# 		########### val_amp modified 
# 		# 		val_amp_X_img = 1e5 -> 1e6
# 		# 		val_amp_X_ksp = 1e3 -> 1e4
# 		# 		val_amp_Y = 1e5 -> 1e6
# 		print('\n  @ myDataloader_fastmri_brain_R8_251012.py ')
# 		print('  Dataset : choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_2')
# 		print('  choh_ifft2c_multi')

# 		N_OUTPUT = 384
# 		N_COIL_CH = 16
# 		n_dim = N_OUTPUT
# 		n_in_ch = N_COIL_CH*2		
# 		val_amp_X_img = 1e6 #1e5 # 551.4780*1e3
# 		val_amp_X_ksp = 1e4 #1e3 # 551.4780
# 		val_amp_Y = 1e6 # 1e5 #1e3
# 		num_slices = 16
# 		num_slices_total = 16*(idx_end-idx_start)
# 		print(f"    num_slices {num_slices}  num_total_set {idx_end-idx_start} idx_start {idx_start} idx_end {idx_end}")
# 		print(f"       val_amp_X_ksp {int(val_amp_X_ksp)}   val_amp_X_img {int(val_amp_X_img)}   val_amp_Y {int(val_amp_Y)}\n")

# 		path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
# 		if not os.path.exists(path_matfolder_input_ksp):
# 			path_matfolder_input_ksp = '/mnt/sdc/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
# 		# path_matfolder_input_ksp = './'

# 		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
# 		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
# 		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)
		
# 		X_ksp_volume_tmp = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
# 		x_ksp_zf_tmp     = np.empty((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
# 		Y_volume_tmp     = np.empty((num_slices, 1, n_dim, n_dim), dtype=np.float32)


# 		file = open('list_brain_train_320.txt', 'r')		
# 		data = file.readlines()
# 		idx=0 #1 #0
# 		idx_segment = 0
# 		idx_total_slice = 0

# 		data_range = data[idx_start:idx_end]

		
# 		import gc
# 		# if disable_gc:
# 		# 	gc.disable()
# 		process = psutil.Process(os.getpid())

# 		with open('list_brain_train_320.txt', 'r') as file:
# 			data = file.readlines()

# 		idx_total_slice = 0

# 		for f_idx, filename in enumerate(data[idx_start:idx_end], start=1):
# 			fname_mat = os.path.join(path_matfolder_input_ksp, filename.replace('.npy\n', '_ksp.mat'))

# 			t0 = time.perf_counter()
# 			# loadmat → 바로 float32/contiguous
# 			f_matfile = sio.loadmat(fname_mat)
# 			XX_ksp = np.asarray(f_matfile['ksp_16ch'], dtype=np.float32, order='C')  # (16,384,384,16,2)
# 			num_slices = XX_ksp.shape[0]  # 보통 16

# 			# ── 1) Ref 이미지(SoS) ────────────────────────────────────────
# 			# (16,384,384,16,2) → (16,16,2,384,384) → (16,32,384,384)
# 			ksp_16ch = np.moveaxis(XX_ksp, (3, 4), (1, 2)).reshape(num_slices, n_in_ch, n_dim, n_dim)
# 			img_16ch = choh_ifft2c_multi_ch(ksp_16ch)  # float32 가정

# 			# Y = sqrt(sum(img_16ch**2, axis=1))  (임시 최소화)
# 			np.square(img_16ch, out=img_16ch)  # in-place
# 			Y_volume_tmp[...] = np.sqrt(np.sum(img_16ch, axis=1, dtype=np.float32, keepdims=True))

# 			# ── 2) Aliased k-space 구성 (R=4 with ACS 32) ─────────────────
# 			X_ksp_volume_tmp.fill(0.0)
# 			# X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
# 			# X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
# 			X_ksp_volume_tmp[:, :, 3:384:4, :, :]     = XX_ksp[:, :, 3:384:4, :, :]
# 			X_ksp_volume_tmp[:, :, 192-16:192+16, :, :] = XX_ksp[:, :, 192-16:192+16, :, :]

# 			# (16,384,384,16,2) → (16,16,2,384,384) → (16,32,384,384)
# 			x_ksp_zf_view = np.moveaxis(X_ksp_volume_tmp, (3, 4), (1, 2)).reshape(num_slices, n_in_ch, n_dim, n_dim)
# 			# contiguous 보장(필요 시) 및 재사용 버퍼로 복사
# 			np.copyto(x_ksp_zf_tmp, x_ksp_zf_view)

# 			img_alias_16ch = choh_ifft2c_multi_ch(x_ksp_zf_tmp)

# 			# ── 3) 스케일 적용: out= 로 임시배열 없이 목적지에 바로 기록 ──
# 			sl = slice(idx_total_slice, idx_total_slice + num_slices)
# 			np.multiply(img_alias_16ch, val_amp_X_img, out=X_img_train[sl])
# 			np.multiply(x_ksp_zf_tmp,  val_amp_X_ksp, out=X_ksp_train[sl])
# 			np.multiply(Y_volume_tmp,  val_amp_Y,     out=Y_train[sl])

# 			idx_total_slice += num_slices

# 			# 큰 임시 참조 제거 (GC는 비활성이라도 수동 수거 가능)
# 			del f_matfile, XX_ksp, ksp_16ch, img_16ch, x_ksp_zf_view, img_alias_16ch

# 			# 주기적 수거로 파편화/레퍼런스 잔존 완화
# 			if (f_idx % 8) == 0:
# 				gc.collect()

# 			mem_rss_gb = process.memory_info().rss / (1024**3)
# 			print(f"  loaded: {filename.strip():<40}  +{num_slices:2d}  "
# 				  f"total={idx_total_slice:<5d}  time={time.perf_counter()-t0:.2f}s  RSS={mem_rss_gb:.1f} GB")

# 		# if disable_gc:
# 		# 	gc.enable()
# 		# 	gc.collect()

# 		# 실제 로드 수에 맞게 슬라이스(예방적)
# 		self.label    = Y_train[:idx_total_slice]
# 		self.data     = X_ksp_train[:idx_total_slice]
# 		self.data_img = X_img_train[:idx_total_slice]
# 	def __len__(self):
# 		return self.label.shape[0]
# 	def __getitem__(self, idx):
# 		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
# 		return sample







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


	# choh_data_train = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4(num_total_set = 2)
	choh_data_train = choh_fastmri_brain_hybrid_ifft_acs16R8_train_v4_1(	idx_start = 320, idx_end = 321)

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
		# img_disp = np.squeeze(sample_batched['data'][0,0,:,:])
		img_disp = np.abs( np.squeeze(sample_batched['data'][0,0,:,:]) )
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



















