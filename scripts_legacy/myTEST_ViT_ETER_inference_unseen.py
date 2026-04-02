
import os
import sys
import torch
import torchvision
import torch.nn as nn
import numpy as np
import time
# import h5py
from types import ModuleType
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import psutil

# Imports from local modules
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_ETER_ViT import choh_Decoder3_ETER_skip_up_tail
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

PATH_FOLDER = 'logs/240916_choh_ViT_ETER_skip_up_tail_B_ch256_fea16_nh10_ep100_cos_rtx2_24/'
PATH_SDB = '/mnt/sdb/choh/shared/W_python/myViT/'
if 'PATH_SDB' in globals():
	PATH_FOLDER = PATH_SDB + PATH_FOLDER

###### 		ENCODER PARAMETERS		######
NUM_VIT_ENCODER_HIDDEN = 768
NUM_VIT_ENCODER_LAYER = 12
NUM_VIT_ENCODER_MLP_SIZE = 3072
NUM_VIT_ENCODER_HEAD = 12

###### 		BIRNN PARAMETERS		######
NUM_ETER_HORI_HIDDEN = 10
NUM_ETER_VERT_HIDDEN = 10

###### 		DECODER PARAMETERS		######
NUM_VIT_DECODER_DIM_HEAD = 64
NUM_VIT_DECODER_DIM = 1280
NUM_VIT_DECODER_DEPTH = 12
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH = 256
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16

BATCH_SIZE = 4
N_OUT_X = 384
N_OUT_Y = 384

process = psutil.Process()

def choh_ifft2c_multi_ch(ksp):
	######## ifftshift fftshift, ifft2, ifftshift fftshift -> fftshift fftshift, ifft2, ifftshift ifftshift
	####### input ksp shape is: (num_slice, num_ch*2, num_dim, num_dim)

	num_ch_for_C = ksp.shape[1]//2
	C = np.zeros((ksp.shape[0], num_ch_for_C, ksp.shape[2], ksp.shape[3]),dtype=np.complex_)
	C = C + ksp[:,0:ksp.shape[1]:2,:,:] + ksp[:,1:ksp.shape[1]:2,:,:]*1j 

	signal = np.fft.fftshift( np.fft.fftshift(C, 2), 3)
	# result_shift = np.fft.ifft2(signal, axes=(-2, -1), norm="ortho" )
	result_shift = np.fft.ifft2(signal, axes=(-2, -1))
	result_complex = np.fft.ifftshift( np.fft.ifftshift(result_shift, 2), 3)

	result = np.zeros(ksp.shape)
	result[:,0:ksp.shape[1]:2,:,:] = result_complex.real
	result[:,1:ksp.shape[1]:2,:,:] = result_complex.imag

	return result

class choh_fastmri_brain_unseen(Dataset):
	def __init__(self, list_file='list_brain_unseen_10.txt'):
		print('\n  Dataset : choh_fastmri_brain_unseen')
		print('\n  choh_ifft2c_multi')

		N_OUTPUT = 384
		N_COIL_CH = 16
		n_dim = N_OUTPUT
		n_in_ch = N_COIL_CH*2		
		val_amp_X_img = 1e6 
		val_amp_X_ksp = 1e4 
		val_amp_Y = 1e6 
		num_slices = 16
		
		# Read the list file
		with open(list_file, 'r') as f:
			data = [line for line in f.readlines() if line.strip()]
		
		num_files = len(data)
		num_slices_total = 16 * num_files
		print("    num_slices {}  num_files {}".format(num_slices, num_files))
		print("       val_amp_X_ksp {}   val_amp_X_img {}   val_amp_Y {}\n".format(int(val_amp_X_ksp), int(val_amp_X_img), int(val_amp_Y)))

		path_matfolder_input_ksp = '/mnt/sda/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_unseen/'
		# path_matfolder_input_ksp = '/mnt/sdb/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# if not os.path.exists(path_matfolder_input_ksp):
		# 	path_matfolder_input_ksp = '/mnt/sdc/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
		# if not os.path.exists(path_matfolder_input_ksp):
		# 	path_matfolder_input_ksp = '/mnt/sda/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'

		X_ksp_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		X_img_train = np.zeros((num_slices_total, n_in_ch, n_dim, n_dim), dtype=np.float32)
		Y_train = np.zeros((num_slices_total, 1, n_dim, n_dim), dtype=np.float32)

		idx=0 
		idx_total_slice = 0

		for filename in data:
			filename = filename.strip()
			str_temp = '%s%s'%(path_matfolder_input_ksp, filename.replace('.npy', '_ksp.mat'))

			f_matfile = sio.loadmat(str_temp)
			f_imgNET = f_matfile['ksp_16ch']
			XX_ksp = f_imgNET ### (16, 384, 384, 16, 2)
			XX_ksp.astype('float32')
			num_slices = XX_ksp.shape[0]

			### ref image
			ksp_16ch = np.transpose( XX_ksp, (0, 3, 4, 1, 2) ) ### XX_ksp.shape (16, 384, 384, 16, 2) -> (16, 16, 2, 384, 384, )
			ksp_16ch = np.reshape( ksp_16ch, [num_slices, n_in_ch, n_dim, n_dim])	### ksp_16ch.shape (16, 16, 2, 384, 384) -> (16, 32, 384, 384)

			img_16ch = choh_ifft2c_multi_ch(ksp_16ch)
			Y_volume = np.zeros((num_slices, 1, n_dim, n_dim), dtype=np.float32)
			Y_volume = np.reshape( np.sqrt( np.sum( np.square(img_16ch), axis=1 ) ), [num_slices, 1, n_dim, n_dim])

			### aliased image
			X_ksp_volume = np.zeros((num_slices, n_dim, n_dim, N_COIL_CH, 2), dtype=np.float32)
			X_ksp_volume[:,:,3:384:4,:,:] = XX_ksp[:,:,3:384:4,:,:]
			X_ksp_volume[:,:,192-16:192+16,:,:] = XX_ksp[:,:,192-16:192+16,:,:]
			
			x_ksp_zf = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			x_ksp_zf = np.reshape( np.transpose( X_ksp_volume, (0, 3, 4, 1, 2) ), [num_slices, n_in_ch, n_dim, n_dim]) ### X_ksp_volume.shape (16, 384, 384, 16, 2) -> (16, 32, 384, 384, )
			img_alias_16ch = np.zeros((num_slices, n_in_ch, n_dim, n_dim), dtype=np.float32)
			img_alias_16ch = choh_ifft2c_multi_ch(x_ksp_zf)

			X_img_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_img*img_alias_16ch
			X_ksp_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_X_ksp*x_ksp_zf
			Y_train[idx_total_slice:idx_total_slice+num_slices,:,:,:] = val_amp_Y*Y_volume

			idx+=1
			idx_total_slice = idx_total_slice + num_slices

			mem_info = process.memory_info()
			mem_rss_gb = mem_info.rss / 1024 / 1024 /1024
			print("  loading done : {}\t{} {} {}\t{} GB".format(filename, idx, XX_ksp.shape, idx_total_slice, mem_rss_gb))

		self.label = Y_train[:idx_total_slice]
		self.data = X_ksp_train[:idx_total_slice]
		self.data_img = X_img_train[:idx_total_slice]
	def __len__(self):
		return self.label.shape[0]
	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'data_img': self.data_img[idx], 'label': self.label[idx]}		
		return sample

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(args):
	print('\n  choh, inference unseen, FastMRI, @main')
	print(PATH_FOLDER)

	vit_choh = choh_ViT(
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

	model = choh_Decoder3_ETER_skip_up_tail(
	encoder = vit_choh,
	eter_n_hori_hidden = NUM_ETER_HORI_HIDDEN,
	eter_n_vert_hidden = NUM_ETER_VERT_HIDDEN,
	decoder_dim = NUM_VIT_DECODER_DIM,      
	decoder_depth = NUM_VIT_DECODER_DEPTH,       
	decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
	decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
	decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
	).cuda()

	filename_trained_weight = 'tensors_entire.pt'
	if args.weight != 0:
		filename_idx = args.weight
		filename_trained_weight = 'tensors_%d.pt'%filename_idx
		if args.weight < 0:
			filename_idx = np.abs(args.weight)
			filename_trained_weight = 'tensors_pre%d.pt'%filename_idx
	print(PATH_FOLDER+filename_trained_weight)
	model = torch.load(PATH_FOLDER+filename_trained_weight, map_location="cuda")
	model.eval()

	print('\n number of params : {}\n'.format(get_n_params(model)))

	# Load the unseen dataset
	choh_data_test = choh_fastmri_brain_unseen(list_file='list_brain_unseen_10.txt')
	print(choh_data_test)
	print(' len(choh_data_test) : %d'%len(choh_data_test))

	testloader = DataLoader(choh_data_test, batch_size=BATCH_SIZE, shuffle=False)
	print(testloader)

	criterion = nn.MSELoss()

	inputs = []
	results = []
	refs = []

	with torch.no_grad():
		print('\n  start inference')
		for i_batch, sample_batched in enumerate(testloader):
			data_in = sample_batched['data']
			data_in = data_in.type(torch.cuda.FloatTensor)
			data_in = data_in.cuda()

			data_ref = sample_batched['label']
			data_ref = data_ref.type(torch.cuda.FloatTensor)
			data_ref = data_ref.cuda()

			data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)

			out = model(data_in_img, data_in)

			inputs = np.append( inputs, data_in_img.cpu().detach().numpy() )
			results = np.append( results, out.cpu().detach().numpy() )
			refs = np.append( refs, data_ref.cpu().detach().numpy() )

			loss = criterion(out, data_ref)
			print('  {} loss: {:.6f}'.format(i_batch, loss ) )

	inputs = np.reshape(inputs, [len(choh_data_test), 16*2*384, 384])
	results = np.reshape(results, [len(choh_data_test), N_OUT_X, N_OUT_Y])
	refs = np.reshape(refs, [len(choh_data_test), N_OUT_X, N_OUT_Y])
	print('inputs.shape {} results.shape {} refs.shape {}'.format(inputs.shape, results.shape, refs.shape))

	## saving files
	if not os.path.exists(PATH_FOLDER+ 'result'):
		os.makedirs(PATH_FOLDER+ 'result')
	
	filename_save_nparry = PATH_FOLDER + 'result/bj_results_unseen_all'
	np.save(filename_save_nparry, results)
	filename_save_nparry = PATH_FOLDER + 'result/bj_refs_unseen_all'
	np.save(filename_save_nparry, refs)
	print(f"Saved results to {PATH_FOLDER}result/bj_results_unseen_all.npy")
	print(f"Saved refs to {PATH_FOLDER}result/bj_refs_unseen_all.npy")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='choh, inference unseen')
	parser.add_argument('-w','--weight', type=int, default=0, help='weight of epoch of given number ')
	args = parser.parse_args()
	main(args)
