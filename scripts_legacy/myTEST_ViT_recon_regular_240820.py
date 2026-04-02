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



### choh
# import u_choh_SSIM
# from myUNet_DF import UNet_choh_skip
# # from myDataloader_fastmri_brain_210324 import choh_fastmri_brain_hybrid_acs32R4_test
# from myDataloader_fastmri_brain_random_220316 import choh_fastmri_brain_hybrid_ifft_random_acs32R4_train
# from myDataloader_fastmri_brain_random_220316 import choh_fastmri_brain_hybrid_ifft_random_acs32R4_val
# from myDataloader_fastmri_brain_random_220316 import choh_fastmri_brain_hybrid_ifft_random_acs32R4_test
# from myDataloader_fastmri_brain_random_220322 import choh_fastmri_brain_hybrid_ifft_random_acs32R4_toggle_test_single
# from myDataloader_fastmri_brain_fixed_220815 import choh_fastmri_brain_hybrid_ifft_fixed_acs32R4_toggle_test	##20220819
# from myDataloader_fastmri_brain_fixed_231022 import choh_fastmri_brain_hybrid_ifft_fixed_acs32R4_toggle_v2
# from myDataloader_fastmri_brain_random_231027 import choh_fastmri_brain_hybrid_ifft_random_acs32R4_toggle_train_v2
from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4
from myDataloader_fastmri_brain_240817 import choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1

from u_choh_model_choh_ViT_for_image_reconstruction import choh_ViT_for_image_reconstruction
from u_choh_model_choh_ViT_for_image_reconstruction import choh_ViT_for_image_reconstruction_with_tail


from torch.utils.data import DataLoader
from torch.autograd import Variable




# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(device)




## choh_ViT_for_image_reconstruction, R4 regular 
# PATH_FOLDER = 'logs/240817_choh_ViT_recon_t300_lssim02_L1reg071_ep30_rtx2_1/'
# PATH_FOLDER = 'logs/240817_choh_ViT_recon_ViT_L_size_t300_lssim02_L1reg071_ep30_rtx2_2/'
PATH_FOLDER = 'logs/240821_choh_ViT_recon_ViT_B_size_t300_lssim02_L1reg071_ep30_rtx2_1/'
PATH_FOLDER = 'logs/240821_choh_ViT_recon_ViT_L_size_t300_lssim02_L1reg071_ep30_rtx2_2/'
PATH_FOLDER = 'logs/240822_choh_ViT_recon_ViT_H_size_t300_v41_lssim02_L1reg071_ep30_rtx2_5/'
PATH_FOLDER = 'logs/240822_choh_ViT_recon_ViT_H_size_t300_v41_lssim02_L1reg071_ep30_rtx2_5/'
# PATH_FOLDER = 'logs/240822_choh_ViT_recon_ViT_B_size_t300_v41_lssim02_L1reg071_ep30_rtx2_6/'
# ## choh_ViT_for_image_reconstruction_with_tail
# PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_L_size_fmlp_4_t300_lssim02_L1reg071_ep30_rtx2_7/'
# PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_B_size_fmlp_8_t300_lssim02_L1reg071_ep30_rtx2_8/'
# PATH_FOLDER = 'logs/240825_choh_ViT_recon_tail_ViT_H_size_fmlp_2_t300_lssim02_L1reg071_ep30_rtx2_9/'
# PATH_FOLDER = 'logs/temp/'


###### 		ViT PARAMETERS		######
### this params equal to ViT-Large	### this params equal to ViT-Large	### this params equal to ViT-Large
NUM_VIT_HIDDEN = 1024
NUM_VIT_LAYER = 24
NUM_VIT_MLP_SIZE = 4096
NUM_VIT_HEAD = 16
NUM_VIT_FINAL_MLP_DIM = 4
### this params equal to ViT-Base	### this params equal to ViT-Base	### this params equal to ViT-Base
NUM_VIT_HIDDEN = 768
NUM_VIT_LAYER = 12
NUM_VIT_MLP_SIZE = 3072
NUM_VIT_HEAD = 12
NUM_VIT_FINAL_MLP_DIM = 8
### this params equal to ViT-Huge	### this params equal to ViT-Huge	### this params equal to ViT-Huge
NUM_VIT_HIDDEN = 1280
NUM_VIT_LAYER = 32
NUM_VIT_MLP_SIZE = 5120
NUM_VIT_HEAD = 16
NUM_VIT_FINAL_MLP_DIM = 2








BATCH_SIZE = 4 #16 #32 #8 #4

N_OUT_X = 384
N_OUT_Y = 384









def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(args):
	print('\n  choh, test, FastMRI, @main')
	print(PATH_FOLDER)
	idx  = args.idx #101
	idx_patient = args.patient #1
	flag_cmap = args.cmap
	value_amplify = args.amp
	flag_type = args.set


	model = choh_ViT_for_image_reconstruction(
	image_size = (384, 384), 
	patch_size = (32, 32),
	num_classes = 384*384,
	dim = NUM_VIT_HIDDEN,
	depth = NUM_VIT_LAYER,
	heads = NUM_VIT_HEAD,
	mlp_dim = NUM_VIT_MLP_SIZE,
	channels=32,
	dropout = 0.1,
	emb_dropout = 0.1
	).cuda()

	# choh_vit_recon = choh_ViT_for_image_reconstruction_with_tail(
	# image_size = (384, 384), 
	# patch_size = (32, 32),
	# num_classes = NUM_VIT_FINAL_MLP_DIM,
	# dim = NUM_VIT_HIDDEN,
	# depth = NUM_VIT_LAYER,
	# heads = NUM_VIT_HEAD,
	# mlp_dim = NUM_VIT_MLP_SIZE,
	# channels=32,
	# dropout = 0.1,
	# emb_dropout = 0.1
	# ).cuda()










	# model = oETER_GRU_DFU()
	filename_trained_weight = 'tensors_entire.pt'
	if args.weight is not 0:
		filename_idx = args.weight
		filename_trained_weight = 'tensors_%d.pt'%filename_idx
		if args.weight < 0:
			filename_idx = np.abs(args.weight)
			filename_trained_weight = 'tensors_pre%d.pt'%filename_idx
	print(PATH_FOLDER+filename_trained_weight)
	model = torch.load(PATH_FOLDER+filename_trained_weight, map_location="cuda")
	# model = torch.load(PATH_FOLDER+'tensors_entire.pt', map_location="cuda")
	model.eval()

	print(' number of params : {}'.format(get_n_params(model)))





	if flag_type=='testset':
		choh_data_test = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4(idx_start=320 , idx_end=320+59)
	# elif flag_type=='train':
	# 	choh_data_test = choh_fastmri_brain_hybrid_ifft_random_acs32R4_train(num_total_set=idx_patient)
	elif flag_type=='single':
		choh_data_test = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4(idx_start=320 , idx_end=321)
	elif flag_type=='v41':
		choh_data_test = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1(idx_start=320 , idx_end=321)
	elif flag_type=='p55':
		print('\n\ntestset : p55_test')
		choh_data_test = choh_fastmri_brain_hybrid_ifft_acs32R4_train_v4_1(idx_start=320+54 , idx_end=320+55)		### p55
	# elif flag_type=='fixed':
	# 	print(flag_type)
	# 	# choh_data_test = choh_fastmri_brain_hybrid_ifft_fixed_acs32R4_toggle_test()
	# 	# choh_data_test = choh_fastmri_brain_hybrid_ifft_fixed_acs32R4_toggle_v2(idx_start=320, idx_end=320+1)
	# 	choh_data_test = choh_fastmri_brain_hybrid_ifft_fixed_acs32R4_toggle_v2(idx_start=320, idx_end=320+59)
	# elif flag_type=='random':
	# 	print(flag_type)
	# 	choh_data_test = choh_fastmri_brain_hybrid_ifft_random_acs32R4_toggle_train_v2(idx_start=320, idx_end=320+1)
	# elif flag_type=='random_all':
	# 	print(flag_type)
	# 	choh_data_test = choh_fastmri_brain_hybrid_ifft_random_acs32R4_toggle_train_v2(idx_start=320, idx_end=320+59)
	else:
		choh_data_test = choh_icsl216_undersample_center_single(idx=idx_patient)
	print(choh_data_test)
	print(' len(choh_data_test) : %d'%len(choh_data_test))


	testloader = DataLoader(choh_data_test, batch_size=BATCH_SIZE, shuffle=False)
	print(testloader)
	total_step = len(testloader)

	criterion = nn.MSELoss()

	inputs = []
	results = []
	refs = []

	with torch.no_grad():
		print('\n  start inferece')
		for i_batch, sample_batched in enumerate(testloader):
			data_in = sample_batched['data']
			data_in = data_in.type(torch.cuda.FloatTensor)
			data_in = data_in.cuda()
			# data_in = data_in.type(torch.FloatTensor)

			data_ref = sample_batched['label']
			data_ref = data_ref.type(torch.cuda.FloatTensor)
			data_ref = data_ref.cuda()
			# data_ref = data_ref.type(torch.FloatTensor)

			data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
			# print(' data_in_img.shape {}'.format(data_in_img.shape))

			# print(data_in.shape)
			# out = model(data_in, data_in_img)
			# out = model(data_in)
			out = model(data_in_img)


			# inputs = np.append( inputs, data_in.cpu().detach().numpy() )
			inputs = np.append( inputs, data_in_img.cpu().detach().numpy() )
			results = np.append( results, out.cpu().detach().numpy() )
			refs = np.append( refs, data_ref.cpu().detach().numpy() )
			# results.append( out.cpu().detach().numpy() )
			# refs.append( data_ref.cpu().detach().numpy() )

			loss = criterion(out, data_ref)



			print('  {} loss: {:.6f}'.format(i_batch, loss ) )
	inputs = np.reshape(inputs, [len(choh_data_test), 16*2*384, 384])
	results = np.reshape(results, [len(choh_data_test), N_OUT_X, N_OUT_Y])
	refs = np.reshape(refs, [len(choh_data_test), N_OUT_X, N_OUT_Y])
	print('inputs.shape {} results.shape {} refs.shape {}'.format(inputs.shape, results.shape, refs.shape))



	flag_while = True
	while flag_while:
		print(' flag_while %r'%flag_while)
		print('  patient : %d, \tidx : %d, value_amplify : %f\n'%(idx_patient, idx, value_amplify))

		plt.figure()
		num_amplify_diff = 1 #5
		num_amplify_diff = value_amplify


		# img_input = np.squeeze( inputs.numpy()[idx,:,:128] )
		# img_pred = np.squeeze( results.numpy()[idx,:,:,:] )
		# img_truth = np.squeeze( refs.numpy()[idx,:,:,:] )

		# img_input = np.squeeze( inputs[idx,:,:128] ) ### wrong!
		# img_input = np.squeeze( inputs[idx,:,:]) ### start:stop:step
		# img_input = np.squeeze( inputs[idx,:,0:384*16*2:32]) ### start:stop:step
		img_input = np.squeeze( inputs[idx,0:384,:]) ### start:stop:step
		img_pred = np.squeeze( results[idx,:,:] )
		img_truth = np.squeeze( refs[idx,:,:] )




		# plt.subplot(4,2,1)
		# # plt.imshow(img_pred, aspect='auto') # aspect='equal'
		# img_input = np.squeeze( inputs[idx,:,0:384]) ### start:stop:step
		# print(' img_input.shape {}'.format(img_input.shape))
		# plt.imshow(img_input, aspect='equal', cmap=flag_cmap) # aspect='equal'
		# plt.title('img_pred')
		# plt.colorbar()

		# plt.subplot(4,2,2)
		# img_input = np.squeeze( inputs[idx,:,0:384*2:2]) ### start:stop:step
		# print(' img_input.shape {}'.format(img_input.shape))
		# plt.imshow(img_input, aspect='equal', cmap=flag_cmap)
		# plt.title('img_truth')
		# plt.colorbar()

		# plt.subplot(4,2,3)
		# img_input = np.squeeze( inputs[idx,:,0:384*32:32]) ### start:stop:step
		# print(' img_input.shape {}'.format(img_input.shape))
		# plt.imshow(img_input, aspect='equal', cmap=flag_cmap)
		# plt.title('img_input')
		# plt.colorbar()

		# plt.subplot(4,2,4)
		# plt.imshow(np.abs(img_truth-img_pred), aspect='equal', cmap=flag_cmap)
		# plt.title('diff')
		# plt.colorbar()
		# plt.show()
		
		
		plt.subplot(4,2,1)
		# plt.imshow(img_pred, aspect='auto') # aspect='equal'
		plt.imshow(img_pred, aspect='equal', cmap=flag_cmap) # aspect='equal'
		plt.title('img_pred')
		plt.colorbar()

		plt.subplot(4,2,2)
		plt.imshow(img_truth, aspect='equal', cmap=flag_cmap)
		plt.title('img_truth')
		plt.colorbar()

		plt.subplot(4,2,3)
		plt.imshow(img_input, aspect='equal', cmap=flag_cmap)
		plt.title('img_input')
		plt.colorbar()

		plt.subplot(4,2,4)
		plt.imshow(np.abs(img_truth-img_pred), aspect='equal', cmap=flag_cmap)
		plt.title('diff')
		plt.colorbar()
		plt.show()




		### min max GT
		min_temp = np.min(np.min(img_truth))
		max_temp = np.max(np.max(img_truth))
		print('min_max_GT: %f \t %f'%( min_temp, max_temp))
		min_max_GT = ( min_temp, max_temp)
		### min max Pred
		min_temp = np.min(np.min(img_pred))
		max_temp = np.max(np.max(img_pred))
		print('min_max_Pred: %f \t %f'%( min_temp, max_temp))
		min_max_Pred = ( min_temp, max_temp)



		flag_evalu=True
		if flag_evalu:
			# print(img_truth.type())
			# print(img_pred.type())
			print(img_truth.dtype)
			print(img_pred.dtype)
			# img_truth = np.float32(img_truth)

			from skimage.measure import compare_ssim as ssim
			val_maximum = np.amax((img_truth.max(), img_pred.max()))

			ssim_between = ssim(img_truth, img_pred, data_range=val_maximum)
			# print('  ssim_between : %f'%ssim_between)

			# numera = np.sum(np.sum(np.square(np.subtract(np.float32(img_truth), img_pred))))
			numera = np.sum(np.sum(np.square(np.subtract(img_truth, img_pred))))
			denomi = np.sum(np.sum(np.square(img_truth)))
			nmse = numera/denomi
			# print('  nMSE : %f'%nmse)
			# print('  MSE : %f'%(numera/128/128))



			# from sklearn.metrics import mean_squared_error
			# val_mse = mean_squared_error(img_truth, img_pred)
			# print('  val_mse : %f'%val_mse)

			# val_mse = np.square(np.subtract(img_truth, img_pred)).mean()
			# print('  val_mse : %f'%val_mse)
			print('\t{:.6f}\t{:.6f}\tssim_between, nMSE'.format(ssim_between, nmse))




		# , clim=(-0.07, 0.43)
		num_display_range = (0, 0.34)
		# num_display_range = (0, 0.54)
		# num_display_range = (0, 0.62)
		num_display_range = (0, 1.0)
		# num_display_range = (-3.2, 3.2)
		num_display_range = min_max_GT

		plt.imshow(img_pred, aspect='equal', cmap=flag_cmap, clim=num_display_range)
		plt.axis('off')
		plt.title('pred')
		plt.colorbar()
		plt.show()

		# plt.subplot(2,1,2)
		plt.imshow(img_truth, aspect='equal', cmap=flag_cmap, clim=num_display_range)
		plt.axis('off')
		plt.title('GT')
		plt.colorbar()
		plt.show()

		# plt.imshow(5*np.abs(img_truth-img_pred), aspect='equal', cmap='gray', clim=(0, 0.62))
		plt.imshow(num_amplify_diff*np.abs(img_truth-img_pred), aspect='equal', cmap=flag_cmap, clim=num_display_range)
		plt.axis('off')
		plt.title('diff, amplified :%d'%num_amplify_diff)
		plt.colorbar()
		plt.show()





		## saving files
		if not os.path.exists(PATH_FOLDER+ 'result'):
			os.makedirs(PATH_FOLDER+ 'result')
		# ### save result
		# filename_save_nparry = PATH_FOLDER + 'result/img_pred_%s%d'%(flag_type,idx)
		# np.save(filename_save_nparry, img_pred)
		# filename_save_nparry = PATH_FOLDER + 'result/img_truth_%s%d'%(flag_type,idx)
		# np.save(filename_save_nparry, img_truth)
		### save result
		# filename_save_nparry = PATH_FOLDER + 'result/bj_inputs'
		# np.save(filename_save_nparry, inputs)
		filename_save_nparry = PATH_FOLDER + 'result/bj_results'
		np.save(filename_save_nparry, results)
		filename_save_nparry = PATH_FOLDER + 'result/bj_refs'
		np.save(filename_save_nparry, refs)
		# filename_save_nparry = PATH_FOLDER + 'result/bj_freqs_sub'
		# # np.save(filename_save_nparry, freqs_sub)
		# ### save result
		# filename_save_nparry = PATH_FOLDER + 'result/img_pred'
		# np.save(filename_save_nparry, img_pred)
		# filename_save_nparry = PATH_FOLDER + 'result/img_truth'
		# np.save(filename_save_nparry, img_truth)



		try:
			print(' ')
			x = int(input("Enter a number (idx): "))
		except ValueError:
			print('    choh, not a number, end')
			flag_while = False
			break
		idx = x






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='    choh, get array idx for display')
	# parser.add_argument('-i', '--idx', type=int, default=88, help='array idx for display')
	# parser.add_argument('-p', '--patient', type=int, default=220, help='index of patient')
	# parser.add_argument('-c','--cmap', type=str, default='gray', help='colormap for display')
	# parser.add_argument('-a','--amp', type=float, default='5', help='amplification for display of difference')
	parser.add_argument('-i', '--idx', type=int, default=3, help='array idx for display')
	parser.add_argument('-p', '--patient', type=int, default=1, help='index of patient')
	parser.add_argument('-c','--cmap', type=str, default='viridis', help='colormap for display')
	parser.add_argument('-a','--amp', type=float, default='5', help='amplification for display of difference')
	parser.add_argument('-s','--set', type=str, default='v41', help='which set will be loaded')
	parser.add_argument('-w','--weight', type=int, default=0, help='weight of epoch of given number ')


	args = parser.parse_args()
	print('    choh, parser args.patient : %d'%args.patient)
	print('    choh, parser args.idx : %d'%args.idx)
	print('    choh, parser args.weight : %d'%args.weight)
	print('    choh, parser args.set : %s'%args.set)



	main(args)
