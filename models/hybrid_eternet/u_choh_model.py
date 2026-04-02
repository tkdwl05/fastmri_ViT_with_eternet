import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange








# ETER hybrid, fastmri multi 16ch, fullsize, 384x384
class ETER_hybrid_GRU_DFU(nn.Module):
	def __init__(self, dim_input_x, dim_input_y, dim_out_x, dim_out_y, n_coil, n_hidden_LRNN_1, n_hidden_LRNN_2, n_unet_depth):
		super(ETER_hybrid_GRU_DFU, self).__init__()
		num_in_x = dim_input_x
		num_in_y = dim_input_y
		# n_coil = dim_coil
		num_out_x = dim_out_x
		num_out_y = dim_out_y
		input_size = num_in_y*n_coil*2
		num_layers = 1 #2
		num_out1 = num_out_y*n_hidden_LRNN_1
		num_in2 = num_in_x*n_hidden_LRNN_1
		num_out2 = num_out_x*n_hidden_LRNN_2
		# num_feat_ch = int(num_out2*2/num_out_x)
		num_feat_ch = int(num_out2*2/num_out_x) + n_coil*2
		n_hidden = n_hidden_LRNN_2 + n_coil
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
		self.unet = UNet_choh_skip(in_channels=num_feat_ch, n_classes=1, depth=n_unet_depth, wf=6, batch_norm=False, up_mode='upconv', n_hidden=n_hidden)


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




class choh_Decoder2(nn.Module):
	def __init__(
		self,
		*,
		encoder,
		decoder_dim,
		decoder_depth = 1,
		decoder_heads = 8,
		decoder_dim_head = 64
	):
		super().__init__()
		# # assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
		# self.masking_ratio = masking_ratio

		# extract some hyperparameters and functions from encoder (vision transformer to be trained)

		self.encoder = encoder
		num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

		self.to_patch = encoder.to_patch_embedding[0]   ### [0] = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
		self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

		# pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
		print(' encoder.to_patch_embedding[2].weight.shape[-1] {}'.format(encoder.to_patch_embedding[2].weight.shape[-1]))
		# print(' encoder.to_patch_embedding[0].weight.shape {}'.format(encoder.to_patch_embedding[0].weight.shape))
		# print(' encoder.to_patch_embedding[1].weight.shape {}'.format(encoder.to_patch_embedding[1].weight.shape))
		print(' encoder.to_patch_embedding[2].weight.shape {}'.format(encoder.to_patch_embedding[2].weight.shape))
		
		# print(encoder.patch_size)
		# pixel_values_per_patch = encoder.patch_size if isinstance(encoder.patch_size, tuple) else encoder.patch_size*encoder.patch_size
		pixel_values_per_patch = encoder.patch_size[0] * encoder.patch_size[1]

		# decoder parameters
		self.decoder_dim = decoder_dim
		self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
		self.mask_token = nn.Parameter(torch.randn(decoder_dim))
		self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
		self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
		self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

		self.patch_to_image = Rearrange('b (nh nw) (p1 p2) -> b 1 (nh p1) (nw p2)', nh=encoder.num_patch_h, nw=encoder.num_patch_w,p1=encoder.patch_size[0], p2=encoder.patch_size[1])

	def forward(self, img):
		device = img.device

		# get patches

		patches = self.to_patch(img)
		batch, num_patches, *_ = patches.shape

		# ref_patches = self.to_patch(ref_img)
		# batch, num_patches, *_ = patches.shape
		print('\nimg.shape {}'.format(img.shape ))
		print('patches.shape {} '.format(patches.shape ))
		# print('\nimg.shape {} ref_img.shape {}'.format(img.shape, ref_img.shape ))
		# print('patches.shape {} ref_patches.shape {}'.format(patches.shape, ref_patches.shape ))

		# patch to encoder tokens and add positions

		tokens = self.patch_to_emb(patches)
		if self.encoder.pool == "cls":
			tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
		elif self.encoder.pool == "mean":
			tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

		# calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

		# num_masked = int(self.masking_ratio * num_patches)
		# rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
		# masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
		#indices = einops.rearragne( torch.arange(batch*num_patches), '(bb pp) -> bb pp', bb=batch,pp=num_patches)
		indices = repeat( torch.arange(num_patches, device = device), 'pp -> bb pp', bb = batch)
		print('indices shape {}   '.format(indices.shape))
		# print('indices  {}   '.format(indices))


		# get the unmasked tokens to be encoded

		batch_range = torch.arange(batch, device = device)[:, None]
		print('batch_range {} '.format(batch_range))
		# tokens = tokens[batch_range, unmasked_indices]
		# print('tokens shape {} {} '.format(tokens.shape, tokens))
		tokens = tokens[batch_range, indices]
		print('tokens shape {}  '.format(tokens.shape))

		# get the patches to be masked for the final reconstruction loss

		# #input_patches = patches[batch_range, :]     ### it was masked_patches, not used, replace ref_pixel_values
		# ref_pixel_values = ref_patches[batch_range, indices] ### choh, 20240807
		# print('ref_pixel_values.shape {} '.format(ref_pixel_values.shape ))

		# attend with vision transformer

		encoded_tokens = self.encoder.transformer(tokens)
		print('encoded_tokens.shape {} '.format(encoded_tokens.shape ))

		# project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

		decoder_tokens = self.enc_to_dec(encoded_tokens)
		print('decoder_tokens.shape {} '.format(decoder_tokens.shape ))

		# reapply decoder position embedding to unmasked tokens

		#unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
		#print(' self.decoder_pos_emb(indices).device {}'.format(self.decoder_pos_emb(indices).device))
		decoder_tokens = decoder_tokens + self.decoder_pos_emb(indices)

		# repeat mask tokens for number of masked, and add the positions using the masked indices derived above

		#mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
		#mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

		# concat the masked tokens to the decoder tokens and attend with decoder
		
		#decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
		#decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
		#decoder_tokens[batch_range, masked_indices] = mask_tokens
		decoded_tokens = self.decoder(decoder_tokens)
		print('decoded_tokens.shape {} '.format(decoded_tokens.shape ))

		# splice out the mask tokens and project to pixel values

		#mask_tokens = decoded_tokens[batch_range, masked_indices]
		#pred_pixel_values = self.to_pixels(mask_tokens)
		pred_pixel_values = self.to_pixels(decoded_tokens)
		print('pred_pixel_values.shape {} '.format(pred_pixel_values.shape ))

		# calculate reconstruction loss

		# #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
		# recon_loss = F.mse_loss(pred_pixel_values, ref_pixel_values)
		# return recon_loss

		# recon_loss = F.mse_loss(pred_pixel_values, ref_pixel_values)
		# out1 = rearrange(pred_pixel_values, 'b (nh nw) (p1 p2) -> b (nh p1) (nw p1)', p1=)
		out = self.patch_to_image(pred_pixel_values)
		# out2 = self.patch_to_image(ref_pixel_values)
		return out




class choh_Decoder(nn.Module):
	def __init__(
		self,
		*,
		encoder,
		decoder_dim,
		decoder_depth = 1,
		decoder_heads = 8,
		decoder_dim_head = 64
	):
		super().__init__()
		# # assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
		# self.masking_ratio = masking_ratio

		# extract some hyperparameters and functions from encoder (vision transformer to be trained)

		self.encoder = encoder
		num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

		self.to_patch = encoder.to_patch_embedding[0]   ### [0] = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
		self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

		# pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
		print(' encoder.to_patch_embedding[2].weight.shape[-1] {}'.format(encoder.to_patch_embedding[2].weight.shape[-1]))
		# print(' encoder.to_patch_embedding[0].weight.shape {}'.format(encoder.to_patch_embedding[0].weight.shape))
		# print(' encoder.to_patch_embedding[1].weight.shape {}'.format(encoder.to_patch_embedding[1].weight.shape))
		print(' encoder.to_patch_embedding[2].weight.shape {}'.format(encoder.to_patch_embedding[2].weight.shape))
		
		# print(encoder.patch_size)
		# pixel_values_per_patch = encoder.patch_size if isinstance(encoder.patch_size, tuple) else encoder.patch_size*encoder.patch_size
		pixel_values_per_patch = encoder.patch_size[0] * encoder.patch_size[1]

		# decoder parameters
		self.decoder_dim = decoder_dim
		self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
		self.mask_token = nn.Parameter(torch.randn(decoder_dim))
		self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
		self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
		self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

		self.patch_to_image = Rearrange('b (nh nw) (p1 p2) -> b 1 (nh p1) (nw p2)', nh=encoder.num_patch_h, nw=encoder.num_patch_w,p1=encoder.patch_size[0], p2=encoder.patch_size[1])

	def forward(self, img, ref_img):
		device = img.device

		# get patches

		patches = self.to_patch(img)
		batch, num_patches, *_ = patches.shape

		ref_patches = self.to_patch(ref_img)
		# batch, num_patches, *_ = patches.shape
		print('\nimg.shape {} ref_img.shape {}'.format(img.shape, ref_img.shape ))
		print('patches.shape {} ref_patches.shape {}'.format(patches.shape, ref_patches.shape ))

		# patch to encoder tokens and add positions

		tokens = self.patch_to_emb(patches)
		if self.encoder.pool == "cls":
			tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
		elif self.encoder.pool == "mean":
			tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

		# calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

		# num_masked = int(self.masking_ratio * num_patches)
		# rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
		# masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
		#indices = einops.rearragne( torch.arange(batch*num_patches), '(bb pp) -> bb pp', bb=batch,pp=num_patches)
		indices = repeat( torch.arange(num_patches, device = device), 'pp -> bb pp', bb = batch)
		print('indices shape {}   '.format(indices.shape))
		# print('indices  {}   '.format(indices))


		# get the unmasked tokens to be encoded

		batch_range = torch.arange(batch, device = device)[:, None]
		print('batch_range {} '.format(batch_range))
		# tokens = tokens[batch_range, unmasked_indices]
		# print('tokens shape {} {} '.format(tokens.shape, tokens))
		tokens = tokens[batch_range, indices]
		print('tokens shape {}  '.format(tokens.shape))

		# get the patches to be masked for the final reconstruction loss

		#input_patches = patches[batch_range, :]     ### it was masked_patches, not used, replace ref_pixel_values
		ref_pixel_values = ref_patches[batch_range, indices] ### choh, 20240807
		print('ref_pixel_values.shape {} '.format(ref_pixel_values.shape ))

		# attend with vision transformer

		encoded_tokens = self.encoder.transformer(tokens)
		print('encoded_tokens.shape {} '.format(encoded_tokens.shape ))

		# project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

		decoder_tokens = self.enc_to_dec(encoded_tokens)
		print('decoder_tokens.shape {} '.format(decoder_tokens.shape ))

		# reapply decoder position embedding to unmasked tokens

		#unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
		#print(' self.decoder_pos_emb(indices).device {}'.format(self.decoder_pos_emb(indices).device))
		decoder_tokens = decoder_tokens + self.decoder_pos_emb(indices)

		# repeat mask tokens for number of masked, and add the positions using the masked indices derived above

		#mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
		#mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

		# concat the masked tokens to the decoder tokens and attend with decoder
		
		#decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
		#decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
		#decoder_tokens[batch_range, masked_indices] = mask_tokens
		decoded_tokens = self.decoder(decoder_tokens)
		print('decoded_tokens.shape {} '.format(decoded_tokens.shape ))

		# splice out the mask tokens and project to pixel values

		#mask_tokens = decoded_tokens[batch_range, masked_indices]
		#pred_pixel_values = self.to_pixels(mask_tokens)
		pred_pixel_values = self.to_pixels(decoded_tokens)
		print('pred_pixel_values.shape {} '.format(pred_pixel_values.shape ))

		# calculate reconstruction loss

		# #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
		# recon_loss = F.mse_loss(pred_pixel_values, ref_pixel_values)
		# return recon_loss

		recon_loss = F.mse_loss(pred_pixel_values, ref_pixel_values)
		# out1 = rearrange(pred_pixel_values, 'b (nh nw) (p1 p2) -> b (nh p1) (nw p1)', p1=)
		out1 = self.patch_to_image(pred_pixel_values)
		out2 = self.patch_to_image(ref_pixel_values)
		return out1, out2






class choh_ViT_for_image_reconstruction(nn.Module):
	def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
		super().__init__()
		print('   \'choh_ViT_for_image_reconstruction\'   ')
		image_height, image_width = image_size
		patch_height, patch_width = patch_size

		assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

		num_patches = (image_height // patch_height) * (image_width // patch_width)
		patch_dim = channels * patch_height * patch_width
		assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_patch_h = image_height // patch_height
		self.num_patch_w = image_width // patch_width
		self.to_patch_embedding = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
			nn.LayerNorm(patch_dim),
			nn.Linear(patch_dim, dim),
			nn.LayerNorm(dim),
		)

		self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
		self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
		self.dropout = nn.Dropout(emb_dropout)

		self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

		self.pool = pool
		self.to_latent = nn.Identity()

		self.mlp_head = nn.Linear(dim, num_classes)
		self.final_rearrange = Rearrange('b (nx ny)  -> b 1 nx ny', nx = image_size[0], ny = image_size[1] )

	def forward(self, img):
		print('\n  img.shape {}'.format( img.shape ))
		x = self.to_patch_embedding(img)
		print(' x.shape {}'.format( x.shape ))
		b, n, _ = x.shape

		cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
		print(' cls_tokens.shape {}'.format( cls_tokens.shape ))
		x = torch.cat((cls_tokens, x), dim=1)
		print(' x.shape {}'.format( x.shape ))
		x += self.pos_embedding[:, :(n + 1)]
		print(' x.shape {}'.format( x.shape ))
		x = self.dropout(x)
		print(' x.shape {}'.format( x.shape ))

		x = self.transformer(x)
		print(' after transformer x.shape {}'.format( x.shape ))

		x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
		print(' x.shape {}'.format( x.shape ))

		x = self.to_latent(x)
		print(' x.shape {}'.format( x.shape ))
		out = self.mlp_head(x)
		print(' out.shape {}'.format( out.shape ))

		out = self.final_rearrange(out)
		print(' final out.shape {}'.format( out.shape ))

		return out



class ViT_choh(nn.Module):
	def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
		super().__init__()
		print('   \'ViT_choh\'   ')
		image_height, image_width = image_size
		patch_height, patch_width = patch_size

		assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

		num_patches = (image_height // patch_height) * (image_width // patch_width)
		patch_dim = channels * patch_height * patch_width
		assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

		self.image_size = image_size
		self.patch_size = patch_size
		self.num_patch_h = image_height // patch_height
		self.num_patch_w = image_width // patch_width
		self.to_patch_embedding = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
			nn.LayerNorm(patch_dim),
			nn.Linear(patch_dim, dim),
			nn.LayerNorm(dim),
		)

		self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
		self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
		self.dropout = nn.Dropout(emb_dropout)

		self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

		self.pool = pool
		self.to_latent = nn.Identity()

		self.mlp_head = nn.Linear(dim, num_classes)

	def forward(self, img):
		x = self.to_patch_embedding(img)
		b, n, _ = x.shape

		cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
		x = torch.cat((cls_tokens, x), dim=1)
		x += self.pos_embedding[:, :(n + 1)]
		x = self.dropout(x)

		x = self.transformer(x)

		x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

		x = self.to_latent(x)
		return self.mlp_head(x)










class FeedForward(nn.Module):
	def __init__(self, dim, hidden_dim, dropout = 0.):
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)

class Attention(nn.Module):
	def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
		super().__init__()
		inner_dim = dim_head *  heads
		project_out = not (heads == 1 and dim_head == dim)

		self.heads = heads
		self.scale = dim_head ** -0.5

		self.norm = nn.LayerNorm(dim)

		self.attend = nn.Softmax(dim = -1)
		self.dropout = nn.Dropout(dropout)

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		) if project_out else nn.Identity()

	def forward(self, x):
		x = self.norm(x)

		qkv = self.to_qkv(x).chunk(3, dim = -1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

		dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

		attn = self.attend(dots)
		attn = self.dropout(attn)

		out = torch.matmul(attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out)

class Transformer(nn.Module):
	def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.layers = nn.ModuleList([])
		for _ in range(depth):
			self.layers.append(nn.ModuleList([
				Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
				FeedForward(dim, mlp_dim, dropout = dropout)
			]))

	def forward(self, x):
		for attn, ff in self.layers:
			x = attn(x) + x
			x = ff(x) + x

		return self.norm(x)
