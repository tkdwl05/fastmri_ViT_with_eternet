import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


# from vit_pytorch.vit import Transformer
# from u_choh_vit import Transformer
from u_choh_model import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]   ### [0] = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss



class MAE_choh(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]   ### [0] = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        #self.decoder_dim = decoder_dim
        self.decoder_dim = encoder.patch_size * encoder.patch_size
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, ref_img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        print('patches.shape {} '.format(patches.shape ))

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        print('num_masked {} '.format(num_masked ))
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        print('rand_indices.shape {} '.format(rand_indices.shape ))
        print('rand_indices {} '.format(rand_indices ))
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        print('batch_range {} '.format( batch_range ))
        print('tokens.shape {} '.format(tokens.shape ))
        tokens = tokens[batch_range, unmasked_indices]
        print('tokens.shape {} '.format(tokens.shape ))

        # get the patches to be masked for the final reconstruction loss

        print('batch_range.shape {} '.format(batch_range.shape ))
        print('masked_indices.shape {} '.format(masked_indices.shape ))
        masked_patches = patches[batch_range, masked_indices]
        print('masked_patches.shape {} '.format(masked_patches.shape ))

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        print('encoded_tokens.shape {} '.format(encoded_tokens.shape ))

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        print('decoder_tokens.shape {} '.format(decoder_tokens.shape ))

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        print('unmasked_decoder_tokens.shape {} '.format(unmasked_decoder_tokens.shape ))

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        print('mask_tokens.shape {} '.format(mask_tokens.shape ))
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        print('mask_tokens.shape {} '.format(mask_tokens.shape ))

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss




class MAE_choh_2(nn.Module):
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

    def forward(self, img, ref_img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        ref_patches = self.to_patch(ref_img)
        # batch, num_patches, *_ = patches.shape
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
        print('indices  {}   '.format(indices))


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

        #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        recon_loss = F.mse_loss(pred_pixel_values, ref_pixel_values)
        return recon_loss
