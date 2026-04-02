import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Helper from the original script
def choh_ifft2c_multi_ch(ksp):
    num_ch_for_C = ksp.shape[1]//2
    C = np.zeros((ksp.shape[0], num_ch_for_C, ksp.shape[2], ksp.shape[3]),dtype=np.complex128)
    C = C + ksp[:,0:ksp.shape[1]:2,:,:] + ksp[:,1:ksp.shape[1]:2,:,:]*1j 

    signal = np.fft.fftshift( np.fft.fftshift(C, 2), 3)
    result_shift = np.fft.ifft2(signal, axes=(-2, -1))
    result_complex = np.fft.ifftshift( np.fft.ifftshift(result_shift, 2), 3)

    result = np.zeros(ksp.shape)
    result[:,0:ksp.shape[1]:2,:,:] = result_complex.real
    result[:,1:ksp.shape[1]:2,:,:] = result_complex.imag

    return result

class FastMRI_H5_Dataloader(Dataset):
    def __init__(self, data_folder, num_files=10):
        print('\n  @ dataloader_h5.py ')
        print('  Dataset : Native .h5 FastMRI Loader')
        
        self.data_folder = data_folder
        self.files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.h5')][:num_files]
        
        self.N_OUTPUT = 384
        self.N_COIL_CH = 16
        self.val_amp_X_img = 1e6
        self.val_amp_X_ksp = 1e4
        self.val_amp_Y = 1e6
        
        print(f"    Loaded {len(self.files)} raw .h5 files.")

    def __len__(self):
        return len(self.files)

    def crop_center(self, kspace, target_H, target_W):
        # kspace shape: (slices, coils, H, W)
        _, _, H, W = kspace.shape
        start_H = max(0, (H - target_H) // 2)
        start_W = max(0, (W - target_W) // 2)
        k_crop = kspace[:, :, start_H:start_H+target_H, start_W:start_W+target_W]
        
        # pad if smaller
        pad_H = max(0, target_H - H)
        pad_W = max(0, target_W - W)
        if pad_H > 0 or pad_W > 0:
            k_crop = np.pad(k_crop, ((0,0), (0,0), (pad_H//2, pad_H - pad_H//2), (pad_W//2, pad_W - pad_W//2)))
        return k_crop

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            kspace_complex = f['kspace'][()] # shape: (slices, coils, H, W)
            
        num_slices = kspace_complex.shape[0]
        coils = kspace_complex.shape[1]
        
        kspace_cropped = self.crop_center(kspace_complex, self.N_OUTPUT, self.N_OUTPUT)
        
        # Ensure exactly 16 coils
        kspace_16 = np.zeros((num_slices, self.N_COIL_CH, self.N_OUTPUT, self.N_OUTPUT), dtype=np.complex128)
        take_coils = min(coils, self.N_COIL_CH)
        kspace_16[:, :take_coils, :, :] = kspace_cropped[:, :take_coils, :, :]
        
        # Convert to separate Real/Imag shape: (n_in_ch, 384, 384) where n_in_ch = 32
        ksp_combined = np.zeros((num_slices, self.N_COIL_CH * 2, self.N_OUTPUT, self.N_OUTPUT), dtype=np.float32)
        for c in range(self.N_COIL_CH):
            ksp_combined[:, 2*c, :, :] = kspace_16[:, c, :, :].real
            ksp_combined[:, 2*c+1, :, :] = kspace_16[:, c, :, :].imag

        # Generate target Image (SoS)
        img_16ch = choh_ifft2c_multi_ch(ksp_combined)
        Y_volume = np.sqrt(np.sum(np.square(img_16ch), axis=1, keepdims=True))

        # Generate Aliased kspace (R=4)
        x_ksp_zf = np.zeros_like(ksp_combined)
        x_ksp_zf[:, :, 3:384:4, :] = ksp_combined[:, :, 3:384:4, :]
        x_ksp_zf[:, :, 192-16:192+16, :] = ksp_combined[:, :, 192-16:192+16, :]

        img_alias_16ch = choh_ifft2c_multi_ch(x_ksp_zf)

        data = (x_ksp_zf * self.val_amp_X_ksp)[0]          # Just returning the 1st slice for iteration testing in Dataset
        data_img = (img_alias_16ch * self.val_amp_X_img)[0]
        label = (Y_volume * self.val_amp_Y)[0]

        return {'data': data, 'data_img': data_img, 'label': label}

if __name__ == '__main__':
    dataset = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_train', num_files=2)
    loader = DataLoader(dataset, batch_size=1)
    for batch in loader:
        print("Data loaded! Shapes:")
        print("data (aliased ksp):", batch['data'].shape)
        print("data_img (aliased img):", batch['data_img'].shape)
        print("label (target img):", batch['label'].shape)
        break
