
import os

# Read the list of used files
with open('list_brain_train_320.txt', 'r') as f:
    used_files_npy = [line.strip() for line in f.readlines() if line.strip()]

# Convert to the format found in the directory
used_files_mat = [f.replace('.npy', '_ksp.mat') for f in used_files_npy]
used_files_set = set(used_files_mat)

# List all files in the directory
directory_path = '/mnt/sda/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'
all_files = sorted([f for f in os.listdir(directory_path) if f.endswith('_ksp.mat')])

# Find files not in the used list
unseen_files = [f for f in all_files if f not in used_files_set]

print(f"Total files in directory: {len(all_files)}")
print(f"Total used files: {len(used_files_set)}")
print(f"Total unseen files: {len(unseen_files)}")

# Select all unseen files
selected_files = unseen_files
print(f"Selected {len(selected_files)} unseen files (ALL).")
# for f in selected_files:
#     print(f)

# Save the selected files to a new text file for the inference script to use
with open('list_brain_unseen_all.txt', 'w') as f:
    for filename in selected_files:
        # Convert back to .npy format if the dataloader expects it, 
        # or just keep it as is and modify the dataloader.
        # The existing dataloader expects .npy in the text file and replaces it.
        # So I should save them as .npy
        npy_filename = filename.replace('_ksp.mat', '.npy')
        f.write(npy_filename + '\n')
