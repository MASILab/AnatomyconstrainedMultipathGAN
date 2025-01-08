import nibabel as nib 
import numpy as np 
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed
import os

#Savinf all training data as normalized nifti images ranging from -1 to 1.

nifti_files_path = "/home-local/Kernel_Conversion/MultipathKernelConversion_forA6000/multipath_data_journalextension"
normalized_nifti_files = "/home-local/Kernel_Conversion/AnatomyConstrainedMultipathGAN/AnatomyconstrainedMultipathGAN/training_data"

folders = ["train_siemens_masked_hard", "train_siemens_masked_soft", "train_ge_bone_hard", "train_ge_bone_soft",
           "train_philips_hard", "train_philips_soft", "train_lung_hard", "train_lung_soft"]

normalizer = interp1d([-1024, 3072], [-1, 1])

for folder in tqdm(folders):
    input_data_path = os.path.join(nifti_files_path, folder)
    output_data_path = os.path.join(normalized_nifti_files, folder)
    os.makedirs(output_data_path, exist_ok=True)

    for file in tqdm(os.listdir(input_data_path)):
        input_file = nib.load(os.path.join(input_data_path, file))
        input_data = input_file.get_fdata()
        input_affine = input_file.affine
        input_header = input_file.header

        # Normalize the data
        clipped = np.clip(input_data, -1024, 3072)
        normalized = normalizer(clipped)
        normalized_nifti = nib.Nifti1Image(normalized, input_affine, input_header)

        # Save the normalized data
        nib.save(normalized_nifti, os.path.join(output_data_path, file))


