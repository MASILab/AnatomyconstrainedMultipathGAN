import os 
import pandas as pd 
import shutil 
from tqdm import tqdm
from scipy.interpolate import interp1d
import nibabel as nib
import numpy as np

#Read the dataframes, read the hard and soft uids. Replace the uids with the pids
b30f_b50f = pd.read_csv("/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B50f_B30f_data.csv")
c_d = pd.read_csv("/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D_data.csv")
standard_bone = pd.read_csv("/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE_data.csv")
standard_lung = pd.read_csv("/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG_data.csv")

nlst_t0 = "/nfs2/NLST/NIfTI/T0_all"

def copy_data(dataframe, hard_kernel_folder, soft_kernel_folder):
    for index, row in tqdm(dataframe.iterrows()):
        pid = int(row['pid'])
        hard_uid = str(row['hard_uid']) + ".nii.gz"
        soft_uid = str(row['soft_uid']) + ".nii.gz" 

        if hard_uid and soft_uid in os.listdir(nlst_t0):
            hard_kernel_image = os.path.join(nlst_t0, hard_uid)
            soft_kernel_image = os.path.join(nlst_t0, soft_uid)

            #Copy the files to the respective folders
            hard_dir = hard_kernel_folder
            soft_dir = soft_kernel_folder

            #Rename the files to the pid and then copy them to the respective folders
            hard_new_name = os.path.join(hard_dir, str(pid) + ".nii.gz")
            soft_new_name = os.path.join(soft_dir, str(pid) + ".nii.gz")

            shutil.copy(hard_kernel_image, hard_new_name)
            shutil.copy(soft_kernel_image, soft_new_name)

# copy_data(b30f_b50f, "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/hard", "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft")
# copy_data(c_d, "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard", "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft")
# copy_data(standard_bone, "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/hard", "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/soft")
# copy_data(standard_lung, "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/hard", "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/soft")

def apply_circular_mask(input_path, out_path):
    """
    Apply a circular mask to the Siemens and Philips 3D volumes.
    """
    for file in tqdm(os.listdir(input_path)):
        print(f"Currently masking {file}")
        fname = file.split(".nii.gz")[0]
        nift_image = nib.load(os.path.join(input_path, file))
        data = nift_image.get_fdata()
        dtype = data.dtype
        masked_image = np.zeros(data.shape, dtype = dtype)

        for i in range(data.shape[2]):
            width, height = data.shape[0], data.shape[1]
            centx = width // 2
            centy = height // 2
            radius = min(centx, centy, width-centx, height-centy)

            Y,X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - centx)**2 + (Y - centy)**2)
            mask = dist_from_center <= radius
            masked_slice = data[:,:,i].copy()
            masked_slice[~mask] = -1024

            clipped = np.clip(masked_slice, -1024, 3072)
            masked_image[:,:,i] = clipped
        
        masked_nifti = nib.Nifti1Image(masked_image, nift_image.affine, nift_image.header)
        nib.save(masked_nifti, os.path.join(out_path, fname + ".nii.gz"))
        print(f"Saved masked image to {out_path}")


directories = [
    "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/hard",
    "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft",
    "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard",
    "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft",
]

for directory in tqdm(directories):
    apply_circular_mask(input_path = directory, out_path = directory + "_masked")