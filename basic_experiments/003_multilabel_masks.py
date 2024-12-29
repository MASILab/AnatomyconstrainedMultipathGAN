import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed

paths = ["/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/hard_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/hard",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/soft",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/hard",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/soft"]

masks = ["lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz", "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
         "skeletal_muscle.nii.gz", "subcutaneous_fat.nii.gz", "liver.nii.gz", "heart.nii.gz", "aorta.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz"]

def process_folder(folder):
    for files in os.listdir(os.path.join(paths[0], folder)):
        fname = files.split('.nii.gz')[0]
        segmentations_path = os.path.join(paths[0], folder, 'segmentations')
        llll = os.path.join(segmentations_path, masks[0])
        llrl = os.path.join(segmentations_path, masks[1])
        lul = os.path.join(segmentations_path, masks[2])
        lur = os.path.join(segmentations_path, masks[3])
        lmr = os.path.join(segmentations_path, masks[4])
        sm = os.path.join(segmentations_path, masks[5])
        sf = os.path.join(segmentations_path, masks[6])
        liver = os.path.join(segmentations_path, masks[7])
        heart = os.path.join(segmentations_path, masks[8])
        aorta = os.path.join(segmentations_path, masks[9])
        kl = os.path.join(segmentations_path, masks[10])
        kr = os.path.join(segmentations_path, masks[11])

        llll_data = nib.load(llll).get_fdata()
        llrl_data = nib.load(llrl).get_fdata()
        lul_data = nib.load(lul).get_fdata()
        lur_data = nib.load(lur).get_fdata()
        lmr_data = nib.load(lmr).get_fdata()
        sm_data = nib.load(sm).get_fdata()
        sf_data = nib.load(sf).get_fdata()
        liver_data = nib.load(liver).get_fdata()
        heart_data = nib.load(heart).get_fdata()
        aorta_data = nib.load(aorta).get_fdata()
        kl_data = nib.load(kl).get_fdata()
        kr_data = nib.load(kr).get_fdata()

        multilabel_mask = np.zeros((llll_data.shape[0], llll_data.shape[1], llll_data.shape[2]), dtype=np.uint8)
        multilabel_mask[llll_data == 1] = 1
        multilabel_mask[llrl_data == 1] = 1
        multilabel_mask[lul_data == 1] = 1
        multilabel_mask[lur_data == 1] = 1
        multilabel_mask[lmr_data == 1] = 1
        multilabel_mask[sm_data == 1] = 2
        multilabel_mask[sf_data == 1] = 3
        multilabel_mask[liver_data == 1] = 4
        multilabel_mask[heart_data == 1] = 5
        multilabel_mask[aorta_data == 1] = 6
        multilabel_mask[kl_data == 1] = 7
        multilabel_mask[kr_data == 1] = 7

        nifti_mask = nib.Nifti1Image(multilabel_mask, affine=nib.load(llll).affine, header=nib.load(llll).header)
        nib.save(nifti_mask, os.path.join(segmentations_path, fname + '_multilabel.nii.gz'))

# Use joblib to parallelize the processing
folders = os.listdir(paths[0])
Parallel(n_jobs=10)(delayed(process_folder)(folder) for folder in tqdm(folders))