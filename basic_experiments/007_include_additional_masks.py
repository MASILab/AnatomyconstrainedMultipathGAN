import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

#Include C7 vertebrae,L1,L2 vertebrae, spinal cord
paths = ["/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/hard_masked", "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/soft_masked", 
         "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/C_D/hard_masked", "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/C_D/soft_masked",
         "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_LUNG/hard", "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_LUNG/soft",
         "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_BONE/hard", "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/STANDARD_BONE/soft"]

masks = ["thyroid_gland.nii.gz", "vertebrae_C7.nii.gz", "vertebrae_L1.nii.gz", "vertebrae_L2.nii.gz", "spinal_cord.nii.gz"]

def process_folder(path, folder):
    for files in os.listdir(os.path.join(path, folder)):
        fname = files.split('.nii.gz')[0]
        segmentations_path = os.path.join(path, folder, 'segmentations')

        multilabel_mask = nib.load(os.path.join(segmentations_path, fname + '_multilabel.nii.gz'))
        multilabel_mask_data = multilabel_mask.get_fdata() #Mask is float. Convert to int
        multilabel_mask_data = multilabel_mask_data.astype(np.uint8)
        new_mask = multilabel_mask_data.copy()
        del multilabel_mask_data
        gc.collect()

        for mask in masks:
            mask_path = os.path.join(segmentations_path, mask)
            if os.path.exists(mask_path):
                mask_data = nib.load(mask_path).get_fdata()
                if mask == masks[0]:
                    new_mask[mask_data == 1] = 41 #thyroid gland
                elif mask == masks[1]:
                    new_mask[mask_data == 1] = 42 #vertebrae C7
                elif mask == masks[2]:
                    new_mask[mask_data == 1] = 43 #vertebrae L1
                elif mask == masks[3]:
                    new_mask[mask_data == 1] = 44 #vertebrae L2
                elif mask == masks[4]:
                    new_mask[mask_data == 1] = 45 #spinal cord
                del mask_data
                gc.collect()
        
        new_mask_nii = nib.Nifti1Image(new_mask, affine = multilabel_mask.affine, header = multilabel_mask.header)
        nib.save(new_mask_nii, os.path.join(segmentations_path, fname + '_multilabel_all.nii.gz'))

for path in tqdm(paths):
    folders = os.listdir(path)
    Parallel(n_jobs=10)(delayed(process_folder)(path, folder) for folder in tqdm(folders))