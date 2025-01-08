import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

# paths = ["/fs5/p_masi/krishar1/MIDL/B30f_B50f/hard_masked",
#          "/fs5/p_masi/krishar1/MIDL/B30f_B50f/soft_masked",
#          "/fs5/p_masi/krishar1/MIDL/C_D/hard_masked",
#          "/fs5/p_masi/krishar1/MIDL/C_D/soft_masked",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_LUNG/hard",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_LUNG/soft",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_BONE/hard",
#          "/fs5/p_masi/krishar1/MIDL/STANDARD_BONE/soft"]


paths = ["/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/hard_masked", "/media/krishar1/Elements1/AnatomyConstrainedMultipathGAN/B30f_B50f/soft_masked"]

masks = ["lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz", "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
         "skeletal_muscle.nii.gz", "subcutaneous_fat.nii.gz", "trachea.nii.gz", "liver.nii.gz", "heart.nii.gz", "aorta.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz",
         "pancreas.nii.gz", "common_carotid_artery_left.nii.gz", "common_carotid_artery_right.nii.gz", "inferior_vena_cava.nii.gz", "pulmonary_vein.nii.gz", 
         "subclavian_artery_left.nii.gz", "subclavian_artery_right.nii.gz", "superior_vena_cava.nii.gz", "rib_left_1.nii.gz", "rib_left_2.nii.gz", "rib_left_3.nii.gz",
         "rib_left_4.nii.gz", "rib_left_5.nii.gz", "rib_left_6.nii.gz", "rib_left_7.nii.gz", "rib_left_8.nii.gz", "rib_left_9.nii.gz", "rib_left_10.nii.gz",
         "rib_left_11.nii.gz", "rib_left_12.nii.gz", "rib_right_1.nii.gz", "rib_right_2.nii.gz", "rib_right_3.nii.gz", "rib_right_4.nii.gz", "rib_right_5.nii.gz",
         "rib_right_6.nii.gz", "rib_right_7.nii.gz", "rib_right_8.nii.gz", "rib_right_9.nii.gz", "rib_right_10.nii.gz", "rib_right_11.nii.gz", "rib_right_12.nii.gz", "spleen.nii.gz", 
         "costal_cartilages.nii.gz", "adrenal_gland_left.nii.gz", "adrenal_gland_right.nii.gz", "colon.nii.gz", "clavicula_left.nii.gz", "clavicula_right.nii.gz", "duodenum.nii.gz", 
         "gallbladder.nii.gz", "portal_vein_and_splenic_vein.nii.gz", "scapula_left.nii.gz", "scapula_right.nii.gz", "sternum.nii.gz"]

def process_folder(path, folder):
    for files in os.listdir(os.path.join(path, folder)):
        fname = files.split('.nii.gz')[0]
        segmentations_path = os.path.join(path, folder, 'segmentations')
        
        # Initialize an empty mask
        llll = os.path.join(segmentations_path, masks[0])
        llll_data = nib.load(llll).get_fdata()
        multilabel_mask = np.zeros(shape=llll_data.shape, dtype=np.uint8)
        del llll_data
        gc.collect()

        # Define groups of masks
        lung_masks = [masks[0], masks[1], masks[2], masks[3], masks[4]]
        kidney_masks = [masks[11], masks[12]]
        carotid_artery_masks = [masks[14], masks[15]]
        subclavian_artery_masks = [masks[18], masks[19]]
        left_rib_masks = masks[21:33]
        right_rib_masks = masks[33:45]
        adrenal_gland_masks = [masks[47], masks[48]]
        clavicle_masks = [masks[50], masks[51]]
        scapula_masks = [masks[55], masks[56]]

        # Load and process each mask one by one
        for mask in masks:
            mask_path = os.path.join(segmentations_path, mask)
            if os.path.exists(mask_path):
                mask_data = nib.load(mask_path).get_fdata()
                if mask in lung_masks:
                    multilabel_mask[mask_data == 1] = 1
                elif mask == masks[5]:  # skeletal_muscle
                    multilabel_mask[mask_data == 1] = 2
                elif mask == masks[6]:  # subcutaneous_fat
                    multilabel_mask[mask_data == 1] = 3
                elif mask == masks[7]:  # trachea
                    multilabel_mask[mask_data == 1] = 4
                elif mask == masks[8]:  # liver
                    multilabel_mask[mask_data == 1] = 5
                elif mask in kidney_masks:
                    multilabel_mask[mask_data == 1] = 6
                elif mask == masks[9]:  # heart
                    multilabel_mask[mask_data == 1] = 7
                elif mask == masks[10]:  # aorta
                    multilabel_mask[mask_data == 1] = 8
                elif mask == masks[13]:  # pancreas
                    multilabel_mask[mask_data == 1] = 9
                elif mask in carotid_artery_masks:
                    multilabel_mask[mask_data == 1] = 10
                elif mask == masks[16]:  # inferior_vena_cava
                    multilabel_mask[mask_data == 1] = 11
                elif mask == masks[17]:  # pulmonary_vein
                    multilabel_mask[mask_data == 1] = 12
                elif mask in subclavian_artery_masks:
                    multilabel_mask[mask_data == 1] = 13
                elif mask == masks[20]:  # superior_vena_cava
                    multilabel_mask[mask_data == 1] = 14
                elif mask in left_rib_masks:
                    multilabel_mask[mask_data == 1] = 15
                elif mask in right_rib_masks:
                    multilabel_mask[mask_data == 1] = 16
                elif mask == masks[45]:  # spleen
                    multilabel_mask[mask_data == 1] = 17
                elif mask == masks[46]:  # costal_cartilages
                    multilabel_mask[mask_data == 1] = 18
                elif mask in adrenal_gland_masks:
                    multilabel_mask[mask_data == 1] = 19
                elif mask == masks[49]:  # colon
                    multilabel_mask[mask_data == 1] = 20
                elif mask in clavicle_masks:
                    multilabel_mask[mask_data == 1] = 21
                elif mask == masks[52]:  # duodenum
                    multilabel_mask[mask_data == 1] = 22
                elif mask == masks[53]:  # gallbladder
                    multilabel_mask[mask_data == 1] = 23
                elif mask == masks[54]:  # portal_vein_and_splenic_vein
                    multilabel_mask[mask_data == 1] = 24
                elif mask in scapula_masks:
                    multilabel_mask[mask_data == 1] = 25
                elif mask == masks[57]: #Sternum
                    multilabel_mask[mask_data == 1] = 26
                del mask_data
                gc.collect()

        # Save the multilabel mask
        nifti_mask = nib.Nifti1Image(multilabel_mask, affine=nib.load(llll).affine, header=nib.load(llll).header)
        nib.save(nifti_mask, os.path.join(segmentations_path, fname + '_multilabel.nii.gz'))

# Use joblib to parallelize the processing
for path in paths:
    folders = os.listdir(path)
    Parallel(n_jobs=10)(delayed(process_folder)(path, folder) for folder in tqdm(folders))