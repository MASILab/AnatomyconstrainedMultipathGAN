# import numpy as np
# import os
# import nibabel as nib
# from tqdm import tqdm
# from joblib import Parallel, delayed

# # paths = ["/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/hard_masked",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft_masked",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard_masked",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft_masked",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/hard",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/soft",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/hard",
# #          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/soft"]

# paths = ["/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft_masked",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard_masked",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft_masked",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/hard",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/soft",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/hard",
#          "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/soft"]

# masks = ["lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz", "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
#          "skeletal_muscle.nii.gz", "subcutaneous_fat.nii.gz", "trachea.nii.gz", "liver.nii.gz", "heart.nii.gz", "aorta.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz",
#          "pancreas.nii.gz", "common_carotid_artery_left.nii.gz", "common_carotid_artery_right.nii.gz", "inferior_vena_cava.nii.gz", "pulmonary_vein.nii.gz", 
#          "subclavian_artery_left.nii.gz", "subclavian_artery_right.nii.gz", "superior_vena_cava.nii.gz", "rib_left_1.nii.gz", "rib_left_2.nii.gz", "rib_left_3.nii.gz",
#          "rib_left_4.nii.gz", "rib_left_5.nii.gz", "rib_left_6.nii.gz", "rib_left_7.nii.gz", "rib_left_8.nii.gz", "rib_left_9.nii.gz", "rib_left_10.nii.gz",
#          "rib_left_11.nii.gz", "rib_left_12.nii.gz", "rib_right_1.nii.gz", "rib_right_2.nii.gz", "rib_right_3.nii.gz", "rib_right_4.nii.gz", "rib_right_5.nii.gz",
#          "rib_right_6.nii.gz", "rib_right_7.nii.gz", "rib_right_8.nii.gz", "rib_right_9.nii.gz", "rib_right_10.nii.gz", "rib_right_11.nii.gz", "rib_right_12.nii.gz", "spleen.nii.gz"]

# def process_folder(folder, path):
#     for files in os.listdir(os.path.join(path, folder)):
#         fname = files.split('.nii.gz')[0]
#         segmentations_path = os.path.join(path, folder, 'segmentations')
#         llll = os.path.join(segmentations_path, masks[0])
#         llrl = os.path.join(segmentations_path, masks[1])
#         lul = os.path.join(segmentations_path, masks[2])
#         lur = os.path.join(segmentations_path, masks[3])
#         lmr = os.path.join(segmentations_path, masks[4])
#         sm = os.path.join(segmentations_path, masks[5])
#         sf = os.path.join(segmentations_path, masks[6])
#         trachea = os.path.join(segmentations_path, masks[7])
#         liver = os.path.join(segmentations_path, masks[8])
#         heart = os.path.join(segmentations_path, masks[9])
#         aorta = os.path.join(segmentations_path, masks[10])
#         kl = os.path.join(segmentations_path, masks[11])
#         kr = os.path.join(segmentations_path, masks[12])
#         pancreas = os.path.join(segmentations_path, masks[13])
#         ccal = os.path.join(segmentations_path, masks[14])
#         ccar = os.path.join(segmentations_path, masks[15])
#         iva = os.path.join(segmentations_path, masks[16])
#         pvein = os.path.join(segmentations_path, masks[17])
#         scal = os.path.join(segmentations_path, masks[18])
#         scar = os.path.join(segmentations_path, masks[19])
#         sva = os.path.join(segmentations_path, masks[20])
#         ribl1 = os.path.join(segmentations_path, masks[21])
#         ribl2 = os.path.join(segmentations_path, masks[22])
#         ribl3 = os.path.join(segmentations_path, masks[23])
#         ribl4 = os.path.join(segmentations_path, masks[24])
#         ribl5 = os.path.join(segmentations_path, masks[25])
#         ribl6 = os.path.join(segmentations_path, masks[26])
#         ribl7 = os.path.join(segmentations_path, masks[27])
#         ribl8 = os.path.join(segmentations_path, masks[28])
#         ribl9 = os.path.join(segmentations_path, masks[29])
#         ribl10 = os.path.join(segmentations_path, masks[30])
#         ribl11 = os.path.join(segmentations_path, masks[31])
#         ribl12 = os.path.join(segmentations_path, masks[32])
#         ribr1 = os.path.join(segmentations_path, masks[33])
#         ribr2 = os.path.join(segmentations_path, masks[34])
#         ribr3 = os.path.join(segmentations_path, masks[35])
#         ribr4 = os.path.join(segmentations_path, masks[36])
#         ribr5 = os.path.join(segmentations_path, masks[37])
#         ribr6 = os.path.join(segmentations_path, masks[38])
#         ribr7 = os.path.join(segmentations_path, masks[39])
#         ribr8 = os.path.join(segmentations_path, masks[40])
#         ribr9 = os.path.join(segmentations_path, masks[41])
#         ribr10 = os.path.join(segmentations_path, masks[42])
#         ribr11 = os.path.join(segmentations_path, masks[43])
#         ribr12 = os.path.join(segmentations_path, masks[44])
#         spleen = os.path.join(segmentations_path, masks[45])


#         llll_data = nib.load(llll).get_fdata()
#         llrl_data = nib.load(llrl).get_fdata()
#         lul_data = nib.load(lul).get_fdata()
#         lur_data = nib.load(lur).get_fdata()
#         lmr_data = nib.load(lmr).get_fdata()
#         sm_data = nib.load(sm).get_fdata()
#         sf_data = nib.load(sf).get_fdata()
#         trachea_data = nib.load(trachea).get_fdata()
#         liver_data = nib.load(liver).get_fdata()
#         heart_data = nib.load(heart).get_fdata()
#         aorta_data = nib.load(aorta).get_fdata()
#         kl_data = nib.load(kl).get_fdata()
#         kr_data = nib.load(kr).get_fdata()
#         pancreas_data = nib.load(pancreas).get_fdata()
#         ccal_data = nib.load(ccal).get_fdata()
#         ccar_data = nib.load(ccar).get_fdata()
#         iva_data = nib.load(iva).get_fdata()
#         pvein_data = nib.load(pvein).get_fdata()
#         scal_data = nib.load(scal).get_fdata()
#         scar_data = nib.load(scar).get_fdata()
#         sva_data = nib.load(sva).get_fdata()
#         ribl1_data = nib.load(ribl1).get_fdata()
#         ribl2_data = nib.load(ribl2).get_fdata()
#         ribl3_data = nib.load(ribl3).get_fdata()
#         ribl4_data = nib.load(ribl4).get_fdata()
#         ribl5_data = nib.load(ribl5).get_fdata()
#         ribl6_data = nib.load(ribl6).get_fdata()
#         ribl7_data = nib.load(ribl7).get_fdata()
#         ribl8_data = nib.load(ribl8).get_fdata()
#         ribl9_data = nib.load(ribl9).get_fdata()
#         ribl10_data = nib.load(ribl10).get_fdata()
#         ribl11_data = nib.load(ribl11).get_fdata()
#         ribl12_data = nib.load(ribl12).get_fdata()
#         ribr1_data = nib.load(ribr1).get_fdata()
#         ribr2_data = nib.load(ribr2).get_fdata()
#         ribr3_data = nib.load(ribr3).get_fdata()
#         ribr4_data = nib.load(ribr4).get_fdata()
#         ribr5_data = nib.load(ribr5).get_fdata()
#         ribr6_data = nib.load(ribr6).get_fdata()
#         ribr7_data = nib.load(ribr7).get_fdata()
#         ribr8_data = nib.load(ribr8).get_fdata()
#         ribr9_data = nib.load(ribr9).get_fdata()
#         ribr10_data = nib.load(ribr10).get_fdata()
#         ribr11_data = nib.load(ribr11).get_fdata()
#         ribr12_data = nib.load(ribr12).get_fdata()
#         spleen_data = nib.load(spleen).get_fdata()

#         multilabel_mask = np.zeros(shape = llll_data.shape, dtype=np.uint8)
#         multilabel_mask[llll_data == 1] = 1
#         multilabel_mask[llrl_data == 1] = 1
#         multilabel_mask[lul_data == 1] = 1
#         multilabel_mask[lur_data == 1] = 1
#         multilabel_mask[lmr_data == 1] = 1
#         multilabel_mask[sm_data == 1] = 2
#         multilabel_mask[sf_data == 1] = 3
#         multilabel_mask[trachea_data == 1] = 4
#         multilabel_mask[liver_data == 1] = 5
#         multilabel_mask[heart_data == 1] = 6
#         multilabel_mask[aorta_data == 1] = 7
#         multilabel_mask[kl_data == 1] = 8
#         multilabel_mask[kr_data == 1] = 8
#         multilabel_mask[pancreas_data == 1] = 9
#         multilabel_mask[ccal_data == 1] = 10
#         multilabel_mask[ccar_data == 1] = 10
#         multilabel_mask[iva_data == 1] = 11
#         multilabel_mask[pvein_data == 1] = 12
#         multilabel_mask[scal_data == 1] = 13
#         multilabel_mask[scar_data == 1] = 13
#         multilabel_mask[sva_data == 1] = 14
#         multilabel_mask[ribl1_data == 1] = 15
#         multilabel_mask[ribl2_data == 1] = 15
#         multilabel_mask[ribl3_data == 1] = 15
#         multilabel_mask[ribl4_data == 1] = 15
#         multilabel_mask[ribl5_data == 1] = 15
#         multilabel_mask[ribl6_data == 1] = 15
#         multilabel_mask[ribl7_data == 1] = 15
#         multilabel_mask[ribl8_data == 1] = 15
#         multilabel_mask[ribl9_data == 1] = 15
#         multilabel_mask[ribl10_data == 1] = 15
#         multilabel_mask[ribl11_data == 1] = 15
#         multilabel_mask[ribl12_data == 1] = 15
#         multilabel_mask[ribr1_data == 1] = 16
#         multilabel_mask[ribr2_data == 1] = 16
#         multilabel_mask[ribr3_data == 1] = 16
#         multilabel_mask[ribr4_data == 1] = 16
#         multilabel_mask[ribr5_data == 1] = 16
#         multilabel_mask[ribr6_data == 1] = 16
#         multilabel_mask[ribr7_data == 1] = 16
#         multilabel_mask[ribr8_data == 1] = 16
#         multilabel_mask[ribr9_data == 1] = 16
#         multilabel_mask[ribr10_data == 1] = 16
#         multilabel_mask[ribr11_data == 1] = 16
#         multilabel_mask[ribr12_data == 1] = 16
#         multilabel_mask[spleen_data == 1] = 17


#         nifti_mask = nib.Nifti1Image(multilabel_mask, affine=nib.load(llll).affine, header=nib.load(llll).header)
#         nib.save(nifti_mask, os.path.join(segmentations_path, fname + '_multilabel.nii.gz'))

# # Use joblib to parallelize the processing
# for path in tqdm(paths):
#     folders = os.listdir(path)
#     Parallel(n_jobs=3)(delayed(process_folder)(folder, path) for folder in tqdm(folders))

import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

paths = ["/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/hard_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/B30f_B50f/soft_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/hard_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/C_D/soft_masked",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/hard",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_LUNG/soft",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/hard",
         "/valiant02/masi/krishar1/TotalSegmentator_masks_CTkernel_MIDL/STANDARD_BONE/soft"]

masks = ["lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz", "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz",
         "skeletal_muscle.nii.gz", "subcutaneous_fat.nii.gz", "trachea.nii.gz", "liver.nii.gz", "heart.nii.gz", "aorta.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz",
         "pancreas.nii.gz", "common_carotid_artery_left.nii.gz", "common_carotid_artery_right.nii.gz", "inferior_vena_cava.nii.gz", "pulmonary_vein.nii.gz", 
         "subclavian_artery_left.nii.gz", "subclavian_artery_right.nii.gz", "superior_vena_cava.nii.gz", "rib_left_1.nii.gz", "rib_left_2.nii.gz", "rib_left_3.nii.gz",
         "rib_left_4.nii.gz", "rib_left_5.nii.gz", "rib_left_6.nii.gz", "rib_left_7.nii.gz", "rib_left_8.nii.gz", "rib_left_9.nii.gz", "rib_left_10.nii.gz",
         "rib_left_11.nii.gz", "rib_left_12.nii.gz", "rib_right_1.nii.gz", "rib_right_2.nii.gz", "rib_right_3.nii.gz", "rib_right_4.nii.gz", "rib_right_5.nii.gz",
         "rib_right_6.nii.gz", "rib_right_7.nii.gz", "rib_right_8.nii.gz", "rib_right_9.nii.gz", "rib_right_10.nii.gz", "rib_right_11.nii.gz", "rib_right_12.nii.gz", "spleen.nii.gz"]

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
                del mask_data
                gc.collect()

        # Save the multilabel mask
        nifti_mask = nib.Nifti1Image(multilabel_mask, affine=nib.load(llll).affine, header=nib.load(llll).header)
        nib.save(nifti_mask, os.path.join(segmentations_path, fname + '_multilabel.nii.gz'))

# Use joblib to parallelize the processing
for path in paths:
    folders = os.listdir(path)
    Parallel(n_jobs=12)(delayed(process_folder)(path, folder) for folder in tqdm(folders))