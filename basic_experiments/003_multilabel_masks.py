import numpy as np 
import os 
import nibabel as nib 
import matplotlib.pyplot as plt
from tqdm import tqdm

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

for folder in tqdm(os.listdir(paths[0])):
    for files in os.listdir(os.path.join(paths[0], folder)):
        segmentations_path = os.path.join(paths[0], folder, files, 'segmentations')
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
        