import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk 
import sklearn.metrics as metrics
import pandas as pd

def make_data(image_path, list_of_files, target_path):
    for file in tqdm(list_of_files):
        fname = file.split(".")[0]
        source_file = os.path.join(image_path, file)
        destination = os.path.join(target_path, fname)
        os.makedirs(destination, exist_ok = True)
        print("cp " + source_file + " " + destination)
        os.system("cp " + source_file + " " + destination)


def make_data_anatomyconstrained():
    bonetob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoB30f"
    stdtob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/STDtoB30f"
    b50ftob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/B50ftoB30f"
    bonetostd = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoSTD"

    tseg_bonetob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/BONEtoB30f"
    tseg_stdtob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/STDtoB30f"
    tseg_b50ftob30f = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/B50ftoB30f"
    tseg_bonetostd = "/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/BONEtoSTD"

    #create directories 

    bonetob30f_files = os.listdir(bonetob30f)
    stdtob30f_files = os.listdir(stdtob30f)
    b50ftob30f_files = os.listdir(b50ftob30f)
    bonetostd_files = os.listdir(bonetostd)

    make_data(bonetob30f, bonetob30f_files, tseg_bonetob30f)
    make_data(stdtob30f, stdtob30f_files, tseg_stdtob30f)
    make_data(b50ftob30f, b50ftob30f_files, tseg_b50ftob30f)
    make_data(bonetostd, bonetostd_files, tseg_bonetostd)

    tseg_b50ftob30f_files = os.listdir(tseg_b50ftob30f)
    tseg_bonetob30f_files = os.listdir(tseg_bonetob30f)
    tseg_stdtob30f_files = os.listdir(tseg_stdtob30f)
    tseg_bonetostd_files = os.listdir(tseg_bonetostd)

    for dir in tqdm(tseg_b50ftob30f_files):
        nifti_file = os.listdir(os.path.join(tseg_b50ftob30f, dir))
        output = os.path.join(tseg_b50ftob30f, dir, "segmentations")
        os.makedirs(output, exist_ok = True)
        print(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
        os.system(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
    
    for dir in tqdm(tseg_bonetob30f_files):
        nifti_file = os.listdir(os.path.join(tseg_bonetob30f, dir))
        output = os.path.join(tseg_bonetob30f, dir, "segmentations")
        os.makedirs(output, exist_ok = True)
        print(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
        os.system(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
    
    for dir in tqdm(tseg_stdtob30f_files):
        nifti_file = os.listdir(os.path.join(tseg_stdtob30f, dir))
        output = os.path.join(tseg_stdtob30f, dir, "segmentations")
        os.makedirs(output, exist_ok = True)
        print(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
        os.system(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
    
    for dir in tqdm(tseg_bonetostd_files):
        nifti_file = os.listdir(os.path.join(tseg_bonetostd, dir))
        output = os.path.join(tseg_bonetostd, dir, "segmentations")
        os.makedirs(output, exist_ok = True)
        print(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")
        os.system(f"TotalSegmentator -i {nifti_file} -o {output} -ta tissue_types")


def dice_coefficient(truth, pred):
    intersect = np.sum(truth * pred)
    union = np.sum(truth) + np.sum(pred)
    dice = (2.0 * intersect ) / union
    return dice


def tissue_statistics_unpaired(original_path, multipath_withanatomy, df_save_path):
    original = sorted(os.listdir(original_path))
    multipath_with_anatomy = sorted(os.listdir(multipath_withanatomy))

    dice_multipath_with_anatomy_muscle = {}
    dice_multipath_with_anatomy_fat = {}


    for file in tqdm(original):
        original_file_muscle = os.path.join(original_path, file, "segmentations", "skeletal_muscle.nii.gz")
        multipath_with_anatomy_file_muscle = os.path.join(multipath_withanatomy, file, "segmentations", "skeletal_muscle.nii.gz")

        original_file_fat = os.path.join(original_path, file, "segmentations", "subcutaneous_fat.nii.gz")
        multipath_with_anatomy_file_fat = os.path.join(multipath_withanatomy, file, "segmentations", "subcutaneous_fat.nii.gz")

        original_img_muscle = nib.load(original_file_muscle).get_fdata()
        multipath_with_anatomy_img_muscle = nib.load(multipath_with_anatomy_file_muscle).get_fdata()

        original_img_fat = nib.load(original_file_fat).get_fdata()
        multipath_with_anatomy_img_fat = nib.load(multipath_with_anatomy_file_fat).get_fdata()
 
        dice_multipath_with_anatomy_muscle[file] = dice_coefficient(original_img_muscle, multipath_with_anatomy_img_muscle)
        dice_multipath_with_anatomy_fat[file] = dice_coefficient(original_img_fat, multipath_with_anatomy_img_fat)

    df = pd.DataFrame()
    df["Patient"] = original
    df["Dice_Multipath_with_anatomy_context_Muscle"] = list(dice_multipath_with_anatomy_muscle.values())
    df["Dice_Multipath_with_anatomy_Fat"] = list(dice_multipath_with_anatomy_fat.values())
    df.to_csv(df_save_path, index = False)
    print("Dataframe saved to", df_save_path)



# make_data_anatomyconstrained()

tissue_statistics_unpaired(original_path = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/TotalSegtissues",
                           multipath_withanatomy="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/BONEtoB30f",
                           df_save_path="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoB30f.csv")

tissue_statistics_unpaired(original_path = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/TotalSegtissues",
                            multipath_withanatomy="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/STDtoB30f",
                            df_save_path="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/STDtoB30f.csv")

# tissue_statistics_unpaired(original_path = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/TotalSegtissues",
#                             multipath_withanatomy="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/B50ftoB30f",
#                             df_save_path="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/B50ftoB30f.csv")

# tissue_statistics_unpaired(original_path = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/TotalSegtissues",
#                             multipath_withanatomy="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/TotalSegtissue_unpaired/BONEtoSTD",
#                             df_save_path="/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoSTD.csv")

