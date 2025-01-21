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



