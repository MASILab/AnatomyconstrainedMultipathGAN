import pandas as pd
from Emphysemamodel.lungmask import ProcessLungMask
import logging
import os
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EmphysemaAnalysis:
    def __init__(self, in_ct_dir, project_dir):
        self.in_ct_dir = in_ct_dir
        self.project_dir = project_dir

    def _generate_lung_mask_config(self):
        return {
            'input': {
                'ct_dir': self.in_ct_dir
            },
            'output': {
                'root_dir': self.project_dir,
                'if_overwrite': True
            },
            'model': {
                'model_lung_mask': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/models/lung_mask' #Must add this model to my location in /nfs/masi/krishar1
            }
        }

    def generate_lung_mask(self):
        """
        Preprocessing, generating masks, level prediction, get the TCI evaluation, etc.
        :return:
        """
        # logger.info(f'##### Start preprocess #####')
        config_preprocess = self._generate_lung_mask_config()
        logger.info(f'Get lung mask\n')
        lung_mask_generator = ProcessLungMask(config_preprocess)
        lung_mask_generator.run()

    def get_emphysema_mask(self):
        print(f'Generate emphysema masks')
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')

        emph_threshold = -950
        ct_list = os.listdir(self.in_ct_dir)

        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')
        os.makedirs(emph_mask_dir, exist_ok=True)

        def _process_single_case(ct_file_name):
            in_ct = os.path.join(self.in_ct_dir, ct_file_name)
            lung_mask = os.path.join(lung_mask_dir, ct_file_name)

            ct_img = nib.load(in_ct)
            lung_img = nib.load(lung_mask)

            ct_data = ct_img.get_fdata()
            lung_data = lung_img.get_fdata()

            emph_data = np.zeros(ct_data.shape, dtype=int)
            emph_data[(ct_data < emph_threshold) & (lung_data > 0)] = 1

            emph_img = nib.Nifti1Image(emph_data,
                                       affine=ct_img.affine,
                                       header=ct_img.header)
            emph_path = os.path.join(emph_mask_dir, ct_file_name)
            nib.save(emph_img, emph_path)

        Parallel(
            n_jobs=10,
            prefer='threads'
        )(delayed(_process_single_case)(ct_file_name)
          for ct_file_name in tqdm(ct_list, total=len(ct_list)))

    def get_emphysema_measurement(self):
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')
        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')

        ct_file_list = os.listdir(lung_mask_dir)
        record_list = []
        for ct_file_name in ct_file_list:
            pid = ct_file_name.replace('.nii.gz', '')

            lung_mask = nib.load(os.path.join(lung_mask_dir, ct_file_name)).get_fdata()
            emph_mask = nib.load(os.path.join(emph_mask_dir, ct_file_name)).get_fdata()

            emph_score = 100. * np.count_nonzero(emph_mask) / np.count_nonzero(lung_mask)

            record_list.append({
                'pid': pid,
                'emph_score': emph_score
            })

        emph_score_df = pd.DataFrame(record_list)
        emph_score_csv = os.path.join(self.project_dir, 'emph.csv')
        print(f'Save to {emph_score_csv}')
        emph_score_df.to_csv(emph_score_csv, index=False)

list_kernels_all = [('DtoC_epoch110_newsubjects', 'starganL2weightschednewkernels', 'DtoC_epoch110_newsubjects_emphysema'),
                ('DtoB30f_epoch110_newsubjects', 'starganL2weightschednewkernels', 'DtoB30f_epoch110_newsubjects_emphysema'),
                ('CtoB30f_epoch110_newsubjects', 'starganL2weightschednewkernels', 'CtoB30f_epoch110_newsubjects_emphysema'),
                ('LUNGtoB30f_epoch110_newsubjects', 'starganL2weightschednewkernels', 'LUNGtoB30f_epoch110_newsubjects_emphysema'),
                ('LUNGtoLUNGSTD_epoch110_newsubjects', 'starganL2weightschednewkernels', 'LUNGtoLUNGSTD_epoch110_newsubjects_emphysema'),
                ('LUNGSTDtoB30f_epoch110_newsubjects', 'starganL2weightschednewkernels', 'LUNGSTDtoB30f_epoch110_newsubjects_emphysema'),
                ('B50ftoB30f_epoch110_newsubjects', 'starganL2weightsched', 'B50ftoB30f_epoch110_newsubjects_emphysema'),
                ('BONEtoB30f_epoch110_newsubjects', 'starganL2weightsched', 'BONEtoB30f_epoch110_newsubjects_emphysema'),
                ('STDtoB30f_epoch110_newsubjects', 'starganL2weightsched', 'STDtoB30f_epoch110_newsubjects_emphysema'), 
                ('BONEtoSTD_epoch110_newsubjects', 'starganL2weightsched', 'BONEtoSTD_epoch110_newsubjects_emphysema'),]



def run_emphysema_conversion():
    for kernel, expname, output in tqdm(list_kernels_all):
        in_ct_dir = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/Journal_results", expname, kernel)
        project_dir = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/Journal_results",expname, output)
        print(f'Processing {in_ct_dir}')
        print("Output directory: ", project_dir)
        os.makedirs(project_dir, exist_ok=True)
        emph_analyzer = EmphysemaAnalysis(in_ct_dir=in_ct_dir, project_dir=project_dir)
        emph_analyzer.generate_lung_mask()
        emph_analyzer.get_emphysema_mask()
        emph_analyzer.get_emphysema_measurement()

def run_emphysema_newinference():
    kernels = ["STANDARD_BONE"]
    kernel_type = ["hard", "soft"]
    inputpath = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application"
    for kernel in tqdm(kernels):
        for ktype in kernel_type:
            kernelpath = os.path.join(inputpath, kernel, ktype, "ct")
            outpath = os.path.join(inputpath, kernel, ktype, "emphysema")
            os.makedirs(outpath, exist_ok = True)
            emph_analyzer = EmphysemaAnalysis(in_ct_dir=kernelpath, project_dir=outpath)
            emph_analyzer.generate_lung_mask()
            emph_analyzer.get_emphysema_mask()
            emph_analyzer.get_emphysema_measurement()


def emphysema_switchganbaseline():
    projdir = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcycleGAN"
    for file in tqdm(os.listdir(projdir)):
            in_ct_dir = os.path.join(projdir, file)
            out_dir = os.path.join(projdir, file + "_emphysema")
            os.makedirs(out_dir, exist_ok = True)
            emph_analyzer = EmphysemaAnalysis(in_ct_dir=in_ct_dir, project_dir=out_dir)
            emph_analyzer.generate_lung_mask()
            emph_analyzer.get_emphysema_mask()
            emph_analyzer.get_emphysema_measurement()


# emphysema_baseline()

# run_emphysema_conversion()

def emphysema_multipathresnet():
    projdir = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images"
    # epochs = ["121", "122", "123", "124", "125", "126", "127"]
    # epochs = ["128", "129", "130", "131", "132"]
    epochs = ["133", "134", "135", "136", "137", "138", "139", "140"]
    # epochs = ["100", "110"]
    for epoch in tqdm(epochs):
        proj_dir = os.path.join(projdir, "epoch_" + epoch)
        for file in tqdm(os.listdir(proj_dir)):
            in_ct_dir = os.path.join(proj_dir, file)
            out_dir = os.path.join(proj_dir, file + "_emphysema")
            print(f'Processing {in_ct_dir}')
            print("Output directory: ", out_dir)
            os.makedirs(out_dir, exist_ok = True)
            emph_analyzer = EmphysemaAnalysis(in_ct_dir=in_ct_dir, project_dir=out_dir)
            emph_analyzer.generate_lung_mask()
            emph_analyzer.get_emphysema_mask()
            emph_analyzer.get_emphysema_measurement()
    


def emphysema_vanillacyclegan():
    projdir = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results"
    # epochs = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120"]
    # epochs = ['121', '122', '123', '124', '125', '126', '127', '128']
    epochs = ["134", "135", "136", "137", "138", "139", "140"]
    for epoch in tqdm(epochs):
        proj_dir = os.path.join(projdir, "epoch_" + epoch)
        for file in tqdm(os.listdir(proj_dir)):
            in_ct_dir = os.path.join(proj_dir, file)
            out_dir = os.path.join(proj_dir, file + "_emphysema")
            print(f'Processing {in_ct_dir}')
            print("Output directory: ", out_dir)
            os.makedirs(out_dir, exist_ok = True)
            emph_analyzer = EmphysemaAnalysis(in_ct_dir=in_ct_dir, project_dir=out_dir)
            emph_analyzer.generate_lung_mask()
            emph_analyzer.get_emphysema_mask()
            emph_analyzer.get_emphysema_measurement()



# emphysema_vanillacyclegan()
# emphysema_multipathresnet()


#Manual input of input and output directories:     
    

class correctEmphysema:
    def __init__(self, in_ct_dir, lung_mask_dir, emph_mask_path):
        self.in_ct_dir = in_ct_dir
        self.lung_mask_dir = lung_mask_dir
        self.emph_mask_path = emph_mask_path


    def emph_mask_measurement(self):
        #Compute corrected emphsyema mask and score, save corrected emphysema masks, scores
        emph_threshold = -950
        ct_list = os.listdir(self.in_ct_dir)
        emph_mask_dir = os.path.join(self.emph_mask_path, 'corrected_emphysema') #to save the corrected emphysema masks
        os.makedirs(emph_mask_dir, exist_ok=True)

        for file in tqdm(ct_list):
            ct_image = nib.load(os.path.join(self.in_ct_dir, file))
            lung_mask = nib.load(os.path.join(self.lung_mask_dir, file))
            ct_data = ct_image.get_fdata()
            lung_data = lung_mask.get_fdata()

            emph_data = np.zeros(ct_data.shape, dtype=int)
            emph_data[(ct_data < emph_threshold) & (lung_data > 0)] = 1

            emph_img = nib.Nifti1Image(emph_data,
                                    affine=ct_image.affine,
                                    header=ct_image.header)
            emph_path = os.path.join(emph_mask_dir, file)
            nib.save(emph_img, emph_path)


        record_list = []
        for ct_file_name in ct_list:
            pid = ct_file_name.replace('.nii.gz', '')

            lung_mask = nib.load(os.path.join(self.lung_mask_dir, ct_file_name)).get_fdata()
            emph_mask = nib.load(os.path.join(emph_mask_dir, ct_file_name)).get_fdata()

            emph_score = 100. * np.count_nonzero(emph_mask) / np.count_nonzero(lung_mask)

            record_list.append({
                'pid': pid,
                'emph_score': emph_score
            })

        emph_score_df = pd.DataFrame(record_list)
        emph_score_csv = os.path.join(self.emph_mask_path, 'emph_corrected.csv')
        print(f'Save to {emph_score_csv}')
        emph_score_df.to_csv(emph_score_csv, index=False)


# #Run this function to get the corrected emphysema masks and scores for cycleGAN and multipathGAN
# emph_correction = correctEmphysema(in_ct_dir = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/BONEtoB30f",
#                       lung_mask_dir="/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/emphysema/lung_mask",
#                       emph_mask_path="/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/BONEtoB30f_emphysema")
# emph_correction.emph_mask_measurement()



