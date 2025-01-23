from math import log10
from scipy.interpolate import make_interp_spline
from scipy.stats import rankdata
from ridgeplot import ridgeplot
from scipy.stats import wilcoxon, ranksums, mannwhitneyu 
from skimage.metrics import mean_squared_error
from scipy.stats import bootstrap
import pandas as pd 
import numpy as np

def get_rmse_w_ci(pred_list, gt_list):
    rmse_val = np.sqrt(mean_squared_error(gt_list, pred_list))

    def get_rmse_sample(sample_pred_list, sample_gt_list):
        return np.sqrt(mean_squared_error(sample_gt_list, sample_pred_list))

    rmse_ci = bootstrap((gt_list, pred_list), get_rmse_sample, vectorized=False,
                        paired=True, confidence_level=0.95, random_state=0, n_resamples=1000)
    rmse_ci = [rmse_ci.confidence_interval.low, rmse_ci.confidence_interval.high]

    return rmse_val, rmse_ci


def get_rmse(pred_list, gt_list):
    return np.sqrt(mean_squared_error(pred_list, gt_list))


def rmse_ci_paired():
    b30f_ref = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/emphysema_masked/emph.csv")
    b30f_ref = b30f_ref.sort_values(by = "pid")
    b50f = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/emphysema_masked/emph.csv")
    b50f = b50f.sort_values(by = "pid")
    bone = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/emphysema/emph.csv")
    bone = bone.sort_values(by = "pid")
    bone_std = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/emphysema/emph.csv")
    bone_std = bone_std.sort_values(by = "pid")

    b50ftob30f_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_cycgan = b50ftob30f_cycgan.sort_values(by = "pid")
    bonetostd_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/BONEtoSTD_emphysema/emph.csv")
    bonetostd_cycgan = bonetostd_cycgan.sort_values(by = "pid")

    b50ftob30f_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_multipath = b50ftob30f_multipath.sort_values(by = "pid")
    bonetostd_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/BONEtoSTD_emphysema/emph.csv")
    bonetostd_multipath = bonetostd_multipath.sort_values(by = "pid")

    b50ftob30f_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_switchgan = b50ftob30f_switchgan.sort_values(by = "pid")
    bonetostd_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/BONEtoSTD_emphysema/emph.csv")
    bonetostd_switchgan = bonetostd_switchgan.sort_values(by = "pid")

    b50ftob30f_multipathanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_multipathanatomy = b50ftob30f_multipathanatomy.sort_values(by = "pid")
    bonetostd_multipathanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoSTD_emphysema/emph.csv")
    bonetostd_multipathanatomy = bonetostd_multipathanatomy.sort_values(by = "pid")

    #Get the RMSE and confidence intervals for before, cyclegan and multi-path harmonization.
    rmse_b50f_b30f, ci_b50f_b30f = get_rmse_w_ci(b50f['emph_score'], b30f_ref['emph_score'])
    rmse_bone_std, ci_bone_std = get_rmse_w_ci(bone['emph_score'], bone_std['emph_score'])

    rmse_b50f_b30f_cyc, ci_b50f_b30f_cyc = get_rmse_w_ci(b50ftob30f_cycgan['emph_score'], b30f_ref['emph_score'])
    rmse_bone_std_cyc, ci_bone_std_cyc = get_rmse_w_ci(bonetostd_cycgan['emph_score'], bone_std['emph_score'])

    rmse_b50f_b30f_multi, ci_b50f_b30f_multi = get_rmse_w_ci(b50ftob30f_multipath['emph_score'], b30f_ref['emph_score'])
    rmse_bone_std_multi, ci_bone_std_multi = get_rmse_w_ci(bonetostd_multipath['emph_score'], bone_std['emph_score'])

    rmse_b50f_b30f_switchgan, ci_b50f_b30f_switchgan = get_rmse_w_ci(b50ftob30f_switchgan['emph_score'], b30f_ref['emph_score'])
    rmse_bone_std_switchgan, ci_bone_std_switchgan = get_rmse_w_ci(bonetostd_switchgan['emph_score'], bone_std['emph_score'])

    rsme_b50f_b30f_multipathwithanatomy, ci_b50f_b30f_multipathwithanatomy = get_rmse_w_ci(b50ftob30f_multipathanatomy['emph_score'], b30f_ref['emph_score'])
    rsme_bone_std_multipathwithanatomy, ci_bone_std_multipathwithanatomy = get_rmse_w_ci(bonetostd_multipathanatomy['emph_score'], bone_std['emph_score'])

    #Save RMSE and confidence interval together. Round RMSE and CI to 3 decimal places.
    rmse_ci = pd.DataFrame([
        [round(rmse_b50f_b30f, 2), round(ci_b50f_b30f[0], 2), round(ci_b50f_b30f[1], 2),
        round(rmse_b50f_b30f_cyc, 2), round(ci_b50f_b30f_cyc[0], 2), round(ci_b50f_b30f_cyc[1], 2),
        round(rmse_b50f_b30f_multi, 2), round(ci_b50f_b30f_multi[0], 2), round(ci_b50f_b30f_multi[1], 2),
        round(rmse_b50f_b30f_switchgan, 2), round(ci_b50f_b30f_switchgan[0], 2), round(ci_b50f_b30f_switchgan[1], 2),
        round(rsme_b50f_b30f_multipathwithanatomy, 2), round(ci_b50f_b30f_multipathwithanatomy[0], 2), round(ci_b50f_b30f_multipathwithanatomy[1], 2)],
        [round(rmse_bone_std, 2), round(ci_bone_std[0], 2), round(ci_bone_std[1], 2),
        round(rmse_bone_std_cyc, 2), round(ci_bone_std_cyc[0], 2), round(ci_bone_std_cyc[1], 2),
        round(rmse_bone_std_multi, 2), round(ci_bone_std_multi[0], 2), round(ci_bone_std_multi[1], 2),
        round(rmse_bone_std_switchgan, 2), round(ci_bone_std_switchgan[0], 2), round(ci_bone_std_switchgan[1], 2), 
        round(rsme_bone_std_multipathwithanatomy, 2), round(ci_bone_std_multipathwithanatomy[0], 2), round(ci_bone_std_multipathwithanatomy[1], 2)]
    ],
        columns=['RMSE_before', 'CI_low_before', 'CI_high_before',
                'RMSE_cyclegan', 'CI_low_cyclegan', 'CI_high_cyclegan',
                'RMSE_multipath', 'CI_low_multipath', 'CI_high_multipath',
                'RMSE_switchgan', 'CI_low_switchgan', 'CI_high_switchgan', 
                'RMSE_multipathwithanatomy', 'CI_low_multipathwithanatomy', 'CI_high_multipathwithanatomy'],
        index=['B50f_B30f', 'BONE_STD']
    )

    print(rmse_ci)
    rmse_ci.to_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/rmse_ci_emphysema_paired.csv")

def mannwhitney_utest():
    #Show the emphysema distribution for the best epoch on the validation data
    b30f_ref = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/emphysema_masked/emph.csv")
    b30f_ref = b30f_ref.sort_values(by = "pid")
    b50f = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/emphysema_masked/emph.csv")
    b50f = b50f.sort_values(by = "pid")
    bone = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/emphysema/emph.csv")
    bone = bone.sort_values(by = "pid")
    bone_std = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/emphysema/emph.csv")
    bone_std = bone_std.sort_values(by = "pid")


    b30f_ref['Kernel'] = 'B30f (reference)'
    b50f['Kernel'] = 'B50f'
    bone['Kernel'] = 'BONE'
    bone_std['Kernel'] = 'STANDARD'

    bonetob30f_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/BONEtoB30f_emphysema/emph.csv")
    bonetob30f_cycgan = bonetob30f_cycgan.sort_values(by = "pid")
    stdtob30f_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/STDtoB30f_emphysema/emph.csv")
    stdtob30f_cycgan = stdtob30f_cycgan.sort_values(by = "pid")
    b50ftob30f_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_cycgan = b50ftob30f_cycgan.sort_values(by = "pid")
    bonetostd_cycgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results/BONEtoSTD_emphysema/emph.csv")
    bonetostd_cycgan = bonetostd_cycgan.sort_values(by = "pid")


    bonetob30f_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/BONEtoB30f_emphysema/emph.csv")
    bonetob30f_multipath = bonetob30f_multipath.sort_values(by = "pid")
    stdtob30f_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/STDtoB30f_emphysema/emph.csv")
    stdtob30f_multipath = stdtob30f_multipath.sort_values(by = "pid")
    b50ftob30f_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_multipath = b50ftob30f_multipath.sort_values(by = "pid")
    bonetostd_multipath = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images/BONEtoSTD_emphysema/emph.csv")
    bonetostd_multipath = bonetostd_multipath.sort_values(by = "pid")

    bonetob30f_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/BONEtoB30f_emphysema/emph.csv")
    bonetob30f_switchgan = bonetob30f_switchgan.sort_values(by = "pid")
    stdtob30f_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/STDtoB30f_emphysema/emph.csv")
    stdtob30f_switchgan = stdtob30f_switchgan.sort_values(by = "pid")
    b50ftob30f_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_switchgan = b50ftob30f_switchgan.sort_values(by = "pid")
    bonetostd_switchgan = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/switchcyclegan_inference_data_results/BONEtoSTD_emphysema/emph.csv")
    bonetostd_switchgan = bonetostd_switchgan.sort_values(by = "pid")

    bonetob30f_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoB30f_emphysema/emph.csv")
    bonetob30f_multipathwithanatomy = bonetob30f_multipathwithanatomy.sort_values(by = "pid")
    stdtob30f_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/STDtoB30f_emphysema/emph.csv")
    stdtob30f_multipathwithanatomy = stdtob30f_multipathwithanatomy.sort_values(by = "pid")
    b50ftob30f_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/B50ftoB30f_emphysema/emph.csv")
    b50ftob30f_multipathwithanatomy = b50ftob30f_multipathwithanatomy.sort_values(by = "pid")
    bonetostd_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoSTD_emphysema/emph.csv")
    bonetostd_multipathwithanatomy = bonetostd_multipathwithanatomy.sort_values(by = "pid")


    bonetob30f_cycgan['Kernel'] = 'BONEtoB30f'
    stdtob30f_cycgan['Kernel'] = 'STDtoB30f'
    b50ftob30f_cycgan['Kernel'] = 'B50ftoB30f'

    bonetob30f_multipath['Kernel'] = 'BONEtoB30f'
    stdtob30f_multipath['Kernel'] = 'STDtoB30f'
    b50ftob30f_multipath['Kernel'] = 'B50ftoB30f'

    bonetob30f_switchgan['Kernel'] = 'BONEtoB30f'
    stdtob30f_switchgan['Kernel'] = 'STDtoB30f'
    b50ftob30f_switchgan['Kernel'] = 'B50ftoB30f'

    bonetob30f_multipathwithanatomy['Kernel'] = 'BONEtoB30f'
    stdtob30f_multipathwithanatomy['Kernel'] = 'STDtoB30f'
    b50ftob30f_multipathwithanatomy['Kernel'] = 'B50ftoB30f'


    u_b50f_b30f, p_b50f_b30f = mannwhitneyu(b50f['emph_score'], b30f_ref['emph_score'])
    u_bone_b30f, p_bone_b30f = mannwhitneyu(bone['emph_score'], b30f_ref['emph_score'])
    u_bone_std_b30f, p_bone_std_b30f = mannwhitneyu(bone_std['emph_score'], b30f_ref['emph_score'])

    #stat test cycleGAN 
    u_b50f_b30f_cyc, p_b50f_b30f_cyc = mannwhitneyu(b50ftob30f_cycgan['emph_score'], b30f_ref['emph_score'])
    u_bone_b30f_cyc, p_bone_b30f_cyc = mannwhitneyu(bonetob30f_cycgan['emph_score'], b30f_ref['emph_score'])
    u_bone_std_b30f_cyc, p_bone_std_b30f_cyc = mannwhitneyu(stdtob30f_cycgan['emph_score'], b30f_ref['emph_score'])


    u_b50f_b30f_multi, p_b50f_b30f_multi = mannwhitneyu(b50ftob30f_multipath['emph_score'], b30f_ref['emph_score'])
    u_bone_b30f_multi, p_bone_b30f_multi = mannwhitneyu(bonetob30f_multipath['emph_score'], b30f_ref['emph_score'])
    u_bone_std_b30f_multi, p_bone_std_b30f_multi = mannwhitneyu(stdtob30f_multipath['emph_score'], b30f_ref['emph_score'])

    u_b50f_b30f_switchgan, p_b50f_b30f_switchgan = mannwhitneyu(b50ftob30f_switchgan['emph_score'], b30f_ref['emph_score'])
    u_bone_b30f_switchgan, p_bone_b30f_switchgan = mannwhitneyu(bonetob30f_switchgan['emph_score'], b30f_ref['emph_score'])
    u_bone_std_b30f_switchgan, p_bone_std_b30f_switchgan = mannwhitneyu(stdtob30f_switchgan['emph_score'], b30f_ref['emph_score'])

    u_b50f_b30f_multipath_anatomy, p_b50f_b30f_multipath_anatomy = mannwhitneyu(b50ftob30f_multipathwithanatomy['emph_score'], b30f_ref['emph_score'])
    u_bone_b30f_multipath_anatomy, p_bone_b30f_multipath_anatomy = mannwhitneyu(bonetob30f_multipathwithanatomy['emph_score'], b30f_ref['emph_score'])
    u_bone_std_b30f_multipath_anatomy, p_bone_std_b30f_multipath_anatomy = mannwhitneyu(stdtob30f_multipathwithanatomy['emph_score'], b30f_ref['emph_score'])
    #Save the p values for before harmonization, after harmonization with cycleGAN and after harmonization with multi-path. 
    #Columns are before, cycleGAN, multi-path. Rows are B50f, BONE, BONE_STD, D, C, LUNG

    p_values = pd.DataFrame([[p_b50f_b30f, p_b50f_b30f_cyc, p_b50f_b30f_multi, p_b50f_b30f_switchgan, p_b50f_b30f_multipath_anatomy],
                            [p_bone_b30f, p_bone_b30f_cyc, p_bone_b30f_multi, p_bone_b30f_switchgan, p_bone_b30f_multipath_anatomy],
                            [p_bone_std_b30f, p_bone_std_b30f_cyc, p_bone_std_b30f_multi, p_bone_std_b30f_switchgan, p_bone_std_b30f_multipath_anatomy],
                          ],
                            columns = ['Before_harmonization', 'After_harmonization (CycleGAN)', 'After_harmonization (Multi-path)', 'After_harmonization (SwitchGAN)', 'After_harmonization (Multi-path with anatomy)'],
                            index = ['B50f', 'BONE', 'BONE_STD'])

    print(p_values)
    p_values.to_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/p_values_emphysema_unpaired.csv")

def cohens_d(group1, group2):
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1) * std1 ** 2 + (n2 -1) * std2**2) / (n1+n2-2))
        d = (mean1 - mean2) / pooled_std
        return d

def effect_size():
    bonetob30f = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/bonetob30f_finalresults.csv")
    stdtob30f = pd.read_csv("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/stdtob30f_finalresults.csv")

    bonetob30f_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/BONEtoB30f.csv")
    stdtob30f_multipathwithanatomy = pd.read_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/STDtoB30f.csv")

    merged_bonetob30f = pd.merge(bonetob30f, bonetob30f_multipathwithanatomy, on = "Patient")
    merged_stdtob30f = pd.merge(stdtob30f, stdtob30f_multipathwithanatomy, on = "Patient")

    bonetob30f_multi_muscle = list(bonetob30f['Dice_Multipath_Muscle'])
    bonetob30f_multi_fat = list(bonetob30f['Dice_Multipath_Fat'])
    bonetob30f_cycgan_muscle = list(bonetob30f['Dice_CycleGAN_Muscle'])
    bonetob30f_cycgan_fat = list(bonetob30f['Dice_CycleGAN_Fat'])
    bonetob30f_switchgan_muscle = list(bonetob30f['Dice_SwitchGAN_Muscle'])
    bonetob30f_switchgan_fat = list(bonetob30f['Dice_SwitchGAN_Fat'])
    bonetob30f_multipath_anatomy = list(merged_bonetob30f['Dice_Multipath_with_anatomy_context_Muscle'])
    bonetob30f_multipath_anatomy_fat = list(merged_bonetob30f['Dice_Multipath_with_anatomy_Fat'])

    stdtob30f_multi_muscle = list(stdtob30f['Dice_Multipath_Muscle'])
    stdtob30f_multi_fat = list(stdtob30f['Dice_Multipath_Fat'])
    stdtob30f_cycgan_muscle = list(stdtob30f['Dice_CycleGAN_Muscle'])
    stdtob30f_cycgan_fat = list(stdtob30f['Dice_CycleGAN_Fat'])
    stdtob30f_switchgan_muscle = list(stdtob30f['Dice_SwitchGAN_Muscle'])
    stdtob30f_switchgan_fat = list(stdtob30f['Dice_SwitchGAN_Fat'])
    stdtob30f_multipath_anatomy = list(merged_stdtob30f['Dice_Multipath_with_anatomy_context_Muscle'])
    stdtob30f_multipath_anatomy_fat = list(merged_stdtob30f['Dice_Multipath_with_anatomy_Fat']) 



    d_bone_muscle_multcyc = cohens_d(bonetob30f_multipath_anatomy, bonetob30f_cycgan_muscle)
    d_bone_fat_multcyc = cohens_d(bonetob30f_multipath_anatomy_fat, bonetob30f_cycgan_fat)
    d_bone_muscle_multanatmult = cohens_d(bonetob30f_multipath_anatomy, bonetob30f_multi_muscle)
    d_bone_fat_multanatmult = cohens_d(bonetob30f_multipath_anatomy_fat, bonetob30f_multi_fat)
    d_bone_muscle_multswitch = cohens_d(bonetob30f_multipath_anatomy, bonetob30f_switchgan_muscle)
    d_bone_fat_multswitch = cohens_d(bonetob30f_multipath_anatomy_fat, bonetob30f_switchgan_fat)

    d_std_muscle_multcyc = cohens_d(stdtob30f_multipath_anatomy, stdtob30f_cycgan_muscle)
    d_std_fat_multcyc = cohens_d(stdtob30f_multipath_anatomy_fat, stdtob30f_cycgan_fat)
    d_std_muscle_multanatmult = cohens_d(stdtob30f_multipath_anatomy, stdtob30f_multi_muscle)
    d_std_fat_multanatmult = cohens_d(stdtob30f_multipath_anatomy_fat, stdtob30f_multi_fat)
    d_std_muscle_multswitch = cohens_d(stdtob30f_multipath_anatomy, stdtob30f_switchgan_muscle)
    d_std_fat_multswitch = cohens_d(stdtob30f_multipath_anatomy_fat, stdtob30f_switchgan_fat)

    effect_size = pd.DataFrame([[d_bone_muscle_multcyc, d_bone_fat_multcyc, d_bone_muscle_multanatmult, d_bone_fat_multanatmult ,d_bone_muscle_multswitch, d_bone_fat_multswitch],
                                [d_std_muscle_multcyc, d_std_fat_multcyc, d_std_muscle_multanatmult, d_std_fat_multanatmult, d_std_muscle_multswitch, d_std_fat_multswitch]],
                                columns = ['Muscle (Multipath_withAnatomy vs CycleGAN)', 'Fat (Multipath_withAnatomy vs CycleGAN)', 'Muscle (Multipath_withAnatomy vs MultipathGAN)', 'Fat (Multipath_wtihAnatomy vs MultipathGAN)', 
                                           'Muscle (Multipath_withAnatomy vs SwitchGAN)', 'Fat (Multipath_withAnatomy vs SwitchGAN)'],
                                index = ['BONE', 'STD'])
    
    print(effect_size)
    effect_size.to_csv("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE/Cohen's_d_dice_unpaired.csv")

rmse_ci_paired()
mannwhitney_utest()
effect_size() 