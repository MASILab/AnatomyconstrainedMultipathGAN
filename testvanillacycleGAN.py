import torch
import os
from glob import glob
from tqdm import tqdm
from test_dataloader_custom import InferenceDataloader
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from models.networks import define_G
from collections import OrderedDict
from utils_emphysema import EmphysemaAnalysis

# from TMI_SwitchableCycleGAN.networks.adain import half_PolyPhase_resUnet_Adain as Generator

class KernelConversion:
    def __init__(self, config, generator, inkernel, outkernel, inct_dir_synthetic):
        self.config = config
        self.generator = generator
        self.inkernel = inkernel
        self.outkernel = outkernel
        self.inct_dir_synthetic = inct_dir_synthetic

    def kernel_conversion_vanillacyclegan(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gen_dict = torch.load(self.generator)
        ord_dict = OrderedDict()
        for k,v in gen_dict.items():
            ord_dict["module." + k] = v

        generator = define_G(input_nc=1, output_nc=1, ngf=64, netG="resnet_9blocks", norm="instance", use_dropout=False, init_type="normal", init_gain=0.02, gpu_ids=[0])
        generator.load_state_dict(ord_dict)

        in_nii_path = glob(os.path.join(self.config[self.inkernel], '*.nii.gz'))
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)
        print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

        with torch.no_grad():
            generator.eval()
            for nii_path in tqdm(in_nii_path, total = len(in_nii_path)):
                test_dataset = InferenceDataloader(nii_path) #Load the volume into the dataloader
                test_dataset.load_nii()
                test_dataloader = DataLoader(dataset=test_dataset, batch_size = 32, shuffle=False, num_workers=6) #returns the pid, normalized data and the slice index
                converted_scan_idx_slice_map = {}
                for i, data in enumerate(test_dataloader):
                    pid = data['pid']
                    norm_data = data['normalized_data'].float().to(device) #Data on the device
                    fake_image = generator(norm_data) #fake image generated. this is a tensor which needs to be converted to numpy array
                    fake_image_numpy = fake_image.cpu().numpy()
                    slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                    for idx, slice_index in enumerate(slice_idx_list):
                        converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

                nii_file_name = os.path.basename(nii_path)
                converted_image = os.path.join(out_nii, nii_file_name)
                test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
                print(f"{nii_file_name} converted!")


    def run_validation_inference(self):
        for i in tqdm(range(141, 190)):
            print(f"Synthesizing images for epoch {i}......")
            b50tob30f = os.path.join(config_fourkernels_vanillaGAN["b50tob30f"], i + "_net_G_A.pth")
            bonetostd = os.path.join(config_fourkernels_vanillaGAN["bonetostd"], i + "_net_G_A.pth")
            bonetob30f = os.path.join(config_fourkernels_vanillaGAN["bonetob30f"], i + "_net_G_A.pth")
            stdtob30f = os.path.join(config_fourkernels_vanillaGAN["stdtob30f"], i + "_net_G_A.pth")
            shss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + i, config_fourkernels_vanillaGAN["shss"])
            os.makedirs(shss, exist_ok=True)
            ghss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + i, config_fourkernels_vanillaGAN["ghss"])
            os.makedirs(ghss, exist_ok=True)
            gsss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + i, config_fourkernels_vanillaGAN["gsss"])
            os.makedirs(gsss, exist_ok=True)
            ghgs = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + i, config_fourkernels_vanillaGAN["ghgs"])
            os.makedirs(ghgs, exist_ok=True)
            self.kernel_conversion_vanillacyclegan()
            self.kernel_conversion_vanillacyclegan()
            self.kernel_conversion_vanillacyclegan()
            self.kernel_conversion_vanillacyclegan()
            print(f"Images synthesized for epoch {i}!")
            emph_analyze = EmphysemaAnalysis(in_ct_dir=shss, project_dir=shss + "_emphysema")
            emph_analyze.generate_lung_mask()
            emph_analyze.get_emphysema_mask()
            emph_analyze.get_emphysema_measurement()
            print("Emphysema analysis complete!")


    def run_validation_inference_stage2(self):
        """
        Run validation for Philips and GE scanners, paired and unpaired data
        """
        print("Running validation for other baseline models......")
        #self.kernel_conversion_vanillacyclegan()
        print("Evaluating emphyema for synthesized images......")
        emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, project_dir=self.inct_dir_synthetic + "_emphysema")
        emph_analyze.generate_lung_mask()
        emph_analyze.get_emphysema_mask()
        emph_analyze.get_emphysema_measurement()
        print("Emphysema analysis completed for synthesized images!")


    def run_withheld_test_inference(self):
        print("Synthesizing images for withheld data......")
        self.kernel_conversion_vanillacyclegan()
        print("Images synthesized for withheld data!")
        emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, project_dir=self.inct_dir_synthetic + "_emphysema")
        emph_analyze.generate_lung_mask()
        emph_analyze.get_emphysema_mask()
        emph_analyze.get_emphysema_measurement()
        print("Emphysema analysis completed for withheld data!")


#Validation data
config_fourkernels_vanillaGAN = {
    "siemens_hard_100": "/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/hard/ct_masked",
    "siemens_soft_100": "/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/soft/ct_masked",
    "ge_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/hard/ct",
    "ge_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/soft/ct",
    "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
    "b50tob30f": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/B50ftoB30f/vanillacycleGAN_B50ftoB30f_continue_train/",
    "bonetostd":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/BONEtoSTD/vanillacycleGAN_BONEtoSTD_continue_train/",
    "bonetob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/BONEtoB30f/vanillacycleGAN_BONEtoB30f_continue_train/",
    "stdtob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/STDtoB30f/vanillacycleGAN_STDtoB30f_continue_train/",
}


config_fourkernels_withheld_data = {
            "siemens_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/ct_masked",
            "siemens_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/ct_masked",
            "ge_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/ct",
            "ge_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/ct",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "b50tob30f": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/B50ftoB30f/vanillacycleGAN_B50ftoB30f/",
            "bonetostd":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/BONEtoSTD/vanillacycleGAN_BONEtoSTD/",
            "bonetob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/BONEtoB30f/vanillacycleGAN_BONEtoB30f/",
            "stdtob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/STDtoB30f/vanillacycleGAN_STDtoB30f/",}


config_kernels_stage2 = {"philips_hard":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/C_D/hard/ct_masked",
                             "philips_soft":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/C_D/soft/ct_masked",
                             "ge_lung":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_LUNG/hard/ct",
                             "g_lung_std": "/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_LUNG/soft/ct",
                             "phss": "DtoB30f", "psss":"CtoB30f", "glss":"LUNGtoB30f", "phps":"DtoC", "glgs":"LUNGtoSTD",
                             "dtob30f": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/DtoB30f/vanillacycleGAN_DtoB30f/",
                             "ctob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/CtoB30f/vanillacycleGAN_CtoB30f/",
                             "lungtob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/LUNGtoB30f/vanillacycleGAN_LUNGtoB30f/",
                             "dtoc":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/DtoC/vanillacycleGAN_DtoC/",
                             "lungtostd":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/LUNGtoSTD/vanillacycleGAN_LUNGtoSTD/"}


config_kernels_stage2_withheld_data = {
    "philips_hard": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/C_D/hard/ct_masked",
    "philips_soft": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/C_D/soft/ct_masked",
    "ge_lung": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_LUNG/hard/ct",
    "ge_lung_std": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_LUNG/soft/ct",
    "phss": "DtoB30f", "psss":"CtoB30f", "glss":"LUNGtoB30f", "phps":"DtoC", "glgs":"LUNGtoSTD",
    "dtob30f": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/DtoB30f/vanillacycleGAN_DtoB30f/",
    "ctob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/CtoB30f/vanillacycleGAN_CtoB30f/",
    "lungtob30f":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/LUNGtoB30f/vanillacycleGAN_LUNGtoB30f/",
    "dtoc":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/DtoC/vanillacycleGAN_DtoC/",
    "lungtostd":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_models/vanillacycleGAN/LUNGtoSTD/vanillacycleGAN_LUNGtoSTD/"
}


def stagetwo_inference():
    print("Synthesizing images for withheld test data......")
    dtob30f = os.path.join(config_kernels_stage2_withheld_data["dtob30f"], "186_net_G_A.pth")
    ctob30f = os.path.join(config_kernels_stage2_withheld_data["ctob30f"], "186_net_G_A.pth")
    lungtob30f = os.path.join(config_kernels_stage2_withheld_data["lungtob30f"], "186_net_G_A.pth")
    lungtostd = os.path.join(config_kernels_stage2_withheld_data["lungtostd"], "186_net_G_A.pth")
    dtoc = os.path.join(config_kernels_stage2_withheld_data["dtoc"], "186_net_G_A.pth")

    phss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_kernels_stage2_withheld_data["phss"])
    #psss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_kernels_stage2_withheld_data["psss"])
    #glss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_kernels_stage2_withheld_data["glss"])
    #phps = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_kernels_stage2_withheld_data["phps"])
    #glgs = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_kernels_stage2_withheld_data["glgs"])

    inference_dtob30f = KernelConversion(config=config_kernels_stage2_withheld_data, generator=dtob30f, inkernel="philips_hard", outkernel=phss, inct_dir_synthetic=phss)
    #inference_ctob30f = KernelConversion(config=config_kernels_stage2_withheld_data, generator=ctob30f, inkernel="philips_soft", outkernel=psss, inct_dir_synthetic=psss)
    #inference_lungtob30f = KernelConversion(config=config_kernels_stage2_withheld_data, generator=lungtob30f, inkernel="ge_lung", outkernel=glss, inct_dir_synthetic=glss)
    #inference_dtoc = KernelConversion(config=config_kernels_stage2_withheld_data, generator=dtoc, inkernel="philips_hard", outkernel=phps, inct_dir_synthetic=phps)
    #inference_lungtostd = KernelConversion(config=config_kernels_stage2_withheld_data, generator=lungtostd, inkernel="ge_lung", outkernel=glgs, inct_dir_synthetic=glgs)

    inference_dtob30f.run_withheld_test_inference()
    #inference_ctob30f.run_withheld_test_inference()
    #inference_lungtob30f.run_withheld_test_inference()
    #inference_dtoc.run_withheld_test_inference()
    #inference_lungtostd.run_withheld_test_inference()



def stageone_inference():
    print("Synthesizing images for withheld test data......")
    b50ftob30f = os.path.join(config_fourkernels_withheld_data["b50tob30f"], "120_net_G_A.pth")
    bonetostd = os.path.join(config_fourkernels_withheld_data["bonetostd"], "120_net_G_A.pth")
    bonetob30f = os.path.join(config_fourkernels_withheld_data["bonetob30f"], "120_net_G_A.pth")
    stdtob30f = os.path.join(config_fourkernels_withheld_data["stdtob30f"], "120_net_G_A.pth")

    shss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_fourkernels_withheld_data["shss"])
    ghss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_fourkernels_withheld_data["ghss"])
    gsss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_fourkernels_withheld_data["gsss"])
    ghgs = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_withheldtest_data_baseline_results", config_fourkernels_withheld_data["ghgs"])

    inferenceb50fotb30f = KernelConversion(config_fourkernels_withheld_data, b50ftob30f, "siemens_hard", shss)
    inferencebonetob30f = KernelConversion(config_fourkernels_withheld_data, bonetob30f, "ge_hard", ghss)
    inferencestdtob30f = KernelConversion(config_fourkernels_withheld_data, stdtob30f, "ge_soft", gsss)
    inferencebonetostd = KernelConversion(config_fourkernels_withheld_data, bonetostd, "ge_hard", ghgs)
    inferenceb50fotb30f.run_withheld_test_inference()
    inferencebonetob30f.run_withheld_test_inference()
    inferencestdtob30f.run_withheld_test_inference()
    inferencebonetostd.run_withheld_test_inference()


def validate_cycleGAN():
    #Validation for vanilla cycleGAN models for models used in Stage 2
    #Epochs saved in steps of 10
    for i in tqdm(range(10,121,10)):
        dtob30f = os.path.join(config_kernels_stage2["dtob30f"], str(i) + "_net_G_A.pth")
        ctob30f = os.path.join(config_kernels_stage2["ctob30f"], str(i) + "_net_G_A.pth")
        lungtob30f = os.path.join(config_kernels_stage2["lungtob30f"], str(i) + "_net_G_A.pth")
        dtoc = os.path.join(config_kernels_stage2["dtoc"], str(i) + "_net_G_A.pth")
        lungtostd = os.path.join(config_kernels_stage2["lungtostd"], str(i) + "_net_G_A.pth")

        dtob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["phss"])
        ctob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["psss"])
        lungtob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["glss"])
        dtoc_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["phps"])
        lungtostd_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["glgs"])

        inference_dtob30f = KernelConversion(config=config_kernels_stage2, generator=dtob30f, inkernel="philips_hard", outkernel=dtob30f_out, inct_dir_synthetic=dtob30f_out)
        inference_ctob30f = KernelConversion(config=config_kernels_stage2, generator=ctob30f, inkernel="philips_soft", outkernel=ctob30f_out, inct_dir_synthetic=ctob30f_out)
        inference_lungtob30f = KernelConversion(config=config_kernels_stage2, generator=lungtob30f, inkernel="ge_lung", outkernel=lungtob30f_out, inct_dir_synthetic=lungtob30f_out)
        inference_dtoc = KernelConversion(config=config_kernels_stage2, generator=dtoc, inkernel="philips_hard", outkernel=dtoc_out, inct_dir_synthetic=dtoc_out)
        inference_lungtostd = KernelConversion(config=config_kernels_stage2, generator=lungtostd, inkernel="ge_lung", outkernel=lungtostd_out, inct_dir_synthetic=lungtostd_out)

        inference_dtob30f.run_validation_inference_stage2()
        inference_ctob30f.run_validation_inference_stage2()
        inference_lungtob30f.run_validation_inference_stage2()
        inference_dtoc.run_validation_inference_stage2()
        inference_lungtostd.run_validation_inference_stage2()



    #epochs saved during continued training in steps of 1
    for i in tqdm(range(121,201,1)):
        dtob30f = os.path.join(config_kernels_stage2["dtob30f"], str(i) + "_net_G_A.pth")
        ctob30f = os.path.join(config_kernels_stage2["ctob30f"], str(i) + "_net_G_A.pth")
        lungtob30f = os.path.join(config_kernels_stage2["lungtob30f"], str(i) + "_net_G_A.pth")
        dtoc = os.path.join(config_kernels_stage2["dtoc"], str(i) + "_net_G_A.pth")
        lungtostd = os.path.join(config_kernels_stage2["lungtostd"], str(i) + "_net_G_A.pth")

        dtob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["phss"])
        ctob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["psss"])
        lungtob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["glss"])
        dtoc_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["phps"])
        lungtostd_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/baseline_results/vanillacyclegan_validation_data_baseline_results", "epoch_" + str(i), config_kernels_stage2["glgs"])

        inference_dtob30f = KernelConversion(config=config_kernels_stage2, generator=dtob30f, inkernel="philips_hard", outkernel=dtob30f_out, inct_dir_synthetic=dtob30f_out)
        inference_ctob30f = KernelConversion(config=config_kernels_stage2, generator=ctob30f, inkernel="philips_soft", outkernel=ctob30f_out, inct_dir_synthetic=ctob30f_out)
        inference_lungtob30f = KernelConversion(config=config_kernels_stage2, generator=lungtob30f, inkernel="ge_lung", outkernel=lungtob30f_out, inct_dir_synthetic=lungtob30f_out)
        inference_dtoc = KernelConversion(config=config_kernels_stage2, generator=dtoc, inkernel="philips_hard", outkernel=dtoc_out, inct_dir_synthetic=dtoc_out)
        inference_lungtostd = KernelConversion(config=config_kernels_stage2, generator=lungtostd, inkernel="ge_lung", outkernel=lungtostd_out, inct_dir_synthetic=lungtostd_out)

        inference_dtob30f.run_validation_inference_stage2()
        inference_ctob30f.run_validation_inference_stage2()
        inference_lungtob30f.run_validation_inference_stage2()
        inference_dtoc.run_validation_inference_stage2()
        inference_lungtostd.run_validation_inference_stage2()


stagetwo_inference()
