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
            shss = os.path.join("/path/to/val_data", "epoch_" + i, config_fourkernels_vanillaGAN["shss"])
            os.makedirs(shss, exist_ok=True)
            ghss = os.path.join("/path/to/val_data", "epoch_" + i, config_fourkernels_vanillaGAN["ghss"])
            os.makedirs(ghss, exist_ok=True)
            gsss = os.path.join("/path/to/val_data", "epoch_" + i, config_fourkernels_vanillaGAN["gsss"])
            os.makedirs(gsss, exist_ok=True)
            ghgs = os.path.join("/path/to/val_data", "epoch_" + i, config_fourkernels_vanillaGAN["ghgs"])
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
    "siemens_hard_100": "/path/to/val_data",
    "siemens_soft_100": "/path/to/val_data",
    "ge_hard_100":"/path/to/val_data",
    "ge_soft_100":"/path/to/val_data",
    "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
    "b50tob30f": "/path/to/checkpoints",
    "bonetostd":"/path/to/checkpoints",
    "bonetob30f":"/path/to/checkpoints",
    "stdtob30f":"/path/to/checkpoints"
}


config_fourkernels_withheld_data = {
            "siemens_hard":"/path/to/test_data",
            "siemens_soft":"/path/to/test_data",
            "ge_hard":"/path/to/test_data",
            "ge_soft":"/path/to/test_data",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "b50tob30f": "/path/to/checkpoints",
            "bonetostd":"/path/to/checkpoints",
            "bonetob30f":"/path/to/checkpoints",
            "stdtob30f":"/path/to/checkpoints"}


def stageone_inference():
    print("Synthesizing images for withheld test data......")
    b50ftob30f = os.path.join(config_fourkernels_withheld_data["b50tob30f"], "120_net_G_A.pth")
    bonetostd = os.path.join(config_fourkernels_withheld_data["bonetostd"], "120_net_G_A.pth")
    bonetob30f = os.path.join(config_fourkernels_withheld_data["bonetob30f"], "120_net_G_A.pth")
    stdtob30f = os.path.join(config_fourkernels_withheld_data["stdtob30f"], "120_net_G_A.pth")

    shss = os.path.join("/path/to/output", config_fourkernels_withheld_data["shss"])
    ghss = os.path.join("/path/to/output", config_fourkernels_withheld_data["ghss"])
    gsss = os.path.join("/path/to/output", config_fourkernels_withheld_data["gsss"])
    ghgs = os.path.join("/path/to/output", config_fourkernels_withheld_data["ghgs"])

    inferenceb50fotb30f = KernelConversion(config_fourkernels_withheld_data, b50ftob30f, "siemens_hard", shss)
    inferencebonetob30f = KernelConversion(config_fourkernels_withheld_data, bonetob30f, "ge_hard", ghss)
    inferencestdtob30f = KernelConversion(config_fourkernels_withheld_data, stdtob30f, "ge_soft", gsss)
    inferencebonetostd = KernelConversion(config_fourkernels_withheld_data, bonetostd, "ge_hard", ghgs)
    inferenceb50fotb30f.run_withheld_test_inference()
    inferencebonetob30f.run_withheld_test_inference()
    inferencestdtob30f.run_withheld_test_inference()
    inferencebonetostd.run_withheld_test_inference()

