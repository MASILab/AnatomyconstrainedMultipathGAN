import torch
import os
from glob import glob
from tqdm import tqdm
from test_dataloader_custom import InferenceDataloader
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from models.networks import ResBlocklatent, ResNetEncoder, ResNetDecoder, G_decoder, G_encoder
from collections import OrderedDict
import torch.nn as nn
from utils_emphysema import EmphysemaAnalysis

#Use the 100 volumes in /nfs as a validation dataset. Do not reuse this dataset during testing (inference on withheld data)!
#Evaluate various checkpoints on this dataset.

class GenerateInferenceMultipathGAN:
    def __init__(self, config, input_encoder, output_decoder, inkernel, outkernel, inct_dir_synthetic):
        self.config = config
        self.input_encoder = input_encoder #Must be a path to a checkpoint (.pth)
        self.output_decoder = output_decoder #Must be path to a checkpoint (.pth)
        self.inkernel = inkernel
        self.outkernel = outkernel
        self.inct_dir_synthetic = inct_dir_synthetic #For emphysema analysis

    def generate_images(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.input_encoder)
        decoder = torch.load(self.output_decoder)
        encoderdict = OrderedDict()
        decoderdict = OrderedDict()
        for k, v in encoder.items():
            encoderdict["module." + k] = v
        for k, v in decoder.items():
            decoderdict["module." + k] = v

        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect')
        resencode = G_encoder(input_nc=1, ngf=64, netG_encoder="resnet_encoder", norm = 'instance', init_type='normal', init_gain=0.02, latent_layer=shared_latent, gpu_ids=[0])
        resdecode = G_decoder(output_nc=1, ngf=64, netG_decoder="resnet_decoder", norm = 'instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
        resencode.load_state_dict(encoderdict)
        resdecode.load_state_dict(decoderdict)

        in_nii_path = glob(os.path.join(self.config[self.inkernel], '*.nii.gz')) #Find nifti images in the specific location.
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)
        print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            for nii_path in tqdm(in_nii_path, total = len(in_nii_path)):
                test_dataset = InferenceDataloader(nii_path) #Load the volume into the dataloader
                test_dataset.load_nii()
                test_dataloader = DataLoader(dataset=test_dataset, batch_size = 25, shuffle=False, num_workers=4) #returns the pid, normalized data and the slice index
                converted_scan_idx_slice_map = {}
                for i, data in enumerate(test_dataloader):
                    pid = data['pid']
                    norm_data = data['normalized_data'].float().to(device) #Data on the device
                    latent = resencode(norm_data)
                    fake_image = resdecode(latent) #fake image generated. this is a tensor which needs to be converted to numpy array
                    fake_image_numpy = fake_image.cpu().numpy()
                    slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                    for idx, slice_index in enumerate(slice_idx_list):
                        converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

                nii_file_name = os.path.basename(nii_path)
                converted_image = os.path.join(out_nii, nii_file_name)
                test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
                print(f"{nii_file_name} converted!")
    
    def generate_single_image(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.input_encoder)
        decoder = torch.load(self.output_decoder)
        encoderdict = OrderedDict()
        decoderdict = OrderedDict()
        for k, v in encoder.items():
            encoderdict["module." + k] = v
        for k, v in decoder.items():
            decoderdict["module." + k] = v

        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect')
        resencode = G_encoder(input_nc=1, ngf=64, netG_encoder="resnet_encoder", norm = 'instance', init_type='normal', init_gain=0.02, latent_layer=shared_latent, gpu_ids=[0])
        resdecode = G_decoder(output_nc=1, ngf=64, netG_decoder="resnet_decoder", norm = 'instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
        resencode.load_state_dict(encoderdict)
        resdecode.load_state_dict(decoderdict)

        in_nii_path = self.inkernel
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            test_dataset = InferenceDataloader(in_nii_path) #Load the volume into the dataloader
            test_dataset.load_nii()
            test_dataloader = DataLoader(dataset=test_dataset, batch_size = 25, shuffle=False, num_workers=4) #returns the pid, normalized data and the slice index
            converted_scan_idx_slice_map = {}
            for i, data in enumerate(test_dataloader):
                pid = data['pid']
                norm_data = data['normalized_data'].float().to(device) #Data on the device
                latent = resencode(norm_data)
                fake_image = resdecode(latent) #fake image generated. this is a tensor which needs to be converted to numpy array
                fake_image_numpy = fake_image.cpu().numpy()
                slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                for idx, slice_index in enumerate(slice_idx_list):
                    converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

            nii_file_name = os.path.basename(in_nii_path)
            converted_image = os.path.join(out_nii, nii_file_name)
            test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
            print(f"{nii_file_name} converted!")

    def generate_images_stage2(self, enc, dec):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.input_encoder)[enc]
        decoder = torch.load(self.output_decoder)[dec]
        encoderdict = OrderedDict()
        decoderdict = OrderedDict()
        for k, v in encoder.items():
            encoderdict["module." + k] = v
        for k, v in decoder.items():
            decoderdict["module." + k] = v

        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect')
        resencode = G_encoder(input_nc=1, ngf=64, netG_encoder="resnet_encoder", norm = 'instance', init_type='normal', init_gain=0.02, latent_layer=shared_latent, gpu_ids=[0])
        resdecode = G_decoder(output_nc=1, ngf=64, netG_decoder="resnet_decoder", norm = 'instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
        resencode.load_state_dict(encoderdict)
        resdecode.load_state_dict(decoderdict)

        in_nii_path = glob(os.path.join(self.config[self.inkernel], '*.nii.gz')) #Find nifti images in the specific location.
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)
        print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            for nii_path in tqdm(in_nii_path, total = len(in_nii_path)):
                test_dataset = InferenceDataloader(nii_path) #Load the volume into the dataloader
                test_dataset.load_nii()
                test_dataloader = DataLoader(dataset=test_dataset, batch_size = 25, shuffle=False, num_workers=6) #returns the pid, normalized data and the slice index
                converted_scan_idx_slice_map = {}
                for i, data in enumerate(test_dataloader):
                    pid = data['pid']
                    norm_data = data['normalized_data'].float().to(device) #Data on the device
                    latent = resencode(norm_data)
                    fake_image = resdecode(latent) #fake image generated. this is a tensor which needs to be converted to numpy array
                    fake_image_numpy = fake_image.cpu().numpy()
                    slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                    for idx, slice_index in enumerate(slice_idx_list):
                        converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

                nii_file_name = os.path.basename(nii_path)
                converted_image = os.path.join(out_nii, nii_file_name)
                test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
                print(f"{nii_file_name} converted!")


    def run_validation_inference(self):
        for i in tqdm(range(170, 181)):
            print(f"Synthesizing images for epoch {i}......")
            # b50f = os.path.join(config_fourkernels["B50f_encoder"], str(i) + "_net_G_SH_encoder.pth")
            #b30f = os.path.join(config_fourkernels["B30f_encoder"], str(i) + "_net_G_SS_encoder.pth")
            bone = os.path.join(self.config["BONE_encoder"], str(i) + "_net_G_GH_encoder.pth")
            #std = os.path.join(config_fourkernels["STD_encoder"], str(i) + "_net_G_GS_encoder.pth")
            b30f_decoder = os.path.join(self.config["B30f_decoder"], str(i) + "_net_G_SS_decoder.pth")
            #std_decoder = os.path.join(config_fourkernels["STD_decoder"], str(i) + "_net_G_GS_decoder.pth")
            #shss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images", "epoch_" + str(i), config_fourkernels["shss"])
            ghss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images", "epoch_" + str(i), config_fourkernels["ghss"])
            #gsss = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images", "epoch_" + str(i), config_fourkernels["gsss"])
            #ghgs = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images", "epoch_" + str(i), config_fourkernels["ghgs"])
            #generate_images(config_fourkernels, b50f, b30f_decoder, "siemens_hard_100", shss)
            self.generate_images()
            #generate_images(config_fourkernels, std, b30f_decoder, "ge_soft_100", gsss)
            #generate_images(config_fourkernels, bone, std_decoder, "ge_hard_100", ghgs)
            emph_analyze = EmphysemaAnalysis(in_ct_dir=ghss, project_dir=ghss + "_emphysema")
            emph_analyze.generate_lung_mask()
            emph_analyze.get_emphysema_mask()
            emph_analyze.get_emphysema_measurement()
            print(f"Images synthsized for epoch {i}!")


    def run_inference_withheld_data(self,enc, dec):
        print("Synthesizing images for withheld data......")
        self.generate_images()
        # self.generate_images_stage2(enc, dec)
        print("Images synthesized for withheld data!")
        # emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, project_dir=self.inct_dir_synthetic + "_emphysema")
        # emph_analyze.generate_lung_mask()
        # emph_analyze.get_emphysema_mask()
        # emph_analyze.get_emphysema_measurement()
        # print("Emphysema analysis completed for withheld data!")


    def run_inference_validation_data_stage2(self, enc, dec):
        print("Synthesizing images for withheld data......")
        #self.generate_images_stage2(enc, dec)
        print("Images synthesized for withheld data!")
        emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, project_dir=self.inct_dir_synthetic + "_emphysema")
        emph_analyze.generate_lung_mask()
        emph_analyze.get_emphysema_mask()
        emph_analyze.get_emphysema_measurement()
        print("Emphysema analysis completed for withheld data!")

#the paths mentioned here are the paths to the validation dataset. These 100 subjects were used in the Medical Physics journal paper as the witheld dataset.
config_fourkernels = {"siemens_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/hard/ct_masked",
            "siemens_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/soft/ct_masked",
            "ge_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/hard/ct",
            "ge_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/soft/ct",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "B50f_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "B30f_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "BONE_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "STD_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "B30f_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "STD_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/"}


config_fourkernels_withheld_data = {
            "siemens_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/ct_masked",
            "siemens_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/ct_masked",
            "ge_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/ct_masked",
            "ge_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/ct_masked",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "B50f_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "B30f_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "BONE_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "STD_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "B30f_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/",
            "STD_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/", 
            "BONE_decoder": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone/"}

config_fourkernels_stage2 = {"philips_hard":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/C_D/hard/ct_masked",
                             "philips_soft":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/C_D/soft/ct_masked",
                             "ge_lung":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_LUNG/hard/ct",
                             "g_lung_std": "/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_LUNG/soft/ct",
                             "phss": "DtoB30f", "psss":"CtoB30f", "glss":"LUNGtoB30f", "phps":"DtoC", "glgs":"LUNGtoSTD",
                             "D_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
                             "C_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
                             "C_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
                             "B30f_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
                             "LUNG_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
                             "STD_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2"}



config_kernels_stage2_withheld_data = {
    "philips_hard": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/C_D/hard/ct_masked",
    "philips_soft": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/C_D/soft/ct_masked",
    "ge_lung": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_LUNG/hard/ct",
    "ge_lung_std": "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_LUNG/soft/ct",
    "phss": "DtoB30f", "psss":"CtoB30f", "glss":"LUNGtoB30f", "phps":"DtoC", "glgs":"LUNGtoSTD",
    "D_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
    "C_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
    "C_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
    "B30f_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
    "LUNG_encoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2",
    "STD_decoder":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/multipathGAN_withL2weightsched_resentbackbone_stage2"
}

def inference_stage2():
    phild_enc = os.path.join(config_kernels_stage2_withheld_data["D_encoder"], "119_net_gendisc_weights.pth")
    philc_enc = os.path.join(config_kernels_stage2_withheld_data["C_encoder"], "119_net_gendisc_weights.pth")
    lung_enc = os.path.join(config_kernels_stage2_withheld_data["LUNG_encoder"], "119_net_gendisc_weights.pth")
    b30f_dec = os.path.join(config_kernels_stage2_withheld_data["B30f_decoder"], "119_net_gendisc_weights.pth")
    std_dec = os.path.join(config_kernels_stage2_withheld_data["STD_decoder"], "119_net_gendisc_weights.pth")
    philc_dec = os.path.join(config_kernels_stage2_withheld_data["C_decoder"], "119_net_gendisc_weights.pth")

    dtob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images_stage2_epoch119", config_kernels_stage2_withheld_data["phss"])
    #ctob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_kernels_stage2_withheld_data["psss"])
    #lungtob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_kernels_stage2_withheld_data["glss"])
    #lungtostd = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_kernels_stage2_withheld_data["glgs"])
    #dtoc = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_kernels_stage2_withheld_data["phps"])

    inference_dtob30f = GenerateInferenceMultipathGAN(config_kernels_stage2_withheld_data, phild_enc, b30f_dec, inkernel="philips_hard", outkernel=dtob30f, inct_dir_synthetic=dtob30f)
    #inference_ctob30f = GenerateInferenceMultipathGAN(config_kernels_stage2_withheld_data, philc_enc, b30f_dec, inkernel="philips_soft", outkernel=ctob30f, inct_dir_synthetic=ctob30f)
    #inference_lungtob30f = GenerateInferenceMultipathGAN(config_kernels_stage2_withheld_data, lung_enc, b30f_dec, inkernel="ge_lung", outkernel=lungtob30f, inct_dir_synthetic=lungtob30f)
    #inference_lungtostd = GenerateInferenceMultipathGAN(config_kernels_stage2_withheld_data, lung_enc, std_dec, inkernel="ge_lung", outkernel=lungtostd, inct_dir_synthetic=lungtostd)
    #inference_dtoc = GenerateInferenceMultipathGAN(config_kernels_stage2_withheld_data, phild_enc, philc_dec, inkernel="philips_hard", outkernel=dtoc, inct_dir_synthetic=dtoc)

    inference_dtob30f.run_inference_withheld_data(enc="G_D_encoder", dec="G_SS_decoder")
    #inference_ctob30f.run_inference_withheld_data(enc= "G_C_encoder", dec="G_SS_decoder")
    #inference_lungtob30f.run_inference_withheld_data(enc = "G_LUNG_encoder", dec="G_SS_decoder")
    #inference_lungtostd.run_inference_withheld_data(enc="G_LUNG_encoder", dec="G_GS_decoder")
    #inference_dtoc.run_inference_withheld_data(enc="G_D_encoder", dec="G_C_decoder")

# inference_stage2()


def validation_stage2():
    #running for validation data for new kernels
    for i in tqdm(range(180,200)):
        phild_enc = os.path.join(config_fourkernels_stage2["D_encoder"], str(i) + "_net_gendisc_weights.pth")
        philc_enc = os.path.join(config_fourkernels_stage2["C_encoder"], str(i) + "_net_gendisc_weights.pth")
        lung_enc = os.path.join(config_fourkernels_stage2["LUNG_encoder"], str(i) + "_net_gendisc_weights.pth")
        b30f_dec = os.path.join(config_fourkernels_stage2["B30f_decoder"], str(i) + "_net_gendisc_weights.pth")
        std_dec = os.path.join(config_fourkernels_stage2["STD_decoder"], str(i) + "_net_gendisc_weights.pth")
        philc_dec = os.path.join(config_fourkernels_stage2["C_decoder"], str(i) + "_net_gendisc_weights.pth")


        # dtob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images_stage2", "epoch_" + str(i), config_fourkernels_stage2["phss"])
        #ctob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images_stage2", "epoch_" + str(i), config_fourkernels_stage2["psss"])
        #lungtob30f = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images_stage2", "epoch_" + str(i), config_fourkernels_stage2["glss"])
        lungtostd = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images_stage2", "epoch_" + str(i), config_fourkernels_stage2["glgs"])
        #dtoc = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/validation_harmonized_images_stage2", "epoch_" + str(i), config_fourkernels_stage2["phps"])

        # inference_dtob30f = GenerateInferenceMultipathGAN(config_fourkernels_stage2, phild_enc, b30f_dec, inkernel="philips_hard", outkernel=dtob30f, inct_dir_synthetic=dtob30f)
        #inference_ctob30f = GenerateInferenceMultipathGAN(config_fourkernels_stage2, philc_enc, b30f_dec, inkernel="philips_soft", outkernel=ctob30f, inct_dir_synthetic=ctob30f)
        #inference_lungtob30f = GenerateInferenceMultipathGAN(config_fourkernels_stage2, lung_enc, b30f_dec, inkernel="ge_lung", outkernel=lungtob30f, inct_dir_synthetic=lungtob30f)
        inference_lungtostd = GenerateInferenceMultipathGAN(config_fourkernels_stage2, lung_enc, std_dec, inkernel="ge_lung", outkernel=lungtostd, inct_dir_synthetic=lungtostd)
        #inference_dtoc = GenerateInferenceMultipathGAN(config_fourkernels_stage2, phild_enc, philc_dec, inkernel="philips_hard", outkernel=dtoc, inct_dir_synthetic=dtoc)

        # inference_dtob30f.run_inference_validation_data_stage2(enc="G_D_encoder", dec="G_SS_decoder")
        #inference_ctob30f.run_inference_validation_data_stage2(enc="G_C_encoder", dec="G_SS_decoder")
        #inference_lungtob30f.run_inference_validation_data_stage2(enc="G_LUNG_encoder", dec="G_SS_decoder")
        inference_lungtostd.run_inference_validation_data_stage2(enc="G_LUNG_encoder", dec="G_GS_decoder")
        #inference_dtoc.run_inference_validation_data_stage2(enc="G_D_encoder", dec="G_C_decoder")

# validation_stage2()

def inference_stage1():
    #Generate images for the withheld dataset
    b50f_enc = os.path.join(config_fourkernels_withheld_data["B50f_encoder"], "100_net_G_SH_encoder.pth")
    bone_enc = os.path.join(config_fourkernels_withheld_data["BONE_encoder"], "100_net_G_GH_encoder.pth")
    std_enc = os.path.join(config_fourkernels_withheld_data["STD_encoder"], "100_net_G_GS_encoder.pth")
    b30f_dec = os.path.join(config_fourkernels_withheld_data["B30f_decoder"], "100_net_G_SS_decoder.pth")
    std_dec = os.path.join(config_fourkernels_withheld_data["STD_decoder"], "100_net_G_GS_decoder.pth")

    b50ftob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_fourkernels_withheld_data["shss"])
    bonetob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_fourkernels_withheld_data["ghss"])
    stdtob30f_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_fourkernels_withheld_data["gsss"])
    bonetostd_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", config_fourkernels_withheld_data["ghgs"])


    inferenceb50ftob30f = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, b50f_enc, b30f_dec,
                                              inkernel="siemens_hard", outkernel=b50ftob30f_out, inct_dir_synthetic=b50ftob30f_out)
    # inferencebonetob30f = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, bone_enc, b30f_dec,
    #                                             inkernel="ge_hard", outkernel=bonetob30f_out, inct_dir_synthetic=bonetob30f_out)
    # inferencestdtob30f = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, std_enc, b30f_dec,
    #                                              inkernel="ge_soft", outkernel=stdtob30f_out, inct_dir_synthetic=stdtob30f_out)
    # inferencebonetostd = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, bone_enc, std_dec,
    #                                                 inkernel="ge_hard", outkernel=bonetostd_out, inct_dir_synthetic=bonetostd_out)

    inferenceb50ftob30f.run_inference_withheld_data()
    # inferencebonetob30f.run_inference_withheld_data()
    # inferencestdtob30f.run_inference_withheld_data()
    # inferencebonetostd.run_inference_withheld_data()

def inference_method_figure():
    real_std = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/ct/119647.nii.gz"
    std_encoder = os.path.join(config_fourkernels_withheld_data["STD_encoder"], "100_net_G_GS_encoder.pth")
    bone_decoder = os.path.join(config_fourkernels_withheld_data["BONE_decoder"], "100_net_G_GH_decoder.pth")

    stdtobone_out = os.path.join("/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/withheld_test_harmonized_images", "STDtoBONE")
    inferencestdtobone = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, std_encoder, bone_decoder,
                                              inkernel=real_std, outkernel=stdtobone_out, inct_dir_synthetic=stdtobone_out)
    inferencestdtobone.generate_single_image()

inference_method_figure()