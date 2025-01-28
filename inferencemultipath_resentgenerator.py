import torch
import os
from glob import glob
from tqdm import tqdm
from test_custom_dataloader import InferenceDataloader
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from models.networks import ResBlocklatent, ResNetEncoder, ResNetDecoder, G_decoder, G_encoder
from collections import OrderedDict
import torch.nn as nn
# from utils_emphysema import EmphysemaAnalysis


#Use the 100 volumes in /nfs as a validation dataset. Do not reuse this dataset during testing (inference on withheld data)!
#Evaluate various checkpoints on this dataset.

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"    

class GenerateInferenceMultipathGAN:
    def __init__(self, config, input_encoder, output_decoder, inkernel, outkernel, inct_dir_synthetic):
        self.config = config
        self.input_encoder = input_encoder #Must be a path to a checkpoint (.pth)
        self.output_decoder = output_decoder #Must be path to a checkpoint (.pth)
        self.inkernel = inkernel
        self.outkernel = outkernel
        self.inct_dir_synthetic = inct_dir_synthetic #For emphysema analysis

    def generate_images(self, enc, dec):
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
                test_dataloader = DataLoader(dataset=test_dataset, batch_size = 32, shuffle=False, num_workers=4) #returns the pid, normalized data and the slice index
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
    
    def emphysema_analysis(self):
        emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, project_dir=self.inct_dir_synthetic + "_emphysema")
        emph_analyze.generate_lung_mask()
        emph_analyze.get_emphysema_mask()
        emph_analyze.get_emphysema_measurement()

    def generate_single_image(self, enc, dec):
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

config_fourkernels_withheld_data = {
            "siemens_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/hard/ct_masked",
            "siemens_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/ct_masked",
            "ge_hard":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/hard/ct",
            "ge_soft":"/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/STANDARD_BONE/soft/ct",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "B50f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
            "B30f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
            "BONE_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
            "STD_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
            "B30f_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
            "STD_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only", 
            "BONE_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only"}

def inference_anatomyGAN():
    b50f_enc = os.path.join(config_fourkernels_withheld_data["B50f_encoder"], "106_net_gendisc_weights.pth")
    b30f_enc = os.path.join(config_fourkernels_withheld_data["B30f_encoder"], "106_net_gendisc_weights.pth")
    bone_enc = os.path.join(config_fourkernels_withheld_data["BONE_encoder"], "106_net_gendisc_weights.pth")
    std_enc = os.path.join(config_fourkernels_withheld_data["STD_encoder"], "106_net_gendisc_weights.pth")
    b30f_dec = os.path.join(config_fourkernels_withheld_data["B30f_decoder"], "106_net_gendisc_weights.pth")
    std_dec = os.path.join(config_fourkernels_withheld_data["STD_decoder"], "106_net_gendisc_weights.pth")

    b50ftob30f_out = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE", config_fourkernels_withheld_data["shss"])
    bonetob30f_out = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE", config_fourkernels_withheld_data["ghss"])
    stdtob30f_out = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE", config_fourkernels_withheld_data["gsss"])
    bonetostd_out = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE", config_fourkernels_withheld_data["ghgs"])

    # print(b50f_enc, b30f_enc, bone_enc, std_enc, b30f_dec, std_dec, b50ftob30f, bonetob30f, stdtob30f, bonetostd)

    inference_b50ftob30f = GenerateInferenceMultipathGAN(config=config_fourkernels_withheld_data, input_encoder=b50f_enc, output_decoder=b30f_dec, inkernel="siemens_hard", outkernel=b50ftob30f_out, inct_dir_synthetic=b50ftob30f_out)
    inference_bonetob30f = GenerateInferenceMultipathGAN(config=config_fourkernels_withheld_data, input_encoder=bone_enc, output_decoder=b30f_dec, inkernel="ge_hard", outkernel=bonetob30f_out, inct_dir_synthetic=bonetob30f_out)
    inference_stdtob30f = GenerateInferenceMultipathGAN(config=config_fourkernels_withheld_data, input_encoder=std_enc, output_decoder=b30f_dec, inkernel="ge_soft", outkernel=stdtob30f_out, inct_dir_synthetic=stdtob30f_out)
    inference_bonetostd = GenerateInferenceMultipathGAN(config=config_fourkernels_withheld_data, input_encoder=bone_enc, output_decoder=std_dec, inkernel="ge_hard", outkernel=bonetostd_out, inct_dir_synthetic=bonetostd_out)

    # inference_bonetob30f.generate_images(enc="G_GH_encoder", dec="G_SS_decoder")
    # inference_stdtob30f.generate_images(enc="G_GS_encoder", dec="G_SS_decoder")
    # inference_b50ftob30f.generate_images(enc="G_SH_encoder", dec="G_SS_decoder")
    # inference_bonetostd.generate_images(enc="G_GH_encoder", dec="G_GS_decoder")

    inference_bonetob30f.emphysema_analysis()
    # inference_stdtob30f.emphysema_analysis()
    # inference_b50ftob30f.emphysema_analysis()
    # inference_bonetostd.emphysema_analysis()

# inference_anatomyGAN()



def inference_method_figure():
    real_b30f = "/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/journal_inference_additional_data/data.application/B30f_B50f/soft/ct_masked/217022.nii.gz"
    
    b30f_encoder = os.path.join(config_fourkernels_withheld_data["B30f_encoder"], "106_net_gendisc_weights.pth")
    bone_decoder = os.path.join(config_fourkernels_withheld_data["BONE_decoder"], "106_net_gendisc_weights.pth")

    b30ftobone_out = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/INFERENCE", "B30ftoBONE")
    inferencestdtobone = GenerateInferenceMultipathGAN(config_fourkernels_withheld_data, b30f_encoder, bone_decoder,
                                              inkernel=real_b30f, outkernel=b30ftobone_out, inct_dir_synthetic=None)
    inferencestdtobone.generate_single_image(enc="G_SS_encoder", dec="G_GH_decoder")

inference_method_figure()