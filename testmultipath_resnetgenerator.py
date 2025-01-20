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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    

#the paths mentioned here are the paths to the validation dataset. These 100 subjects were used in the Medical Physics journal paper as the witheld dataset.
# config_fourkernels = {"siemens_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/hard/ct_masked",
#             "siemens_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/soft/ct_masked",
#             "ge_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/hard/ct",
#             "ge_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/soft/ct",
#             "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
#             "B50f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
#             "B30f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
#             "BONE_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
#             "STD_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
#             "B30f_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only",
#             "STD_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_with_context_seg_loss_only"}

config_fourkernels = {"siemens_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/hard/ct_masked",
            "siemens_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/B30f_B50f/soft/ct_masked",
            "ge_hard_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/hard/ct",
            "ge_soft_100":"/nfs/masi/krishar1/Kernel_conversion_outputs/TEST/data.application/STANDARD_BONE/soft/ct",
            "shss": "B50ftoB30f", "ghss":"BONEtoB30f", "gsss":"STDtoB30f", "ghgs":"BONEtoSTD",
            "B50f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data",
            "B30f_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data",
            "BONE_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data",
            "STD_encoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data",
            "B30f_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data",
            "STD_decoder":"/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/MultipathGAN_identity_context_with_subset_data"}



def validation():
    for i in tqdm(range(2, 102, 2)): #config to run inference on model with segmentation loss only. 
        print(f"Synthesizing images for epoch {i}......")
        b50f_enc = os.path.join(config_fourkernels["B50f_encoder"], str(i) + "_net_gendisc_weights.pth")
        bone_enc = os.path.join(config_fourkernels["BONE_encoder"], str(i) + "_net_gendisc_weights.pth")
        b30f_enc = os.path.join(config_fourkernels["B30f_encoder"], str(i) + "_net_gendisc_weights.pth")
        std_enc = os.path.join(config_fourkernels["STD_encoder"], str(i) + "_net_gendisc_weights.pth")
        b30f_dec = os.path.join(config_fourkernels["B30f_decoder"], str(i) + "_net_gendisc_weights.pth")
        std_dec = os.path.join(config_fourkernels["STD_decoder"], str(i) + "_net_gendisc_weights.pth")

        #Run inference on the validation dataset
        b50ftob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_with_context_seg_loss_only", "epoch_" + str(i), config_fourkernels["shss"])
        bonetob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_with_context_seg_loss_only", "epoch_" + str(i), config_fourkernels["ghss"])
        stdtob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_with_context_seg_loss_only", "epoch_" + str(i), config_fourkernels["gsss"])
        bonetostd = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_with_context_seg_loss_only", "epoch_" + str(i), config_fourkernels["ghgs"])

        validate_b50ftob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=b50f_enc, output_decoder=b30f_dec, inkernel="siemens_hard_100", outkernel=b50ftob30f, inct_dir_synthetic=None)
        validate_bonetob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=bone_enc, output_decoder=b30f_dec, inkernel="ge_hard_100", outkernel=bonetob30f, inct_dir_synthetic=None)
        validate_stdtob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=std_enc, output_decoder=b30f_dec, inkernel="ge_soft_100", outkernel=stdtob30f, inct_dir_synthetic=None)
        validate_bonetostd = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=bone_enc, output_decoder=std_dec, inkernel="ge_hard_100", outkernel=bonetostd, inct_dir_synthetic=None)

        validate_bonetob30f.generate_images(enc="G_GH_encoder", dec="G_SS_decoder")
        validate_b50ftob30f.generate_images(enc="G_SH_encoder", dec="G_SS_decoder")
        validate_stdtob30f.generate_images(enc="G_GS_encoder", dec="G_SS_decoder")
        validate_bonetostd.generate_images(enc="G_GH_encoder", dec="G_GS_decoder")

def validation_exptwo():
    for i in tqdm(range(2, 48, 2)): #config to run inference on model with segmentation loss only. 
        print(f"Synthesizing images for epoch {i}......")
        b50f_enc = os.path.join(config_fourkernels["B50f_encoder"], str(i) + "_net_gendisc_weights.pth")
        bone_enc = os.path.join(config_fourkernels["BONE_encoder"], str(i) + "_net_gendisc_weights.pth")
        b30f_enc = os.path.join(config_fourkernels["B30f_encoder"], str(i) + "_net_gendisc_weights.pth")
        std_enc = os.path.join(config_fourkernels["STD_encoder"], str(i) + "_net_gendisc_weights.pth")
        b30f_dec = os.path.join(config_fourkernels["B30f_decoder"], str(i) + "_net_gendisc_weights.pth")
        std_dec = os.path.join(config_fourkernels["STD_decoder"], str(i) + "_net_gendisc_weights.pth")

        #Run inference on the validation dataset
        b50ftob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_identity_context_with_subset_data", "epoch_" + str(i), config_fourkernels["shss"])
        bonetob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_identity_context_with_subset_data", "epoch_" + str(i), config_fourkernels["ghss"])
        stdtob30f = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_identity_context_with_subset_data", "epoch_" + str(i), config_fourkernels["gsss"])
        bonetostd = os.path.join("/valiant02/masi/krishar1/MIDL_experiments/multipathgan_seg_identity_experiments_1-19-25/VALIDATION/MultipathGAN_identity_context_with_subset_data", "epoch_" + str(i), config_fourkernels["ghgs"])

        validate_b50ftob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=b50f_enc, output_decoder=b30f_dec, inkernel="siemens_hard_100", outkernel=b50ftob30f, inct_dir_synthetic=None)
        validate_bonetob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=bone_enc, output_decoder=b30f_dec, inkernel="ge_hard_100", outkernel=bonetob30f, inct_dir_synthetic=None)
        validate_stdtob30f = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=std_enc, output_decoder=b30f_dec, inkernel="ge_soft_100", outkernel=stdtob30f, inct_dir_synthetic=None)
        validate_bonetostd = GenerateInferenceMultipathGAN(config=config_fourkernels, input_encoder=bone_enc, output_decoder=std_dec, inkernel="ge_hard_100", outkernel=bonetostd, inct_dir_synthetic=None)

        validate_bonetob30f.generate_images(enc="G_GH_encoder", dec="G_SS_decoder")
        validate_b50ftob30f.generate_images(enc="G_SH_encoder", dec="G_SS_decoder")
        validate_stdtob30f.generate_images(enc="G_GS_encoder", dec="G_SS_decoder")
        validate_bonetostd.generate_images(enc="G_GH_encoder", dec="G_GS_decoder")

validation_exptwo()