# AnatomyconstrainedMultipathGAN
Segmentation guided CT kernel harmonization using MultipathcycleGAN and TotalSegmentator. 

This is companion code for the paper "Anatomy-Guided Multi-Path CycleGAN for Lung CT Kernel Harmonization" submitted to MIDL 2025. The code for the multipathGAN heavily relies on the original cycleGAN model [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] 

Paper:[https://openreview.net/pdf?id=w3p7GddsQ8]


## How to prepare data: 
1) Ensure that the images and multilabel masks are CT slices of size 512 x 512 pixels for different kinds of kernels that are available. All the images must be NIfTI (.nii.gz). TotalSegmentator must be run on all the training data and multilabel masks need to be obtained. Here is the link for TotalSegmentator on how to generate labels: [https://github.com/wasserth/TotalSegmentator]

2) All images need to be in the following folder structure:
/Training_data
|----->Kernel_A
       |------>pid_000.nii.gz
       |------>pid_001.nii.gz
       :
       :
       :          
|----->Kernel_B
|----->Kernel_C
|----->Kernel_D

## Model training and testing 
### Training script: 004_multipathwithseglossonly.sh
* Training script: trainmultipath_4kernels.py
* --dataroot: Path to data with all individual slices 
* --name: Name of the experiment 
* --model: which multipath cycleGAN model to use unders /models 
* --input_nc, output_nc: Number of input and output channels. Current model handles only single channel 
* --dataset_mode: Which dataloader to use under /data folder 
* --netG_encoder: which encoder to use 
* --netG_decoder: which decoder to use 

### Testing script: inferencemultipath_resnetgenerator.py 

All the checkpoints are saved as xxx_net_gendisc_weights.pth and optimizer weights are stored as xxx_optimizer.pth

