# AnatomyconstrainedMultipathGAN
Segmentation guided CT kernel harmonization using MultipathcycleGAN and TotalSegmentator. 

This is companion code for the paper "Anatomy-Guided Multi-Path CycleGAN for Lung CT Kernel Harmonization" submitted to MIDL 2025. The code for the multipathGAN heavily relies on the original cycleGAN model [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] 

Paper:[https://openreview.net/pdf?id=w3p7GddsQ8]


## How to prepare data: 
1) Ensure that the images and multilabel masks are CT slices of size 512 x 512 pixels for different kinds of kernels that are available. All the images must be NIfTI (.nii.gz). TotalSegmentator must be run on all the training data and multilabel masks need to be obtained. Here is the link for TotalSegmentator on how to generate labels: [https://github.com/wasserth/TotalSegmentator]

2) All images need to follow the folder structure that the cycleGAN uses.

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


## Citation
@inproceedings{
krishnan2025context,
title={Context based harmonization of low-dose lung cancer computed tomography reconstruction kernels using multipath cycle{GAN}},
author={Aravind Krishnan and Thomas Li and Lucas Walker Remedios and Kaiwen Xu and Lianrui Zuo and Kim L. Sandler and Fabien Maldonado and Bennett Allan Landman},
booktitle={Submitted to Medical Imaging with Deep Learning},
year={2025},
url={https://openreview.net/forum?id=w3p7GddsQ8},
note={under review}
}