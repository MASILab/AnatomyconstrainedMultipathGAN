#!/bin/bash

python trainmultipath_4kernels.py --dataroot /local_storage/krishar1/AnatomyconstrainedMultipathGAN/training_data --name MultipathGAN_with_context_seg_loss_only --model resnetmultipathwithoutidentitycycle_gan --input_nc 1 --output_nc 1 --dataset_mode unaligned --batch_size 4 --load_size 512 --crop_size 512 --n_epochs 100 --n_epochs_decay 100 --gpu_ids 0,1 --display_id 0 --no_flip --norm instance --netG_encoder resnet_encoder --netG_decoder resnet_decoder --netD basic --num_threads 40
