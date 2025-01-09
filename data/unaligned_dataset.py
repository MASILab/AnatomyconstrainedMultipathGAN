import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import numpy as np 
import nibabel as nib 
import torch
from torch.utils.data import DataLoader, Dataset
 
 #Need to confirm if massk has to be float or int. 
class UnalignedDataset(BaseDataset):
    """
    Dataset class for mulitpath kernel conversion. Loads in nine kernels and returns the corresponding images.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        #When training for four domains, use this code from lines 33-51
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_hard')  
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_soft')  
        self.dir_C = os.path.join(opt.dataroot, opt.phase + '_ge_bone_hard')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + '_ge_bone_soft')
        self.mask_dirA = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_hard_slices')
        self.mask_dirB = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_soft_slices')
        self.mask_dirC = os.path.join(opt.dataroot, opt.phase + '_ge_bone_hard_slices')
        self.mask_dirD = os.path.join(opt.dataroot, opt.phase + '_ge_bone_soft_slices')
        print(self.dir_A)
        print(self.dir_B)
        print(self.dir_C)
        print(self.dir_D)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))
        self.mask_A_paths = sorted(make_dataset(self.mask_dirA, opt.max_dataset_size))
        self.mask_B_paths = sorted(make_dataset(self.mask_dirB, opt.max_dataset_size))
        self.mask_C_paths = sorted(make_dataset(self.mask_dirC, opt.max_dataset_size))
        self.mask_D_paths = sorted(make_dataset(self.mask_dirD, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        # input_nc = self.opt.input_nc    
        # output_nc = self.opt.output_nc    
        self.subset_A = int(0.2 * self.A_size) 
        self.subset_B = int(0.2 * self.B_size)
        self.subset_C = int(0.2 * self.C_size)
        self.subset_D = int(0.2 * self.D_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # Get the dataitems for 4 domains
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path = self.mask_A_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_D = index % self.D_size
        else:   # randomize the index
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)
        B_path = self.B_paths[index_B]
        B_mask_path = self.mask_B_paths[index_B]
        C_path = self.C_paths[index_C]
        C_mask_path = self.mask_C_paths[index_C]
        D_path = self.D_paths[index_D]
        D_mask_path = self.mask_D_paths[index_D]

        # print("A index:",index % self.A_size)
        # print("B index:",index_B)
        # print("C index:",index_C)
        # print("D index:",index_D)
        A, A_mask = self.normalize(A_path, A_mask_path)
        B, B_mask = self.normalize(B_path, B_mask_path)
        C, C_mask = self.normalize(C_path, C_mask_path)
        D, D_mask = self.normalize(D_path, D_mask_path) 

        #Return a tuple of the kernel data instead of an indivodual kernel. (Needs to be implemented)
        return {'A': A, 'B': B, 'C': C, 'D': D, 'A_mask': A_mask, 'B_mask': B_mask, 'C_mask': C_mask, 'D_mask': D_mask,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path, } 

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have different datasets with potentially different number of images,
        we take a maximum of all the datasets.
        """
        return max(self.A_size, self.B_size, self.C_size, self.D_size)
        # return max(self.subset_A, self.subset_B, self.subset_C, self.subset_D)


    def normalize(self, input_slice_path, input_mask_path):
        nift_data = nib.load(input_slice_path).get_fdata()[:,:,0]
        torch_tensor = torch.from_numpy(nift_data).unsqueeze(0).float()
        nift_mask = nib.load(input_mask_path).get_fdata()[:,:,0]
        mask = torch.from_numpy(nift_mask)
        mask_tensor = mask.unsqueeze(0).long()
        return torch_tensor, mask_tensor
