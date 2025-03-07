import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm


class VanillaCycleGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_seg', type=float, default=1.0, help='weight for tissue statistic loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'seg_A', 'seg_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.scalar = GradScaler() 

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_A_mask = input['A_mask'].to(self.device)
        self.real_B_mask = input['B_mask'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def tissue_statistic_loss(self, real_A, fake_A, real_B, fake_B, real_mask_A, real_mask_B): 
        ##Might have to weight the mean by the area of the structure
        real_mean_A, fake_mean_A = 0.0, 0.0
        real_mean_B, fake_mean_B = 0.0, 0.0

        unique_labels_forward = torch.unique(real_mask_A)
        unique_labels_forward = unique_labels_forward[unique_labels_forward != 0] #Remove background label
        unique_labels_backward = torch.unique(real_mask_B)
        unique_labels_backward = unique_labels_backward[unique_labels_backward != 0] #Remove background label
        
        for label in unique_labels_forward:
            label_mask = (real_mask_A == label)
            label_count = torch.sum(label_mask)

            if label_count > 0:
                real_mean_A += torch.mean(real_A[label_mask])
                fake_mean_B += torch.mean(fake_B[label_mask])
             

        for label in unique_labels_backward:
            label_mask = (real_mask_B == label)
            label_count = torch.sum(label_mask)

            if label_count > 0:
                real_mean_B += torch.mean(real_B[label_mask])
                fake_mean_A += torch.mean(fake_A[label_mask])
             
        # Compute L2 loss for the forward and backward cycle
        loss_seg_A = self.criterionSeg(real_mean_A, fake_mean_B) * self.opt.lambda_seg
        loss_seg_B = self.criterionSeg(real_mean_B, fake_mean_A) * self.opt.lambda_seg

        return loss_seg_A, loss_seg_B


    def forward(self):
        with autocast():
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
       
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

        self.loss_D = self.loss_D_A + self.loss_D_B
        return self.loss_D

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_seg = self.opt.lambda_seg
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        #4 computations for the tissue statistic loss: mean of real A , MEan of Fake A, mean of real B, mean of fake B
        # L2 forward = ||mean(real_A) - mean(fake_A)||^2, L2 backward = ||mean(real_B) - mean(fake_B)||^2
        self.loss_seg_A, self.loss_seg_B = self.tissue_statistic_loss(self.real_A, self.fake_A, self.real_B, self.fake_B, self.real_A_mask, self.real_B_mask)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B +  self.loss_seg_A + self.loss_seg_B
        return self.loss_G

    def optimize_parameters(self):
     
        self.forward()      
     
        self.set_requires_grad([self.netD_A, self.netD_B], False) 
        self.optimizer_G.zero_grad() 
        with autocast():
            loss_G = self.backward_G()             
        # self.optimizer_G.step()      
        self.scalar.scale(loss_G).backward()
        self.scalar.step(self.optimizer_G)

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  
        with autocast():
            loss_D = self.backward_D()
        # self.optimizer_D.step()
        self.scalar.scale(loss_D).backward()
        self.scalar.step(self.optimizer_D)
        self.scalar.update()