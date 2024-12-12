import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.networks import ResBlocklatent
import torch.nn.functional as F 
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class ResnetMultipathNewKernelsCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)') #Forward lambda
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)') #Backward lambda
            # parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_L2', type = float, default=10e6, help='Weight for L2 function between generated outout and input for a given path. Will begin decaying once the images are forced to identity.') 

        return parser

    def __init__(self, opt):
        """Initialize the MultipathCycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
          # A = B50f, B = B30f, C = GE BONE, D = GE STD, E= Philips D, F = Philips C, G = GE LUNG, H = GE LUNGSTD
        #Printing losses
        self.loss_names = ['D_AE', 'G_AE', 'L2B50fD', 'cycle_AE', 'D_EA', 'G_EA', 'L2DB50f', 'cycle_EA', #B50f to Philips D
                           'D_AF', 'G_AF', 'L2B50fC', 'cycle_AF', 'D_FA', 'G_FA', 'L2CB50f', 'cycle_FA', #B50f to Philips C
                           'D_AG', 'G_AG', 'L2B50fLUNG', 'cycle_AG', 'D_GA', 'G_GA', 'L2LUNGB50f', 'cycle_GA', #B50f to LUNG
                           'D_BE', 'G_BE', 'L2B30fD', 'cycle_BE', 'D_EB', 'G_EB', 'L2DB30f', 'cycle_EB', #B30f to Philips D
                           'D_BF', 'G_BF', 'L2B30fC', 'cycle_BF', 'D_FB', 'G_FB', 'L2CB30f', 'cycle_FB', #B30f to Philips C
                           'D_BG', 'G_BG', 'L2B30fLUNG', 'cycle_BG', 'D_GB', 'G_GB', 'L2LUNGB30f', 'cycle_GB', #B30f to LUNG
                           'D_CE', 'G_CE', 'L2BONED', 'cycle_CE', 'D_EC', 'G_EC', 'L2DBONE', 'cycle_EC', #GE BONE to Philips D
                           'D_CF', 'G_CF', 'L2BONEC', 'cycle_CF', 'D_FC', 'G_FC', 'L2CBONE', 'cycle_FC', #GE BONE to Philips C
                           'D_CG', 'G_CG', 'L2BONELUNG', 'cycle_CG', 'D_GC', 'G_GC', 'L2LUNGBONE', 'cycle_GC', #GE BONE to LUNG
                           'D_DE', 'G_DE', 'L2STDD', 'cycle_DE', 'D_ED', 'G_ED', 'L2DSTD', 'cycle_ED', #GE STD to Philips D
                           'D_DF', 'G_DF', 'L2STDC', 'cycle_DF', 'D_FD', 'G_FD', 'L2CSTD', 'cycle_FD', #GE STD to Philips C
                           'D_EF', 'G_EF', 'L2DC', 'cycle_EF', 'D_FE', 'G_FE', 'L2CD', 'cycle_FE', #Philips D to C
                           'D_EG', 'G_EG', 'L2DLUNG', 'cycle_EG', 'D_GE', 'G_GE', 'L2LUNGD', 'cycle_GE', #Philips D to LUNG
                           'D_FG', 'G_FG', 'L2CLUNG', 'cycle_FG', 'D_GF', 'G_GF', 'L2LUNGC', 'cycle_GF', #Philips C to LUNG
                           'D_GD', 'G_GD', 'L2LUNGSTD', 'cycle_GD', 'D_GD', 'G_GD', 'L2STDLUNG', 'cycle_DG'] #LUNG to LUNGSTD 
 
        visual_names_AE = ['B50f', 'fake_EA', 'rec_AE'] #B50f to Philips D
        visual_names_EA = ['PHILD', 'fake_AE', 'rec_EA'] # Philips D to B50f
        visual_names_AF = ['B50f', 'fake_FA', 'rec_AF'] #B50f to Philips C
        visual_names_FA = ['PHILC', 'fake_AF', 'rec_FA'] # Philips C to B50f
        visual_names_AG = ['B50f', 'fake_GA', 'rec_AG'] #B50f to LUNG
        visual_names_GA = ['LUNG', 'fake_AG', 'rec_GA'] # LUNG to B50f
        visual_names_BE = ['B30f', 'fake_EB', 'rec_BE'] #B30f to Philips D
        visual_names_EB = ['PHILD', 'fake_BE', 'rec_EB'] # Philips D to B30f
        visual_names_BF = ['B30f', 'fake_FB', 'rec_BF'] #B30f to Philips C
        visual_names_FB = ['PHILC', 'fake_BF', 'rec_FB'] # Philips C to B30f
        visual_names_BG = ['B30f', 'fake_GB', 'rec_BG'] #B30f to LUNG
        visual_names_GB = ['LUNG', 'fake_BG', 'rec_GB'] # LUNG to B30f
        visual_names_CE = ['BONE', 'fake_EC', 'rec_CE'] #GE BONE to Philips D
        visual_names_EC = ['PHILD', 'fake_CE', 'rec_EC'] # Philips D to GE BONE
        visual_names_CF = ['BONE', 'fake_FC', 'rec_CF'] #GE BONE to Philips C
        visual_names_FC = ['PHILC', 'fake_CF', 'rec_FC'] # Philips C to GE BONE
        visual_names_CG = ['BONE', 'fake_GC', 'rec_CG'] #GE BONE to LUNG
        visual_names_GC = ['LUNG', 'fake_CG', 'rec_GC'] # LUNG to GE BONE
        visual_names_DE = ['STD', 'fake_ED', 'rec_DE'] #GE STD to Philips D
        visual_names_ED = ['PHILD', 'fake_DE', 'rec_ED'] # Philips D to GE STD
        visual_names_DF = ['STD', 'fake_FD', 'rec_DF'] #GE STD to Philips C
        visual_names_FD = ['PHILC', 'fake_DF', 'rec_FD'] # Philips C to GE STD
        visual_names_EF = ['PHILD', 'fake_FE', 'rec_EF'] #Philips D to Philips C
        visual_names_FE = ['PHILC', 'fake_EF', 'rec_FE'] # Philips C to Philips D
        visual_names_EG = ['PHILD', 'fake_GE', 'rec_EG'] #Philips D to LUNG
        visual_names_GE = ['LUNG', 'fake_EG', 'rec_GE'] # LUNG to Philips D
        visual_names_FG = ['PHILC', 'fake_GF', 'rec_FG'] #Philips C to LUNG
        visual_names_GF = ['LUNG', 'fake_FG', 'rec_GF'] # LUNG to Philips C
        visual_names_GD = ['LUNG', 'fake_DG', 'rec_GD'] # LUNG to LUNGSTD
        visual_names_DG = ['LUNGSTD', 'fake_GD', 'rec_DG'] # LUNGSTD to LUNG


        #Multipath GAN 
        # self.visual_names = list(itertools.chain(*filtered_nested_list))
        self.visual_names = visual_names_AE + visual_names_EA + visual_names_AF + visual_names_FA + visual_names_AG + visual_names_GA + visual_names_BE + visual_names_EB + \
                            visual_names_BF + visual_names_FB + visual_names_BG + visual_names_GB + visual_names_CE + visual_names_EC + visual_names_CF + visual_names_FC + \
                            visual_names_CG + visual_names_GC + visual_names_DE + visual_names_ED + visual_names_DF + visual_names_FD + \
                            visual_names_EF + visual_names_FE + visual_names_EG + visual_names_GE + \
                            visual_names_FG + visual_names_GF + visual_names_GD + visual_names_DG


        if self.isTrain:
            #Multipath cycleGAN
            self.model_names = ['G_SH_encoder', 'G_SH_decoder','G_SS_decoder', 'G_SS_encoder', 'G_GS_encoder', 'G_GS_decoder', 'G_GH_encoder', 'G_GH_decoder',
                                'G_LUNG_encoder', 'G_LUNG_decoder','G_C_encoder', 'G_C_decoder', 'G_D_encoder', 'G_D_decoder',
                                'D_A','D_B', 'D_C','D_D', 'D_E', 'D_F', 'D_G']


        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect').to(self.device)
        #Encoder initalizations
        self.netG_SH_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #Siemens hard encoder
        self.netG_SS_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #Siemens soft encoder
        self.netG_GH_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #Siemens hard encoder
        self.netG_GS_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #GE soft encoder
        self.netG_C_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #C encoder
        self.netG_D_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #D encoder
        self.netG_LUNG_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #GE LUNG encoder

        #Decoder initalizations
        self.netG_SH_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #Siemens hard decoder
        self.netG_SS_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #Siemens soft decoder
        self.netG_GH_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #GE hard decoder
        self.netG_GS_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #GE soft decoder 
        self.netG_C_decoder = networks.G_decoder(opt.input_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #C decoder
        self.netG_D_decoder = networks.G_decoder(opt.input_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #D decoder
        self.netG_LUNG_decoder = networks.G_decoder(opt.input_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #GE LUNG decoder

        
        if self.isTrain:  #Discriminator initalizations
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Siemens hard
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens soft
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #GE BONE
            self.netD_D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE soft (STANDARD)
            self.netD_E = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Philips C
            self.netD_F = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Philips D 
            self.netD_G = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #LUNG


        if self.isTrain:
            self.fake_pools = {}
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            excluded_combinations = ['AB', 'BA', 'AC', 'CA', 'AD', 'DA', 'BC', 'CB', 'BD', 'DB', 'CD', 'DC', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG','HH', 
                                     'AH', 'HA', 'BH', 'HB', 'CH', 'HC', 'DH', 'HD', 'EH', 'HE', 'FH', 'HF', 'HG', 'GH'] 

            for combination in itertools.product(letters, repeat=2):
                key = ''.join(combination)
                if key not in excluded_combinations:
                    self.fake_pools['fake_' + key + '_pool'] = ImagePool(opt.pool_size)
            
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")
            print(f"Created image buffers for all fake images. Length of dictionary with image buffers is {len(self.fake_pools)}")
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")

             
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss() #cycle loss
            self.L2 = torch.nn.MSELoss() # additional L2 loss between generated output and input for a given path

            self.set_requires_grad([self.netG_SH_encoder, self.netG_SS_encoder, self.netG_GH_encoder, self.netG_GS_encoder, 
                                    self.netG_SH_decoder, self.netG_SS_decoder, self.netG_GH_decoder, self.netG_GS_decoder], False) 
            
            self.optimizer_G = torch.optim.Adam(itertools.chain( 
                                                                self.netG_C_encoder.parameters(),
                                                                self.netG_C_decoder.parameters(),
                                                                self.netG_D_encoder.parameters(),
                                                                self.netG_D_decoder.parameters(),
                                                                self.netG_LUNG_encoder.parameters(),
                                                                self.netG_LUNG_decoder.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                                self.netD_B.parameters(),
                                                                self.netD_C.parameters(),
                                                                self.netD_D.parameters(), 
                                                                self.netD_E.parameters(), 
                                                                self.netD_F.parameters(), 
                                                                self.netD_G.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.scalar = GradScaler() #Mixed precision training

        #Set gradients for trained encoders as False.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # Multipath GAN
        self.B50f = input['A'].to(self.device)
        self.B30f = input['B'].to(self.device) 
        self.BONE = input['C'].to(self.device)
        self.STD = input['D'].to(self.device)
        self.PHILD = input['E'].to(self.device)
        self.PHILC = input['F'].to(self.device)
        self.LUNG = input['G'].to(self.device)
        self.LUNGSTD = input['H'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths'] 

    def L2loss_decay(self, real_image, fake_image):
        """
        Compute a loss on smoothened input and synthetic image for a given path
        """
        interp_real = F.interpolate(real_image, size = [256,256], mode = 'bilinear', align_corners=True)
        interp_fake = F.interpolate(fake_image, size = [256,256], mode = 'bilinear', align_corners=True)
        L2loss = self.L2(interp_real, interp_fake)
        return L2loss

    def cyclicpath(self, target_decoder, target_encoder, source_decoder, latent):
        """
        Code snippet for a given cyclic path in the multipath GAN model
        """
        fake_image = target_decoder(latent)
        latent_rec = target_encoder(fake_image)
        reconstructed = source_decoder(latent_rec)
        return fake_image, reconstructed

    def forward(self): 
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        latentA = self.netG_SH_encoder(self.B50f) 
        latentB = self.netG_SS_encoder(self.B30f) 
        latentC = self.netG_GH_encoder(self.BONE)
        latentD = self.netG_GS_encoder(self.STD)
        latentE = self.netG_D_encoder(self.PHILD)
        latentF = self.netG_C_encoder(self.PHILC)
        latentG = self.netG_LUNG_encoder(self.LUNG)
        latentH = self.netG_GS_encoder(self.LUNGSTD)

        #Siemens B50f paths
        #Gen 1: B50f to Philips D
        self.fake_EA, self.rec_AE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_SH_decoder, latentA)
        #Gen 2: Philips D to B50f
        self.fake_AE, self.rec_EA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_D_decoder, latentE)
        #Gen 3: B50f to Philips C 
        self.fake_FA, self.rec_AF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_SH_decoder, latentA)
        #Gen 4: Philips C to B50f
        self.fake_AF, self.rec_FA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_C_decoder, latentF)
        #Gen 5: B50f to LUNG
        self.fake_GA, self.rec_AG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_SH_decoder, latentA)
        #Gen 6: LUNG to B50f
        self.fake_AG, self.rec_GA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_LUNG_decoder, latentG)

        #Siemens B30f paths
        #Gen 7: B30f to Philips D
        self.fake_EB, self.rec_BE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_SS_decoder, latentB)
        #Gen 8: Philips D to B30f
        self.fake_BE, self.rec_EB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_D_decoder, latentE)
        #Gen 9: B30f to Philips C 
        self.fake_FB, self.rec_BF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_SS_decoder, latentB)
        #Gen 10: Philips C to B30f
        self.fake_BF, self.rec_FB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_C_decoder, latentF)
        #Gen 11: B30f to LUNG 
        self.fake_GB, self.rec_BG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_SS_decoder, latentB)
        #Gen 12: LUNG to B30f
        self.fake_BG, self.rec_GB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_LUNG_decoder, latentG)

        #GE BONE paths
        #Gen 13: BONE to Philips D
        self.fake_EC, self.rec_CE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_GH_decoder, latentC)
        #Gen 14: Philips D to BONE
        self.fake_CE, self.rec_EC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_D_decoder, latentE)
        #Gen 15: BONE to Philips C 
        self.fake_FC, self.rec_CF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_GH_decoder, latentC)
        #Gen 16: Philips C to BONE
        self.fake_CF, self.rec_FC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_C_decoder, latentF)
        #Gen 17: BONE to LUNG
        self.fake_GC, self.rec_CG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_GH_decoder, latentC)
        #Gen 18: LUNG to BONE
        self.fake_CG, self.rec_GC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_LUNG_decoder, latentG)

        #GE STD paths
        #Gen 19: STD to Philips D
        self.fake_ED, self.rec_DE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_GS_decoder, latentD)
        #Gen 20: Philips D to STD
        self.fake_DE, self.rec_ED = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_D_decoder, latentE)
        #Gen 21: STD to Philips C 
        self.fake_FD, self.rec_DF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_GS_decoder, latentD)
        #Gen 22: Philips C to STD
        self.fake_DF, self.rec_FD = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_C_decoder, latentF)
        #Gen 23: STD to LUNG 
        self.fake_GD, self.rec_DG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_GS_decoder, latentH)
        #Gen 24: LUNG to STD
        self.fake_DG, self.rec_GD = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_LUNG_decoder, latentG)

        #Philips D paths
        #Gen 25: Philips D to Philips C 
        self.fake_FE, self.rec_EF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_D_decoder, latentE)
        #Gen 26: Philips C to Philips D
        self.fake_EF, self.rec_FE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_C_decoder, latentF) #Forward pass fails here when batch size = 1/2 for the model trained on the A6000.
        #Gen 27: Philips D to LUNG 
        self.fake_GE, self.rec_EG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_D_decoder, latentE)
        #Gen 28: LUNG to Philips D
        self.fake_EG, self.rec_GE = self.cyclicpath(self.netG_D_decoder, self.netG_D_encoder, self.netG_LUNG_decoder, latentG)
  
        #Philips C paths
        #Gen 29: Philips C to LUNG 
        self.fake_GF, self.rec_FG = self.cyclicpath(self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_C_decoder, latentF)
        #Gen 30: LUNG to Philips C
        self.fake_FG, self.rec_GF = self.cyclicpath(self.netG_C_decoder, self.netG_C_encoder, self.netG_LUNG_decoder, latentG)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D


    # Mulitpath cycleGAN 
    # A = B50f, B = B30f, C = GE BONE, D = GE STD, E= Philips D, F = Philips C, G = LUNG, H = LUNG STD
    def backward_D(self):
        """Calculate discriminator loss for all combinations of all discriminators for new paths"""
        #D for all combinations with A
        #D1: backward DAE
        fake_EA = self.fake_pools["fake_EA_pool"].query(self.fake_EA)
        self.loss_D_AE = self.backward_D_basic(self.netD_A, self.PHILD, fake_EA)
        #D2:BAckward DEA
        fake_AE = self.fake_pools["fake_AE_pool"].query(self.fake_AE)
        self.loss_D_EA = self.backward_D_basic(self.netD_E, self.B50f, fake_AE)
        #D3 :Backward DAF
        fake_FA = self.fake_pools["fake_FA_pool"].query(self.fake_FA)
        self.loss_D_AF = self.backward_D_basic(self.netD_A, self.PHILC, fake_FA) 
        #D4:Backward DFA
        fake_AF = self.fake_pools["fake_AF_pool"].query(self.fake_AF)
        self.loss_D_FA = self.backward_D_basic(self.netD_F, self.B50f, fake_AF)
        #D5: Backward DAG
        fake_GA = self.fake_pools["fake_GA_pool"].query(self.fake_GA)
        self.loss_D_AG = self.backward_D_basic(self.netD_A, self.LUNG, fake_GA) 
        #D6:BAckward DGA
        fake_AG = self.fake_pools["fake_AG_pool"].query(self.fake_AG)
        self.loss_D_GA = self.backward_D_basic(self.netD_G, self.B50f, fake_AG)
        
        #D for all combinations with B
        #D7
        fake_EB = self.fake_pools["fake_EB_pool"].query(self.fake_EB)
        self.loss_D_BE = self.backward_D_basic(self.netD_B, self.PHILD, fake_EB)
        #D8
        fake_BE = self.fake_pools["fake_BE_pool"].query(self.fake_BE)
        self.loss_D_EB = self.backward_D_basic(self.netD_E, self.B30f, fake_BE)
        #D9
        fake_FB = self.fake_pools["fake_FB_pool"].query(self.fake_FB)
        self.loss_D_BF = self.backward_D_basic(self.netD_B, self.PHILC, fake_FB)
        #D10
        fake_BF = self.fake_pools["fake_BF_pool"].query(self.fake_BF)
        self.loss_D_FB = self.backward_D_basic(self.netD_F, self.B30f, fake_BF)
        #D11
        fake_GB = self.fake_pools["fake_GB_pool"].query(self.fake_GB)
        self.loss_D_BG = self.backward_D_basic(self.netD_B, self.LUNG, fake_GB)
        #D12
        fake_BG = self.fake_pools["fake_BG_pool"].query(self.fake_BG)
        self.loss_D_GB = self.backward_D_basic(self.netD_G, self.B30f, fake_BG)

        #D for all combinations with C
        #D13
        fake_EC = self.fake_pools["fake_EC_pool"].query(self.fake_EC)
        self.loss_D_CE = self.backward_D_basic(self.netD_C, self.PHILD, fake_EC)
        #D14
        fake_CE = self.fake_pools["fake_CE_pool"].query(self.fake_CE)
        self.loss_D_EC = self.backward_D_basic(self.netD_E, self.BONE, fake_CE)
        #D15
        fake_FC = self.fake_pools["fake_FC_pool"].query(self.fake_FC)
        self.loss_D_CF = self.backward_D_basic(self.netD_C, self.PHILC, fake_FC)
        #D16
        fake_CF = self.fake_pools["fake_CF_pool"].query(self.fake_CF)
        self.loss_D_FC = self.backward_D_basic(self.netD_F, self.BONE, fake_CF)
        #D17
        fake_GC = self.fake_pools["fake_GC_pool"].query(self.fake_GC)
        self.loss_D_CG = self.backward_D_basic(self.netD_C, self.LUNG, fake_GC)
        #D18
        fake_CG = self.fake_pools["fake_CG_pool"].query(self.fake_CG)
        self.loss_D_GC = self.backward_D_basic(self.netD_G, self.BONE, fake_CG)

        #Ds for all combinations with D
        #D19
        fake_ED = self.fake_pools["fake_ED_pool"].query(self.fake_ED)
        self.loss_D_DE = self.backward_D_basic(self.netD_D, self.PHILD, fake_ED)
        #D20
        fake_DE = self.fake_pools["fake_DE_pool"].query(self.fake_DE)
        self.loss_D_ED = self.backward_D_basic(self.netD_E, self.STD, fake_DE)
        #D21
        fake_FD = self.fake_pools["fake_FD_pool"].query(self.fake_FD)
        self.loss_D_DF = self.backward_D_basic(self.netD_D, self.PHILC, fake_FD)
        #D22
        fake_DF = self.fake_pools["fake_DF_pool"].query(self.fake_DF)
        self.loss_D_FD = self.backward_D_basic(self.netD_F, self.STD, fake_DF) 
        #D23
        fake_GD = self.fake_pools["fake_GD_pool"].query(self.fake_GD)
        self.loss_D_DG = self.backward_D_basic(self.netD_D, self.LUNG, fake_GD)
        #D24
        fake_DG = self.fake_pools["fake_DG_pool"].query(self.fake_DG)
        self.loss_D_GD = self.backward_D_basic(self.netD_G, self.LUNGSTD, fake_DG)

        #D for all combinations with E 
        #D25
        fake_FE = self.fake_pools["fake_FE_pool"].query(self.fake_FE)
        self.loss_D_EF = self.backward_D_basic(self.netD_E, self.PHILC, fake_FE)
        #D26
        fake_EF = self.fake_pools["fake_EF_pool"].query(self.fake_EF)
        self.loss_D_FE = self.backward_D_basic(self.netD_F, self.PHILD, fake_EF)
        #D27
        fake_GE = self.fake_pools["fake_GE_pool"].query(self.fake_GE)
        self.loss_D_EG = self.backward_D_basic(self.netD_E, self.LUNG, fake_GE)
        #D28
        fake_EG = self.fake_pools["fake_EG_pool"].query(self.fake_EG)
        self.loss_D_GE = self.backward_D_basic(self.netD_G, self.PHILD, fake_EG)

        #D for all combinations with F
        #D29
        fake_GF = self.fake_pools["fake_GF_pool"].query(self.fake_GF)
        self.loss_D_FG = self.backward_D_basic(self.netD_F, self.LUNG, fake_GF)
        #D30
        fake_FG = self.fake_pools["fake_FG_pool"].query(self.fake_FG)
        self.loss_D_GF = self.backward_D_basic(self.netD_G, self.PHILC, fake_FG)

        self.loss_D = self.loss_D_AE + self.loss_D_EA + self.loss_D_AF + self.loss_D_FA + self.loss_D_AG + self.loss_D_GA + \
                      self.loss_D_BE + self.loss_D_EB + self.loss_D_BF + self.loss_D_FB + self.loss_D_BG + self.loss_D_GB + \
                      self.loss_D_CE + self.loss_D_EC + self.loss_D_CF + self.loss_D_FC + self.loss_D_CG + self.loss_D_GC + \
                      self.loss_D_DE + self.loss_D_ED + self.loss_D_DF + self.loss_D_FD + self.loss_D_DG + self.loss_D_GD + \
                      self.loss_D_EF + self.loss_D_FE + self.loss_D_EG + self.loss_D_GE + \
                      self.loss_D_FG + self.loss_D_GF

        return self.loss_D


    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_L2 = self.opt.lambda_L2
        # GAN loss D_A(G_A(A))
        # GAN loss D_B(G_B(B))

        # Least squares loss for all the generators
        #Domain A
        self.loss_G_AE = self.criterionGAN(self.netD_A(self.fake_EA), True)
        self.loss_G_EA = self.criterionGAN(self.netD_E(self.fake_AE), True)
        self.loss_G_AF = self.criterionGAN(self.netD_A(self.fake_FA), True)
        self.loss_G_FA = self.criterionGAN(self.netD_F(self.fake_AF), True)
        self.loss_G_AG = self.criterionGAN(self.netD_A(self.fake_GA), True)
        self.loss_G_GA = self.criterionGAN(self.netD_G(self.fake_AG), True)

        #Domain B
        self.loss_G_BE = self.criterionGAN(self.netD_B(self.fake_EB), True)
        self.loss_G_EB = self.criterionGAN(self.netD_E(self.fake_BE), True)
        self.loss_G_BF = self.criterionGAN(self.netD_B(self.fake_FB), True)
        self.loss_G_FB = self.criterionGAN(self.netD_F(self.fake_BF), True)
        self.loss_G_BG = self.criterionGAN(self.netD_B(self.fake_GB), True)
        self.loss_G_GB = self.criterionGAN(self.netD_G(self.fake_BG), True)

        #Domain C
        self.loss_G_CE = self.criterionGAN(self.netD_C(self.fake_EC), True)
        self.loss_G_EC = self.criterionGAN(self.netD_E(self.fake_CE), True)
        self.loss_G_CF = self.criterionGAN(self.netD_C(self.fake_FC), True)
        self.loss_G_FC = self.criterionGAN(self.netD_F(self.fake_CF), True)
        self.loss_G_CG = self.criterionGAN(self.netD_C(self.fake_GC), True)
        self.loss_G_GC = self.criterionGAN(self.netD_G(self.fake_CG), True)

        #Domain D
        self.loss_G_DE = self.criterionGAN(self.netD_D(self.fake_ED), True)
        self.loss_G_ED = self.criterionGAN(self.netD_E(self.fake_DE), True)
        self.loss_G_DF = self.criterionGAN(self.netD_D(self.fake_FD), True)
        self.loss_G_FD = self.criterionGAN(self.netD_F(self.fake_DF), True)
        self.loss_G_DG = self.criterionGAN(self.netD_D(self.fake_GD), True)
        self.loss_G_GD = self.criterionGAN(self.netD_G(self.fake_DG), True)

        #Domain E
        self.loss_G_EF = self.criterionGAN(self.netD_E(self.fake_FE), True)
        self.loss_G_FE = self.criterionGAN(self.netD_F(self.fake_EF), True)
        self.loss_G_EG = self.criterionGAN(self.netD_E(self.fake_GE), True)
        self.loss_G_GE = self.criterionGAN(self.netD_G(self.fake_EG), True)

        #Domain F
        self.loss_G_FG = self.criterionGAN(self.netD_F(self.fake_GF), True)
        self.loss_G_GF = self.criterionGAN(self.netD_G(self.fake_FG), True)


        # Forward cycle loss || G_B(G_A(A)) - A||
        # Backward cycle loss || G_A(G_B(B)) - B||

        #Multipath cycleGAN: Cycle consistency losses
        self.loss_cycle_AE = self.criterionCycle(self.rec_AE, self.B50f) * lambda_A
        self.loss_cycle_EA = self.criterionCycle(self.rec_EA, self.PHILD) * lambda_B
        self.loss_cycle_AF = self.criterionCycle(self.rec_AF, self.B50f) * lambda_A
        self.loss_cycle_FA = self.criterionCycle(self.rec_FA, self.PHILC) * lambda_B
        self.loss_cycle_AG = self.criterionCycle(self.rec_AG, self.B50f) * lambda_A
        self.loss_cycle_GA = self.criterionCycle(self.rec_GA, self.LUNG) * lambda_B

        self.loss_cycle_BE = self.criterionCycle(self.rec_BE, self.B30f) * lambda_A
        self.loss_cycle_EB = self.criterionCycle(self.rec_EB, self.PHILD) * lambda_B
        self.loss_cycle_BF = self.criterionCycle(self.rec_BF, self.B30f) * lambda_A
        self.loss_cycle_FB = self.criterionCycle(self.rec_FB, self.PHILC) * lambda_B
        self.loss_cycle_BG = self.criterionCycle(self.rec_BG, self.B30f) * lambda_A
        self.loss_cycle_GB = self.criterionCycle(self.rec_GB, self.LUNG) * lambda_B
    
        self.loss_cycle_CE = self.criterionCycle(self.rec_CE, self.BONE) * lambda_A
        self.loss_cycle_EC = self.criterionCycle(self.rec_EC, self.PHILD) * lambda_B
        self.loss_cycle_CF = self.criterionCycle(self.rec_CF, self.BONE) * lambda_A
        self.loss_cycle_FC = self.criterionCycle(self.rec_FC, self.PHILC) * lambda_B
        self.loss_cycle_CG = self.criterionCycle(self.rec_CG, self.BONE) * lambda_A
        self.loss_cycle_GC = self.criterionCycle(self.rec_GC, self.LUNG) * lambda_B

        self.loss_cycle_DE = self.criterionCycle(self.rec_DE, self.STD) * lambda_A
        self.loss_cycle_ED = self.criterionCycle(self.rec_ED, self.PHILD) * lambda_B
        self.loss_cycle_DF = self.criterionCycle(self.rec_DF, self.STD) * lambda_A
        self.loss_cycle_FD = self.criterionCycle(self.rec_FD, self.PHILC) * lambda_B
        self.loss_cycle_DG = self.criterionCycle(self.rec_DG, self.LUNGSTD) * lambda_A
        self.loss_cycle_GD = self.criterionCycle(self.rec_GD, self.LUNG) * lambda_B

        self.loss_cycle_FE = self.criterionCycle(self.rec_FE, self.PHILC) * lambda_A
        self.loss_cycle_EF = self.criterionCycle(self.rec_EF, self.PHILD) * lambda_B
        self.loss_cycle_GE = self.criterionCycle(self.rec_GE, self.LUNG) * lambda_A
        self.loss_cycle_EG = self.criterionCycle(self.rec_EG, self.PHILD) * lambda_B
       
        self.loss_cycle_FG = self.criterionCycle(self.rec_FG, self.PHILC) * lambda_A
        self.loss_cycle_GF = self.criterionCycle(self.rec_GF, self.LUNG) * lambda_B

        #Additional L2 loss for the objective function: Downsample real and fake tensors, compute MSE between them
        #Domain A
        self.loss_L2B50fD = self.L2loss_decay(self.B50f, self.fake_EA) * lambda_L2
        self.loss_L2DB50f = self.L2loss_decay(self.PHILD, self.fake_AE) * lambda_L2
        self.loss_L2B50fC = self.L2loss_decay(self.B50f, self.fake_FA) * lambda_L2
        self.loss_L2CB50f = self.L2loss_decay(self.PHILC, self.fake_AF) * lambda_L2
        self.loss_L2B50fLUNG = self.L2loss_decay(self.B50f, self.fake_GA) * lambda_L2
        self.loss_L2LUNGB50f = self.L2loss_decay(self.LUNG, self.fake_AG) * lambda_L2

        #Domain B
        self.loss_L2B30fD = self.L2loss_decay(self.B30f, self.fake_EB) * lambda_L2
        self.loss_L2DB30f = self.L2loss_decay(self.PHILD, self.fake_BE) * lambda_L2
        self.loss_L2B30fC = self.L2loss_decay(self.B30f, self.fake_FB) * lambda_L2
        self.loss_L2CB30f = self.L2loss_decay(self.PHILC, self.fake_BF) * lambda_L2
        self.loss_L2B30fLUNG = self.L2loss_decay(self.B30f, self.fake_GB) * lambda_L2
        self.loss_L2LUNGB30f = self.L2loss_decay(self.LUNG, self.fake_BG) * lambda_L2

        #Domain C
        self.loss_L2BONED = self.L2loss_decay(self.BONE, self.fake_EC) * lambda_L2
        self.loss_L2DBONE = self.L2loss_decay(self.PHILD, self.fake_CE) * lambda_L2
        self.loss_L2BONEC = self.L2loss_decay(self.BONE, self.fake_FC) * lambda_L2
        self.loss_L2CBONE = self.L2loss_decay(self.PHILC, self.fake_CF) * lambda_L2
        self.loss_L2BONELUNG = self.L2loss_decay(self.BONE, self.fake_GC) * lambda_L2
        self.loss_L2LUNGBONE = self.L2loss_decay(self.LUNG, self.fake_CG) * lambda_L2

        #Domain D
        self.loss_L2STDD = self.L2loss_decay(self.STD, self.fake_ED) * lambda_L2
        self.loss_L2DSTD = self.L2loss_decay(self.PHILD, self.fake_DE) * lambda_L2
        self.loss_L2STDC = self.L2loss_decay(self.STD, self.fake_FD) * lambda_L2
        self.loss_L2CSTD = self.L2loss_decay(self.PHILC, self.fake_DF) * lambda_L2
        self.loss_L2STDLUNG = self.L2loss_decay(self.LUNGSTD, self.fake_GD) * lambda_L2
        self.loss_L2LUNGSTD = self.L2loss_decay(self.LUNG, self.fake_DG) * lambda_L2

        #Domain E
        self.loss_L2DC = self.L2loss_decay(self.PHILD, self.fake_FE) * lambda_L2
        self.loss_L2CD = self.L2loss_decay(self.PHILC, self.fake_EF) * lambda_L2
        self.loss_L2DLUNG = self.L2loss_decay(self.PHILD, self.fake_GE) * lambda_L2
        self.loss_L2LUNGD = self.L2loss_decay(self.LUNG, self.fake_EG) * lambda_L2

        #Domain F
        self.loss_L2CLUNG = self.L2loss_decay(self.PHILC, self.fake_GF) * lambda_L2
        self.loss_L2LUNGC = self.L2loss_decay(self.LUNG, self.fake_FG) * lambda_L2

        #this is loss function for multipath cycleGAN: Adversarial losses + L2 losses
        self.loss_G = self.loss_G_AE + self.loss_G_EA + self.loss_G_AF + self.loss_G_FA + self.loss_G_AG + self.loss_G_GA + \
                       self.loss_G_BE + self.loss_G_EB + self.loss_G_BF + self.loss_G_FB + self.loss_G_BG + self.loss_G_GB  + \
                       self.loss_G_CE + self.loss_G_EC + self.loss_G_CF + self.loss_G_FC + self.loss_G_CG + self.loss_G_GC  + \
                       self.loss_G_DE + self.loss_G_ED + self.loss_G_DF + self.loss_G_FD + self.loss_G_GD + self.loss_G_DG  + \
                       self.loss_G_EF + self.loss_G_FE + self.loss_G_EG + self.loss_G_GE  + \
                       self.loss_G_FG + self.loss_G_GF + \
                       self.loss_cycle_AE + self.loss_cycle_EA + self.loss_cycle_AF + self.loss_cycle_FA + self.loss_cycle_AG + self.loss_cycle_GA +  \
                       self.loss_cycle_BE + self.loss_cycle_EB + self.loss_cycle_BF + self.loss_cycle_FB + self.loss_cycle_BG + self.loss_cycle_GB +  \
                       self.loss_cycle_CE + self.loss_cycle_EC + self.loss_cycle_CF + self.loss_cycle_FC + self.loss_cycle_CG + self.loss_cycle_GC + \
                       self.loss_cycle_DE + self.loss_cycle_ED + self.loss_cycle_DF + self.loss_cycle_FD + self.loss_cycle_DG + self.loss_cycle_GD +  \
                       self.loss_cycle_FE + self.loss_cycle_EF + self.loss_cycle_GE + self.loss_cycle_EG + \
                       self.loss_cycle_FG + self.loss_cycle_GF + \
                       self.loss_L2B50fD + self.loss_L2DB50f + self.loss_L2B50fC + self.loss_L2CB50f + self.loss_L2B50fLUNG + self.loss_L2LUNGB50f + \
                       self.loss_L2B30fD + self.loss_L2DB30f + self.loss_L2B30fC + self.loss_L2CB30f + self.loss_L2B30fLUNG + self.loss_L2LUNGB30f + \
                       self.loss_L2BONED + self.loss_L2DBONE + self.loss_L2BONEC + self.loss_L2CBONE + self.loss_L2BONELUNG + self.loss_L2LUNGBONE + \
                       self.loss_L2STDD + self.loss_L2DSTD + self.loss_L2STDC + self.loss_L2CSTD + self.loss_L2STDLUNG + self.loss_L2LUNGSTD + \
                       self.loss_L2DC + self.loss_L2CD  + self.loss_L2DLUNG + self.loss_L2LUNGD + \
                       self.loss_L2CLUNG + self.loss_L2LUNGC
        # self.loss_G.backward()
        return self.loss_G


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.optimizer_G.zero_grad()  
        self.set_requires_grad([self.netD_A,self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G], False) 
        with autocast():
            self.forward()
            loss_G = self.backward_G()             

        self.scalar.scale(loss_G).backward()
        self.scalar.step(self.optimizer_G)
        self.scalar.update()       
        # self.backward_G()             
        # self.optimizer_G.step()
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G], True) 
        
        with autocast():
            loss_D = self.backward_D()      # Gradients for all possible combinations of D's  
        #self.optimizer_D.step()  # update all discriminator weights
        self.scalar.scale(loss_D).backward()
        self.scalar.step(self.optimizer_D)

        self.scalar.update()
    
    # def optimize_parameters(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     # forward
    #     self.forward()      # compute fake images and reconstruction images.
    #     self.set_requires_grad([self.netD_A,self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G], False) #Multipath 
    #     self.optimizer_G.zero_grad()  # set G gradients to zero
    #     self.backward_G()             # calculate gradients for all G's
    #     self.optimizer_G.step()       # update weights for the encoders and decoders
    #     self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G], True) #Multipath
    #     self.optimizer_D.zero_grad()  
    #     self.backward_D()      # Gradients for all possible combinations of D's  
    #     self.optimizer_D.step()  # update all discriminator weights