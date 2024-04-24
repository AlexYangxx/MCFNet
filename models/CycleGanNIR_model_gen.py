import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import CycleGanNIR_net
from . import gradnet
import torchvision.transforms as transforms


class CycleGANModel_gen(BaseModel):

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','pair_A', 'pair_B'] #, 'edge_A', 'edge_B'
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            visual_names_A = ['real_A', 'fake_B']
            # visual_names_B = ['real_B', 'fake_A', 'rec_B','real_B_gray']
        else:
            visual_names_A = ['fake_B']
            # visual_names_B = ['real_B', 'fake_A', 'rec_B','real_B_gray']  

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            # visual_names_B.append('idt_A')

        self.visual_names = visual_names_A   # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        #self.netG_A = CycleGanNIR_net.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG_B = CycleGanNIR_net.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        # not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        #using self defined NN
        self.netG_A = CycleGanNIR_net.define_G(opt.input_nc, opt.output_nc, opt.ngf, "dscale_unet_256", opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = CycleGanNIR_net.define_G(opt.output_nc, opt.input_nc, opt.ngf, "dscale_unet_RGB2NIR256", opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) 
        # self.edgenet = gradnet.Gradient_Net()                                

        if self.isTrain:  # define discriminators
            self.netD_A = CycleGanNIR_net.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = CycleGanNIR_net.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.edgenet = gradnet.Gradient_Net()


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = CycleGanNIR_net.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionPaired = torch.nn.L1Loss()
            self.criterionEdge = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G_paired = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_G_paired)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = torch.Tensor(input['A']).float().to(self.device)
        # self.real_A_edge = self.edgenet(torch.Tensor(input['A']).float().to(self.device))
        # self.real_A = torch.cat([self.real_A1,self.real_A_edge],dim = 1)
        # self.real_B = torch.Tensor(input['B']).float().to(self.device)
        # self.real_B_gray = torch.Tensor(input['B_gray']).float().to(self.device)

        if self.isTrain:
            self.real_B_RGB_pair = torch.Tensor(input['B_RGB']).float().to(self.device)
        self.image_paths = input['A_paths']
        # self.image_paths = input['B_paths']
    

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.real_A_in = torch.cat([self.real_A, self.edgenet(self.real_A)],dim=1)
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # # self.fake_A = self.netG_B(self.real_B_gray)  # G_B(B)
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # # self.fake_A_in = torch.cat([self.fake_A, self.edgenet(self.fake_A)],dim=1)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

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
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self,flag):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * 3
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * 3
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0



        # Decay of Lambda:
        #print("Lambda_A = %f" %lambda_A)
        #print("Lambda_B = %f" %lambda_B)

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) 
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) 

        # combined loss and calculate gradients
        # edge_real_NIR = self.edgenet(self.real_A)
        # edge_real_RGB = self.edgenet(self.real_B)
        # edge_fake_RGB = self.edgenet(self.fake_B)
        # edge_fake_NIR = self.edgenet(self.fake_A)
        # edge_real_RGB_gray = self.edgenet(self.real_B_gray)
        # edge_real_RGB_pair = self.edgenet(self.real_B_RGB_pair)
        
        # edge_rec_RGB = self.edgenet(self.rec_B)
        # edge_rec_NIR = self.edgenet(self.rec_A)

        if flag == 1:
            self.loss_pair_A = self.criterionPaired(self.fake_B,self.real_B)
            self.loss_pair_B = self.criterionPaired(self.fake_A,self.real_A)
            # loss_edge_A1 = self.criterionEdge(edge_real_NIR,edge_fake_NIR)*3
            # loss_edge_B1 = self.criterionEdge(edge_real_RGB,edge_fake_RGB) 
            loss_edge_A1 = 0
            loss_edge_B1 = 0 
        else:
            self.loss_pair_A = self.criterionPaired(self.fake_B,self.real_B_RGB_pair)
            # self.loss_pair_B = 0
            self.loss_pair_B = 0
            # loss_edge_A1 = self.criterionEdge(edge_real_NIR,edge_fake_RGB)
            # loss_edge_B1 = self.criterionEdge(edge_real_RGB,edge_fake_NIR)
            # loss_edge_A1 = self.criterionEdge(edge_real_RGB_gray,edge_fake_NIR)*3
            # loss_edge_A1 = self.criterionEdge(edge_real_RGB_gray,edge_fake_NIR)*3
            # loss_edge_A1 = 0
            # loss_edge_B1 = self.criterionEdge(edge_real_RGB_pair,edge_fake_RGB)*3
            # lambda_edge = 10


        # self.loss_edge_A =   self.criterionEdge(edge_rec_NIR,edge_real_NIR)*5
        # self.loss_edge_B =   self.criterionEdge(edge_rec_RGB,edge_real_RGB)*5+self.criterionEdge(edge_real_RGB_pair,edge_fake_RGB)*3

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A * lambda_A + self.loss_cycle_B * lambda_B  #+ self.loss_idt_A + self.loss_idt_B 
        self.loss_G = self.loss_G + self.loss_pair_A * 60 + self.loss_pair_B * 60 #
        self.loss_G.backward()

    # def backward_G_paired(self,flag):

    #     if flag == 1:
    #         self.loss_pair_A = self.criterionPaired(self.netG_A(self.fake_B),self.real_B)
    #         self.loss_pair_B = self.criterionPaired(self.netG_B(self.fake_A),self.real_A)
    #     else:
    #         self.loss_pair_A = self.loss_pair_A 
    #         self.loss_pair_B = self.loss_pair_B

    #     self.loss_G_paired =  self.loss_pair_A + self.loss_pair_B
    #     self.loss_G_paired.backward()

    def optimize_parameters(self,flag):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(flag)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_Bs
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # # G_A and G_B paired
        # if flag == 1:
        #     self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        #     self.optimizer_G_paired.zero_grad()  # set G_A and G_B's gradients to zero
        #     self.backward_G_paired(flag)         # calculate gradients for G_A and G_B
        #     self.optimizer_G_paired.step()       # update G_A and G_B's weights
