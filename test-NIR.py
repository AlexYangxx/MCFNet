
import os
from options.test_options import TestOptions

from data.VCIP_nir2rgb_dataset import *
from models.CycleGanNIR_model import *

from util.visualizer import save_images
from util import html
import pytorch_ssim

import numpy as np
from numpy import linalg as LA
import torch



if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试选项 get test options

    opt.num_threads = 0   # 测试代码仅支持 num_threads = 1 test code only supports num_threads = 1
    opt.batch_size = 1    # 测试代码仅支持 batch_size = 1 test code only supports batch_size = 1
    opt.serial_batches = True  # 禁用数据变换;如果需要随机选择的图像的结果，请注释这一行 disable data shuffling; comment this line if results on randomly chosen images are needed.


    dataset= VCIPNir2RGBDataset_test(opt) # 创建测试数据集 create dataset
    print("dataset [%s] was created" % type(dataset).__name__)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                    shuffle=not opt.serial_batches, num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

    model = CycleGANModel(opt)       # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    #opt.eval = True
    if opt.eval:
        model.eval()


    for i, data in enumerate(dataloader):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML




import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob, os
from PIL import Image
import cv2

# cwd = os.getcwd()
GT_path = "H:/Desktop/ATCycleGAN/submit_Final_VCIP_ASTARTREK/datasets/Testing"
GT_path = opt.dataroot
im_path = webpage.img_dir


GT_file = os.listdir(GT_path)
GT_file = [f for f in GT_file if f.endswith('.png')]
GT_file =[f for f in GT_file if f.endswith('_rgb_reg.png') and f.startswith("Testing_")]
GT_file.sort()


im_file = os.listdir(im_path)
if "Testing" in opt.dataroot:
    im_file =[f for f in im_file if f.endswith('rgb_reg_fake_B.png') and f.startswith("Testing_")]
    classname = "Testing"

im_file.sort()



PSNR_list = []
SSIM_list = []
AE_list = []

# len(GT_file)
for i in range(0, len(GT_file)):
    # GT = transforms.functional.to_grayscale(Image.open(os.path.join(GT_path,GT_file[i])).convert('RGB'))
    # im = transforms.functional.to_grayscale(Image.open(os.path.join(im_path,im_file[i])).convert('RGB'))
    GT = Image.open(os.path.join(GT_path,GT_file[i])).convert('RGB')
    im = Image.open(os.path.join(im_path,im_file[i])).convert('RGB')
    GT_arr = np.float32(np.asarray(GT))/255
    im_arr = np.float32(np.asarray(im))/255

    PSNR = -10*np.log10(np.mean((im_arr - GT_arr)**2))
    PSNR_list.append(PSNR)

    SSIM = ssim(GT_arr, im_arr, data_range=im_arr.max() - im_arr.min(), multichannel=True)
    SSIM_list.append(SSIM)

    eps = 1e-6
    dotP = np.sum(GT_arr * im_arr, axis = 2)
    Norm_pred = np.sqrt(np.sum(im_arr * im_arr, axis = 2))
    Norm_true = np.sqrt(np.sum(GT_arr * GT_arr, axis = 2))
    AE = 180 / np.pi * np.arccos(dotP / (Norm_pred * Norm_true + eps))
    AE = AE.ravel().mean()
    AE_list.append(AE)

print(sum(PSNR_list)/len(PSNR_list))
print(sum(SSIM_list)/len(SSIM_list))
print(sum(AE_list)/len(AE_list))

cwd = os.getcwd()
iteration = opt.load_iter

with open(cwd+'/'+classname+'_'+str(iteration)+'_Losses_test.txt','w') as f:
    f.write("Average PSNR %f\n" %(sum(PSNR_list)/len(PSNR_list)))
    f.write("Average SSIM %f\n" %(sum(SSIM_list)/len(SSIM_list)))
    f.write("Average AE %f \n"  %(sum(AE_list)/len(AE_list)))

    f.write("PSNR_list:\n")
    for index, item in enumerate(PSNR_list):
        f.write("{}. {}\n".format(index + 1, item))

    f.write("SSIM_list:\n")
    for index, item in enumerate(SSIM_list):
        f.write("{}. {}\n".format(index + 1, item))

    f.write("AE_list:\n")
    for index, item in enumerate(AE_list):
        f.write("{}. {}\n".format(index + 1, item))