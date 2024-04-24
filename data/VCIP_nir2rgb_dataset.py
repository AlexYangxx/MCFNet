import os.path
from pathlib import Path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import random

import torchvision.transforms as transforms

# 包含支持的图像文件扩展名
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

# 用于检查文件是否为图像文件
def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS

# 随机获取图像调整位置
def randomCrop(img, width, height):
    assert img.size[0] >= height
    assert img.size[1] >= width

    x = random.randint(0, img.size[0] - width)
    y = random.randint(0, img.size[1] - height)
    position = [x,y, x+width,y+height]
    img = img.crop((x,y, x+width,y+height))

    return img, [x,y, x+width,y+height]

# 裁剪图片和调整图片大小
def crop_resize(img, position, resize_size):
    img = img.crop(position) # 根据给定的位置坐标进行裁剪
    img = img.resize(resize_size,Image.BICUBIC) # 使用双三次插值法调整图像大小

    return img

# 这是一个用于处理VCIP（Video and Image Processing）NIR（Near-Infrared）到RGB（Red, Green, Blue）转换任务的数据集类，
# 继承自BaseDataset。该数据集包含两个域，一个是NIR域（dir_A），另外两个是RGB域，分别是RGB-Registered（dir_B_1）和RGB-Online（dir_B_2）
class VCIPNir2RGBDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir_A = Path(opt.dataroot, 'NIR')  # create a path '/path/to/data/trainA'
        self.dir_B_1 = Path(opt.dataroot, 'RGB-Registered')  # create a path '/path/to/data/trainB1'
        self.dir_B_2 = Path(opt.dataroot, 'RGB-Online') # create a path '/path/to/data/trainB2'

        self.A_paths= [f for f in self.dir_A.glob('*') if is_image(f)]
        self.A_pair_paths = [f for f in self.dir_A.glob('*') if is_image(f)]
        self.B1_paths= [f for f in self.dir_B_1.glob('*') if is_image(f)]
        self.B1_pair_paths= [f for f in self.dir_B_1.glob('*') if is_image(f)]
        self.B2_paths= [f for f in self.dir_B_2.glob('*') if is_image(f)]
        # print(self.B2_paths)

        self.A_paths.sort()
        self.A_pair_paths.sort()
        self.B1_paths.sort()
        self.B1_pair_paths.sort()

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B1_paths) + len(self.B2_paths)  # get the size of dataset B
        # print(self.A_size)
        # print(self.B_size)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]  # 确保索引在范围内 make sure index is within then range
        B_pair_RGB = self.B1_pair_paths[index % self.A_size]

        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # 随机化域 B 的索引以避免固定对 randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        # 从两个不同的图像源中获取对应索引的图像文件路径
        if index_B < len(self.B1_paths):
            B_path = self.B1_paths[index_B]
            A_pair_NIR = self.A_pair_paths[index_B]
        else:
            B_path = self.B2_paths[index_B-len(self.B1_paths)]
            A_pair_NIR = self.A_pair_paths[index_B-len(self.B1_paths)]

        rad = random.uniform(0,1)
        rad2= random.uniform(0,1)

        if rad <=0.2:
            real_A = Image.open(A_path).convert('RGB')
            real_B = Image.open(B_path).convert('RGB')
            real_A_gray = transforms.functional.to_grayscale(Image.open(B_pair_RGB),3)
            real_B_gray = transforms.functional.to_grayscale(Image.open(B_path),3)
            fake_B_pair = Image.open(B_pair_RGB).convert('RGB')
            fake_A_pair = Image.open(A_pair_NIR).convert('RGB')

            if rad2<=0.5:
                A_img = np.float32(np.asarray(real_A).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(fake_A_pair).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(real_A_gray).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(real_B).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(real_B_gray).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(fake_B_pair).transpose(2,0,1)) / 255

            else: # mirrored_image()水平翻转图像
                A_img = np.float32(np.asarray(ImageOps.mirror(real_A)).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(ImageOps.mirror(fake_A_pair)).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(ImageOps.mirror(real_A_gray)).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(ImageOps.mirror(real_B)).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(ImageOps.mirror(real_B_gray)).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(ImageOps.mirror(fake_B_pair)).transpose(2,0,1)) / 255

        else:
            # 对图像A对比度调整
            real_A = Image.open(A_path).convert('RGB')
            factor_contrast =random.uniform(0.5,1.5)
            enhancer_contrast = ImageEnhance.Contrast(real_A)
            real_A = enhancer_contrast.enhance(factor_contrast)

            fake_A_pair = Image.open(A_pair_NIR).convert('RGB')
            enhancer_contrast = ImageEnhance.Contrast(fake_A_pair)
            fake_A_pair = enhancer_contrast.enhance(factor_contrast)

            real_A_gray = transforms.functional.to_grayscale(Image.open(B_pair_RGB), 3)
            real_B = Image.open(B_path).convert('RGB')
            real_B_gray = transforms.functional.to_grayscale(Image.open(B_path),3)
            fake_B_pair = Image.open(B_pair_RGB).convert('RGB')

            real_A,position = randomCrop(real_A,200,200)
            real_A = real_A.resize([256,256],Image.BICUBIC) # 调整A的像素大小

            # 裁剪并恢复大小
            real_A_gray = crop_resize(real_A_gray, position, [256, 256])
            fake_A_pair = crop_resize(fake_A_pair, position, [256, 256])
            real_B = crop_resize(real_B,position,[256,256])
            real_B_gray = crop_resize(real_B_gray,position,[256,256])
            fake_B_pair = crop_resize(fake_B_pair,position,[256,256])

            if rad2<=0.5:
                A_img = np.float32(np.asarray(real_A).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(fake_A_pair).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(real_A_gray).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(real_B).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(real_B_gray).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(fake_B_pair).transpose(2,0,1)) / 255
            else:
                A_img = np.float32(np.asarray(ImageOps.mirror(real_A)).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(ImageOps.mirror(fake_A_pair)).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(ImageOps.mirror(real_A_gray)).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(ImageOps.mirror(real_B)).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(ImageOps.mirror(real_B_gray)).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(ImageOps.mirror(fake_B_pair)).transpose(2,0,1)) / 255



        return {'A': A_img[0:1], 'A_NIR': A_pair_NIR_img[0:1], 'A_gray': A_pair_GRAY_img[0:1],'B': B_img[0:self.opt.output_nc],'B_gray':B_gray_img[0:1],'B_RGB':B_pair_RGB_img[0:self.opt.input_nc],'A_paths': str(A_path), 'B_paths': str(B_path)}
    """ 
        'A': NIR域的图像数据,通过切片A_img[0:1]获取。
        'B': RGB域的图像数据,通过切片B_img[0:self.opt.output_nc]获取。这里使用了self.opt.output_nc指定的通道数,可能是为了处理多通道的RGB图像。
        'B_gray': RGB-Registered域灰度图像的数据,通过切片B_gray_img[0:1]获取。
        'B_RGB': RGB-Registered域的彩色图像数据,通过切片B_pair_RGB_img[0:self.opt.input_nc]获取。这里使用了self.opt.input_nc指定的通道数,可能是为了处理多通道的RGB图像。
        'A_paths': NIR域图像的文件路径,以字符串形式记录。
        'B_paths': RGB域图像的文件路径,以字符串形式记录。
    """

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

class VCIPNir2RGBDataset_paired(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir_A = Path(opt.dataroot, 'NIR')  # create a path '/path/to/data/trainA'
        self.dir_B_1 = Path(opt.dataroot, 'RGB-Registered')  # create a path '/path/to/data/trainB1'
        # self.dir_B_2 = Path(opt.dataroot, 'RGB-Online') # create a path '/path/to/data/trainB2'

        self.A_paths= [f for f in self.dir_A.glob('*') if is_image(f)]
        self.B1_paths= [f for f in self.dir_B_1.glob('*') if is_image(f)]
        # self.B2_paths= [f for f in self.dir_B_2.glob('*') if is_image(f)]

        self.A_paths.sort()
        self.B1_paths.sort()

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B1_paths)  # get the size of dataset B

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_pair_NIR = A_path
        B_path = self.B1_paths[index % self.B_size]
        B_pair_RGB = B_path


        rad = random.uniform(0,1)
        rad2= random.uniform(0,1)

        if rad <=0.2:
            real_A = Image.open(A_path).convert('RGB')
            fake_A_pair = Image.open(A_pair_NIR).convert('RGB')
            real_A_gray = transforms.functional.to_grayscale(Image.open(B_pair_RGB), 3)
            real_B = Image.open(B_path).convert('RGB')
            real_B_gray = transforms.functional.to_grayscale(Image.open(B_path),3)
            fake_B_pair = Image.open(B_pair_RGB).convert('RGB')

            if rad2<=0.5:
                A_img = np.float32(np.asarray(real_A).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(fake_A_pair).transpose(2,0,1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(real_A_gray).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(real_B).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(real_B_gray).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(fake_B_pair).transpose(2,0,1)) / 255
            else:
                A_img = np.float32(np.asarray(ImageOps.mirror(real_A)).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(ImageOps.mirror(fake_A_pair)).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(ImageOps.mirror(real_A_gray)).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(ImageOps.mirror(real_B)).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(ImageOps.mirror(real_B_gray)).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(ImageOps.mirror(fake_B_pair)).transpose(2,0,1)) / 255

        else:
            real_A = Image.open(A_path).convert('RGB')
            factor_contrast =random.uniform(0.5,1.5)
            enhancer_contrast = ImageEnhance.Contrast(real_A)
            real_A = enhancer_contrast.enhance(factor_contrast)

            fake_A_pair = Image.open(A_pair_NIR).convert('RGB')
            enhancer_contrast = ImageEnhance.Contrast(fake_A_pair)
            fake_A_pair = enhancer_contrast.enhance(factor_contrast)

            real_A_gray = transforms.functional.to_grayscale(Image.open(B_pair_RGB), 3)
            real_B = Image.open(B_path).convert('RGB')
            real_B_gray = transforms.functional.to_grayscale(Image.open(B_path),3)
            fake_B_pair = Image.open(B_pair_RGB).convert('RGB')

            real_A,position = randomCrop(real_A,200,200)
            real_A = real_A.resize([256,256],Image.BICUBIC)

            real_A_gray = crop_resize(real_A_gray, position, [256, 256])
            fake_A_pair = crop_resize(fake_A_pair,position,[256, 256])
            real_B = crop_resize(real_B,position,[256,256])
            real_B_gray = crop_resize(real_B_gray,position,[256,256])
            fake_B_pair = crop_resize(fake_B_pair,position,[256,256])

            if rad2<=0.5:
                A_img = np.float32(np.asarray(real_A).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(fake_A_pair).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(real_A_gray).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(real_B).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(real_B_gray).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(fake_B_pair).transpose(2,0,1)) / 255
            else:
                A_img = np.float32(np.asarray(ImageOps.mirror(real_A)).transpose(2,0,1)) / 255
                A_pair_NIR_img = np.float32(np.asarray(ImageOps.mirror(fake_A_pair)).transpose(2, 0, 1)) / 255
                A_pair_GRAY_img = np.float32(np.asarray(ImageOps.mirror(real_A_gray)).transpose(2, 0, 1)) / 255
                B_img = np.float32(np.asarray(ImageOps.mirror(real_B)).transpose(2,0,1)) / 255
                B_gray_img = np.float32(np.asarray(ImageOps.mirror(real_B_gray)).transpose(2,0,1)) / 255
                # B_gray_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
                B_pair_RGB_img = np.float32(np.asarray(ImageOps.mirror(fake_B_pair)).transpose(2,0,1)) / 255


        return {'A': A_img[0:1], 'A_NIR': A_pair_NIR_img[0:1], 'A_gray': A_pair_GRAY_img[0:1],'B': B_img[0:self.opt.output_nc],'B_gray':B_gray_img[0:1],'B_RGB':B_pair_RGB_img[0:self.opt.input_nc], 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)




class VCIPNir2RGBDataset_test(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir_A = Path(opt.dataroot)  # create a path '/path/to/data/trainA'
        self.dir_B = Path(opt.dataroot)  # create a path '/path/to/data/trainB'

        self.A_paths= [f for f in self.dir_A.glob('*_nir_reg.png') if is_image(f)]
        self.B_paths= [f for f in self.dir_B.glob('*_rgb_reg.png') if is_image(f)]

        self.A_paths.sort()
        self.B_paths.sort()

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        # 获取当前索引下的图像路径
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        # 读取图像并进行预处理
        A_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
        A_pair_NIR_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
        A_pair_GRAY_img = np.float32(np.asarray(transforms.functional.to_grayscale(Image.open(B_path), 3)).transpose(2, 0, 1)) / 255
        B_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1)) / 255
        B_gray_img = np.float32(np.asarray(transforms.functional.to_grayscale(Image.open(B_path),3)).transpose(2,0,1)) / 255
        B_pair_RGB_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1)) / 255

       # 返回包含图像及其路径的字典
        return {'A': A_img[0:1], 'A_NIR': A_pair_NIR_img[0:1], 'A_gray': A_pair_GRAY_img[0:1],'B': B_img,'B_gray':B_gray_img[0:1],'B_RGB':B_pair_RGB_img[0:self.opt.input_nc], 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        返回数据集中的图像总数。

        由于我们有两个数据集，其图像数量可能不同，
        我们最多采取
        """

        # 返回数据集中图像的总数（取两个数据集中的最大值）
        return max(self.A_size, self.B_size)



class VCIPNir2RGBDataset_gen(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir_A = Path(opt.dataroot)  # create a path '/path/to/data/trainA'
        # self.dir_B = Path(opt.dataroot)  # create a path '/path/to/data/trainB'

        self.A_paths= [f for f in self.dir_A.glob('*nir*.png') if is_image(f)]
        # self.B_paths= [f for f in self.dir_B.glob('*_rgb_reg.png') if is_image(f)]

        self.A_paths.sort()
        # self.B_paths.sort()

        self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        # B_path = self.B_paths[index]

        A_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1)) / 255
        # B_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1)) / 255
        # B_gray_img = np.float32(np.asarray(transforms.functional.to_grayscale(Image.open(B_path),3)).transpose(2,0,1)) / 255
        # B_pair_RGB_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1)) / 255


        return {'A': A_img[0:1], 'A_paths': str(A_path)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size


