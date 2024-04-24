import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

def imshow(im):

    assert im.min()>=0 and im.max()<= 255 and im.squeeze().ndim<=3, 'input array should be in range [0,1] or [0,255]'

    if isinstance(im, torch.Tensor):
        im= im.numpy()

    if isinstance(im, np.ndarray):
        # first, check data range
        if(im.max()<=1.0):
            im= 255*im
        # second, check data type
        if im.dtype != np.uint8:
            im= np.uint8(im)
        # third, check dimension orders        
        ndims = im.squeeze().ndim
        if ndims== 3:
            if im.shape[0]==3:
                im= im.transpose(1,2,0)               

        # Image.fromarray(im).show()
        plt.show(plt.imshow(im))



if __name__ == "__main__":
    import sys
    sys.path.append('.')

    # from data.deepHDR17_dataset import *

    # hci_data= deepHDR17Dataset('D:/Datasets/HDR/DeepHDR17/DeepHDR_train_dataset.hdf5', patch_size= 500, location= (100, 100))
    # print(len(hci_data))
    # x= hci_data[35] # 34, 35, 37, 55, 70, 73

    # im= x['IN_S_CAM']
    
