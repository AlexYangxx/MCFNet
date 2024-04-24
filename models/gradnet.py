import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class Gradient_Net(nn.Module):

    def __init__(self):
        super(Gradient_Net,self).__init__()

        kernel_x = [[-1., 0., 1.], [-2.,0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        # kernel_x = kernel_x.repeat(1,3,1,1)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        # kernel_y = kernel_y.repeat(1,3,1,1)
        self.weight_x = nn.Parameter(data=kernel_x,requires_grad=False).cuda()
        self.weight_y = nn.Parameter(data=kernel_y,requires_grad=False).cuda()
    
    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding = 1)
        grad_y = F.conv2d(x,self.weight_y, padding = 1)
        gradient = torch.sqrt(torch.pow(grad_x,2)+ torch.pow(grad_y,2))
        
        return gradient

        # return grad_x, grad_y