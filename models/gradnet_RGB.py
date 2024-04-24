import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class Gradient_Net_RGB(nn.Module):

    def __init__(self):
        super(Gradient_Net_RGB,self).__init__()

        kernel_x = [[-1., 0., 1.], [-2.,0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        # kernel_x = kernel_x.repeat(1,3,1,1)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        # kernel_y = kernel_y.repeat(1,3,1,1)
        self.weight_x = nn.Parameter(data=kernel_x,requires_grad=False).cuda()
        self.weight_y = nn.Parameter(data=kernel_y,requires_grad=False).cuda()
    
    def forward(self, x):
        R = x[:, 0, :, :].unsqueeze(1)
        G = x[:, 1, :, :].unsqueeze(1)
        B = x[:, 2, :, :].unsqueeze(1)

        x = x.unsqueeze(1)
        grad_xR = F.conv2d(R, self.weight_x, padding = 1)
        grad_yR = F.conv2d(R, self.weight_y, padding = 1)
        grad_xG = F.conv2d(G, self.weight_x, padding = 1)
        grad_yG = F.conv2d(G, self.weight_y, padding = 1)
        grad_xB = F.conv2d(B, self.weight_x, padding = 1)
        grad_yB = F.conv2d(B, self.weight_y, padding = 1)
            
        # gradient = torch.abs(grad_x) + torch.abs(grad_y)
        
        gradientR = torch.sqrt(torch.pow(grad_xR,2)+ torch.pow(grad_yR,2))
        gradientG = torch.sqrt(torch.pow(grad_xG,2)+ torch.pow(grad_yG,2))
        gradientB = torch.sqrt(torch.pow(grad_xB,2)+ torch.pow(grad_yB,2))
        gradient =  gradientR + gradientG + gradientB

        return gradient

        # return grad_x, grad_y