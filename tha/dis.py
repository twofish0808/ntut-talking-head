import torch
from torch import Tensor
from torch.nn import Sequential, Sigmoid, Tanh

from tha.batch_input_module import BatchInputModule, BatchInputModuleSpec
from nn.conv import Conv7
from nn.u_net_module import UNetModule
from torch import nn



class NetD(nn.Module):
    """
    判别器定义
    """

    def __init__(self):
        super(NetD, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # 输入 3*96*96
            nn.Conv2d(4, ndf, (5, 5), (3, 3), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf)*32*32

            nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2)*16*16

            nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4)*8*8

            nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8)*4*4

            nn.Conv2d(ndf * 8, 1, (4, 4), (1, 1), (0, 0), bias=False),
            nn.Sigmoid()  # 输出一个数：概率
        )

    def forward(self, inp):
        return self.main(inp).view(-1)