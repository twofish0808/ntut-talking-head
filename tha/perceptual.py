from torch.autograd.variable import Variable
from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from torch import nn
import torch
from torch.nn import MSELoss
from torch import Tensor
import numpy as np
class perceptual_loss:
    def __init__(self,change,target) :
        self.loss=0
        self.change=change
        self.target=target
    def forword(self):
        model=vgg16(pretrained=True)
        model.features[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.eval()
        model.cuda()
        print(model)
        print(self.change[0])
        ch=self.change
        tar=self.target
        cov_change=model(Variable(ch).cuda())
        cov_target=model(Variable(tar).cuda())
        self.loss=MSELoss()(cov_change,cov_target)
        return self.loss

def getloss(change,target):
        model=vgg16(pretrained=True)
        model.features[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.eval()
        model.cuda()

        ch=change
        tar=target
        cov_change=model(Variable(ch).cuda())
        cov_target=model(Variable(tar).cuda())
        loss=MSELoss()(cov_change,cov_target)
        return loss
