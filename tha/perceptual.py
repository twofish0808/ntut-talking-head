from torch.autograd.variable import Variable
from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from cnn_finetune import make_model
import torch
from torch.nn import MSELoss
from torch import Tensor
import numpy as np
class perceptual_loss:
    def __init__(self,change,target,label) :
        self.loss=0
        self.change=change
        self.label=label
        self.target=target
    def forword(self):
        model=vgg16(pretrained=True)

        print(self.change[0])
        ch=self.change
        tar=self.change
        cov_change=model(ch)
        cov_target=model(tar)
        self.loss=MSELoss(cov_change,cov_target)
        return self.loss
