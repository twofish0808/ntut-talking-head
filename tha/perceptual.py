from matplotlib.pyplot import get
from torch.autograd.variable import Variable
from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from torch import nn
import torch
from torch.nn import MSELoss, L1Loss
from torch import Tensor
import numpy as np
class perceptual_loss:
    def __init__(self,change,target) :
        self.loss=0
        self.change=change
        self.target=target
    def forword(self):
        model=vgg16(pretrained=True).features
        model[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for i in range(16,31):
            del model[16]

        ch=self.change
        tar=self.target
        loss=0
        new_model=get_newm(model,15,15)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)
        torch.cuda.empty_cache()


        new_model=get_newm(model,8,15)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)+loss
        torch.cuda.empty_cache()

        new_model=get_newm(model,3,8)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)+loss
        torch.cuda.empty_cache()

        return loss


def getloss(change,target):
    model=vgg16(pretrained=True).features
    model[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    for i in range(16,31):
        del model[16]

    ch=change
    tar=target
    loss=0
    new_model=get_newm(model,15,15)
    new_model.eval()
    new_model.cuda()
    result=new_model(ch)
    vgg_target=new_model(tar)
    loss=L1Loss()(result,vgg_target)
    torch.cuda.empty_cache()


    new_model=get_newm(model,8,15)
    new_model.eval()
    new_model.cuda()
    result=new_model(ch)
    vgg_target=new_model(tar)
    loss=L1Loss()(result,vgg_target)+loss
    torch.cuda.empty_cache()

    new_model=get_newm(model,3,8)
    new_model.eval()
    new_model.cuda()
    result=new_model(ch)
    vgg_target=new_model(tar)
    loss=L1Loss()(result,vgg_target)+loss
    torch.cuda.empty_cache()


            
    return loss

def get_newm(model,num,last):
    for i in range(num+1,last+1):
        # try:
            del model[num+1]
        # except:
        #     print("out of ramge")
    return model






