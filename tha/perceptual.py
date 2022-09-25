from operator import mod
from matplotlib.pyplot import get
from torch.autograd.variable import Variable
from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from torchvision import models
from torch import nn
import torch
from torch.nn import MSELoss, L1Loss
from torch import Tensor
import numpy as np



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        features[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # print(features)
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        # self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        # for x in range(16, 23):
        #     self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        # h = self.to_relu_4_3(h)
        # h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

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
    # model=vgg16(pretrained=True).features
    # model[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # for i in range(16,31):
    #     del model[16]

    ch=change
    tar=target
    # loss=0
    # new_model=get_newm(model,3,15)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)
    # torch.cuda.empty_cache()
    # print(model)
    # print(new_model)


    # new_model=get_newm(model,8,15)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)+loss
    # torch.cuda.empty_cache()

    # new_model=get_newm(model,3,8)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)+loss
    # torch.cuda.empty_cache()


            
    # return loss*4/3
    nmodel=Vgg16()
    nmodel.eval()
    nmodel.cuda()
    result=nmodel(ch)
    torch.cuda.empty_cache()
    vgg_target=nmodel(tar)
    torch.cuda.empty_cache()
    # print(result)
    # print(vgg_target)
    loss=0
    for i in range(0,3):
        loss=loss+L1Loss()(result[i],vgg_target[i])
    torch.cuda.empty_cache()
    return loss

def get_newm(model,num,last):
    model1=model
    for i in range(num+1,last+1):
        # try:
            del model[num+1]
        # except:
        #     print("out of ramge")
    return model

class vgg_discriminator(nn.Module):
    def __init__(self):
        super(vgg_discriminator, self).__init__()
        self.conv1_1=conv_block(7,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv1_2=conv_block(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.maxPool1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        self.conv2_1=conv_block(64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv2_2=conv_block(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.maxPool2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        self.conv3_1=conv_block(128,256,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv3_2=conv_block(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv3_3=conv_block(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1))

    def forward(self,x):
        x=self.conv1_1(x)
        a1=self.conv1_2(x)
        a2=self.maxPool1(a1)
        a2=self.conv2_1(a2)
        a2=self.conv2_2(a2)
        a3=self.maxPool2(a2)
        a3=self.conv3_1(a3)
        a3=self.conv3_2(a3)
        a3=self.conv3_3(a3)

        a1=nn.Tanh()(a1)
        a2=nn.Tanh()(a2)
        a3=nn.Tanh()(a3)

        return a1,a2,a3


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

def return_vgg():
    model=vgg_discriminator()
    print(model)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters,dp=0.2, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Dropout(dp))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(7, 64,normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Tanh()
            )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input).view(-1)





class Vgg_discriminator(nn.Module):
    def __init__(self) :
        super(Vgg_discriminator, self).__init__()
        self.loss=0
        self.model=vgg16(pretrained=True).features
        self.model[0]=nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self,image_A,image_B):
        image=torch.cat((image_A,image_B),1)
        
        

        return self.model(image).view(-1)