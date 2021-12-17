from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from torch.nn import MSELoss
from torch import Tensor
class perceptual_loss:
    def __init__(self,change,target,label) :
        self.loss=0
        self.change=change
        self.label=label
        self.target=target
    def forword(self):
        vg=vgg16(pretrained=True)
        cov_change=vg(self.change,self.label)
        cov_target=vg(self.target,self.label)
        self.loss=MSELoss(cov_change,cov_target)
        return self.loss
