import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet, att_mode, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.att_mode = att_mode
        self.device = device

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        if self.att_mode == 'without_ESVR':
            x = F.adaptive_max_pool2d(x, (1, 1)).squeeze((2, 3))

        else:
            x = self.resnet.avgpool(x)
            x = x.view(x.size(0), -1)


        return x, fc, att


