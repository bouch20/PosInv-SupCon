
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Feature(nn.Module):
    def __init__(self, model='resnet18'):
        nn.Module.__init__(self)
        self.model = model

        self.base = models.__dict__[model](pretrained=False)
 
    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
def pytorch_resnet18(**kwargs):
    return Feature('resnet18')

def pytorch_resnet34(**kwargs):
    return Feature('resnet34')

def pytorch_resnet50(**kwargs):
    return Feature('resnet50')