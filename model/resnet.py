import torch.nn as nn
import  torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class Basic_attention_Block(nn.Module):
    """
    this is the attention module before Residual structure
    """

    def __init__(self, channel, up_size=None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(Basic_attention_Block, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size, up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity, x)+identity
        x = self.bn(x)
        x = self.relu(x)

        return x


class attention_Block(nn.Module):
    """
    this is the attention module before Residual structure
    """

    def __init__(self, channel, up_size=None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(attention_Block, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size, up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity, x)+identity
        return x