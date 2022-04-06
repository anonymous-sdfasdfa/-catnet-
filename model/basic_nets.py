import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class res_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(res_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity,x)
        return x



class attenNet_basic(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(attenNet_basic,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.at3 = res_unit(128,up_size=9)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear((9*9*128),512)
        # self.dp  = nn.Dropout()
        self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x

        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        # x = self.dp(x)
        x = self.fc2(x)
        return x




class CNN_basic(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=5),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((9, 9))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 9 * 9, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(640, 2),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
