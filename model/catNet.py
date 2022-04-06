import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from  model.IPCGANs import Generator,BasicBlock,Generator_ffhq
import torchvision.models as models
import numpy as np
from torchvision.datasets.folder import pil_loader
from torchvision.utils import save_image
from utils.network import Conv2d #same padding
from model.resnet import BasicBlock
from model.resnet import BasicBlock,Basic_attention_Block





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


class BasicConv2d(nn.Module):
    """
    basic convoluation model
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class Spatial_attention_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(Spatial_attention_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)

        return x



class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class KMM(nn.Module):
    def __init__(self,channel):
        super(KMM, self).__init__()
        # self.conv1 = BasicConv2d(channel,channel,3,1,1)
        # self.conv2 = BasicConv2d(channel, channel, 3, 1, 1)
        # self.conv3 = BasicConv2d(channel, channel, 3, 1, 1)
        self.spatial = res_unit(channel)
        self.channel = SELayer(channel,8)

    def forward(self,x):
        identity = x
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.channel(x)
        x = self.spatial(x)
        x  = x+identity
        return x




########################################### best
class CatNet_kmm(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(64, 64, 1, 1)
        self.cn_f2 = BasicConv2d(64, 64, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base(x_p)
        x_c = self.base(x_c)

        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)

        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c

class CatNet_two(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=2)

        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.relu = nn.ReLU()

        self.repeat_blocks = nn.Sequential(
            BasicBlock(64, 64),
            Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
        )

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 64, 1, 1)
        self.cn_f2 = BasicConv2d(128, 64, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    # def base_forward(self,x):
    #     x1 = self.base(x)
    #
    #     x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x))) # 32x128x128
    #     x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1))) # 64x64x64
    #     x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2))) #64x32x32
    #     x2 = self.repeat_blocks(x_layer3) #64x32x32
    #     x = torch.cat((x1, x2), 1)
    #     return  x

    def base_forward(self, x):
        x1 = self.base(x)

        x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x)))  # 32x128x128
        x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1)))  # 64x64x64
        x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2)))  # 64x32x32
        x2 = self.repeat_blocks(x_layer3)  # 64x32x32
        x = torch.cat((x1, x2), 1)
        return x
    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base_forward(x_p)
        x_c = self.base_forward(x_c)




        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)

        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c

class CatNet_two_fc(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=2)

        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.relu = nn.ReLU()

        self.repeat_blocks = nn.Sequential(
            BasicBlock(64, 64),
            Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
        )

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(64)
        self.bn_kmm = nn.BatchNorm2d(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 64, 1, 1)
        self.cn_f2 = BasicConv2d(128, 64, 1, 1)
        self.final_layer1 = BasicConv2d(128,64,5,1)
        self.final_layer2 = BasicConv2d(64, 64,3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear((64*6*6),256)
        self.fc2 = nn.Linear(256,2)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def base_forward(self, x):
        x1 = self.base(x)

        x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x)))  # 32x128x128
        x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1)))  # 64x64x64
        x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2)))  # 64x32x32
        x2 = self.repeat_blocks(x_layer3)  # 64x32x32
        x = torch.cat((x1, x2), 1)
        return x
    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base_forward(x_p)
        x_c = self.base_forward(x_c)




        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.bn_kmm(self.KMM(k_p))
        k_c = self.bn_kmm(self.KMM(k_c))

        feature = torch.cat((k_p,k_c),dim=1)

        f1 = self.final_layer1(feature)
        f2 = self.pool(f1)
        f3 = self.final_layer2(f2)
        f4 = self.pool(f3)
        f5 = f4.view(f4.shape[0],-1)
        f6 = F.relu(self.fc1(f5))
        f7 = self.fc2(f6)
        return f7


class CatNet_wo_branch(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(64, 64, 1, 1)
        self.cn_f2 = BasicConv2d(64, 64, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base(x_p)
        x_c = self.base(x_c)

        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)

        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c

class CatNet_wo_kmm(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=2)

        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.relu = nn.ReLU()

        self.repeat_blocks = nn.Sequential(
            BasicBlock(64, 64),
            Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
        )

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        # self.KMM = KMM(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 64, 1, 1)
        self.cn_f2 = BasicConv2d(128, 64, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    # def base_forward(self,x):
    #     x1 = self.base(x)
    #
    #     x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x))) # 32x128x128
    #     x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1))) # 64x64x64
    #     x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2))) #64x32x32
    #     x2 = self.repeat_blocks(x_layer3) #64x32x32
    #     x = torch.cat((x1, x2), 1)
    #     return  x

    def base_forward(self, x):
        x1 = self.base(x)

        x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x)))  # 32x128x128
        x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1)))  # 64x64x64
        x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2)))  # 64x32x32
        x2 = self.repeat_blocks(x_layer3)  # 64x32x32
        x = torch.cat((x1, x2), 1)
        return x
    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base_forward(x_p)
        x_c = self.base_forward(x_c)

        k_p = x_p
        k_c = x_c
        # k_p = self.cn_f1(x_p)
        # k_c = self.cn_f1(x_c)
        # k_p = self.KMM(k_p)
        # k_c = self.KMM(k_c)

        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c




class CatNet_two_ffhq(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator_ffhq(pretrained).encoder

        self.conv1 = Conv2d(13, 32, kernel_size=7, stride=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=2)

        self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.relu = nn.ReLU()

        self.repeat_blocks = nn.Sequential(
            BasicBlock(64, 64),
            Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
            BasicBlock(64, 64),
            # Basic_attention_Block(64),
        )

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(64)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 64, 1, 1)
        self.cn_f2 = BasicConv2d(128, 64, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def base_forward(self,x):
        x1 = self.base(x)

        x_layer1 = self.base.relu(self.base.bn1(self.base.conv1(x))) # 32x128x128
        x_layer2 = self.base.relu(self.base.bn2(self.base.conv2(x_layer1))) # 64x64x64
        x_layer3 = self.base.relu(self.base.bn3(self.base.conv3(x_layer2))) #64x32x32
        x2 = self.repeat_blocks(x_layer3) #64x32x32
        x = torch.cat((x1, x2), 1)
        return  x
    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base_forward(x_p)
        x_c = self.base_forward(x_c)




        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)

        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c


class CatNet_kmm_combine(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(128)
        # self.KMM2 = KMM(128)
        # self.cn_f1 = BasicConv2d(128, 32, 1, 1)
        # self.cn_f2 = BasicConv2d(128, 32, 1, 1)
        self.conv = nn.Sequential(
            BasicConv2d(256,16,3,1),
            # nn.MaxPool2d(2, 2),
            # BasicConv2d(8, 2, 3, 1),
            # nn.MaxPool2d(2, 2),
            # BasicConv2d(2, 2, 6, 1)

        )
        self.conv2 = BasicConv2d(16,8,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3= BasicConv2d(8,2,6,1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base(x_p)
        x_c = self.base(x_c)

        # k_p = self.cn_f1(x_p)
        # k_c = self.cn_f1(x_c)
        x_p = self.KMM(x_p)
        x_c = self.KMM(x_c)
        f =  torch.cat((x_p,x_c),1)

        f = self.pool(self.conv(f))

        f = self.conv2(f)

        f = self.pool(f)

        f = self.conv3(f)

        # f = self.conv(f)
        return f.squeeze()


class CatNet_kmm_64(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(128)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 128, 1, 1)
        self.cn_f2 = BasicConv2d(128, 128, 1, 1)

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x_p,x_c,condition):
        if condition is not None:
            x_p=torch.cat((x_p,condition),1)
            x_c = torch.cat((x_c, condition), 1)

        x_p = self.base(x_p)
        x_c = self.base(x_c)

        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)



        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c





class CatNet_kmm_age(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.base = Generator(pretrained).encoder

        if pretrained:
            self.freeze(self.base)
            self.base.eval()
            print('get pretrained')

        self.KMM = KMM(128)
        # self.KMM2 = KMM(128)
        self.cn_f1 = BasicConv2d(128, 128, 1, 1)
        self.cn_f2 = BasicConv2d(128, 128, 1, 1)

        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((5,128, 128), dtype=np.float32)
        full_zero[4, :, :] = full_one
        self.label_p = torch.tensor(np.expand_dims(full_zero,axis=0)).cuda()
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((5,128, 128), dtype=np.float32)
        full_zero[0, :, :] = full_one
        self.label_c = torch.tensor(np.expand_dims(full_zero,axis=0)).cuda()

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x_p,x_c,condition):
        batch = x_p.size()[0]
        condition_p = self.label_p.repeat(batch, 1, 1, 1)
        condition_c = self.label_p.repeat(batch, 1, 1, 1)

        x_p=torch.cat((x_p,condition_p),1)
        x_c = torch.cat((x_c, condition_c), 1)

        x_p = self.base(x_p)
        x_c = self.base(x_c)

        k_p = self.cn_f1(x_p)
        k_c = self.cn_f1(x_c)
        k_p = self.KMM(k_p)
        k_c = self.KMM(k_c)



        k_p = k_p.view(k_p.shape[0], -1)
        k_c = k_c.view(k_c.shape[0], -1)

        return k_p,k_c




class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class _NonLocalBlockND_pair(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND_pair, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.g_2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W1 = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)

            )
            nn.init.constant_(self.W1[1].weight, 0)
            nn.init.constant_(self.W1[1].bias, 0)

            self.W2 = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W2[1].weight, 0)
            nn.init.constant_(self.W2[1].bias, 0)

        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g_1 = nn.Sequential(self.g_1, max_pool_layer)
            self.g_2 = nn.Sequential(self.g_2, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, fea1,fea2, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        # x = torch.cat((fea1,fea2))#

        x1 = fea1
        x2 = fea2

        batch_size = x1.size(0)

        g_x1 = self.g_1(fea1).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)

        g_x2 = self.g_2(fea2).view(batch_size, self.inter_channels, -1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)



        phi_x = self.phi(x2).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)



        y1 = torch.matmul(f_div_C, g_x1)
        y1 = y1.permute(0, 2, 1).contiguous()

        y2 = torch.matmul(f_div_C, g_x2)
        y2 = y2.permute(0, 2, 1).contiguous()

        y1 = y1.view(batch_size, self.inter_channels, *x1.size()[2:])
        W_y1 = self.W1(y1)
        z1 = W_y1 + fea1
        # z1 = W_y1
        y2 = y2.view(batch_size, self.inter_channels, *x2.size()[2:])
        W_y2 = self.W2(y2)
        z2 = W_y2 + fea2
        # z2 = W_y2
        # if return_nl_map:
        #     return z1, f_div_C
        return z1,z2


# class Comp_module_cos(nn.Module):
#     def __init__(self):
#         super(Comp_module_cos, self).__init__()
#         self.bn = nn.BatchNorm2d(128)
#         self.bn_f1 = nn.BatchNorm2d(128)
#         self.bn_f2 = nn.BatchNorm2d(128)
#         self.pooling = nn.MaxPool2d(2,2)
#         self.chanel_pool = nn.MaxPool3d((2,2,2))
#         self.se = SE_cos_Layer(128,16)
#         self.element_cos = nn.CosineSimilarity(dim=1)
#         self.channel_cos = nn.CosineSimilarity(dim=2)
#     def forward(self,fea1, fea2):
#
#         fea1 = self.pooling(fea1)
#
#         fea2 = self.pooling(fea2)
#         fea1 = self.bn_f1(fea1)
#         fea2 = self.bn_f2(fea2)
#         element_wise = self.element_cos(fea1,fea2)
#         element_wise = torch.unsqueeze(element_wise,dim=1).expand(fea1.size())
#         c_fea1 = fea1.view(fea1.size()[0],fea1.size()[1],-1)
#         c_fea2 = fea2.view(fea2.size()[0],fea2.size()[1],-1)
#         channel_wise = self.channel_cos(c_fea1,c_fea2)
#         # channel_wise = torch.unsqueeze(torch.unsqueeze(channel_wise,2),3).expand(fea1.size())
#         fea1,fea2 = self.se(element_wise,channel_wise,fea1,fea2)
#
#         return fea1, fea2


if __name__=="__main__":
        pass
    # label_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    # ])
    # full_zero = np.zeros((128, 128, 5), dtype=np.float32)
    # label = label_transforms(full_zero).unsqueeze(0)
    # img=torch.ones((1,3,128,128))
    # # condition=torch.ones((2,5,64,64))
    #
    # cat=catNet()
    # from torch.utils.tensorboard import SummaryWriter
    # print(cat(img,img,label).size())
    # with SummaryWriter(comment='Net1')as w:
    #     w.add_graph(cat, (img,img,label))
    # class Img_to_zero_center(object):
    #     def __int__(self):
    #         pass
    #
    #     def __call__(self, t_img):
    #         '''
    #         :param img:tensor be 0-1
    #         :return:
    #         '''
    #         t_img = (t_img - 0.5) * 2
    #         return t_img
    # pth = '../data/pretrained/IPCGANs/gepoch_6_iter_1000.pth'
    # net = CatNet_v1()
    #
    # pretrained_dict = torch.load(pth)
    # # step2: get model state_dict
    # model_dict = net.state_dict()
    # # step3: remove pretrained_dict params which is not in model_dict
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # step4: update model_dict using pretrained_dict
    # model_dict.update(pretrained_dict)
    # # step5: update model using model_dict
    # net.load_state_dict(model_dict)
    #
    # # print(net)
    # timg = pil_loader('./f36-2_2.jpg')
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((128, 128)),
    #     torchvision.transforms.ToTensor(),
    #     Img_to_zero_center()
    # ])
    #
    #
    # class Reverse_zero_center(object):
    #     def __init__(self):
    #         pass
    #
    #     def __call__(self, t_img):
    #         t_img = t_img / 2 + 0.5
    #         return t_img
    #
    # label_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    # ])
    # image = transforms(timg).unsqueeze(0)
    # full_one = np.ones((128, 128), dtype=np.float32)
    # full_zero = np.zeros((128, 128, 5), dtype=np.float32)
    # full_zero[:, :, 2] = full_one
    # label = label_transforms(full_zero).unsqueeze(0)
    # img = image.cuda()
    # lbl = label.cuda()
    # net.cuda()
    # # x,x1,x2 = net(img,img,lbl)
    # net.eval()
    # k_p,k_c = net(img,img,lbl)
    #
    # print(k_p)
    # # save_image(Reverse_zero_center()(gx),
    # #            filename="./generated4{}.jpg".format(2 + 1))