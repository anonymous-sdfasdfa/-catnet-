# # ########################################
# # copy from  https://github.com/guyuchao/IPCGANs-Pytorch
# ##########################################
# import torchvision
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# import os
# from torch.nn.functional import pad
# from torch.nn.parameter import Parameter
# import math
# from torch.nn.modules.utils import _single, _pair, _triple
# from torch.nn.modules import Module
# import numpy as np
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
# class BasicBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         return out
#
#
#
# class _ConvNd(Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, transposed, output_padding, groups, bias):
#         super(_ConvNd, self).__init__()
#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.transposed = transposed
#         self.output_padding = output_padding
#         self.groups = groups
#         if transposed:
#             self.weight = Parameter(torch.Tensor(
#                 in_channels, out_channels // groups, *kernel_size))
#         else:
#             self.weight = Parameter(torch.Tensor(
#                 out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def __repr__(self):
#         s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
#              ', stride={stride}')
#         if self.padding != (0,) * len(self.padding):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len(self.dilation):
#             s += ', dilation={dilation}'
#         if self.output_padding != (0,) * len(self.output_padding):
#             s += ', output_padding={output_padding}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         if self.bias is None:
#             s += ', bias=False'
#         s += ')'
#         return s.format(name=self.__class__.__name__, **self.__dict__)
#
# class Conv2d(_ConvNd):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(Conv2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias)
#
#     def forward(self, input):
#         return _conv2d_same_padding(input, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
# # custom conv2d, because pytorch don't have "padding='same'" option.
# def _conv2d_same_padding(input, weight, bias=None, stride=(1,1), padding=1, dilation=(1,1), groups=1):
#
#     input_rows = input.size(2)
#     filter_rows = weight.size(2)
#     effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
#     out_rows = (input_rows + stride[0] - 1) // stride[0]
#     padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
#                   input_rows)
#     padding_rows = max(0, (out_rows - 1) * stride[0] +
#                         (filter_rows - 1) * dilation[0] + 1 - input_rows)
#     rows_odd = (padding_rows % 2 != 0)
#     padding_cols = max(0, (out_rows - 1) * stride[0] +
#                         (filter_rows - 1) * dilation[0] + 1 - input_rows)
#     cols_odd = (padding_rows % 2 != 0)
#
#     if rows_odd or cols_odd:
#         input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
#
#     return F.conv2d(input, weight, bias, stride,
#                   padding=(padding_rows // 2, padding_cols // 2),
#                   dilation=dilation, groups=groups)
#
#
#
# class Basic_attention_Block(nn.Module):
#     """
#     this is the attention module before Residual structure
#     """
#
#     def __init__(self, channel, up_size=None):
#         """
#
#         :param channel: channels of input feature map
#         :param up_size: upsample size
#         """
#         super(Basic_attention_Block, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv = nn.Conv2d(channel, channel, 3, padding=1)
#         if up_size == None:
#             self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         else:
#             self.upsample = nn.Upsample(size=(up_size, up_size), mode='bilinear', align_corners=False)
#         self.sigmoid = nn.Sigmoid()
#         self.bn = nn.BatchNorm2d(channel)
#         self.relu = nn.ReLU(inplace=True)
#
#
#     def forward(self, x):
#         identity = x
#         x = self.pool(x)
#         x = self.conv(x)
#         x = self.upsample(x)
#         x = self.sigmoid(x)
#         x = torch.mul(identity, x)+identity
#         x = self.bn(x)
#         x = self.relu(x)
#
#         return x
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
#         self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
#         self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2)
#
#         self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
#         self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
#         self.bn3 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
#         self.relu = nn.ReLU()
#
#         self.repeat_blocks = nn.Sequential(
#             BasicBlock(128, 128),
#             Basic_attention_Block(128),
#             BasicBlock(128, 128),
#             Basic_attention_Block(128),
#             BasicBlock(128, 128),
#             Basic_attention_Block(128),
#         )
#
#         # self.repeat_blocks = self._make_repeat_blocks(BasicBlock(128, 128), 6)
#
#     def _make_repeat_blocks(self,block,repeat_times):
#         layers=[]
#         for i in range(repeat_times):
#             layers.append(block)
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.repeat_blocks(x)
#
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
#         self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
#         self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
#         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=1)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#
#     def forward(self,x):
#         x = self.deconv1(x)
#         x = self.relu(self.bn4(x))
#         x = self.deconv2(x)
#         x = self.relu(self.bn5(x))
#         x = self.tanh(self.conv4(x))
#         return x
#
#
# class Generator(nn.Module):
#     def __init__(self,pretrained=None):
#         super(Generator, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         if pretrained:
#             pth = pretrained
#             state_dict = torch.load(pth)
#             model_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#             model_dict.update(pretrained_dict)
#             self.load_state_dict(model_dict)
#
#     def forward(self, x,condition=None):
#         if condition is not None:
#             x=torch.cat((x,condition),1)
#
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
# #
# # class Generator(nn.Module):
# #     def __init__(self,pretrained = None):
# #         super().__init__()
# #         self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
# #         self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
# #         self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2)
# #         # self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
# #         self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
# #         self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
# #         self.bn3 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
# #         # self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
# #         # self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
# #         self.repeat_blocks=self._make_repeat_blocks(BasicBlock(128,128),6)
# #         # self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
# #         # self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0,output_padding=1)
# #         self.relu=nn.ReLU()
# #         # self.tanh=nn.Tanh()
# #
# #         if pretrained:
# #             pth = pretrained
# #             state_dict = torch.load(pth)
# #             model_dict = self.state_dict()
# #             pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
# #             model_dict.update(pretrained_dict)
# #             self.load_state_dict(model_dict)
# #
# #     def _make_repeat_blocks(self,block,repeat_times):
# #         layers=[]
# #         for i in range(repeat_times):
# #             layers.append(block)
# #         return nn.Sequential(*layers)
# #
# #
# #     def forward(self, x,condition=None):
# #         if condition is not None:
# #             x=torch.cat((x,condition),1)
# #
# #         x = self.relu(self.bn1(self.conv1(x)))
# #         x = self.relu(self.bn2(self.conv2(x)))
# #         x = self.relu(self.bn3(self.conv3(x)))
# #         x = self.repeat_blocks(x)# size 1,128,32,32
# #         # x = self.deconv1(x)
# #         # x = self.relu(self.bn4(x))
# #         # x = self.deconv2(x)
# #         # x = self.relu(self.bn5(x))
# #         # x = self.tanh(self.conv4(x))
# #         return x
#
#
# class Generator_ori(nn.Module):
#     def __init__(self):
#         super(Generator_ori, self).__init__()
#         self.conv1 = Conv2d(8, 32, kernel_size=7, stride=1)
#         self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2)
#         self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2)
#         self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
#         self.bn1 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
#         self.bn2 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
#         self.bn3 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
#         self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
#         self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
#         self.repeat_blocks=self._make_repeat_blocks(BasicBlock(128,128),6)
#         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0,output_padding=1)
#         self.relu=nn.ReLU()
#         self.tanh=nn.Tanh()
#
#         # if pretrained:
#         #     pth = pretrained
#         #     state_dict = torch.load(pth)
#         #     model_dict = self.state_dict()
#         #     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#         #     model_dict.update(pretrained_dict)
#         #     self.load_state_dict(model_dict)
#
#     def _make_repeat_blocks(self,block,repeat_times):
#         layers=[]
#         for i in range(repeat_times):
#             layers.append(block)
#         return nn.Sequential(*layers)
#
#     def forward(self, x,condition=None):
#         if condition is not None:
#             x=torch.cat((x,condition),1)
#
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x_em = self.repeat_blocks(x)# size 1,128,32,32
#         x = self.deconv1(x_em) # size 1,64,63,63
#         x = self.relu(self.bn4(x))
#         x = self.deconv2(x)
#         x = self.relu(self.bn5(x)) # size 1,32,128,128
#         x = self.tanh(self.conv4(x))
#         return x_em, x
#
# if __name__=='__main__':
#     # label_transforms = torchvision.transforms.Compose([
#     #     torchvision.transforms.ToTensor(),
#     # ])
#     # full_zero = np.zeros((128, 128, 5), dtype=np.float32)
#     # label = label_transforms(full_zero).unsqueeze(0)
#     # tensor=torch.ones((1,3,128,128))
#     # # condition=torch.ones((2,5,64,64))
#     #
#     # G=Generator()
#     # from torch.utils.tensorboard import SummaryWriter
#     # print(G(tensor,label).size())
#     # with SummaryWriter(comment='Net1')as w:
#     #     w.add_graph(G, (tensor,label))
#
#     print(Generator())