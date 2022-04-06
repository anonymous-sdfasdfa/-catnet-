import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from model.resnet import BasicBlock,Basic_attention_Block
from model.faceAlexnet import AgeAlexNet,AgeAlexNet_64,AgeAlexNet_ffhq
from utils.network import Conv2d #same padding
from utils.io_IPCGANS import Img_to_zero_center
import torchvision.models as torch_model

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = Conv2d(69, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = Conv2d(512, 512, kernel_size=4, stride=2)

    def forward(self, x,condition):
        x = self.lrelu(self.conv1(x))
        x=torch.cat((x,condition),1)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class PatchDiscriminator_ffhq(nn.Module):
    def __init__(self):
        super(PatchDiscriminator_ffhq, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = Conv2d(74, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = Conv2d(512, 512, kernel_size=4, stride=2)

    def forward(self, x,condition):
        x = self.lrelu(self.conv1(x))
        x=torch.cat((x,condition),1)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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

        # self.repeat_blocks = self._make_repeat_blocks(BasicBlock(128, 128), 6)

    def _make_repeat_blocks(self,block,repeat_times):
        layers=[]
        for i in range(repeat_times):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.repeat_blocks(x)

        return x

class Encoder_ffhq(nn.Module):
    def __init__(self):
        super(Encoder_ffhq, self).__init__()
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

        # self.repeat_blocks = self._make_repeat_blocks(BasicBlock(128, 128), 6)

    def _make_repeat_blocks(self,block,repeat_times):
        layers=[]
        for i in range(repeat_times):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.repeat_blocks(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(32, 3, kernel_size=7, stride=1)
        self.bn4 = nn.BatchNorm2d(64, eps=0.001, track_running_stats=True)
        self.bn5 = nn.BatchNorm2d(32, eps=0.001, track_running_stats=True)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.deconv1(x)
        x = self.relu(self.bn4(x))
        x = self.deconv2(x)
        x = self.relu(self.bn5(x))
        x = self.tanh(self.conv4(x))
        return x


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     def forward(self, x,condition=None):
#         if condition is not None:
#             x=torch.cat((x,condition),1)
#
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class Generator(nn.Module):
    def __init__(self,pretrained=None):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained:
            pth = pretrained
            state_dict = torch.load(pth)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x,condition=None):
        if condition is not None:
            x=torch.cat((x,condition),1)

        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Generator_ffhq(nn.Module):
    def __init__(self,pretrained=None):
        super(Generator_ffhq, self).__init__()
        self.encoder = Encoder_ffhq()
        self.decoder = Decoder()
        if pretrained:
            pth = pretrained
            state_dict = torch.load(pth)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x,condition=None):
        if condition is not None:
            x=torch.cat((x,condition),1)

        x = self.encoder(x)
        x = self.decoder(x)
        return x





# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
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
#         x = self.repeat_blocks(x)
#         x = self.deconv1(x)
#         x = self.relu(self.bn4(x))
#         x = self.deconv2(x)
#         x = self.relu(self.bn5(x))
#         x = self.tanh(self.conv4(x))
#         return x

class feature_alexnet(nn.Module):
    def __init__(self):
        super(feature_alexnet, self).__init__()
        self.net = torch_model.alexnet(pretrained=True)
    def forward(self,x):
        self.fea1 = self.net.features(x)
        self.fea2 = self.net.avgpool(self.fea1)



class feature_inception3(nn.Module):
    def __init__(self):
        super(feature_inception3, self).__init__()
        self.inception = torch_model.inception_v3(pretrained=True).cuda()

    def forward(self, x):
        if self.inception.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        self.fea1 = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        # if self.inception.training and self.inception.aux_logits:
        #     aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(self.fea1)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        self.fea2 = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(self.fea2, (1, 1))
        # # N x 2048 x 1 x 1
        # x = F.dropout(x, training=self.training)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)

        # return x



class IPCGANs_cyc:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator().cuda()
        self.discriminator=PatchDiscriminator().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet(pretrainded=False).cuda()
        self.feature_net = feature_inception3().cuda()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label,source_label_128):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source = self.generator(source_img_128, condition=true_label_128)

        g_cyc = self.generator(self.g_source, condition=source_label_128)

        # real img, right age label
        # logit means prob which hasn't been normalized

        # d1 logit ,discriminator 1 means true,0 means false.
        d1_logit = self.discriminator(true_label_img, condition=true_label_64)

        d1_real_loss = self.MSEloss(d1_logit, torch.ones((d1_logit.size())).cuda())

        # real img, false label
        d2_logit = self.discriminator(true_label_img, condition=fake_label_64)
        d2_fake_loss = self.MSEloss(d2_logit, torch.zeros((d1_logit.size())).cuda())

        # fake img,real label
        d3_logit = self.discriminator(self.g_source, condition=true_label_64)
        d3_fake_loss = self.MSEloss(d3_logit, torch.zeros((d1_logit.size())).cuda())  # use this for discriminator
        d3_real_loss = self.MSEloss(d3_logit, torch.ones((d1_logit.size())).cuda())  # use this for genrator

        self.d_loss = (1. / 2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight
        g_loss = (1. / 2 * d3_real_loss) * self.gan_loss_weight

        ################################feature_loss#############################

        # self.age_classifier(source_img_227)
        # source_feature=self.age_classifier.conv5_feature
        self.feature_net(source_img_227)
        source_feature_cyc = self.feature_net.fea1
        source_feature = self.feature_net.fea1

        generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(generate_img_227)

        # self.age_classifier(generate_img_227)
        # generate_feature =self.age_classifier.conv5_feature

        # generate_feature = self.feature_net(generate_img_227)

        self.feature_net(generate_img_227)
        generate_feature = self.feature_net.fea1

        g_cyc_227 = F.interpolate(g_cyc, (227, 227), mode="bilinear", align_corners=True)
        g_cyc_227 = Img_to_zero_center()(g_cyc_227)
        self.feature_net(g_cyc_227)
        cyc_feature = self.feature_net.fea1

        self.feature_loss = self.feature_loss_weight * self.MSEloss(source_feature, generate_feature)

        ################################ recontract_loss##############################

        self.cyc_loss = self.feature_loss_weight * self.MSEloss(source_feature_cyc, cyc_feature)

        ################################age_cls_loss##############################

        age_logit = self.age_classifier(generate_img_227)
        self.age_loss = self.CrossEntropyLoss(age_logit, age_label) * self.age_loss_weight

        self.g_loss = self.age_loss + g_loss + 1/3*(self.feature_loss + 2*self.cyc_loss)


class IPCGANs_cyc_ffhq:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator_ffhq().cuda()
        self.discriminator=PatchDiscriminator_ffhq().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet_ffhq(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet_ffhq(pretrainded=False).cuda()
        self.feature_net = feature_alexnet().cuda()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label,source_label_128):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source = self.generator(source_img_128, condition=true_label_128)

        g_cyc = self.generator(self.g_source, condition=source_label_128)

        # real img, right age label
        # logit means prob which hasn't been normalized

        # d1 logit ,discriminator 1 means true,0 means false.
        d1_logit = self.discriminator(true_label_img, condition=true_label_64)

        d1_real_loss = self.MSEloss(d1_logit, torch.ones((d1_logit.size())).cuda())

        # real img, false label
        d2_logit = self.discriminator(true_label_img, condition=fake_label_64)
        d2_fake_loss = self.MSEloss(d2_logit, torch.zeros((d1_logit.size())).cuda())

        # fake img,real label
        d3_logit = self.discriminator(self.g_source, condition=true_label_64)
        d3_fake_loss = self.MSEloss(d3_logit, torch.zeros((d1_logit.size())).cuda())  # use this for discriminator
        d3_real_loss = self.MSEloss(d3_logit, torch.ones((d1_logit.size())).cuda())  # use this for genrator

        self.d_loss = (1. / 2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight
        g_loss = (1. / 2 * d3_real_loss) * self.gan_loss_weight

        ################################feature_loss#############################

        # self.age_classifier(source_img_227)
        # source_feature=self.age_classifier.conv5_feature
        self.feature_net(source_img_227)
        source_feature_cyc = self.feature_net.fea1
        source_feature = self.feature_net.fea1

        generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(generate_img_227)

        # self.age_classifier(generate_img_227)
        # generate_feature =self.age_classifier.conv5_feature

        # generate_feature = self.feature_net(generate_img_227)

        self.feature_net(generate_img_227)
        generate_feature = self.feature_net.fea1

        g_cyc_227 = F.interpolate(g_cyc, (227, 227), mode="bilinear", align_corners=True)
        g_cyc_227 = Img_to_zero_center()(g_cyc_227)
        self.feature_net(g_cyc_227)
        cyc_feature = self.feature_net.fea1

        self.feature_loss = self.feature_loss_weight * self.MSEloss(source_feature, generate_feature)

        ################################ recontract_loss##############################

        self.cyc_loss = self.feature_loss_weight * self.MSEloss(source_feature_cyc, cyc_feature)

        ################################age_cls_loss##############################

        age_logit = self.age_classifier(generate_img_227)
        self.age_loss = self.CrossEntropyLoss(age_logit, age_label) * self.age_loss_weight

        self.g_loss = self.age_loss + g_loss + 1/3*(self.feature_loss + 2*self.cyc_loss)



class IPCGANs:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator().cuda()
        self.discriminator=PatchDiscriminator().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet(pretrainded=False).cuda()
        self.feature_net = feature_alexnet()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source=self.generator(source_img_128,condition=true_label_128)

        #real img, right age label
        #logit means prob which hasn't been normalized

        #d1 logit ,discriminator 1 means true,0 means false.
        d1_logit=self.discriminator(true_label_img,condition=true_label_64)

        d1_real_loss=self.MSEloss(d1_logit,torch.ones((d1_logit.size())).cuda())

        #real img, false label
        d2_logit=self.discriminator(true_label_img,condition=fake_label_64)
        d2_fake_loss=self.MSEloss(d2_logit,torch.zeros((d1_logit.size())).cuda())

        #fake img,real label
        d3_logit=self.discriminator(self.g_source,condition=true_label_64)
        d3_fake_loss=self.MSEloss(d3_logit,torch.zeros((d1_logit.size())).cuda())#use this for discriminator
        d3_real_loss=self.MSEloss(d3_logit,torch.ones((d1_logit.size())).cuda())#use this for genrator

        self.d_loss=(1./2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight

        g_loss=(1./2*d3_real_loss)*self.gan_loss_weight


        ################################feature_loss#############################

        # self.age_classifier(source_img_227)
        # source_feature=self.age_classifier.conv5_feature
        source_feature = self.feature_net(source_img_227)

        generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(generate_img_227)

        # self.age_classifier(generate_img_227)
        # generate_feature =self.age_classifier.conv5_feature

        generate_feature = self.feature_net(generate_img_227)
        self.feature_loss=self.feature_loss_weight*self.MSEloss(source_feature,generate_feature)

        ################################age_cls_loss##############################



        age_logit=self.age_classifier(generate_img_227)
        self.age_loss=self.CrossEntropyLoss(age_logit,age_label)*self.age_loss_weight

        self.g_loss=self.age_loss+g_loss+self.feature_loss




class IPCGANs_64:
    def __init__(self,lr=0.01,age_classifier_path=None,gan_loss_weight=75,feature_loss_weight=0.5e-4,age_loss_weight=30):

        self.d_lr=lr
        self.g_lr=lr

        self.generator=Generator().cuda()
        self.discriminator=PatchDiscriminator().cuda()
        if age_classifier_path is not None:
            self.age_classifier=AgeAlexNet_64(pretrainded=True,modelpath=age_classifier_path).cuda()
        else:
            self.age_classifier = AgeAlexNet_64(pretrainded=False).cuda()
        self.feature_net = feature_alexnet()
        self.MSEloss=nn.MSELoss().cuda()
        self.CrossEntropyLoss=nn.CrossEntropyLoss().cuda()

        self.gan_loss_weight=gan_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.age_loss_weight=age_loss_weight

        self.d_optim = torch.optim.Adam(self.discriminator.parameters(),self.d_lr,betas=(0.5,0.99))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.99))

    def save_model(self,dir,filename):
        torch.save(self.generator.state_dict(),os.path.join(dir,"g"+filename))
        torch.save(self.discriminator.state_dict(),os.path.join(dir,"d"+filename))

    def load_generator_state_dict(self,state_dict):
        pretrained_dict = state_dict
        # step2: get model state_dict
        model_dict = self.generator.state_dict()
        # step3: remove pretrained_dict params which is not in model_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # step4: update model_dict using pretrained_dict
        model_dict.update(pretrained_dict)
        # step5: update model using model_dict
        self.generator.load_state_dict(model_dict)

    def test_generate(self,source_img_128,condition):
        self.generator.eval()
        with torch.no_grad():
            generate_image=self.generator(source_img_128,condition)
        return generate_image

    def cuda(self):
        self.generator=self.generator.cuda()

    def train(self,source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,\
               fake_label_64, age_label):
        '''

        :param source_img_227: use this img to extract conv5 feature
        :param source_img_128: use this img to generate face in target age
        :param true_label_img:
        :param true_label_128:
        :param true_label_64:
        :param fake_label_64:
        :param age_label:
        :return:
        '''

        ###################################gan_loss###############################
        self.g_source=self.generator(source_img_128,condition=true_label_128)

        #real img, right age label
        #logit means prob which hasn't been normalized

        #d1 logit ,discriminator 1 means true,0 means false.
        d1_logit=self.discriminator(true_label_img,condition=true_label_64)

        d1_real_loss=self.MSEloss(d1_logit,torch.ones((d1_logit.size())).cuda())

        #real img, false label
        d2_logit=self.discriminator(true_label_img,condition=fake_label_64)
        d2_fake_loss=self.MSEloss(d2_logit,torch.zeros((d1_logit.size())).cuda())

        #fake img,real label
        d3_logit=self.discriminator(self.g_source,condition=true_label_64)
        d3_fake_loss=self.MSEloss(d3_logit,torch.zeros((d1_logit.size())).cuda())#use this for discriminator
        d3_real_loss=self.MSEloss(d3_logit,torch.ones((d1_logit.size())).cuda())#use this for genrator

        self.d_loss=(1./2 * (d1_real_loss + 1. / 2 * (d2_fake_loss + d3_fake_loss))) * self.gan_loss_weight
        g_loss=(1./2*d3_real_loss)*self.gan_loss_weight


        ################################feature_loss#############################

        # self.age_classifier(source_img_227)
        # source_feature=self.age_classifier.conv5_feature
        source_feature = self.feature_net(source_img_227)

        # generate_img_227 = F.interpolate(self.g_source, (227, 227), mode="bilinear", align_corners=True)
        generate_img_227 = Img_to_zero_center()(self.g_source)

        # self.age_classifier(generate_img_227)
        # generate_feature =self.age_classifier.conv5_feature

        generate_feature = self.feature_net(generate_img_227)
        self.feature_loss=self.feature_loss_weight*self.MSEloss(source_feature,generate_feature)

        ################################age_cls_loss##############################



        age_logit=self.age_classifier(generate_img_227)
        self.age_loss=self.CrossEntropyLoss(age_logit,age_label)*self.age_loss_weight

        self.g_loss=self.age_loss+g_loss+self.feature_loss


if __name__=="__main__":
    tensor=torch.ones((2,3,128,128)).cuda()
    condition=torch.ones((2,5,64,64)).cuda()

    discriminator=PatchDiscriminator().cuda()
    from torch.utils.tensorboard import SummaryWriter
    print(discriminator(tensor,condition).size())
    with SummaryWriter(comment='Net1')as w:
        w.add_graph(discriminator, (tensor,condition))

