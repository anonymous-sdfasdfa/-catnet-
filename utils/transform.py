import torchvision.transforms as transforms
import numpy as np
import torch
import os
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
############################## ipcgan

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def tensor2pil(tensor,norm="zero_center"):
    assert len(tensor.size())==3 and tensor.size()[0]==3,"need to be [3,h,w] tensor"
    assert norm is "zero_center" or norm is "normalization" or norm is None,"only support zero_center or normalization"
    if norm is "zero_center":
        tensor=(tensor+1)*127.5
    np_tensor=tensor.detach().cpu().numpy().transpose(1,2,0)
    return Image.fromarray(np_tensor.astype(np.uint8))

class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''
        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img

class Reverse_zero_center(object):
    def __init__(self):
        pass
    def __call__(self,t_img):
        t_img=t_img/2+0.5
        return t_img

#########################



def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y

#################################### full transform
# train_full_transform = transforms.Compose(
#     [
#      transforms.Resize((64,64)),
#      transforms.ColorJitter(brightness=0.3,
#                             contrast=0.3,
#                             saturation=0.3,
#                             hue=0.3
#                             ),
#      # transforms.CenterCrop((64,64)),
#      # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
#      transforms.RandomGrayscale(),
#      transforms.RandomHorizontalFlip(p=0.4),
#      transforms.RandomPerspective(distortion_scale=0.2,p=0.3),
#      transforms.RandomResizedCrop(size=(64,64), scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
#
#      transforms.ToTensor(),
#      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#      transforms.Lambda(lambda crops: torch.unsqueeze(crops, 0)),
#      # transforms.RandomErasing()
#      ])
#
#
# test_full_transform = transforms.Compose(
#     [
#         transforms.Resize((64,64)),
#         # transforms.Resize((73,73)),
#         # transforms.CenterCrop((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         transforms.Lambda(lambda crops: torch.unsqueeze(crops, 0)),
#
#     ])
#
#
#
# crop_full_transform = transforms.Compose(
#     [
#         transforms.Resize((64,64)),
#         transforms.FiveCrop((45,45)),
#         transforms.Lambda(lambda crops: torch.stack(
#                [mid_transform(crop) for crop in
#                 crops])),
#
#     ])
#
# mid_transform = transforms.Compose([
#         transforms.Resize((64,64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
#



##################################### cat net transform
#todo: add  Img_to_zero_center

class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''
        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img

train_cat_transform = transforms.Compose(
    [
     transforms.Resize((128,128)),
     # transforms.RandomCrop(64,64),
     # transforms.ColorJitter(brightness=0.3,
     #                        contrast=0.3,
     #                        saturation=0.3,
     #                        hue=0.3
     #                        ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
     # transforms.RandomResizedCrop(size=(128,128), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
     # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),

        # transforms.RandomErasing(),
     # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     Img_to_zero_center(),
     # transforms.RandomErasing()
     ])

train_cat_transform_64 = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.RandomCrop(64,64),
     transforms.ColorJitter(brightness=0.05,
                            contrast=0.05,
                            saturation=0.05,
                            hue=0.05
                            ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     # transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
     # transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
     # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),

        # transforms.RandomErasing(),
     # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     Img_to_zero_center(),
     # transforms.RandomErasing()
     ])


test_cat_transform = transforms.Compose(
    [
     # transforms.Resize((64,64)),           #### changed
     transforms.Resize((128,128)),

     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),

        # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    Img_to_zero_center()
    ]
)

test_cat_transform_64 = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.Resize((128,128)),

     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),

        # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    Img_to_zero_center()
    ]
)
train_kin__transform = transforms.Compose(
    [
     transforms.Resize((128,128)),
     transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               saturation=0.3,
                               hue=0.3
                               ),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     Img_to_zero_center(),
     transforms.RandomErasing()
     ])



test_kin_transform = transforms.Compose(
    [
     transforms.Resize((128,128)),
     transforms.ToTensor(),
    Img_to_zero_center()
    ]
)

label_transforms = transforms.Compose([
    transforms.ToTensor(),
])


#################################### kinFaceW transform
train_transform = transforms.Compose(
	[
	 transforms.Resize((73,73)),
	 # transforms.ColorJitter(brightness=0.3,
	 #                        contrast=0.3,
	 #                        saturation=0.3,
	 #                        hue=0.3
	 #                        ),
	 # transforms.CenterCrop((64,64)),
	 # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
	 # transforms.RandomGrayscale(),
	 transforms.RandomHorizontalFlip(),
	 transforms.RandomCrop((64,64)),
	 # transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
	 # transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
	 # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),

		# transforms.RandomErasing(),
	 # transforms.Grayscale(num_output_channels=3),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	 # transforms.RandomErasing()
	 ])


test_transform = transforms.Compose(
	[
	 transforms.Resize((64,64)),
	 # transforms.Resize((73,73)),
	 # transforms.CenterCrop((64, 64)),

		# transforms.Grayscale(num_output_channels=3),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# train_transform = transforms.Compose(
#     [
#      transforms.Resize((64,64)),
#      # transforms.RandomCrop(64,64),
#      transforms.ColorJitter(brightness=0.3,
#                             contrast=0.3,
#                             saturation=0.3,
#                             hue=0.3
#                             ),
#      # transforms.CenterCrop((64,64)),
#      # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
#      transforms.RandomGrayscale(),
#      transforms.RandomHorizontalFlip(p=0.4),
#      transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
#      transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
#      # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
#
#         # transforms.RandomErasing(),
#      # transforms.Grayscale(num_output_channels=3),
#      transforms.ToTensor(),
#      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#      transforms.RandomErasing()
#      ])
#
#
# test_transform = transforms.Compose(
#     [
#      transforms.Resize((64,64)),
#      # transforms.Resize((73,73)),
#      # transforms.CenterCrop((64, 64)),
#
#         # transforms.Grayscale(num_output_channels=3),
#      transforms.ToTensor(),
#      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


######################################### faceNet transform

facenet_testrans = transforms.Compose(
     [
     transforms.Resize((160,160)),
     np.float32,
     transforms.ToTensor(),
     prewhiten
     # transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# facenet_trans = transforms.Compose(
#      [
#      transforms.Resize((160,160)),
#      transforms.ColorJitter(brightness=0.3,
#                             contrast=0.3,
#                             saturation=0.3,
#                             hue=0.3
#                             ),
#      # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
#      transforms.RandomGrayscale(),
#      transforms.RandomHorizontalFlip(p=0.2),
#      transforms.RandomPerspective(distortion_scale=0.05,p=0.2),
#      transforms.RandomResizedCrop(size=(160,160), scale=(0.98, 1.02), ratio=(0.98, 1.02), interpolation=2),
#      np.float32,
#      transforms.ToTensor(),
#      # transforms.RandomErasing(),
#      prewhiten
#      # transforms.ToTensor(),
#      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#      ])



#
############################################# attention transform




#
# def get_transform(name):
#     if name =='train_trans':
#         return train_transform
#     elif name =='test_trans':
#         return test_transform
#     elif name =='facenet_train_trans':
#         return facenet_trans
#     elif name =='facenet_test_trans':
#         return facenet_testrans







train_atten_transform = transforms.Compose(
	[
	 transforms.Resize((73,73)),
	 transforms.RandomHorizontalFlip(),
	 transforms.RandomCrop((64,64)),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

	 ])


test_atten_transform = transforms.Compose(
	[
	 transforms.Resize((64,64)),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_cnn_transform = transforms.Compose(
	[
	 transforms.Resize((64,64)),
	 transforms.RandomHorizontalFlip(),
	 # transforms.RandomCrop((64,64)),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

	 ])


test_cnn_transform = transforms.Compose(
	[
	 transforms.Resize((64,64)),
	 transforms.ToTensor(),
	 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

