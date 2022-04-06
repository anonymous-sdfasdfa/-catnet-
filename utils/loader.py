import scipy.io
import os
import torch
from torch.utils.data import Dataset,DataLoader
import random
from random import shuffle
import pickle
from utils.transform import train_transform,label_transforms
from PIL import Image
import copy
import numpy as np

class cross_validation():
    """
    get  n-folds cross validation,
    generate [1,2,3,4,...n]
    yeild [1,..del(remove),..n] and [remove]
    """
    def __init__(self,n):
        self.n = n

    def __iter__(self):
        for i in range(self.n,0,-1):
            train_ls = self._tra_ls(i)
            yield train_ls, [i]

    def _tra_ls(self,remove):
        return [i for i in range(1,self.n+1) if i !=remove]


class Nemo_CA_Dataset(Dataset):
    """
    Nemo-children-adult dataset loader
    """
    def __init__(self,label_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        :param label_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle training list after each epoch
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param sf_fimg1:      whether fix img1 position while shuffling neg pairs
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_label(label_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.alln_list,self.img2_list = self.get_names(self.kin_list)
        self.sf_aln = sf_aln
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)


    def get_names(self,ls):
        img2_name = []
        all_name = []
        for i in ls:
            if i[1]==1:
                all_name.append(i[2])
                all_name.append(i[3])
                img2_name.append(i[3])
        return all_name,img2_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.test:
            # extract img1
            img1_n = self.kin_list[item][2]+'.jpg'
            img1_path = os.path.join(self.im_root,img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3]+'.jpg'
            img2_path = os.path.join(self.im_root,img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1,img2))

            return imgs,kin,img1_n,img2_n
        else:
            # extract img1
            img1_n = self.kin_list[item][2] + '.jpg'
            img1_path = os.path.join(self.im_root, img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3] + '.jpg'
            img2_path = os.path.join(self.im_root, img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

            return imgs, kin, img1_n, img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """

        new_pair_list = []

        if self.sf_fimg1: # fix img1's position when cross shuffling
            if self.sf_aln:
                neg_ls = self.alln_list
            else:
                neg_ls = self.img2_list
            rand_lth = len(neg_ls)

            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)
        else:
            neg_ls = self.alln_list
            rand_lth = len(neg_ls)
            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img1 = neg_ls[random.randint(0, rand_lth-1)]
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[2] = new_img1
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_label(self,label_path):
        with open (label_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        return nemo_ls


    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]

class KinDataset(Dataset):
    # def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
    #              sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False):
    def __init__(self, mat_path, im_root, cross_vali=None, transform=None,
                 sf_sequence=False, cross_shuffle=False, sf_aln=False, test=False, test_each=False, real_sn=False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_mat(mat_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)


    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_n = self.kin_list[item][2]
        img1_path = os.path.join(self.im_root,img1_n)
        img1 = Image.open(img1_path)
        # extract img2
        img2_n = self.kin_list[item][3]
        img2_path = os.path.join(self.im_root,img2_n)
        img2 = Image.open(img2_path)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout +=1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle()


        return imgs,kin,img1_n,img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        # if self.crs_insect%2 ==0:
        #     ls_bak = copy.deepcopy(self.kin_list_bak)
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)
        # else:
        #     for pair_l in self.kin_list:
        #         if pair_l[1] == 0:
        #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
        #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
        #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             pair_l[3] = new_img2
        #             pair_l[2] = new_img1
        #         new_pair_list.append(pair_l)
        # infact it's no use to use this line:
        # self.crs_insect +=1
        self.kin_list = new_pair_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]



class Nemo_catnet_Dataset(Dataset):
    """
    Nemo-children-adult dataset loader
    """
    def __init__(self,label_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        :param label_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle training list after each epoch
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param sf_fimg1:      whether fix img1 position while shuffling neg pairs
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_label(label_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.alln_list,self.img2_list = self.get_names(self.kin_list)
        self.sf_aln = sf_aln
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 5), dtype=np.float32)
        full_zero[:, :, 3] = full_one
        self.label = label_transforms(full_zero)

    def get_names(self,ls):
        img2_name = []
        all_name = []
        for i in ls:
            if i[1]==1:
                all_name.append(i[2])
                all_name.append(i[3])
                img2_name.append(i[3])
        return all_name,img2_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.test:
            # extract img1
            img1_n = self.kin_list[item][2]+'.jpg'
            img1_path = os.path.join(self.im_root,img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3]+'.jpg'
            img2_path = os.path.join(self.im_root,img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1,img2))

            return img1,img2,kin,self.label,img1_n,img2_n
        else:
            # extract img1
            img1_n = self.kin_list[item][2] + '.jpg'
            img1_path = os.path.join(self.im_root, img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3] + '.jpg'
            img2_path = os.path.join(self.im_root, img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1, img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

            return img1,img2, kin,self.label, img1_n, img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """

        new_pair_list = []

        if self.sf_fimg1: # fix img1's position when cross shuffling
            if self.sf_aln:
                neg_ls = self.alln_list
            else:
                neg_ls = self.img2_list
            rand_lth = len(neg_ls)

            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)
        else:
            neg_ls = self.alln_list
            rand_lth = len(neg_ls)
            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img1 = neg_ls[random.randint(0, rand_lth-1)]
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[2] = new_img1
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_label(self,label_path):
        with open (label_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        return nemo_ls


    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]

class Nemo_catnet_Dataset_ffhq(Dataset):
    """
    Nemo-children-adult dataset loader
    """
    def __init__(self,label_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        :param label_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle training list after each epoch
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param sf_fimg1:      whether fix img1 position while shuffling neg pairs
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_label(label_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.alln_list,self.img2_list = self.get_names(self.kin_list)
        self.sf_aln = sf_aln
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 10), dtype=np.float32)
        full_zero[:, :, 6] = full_one
        self.label = label_transforms(full_zero)

    def get_names(self,ls):
        img2_name = []
        all_name = []
        for i in ls:
            if i[1]==1:
                all_name.append(i[2])
                all_name.append(i[3])
                img2_name.append(i[3])
        return all_name,img2_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.test:
            # extract img1
            img1_n = self.kin_list[item][2]+'.jpg'
            img1_path = os.path.join(self.im_root,img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3]+'.jpg'
            img2_path = os.path.join(self.im_root,img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1,img2))

            return img1,img2,kin,self.label,img1_n,img2_n
        else:
            # extract img1
            img1_n = self.kin_list[item][2] + '.jpg'
            img1_path = os.path.join(self.im_root, img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3] + '.jpg'
            img2_path = os.path.join(self.im_root, img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1, img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

            return img1,img2, kin,self.label, img1_n, img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """

        new_pair_list = []

        if self.sf_fimg1: # fix img1's position when cross shuffling
            if self.sf_aln:
                neg_ls = self.alln_list
            else:
                neg_ls = self.img2_list
            rand_lth = len(neg_ls)

            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)
        else:
            neg_ls = self.alln_list
            rand_lth = len(neg_ls)
            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img1 = neg_ls[random.randint(0, rand_lth-1)]
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[2] = new_img1
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_label(self,label_path):
        with open (label_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        return nemo_ls


    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]

#################################################################
# class Kin_fivecrop_Dataset(Dataset):
#     def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
#                  sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False):
#         """
#         add five crop data augmentation
#         """
#
#         # kin_list is the whole 1,2,3,4,5 folds from mat
#         self.kin_list = self._read_mat(mat_path)
#         self.im_root = im_root
#         self.transform = transform
#         self.cout = 0
#         if cross_vali is not None:
#             #extract matched folds e.g. [1,2,4,5]
#             self.kin_list = self._get_cross(cross_vali)
#         # if cross_shuffle:
#         self.crf = cross_shuffle  # cross shuffle
#         self.lth = len(self.kin_list)
#         # store all img2/ all_(img1+img2) list for cross shuffle,
#         self.img2_list = [i[3] for i in self.kin_list]
#         self.alln_list = self.get_alln(self.kin_list)
#         self.sf_aln = sf_aln
#         # self.trf_ct = False # transform count
#         self.test = test
#         # self.crs_insect = 0
#         # self.kin_list_bak= copy.deepcopy(self.kin_list)
#
#
#     def get_alln(self,ls):
#         all_name = []
#         for i in ls:
#             all_name.append(i[2])
#             all_name.append(i[3])
#         return all_name
#
#     def __len__(self):
#         return len(self.kin_list)
#
#     def __getitem__(self, item):
#
#         if torch.is_tensor(item):
#             item = item.tolist()
#         # extract img1
#         img1_n = self.kin_list[item][2]
#         img1_path = os.path.join(self.im_root,img1_n)
#         img1 = Image.open(img1_path)
#         # extract img2
#         img2_n = self.kin_list[item][3]
#         img2_path = os.path.join(self.im_root,img2_n)
#         img2 = Image.open(img2_path)
#         # get kin label 0/1
#         kin = self.kin_list[item][1]
#
#         if self.transform:
#             if self.test:
#                 img1 = self.transform(img1)
#                 img2 = self.transform(img2)
#                 kin = torch.unsqueeze(torch.tensor(kin), 0)
#             else:
#                 # self.trf_ct +=1
#                 # if self.trf_ct:
#                 img1_crop = crop_full_transform(img1)
#                 img2_crop = crop_full_transform(img2)
#                 # kin = torch.tensor((kin,kin,kin,kin,kin))
#                 # else:
#                 img1_tran = self.transform(img1)
#                 img2_tran = self.transform(img2)
#                 img1 = torch.cat((img1_crop,img1_tran,img1_tran,img1_tran,img1_tran,img1_tran),dim=0)
#                 img2 = torch.cat((img2_crop,img2_tran,img2_tran,img2_tran,img2_tran,img2_tran),dim=0)
#                 kin = torch.tensor((kin,kin,kin,kin,kin,kin,kin,kin,kin,kin))
#                 # kin = torch.unsqueeze(torch.tensor(kin),0)
#         imgs = torch.cat((img1,img2),dim=1)
#         # after each epoch, shuffle once
#         if self.crf:
#             self.cout +=1
#             if self.cout == self.lth:
#                 self.cout = 0
#                 # self.trf_ct = not self.trf_ct
#                 self._cross_shuffle()
#
#
#         return imgs,kin,img1_n,img2_n
#
#     def _cross_shuffle(self):
#         """
#         shuffle the second images name after each epoch
#         :return:
#         """
#         if self.sf_aln:
#             im2_ls = self.alln_list
#         else:
#             im2_ls = self.img2_list
#         rand_lth = len(im2_ls)
#         new_pair_list = []
#         # if self.crs_insect%2 ==0:
#         #     ls_bak = copy.deepcopy(self.kin_list_bak)
#         for pair_l in self.kin_list:
#             if pair_l[1] == 0:
#                 new_img2 = im2_ls[random.randint(0, rand_lth-1)]
#                 while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
#                     new_img2 = im2_ls[random.randint(0, rand_lth-1)]
#                 pair_l[3] = new_img2
#             new_pair_list.append(pair_l)
#         # else:
#         #     for pair_l in self.kin_list:
#         #         if pair_l[1] == 0:
#         #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
#         #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
#         #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
#         #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
#         #             pair_l[3] = new_img2
#         #             pair_l[2] = new_img1
#         #         new_pair_list.append(pair_l)
#         # infact it's no use to use this line:
#         # self.crs_insect +=1
#         self.kin_list = new_pair_list
#
#     def _read_mat(self,mat_path):
#         mat = scipy.io.loadmat(mat_path)
#         conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
#         pair_list = [conv_type(ls) for ls in mat['pairs']]
#         return pair_list
#
#     def _get_cross(self,cross):
#
#         return [i for i in self.kin_list if i[0] in cross]
class Nemo_catnet_Dataset_age0(Dataset):
    """
    Nemo-children-adult dataset loader
    """
    def __init__(self,label_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        :param label_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle training list after each epoch
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param sf_fimg1:      whether fix img1 position while shuffling neg pairs
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_label(label_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.alln_list,self.img2_list = self.get_names(self.kin_list)
        self.sf_aln = sf_aln
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 5), dtype=np.float32)
        full_zero[:, :, 2] = full_one
        self.label_p = label_transforms(full_zero)
        c_zero = np.zeros((128, 128, 5), dtype=np.float32)
        c_zero[:, :, 0] = full_one
        self.label_c = label_transforms(c_zero)


    def get_names(self,ls):
        img2_name = []
        all_name = []
        for i in ls:
            if i[1]==1:
                all_name.append(i[2])
                all_name.append(i[3])
                img2_name.append(i[3])
        return all_name,img2_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.test:
            # extract img1
            img1_n = self.kin_list[item][2]+'.jpg'
            img1_path = os.path.join(self.im_root,img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3]+'.jpg'
            img2_path = os.path.join(self.im_root,img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1,img2))

            return img1,img2,kin,self.label_p,self.label_c,img1_n,img2_n
        else:
            # extract img1
            img1_n = self.kin_list[item][2] + '.jpg'
            img1_path = os.path.join(self.im_root, img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3] + '.jpg'
            img2_path = os.path.join(self.im_root, img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imgs = torch.cat((img1, img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

            return img1,img2, kin,self.label_p,self.label_c, img1_n, img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """

        new_pair_list = []

        if self.sf_fimg1: # fix img1's position when cross shuffling
            if self.sf_aln:
                neg_ls = self.alln_list
            else:
                neg_ls = self.img2_list
            rand_lth = len(neg_ls)

            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)
        else:
            neg_ls = self.alln_list
            rand_lth = len(neg_ls)
            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img1 = neg_ls[random.randint(0, rand_lth-1)]
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[2] = new_img1
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_label(self,label_path):
        with open (label_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        return nemo_ls


    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]


# class Kin_catnet_Dataset(Dataset):
#     def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
#                  cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
#         """
#         THE dataset of KinfaceW-I/ KinfaceW-II
#         :param mat_path:
#         :param im_root:
#         :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
#         :param transform:     add data augmentation
#         :param cross_shuffle: shuffle names among pair list
#         :param sf_aln:        whether shuffle all names or only img2s' names
#         out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
#         """
#
#         # kin_list is the whole 1,2,3,4,5 folds from mat
#         self.kin_list = self._read_mat(mat_path)
#         self.im_root = im_root
#         self.transform = transform
#         self.cout = 0
#         if cross_vali is not None:
#             #extract matched folds e.g. [1,2,4,5]
#             self.kin_list = self._get_cross(cross_vali)
#         # if cross_shuffle:
#         self.crf = cross_shuffle  # cross shuffle
#         self.lth = len(self.kin_list)
#         # store all img2/ all_(img1+img2) list for cross shuffle,
#         self.img2_list = [i[3] for i in self.kin_list]
#         self.alln_list = self.get_alln(self.kin_list)
#         self.sf_aln = sf_aln
#         # self.crs_insect = 0
#         # self.kin_list_bak= copy.deepcopy(self.kin_list)
#         self.sf_fimg1 = sf_fimg1
#         self.test = test
#         # self.kin_list_bak= copy.deepcopy(self.kin_list)
#         full_one = np.ones((128, 128), dtype=np.float32)
#         full_zero = np.zeros((128, 128, 5), dtype=np.float32)
#         full_zero[:, :, 2] = full_one
#         self.label = label_transforms(full_zero)
#
#
#     def get_alln(self,ls):
#         all_name = []
#         for i in ls:
#             all_name.append(i[2])
#             all_name.append(i[3])
#         return all_name
#
#     def __len__(self):
#         return len(self.kin_list)
#
#     def __getitem__(self, item):
#
#         if torch.is_tensor(item):
#             item = item.tolist()
#         # extract img1
#         img1_n = self.kin_list[item][2]
#         img1_path = os.path.join(self.im_root,img1_n)
#         img1 = Image.open(img1_path)
#         # extract img2
#         img2_n = self.kin_list[item][3]
#         img2_path = os.path.join(self.im_root,img2_n)
#         img2 = Image.open(img2_path)
#         # get kin label 0/1
#         kin = self.kin_list[item][1]
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#
#         # imgs = torch.cat((img1,img2))
#         # after each epoch, shuffle once
#         if not self.test:
#             if self.crf:
#                 self.cout +=1
#                 if self.cout == self.lth:
#                     self.cout = 0
#                     self._cross_shuffle()
#
#         return img1,img2,kin,self.label,img1_n,img2_n
#
#     def _cross_shuffle(self):
#         """
#         shuffle the second images name after each epoch
#         :return:
#         """
#         if self.sf_aln:
#             im2_ls = self.alln_list
#         else:
#             im2_ls = self.img2_list
#         rand_lth = len(im2_ls)
#         new_pair_list = []
#         # if self.crs_insect%2 ==0:
#         #     ls_bak = copy.deepcopy(self.kin_list_bak)
#         for pair_l in self.kin_list:
#             if pair_l[1] == 0:
#                 new_img2 = im2_ls[random.randint(0, rand_lth-1)]
#                 while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
#                     new_img2 = im2_ls[random.randint(0, rand_lth-1)]
#                 pair_l[3] = new_img2
#             new_pair_list.append(pair_l)
#         # else:
#         #     for pair_l in self.kin_list:
#         #         if pair_l[1] == 0:
#         #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
#         #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
#         #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
#         #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
#         #             pair_l[3] = new_img2
#         #             pair_l[2] = new_img1
#         #         new_pair_list.append(pair_l)
#         # infact it's no use to use this line:
#         # self.crs_insect +=1
#         self.kin_list = new_pair_list
#
#     def _read_mat(self,mat_path):
#         mat = scipy.io.loadmat(mat_path)
#         conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
#         pair_list = [conv_type(ls) for ls in mat['pairs']]
#         return pair_list
#
#     def _get_cross(self,cross):
#
#         return [i for i in self.kin_list if i[0] in cross]

class Kin_catnet_Dataset(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_mat(mat_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 5), dtype=np.float32)
        full_zero[:, :, 2] = full_one
        self.label = label_transforms(full_zero)


    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_n = self.kin_list[item][2]
        img1_path = os.path.join(self.im_root,img1_n)
        img1 = Image.open(img1_path)
        # extract img2
        img2_n = self.kin_list[item][3]
        img2_path = os.path.join(self.im_root,img2_n)
        img2 = Image.open(img2_path)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if not self.test:
            if self.crf:
                self.cout +=1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

        return img1,img2,kin,self.label,img1_n,img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        # if self.crs_insect%2 ==0:
        #     ls_bak = copy.deepcopy(self.kin_list_bak)
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)
        # else:
        #     for pair_l in self.kin_list:
        #         if pair_l[1] == 0:
        #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
        #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
        #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             pair_l[3] = new_img2
        #             pair_l[2] = new_img1
        #         new_pair_list.append(pair_l)
        # infact it's no use to use this line:
        # self.crs_insect +=1
        self.kin_list = new_pair_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]

class Kin_catnet_Dataset_ffhq(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_mat(mat_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 10), dtype=np.float32)
        full_zero[:, :, 6] = full_one
        self.label = label_transforms(full_zero)


    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_n = self.kin_list[item][2]
        img1_path = os.path.join(self.im_root,img1_n)
        img1 = Image.open(img1_path)
        # extract img2
        img2_n = self.kin_list[item][3]
        img2_path = os.path.join(self.im_root,img2_n)
        img2 = Image.open(img2_path)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if not self.test:
            if self.crf:
                self.cout +=1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

        return img1,img2,kin,self.label,img1_n,img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        # if self.crs_insect%2 ==0:
        #     ls_bak = copy.deepcopy(self.kin_list_bak)
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)
        # else:
        #     for pair_l in self.kin_list:
        #         if pair_l[1] == 0:
        #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
        #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
        #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             pair_l[3] = new_img2
        #             pair_l[2] = new_img1
        #         new_pair_list.append(pair_l)
        # infact it's no use to use this line:
        # self.crs_insect +=1
        self.kin_list = new_pair_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]


class Kin_catnet_Dataset_64(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_mat(mat_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((64, 64), dtype=np.float32)
        full_zero = np.zeros((64, 64, 5), dtype=np.float32)
        full_zero[:, :, 2] = full_one
        self.label = label_transforms(full_zero)


    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_n = self.kin_list[item][2]
        img1_path = os.path.join(self.im_root,img1_n)
        img1 = Image.open(img1_path)
        # extract img2
        img2_n = self.kin_list[item][3]
        img2_path = os.path.join(self.im_root,img2_n)
        img2 = Image.open(img2_path)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if not self.test:
            if self.crf:
                self.cout +=1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()

        return img1,img2,kin,self.label,img1_n,img2_n

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        # if self.crs_insect%2 ==0:
        #     ls_bak = copy.deepcopy(self.kin_list_bak)
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('_')[1] == new_img2.split('_')[1]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)
        # else:
        #     for pair_l in self.kin_list:
        #         if pair_l[1] == 0:
        #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
        #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
        #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             pair_l[3] = new_img2
        #             pair_l[2] = new_img1
        #         new_pair_list.append(pair_l)
        # infact it's no use to use this line:
        # self.crs_insect +=1
        self.kin_list = new_pair_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]



class IPCGAN_data(Dataset):
    """
    Nemo-children-adult dataset loader
    """
    def __init__(self,label_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,sf_fimg1 = True,test = False,sf_sequence = False):
        """
        :param label_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle training list after each epoch
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param sf_fimg1:      whether fix img1 position while shuffling neg pairs
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_label(label_path)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.alln_list,self.img2_list = self.get_names(self.kin_list)
        self.sf_aln = sf_aln
        self.sf_fimg1 = sf_fimg1
        self.test = test
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        full_one = np.ones((128, 128), dtype=np.float32)
        full_zero = np.zeros((128, 128, 5), dtype=np.float32)
        full_zero[:, :, 2] = full_one
        self.condition = label_transforms(full_zero)

        self.im_label_young = [i[3] for i in self.kin_list]
        self.im_label_old = [i[2] for i in self.kin_list]
        self.labe_group_images = [self.im_label_young,[],self.im_label_old]
        self.labe_pairs = self.get_label_pairs(self.kin_list)
        self.list_lth = len(self.kin_list)
        ############## condition
        self.condition128 = []
        full_one = np.ones((128, 128), dtype=np.float32)
        for i in range(5):
            full_zero = np.zeros((128, 128, 5), dtype=np.float32)
            full_zero[:, :, i] = full_one
            self.condition128.append(full_zero)

        # define label 64*64 for condition discriminate image
        self.condition64 = []
        full_one = np.ones((64, 64), dtype=np.float32)
        for i in range(5):
            full_zero = np.zeros((64, 64, 5), dtype=np.float32)
            full_zero[:, :, i] = full_one
            self.condition64.append(full_zero)

    def get_label_pairs(self,ls):
        lenth = len(ls)
        label_pairs = []
        for i in range(lenth):
            if i%2==0:
                label_pairs.append([0,2])
            else:
                label_pairs.append([2,0])
        return label_pairs



    def get_names(self,ls):
        img2_name = []
        all_name = []
        for i in ls:
            if i[1]==1:
                all_name.append(i[2])
                all_name.append(i[3])
                img2_name.append(i[3])
        return all_name,img2_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        if self.test:
            # extract img1
            img1_n = self.kin_list[item][2]+'.jpg'
            img1_path = os.path.join(self.im_root,img1_n)
            img1 = Image.open(img1_path)
            # extract img2
            img2_n = self.kin_list[item][3]+'.jpg'
            img2_path = os.path.join(self.im_root,img2_n)
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            # if self.transform:
            #     img1 = self.transform(img1)
            #     img2 = self.transform(img2)

            # imgs = torch.cat((img1,img2))

            # return img1,img2,kin,self.condition,img1_n,img2_n


            # if self.transforms is not None:
            source_img_128 = img2.resize((128, 128))
            source_img_128=self.transform(source_img_128)
            condition_128_tensor_li=[]
            # if self.label_transforms is not None:
            for condition in self.condition128:
                condition_128_tensor_li.append(label_transforms(condition).cuda())
            return source_img_128.cuda(),condition_128_tensor_li
        else:
            # extract img1
            img1_n = self.kin_list[item][2] + '.jpg'
            img1_path = os.path.join(self.im_root, img1_n)
            img1 = Image.open(img1_path).resize((128,128))
            # extract img2
            img2_n = self.kin_list[item][3] + '.jpg'
            img2_path = os.path.join(self.im_root, img2_n)
            img2 = Image.open(img2_path).resize((128,128))

            # # get kin label 0/1
            kin = self.kin_list[item][1]


            if random.randint(0, 1)==0:
                source_img = img1
            else:
                source_img = img2

            source_img_227 = source_img.resize((227, 227))
            source_img_128 = source_img.resize((128, 128))

            true_label = int(self.labe_pairs[item][0])
            fake_label = int(self.labe_pairs[item][1])
            true_label_128=self.condition128[true_label]
            true_label_64=self.condition64[true_label]
            fake_label_64=self.condition64[fake_label]
            true_label_im = self.labe_group_images[true_label][random.randint(0, self.list_lth-1)]
            true_label_img_nm = true_label_im+'.jpg'
            true_label_img_pth = os.path.join(self.im_root, true_label_img_nm)
            true_label_img = Image.open(true_label_img_pth).resize((128,128))


            true_label_img=self.transform(true_label_img)
            source_img_227=self.transform(source_img_227)
            source_img_128=self.transform(source_img_128)

            true_label_128=label_transforms(true_label_128)
            true_label_64=label_transforms(true_label_64)
            fake_label_64=label_transforms(fake_label_64)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()
            return source_img_227, source_img_128, true_label_img, \
                   true_label_128, true_label_64, fake_label_64, true_label



    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """

        new_pair_list = []

        if self.sf_fimg1: # fix img1's position when cross shuffling
            if self.sf_aln:
                neg_ls = self.alln_list
            else:
                neg_ls = self.img2_list
            rand_lth = len(neg_ls)

            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)
        else:
            neg_ls = self.alln_list
            rand_lth = len(neg_ls)
            for pair_l in self.kin_list:
                if pair_l[1] == 0:
                    new_img1 = neg_ls[random.randint(0, rand_lth-1)]
                    new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                        new_img2 = neg_ls[random.randint(0, rand_lth-1)]
                    pair_l[2] = new_img1
                    pair_l[3] = new_img2
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_label(self,label_path):
        with open (label_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        return nemo_ls


    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]






# class CACD(Dataset):
#     def __init__(self,split="train",transforms=None, label_transforms=None):
#
#         self.split=split
#
#         #define label 128*128 for condition generate image
#         list_root = os.path.abspath("./data/cacd2000-lists")
#         data_root = "/home/guyuchao/Dataset/ExperimentDataset/CACD2000-aligned"
#
#         self.condition128=[]
#         full_one=np.ones((128,128),dtype=np.float32)
#         for i in range(5):
#             full_zero=np.zeros((128,128,5),dtype=np.float32)
#             full_zero[:,:,i]=full_one
#             self.condition128.append(full_zero)
#
#         # define label 64*64 for condition discriminate image
#         self.condition64 = []
#         full_one = np.ones((64, 64),dtype=np.float32)
#         for i in range(5):
#             full_zero = np.zeros((64, 64, 5),dtype=np.float32)
#             full_zero[:, :, i] = full_one
#             self.condition64.append(full_zero)
#
#         #define label_pairs
#         label_pair_root=os.path.join(list_root,"train_label_pair.txt")
#         with open(label_pair_root,'r') as f:
#             lines=f.readlines()
#         lines=[line.strip() for line in lines]
#         shuffle(lines)
#         self.label_pairs=[]
#         for line in lines:
#             label_pair=[]
#             items=line.split()
#             label_pair.append(int(items[0]))
#             label_pair.append(int(items[1]))
#             self.label_pairs.append(label_pair)
#
#         #define group_images
#         group_lists = [
#             os.path.join(list_root, 'train_age_group_0.txt'),
#             os.path.join(list_root, 'train_age_group_1.txt'),
#             os.path.join(list_root, 'train_age_group_2.txt'),
#             os.path.join(list_root, 'train_age_group_3.txt'),
#             os.path.join(list_root, 'train_age_group_4.txt')
#         ]
#
#         self.label_group_images = []
#         for i in range(len(group_lists)):
#             with open(group_lists[i], 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]
#             group_images = []
#             for l in lines:
#                 items = l.split()
#                 group_images.append(os.path.join(data_root, items[0]))
#             self.label_group_images.append(group_images)
#
#         #define train.txt
#         if self.split is "train":
#             self.source_images = []#which use to aging transfer
#             with open(os.path.join(list_root, 'train.txt'), 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]
#             shuffle(lines)
#             for l in lines:
#                 items = l.split()
#                 self.source_images.append(os.path.join(data_root, items[0]))
#         else:
#             self.source_images = []  # which use to aging transfer
#             with open(os.path.join(list_root, 'test.txt'), 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]
#             shuffle(lines)
#             for l in lines:
#                 items = l.split()
#                 self.source_images.append(os.path.join(data_root, items[0]))
#
#         #define pointer
#         self.train_group_pointer=[0,0,0,0,0]
#         self.source_pointer=0
#         self.batch_size=32
#         self.transforms=transforms
#         self.label_transforms=label_transforms
#
#     def __getitem__(self, idx):
#         if self.split is "train":
#             pair_idx=idx//self.batch_size #a batch train the same pair
#             true_label=int(self.label_pairs[pair_idx][0])
#             fake_label=int(self.label_pairs[pair_idx][1])
#
#             true_label_128=self.condition128[true_label]
#             true_label_64=self.condition64[true_label]
#             fake_label_64=self.condition64[fake_label]
#
#             true_label_img=pil_loader(self.label_group_images[true_label][self.train_group_pointer[true_label]]).resize((128,128))
#             source_img=pil_loader(self.source_images[self.source_pointer])
#
#             source_img_227=source_img.resize((227,227))
#             source_img_128=source_img.resize((128,128))
#
#             if self.train_group_pointer[true_label]<len(self.label_group_images[true_label])-1:
#                 self.train_group_pointer[true_label]+=1
#             else:
#                 self.train_group_pointer[true_label]=0
#
#             if self.source_pointer<len(self.source_images)-1:
#                 self.source_pointer+=1
#             else:
#                 self.source_pointer=0
#
#             if self.transforms is not None:
#                 true_label_img=self.transforms(true_label_img)
#                 source_img_227=self.transforms(source_img_227)
#                 source_img_128=self.transforms(source_img_128)
#
#             if self.label_transforms is not None:
#                 true_label_128=self.label_transforms(true_label_128)
#                 true_label_64=self.label_transforms(true_label_64)
#                 fake_label_64=self.label_transforms(fake_label_64)
#             #source img 227 : use it to extract face feature
#             #source img 128 : use it to generate different age face -> then resize to (227,227) to extract feature, compile with source img 227
#             #ture_label_img : img in target age group -> use to train discriminator
#             #true_label_128 : use this condition to generate
#             #true_label_64 and fake_label_64 : use this condition to discrimination
#             #true_label : label
#
#             return source_img_227,source_img_128,true_label_img,\
#                    true_label_128,true_label_64,fake_label_64, true_label
#         else:
#             source_img_128=pil_loader(self.source_images[idx]).resize((128,128))
#             if self.transforms is not None:
#                 source_img_128=self.transforms(source_img_128)
#             condition_128_tensor_li=[]
#             if self.label_transforms is not None:
#                 for condition in self.condition128:
#                     condition_128_tensor_li.append(self.label_transforms(condition).cuda())
#             return source_img_128.cuda(),condition_128_tensor_li
#
#     def __len__(self):
#         if self.split is "train":
#             return len(self.label_pairs)
#         else:
#             return len(self.source_images)


if __name__=='__main__':
    # father-daughter
    train_ls = '../data/label/fd.pkl'
    data_pth = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/f-d'
    # nemo_data = ipcgan_data(train_ls,data_pth,[1,2,3,4],transform= train_transform,sf_sequence= True,cross_shuffle =True,sf_aln = True)
    # nemoloader = DataLoader(nemo_data,shuffle=True)
    # for j in range(3):
    #     for i,data in enumerate(nemoloader):
    #         # print(i)
    #         pass
    #     print(i)