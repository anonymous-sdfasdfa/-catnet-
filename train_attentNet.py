from config import attention_config
from utils.base_train import base_train,pint_acc,pint_acc_ub
import argparse
from utils.loader import *
import  numpy as np
class attenNet_train(base_train):
    def __init__(self,config):
        super().__init__(config)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train attention Net')
    parser.add_argument('--datatype','--dt',type=str,default='kfw1',help='The dataset trained on')
    args = parser.parse_args()
    mats = ['fd', 'fs', 'md', 'ms']
    imgs = ['father-dau', 'father-son', 'mother-dau', 'mother-son']

    if args.datatype =='kfw1':
        print('start training on kfw1')
        acc_ls = []
        for mat,img in zip(mats,imgs):
            attention_config.kin_config.data_name = 'kfw1'
            attention_config.kin_config.kintype = mat
            ## list path

            attention_config.kin_config.list_path = '/home/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(mat)
            ## data path
            attention_config.kin_config.img_root = '/home/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(img)
            netmodel = attenNet_train(attention_config.kin_config)

            acc_ls.append(netmodel.cross_run())

        pint_acc(acc_ls)


    elif args.datatype =='nemo':

        print('start training on nemo-kinship-children')

        mats = ['fd', 'fs', 'md', 'ms']
        imgs = ['f-d','f-s','m-d','m-s']
        acc_ls = []
        for mat, img in zip(mats, imgs):
            attention_config.kin_config.data_name = 'nemo'
            attention_config.kin_config.kintype = mat
            attention_config.kin_config.Dataset = Nemo_CA_Dataset
            ## list path
            attention_config.kin_config.epoch_num = 60
            attention_config.kin_config.list_path = './data/label/{}.pkl'.format(mat)
            ## data path
            attention_config.kin_config.img_root = '/home/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(img)
            netmodel = attenNet_train(attention_config.kin_config)

            acc_ls.append(netmodel.cross_run())

        pint_acc(acc_ls)

