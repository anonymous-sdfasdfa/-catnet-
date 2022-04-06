from utils.contrastive_train import contrastive_train
from config import contrastive_catnet_config
from utils.loader import *
from utils.transform import *
import argparse




if __name__=='__main__':
    #
    parser = argparse.ArgumentParser(description='train CATnet')
    parser.add_argument('--datatype','--dt',type=str,default='nemo',help='The dataset trained on')
    parser.add_argument('--modelname', '--mn', type=str, default='CatNet_two_fc-cat', help='The dataset trained on')
    args = parser.parse_args()
    types = ['fd','fs','md','ms']
    imgs = ['f-d','f-s','m-d','m-s']

    if args.datatype == 'nemo':
        print('start training on memo cat')
        # types = [ 'ms']
        # imgs = [ 'm-s']
        for ty,img in zip(types,imgs):
            print('start the training of {}'.format(ty))
            contrastive_catnet_config.kin_config.model_name = args.modelname
            contrastive_catnet_config.kin_config.kintype = '{}-cat'.format(ty)

            ## list path

            contrastive_catnet_config.kin_config.lr = 0.001
            contrastive_catnet_config.kin_config.train_batch = 36

            contrastive_catnet_config.kin_config.epoch_num = 10
            # contrastive_catnet_config.kin_config.trans = [test_kin_transform, test_kin_transform]

            contrastive_catnet_config.kin_config.list_path = './data/label/{}.pkl'.format(ty)
            ## data path
            # contrastive_catnet_config.kin_config.img_root = '/var/scratch/wwang/DATA/ca-images/{}'.format(img)
            contrastive_catnet_config.kin_config.img_root = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(
                img)
            netmodel = contrastive_train(contrastive_catnet_config.kin_config)

            netmodel.cross_run()

    elif args.datatype == 'kfw1':
        mats = ['fd', 'fs', 'md', 'ms']
        imgs = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        print('start training on kfw1')
        for ty, img in zip(types, imgs):
            print('start the training of {}'.format(ty))
            contrastive_catnet_config.kin_config.model_name = args.modelname #'CatNet_kmm'
            contrastive_catnet_config.kin_config.kintype = '{}-kfw1'.format(ty)
            contrastive_catnet_config.kin_config.Dataset = Kin_catnet_Dataset
            contrastive_catnet_config.kin_config.train_batch = 36
            contrastive_catnet_config.kin_config.lr = 0.001
            # contrastive_catnet_config.kin_config.trans = [test_kin_transform,test_kin_transform]
            ## list path
            contrastive_catnet_config.kin_config.epoch_num = 10
            contrastive_catnet_config.kin_config.gpth = './checkpoint/IPCGANS/2022-02-07_23-04-56/saved_parameters/gepoch_7_iter_6250.pth'
            # contrastive_catnet_config.kin_config.gpth = './checkpoint/IPCGANS/2022-02-07_23-04-56/saved_parameters/gepoch_8_iter_6250.pth'
            contrastive_catnet_config.kin_config.trans =  [train_cat_transform,test_cat_transform]
            # contrastive_catnet_config.kin_config.list_path = '/var/scratch/wwang/DATA/KinFaceW-I/meta_data/{}_pairs.mat'.format(ty)
            contrastive_catnet_config.kin_config.list_path = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(
                ty)
            ## data path
            # contrastive_catnet_config.kin_config.img_root = '/var/scratch/wwang/DATA//KinFaceW-I/images/{}'.format(img)
            contrastive_catnet_config.kin_config.img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(
                img)
            # contrastive_catnet_config.kin_config.img_root = '/home/wei/Documents/CODE/ECCV2022/CATNet3/data/KinFaceW-I-new/images/{}'.format(
            #     img)
            netmodel = contrastive_train(contrastive_catnet_config.kin_config)

            netmodel.cross_run()

    elif args.datatype == 'kfw2':
        mats = ['fd', 'fs', 'md', 'ms']
        imgs = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        print('start training on kfw2')
        for ty, img in zip(types, imgs):
            print('start the training of {}'.format(ty))
            contrastive_catnet_config.kin_config.model_name = args.modelname
            contrastive_catnet_config.kin_config.kintype = '{}-kfw2'.format(ty)
            contrastive_catnet_config.kin_config.Dataset = Kin_catnet_Dataset
            ## list path
            contrastive_catnet_config.kin_config.train_batch = 36
            contrastive_catnet_config.kin_config.lr = 0.001
            contrastive_catnet_config.kin_config.epoch_num = 10
            # contrastive_catnet_config.kin_config.trans = [test_kin_transform, test_kin_transform]
            # contrastive_catnet_config.kin_config.list_path = '/var/scratch/wwang/DATA/KinFaceW-II/meta_data/{}_pairs.mat'.format(ty)
            contrastive_catnet_config.kin_config.list_path = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(
                ty)
            ## data path
            # contrastive_catnet_config.kin_config.img_root = '/var/scratch/wwang/DATA//KinFaceW-II/images/{}'.format(img)
            contrastive_catnet_config.kin_config.img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(
                img)
            netmodel = contrastive_train(contrastive_catnet_config.kin_config)

            netmodel.cross_run()

    elif args.datatype == 'ub':
        types = ['cy', 'co']
        imgs = ['cy', 'co']
        print('start training on ub')
        for ty, img in zip(types, imgs):
            print('start the training of {}'.format(ty))
            contrastive_catnet_config.kin_config.model_name = 'CatNet_kmm'
            contrastive_catnet_config.kin_config.kintype = '{}-ub'.format(ty)
            contrastive_catnet_config.kin_config.Dataset = Kin_catnet_Dataset
            ## list path
            contrastive_catnet_config.kin_config.train_batch = 36
            contrastive_catnet_config.kin_config.epoch_num = 60
            contrastive_catnet_config.kin_config.lr = 0.001
            # contrastive_catnet_config.kin_config.weight_decay= 5e-5
            # contrastive_catnet_config.kin_config.gpth = './data/pretrained/IPCGANs/gepoch_6_iter_1000.pth'
            contrastive_catnet_config.kin_config.trans = [test_kin_transform, test_kin_transform]
            # contrastive_catnet_config.kin_config.list_path = '/var/scratch/wwang/DATA/UB_Kinface/UB/label/{}.mat'.format(ty)
            contrastive_catnet_config.kin_config.list_path = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/label/{}.mat'.format(
                ty)
            ## data path
            # contrastive_catnet_config.kin_config.img_root = '/var/scratch/wwang/DATA/UB_Kinface/UB/{}'.format(img)
            contrastive_catnet_config.kin_config.img_root = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/{}'.format(
                img)
            netmodel = contrastive_train(contrastive_catnet_config.kin_config)

            netmodel.cross_run()