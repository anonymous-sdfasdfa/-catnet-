import torch.nn as nn
from model import catNet
from utils.transform import *
from utils.loader import *
from utils.losses import ContrastiveLoss


class kin_config(object):
    ################### data ####################

    des = 'Nemo_CA_Dataset training&testing using basic attention Network'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    kintype  = 'fd'
    ## list path
    list_path = '../data/label/fd.pkl'
    ## data path
    img_root  = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/f-d'
    ## pretrain
    # gpth = './data/pretrained/IPCGANs/gepoch_9_iter_12500.pth'
    gpth = './checkpoint/IPCGANS/2022-02-07_23-04-56/saved_parameters/gepoch_7_iter_6250.pth'
    ## dataset
    Dataset =  Nemo_catnet_Dataset
    ## transformer
    trans =  [train_cat_transform,test_cat_transform]
    ################### loader ####################
    # shuffle sequential list of image
    sf_sequence = True
    ## shuflle the list after each epoch
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    sf_fimg1 = False #whether fix img1 position while shuffling neg pairs

    gpu = 0
    ################### train ####################
    ######## model

    ## modelname
    model = catNet
    ## structure
    model_name = 'CatNet'
    ## image size
    imsize = (6,128,128)
    ## epoch numbers
    epoch_num = 1000
    ## batch
    train_batch =  64
    test_batch = 64
    ## learning rate
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    # lr_milestones = [180, 250, 300, 400, 500, 550]
    lr_milestones = [200,300]
    ## regularization
    weight_decay = 5e-3

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 1
    ##  frequent of printing the evaluation acc
    prt_fr = 50
    ## loss
    loss = ContrastiveLoss
    ## optimal
    optim = 'adam'

    ######## record
    ## save the training accuracy
    save_tacc = False
    ## whether load pretrained model
    reload = ''
    ## save trained model
    save_ck = True
    ##
    logs_name = 'data/logs'
    ##
    savemis = False
    ##
    save_graph = False


if __name__ =='__main__':
    d = kin_config.__dict__
    print(d)