import torch.nn as nn
import utils.loader
from model import modules
from utils.transform import *
from utils.loader import *
# from utils.loss import *


####################### KinfaceW I&II ############################
class kin_config(object):
    ################### data ####################

    des = 'stage2_logic'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    data_name = 'kfw1'
    kintype  = 'F-D'
    ## list path
    list_path = '/home/Documents/DATA/kinship/KinFaceW-I/meta_data/fd_pairs.mat'
    ## data path
    img_root  = '/home/Documents/DATA/kinship/KinFaceW-I/images/father-dau'
    ## dataset
    Dataset =  KinDataset
    ## transformer
    trans =  [train_atten_transform,test_atten_transform]
    ################### loader ####################

    sf_sequence = True
    ## shuflle the list after each epoch
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    ################### train ####################
    ######## model

    ## modelname
    model = modules
    ## structure
    model_name = 'attenNet'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers
    epoch_num = 60
    ## batch
    train_batch =  32
    test_batch = 32
    ## learning rate
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    # lr_milestones = [180, 250, 300, 400, 500, 550]
    lr_milestones = [60,300]
    ## regularization
    weight_decay = 5e-3

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 4
    ##  frequent of printing the evaluation acc
    prt_fr = 1
    ## loss
    loss = nn.CrossEntropyLoss
    ## optimal
    optimal = 'adam'

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
