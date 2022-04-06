# from model.IPCGANs import  Generator
# from model import catNet
# from model.catNet import *
# from utils.loader import  *
# from utils.transform import *
# # import scipy.io
# from torchvision.datasets.folder import pil_loader
# import timeit
#
# # import scipy.io
# # import os
# import pickle
# import numpy as np

from torchvision.datasets.folder import pil_loader
from utils.transform import *
from model.IPCGANs import  Generator
from model import catNet
import pickle
import scipy.io

def get_features(g,condition, pth,img_root):
    path = os.path.join(img_root,pth)
    timg = pil_loader(path)
    image = test_cat_transform(timg).unsqueeze(0)
    image= image.cuda()
    if condition is not None:
        image = torch.cat((image, condition), 1)
        # x_c = torch.cat((image, condition), 1)
    fea = g(image)
    fea = fea.view(fea.shape[0], -1).cpu().data.numpy()
    return fea




def gen_mat(kintype_lb,kintype_pth,dt):
    ################## net
    # def freeze(feat):
    #     for name, child in feat.named_children():
    #         if name == 'repeat_3':
    #             return
    #         for param in child.parameters():
    #             param.requires_grad = False

    # pth = '/home/wei/Documents/CODE/BMVC/IPCGANs-Pytorch-master/checkpoint/IPCGANS/2020-10-12_01-04-00/saved_parameters/gepoch_9_iter_12500.pth'
    # pth = '/home/wei/Documents/CODE/BMVC/cat-Net/data/pretrained/IPCGANs/gepoch_6_iter_1000.pth'
    pth = '../data/pretrained/IPCGANs/gepoch_4_iter_5000.pth'
    g = Generator(pretrained=pth).encoder
    # freeze(g)
    g.eval()
    g.cuda()

    ################## condition
    full_one = np.ones((128, 128), dtype=np.float32)
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)
    full_zero[:, :, 3] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition = label.cuda()

    if dt == 'cat':
        pth = '../data/label/{}.pkl'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(kintype_pth)
        with open(pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)
    elif dt == 'kfw1':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'kfw2':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'ub':
        pth = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/label/{}.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)

    #######require
    leth = len(nemo_ls)
    fold = np.zeros((leth,1))
    idxa = np.zeros((leth,1))
    idxb = np.zeros((leth,1))
    idxa_name = np.zeros((leth,1),dtype=np.object)
    idxb_name = np.zeros((leth,1),dtype=np.object)
    matches = np.zeros((leth,1))
    ux = []

    im_dict = {}
    i = 0

    for j,item in enumerate(nemo_ls):
        im1pth = item[2]+'.jpg'
        if im1pth in im_dict:
            im1_num = im_dict[im1pth]
        else:
            i += 1
            im1_num = i
            im_dict[im1pth]=im1_num

            fea = get_features(g,condition, im1pth,img_root)
            if i ==1:
                ux = fea
            else:
                ux =  np.concatenate((ux,fea),axis=0)

        im2pth = item[3]+'.jpg'
        if im2pth in im_dict:
            im2_num = im_dict[im2pth]
        else:
            i += 1
            im2_num = i
            im_dict[im2pth]=im2_num
            fea = get_features(g,condition,im2pth,img_root)
            if i ==1:
                ux = fea
            else:
                ux =  np.concatenate((ux,fea),axis=0)

        fold[j,0] = np.array([[item[0]]])
        idxa_name[j,0] = np.array([im1pth])
        idxa[j,0] = np.array([[im1_num]])
        idxb_name[j,0] = np.array([im2pth])
        idxb[j,0] = np.array([[im2_num]])
        matches[j,0] =np.array([[item[1]]])
        ux = np.array(ux, dtype=np.double)
        matches = np.array(matches, dtype=bool)

    scipy.io.savemat('/home/wei/Documents/CODE/CVPR/NRML/data/bmvc/gan_only/{}/{}_gfeature.mat'.format(dt,kintype_lb), mdict={'fold':fold,'idxa':idxa,'idxb':idxb,'matches':matches,'ux':ux})



def get_ckps(pth):
    ckps= os.listdir(pth)
    ckps= sorted(ckps)
    ckp_ls = [os.path.join(pth,it) for it in ckps]
    ckp_ls = ckp_ls[::-1]
    return ckp_ls



def load_dict(net,ckps):
    pretrained_dict = torch.load(ckps)
    # # step2: get model state_dict
    # model_dict = net.state_dict()
    # # step3: remove pretrained_dict params which is not in model_dict
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # step4: update model_dict using pretrained_dict
    # model_dict.update(pretrained_dict)
    # # step5: update model using model_dict
    net.load_state_dict(pretrained_dict['arch'])
    net.eval()
    net.cuda()
    return net

def get_features_contrastive_catnet(net,condition,pth,img_root,t):
    path = os.path.join(img_root,pth)
    timg = pil_loader(path)
    image = test_cat_transform(timg).unsqueeze(0)
    image= image.cuda()
    x_p,x_c = net(image,image,condition)
    if t =='p':
        fea = x_p.cpu().data.numpy()
        return fea
    if t =='c':
        fea = x_c.cpu().data.numpy()
        return fea

def get_features_woage(net,condition,pth,img_root,t):
    path = os.path.join(img_root,pth)
    timg = pil_loader(path)
    image = test_cat_transform(timg).unsqueeze(0)
    image= image.cuda()
    x_p,x_c = net(image,image,condition)
    if t =='p':
        fea = x_p.cpu().data.numpy()
        return fea
    if t =='c':
        fea = x_c.cpu().data.numpy()
        return fea

def get_features_contrastive_catnet_age0(net,condition1,condition2,pth,img_root,t):
    path = os.path.join(img_root,pth)
    timg = pil_loader(path)
    image = test_cat_transform(timg).unsqueeze(0)
    image= image.cuda()
    x_p,x_c = net(image,image,condition1,condition2)
    if t =='p':
        fea = x_p.cpu().data.numpy()
        return fea
    if t =='c':
        fea = x_c.cpu().data.numpy()
        return fea


def read_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
    pair_list = [conv_type(ls) for ls in mat['pairs']]
    new_ls = []
    for item in pair_list:
        new_ls.append([item[0],item[1],item[2][:-4],item[3][:-4]])
    return new_ls

def gen_mat_contrastive_catnet_v2(kintype_lb,kintype_pth,ckp_pth,VERSION,dt):

    if dt =='cat':
        pth = '../data/label/{}.pkl'.format(kintype_lb)
        img_root  = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(kintype_pth)
        with open(pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)
    elif dt == 'kfw1':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(kintype_pth)
        # img_root = '/home/wei/Documents/CODE/ECCV2022/CATNet3/data/KinFaceW-I-new/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'kfw2':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'ub':
        pth = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/label/{}.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)

    ################## net

    ckp_ls = get_ckps(ckp_pth)
    ## remove the name after '-'
    VERSION_ = VERSION.split('-')[0]
    catnet = getattr(catNet, VERSION_)()


    ################## condition
    full_one = np.ones((128, 128), dtype=np.float32)########
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)####
    full_zero[:, :, 2] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition = label.cuda()


    #######require
    leth = len(nemo_ls)
    fold = np.zeros((leth,1))
    idxa = np.zeros((leth,1))
    idxb = np.zeros((leth,1))
    idxa_name = np.zeros((leth,1),dtype=np.object)
    idxb_name = np.zeros((leth,1),dtype=np.object)
    matches = np.zeros((leth,1))
    ux = []

    im_dict = {}
    i = 0
    for fold_num in range(5):
        ckp_catnet_pth_up = ckp_ls[fold_num]
        catnet = load_dict(catnet, ckp_catnet_pth_up)
        ### add
        catnet.eval()
        for j,item in enumerate(nemo_ls):



            im1pth = item[2]+'.jpg'
            if im1pth in im_dict:
                im1_num = im_dict[im1pth]
            else:
                i += 1
                im1_num = i
                im_dict[im1pth]=im1_num

                fea = get_features_contrastive_catnet(catnet,condition, im1pth,img_root,'p')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            im2pth = item[3]+'.jpg'
            if im2pth in im_dict:
                im2_num = im_dict[im2pth]
            else:
                i += 1
                im2_num = i
                im_dict[im2pth]=im2_num
                fea = get_features_contrastive_catnet(catnet,condition, im2pth,img_root,'c')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            fold[j,0] = np.array([[item[0]]])
            idxa_name[j,0] = np.array([im1pth])
            idxa[j,0] = np.array([[im1_num]])
            idxb_name[j,0] = np.array([im2pth])
            idxb[j,0] = np.array([[im2_num]])
            matches[j,0] =np.array([[item[1]]])
            ux = np.array(ux, dtype=np.double)
            matches = np.array(matches, dtype=bool)



        output = open('kfw1_{}_dict.pkl'.format(kintype_lb),'wb')
        pickle.dump(im_dict,output)
        output.close()
        print(66)
        # scipy.io.savemat('/home/wei/Documents/CODE/CVPR/NRML/data/cviu/catnet/{}/{}_{}_contrastive_{}.mat'.format(dt,kintype_lb,fold_num+1,VERSION), mdict={'fold':fold,'idxa':idxa,'idxb':idxb,'matches':matches,'ux':ux})

def gen_mat_contrastive_catnet_woage(kintype_lb,kintype_pth,ckp_pth,VERSION,dt):

    if dt =='cat':
        pth = '../data/label/{}.pkl'.format(kintype_lb)
        img_root  = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(kintype_pth)
        with open(pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)
    elif dt == 'kfw1':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(kintype_pth)
        # img_root = '/home/wei/Documents/CODE/ECCV2022/CATNet3/data/KinFaceW-I-new/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'kfw2':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'ub':
        pth = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/label/{}.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/UB_Kinface/UB/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)

    ################## net

    ckp_ls = get_ckps(ckp_pth)
    ## remove the name after '-'
    VERSION_ = VERSION.split('-')[0]
    catnet = getattr(catNet, VERSION_)()


    ################## condition
    full_one = np.ones((128, 128), dtype=np.float32)########
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)####
    full_zero[:, :, 0] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition1 = label.cuda()

    full_one = np.ones((128, 128), dtype=np.float32)  ########
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)  ####
    full_zero[:, :, 2] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition2 = label.cuda()

    #######require
    leth = len(nemo_ls)
    fold = np.zeros((leth,1))
    idxa = np.zeros((leth,1))
    idxb = np.zeros((leth,1))
    idxa_name = np.zeros((leth,1),dtype=np.object)
    idxb_name = np.zeros((leth,1),dtype=np.object)
    matches = np.zeros((leth,1))
    ux = []

    im_dict = {}
    i = 0
    for fold_num in range(5):
        ckp_catnet_pth_up = ckp_ls[fold_num]
        catnet = load_dict(catnet, ckp_catnet_pth_up)
        ### add
        catnet.eval()
        for j,item in enumerate(nemo_ls):



            im1pth = item[2]+'.jpg'
            if im1pth in im_dict:
                im1_num = im_dict[im1pth]
            else:
                i += 1
                im1_num = i
                im_dict[im1pth]=im1_num

                fea = get_features_woage(catnet,condition2, im1pth,img_root,'p')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            im2pth = item[3]+'.jpg'
            if im2pth in im_dict:
                im2_num = im_dict[im2pth]
            else:
                i += 1
                im2_num = i
                im_dict[im2pth]=im2_num
                fea = get_features_woage(catnet,condition1, im2pth,img_root,'c')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            fold[j,0] = np.array([[item[0]]])
            idxa_name[j,0] = np.array([im1pth])
            idxa[j,0] = np.array([[im1_num]])
            idxb_name[j,0] = np.array([im2pth])
            idxb[j,0] = np.array([[im2_num]])
            matches[j,0] =np.array([[item[1]]])
            ux = np.array(ux, dtype=np.double)
            matches = np.array(matches, dtype=bool)




        scipy.io.savemat('/home/wei/Documents/CODE/CVPR/NRML/data/cviu/catnet/{}/{}_{}_contrastive_{}-woage.mat'.format(dt,kintype_lb,fold_num+1,VERSION), mdict={'fold':fold,'idxa':idxa,'idxb':idxb,'matches':matches,'ux':ux})

def gen_mat_contrastive_catnet_ck(kintype_lb, kintype_pth, ckp_pth, VERSION, dt):
    if dt == 'cat':
        pth = '../data/label/{}.pkl'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(kintype_pth)
        with open(pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)
    elif dt == 'kfw1':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'kfw2':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'ub':
        pass

    ################## net

    ckp_ls = get_ckps(ckp_pth)
    catnet = getattr(catNet, VERSION)(pretrained = '/home/wei/Documents/CODE/BMVC/cat-Net/data/pretrained/IPCGANs/gepoch_16_iter_1000.pth')
    catnet.cuda()
    ################## condition
    full_one = np.ones((128, 128), dtype=np.float32)
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)
    full_zero[:, :, 2] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition = label.cuda()

    #######require
    leth = len(nemo_ls)
    fold = np.zeros((leth, 1))
    idxa = np.zeros((leth, 1))
    idxb = np.zeros((leth, 1))
    idxa_name = np.zeros((leth, 1), dtype=np.object)
    idxb_name = np.zeros((leth, 1), dtype=np.object)
    matches = np.zeros((leth, 1))
    ux = []

    im_dict = {}
    i = 0
    for fold_num in range(5):
        # ckp_catnet_pth_up = ckp_ls[fold_num]
        # catnet = load_dict(catnet, ckp_catnet_pth_up)
        ### add
        catnet.eval()
        for j, item in enumerate(nemo_ls):

            im1pth = item[2] + '.jpg'
            if im1pth in im_dict:
                im1_num = im_dict[im1pth]
            else:
                i += 1
                im1_num = i
                im_dict[im1pth] = im1_num

                fea = get_features_contrastive_catnet(catnet, condition, im1pth, img_root, 'p')
                if i == 1:
                    ux = fea
                else:
                    ux = np.concatenate((ux, fea), axis=0)

            im2pth = item[3] + '.jpg'
            if im2pth in im_dict:
                im2_num = im_dict[im2pth]
            else:
                i += 1
                im2_num = i
                im_dict[im2pth] = im2_num
                fea = get_features_contrastive_catnet(catnet, condition, im2pth, img_root, 'c')
                if i == 1:
                    ux = fea
                else:
                    ux = np.concatenate((ux, fea), axis=0)

            fold[j, 0] = np.array([[item[0]]])
            idxa_name[j, 0] = np.array([im1pth])
            idxa[j, 0] = np.array([[im1_num]])
            idxb_name[j, 0] = np.array([im2pth])
            idxb[j, 0] = np.array([[im2_num]])
            matches[j, 0] = np.array([[item[1]]])
            ux = np.array(ux, dtype=np.double)
            matches = np.array(matches, dtype=bool)

        scipy.io.savemat(
            '/home/wei/Documents/CODE/CVPR/NRML/data/bmvc/catnet/{}/{}_{}_contrastive_{}.mat'.format(dt, kintype_lb,
                                                                                                     fold_num + 1,
                                                                                                     VERSION),
            mdict={'fold': fold, 'idxa': idxa, 'idxb': idxb, 'matches': matches, 'ux': ux})


#

def gen_mat_contrastive_catnet_age0(kintype_lb,kintype_pth,ckp_pth,VERSION,dt):

    if dt =='nemo':
        pth = '../data/label/{}.pkl'.format(kintype_lb)
        img_root  = '/home/wei/Documents/DATA/kinship/ca-nemo/kin/ca-images/{}'.format(kintype_pth)
        with open(pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)
    elif dt == 'kfw1':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'kfw2':
        pth = '/home/wei/Documents/DATA/kinship/KinFaceW-II/meta_data/{}_pairs.mat'.format(kintype_lb)
        img_root = '/home/wei/Documents/DATA/kinship/KinFaceW-II/images/{}'.format(kintype_pth)
        nemo_ls = read_mat(pth)
    elif dt == 'ub':
        pass


    ################## net

    # ckp_ls = get_ckps(ckp_pth)
    # catnet = getattr(catNet, VERSION)()
    ckp_ls = get_ckps(ckp_pth)
    ## remove the name after '-'
    VERSION_ = VERSION.split('-')[0]
    catnet = getattr(catNet, VERSION_)()
    ################## condition
    full_one = np.ones((128, 128), dtype=np.float32)
    full_zero = np.zeros((128, 128, 5), dtype=np.float32)
    full_zero[:, :, 3] = full_one
    label = label_transforms(full_zero).unsqueeze(0)
    condition1 = label.cuda()

    c_one = np.ones((128, 128), dtype=np.float32)
    c_zero = np.zeros((128, 128, 5), dtype=np.float32)
    c_zero[:, :, 0] = c_one
    label = label_transforms(c_zero).unsqueeze(0)
    condition2 = label.cuda()



    # pth = '../data/label/{}.pkl'.format(kintype_lb)
    # img_root  = '/home/wei/Documents/DATA/kinship/bmvc/ca-images/{}'.format(kintype_pth)
    # with open(pth, 'rb') as fp:
    #     nemo_ls = pickle.load(fp)

    #######require
    leth = len(nemo_ls)
    fold = np.zeros((leth,1))
    idxa = np.zeros((leth,1))
    idxb = np.zeros((leth,1))
    idxa_name = np.zeros((leth,1),dtype=np.object)
    idxb_name = np.zeros((leth,1),dtype=np.object)
    matches = np.zeros((leth,1))
    ux = []

    im_dict = {}
    i = 0
    for fold_num in range(5):
        ckp_catnet_pth_up = ckp_ls[fold_num]
        catnet = load_dict(catnet, ckp_catnet_pth_up)
        ### add
        catnet.eval()
        for j,item in enumerate(nemo_ls):



            im1pth = item[2]+'.jpg'
            if im1pth in im_dict:
                im1_num = im_dict[im1pth]
            else:
                i += 1
                im1_num = i
                im_dict[im1pth]=im1_num

                fea = get_features_contrastive_catnet_age0(catnet,condition1,condition2, im1pth,img_root,'p')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            im2pth = item[3]+'.jpg'
            if im2pth in im_dict:
                im2_num = im_dict[im2pth]
            else:
                i += 1
                im2_num = i
                im_dict[im2pth]=im2_num
                fea = get_features_contrastive_catnet_age0(catnet,condition1,condition2, im2pth,img_root,'c')
                if i ==1:
                    ux = fea
                else:
                    ux =  np.concatenate((ux,fea),axis=0)

            fold[j,0] = np.array([[item[0]]])
            idxa_name[j,0] = np.array([im1pth])
            idxa[j,0] = np.array([[im1_num]])
            idxb_name[j,0] = np.array([im2pth])
            idxb[j,0] = np.array([[im2_num]])
            matches[j,0] =np.array([[item[1]]])
            ux = np.array(ux, dtype=np.double)
            matches = np.array(matches, dtype=bool)



        scipy.io.savemat('/home/wei/Documents/CODE/CVPR/NRML/data/cviu/catnet/{}/{}_{}_contrastive_{}-woage.mat'.format(dt,kintype_lb,fold_num+1,VERSION), mdict={'fold':fold,'idxa':idxa,'idxb':idxb,'matches':matches,'ux':ux})

if __name__ == '__main__':

    #####################1. get CATNET feature
    dt = 'kfw1'
    if dt == 'cat'or dt =='cat0':
        ls = ['fd','fs','md','ms']
        pl = ['f-d','f-s','m-d','m-s']
    elif dt == 'kfw1' or dt =='kfw2':
        ls = ['fd','fs','md','ms']
        pl = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    elif dt == 'ub':
        ls = ['cy', 'co']
        pl = ['cy', 'co']
    # ls= ['fd']
    # pl = ['f-d']
    # ls = ['md']
    # pl = ['m-d']
    print('generating')
    VERSION = 'CatNet_wo_branch-kfw1' #'CatNet_kmm-epoch10-crop'
    for ll,pp in zip(ls,pl):
        ckp_pths = '/home/wei/Documents/CODE/ECCV2022/CATNet3/data/checkpoints/{}-{}-{}'.format(ll,dt,VERSION)
        gen_mat_contrastive_catnet_v2(ll,pp,ckp_pths,VERSION,dt=dt)



    #####################2. get GAN feature

    # dt = 'ub'
    # if dt == 'cat':
    #     ls = ['fd','fs','md','ms']
    #     pl = ['f-d','f-s','m-d','m-s']
    # elif dt == 'kfw1' or dt =='kfw2':
    #     ls = ['fd','fs','md','ms']
    #     pl = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    # elif dt == 'ub':
    #     ls = ['cy', 'co']
    #     pl = ['cy', 'co']
    #
    # print('generating')
    # for ll,pp in zip(ls,pl):
    #     gen_mat(ll,pp,dt)


    #################### 3. get Cat age0 feature
    #     gen_mat_contrastive_catnet_age0(ll,pp,ckp_pths,VERSION,dt = 'nemo')
    # dt = 'kfw1'
    # if dt == 'cat' or dt == 'cat0':
    #     ls = ['fd', 'fs', 'md', 'ms']
    #     pl = ['f-d', 'f-s', 'm-d', 'm-s']
    # elif dt == 'kfw1' or dt == 'kfw2':
    #     ls = ['fd', 'fs', 'md', 'ms']
    #     pl = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    # elif dt == 'ub':
    #     ls = ['cy', 'co']
    #     pl = ['cy', 'co']
    # # ls= ['fd']
    # # pl = ['f-d']
    # # ls = ['md']
    # # pl = ['m-d']
    # print('generating')
    # VERSION = 'CatNet_two-layer3-epoch10'  # 'CatNet_kmm-epoch10-crop'
    # for ll, pp in zip(ls, pl):
    #     ckp_pths = '/home/wei/Documents/CODE/ECCV2022/CATNet3/data/checkpoints/{}-{}-{}'.format(ll, dt, VERSION)
    #     gen_mat_contrastive_catnet_woage(ll, pp, ckp_pths, VERSION, dt=dt)