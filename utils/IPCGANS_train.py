import torch
import torchvision
import torch.utils.data as data
import os
from os.path import join
import argparse
import logging
from tqdm import tqdm
from torchvision.utils import save_image
#user import
from data_generator.DataLoader_IpcGans import CACD
from model.IPCGANs import IPCGANs
from utils.io_IPCGANS import check_dir,Img_to_zero_center,Reverse_zero_center
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

#step1: define argument
parser = argparse.ArgumentParser(description='train IPCGANS')
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Optimizer
parser.add_argument('--learning_rate', '--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=256)
# parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=50)
parser.add_argument('--max_epoches', type=int, help='Number of epoches to run', default=20)
parser.add_argument('--val_interval', type=int, help='Number of steps to validate', default=1)
parser.add_argument('--save_interval', type=int, help='Number of batches to save model', default=1)

parser.add_argument('--d_iter', type=int, help='Number of steps for discriminator', default=1)
parser.add_argument('--g_iter', type=int, help='Number of steps for generator', default=2)
# Model
parser.add_argument('--gan_loss_weight', type=float, help='gan_loss_weight', default=70)
parser.add_argument('--feature_loss_weight', type=float, help='fea_loss_weight', default=2)
parser.add_argument('--age_loss_weight', type=float, help='age_loss_weight', default=1)
parser.add_argument('--age_groups', type=int, help='the number of different age groups', default=5)
# parser.add_argument('--age_classifier_path', type=str, help='directory of age classification model', default='./checkpoint/pretrained/age_cls_epoch_47_iter_0.pth')
parser.add_argument('--age_classifier_path', type=str, help='directory of age classification model', default='./checkpoint/pretrain_alexnet/saved_parameters/epoch_199.pth')
#parser.add_argument('--feature_extractor_path', type=str, help='directory of pretrained alexnet', default='/home/guyuchao/Dataset/Pretrain Model/alexnet-owt-4df8aa71.pth')

# Data and IO
parser.add_argument('--checkpoint', type=str, help='logs and checkpoints directory', default='./checkpoint/IPCGANS/%s'%(TIMESTAMP))
parser.add_argument('--saved_model_folder', type=str,
                    help='the path of folder which stores the parameters file',
                    default='./checkpoint/IPCGANS/%s/saved_parameters/'%(TIMESTAMP))
parser.add_argument('--saved_validation_folder', type=str,
                    help='the path of folder which stores the val img',
                    default='./checkpoint/IPCGANS/%s/validation/'%(TIMESTAMP))
parser.add_argument('--tensorboard_log_folder', type=str,
                    help='the path of folder which stores the tensorboard log',
                    default='./checkpoint/IPCGANS/%s/tensorboard/'%(TIMESTAMP))

args = parser.parse_args()

# define tensorboard
writer = SummaryWriter(os.path.join(args.tensorboard_log_folder,TIMESTAMP))


check_dir(args.checkpoint)
check_dir(args.saved_model_folder)
check_dir(args.saved_validation_folder)


#step2: define logging output
logger = logging.getLogger("IPCGANS Train")
file_handler = logging.FileHandler(join(args.checkpoint, 'log.txt'), "w")
stdout_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.setLevel(logging.INFO)


def main():
    logger.info("Start to train:\n arguments: %s" % str(args))
    #step3: define transform
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Img_to_zero_center()
    ])
    label_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    #step4: define train/test dataloader
    train_dataset = CACD("train",transforms, label_transforms)
    test_dataset = CACD("test", transforms, label_transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    #step5: define model,optim
    model=IPCGANs(lr=args.learning_rate,age_classifier_path=args.age_classifier_path,gan_loss_weight=args.gan_loss_weight,feature_loss_weight=args.feature_loss_weight,age_loss_weight=args.age_loss_weight)
                  #,feature_extractor_path=args.feature_extractor_path)
    d_optim=model.d_optim
    g_optim=model.g_optim

    for epoch in range(args.max_epoches):


        # if epoch ==0:
        #     model.fea_loss_weight = 5
        #     model.age_loss_weight = 0.1
        #     args.g_iter = 4
        # elif epoch ==1:
        #     args.g_iter = 2
        #     model.fea_loss_weight = 10
        #     model.age_loss_weight = 0.5
        # elif epoch ==2:
        #     model.fea_loss_weight = 20
        #     model.age_loss_weight = 1
        # elif epoch ==3:
        #     model.fea_loss_weight = 20
        #     model.age_loss_weight = 2
        # elif epoch ==5:
        #     model.age_loss_weight = 5
        # elif epoch == 6:
        #     model.age_loss_weight = 10


        idx = 0
        for idx, (source_img_227,source_img_128,true_label_img,true_label_128,true_label_64,fake_label_64, true_label) in enumerate(train_loader,1):

            running_d_loss=None
            running_g_loss=None
            n_iter = epoch * len(train_loader) + idx


            #mv to gpu
            source_img_227=source_img_227.cuda()
            source_img_128=source_img_128.cuda()
            true_label_img=true_label_img.cuda()
            true_label_128=true_label_128.cuda()
            true_label_64=true_label_64.cuda()
            fake_label_64=fake_label_64.cuda()
            true_label=true_label.cuda()

            #train discriminator
            for d_iter in range(args.d_iter):
                #d_lr_scheduler.step()
                d_optim.zero_grad()
                model.train(
                    source_img_227=source_img_227,
                    source_img_128=source_img_128,
                    true_label_img=true_label_img,
                    true_label_128=true_label_128,
                    true_label_64=true_label_64,
                    fake_label_64=fake_label_64,
                    age_label=true_label
                )
                d_loss=model.d_loss
                running_d_loss=d_loss
                d_loss.backward()
                d_optim.step()

            #visualize params
            # for name, param in model.discriminator.named_parameters():
            #     writer.add_histogram("discriminator:%s"%name, param.clone().cpu().detach().numpy(), n_iter)

            #train generator
            for g_iter in range(args.g_iter):
                #g_lr_scheduler.step()
                g_optim.zero_grad()
                model.train(
                    source_img_227=source_img_227,
                    source_img_128=source_img_128,
                    true_label_img=true_label_img,
                    true_label_128=true_label_128,
                    true_label_64=true_label_64,
                    fake_label_64=fake_label_64,
                    age_label=true_label
                )
                g_loss = model.g_loss
                running_g_loss=g_loss
                g_loss.backward()
                g_optim.step()

            # for name, param in model.generator.named_parameters():
            #     writer.add_histogram("generator:%s" % name, param.clone().cpu().detach().numpy(), n_iter)

            format_str = ('step %d/%d, g_loss = %.3f, d_loss = %.3f')
            logger.info(format_str % (idx, len(train_loader),running_g_loss,running_d_loss))

            writer.add_scalars('data/loss', {'G_loss':running_g_loss,'D_loss':running_d_loss}, n_iter)

        # save the parameters at the end of each save interval
        if (epoch+1) % args.save_interval == 0:
            model.save_model(dir=args.saved_model_folder,
                             filename='epoch_%d_iter_%d.pth'%(epoch, idx))
            logger.info('checkpoint has been created!')

        #val step
        if (epoch+1) % args.val_interval == 0:
            save_dir = os.path.join(args.saved_validation_folder, "epoch_%d" % epoch, "idx_%d" % idx)
            check_dir(save_dir)
            for val_idx, (source_img_128, true_label_128) in enumerate(tqdm(test_loader)):
                save_image(Reverse_zero_center()(source_img_128),os.path.join(save_dir,"batch_%d_source.jpg"%(val_idx)))

                pic_list = []
                pic_list.append(source_img_128)
                for age in range(args.age_groups):
                    img = model.test_generate(source_img_128, true_label_128[age])
                    save_image(Reverse_zero_center()(img),os.path.join(save_dir,"batch_%d_age_group_%d.jpg"%(val_idx,age)))
            logger.info('validation image has been created!')

if __name__ == '__main__':
    main()
