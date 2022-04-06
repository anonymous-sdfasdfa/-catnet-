from utils.loader import cross_validation
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime,date
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class contrastive_train(object):
    """
    basic training class
    """

    def __init__(self, config):
        ## init param
        self.kintype = config.kintype
        self.epoch_num = config.epoch_num
        self.list_path = config.list_path
        self.img_root = config.img_root
        self.show_lstep = config.show_lstep
        self.batch = config.train_batch
        self.model = config.model
        self.lr = config.lr
        self.lr_de = config.lr_decay
        self.mom = config.momentum
        self.lr_mil = config.lr_milestones
        self.sf_sq = config.sf_sequence
        self.model_name = config.model_name
        self.cs = config.cross_shuffle
        self.cr_num = config.cross_num
        self.print_frq = config.prt_fr
        self.sf_aln = config.sf_aln
        self.reload = config.reload
        self.sf_fimg1 = config.sf_fimg1
        self.save_ck = config.save_ck
        self.we_de = config.weight_decay
        self.logs_name = config.logs_name
        self.Dataset = config.Dataset
        self.savemis = config.savemis
        self.test_batch = config.test_batch
        self.save_tacc = config.save_tacc
        self.loss = config.loss
        self.optim = config.optim
        self.dl_sf = config.dataloder_shuffle
        self.description = config.__dict__
        self.gpth = config.gpth
        self.gpu = config.gpu
        ##transform
        if config.trans is not None:
            self.train_transform = config.trans[0]
            self.test_transform = config.trans[1]

        else:
            self.train_transform = config.train_transform
            self.test_transform = config.test_transform
        ## add save model params
        if config.save_ck:
            self.ck_pth = 'data/checkpoints/{}-{}'.format(config.kintype,config.model_name)
            if not os.path.isdir(self.ck_pth):
                os.makedirs(self.ck_pth)

        self.device = torch.device("cuda:{}".format(self.gpu) if torch.cuda.is_available() else "cpu")

        if config.save_graph:
            net = getattr(config.model, config.model_name)(pretrained=self.gpth)
            net.to(self.device)
            graph_in = torch.randn(2, config.imsize[0], config.imsize[1], config.imsize[2]).to(self.device)
            with SummaryWriter('{}/{}/graph'.format(self.logs_name, self.model_name)) as ww:
                ww.add_graph(net, graph_in)
                ww.add_text('{}_parameters'.format(self.model_name), net.__repr__())
                ww.add_text('Training Description', config.des)
                ww.add_text('Training parameters',
                            'Training parameters:{}'.format(self.__dict__))  # loss and optimizer
        self.test_acc = 0

    def object_fun(self):
        # add split '-' name
        model_name_ = self.model_name.split('-')[0]
        net = getattr(self.model, model_name_)(pretrained=self.gpth)
        net.to(self.device)
        criterion = self.loss()
        if self.optim == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, momentum=self.mom, weight_decay=self.we_de)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, weight_decay=self.we_de)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_mil, gamma=self.lr_de)
        return net, criterion, optimizer, scheduler

    def cross_run(self):
        """
        run cross validation and save
        :return:
        """
        with SummaryWriter('{}/{}-{}/{}_kv_cross_description'.format(self.logs_name, self.kintype,self.model_name,
                                                                     datetime.now())) as ww:
            ww.add_text('{}_{}_parameters'.format(self.logs_name, self.model_name),
                        '{}'.format(self.description).replace(',', '<br />'))

        total_acc = 0
        for tra_id, tes_id in cross_validation(self.cr_num):
            print('#' * 10 + 'cross validation{}'.format(self.cr_num + 1 - tes_id[0]) + '#' * 10)
            self.run(tra_id, tes_id, self.cr_num + 1 - tes_id[0])
            acc = self.test_acc
            total_acc += acc

        with SummaryWriter('{}/{}-{}/{}_kv_cross_vali_final'.format(self.logs_name, self.kintype,self.model_name,
                                                                    datetime.now())) as ww:
            ww.add_scalar(tag='final test accuracy',
                          scalar_value=total_acc / self.cr_num)
            print('the accuracy after cross validation is {}'.format(total_acc / self.cr_num))

    def run(self, train_id=None, test_id=None, w_num=1):
        """
        :param train_id: cross validation split list [1,2,3,4]
        :param test_id: [5]
        :param w_num: the number of validation 6-test_id
        :return:
        """

        self.writer = SummaryWriter(
            '{}/{}-{}/{}_kv_cross_vali0{}'.format(self.logs_name, self.kintype, self.model_name, datetime.now(),
                                                  w_num))
        if train_id is None:
            train_id = [1, 2, 3, 4]
            test_id = [5]

        train_set = self.Dataset(self.list_path, self.img_root, train_id, transform=self.train_transform,
                                 cross_shuffle=self.cs, sf_aln=self.sf_aln, sf_fimg1 = self.sf_fimg1,sf_sequence=self.sf_sq)
        test_set = self.Dataset(self.list_path, self.img_root, test_id, transform=self.test_transform, test=True)
        train_loader = DataLoader(train_set, batch_size=self.batch, shuffle=self.dl_sf)
        test_loader = DataLoader(test_set, batch_size=self.test_batch)
        # final_loader = DataLoader(test_set, batch_size=1)
        self.train(train_loader, test_loader, w_num)
        self.writer.close()

    def train(self, train_loader, test_loader, w_num):
        ### get model
        # net = getattr(self.model, self.model_name)()
        # net.to(self.device)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.mom,weight_decay=self.we_de)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_mil, gamma=self.lr_de)

        net, criterion, optimizer, scheduler = self.object_fun()

        ### add model reload
        if self.reload != '':
            checkpoints = torch.load(self.reload)
            net.load_state_dict(checkpoints['arch'])

        global_step = 0
        # net.base.eval()
        epoch = 0
        for epoch in range(self.epoch_num):  # loop over the dataset multiple times
            try:
                net.KMM.train()
            except:
                pass
            # net.KMM.train()
            net.cn_f1.train()
            net.cn_f2.train()


            print('epoch: {}'.format(epoch))
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs1,inputs2, labels,condition, img1_n, img2_n = data
                inputs1,inputs2, labels,condition = inputs1.to(self.device),inputs2.to(self.device),\
                                                    labels.to(self.device),condition.to(self.device)

                # labels = 1-labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                k_p,k_c = net(inputs1,inputs2,condition)
                loss = criterion(k_p,k_c, labels)
                loss.backward()
                optimizer.step()

                ############# not sure
                # scheduler.step()
                # for param_group in optimizer.param_groups:
                # 	print('epoch:{}'.format(epoch))
                # 	print(param_group['lr'])
                # print statistics
                # adjust_learning_rate(optimizer,epoch,self.lr)

                # print statistics
                running_loss += loss.item()
                # running_loss = loss.item()

                if i % self.show_lstep == (self.show_lstep - 1):
                    # print every 20 mini-batches
                    print('[epoch %d, global step %5d] loss: %.3f' %
                          (epoch + 1, global_step + 1, running_loss / self.show_lstep))

                    # ...log the running loss
                    self.writer.add_scalar(tag='training loss',
                                           scalar_value=running_loss / self.show_lstep,
                                           global_step=global_step)

                    running_loss = 0.0

                # update global step
                global_step += 1
            if self.save_tacc:
                self.acc_records(net, train_loader, epoch, t='train')
            if (epoch + 1) % self.print_frq == 0:
                self.acc_records(net, test_loader, epoch, t='test')

            # update learning rate
            scheduler.step()
        self.mis_record(net, test_loader, w_num, self.savemis)
        # self.acc_records(net, test_loader, epoch, t='test')
        ## save model
        if self.save_ck:
            torch.save({
                'epoch': epoch,
                'arch': net.state_dict(),
                # 'optimizer_state': optimizer.state_dict()
            }, '{}/{}-{}-cv{}-{}-{}.pth'.format(self.ck_pth,
                                                 self.kintype, self.model_name,w_num, date.today(),
                                                datetime.now().hour))

    def acc_records(self, net, dloader, epoch, t='train'):
        """
        :param dloader: train loader or test loader
        :param epoch: training epochs
        :param t: 'train' or 'test'
        :return: record accuracy
        """
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in dloader:
                # images, labels, _, _ = data
                # images, labels = images.to(self.device), labels.to(self.device)
                inputs1, inputs2, labels, condition, img1_n, img2_n = data
                inputs1, inputs2, labels, condition = inputs1.to(self.device), inputs2.to(self.device), \
                                          labels.to(self.device), condition.to(self.device)

                # labels = 1 - labels
                k_p,k_c = net(inputs1,inputs2,condition)
                loss_score = F.pairwise_distance(k_p, k_c)
                predicted = loss_score < 10
                predicted = predicted.type(torch.int64)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %s images: %d %%' % (t,
                                                                   100 * correct / total))
        acc = correct / total
        self.writer.add_scalar(tag='{}_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)

    def mis_record(self, net, dloader, w_num, savemis):
        """
        mismatch record
        :param dloader: testloader
        :param w_num: writer cross valid number: 0-5
        :return:
        TODO: seperate the imags into image1 and image2
        """
        total = 0
        correct = 0
        net.eval()
        with torch.no_grad():
            for data in dloader:
                inputs1, inputs2, labels, condition, img1_n, img2_n = data
                inputs1, inputs2, labels, condition = inputs1.to(self.device), inputs2.to(self.device), \
                                                      labels.to(self.device), condition.to(self.device)
                # labels = 1 - labels
                k_p,k_c = net(inputs1, inputs2, condition)
                dist = F.pairwise_distance(k_p,k_c)
                predicted = dist < 1
                predicted = predicted.type(torch.int64)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #  add writer for misclassified pairs
                if savemis:
                    for i, val in enumerate(predicted == labels):
                        if not val:
                            # probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(predicted, outputs)]
                            # probs = [it.item() for it in F.softmax(outputs[i], dim=0)]
                            # img1, img2 = images[i:i + 1][:, 0:3, :, :], images[i:i + 1][:, 3:6, :, :]
                            img1,img2 = inputs1[i],inputs2[i]
                            img1 = img1 / 2 + 0.5
                            img2 = img2 / 2 + 0.5
                            imgs = torch.cat((img1, img2))
                            if predicted[i] == 1:
                                self.writer.add_images(tag='{}/false_positive_cross_0{}_{}/{}distance:({:.4})'.
                                                       format(self.model_name, w_num, img1_n[i], img2_n[i],
                                                              dist[i]),
                                                       img_tensor=imgs)
                            else:
                                self.writer.add_images(tag='{}/false_negative_cross_0{}_{}/{}distance:({:.4})'.
                                                       format(self.model_name, w_num, img1_n[i], img2_n[i],
                                                              dist[i]),
                                                       img_tensor=imgs)
        self.test_acc = correct / total

    def test(self):
        pass


class record_acc():
    def __init__(self):
        self.acc_dict = {1:[],
                         2:[],
                         3:[],
                         4:[],
                         5:[]}
        self.epoch = []
    def update(self,cross, epoch, acc):
        self.acc_dict[cross].append(acc)
        self.epoch.append(epoch)

    def acc_print(self):
        accs = np.zeros(len(self.acc_dict[1]))
        for cs in self.acc_dict:
            accs += np.array(self.acc_dict[cs])

        ls = list(map(lambda l: '{:.04f}'.format(l),accs / 5))
        print(list(zip(self.epoch,ls)))
        print( )


class convfc_train(object):
    """
    basic training class
    """

    def __init__(self, config):
        ## init param
        self.kintype = config.kintype
        self.epoch_num = config.epoch_num
        self.list_path = config.list_path
        self.img_root = config.img_root
        self.show_lstep = config.show_lstep
        self.batch = config.train_batch
        self.model = config.model
        self.lr = config.lr
        self.lr_de = config.lr_decay
        self.mom = config.momentum
        self.lr_mil = config.lr_milestones
        self.sf_sq = config.sf_sequence
        self.model_name = config.model_name
        self.cs = config.cross_shuffle
        self.cr_num = config.cross_num
        self.print_frq = config.prt_fr
        self.sf_aln = config.sf_aln
        self.reload = config.reload
        self.sf_fimg1 = config.sf_fimg1
        self.save_ck = config.save_ck
        self.we_de = config.weight_decay
        self.logs_name = config.logs_name
        self.Dataset = config.Dataset
        self.savemis = config.savemis
        self.test_batch = config.test_batch
        self.save_tacc = config.save_tacc
        self.loss = config.loss
        self.optim = config.optim
        self.dl_sf = config.dataloder_shuffle
        self.description = config.__dict__
        self.gpth = config.gpth
        self.gpu = config.gpu
        ##transform
        if config.trans is not None:
            self.train_transform = config.trans[0]
            self.test_transform = config.trans[1]

        else:
            self.train_transform = config.train_transform
            self.test_transform = config.test_transform
        ## add save model params
        if config.save_ck:
            self.ck_pth = 'data/checkpoints/{}-{}'.format(config.kintype,config.model_name)
            if not os.path.isdir(self.ck_pth):
                os.makedirs(self.ck_pth)

        self.device = torch.device("cuda:{}".format(self.gpu) if torch.cuda.is_available() else "cpu")

        if config.save_graph:
            net = getattr(config.model, config.model_name)(pretrained=self.gpth)
            net.to(self.device)
            graph_in = torch.randn(2, config.imsize[0], config.imsize[1], config.imsize[2]).to(self.device)
            with SummaryWriter('{}/{}/graph'.format(self.logs_name, self.model_name)) as ww:
                ww.add_graph(net, graph_in)
                ww.add_text('{}_parameters'.format(self.model_name), net.__repr__())
                ww.add_text('Training Description', config.des)
                ww.add_text('Training parameters',
                            'Training parameters:{}'.format(self.__dict__))  # loss and optimizer
        self.test_acc = 0
        self.record_acc_dict = record_acc()

    def object_fun(self):
        # add split '-' name
        model_name_ = self.model_name.split('-')[0]
        net = getattr(self.model, model_name_)(pretrained=self.gpth)
        net.to(self.device)
        criterion = self.loss()
        if self.optim == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, momentum=self.mom, weight_decay=self.we_de)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, weight_decay=self.we_de)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_mil, gamma=self.lr_de)
        return net, criterion, optimizer, scheduler

    def cross_run(self):
        """
        run cross validation and save
        :return:
        """
        with SummaryWriter('{}/{}-{}/{}_kv_cross_description'.format(self.logs_name, self.kintype,self.model_name,
                                                                     datetime.now())) as ww:
            ww.add_text('{}_{}_parameters'.format(self.logs_name, self.model_name),
                        '{}'.format(self.description).replace(',', '<br />'))

        total_acc = 0
        for tra_id, tes_id in cross_validation(self.cr_num):
            print('#' * 10 + 'cross validation{}'.format(self.cr_num + 1 - tes_id[0]) + '#' * 10)
            self.run(tra_id, tes_id, self.cr_num + 1 - tes_id[0])
            acc = self.test_acc
            total_acc += acc

        with SummaryWriter('{}/{}-{}/{}_kv_cross_vali_final'.format(self.logs_name, self.kintype,self.model_name,
                                                                    datetime.now())) as ww:
            ww.add_scalar(tag='final test accuracy',
                          scalar_value=total_acc / self.cr_num)
            print('the accuracy after cross validation is {}'.format(total_acc / self.cr_num))

    def run(self, train_id=None, test_id=None, w_num=1):
        """
        :param train_id: cross validation split list [1,2,3,4]
        :param test_id: [5]
        :param w_num: the number of validation 6-test_id
        :return:
        """

        self.writer = SummaryWriter(
            '{}/{}-{}/{}_kv_cross_vali0{}'.format(self.logs_name, self.kintype, self.model_name, datetime.now(),
                                                  w_num))
        if train_id is None:
            train_id = [1, 2, 3, 4]
            test_id = [5]

        train_set = self.Dataset(self.list_path, self.img_root, train_id, transform=self.train_transform,
                                 cross_shuffle=self.cs, sf_aln=self.sf_aln, sf_fimg1 = self.sf_fimg1,sf_sequence=self.sf_sq)
        test_set = self.Dataset(self.list_path, self.img_root, test_id, transform=self.test_transform, test=True)
        train_loader = DataLoader(train_set, batch_size=self.batch, shuffle=self.dl_sf)
        test_loader = DataLoader(test_set, batch_size=self.test_batch)
        # final_loader = DataLoader(test_set, batch_size=1)
        self.train(train_loader, test_loader, w_num)
        self.writer.close()

    def train(self, train_loader, test_loader, w_num):
        ### get model
        # net = getattr(self.model, self.model_name)()
        # net.to(self.device)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.mom,weight_decay=self.we_de)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_mil, gamma=self.lr_de)

        net, criterion, optimizer, scheduler = self.object_fun()

        ### add model reload
        if self.reload != '':
            checkpoints = torch.load(self.reload)
            net.load_state_dict(checkpoints['arch'])

        global_step = 0
        # net.base.eval()
        epoch = 0
        for epoch in range(self.epoch_num):  # loop over the dataset multiple times

            net.train()

            print('epoch: {}'.format(epoch))
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs1,inputs2, labels,condition, img1_n, img2_n = data
                inputs1,inputs2, labels,condition = inputs1.to(self.device),inputs2.to(self.device),\
                                                    labels.to(self.device),condition.to(self.device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs= net(inputs1,inputs2,condition)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # running_loss = loss.item()

                if i % self.show_lstep == (self.show_lstep - 1):
                    # print every 20 mini-batches
                    print('[epoch %d, global step %5d] loss: %.3f' %
                          (epoch + 1, global_step + 1, running_loss / self.show_lstep))

                    # ...log the running loss
                    self.writer.add_scalar(tag='training loss',
                                           scalar_value=running_loss / self.show_lstep,
                                           global_step=global_step)

                    running_loss = 0.0

                # update global step
                global_step += 1
            # if self.save_tacc:
            #     self.acc_records(net, train_loader, epoch, t='train')
            if (epoch + 1) % self.print_frq == 0:
                self.acc_records(net, test_loader, epoch+1, w_num,t='test')

            # update learning rate
            scheduler.step()
        # self.mis_record(net, test_loader, w_num, self.savemis)
        # self.acc_records(net, test_loader, epoch, t='test')
        ## save model
        if self.save_ck:
            torch.save({
                'epoch': epoch,
                'arch': net.state_dict(),
                # 'optimizer_state': optimizer.state_dict()
            }, '{}/{}-{}-cv{}-{}-{}.pth'.format(self.ck_pth,
                                                 self.kintype, self.model_name,w_num, date.today(),
                                                datetime.now().hour))

    def acc_records(self, net, dloader, epoch, w_num,t='train'):
        """
        :param dloader: train loader or test loader
        :param epoch: training epochs
        :param t: 'train' or 'test'
        :return: record accuracy
        """
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in dloader:

                inputs1, inputs2, labels, condition, img1_n, img2_n = data
                inputs1, inputs2, labels, condition = inputs1.to(self.device), inputs2.to(self.device), \
                                          labels.to(self.device), condition.to(self.device)

                outputs =net(inputs1,inputs2,condition)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %s images: %d %%' % (t,
                                                                   100 * correct / total))
        acc = correct / total
        self.record_acc_dict.update(w_num,epoch,acc)


        # self.writer.add_scalar(tag='{}_accuracy/epoch'.format(t),
        #                        scalar_value=acc,
        #                        global_step=epoch)

    def mis_record(self, net, dloader, w_num, savemis):
        """
        mismatch record
        :param dloader: testloader
        :param w_num: writer cross valid number: 0-5
        :return:
        TODO: seperate the imags into image1 and image2
        """
        total = 0
        correct = 0
        net.eval()
        with torch.no_grad():
            for data in dloader:
                inputs1, inputs2, labels, condition, img1_n, img2_n = data
                inputs1, inputs2, labels, condition = inputs1.to(self.device), inputs2.to(self.device), \
                                                      labels.to(self.device), condition.to(self.device)
                # labels = 1 - labels
                k_p,k_c = net(inputs1, inputs2, condition)
                dist = F.pairwise_distance(k_p,k_c)
                predicted = dist < 1
                predicted = predicted.type(torch.int64)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #  add writer for misclassified pairs
                if savemis:
                    for i, val in enumerate(predicted == labels):
                        if not val:
                            # probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(predicted, outputs)]
                            # probs = [it.item() for it in F.softmax(outputs[i], dim=0)]
                            # img1, img2 = images[i:i + 1][:, 0:3, :, :], images[i:i + 1][:, 3:6, :, :]
                            img1,img2 = inputs1[i],inputs2[i]
                            img1 = img1 / 2 + 0.5
                            img2 = img2 / 2 + 0.5
                            imgs = torch.cat((img1, img2))
                            if predicted[i] == 1:
                                self.writer.add_images(tag='{}/false_positive_cross_0{}_{}/{}distance:({:.4})'.
                                                       format(self.model_name, w_num, img1_n[i], img2_n[i],
                                                              dist[i]),
                                                       img_tensor=imgs)
                            else:
                                self.writer.add_images(tag='{}/false_negative_cross_0{}_{}/{}distance:({:.4})'.
                                                       format(self.model_name, w_num, img1_n[i], img2_n[i],
                                                              dist[i]),
                                                       img_tensor=imgs)
        self.test_acc = correct / total

    def test(self):
        pass

