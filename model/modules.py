from .basic_nets import *
import torch


class attenNet(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = attenNet_basic().to(self.device)

    def load(self,ck_pth):
        checkpoints = torch.load(ck_pth)
        self.net.load_state_dict(checkpoints['arch'])


        # state_dict = torch.load(ck_pth)
        # model_dict = self.net.state_dict()
        # pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.net.load_state_dict(model_dict)

    def inference(self, images):
        outputs = self.net(images)
        return outputs

    def eval(self,dloader):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                images, labels, _, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        return  acc


class CNN(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = CNN_basic().to(self.device)

    def load(self,ck_pth):
        checkpoints = torch.load(ck_pth)
        self.net.load_state_dict(checkpoints['arch'])

    def inference(self, images):
        outputs = self.net(images)
        return outputs

    def eval(self,dloader):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                images, labels, _, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        return  acc
