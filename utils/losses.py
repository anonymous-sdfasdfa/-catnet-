import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1,output2, label):
        # output1 = outputs[0]
        # output2 = outputs[1]
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        term1 =  (label)* torch.pow(euclidean_distance, 2)
        term2 =  (1-label)* torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # m1 = torch.mean(term1)
        # m2 = torch.mean(term2)
        loss_contrastive = torch.mean(term1+term2)


        return loss_contrastive