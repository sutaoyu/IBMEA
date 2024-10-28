import torch
from torch import nn
from pytorch_metric_learning import losses, miners

try:
    from models import *
    from utils import *
except:
    from src.models import *
    from src.utils import *


class MsLoss(nn.Module):
    def __init__(self, device, thresh=0.5, scale_pos=0.1, scale_neg=40.0):
        super(MsLoss, self).__init__()
        self.device = device
        alpha, beta, base = scale_pos, scale_neg, thresh
        self.loss_func = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]
        labels = torch.arange(emb_train_left.size(0))
        embeddings = torch.cat([emb_train_left, emb_train_right], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.loss_func(embeddings, labels)
        return loss


class InfoNCE_loss(nn.Module):
    def __init__(self, device, temperature=0.05) -> None:
        super().__init__()
        self.device = device
        self.t = temperature

        self.ce_loss = nn.CrossEntropyLoss()

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]

        score = self.sim(emb_train_left, emb_train_right)

        bsize = emb_train_left.size()[0]
        label = torch.arange(bsize, dtype=torch.long).cuda(self.device)

        loss = self.ce_loss(score / self.t, label)
        return loss
