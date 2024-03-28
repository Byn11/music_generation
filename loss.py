import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.functional as F
import numpy as np

class _BaseLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_BaseLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        raise RuntimeError()


class _BaseWeightedLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_BaseWeightedLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        raise RuntimeError()


class NLLLoss(_BaseWeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        nll_loss = F.nll_loss(input, target, weight=self.weight)

        return {'nll': nll_loss}


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, logp=True, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.logp = logp
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if self.logp:
            logpt = input #（2048,50）
        else:
            logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target) #(2048,1)
        logpt = logpt.view(-1) #(2048)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class get_cross_entorpy(nn.Module):
    def __init__(self):
        super(get_cross_entorpy,self).__init__()
    def forward(self,y,target):
        ce=-(target*np.log(y)+(1-target)*np.log(1-y))

        return ce