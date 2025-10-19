import torch
import torch.nn as nn
import torch.nn.functional as F

class Distill(nn.Module):
    def __init__(self, alpha=1, tem=3):
        super(Distill, self).__init__()
        self.tem = tem
        if 1 > alpha >= 0:
            self.alpha = alpha
        else:
            self.alpha = 1
        self.kiv = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, stu_log, tea_log, labels):
        p_stu = F.log_softmax(stu_log / self.tem, dim = 1)
        p_tea = F.softmax(tea_log / self.tem, dim = 1)
        kl = self.kiv(p_stu, p_tea) * (self.tem**2)
        ce = self.ce(stu_log, labels)

        return self.alpha * ce + (1- self.alpha) * kl
