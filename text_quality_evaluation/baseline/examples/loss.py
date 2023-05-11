import torch
from torch.nn import ReLU
import torch.nn.functional as F


def compute_kl_loss(predict, target):
    pt_kl = F.kl_div(F.log_softmax(predict, dim=-1), F.softmax(target, dim=-1), reduction='mean')
    pp_kl = 0
    for length in range(len(target)):
        predict_new = predict[length]
        temp = F.kl_div(F.log_softmax(predict_new, dim=-1), F.softmax(predict, dim=-1), reduction='mean')
        pp_kl += temp
    pp_kl /= len(target)
    action = ReLU()
    loss = action(pt_kl - pp_kl)
    return loss


if __name__ == '__main__':
    pass
