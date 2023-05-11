import torch


def multilabel_crossentropy(output,label):
    """
    多标签分类的交叉熵
    说明：label和output的shape一致，label的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证output的值域是全体实数，换言之一般情况下output
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出output大于0的类。如有疑问，请仔细阅读并理解
         本文。
    :param output: [B,C]
    :param label:  [B,C]
    :return:
    """
    output = (1-2*label)*output

    #得分变为负1e12
    output_neg = output - label* 1e12
    output_pos = output-(1-label)* 1e12

    zeros = torch.zeros_like(output[:,:1])

    # [B, C + 1]
    output_neg = torch.cat([output_neg,zeros],dim=1)
    # [B, C + 1]
    output_pos = torch.cat([output_pos,zeros],dim=1)


    loss_pos = torch.logsumexp(output_pos,dim=1)
    loss_neg = torch.logsumexp(output_neg,dim=1)
    loss = (loss_neg + loss_pos).sum()

    return loss
