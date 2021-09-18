import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F


class CriterionCrossEntropyEdgeParsing(nn.Module):
    """
    Weighted CE2P loss for face parsing.
    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """

    def __init__(self, loss_weight=(1.0, 1.0, 1.0), ignore_index=255):
        super(CriterionCrossEntropyEdgeParsing, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.loss_weight = loss_weight

    def forward(self, inputs, target):
        device = target[0].device
        h, w = target[0].size(1), target[0].size(2)

        input_labels = target[1].data.cpu().numpy().astype(np.int64)
        pos_num = np.sum(input_labels == 1).astype(np.float)
        neg_num = np.sum(input_labels == 0).astype(np.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = (weight_neg, weight_pos)
        weights = torch.from_numpy(np.array(weights)).float().to(device)

        edge_p_num = target[1].cpu().numpy().reshape(target[1].size(0), -1).sum(axis=1)
        edge_p_num = np.tile(edge_p_num, [h, w, 1]).transpose(2, 1, 0)
        edge_p_num = torch.from_numpy(edge_p_num).to(device).float()

        scale_parse, scale_edge = inputs
        loss_parse = self.criterion(scale_parse, target[0])
        loss_edge = F.cross_entropy(scale_edge, target[1], weights)
        loss_att_edge = self.criterion_weight(scale_parse, target[0]) * target[1].float()
        loss_att_edge = loss_att_edge / edge_p_num  # only compute the edge pixels
        loss_att_edge = torch.sum(loss_att_edge) / target[1].size(0)  # mean for batchsize

        return self.loss_weight[0] * loss_parse + self.loss_weight[1] * loss_edge + self.loss_weight[2] * loss_att_edge
