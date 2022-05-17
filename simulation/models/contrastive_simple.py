"""
Original:
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
It also supports the unsupervised contrastive loss in SimCLR
!!!!!!!!! pair-wise !!!!!!!!!!!
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLossPairPNMask(nn.Module):
    def __init__(self, temperature=0.07, mask_weight=1.0):
        super(SupConLossPairPNMask, self).__init__()
        self.base_temperature = 0.07
        self.temperature = temperature
        self.mask_weight = mask_weight

    def forward(self, features, labels=None, pair=None):
        """
        :param features: hidden vector of shape [bsz, ...].
        :param labels: ground truth of shape [bsz].
        :param mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        :return: A loss scalar.
        """
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)  # torch.Size([16, 128])

        batch_size = features.shape[0]
        assert labels is not None
        labels = labels.contiguous().view(-1, 1)  # bsz*1
        assert labels.shape[0] == batch_size
        mask_label = torch.eq(labels, labels.T).float().to(device)  # (i,j)=1表示标签相同
        maskn_label = 1 - mask_label  # (i,j)=1表示标签不同
        mask_pair = torch.ones((batch_size, batch_size)).float().to(device)
        if pair is not None:
            mask_pair += pair * self.mask_weight  # 配对的样本处=1+权重（只会是neg对）
        assert maskn_label[torch.where(mask_pair > 1)].sum() == pair.sum()  # pair处的样本都是neg对

        # compute logits
        anchor_dot_contrast = torch.div(  # 逐元素除以self.temperature 0.07
            torch.matmul(features, features.T),  # 对称阵，对角是1
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 每行的最大值，应该都是1/0.07=14.2857
        logits = anchor_dot_contrast - logits_max.detach()  # 这样对角都是0，其他是负数

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_label),  # [400]的全1矩阵
            1,
            torch.arange(batch_size).view(-1, 1).to(device),  # 0~31 变成列向量torch.Size([32, 1])
            0
        )  # 只有对角线是0，其他都是1
        mask = mask_label * logits_mask  # 32,32的矩阵，其中对角线都是0
        # maskn = maskn_label * logits_mask  # 本来就对角线都是0了，不用乘

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 除法转换成减法

        # compute mean of log-likelihood over positive 只看配对pos样本的log_prob
        mean_log_prob_pos = (mask * mask_pair * log_prob).sum(1) / mask.sum(1)  # mask * mask_pair == mask
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        mean_log_prob_neg = (maskn_label * mask_pair * log_prob).sum(1) / maskn_label.sum(1)
        lossn = - (self.temperature / self.base_temperature) * mean_log_prob_neg
        lossn = lossn.view(1, batch_size).mean()

        return loss, lossn


class SupConLossPair(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLossPair, self).__init__()
        self.base_temperature = 0.07
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        :param features: hidden vector of shape [bsz, ...].
        :param labels: ground truth of shape [bsz].
        :param mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        :return: A loss scalar.
        """
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)  # torch.Size([16, 128])

        batch_size = features.shape[0]
        assert labels is not None
        labels = labels.contiguous().view(-1, 1)  # bsz*1
        assert labels.shape[0] == batch_size
        mask_label = torch.eq(labels, labels.T).float().to(device)  # (i,j)=1表示标签相同

        # compute logits
        anchor_dot_contrast = torch.div(  # 逐元素除以self.temperature 0.07
            torch.matmul(features, features.T),  # 对称阵，对角是1
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 每行的最大值，应该都是1/0.07=14.2857
        logits = anchor_dot_contrast - logits_max.detach()  # 这样对角都是0，其他是负数

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_label),  # [400]的全1矩阵
            1,
            torch.arange(batch_size).view(-1, 1).to(device),  # 0~31 变成列向量torch.Size([32, 1])
            0
        )  # 只有对角线是0，其他都是1
        mask = mask_label * logits_mask  # 32,32的矩阵，其中对角线都是0
        # maskn = maskn_label * logits_mask  # 本来就对角线都是0了，不用乘

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 除法转换成减法

        # compute mean of log-likelihood over positive 只看配对pos样本的log_prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # mask * mask_pair == mask
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss


class SupConLossPairNegMask(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLossPairNegMask, self).__init__()
        self.base_temperature = 0.07
        self.temperature = temperature

    def forward(self, features, mask=None):
        """
        :param features: hidden vector of shape [bsz, ...].
        :param mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                comes from the same subject as sample i. Can be asymmetric.
        :return: A loss scalar.
        """
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)  # torch.Size([16, 128])

        indi_samples = torch.where(mask.sum(1) > 0)[0]
        features = features[indi_samples, :]
        mask = mask[indi_samples, :][:, indi_samples]
        assert torch.eq(mask, mask.T).all()
        batch_size = features.shape[0]

        # compute logits
        anchor_dot_contrast = torch.div(  # 逐元素除以self.temperature 0.07
            torch.matmul(features, features.T),  # 对称阵，对角是1
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 每行的最大值，应该都是1/0.07=14.2857
        logits = anchor_dot_contrast - logits_max.detach()  # 这样对角都是0，其他是负数

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # [400]的全1矩阵
            1,
            torch.arange(batch_size).view(-1, 1).to(device),  # 0~31 变成列向量torch.Size([32, 1])
            0
        )  # 只有对角线是0，其他都是1

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 除法转换成减法

        mean_log_prob_neg = (mask * log_prob).sum(1) / mask.sum(1)
        lossn = - (self.temperature / self.base_temperature) * mean_log_prob_neg
        lossn = lossn.view(1, batch_size).mean()

        return lossn
