

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, n_trans, ad_temp=1.0):
        super().__init__()

        self.n_trans = n_trans
        self.ad_temp = ad_temp


# anomaly scores 
class MultiheadCrossEntropyAnomalyScore(Loss):
    def forward(self, mt_zs_ce, mt_labels):
        n_heads = len(mt_zs_ce)
        minibatch_size = mt_zs_ce[0].shape[0]
        ori_size = minibatch_size // self.n_trans

        ce = 0.
        for t_ind, zs_ce in enumerate(mt_zs_ce):
            prob = F.softmax(self.ad_temp*zs_ce, dim=1)
            prob_t = prob[np.arange(minibatch_size), mt_labels[:, t_ind]]
            prob_t[prob_t < 1e-9] = 1e-9 # prevents NaN
            neg_logp_t = -torch.log(prob_t)

            ce += torch.reshape(neg_logp_t, (ori_size, self.n_trans)).sum(1) # (36,)
        ce /= n_heads
        return ce


class LatentGaussianRepresentationAnomalyScore(Loss):
    def forward(self, zs, means=None):
        if means is None:
            means = zs.mean(0).unsqueeze(0).detach()
        diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
        logp_sz = F.log_softmax(-self.ad_temp*diffs, dim=2)
        # logp_sz = -torch.diagonal(logp_sz, dim1=1, dim2=2).reshape(batch_range).detach()
        logp_sz = -torch.diagonal(logp_sz, dim1=1, dim2=2).sum(1).detach()
        return logp_sz


class MultiheadCrossEntropyAgainstUniformAnomalyScore(Loss):
    def forward(self, mt_zs_ce):
        n_heads = len(mt_zs_ce)
        minibatch_size = mt_zs_ce[0].shape[0]
        ori_size = minibatch_size // self.n_trans

        ce = 0
        for t_ind, zs_ce in enumerate(mt_zs_ce):
            # for each head
            t_classes = zs_ce.shape[1]
            uniform_dist = torch.ones((minibatch_size, t_classes)).to(zs_ce)/t_classes
            ce -= (F.log_softmax(zs_ce, dim=1) * uniform_dist).sum(1)
        ce /= n_heads

        ce = torch.reshape(ce, (ori_size, self.n_trans)).sum(1)
        return ce


# losses
class MultiheadCrossEntropy(Loss):
    def forward(self, mt_zs_ce, mt_labels):
        n_heads = len(mt_zs_ce)
        minibatch_size = mt_zs_ce[0].shape[0]
        ori_size = minibatch_size // self.n_trans

        ce = 0.
        for t_ind, zs_ce in enumerate(mt_zs_ce):
            prob = F.softmax(self.ad_temp*zs_ce, dim=1)
            prob_t = prob[np.arange(minibatch_size), mt_labels[:, t_ind]]
            prob_t[prob_t < 1e-9] = 1e-9 # prevents NaN
            neg_logp_t = -torch.log(prob_t)

            ce += neg_logp_t
        ce /= n_heads
        return ce


class MultiheadCrossEntropyZeroDriven(Loss):
    def forward(self, mt_zs_ce, mt_labels):
        n_heads = len(mt_zs_ce)
        minibatch_size = mt_zs_ce[0].shape[0]
        ori_size = minibatch_size // self.n_trans

        ce = 0.
        for t_ind, zs_ce in enumerate(mt_zs_ce):
            prob = F.softmax(self.ad_temp*zs_ce, dim=1)
            prob_t = prob[np.arange(minibatch_size), mt_labels[:, t_ind]]
            prob_t[prob_t < 1e-9] = 1e-9 # prevents NaN
            neg_logp_t = -torch.log(1-prob_t)

            ce += neg_logp_t 
        ce /= n_heads
        return ce


class MultiheadCrossEntropyAgainstUniform(Loss):
    def forward(self, mt_zs_ce):
        n_heads = len(mt_zs_ce)
        minibatch_size = mt_zs_ce[0].shape[0]

        ce = 0
        for t_ind, zs_ce in enumerate(mt_zs_ce):
            # for each head
            t_classes = zs_ce.shape[1]
            uniform_dist = torch.ones((minibatch_size, t_classes)).to(zs_ce)/t_classes
            ce -= (F.log_softmax(zs_ce, dim=1) * uniform_dist).sum(1)
        ce /= n_heads
        return ce