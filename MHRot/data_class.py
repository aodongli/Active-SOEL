

import numpy as np
import torch
from torch.utils.data import Dataset


class MultitaskDataset(Dataset):
    def __init__(self, x_aug, mt_labels, y, num_trans, test=False):
        self.x_aug = x_aug
        self.mt_labels = mt_labels
        self.y = y
        self.num_trans = num_trans
        self.num_aug_samples = self.x_aug.shape[0]
        self.num_ori_samples = self.num_aug_samples // self.num_trans
        self.test = test

        self.p_labels = np.zeros_like(self.y)
        self.weights = np.ones_like(self.y)

    def __len__(self):
        return self.num_ori_samples

    def augment_indices(self, idx):
        return np.arange(self.num_trans) + idx*self.num_trans

    def __getitem__(self, idx):
        data = self.x_aug[self.augment_indices(idx)]
        mt_labels = self.mt_labels[self.augment_indices(idx)]
        if self.test:
            ori_labels = self.y[idx]
            p_labels = self.p_labels[idx]
            weights = self.weights[idx]
        else:
            ori_labels = self.y[self.augment_indices(idx)]
            p_labels = self.p_labels[self.augment_indices(idx)]
            weights = self.weights[self.augment_indices(idx)]

        return {'id': idx, 
                'data': data, 
                'mt_labels': mt_labels, 
                'gt_labels': ori_labels,
                'p_labels': p_labels,
                'weights': weights}