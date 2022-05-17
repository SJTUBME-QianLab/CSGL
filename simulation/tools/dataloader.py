from __future__ import print_function

import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
import random
import os
import itertools
from sklearn.model_selection import StratifiedKFold
from scipy import sparse


class MethylationData(data.Dataset):
    def __init__(self, data_dir, data_name, device):
        self.root = data_dir
        self.data_name = data_name

        if 'cov' in data_name:
            self.Xall = np.load(os.path.join(self.root, 'X_normL2.npy'))
            self.Yall = np.load(os.path.join(self.root, 'Y.npy'))
            # self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            # self.split = self.skf.split(self.Xall, self.Yall)
            self.Xall = torch.from_numpy(self.Xall).float()  # 600, 3000, dtype=torch.float32
            self.Yall = torch.from_numpy(self.Yall.T).long()  # 600, dtype=torch.int64
        else:
            raise ValueError(data_name)
        if 'cov' in data_name:
            pair = sparse.load_npz(os.path.join(self.root, 'pair_sparse.npz'))
            self.pair = torch.tensor(pair.toarray(), dtype=torch.long).to(device)

        self.feature_num = self.Xall.shape[1]  # 3000
        self.class_num = len(np.unique(self.Yall))  # 2
        self.Xall, self.Yall = self.Xall.to(device), self.Yall.to(device)

        self.fea_info = pd.read_csv(os.path.join(self.root, 'basic_info.csv'))
        # fea_name, gp_label, loc, true_01, isol_01, beta
        self.fea_idxes = self.fea_info.index.values
        self.locs = self.fea_info['loc'].values
        self.fea_name = self.fea_info['fea_name'].values
        self.gp_info = self.fea_info['gp_label'].values
        self.true_beta = self.fea_info['beta'].values
        self.TorF = self.fea_info['true_01'].values

        self.gp_idx_list = [np.where(self.fea_info['gp_label'] == xx)[0] for xx in np.unique(self.gp_info)]
        # self.gp_idx_list = [torch.from_numpy(xx).long().to(device) for xx in self.gp_idx_list]
        # self.locs = torch.from_numpy(self.locs).float().to(device)
        print('Feature information imported.')

    def get_basic_items(self, idx):
        inputs = self.Xall[idx, :]
        labels = self.Yall[idx]
        return inputs, labels

    def __getitem__(self, idx):
        return self.get_basic_items(idx)

    def __len__(self):
        return len(self.Xall)


class SubData(data.Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset

        self.X = dataset.Xall[idx, :]
        self.Y = dataset.Yall[idx]
        self.pair = torch.tensor(dataset.pair.toarray()[idx, :][:, idx], dtype=torch.long).to(self.Y.device)

    def get_basic_items(self, idx):
        inputs = self.X[idx, :]
        labels = self.Y[idx]
        return inputs, labels

    def __getitem__(self, idx):
        return self.get_basic_items(idx)

    def __len__(self):
        return len(self.X)


class Dataloader:
    def __init__(self, dataset):
        self.dataset = dataset
        # self.fold = fold

        # if 'cov' in self.dataset.data_name:
        #     self.train_idx_list, self.test_idx_list = [], []
        #     save_idx_dir = os.path.join(self.dataset.root, 'test_idx')
        #     os.makedirs(save_idx_dir, exist_ok=True)
        #     for i, (train, test) in enumerate(self.dataset.split):
        #         self.train_idx_list.append(train)
        #         self.test_idx_list.append(test)
        #         if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
        #             test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
        #             assert (test_save == test).all()
        #         else:
        #             np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")
        #
        # elif self.dataset.data_name == 'LUAD':
        #     pass
        # else:
        #     raise ValueError(self.dataset.data_name)

    def get_individual(self, batch_size, shuffle=True):
        # if partition == 'train':
        #     dataset = SubData(self.dataset, self.train_idx_list[self.fold])
        # elif partition == 'test':
        #     dataset = SubData(self.dataset, self.test_idx_list[self.fold])
        # else:
        #     raise ValueError
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(len(self.dataset), batch_size), shuffle=shuffle)
        return loader

    def get_pair(self, batch_size, shuffle=True):
        # if partition == 'train':
        #     dataset = SubData(self.dataset, self.train_idx_list[self.fold])
        # elif partition == 'test':
        #     dataset = SubData(self.dataset, self.test_idx_list[self.fold])
        # else:
        #     raise ValueError
        X, Y, sub_pair = self.dataset.Xall, self.dataset.Yall, self.dataset.pair
        batch_size = min(len(X), batch_size)  # 400
        load = []
        idx = np.arange(len(X))
        if shuffle:
            idx = np.random.permutation(idx)
        for index in range(len(idx) // batch_size + 1):
            c_idx = np.arange((index * batch_size), (min(len(idx), (index + 1) * batch_size)))
            if len(c_idx) == 0:
                continue
            choose = idx[c_idx]
            input = X[choose, :]
            label = Y[choose]
            mask12 = sub_pair[choose, :][:, choose]
            load.append((input, label, mask12))
        return load

    # def get_whole(self):
    #     dataset_tr = SubData(self.dataset, self.train_idx_list[self.fold])
    #     dataset_te = SubData(self.dataset, self.test_idx_list[self.fold])
    #     return dataset_tr.X, dataset_tr.Y, dataset_te.X, dataset_te.Y

