"""
"""

import pandas as pd
import numpy as np
import random
from scipy import stats


class GetData:
    def __init__(self, cov_kind, y_kind, n, rho=None):
        self.cov_kind = cov_kind
        self.y_kind = y_kind
        self.Xs = 2.
        assert self.cov_kind in ['Gaussian', 'Laplacian', 'eye']
        assert self.y_kind in ['logistic', 'linear']
        assert (rho and self.cov_kind == 'Laplacian') or self.cov_kind != 'Laplacian'
        self.rho = rho
        self.n = n

    def get_L(self, loc):
        p = len(loc)
        ss = np.std(np.diff(loc))
        Sig = np.eye(p)
        for i in range(p):
            for j in range(i):
                if self.cov_kind == 'Gaussian':
                    Sig[i, j] = np.exp(- (loc[i] - loc[j]) ** 2 / (2 * ss ** 2))
                elif self.cov_kind == 'Laplacian':
                    Sig[i, j] = np.power(self.rho, - abs(loc[i] - loc[j]) / (2 * ss))
                Sig[j, i] = Sig[i, j]

        return Sig

    def get_X(self, Sig):
        mu = - 0.1 * np.ones(Sig.shape[0])
        tg = self.Xs * np.random.multivariate_normal(mu, Sig, self.n)
        assert tg.shape[0] == self.n
        X = 1. / (1. + np.exp(-tg))

        return X

    def get_Y(self, X, beta):
        Y0 = np.dot(X, beta)
        if self.y_kind == 'logistic':
            p = 1. / (1. + np.exp(-Y0))
            Y = stats.bernoulli.rvs(p, size=(self.n, len(p)))
        elif self.y_kind == 'linear':
            eps = np.random.normal(loc=0, scale=np.std(Y0) / 10, size=self.n)
            bias = np.median(Y0)
            Y = ((Y0 + eps) > bias).astype(float)
        else:
            raise ValueError(self.y_kind)
        return Y

    def get_data(self, all_info):
        Xg = []
        betag = []
        for ddi in all_info:
            cov = self.get_L(ddi['coordinate'])
            Xg.append(self.get_X(cov))
            betag.append(ddi['beta'])

        X = np.concatenate(Xg, axis=1)  # sample * feature
        beta = np.concatenate(betag, axis=0)
        Y = self.get_Y(X, beta=beta)

        return X, Y


class Coordinates:
    def __init__(self, distribution):
        self.distribution = distribution

    def interval(self, num, **kwargs):
        if self.distribution == 'uniform':
            lower, upper = kwargs['lower'], kwargs['upper']
            intervals = [int(random.uniform(lower, upper)) for i in range(num)]
        elif self.distribution == 'randint':
            lower, upper = kwargs['lower'], kwargs['upper']
            intervals = [random.randint(lower, upper) for i in range(num)]
        elif self.distribution == 'normal':
            mean, std = kwargs['mean'], kwargs['std']
            intervals = [int(random.normalvariate(mean, std)) for i in range(num)]
        else:
            raise ValueError(self.distribution)

        return intervals

    def get_sites_2parts(self, num, true_idx):
        clustered = self.interval(len(true_idx) - 1, lower=5, upper=100)
        others = self.interval(num - len(true_idx) + 1, lower=50, upper=1000)
        intervals = np.zeros((num, ))
        intervals[true_idx[1:]] = clustered
        other_idx = sorted(set(range(num)) - set(true_idx))
        intervals[other_idx + [true_idx[0]]] = others
        assert np.min(intervals) > 0
        positions = [int(np.sum(intervals[:i])) for i in range(1, len(intervals) + 1)]
        return positions

    def get_sites_even(self, num):
        clustered = self.interval(num, lower=5, upper=200)
        positions = [int(np.sum(clustered[:i])) for i in range(1, len(clustered) + 1)]
        return positions


def concat_data(all_info):
    df_all = []
    for ddi in all_info:
        dfi = pd.DataFrame({
            'fea_name': ddi['fea_name'],
            'gp_label': [ddi['gp_label'] for i in range(ddi['gp_dim'])],
            'loc': ddi['coordinate'],
            'true_01': np.zeros(ddi['gp_dim']),
            'isol_01': np.zeros(ddi['gp_dim']),
            'beta': ddi['beta']
        })
        dfi.loc[ddi['true_idx'], 'true_01'] = 1.0
        dfi.loc[ddi['isol_idx'], 'isol_01'] = 1.0
        df_all.append(dfi)

    df_all = pd.concat(df_all, axis=0)
    return df_all

