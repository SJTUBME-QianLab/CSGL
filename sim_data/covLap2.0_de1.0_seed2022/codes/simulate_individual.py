"""
"""

import pandas as pd
import os
import random
import argparse
import numpy as np
from gen_data import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def build_args():
    parser = argparse.ArgumentParser('Arguments Setting.')

    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--note', default='', type=str, help='Note of this simulation')
    parser.add_argument('--save_dir', default='/home/data/tangxl/ContrastSGL/sim_data/', type=str, help='Save directory')
    parser.add_argument('--y_dis', default='linear', type=str, choices=['linear', 'logistic'],
                        help='The distribution of Y')
    parser.add_argument('--cov', default='Gaussian', type=str, choices=['Gaussian', 'Laplacian'],
                        help='Covariance matrix')
    parser.add_argument('--rho', default=1.0, type=float, help='Coefficient for Laplacian')
    parser.add_argument('--delta', default=1.0, type=float, help='Coefficient for beta')
    parser.add_argument('--beta_sigma', default=0.0, type=float, help='Std for beta')
    parser.add_argument('--sample', default=500, type=int, help='Sample size')

    args = parser.parse_args()

    assert args.y_dis == 'linear'
    args.name = 'cov{}_de{:.1f}_seed{:d}'.format(
        args.cov[:3] + ('{:.1f}'.format(args.rho) if args.cov == 'Laplacian' else ''),
        args.delta, args.seed)

    args.save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    # save all the codes
    codes_dir = os.path.join(args.save_dir, 'codes')
    os.makedirs(codes_dir, exist_ok=True)
    os.system('cp ./*.py ' + codes_dir)
    # save argparse
    argsDict = args.__dict__
    with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------' + '\n')

    return args


def main():
    # parameters
    args = build_args()
    set_seed(args.seed)

    # initialize
    all_info = []
    for i in range(30):
        ddi = {
            'gp_label': i + 1,
            'gp_dim': 0,
            'fea_name': [],
            'true_idx': [],
            'isol_idx': [],
            'coordinate': [],
            'beta_sgn': 0,
            'beta': np.array([]),
        }
        if i < 10:
            ddi['gp_dim'] = 150
        elif i < 20:
            ddi['gp_dim'] = 100
        else:
            ddi['gp_dim'] = 50
        ddi['fea_name'] = [('G%d-%d' % (ddi['gp_label'], x + 1)) for x in range(ddi['gp_dim'])]
        ddi['beta'] = np.zeros(ddi['gp_dim'])

        if i % 10 < 3:
            ddi['beta_sgn'] = 1
        elif i % 10 < 6:
            ddi['beta_sgn'] = -1
        else:
            ddi['beta_sgn'] = 0

        all_info.append(ddi)

    # setting for coefficients
    interval = 10  # idx distance of true sites and other isolated false sites
    beta_mu = 1 / np.sqrt(50) * args.delta
    sites = Coordinates('uniform')

    for ddi in all_info:

        if ddi['beta_sgn'] != 0:
            # 中间10%的位点是真实位点
            ddi['true_idx'] = list(range(int(ddi['gp_dim'] * 0.45), int(ddi['gp_dim'] * 0.55)))
            # 6%的位点是孤立位点
            isol_idx0 = list(range(0, min(ddi['true_idx']) - interval)) + \
                       list(range(max(ddi['true_idx']) + interval + 1, ddi['gp_dim']))
            isol_idx = np.random.choice(isol_idx0, size=int(ddi['gp_dim'] * 0.06), replace=False)
            while min(np.diff(np.unique(isol_idx))) < 4:
                isol_idx = np.random.choice(isol_idx0, size=int(ddi['gp_dim'] * 0.06), replace=False)
            ddi['isol_idx'] = sorted(isol_idx)

            # 赋值系数
            ddi['beta'][ddi['true_idx'] + ddi['isol_idx']] = ddi['beta_sgn'] \
                * np.random.normal(loc=beta_mu, scale=args.beta_sigma, size=len(ddi['true_idx'] + ddi['isol_idx']))
            ddi['beta'][ddi['isol_idx']] = (np.sign(np.random.randn(len(isol_idx)))) * ddi['beta'][ddi['isol_idx']]

        #     # 坐标
        #     ddi['coordinate'] = sites.get_sites_2parts(ddi['gp_dim'], ddi['true_idx'])
        #
        # else:
        #     tmp = list(range(int(ddi['gp_dim'] * 0.45), int(ddi['gp_dim'] * 0.55)))
        #     ddi['coordinate'] = sites.get_sites_2parts(ddi['gp_dim'], tmp)

        ddi['coordinate'] = sites.get_sites_even(ddi['gp_dim'])

    # generate data
    gen = GetData(args.cov, args.y_dis, int(args.sample * 0.4), rho=args.rho)
    X, Y = gen.get_data(all_info)
    # save all the information
    df_all = concat_data(all_info)
    df_all.to_csv(os.path.join(args.save_dir, 'basic_info.csv'))

    # individual samples
    spacial_num = args.sample - int(args.sample * 0.4)
    other_gr = [k+i*10 for i in range(3) for k in range(6, 10)]
    spacial_mask = np.zeros((X.shape[1], args.sample))
    X_indi, Y_indi = [], []

    for k in range(args.sample - spacial_num, args.sample, 2):
        all_info_1 = all_info.copy()
        spacial = np.random.choice(other_gr, size=1, replace=False)[0]
        ddi = all_info_1[spacial]
        spac_idx = np.random.choice(list(range(0, ddi['gp_dim'] - int(ddi['gp_dim'] * 0.1))), size=1, replace=False)[0]
        spac_idx = list(range(spac_idx, spac_idx + int(ddi['gp_dim'] * 0.1)))
        ddi['beta'][spac_idx] = (np.sign(np.random.randn(1))) \
            * np.random.normal(loc=beta_mu, scale=args.beta_sigma, size=len(spac_idx))
        beta_idx0 = sum([ddi['gp_dim'] for ddi in all_info_1[:spacial]])
        beta_idx = [x + beta_idx0 for x in spac_idx]
        spacial_mask[beta_idx, k] = 1
        spacial_mask[beta_idx, k + 1] = 1

        gen = GetData(args.cov, args.y_dis, 2, rho=args.rho)
        Xi, Yi = gen.get_data(all_info_1)  # (2, 3000), 2
        X_indi.append(Xi)
        Y_indi.append(Yi)

    X_indi_c = np.concatenate(X_indi, axis=0)
    X = np.vstack([X, X_indi_c])
    Y_indi_c = np.concatenate(Y_indi, axis=0)
    Y = np.hstack([Y, Y_indi_c])

    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    nn = np.linalg.norm(X, ord=2, axis=0)
    X_norm = X / nn[None, :]  # n*d / 1*d
    np.save(os.path.join(args.save_dir, 'X_normL2.npy'), X_norm)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
    np.save(os.path.join(args.save_dir, 'spacial_mask.npy'), spacial_mask)

    np.savetxt(os.path.join(args.save_dir, 'X.csv'), X, delimiter=',')
    np.savetxt(os.path.join(args.save_dir, 'X_normL2.csv'), X_norm, delimiter=',')
    np.savetxt(os.path.join(args.save_dir, 'Y.csv'), Y, delimiter=',')
    np.savetxt(os.path.join(args.save_dir, 'spacial_mask.csv'), spacial_mask, delimiter=',')

    spacial_mask = np.load(os.path.join(args.save_dir, 'spacial_mask.npy'))
    sample_num = spacial_mask.shape[1]
    pair = np.zeros((sample_num, sample_num))
    non_zeros = np.where(spacial_mask.sum(axis=0) > 0)[0]
    for i in range(non_zeros[0], sample_num, 2):
        pair[i, i + 1] = 1
        pair[i + 1, i] = 1
    from scipy import sparse
    pair_sparse = sparse.csr_matrix(pair)
    sparse.save_npz(os.path.join(args.save_dir, 'pair_sparse.npz'), pair_sparse)


if __name__ == '__main__':
    main()
