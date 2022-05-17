# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 14:02
# @Author  : tangxl
# @FileName: generate_batch.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed


def main():
    result_dir = '/home/data/tangxl/ContrastSGL/sim_data/'
    rand_list = range(2021, 2030)

    cmd_list = []
    # Laplacian model
    for rho in [1.1, 2.0]:
        for delta in [1, 2]:
            for seed in rand_list:
                model_name = 'covLap{:.1f}_de{:1f}_seed{:d}'.format(rho, delta, seed)
                if (not os.path.isfile(os.path.join(result_dir, model_name, 'pair_sparse.npz'))) \
                        or (os.stat(os.path.join(result_dir, model_name, 'pair_sparse.npz')).st_size == 0):
                    cmd = 'python simulate_individual.py ' \
                          '--cov Laplacian --rho {:.1f} --delta {:.1f} --seed {:d}'.format(rho, delta, seed)
                    cmd_list.append(cmd)

    # Gaussian model
    for delta in [1, 2]:
        for seed in rand_list:
            model_name = 'covGau_de{:1f}_seed{:d}'.format(delta, seed)
            if (not os.path.isfile(os.path.join(result_dir, model_name, 'pair_sparse.npz'))) \
                    or (os.stat(os.path.join(result_dir, model_name, 'pair_sparse.npz')).st_size == 0):
                cmd = 'python simulate_individual.py ' \
                      '--cov Gaussian --delta {:d} --seed {:d}'.format(delta, seed)
                cmd_list.append(cmd)

    # running
    Parallel(n_jobs=6)(delayed(os.system)(cmd) for cmd in cmd_list)


if __name__ == '__main__':
    main()
