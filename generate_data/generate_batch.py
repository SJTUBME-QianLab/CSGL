import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed


def main():
    result_dir = './../sim_data/'
    rand_list = range(2020, 2021)

    cmd_list = []
    for rho in [2.0]:
        for delta in [1]:
            for seed in rand_list:
                model_name = 'covLap{:.1f}_de{:1f}_seed{:d}'.format(rho, delta, seed)
                if (not os.path.isfile(os.path.join(result_dir, model_name, 'pair_sparse.npz'))) \
                        or (os.stat(os.path.join(result_dir, model_name, 'pair_sparse.npz')).st_size == 0):
                    cmd = 'python simulate_individual.py ' \
                          '--cov Laplacian --rho {:.1f} --delta {:.1f} --seed {:d}'.format(rho, delta, seed)
                    cmd_list.append(cmd)

    # running
    Parallel(n_jobs=6)(delayed(os.system)(cmd) for cmd in cmd_list)


if __name__ == '__main__':
    main()
