This repository holds the code for the paper

**Contrastive Spatial Group LASSO for Biomarker Discovery in DNA Methylation Data**

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Zhanfeng Mo, Xiaohua Qian\*

# Abstract

**Motivation**: Identifying candidate biomarkers in DNA methylation data is crucial for early diagnosis of numerous diseases. A key challenge for this identification process is the confounding factors, such as subject-specific anomalies, that may conceal the actual methylation sites. Additionally, although earlier methods have considered the co-regulation of adjacent Cytosine-phosphate-Guanine (CpG) sites, these methods overlooked the micromesh patterns of spatial correlation associated with inter-site distance variations.

**Results**: We integrate deep learning with traditional regression to propose a feature selection model based on contrastive learning and spatial correlation, thereby discovering causal and spatially-clustered CpG sites. Specifically, contrastive learning is employed to reduce the adverse influence of the confounding factors, through implicitly emphasizing the selection of discriminative features for differentiating between case-control pairs from the same subject (such as cancer and adjacent normal tissues). Also, a spatially-weighted total variation regularization is designed for biomarker discovery, thereby aligning spatial correlation with weight smoothness so that adjacent sites have similar weights. Our simulations show that the proposed method not only outperforms other methods in terms of feature selection accuracy, but can also remove biased features and improve feature aggregation. With further experiments on real DNA methylation data, these strengths of our method are verified through gene function analysis and visualization of short-range site weights. Overall, we provide a novel strategy for biomarker discovery in DNA methylation data, where biological significance, applicability, and reliability are adequately accounted for.

# required

Our code is based on **Python3.8** There are a few dependencies to run the code. The major libraries we depend are

- PyTorch1.10.0 (http://pytorch.org/)
- numpy
- pandas
- scikit-learn

# set up

```
conda env create -f ContrastSGL.yaml
```

Attention: Please run this project on linux. In different pytorch environment, the model may obtain different results.

# quickly start a simulation

1. generate simulation data

   Go to the folder `./generate_data` and then run this command:

   ```shell
   python simulate_individual.py --cov Laplacian --rho 2.0 --seed 2022
   ```

   The data will be saved in  `./sim_data` folder.

2. train and test 

   Go to the folder `./simulation` and then run the `main.py` by this command:

   ```shell
   python main.py --data_name covLap2.0_de1.0_seed2022 --L1 0.2 --L21 0.05 --Lg 1.0 --Lcon 0.3 --lr 0.1
   ```

   The weights and metrices will be saved in `./result` folder.

# Contact

For any question, feel free to contact

```
Xinlu Tang : tangxl20@sjtu.edu.cn
```
