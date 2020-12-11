# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:49:43 2020

@author: Louis
"""

import pandas as pd

algos = ["KMeans", "AffinityPropagation", "MeanShift", "SpectralClustering", "AgglomerativeClustering", "DBSCAN", "OPTICS", "Birch"]
pca_offsets = np.linspace(0.7,1,6)
scalers = ["RobustScaler", "MinMax", "MaxAbs", "Standard"] #, "PowerTransformer"]

pca_offsets = [x for x in pca_offsets if x>0.8]

for algo in algos:
    best_pca = (0, 0)
    for pca in pca_offsets:
        pca = round(pca,2)
        best_scaler = ("scaler", 0)
        for scaler in scalers:
            result = pd.read_csv("gridsearch_{}_{}_{}.csv".format(algo, pca, scaler))
            
            ranks = result.rank_test_score
            index = ranks.idxmin()
            params = result.params
            scores = result.mean_test_score
            
            best_params = result.params.loc[index]
            best_score = result.mean_test_score.loc[index]
            
            if best_scaler[1] < best_score:
                best_scaler = (scaler, best_score)
        print("best for {} {} {} is {}".format(algo, pca, best_scaler[0], best_scaler[1]))
        if best_pca[1] < best_scaler[1]:
                best_pca = (pca, best_scaler[1])
    print("best for {} {} {} is {} ({})".format(algo, best_pca[0], best_scaler[0],best_pca[1], best_params))
            
        