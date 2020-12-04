# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:06:19 2020

@author: Louis
"""

from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings

# from analyse_refactor import open_csv
# from analyse_refactor import scale_numeric
# from analyse_refactor import perform_pca

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import analyse_refactor
from track import audio_features

from sklearn.metrics.pairwise import euclidean_distances

from spotify_helper import SpotifyHelper
from playlist import Playlist

import seaborn as sns


info_features = ['id', 'name','artists','year']

# =============================================================================
# Prep data
# =============================================================================
data = analyse_refactor.open_csv("datasets\output\mySavedSongs.csv")
all_data = analyse_refactor.open_csv("datasets\data.csv")

scale_method = "RobustScaler"   # RobustScaler Standard MaxAbs QuantileTransformer PowerTransformer MinMax
normalize = False
norm = "l1"         # "l1" "max"

scaled_audio_features = ["scaled" + s for s in audio_features]

data[audio_features] = analyse_refactor.scale_numeric(data[audio_features], method=scale_method, normalize = normalize, norm=norm)
all_X = analyse_refactor.scale_numeric(all_data[audio_features], method=scale_method , normalize = normalize, norm=norm)
features = list(data[audio_features].columns) 
print(features)
pca_df, pca = analyse_refactor.perform_pca(data[audio_features], len(features))
var_ratio = pca.explained_variance_ratio_

# Select the number of componants based on an offset
i=0
cum_ratio = 0
pca_rep_offset = 0.9
while(cum_ratio < pca_rep_offset):
    cum_ratio+=var_ratio[i]
    i+=1
    
print("PCA {} componants (represents {:.0f}% of variance)".format(i,100*cum_ratio))

cols = ["PCA_{}".format(j) for j in range(1,i+1)]
X = pca_df[cols].copy() 

all_X_pca, pca = analyse_refactor.perform_pca(all_X, n=len(audio_features), pca = pca)
all_X_pca = all_X_pca[cols]


# =============================================================================
# Define best algo
# =============================================================================
mbkm = MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++',
                init_size=None, max_iter=100, max_no_improvement=10,
                n_clusters=8, n_init=3, random_state=None,
                reassignment_ratio=0.01, tol=0.0, verbose=0)

ap = AffinityPropagation(affinity='euclidean', convergence_iter=35, copy=True,
                    damping=0.803030303030303, max_iter=200, preference=-245,
                    verbose=False)

ms = MeanShift(bandwidth=None, bin_seeding=False, cluster_all=True, max_iter=300,
          min_bin_freq=1, n_jobs=None, seeds=None)

sc = SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=7, n_components=None,
                   n_init=10, n_jobs=None, n_neighbors=10, random_state=None)

ac = AgglomerativeClustering(affinity='cosine', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='average', memory=None, n_clusters=2)

dbscan = DBSCAN(algorithm='auto', eps=2.7263157894736842, leaf_size=30,
       metric='euclidean', metric_params=None, min_samples=2, n_jobs=None,
       p=None)

optics = OPTICS(algorithm='auto', cluster_method='xi', eps=None, leaf_size=30,
       max_eps=np.inf, metric='euclidean', metric_params=None,
       min_cluster_size=None, min_samples=0.10526315789473684, n_jobs=None, p=2,
       predecessor_correction=True, xi=0.0)

birch = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=6,
      threshold=0.5)


clustering_algorithms = (
        ('MiniBatchKMeans', mbkm),
        ('AffinityPropagation', ap),
        #('MeanShift', ms),
        ('SpectralClustering', sc),
        ('AgglomerativeClustering', ac),
        ('DBSCAN', dbscan),
        #('OPTICS', optics),
        #('Birch', birch)
    )

centers = {}
for name, algorithm in clustering_algorithms:
    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.", category=UserWarning)
        warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding may not work as expected.", category=UserWarning)
        algorithm.fit(X)
        
    # calculate the prediction
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    
    # Get some sense of how good the clustering is
    silhouette = silhouette_score(X[cols], y_pred, metric='euclidean')  # bad -1 - 1 good
    chs = calinski_harabasz_score(X[cols], y_pred)                      # higher is better
    # a lower Davies-Bouldin index relates to a model with better separation between the clusters.
    dhs = davies_bouldin_score(X[cols], y_pred)                         # good 0 - 1 bad 
    score = silhouette*chs/dhs
    print("{:.3f} * {:.3f} / {:.3f} = {} : [{} clusters] {}".format(silhouette, chs, dhs,score, len(pd.Series(y_pred).unique()), name))
    
    if score>10:
        # append the clustering result to mySavedSongs
        X[name] = pd.Series(y_pred)
        
        for c in X.loc[:,name].unique():
            # choose a random song from each cluster (random or by popularity)
            dd = pd.concat([data, X], axis=1)
            
            clustered_songs = dd[ dd[name] == c].sort_values(by="popularity", ascending=False)
            
            if(False):
                fig = plt.figure()
                ax = sns.violinplot(data=clustered_songs[audio_features])
                ax.set_title(scale_method + "_" + name + "_" + str(c))
            
            
            p = Playlist()
            #print(name + str(c))
            p = p.df_to_playlist(clustered_songs.sample(min(25,clustered_songs.shape[0])), scale_method + "_" + name + "_" + str(c))
            
            cp = SpotifyHelper()
            p_id = cp.create_playlist(p)
    
        plot = False    
        if(plot):    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X["PCA_1"], X["PCA_2"], X["PCA_3"], c = X.loc[:,name], marker=",") 
            ax.set_xlabel('PCA_1')
            ax.set_ylabel('PCA_2')
            ax.set_zlabel('PCA_3')
            ax.set_title(name)

    
final_df = pd.concat([data[info_features], data[features], X], axis=1)    
final_df.to_csv("datasets\output\mySavedSong_clustered_by_best_estimator.csv")

all_final_df = pd.concat([all_data, all_X_pca], axis=1)    
#all_final_df.to_csv("all_clusteredSongs_best_estimator.csv")


if(False):
    for name in ["AffinityPropagation", "DBSCAN", "Birch"]:
        s_col = name + "_dist_to_center"
        for c in all_X_pca[name].unique():
            df = all_final_df[all_final_df[name] == c].sort_values(by=s_col).loc[:,["id","name","artists",s_col]].head(20)
            p = Playlist()
            p = p.df_to_playlist(df, name + str(c))
            
            cp = SpotifyHelper()
            p_id = cp.create_playlist(p)
            file.write(p_id) 
            file.write("\n") 
    file.close()
    
