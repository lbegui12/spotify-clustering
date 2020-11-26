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

from helloworld import MapPlaylists
from playlist import Playlist



info_features = ['id', 'name','artist','year']

# =============================================================================
# Prep data
# =============================================================================
data = analyse_refactor.open_csv("mySavedSongs.csv")
all_data = analyse_refactor.open_csv("data.csv")

scale_method = "MaxAbs"   # RobustScaler Standard MaxAbs QuantileTransformer PowerTransformer
normalize = True
norm = "l1"         # "l1" "max"

X = analyse_refactor.scale_numeric(data, method=scale_method, normalize = normalize, norm=norm)
all_X = analyse_refactor.scale_numeric(all_data[audio_features], method=scale_method , normalize = normalize, norm=norm)
features = list(X.columns) 
print(features)
pca_df, pca = analyse_refactor.perform_pca(X, len(features))
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
                #n_clusters=5, n_init=3, random_state=None,
                n_clusters=7, n_init=3, random_state=None,
                reassignment_ratio=0.01, tol=0.0, verbose=0)

ap = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
                    damping=0.7, max_iter=200, preference=-235, verbose=False)

ms = MeanShift(bandwidth=None, bin_seeding=False, cluster_all=True, max_iter=300,
          min_bin_freq=1, n_jobs=None, seeds=None) 

sc = SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=3, n_components=None,
                   n_init=10, n_jobs=None, n_neighbors=10, random_state=None)

ac = AgglomerativeClustering(affinity='cityblock', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        #linkage='average', memory=None, n_clusters=2)
                        linkage='average', memory=None, n_clusters=7)

dbscan = DBSCAN(algorithm='auto', eps=2.7052631578947373, leaf_size=30,
       metric='euclidean', metric_params=None, min_samples=85, n_jobs=None,
       p=None)

optics = OPTICS(algorithm='auto', cluster_method='xi', eps=None, leaf_size=30,
       max_eps=np.inf, metric='cosine', metric_params=None, min_cluster_size=0.05,
       min_samples=0.1, n_jobs=None, p=2, predecessor_correction=True, xi=0.0)

#birch = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=2, threshold=0.5)
birch = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=7, threshold=0.5)

clustering_algorithms = (
        ('MiniBatchKMeans', mbkm),
        ('AffinityPropagation', ap),
        #('MeanShift', ms),
        ('SpectralClustering', sc),
        ('AgglomerativeClustering', ac),
        #('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('Birch', birch)
    )

centers = {}
file = open("created_playlist_ids.txt","a+")
for name, algorithm in clustering_algorithms:
    print(name.upper())
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
    dhs = davies_bouldin_score(X[cols], y_pred)                         # good 0 - 1 bad
    score = silhouette*chs/dhs
    print("{:.3f} - {:.3f} - {:.3f} = {} : [{} clusters] {}".format(silhouette, chs, dhs,score, len(pd.Series(y_pred).unique()), name))
    
    # append the clustering result to mySavedSongs
    X[name] = pd.Series(y_pred)
    
    # Find clusters centers and assign each song from data.csv to a cluster (keeping track of its distance to the cluster center)
    cluster_centers = []
    s_col = name + "_dist_to_center"
    for c in X.loc[:,name].unique():
        cluster_center = X[ X[name] == c].loc[:,cols].mean(axis=0)
        cluster_centers.append(cluster_center)    
    
    d = euclidean_distances(all_X_pca[cols], np.array(cluster_centers)) 
    
    all_X_pca[name] = d.argmin(axis=1)
    all_X_pca[s_col] = pd.Series(d.min(axis=1))
    
    # Using cluster centers may not be a good idea for concave clusters
    # Let's select X points from the cluster randomly and keep the Y closest points
    n_anchor_songs = 3
    for c in X.loc[:,name].unique():
        # choose a random song from each cluster (random or by popularity)
        ddd = pd.concat([data, X], axis=1)
        clustered_songs = ddd[ ddd[name] == c].sort_values(by="popularity", ascending=False)
        samples = clustered_songs.loc[:,cols].sample(min(n_anchor_songs,clustered_songs.shape[0]))
        d = euclidean_distances(all_X_pca[cols], np.array(samples))
        
        d_df = pd.DataFrame(data=d, columns=[str(i) for i in range(1,min(n_anchor_songs,clustered_songs.shape[0])+1)])
        close_songs = pd.concat([all_data, d_df], axis=1).drop_duplicates()
        
        con = pd.DataFrame()
        for by in [str(i) for i in range(1,min(n_anchor_songs,clustered_songs.shape[0])+1)]:
            c1 = close_songs.sort_values(by=by).loc[:,["id", "name","artists"]].head(5)
            con=pd.concat([con,c1], axis=0).drop_duplicates()
        
        p = Playlist()
        print(name + str(c))
        p = p.df_to_playlist(con, name + str(c))
        
        cp = MapPlaylists()
        p_id = cp.create_playlist(p)
        file.write(p_id) 
        file.write("\n") 

    plot = True    
    if(plot):    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X["PCA_1"], X["PCA_2"], X["PCA_3"], c = X.loc[:,name], marker=",") 
        ax.set_xlabel('PCA_1')
        ax.set_ylabel('PCA_2')
        ax.set_zlabel('PCA_3')
        ax.set_title(name)
        
file.close()
    
final_df = pd.concat([data[info_features], data[features], X], axis=1)    
final_df.to_csv("clusteredSongs_best_estimator.csv")

all_final_df = pd.concat([all_data, all_X_pca], axis=1)    
#all_final_df.to_csv("all_clusteredSongs_best_estimator.csv")


if(False):
    file = open("created_playlist_ids.txt","w")
    for name in ["AffinityPropagation", "DBSCAN", "Birch"]:
        s_col = name + "_dist_to_center"
        for c in all_X_pca[name].unique():
            df = all_final_df[all_final_df[name] == c].sort_values(by=s_col).loc[:,["id","name","artists",s_col]].head(20)
            p = Playlist()
            p = p.df_to_playlist(df, name + str(c))
            
            cp = MapPlaylists()
            p_id = cp.create_playlist(p)
            file.write(p_id) 
            file.write("\n") 
    file.close()
    
