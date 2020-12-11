# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:41:59 2020

@author: Louis
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:06:19 2020

@author: Louis
"""

import umap
import hdbscan as hd
from sklearn.cluster import Birch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


import spotify_helper
from spotify_helper import SpotifyHelper
from playlist import Playlist
import analyse_refactor 


def generate_clusters(scaler, features, pca_rep_offset=0.9, normalize=True, is_playlist_created=False):
    
    data = analyse_refactor.open_csv("datasets\output\mySavedSongs.csv")
    all_data = analyse_refactor.open_csv("datasets\data.csv")

    data[audio_features] = analyse_refactor.scale_numeric(data[audio_features], method=scale_method, features=audio_features, normalize=normalize)
    all_X = analyse_refactor.scale_numeric(all_data[audio_features], method=scale_method , features=audio_features, normalize=normalize)
    
    pca_df, pca = analyse_refactor.perform_pca(data[audio_features], len(features))
    var_ratio = pca.explained_variance_ratio_
    
    # Select the number of componants based on an offset
    i=0
    cum_ratio = 0
    while(cum_ratio < pca_rep_offset):
        cum_ratio+=var_ratio[i]
        i+=1
        
    print("PCA {} componants (represents {:.0f}% of variance)".format(i,100*cum_ratio))
    
    cols = ["PCA_{}".format(j) for j in range(1,i+1)]
    X = pca_df[cols].copy() 
    
    reducer = umap.UMAP(
        n_neighbors= 5,          # the lower the value the higher the focus on local structure 
        min_dist=0.0,
        n_components=2,
        )
    X = reducer.fit_transform(data[audio_features])
    X = pd.DataFrame(X)
    
    
    birch = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=9,
          threshold=0.5)
    
    hdbscan = hd.HDBSCAN(
        min_samples=2, 
        min_cluster_size=30, 
        cluster_selection_epsilon=.3
        )
    
    
    clustering_algorithms = (
            ('Birch', birch),
            ("HDBSCAN", hdbscan)# HDBSCAN https://github.com/scikit-learn-contrib/hdbscan
        )
    
    name="HDBSCAN"
    algorithm = hdbscan
 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.", category=UserWarning)
        warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding may not work as expected.", category=UserWarning)
        algorithm.fit(X)
        
    # calculate the prediction
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    
    # Get some sense of how good the clustering is doing
    silhouette = silhouette_score(X, y_pred, metric='euclidean')  # bad -1 - 1 good
    chs = calinski_harabasz_score(X, y_pred)                      # higher is better
    dhs = davies_bouldin_score(X, y_pred)                         # good 0 - 1 bad      # a lower Davies-Bouldin index relates to a model with better separation between the clusters.
    score = silhouette*chs/dhs
    print(pd.Series(y_pred).value_counts())
    print("{:.3f} * {:.3f} / {:.3f} = {} : [{} clusters] {}".format(silhouette, chs, dhs,score, len(pd.Series(y_pred).unique()), name))
    
    
    # append the clustering result to mySavedSongs
    X[name] = pd.Series(y_pred)
    
    N = len(audio_features)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(13, 8))
    ax = plt.subplot(121, polar=True, )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], audio_features)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.3,0.5,0.7], [".3",".5",".7"], color="grey", size=7)
    plt.ylim(-1,2)
    
    for c in sorted(X.loc[:,name].unique()):
        # choose a random song from each cluster (random or by popularity)
        dd = pd.concat([data, X], axis=1)
        
        clustered_songs = dd[ dd[name] == c].sort_values(by="popularity", ascending=False)
        
        values=clustered_songs[audio_features].mean().values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=str(c))
        
        
        if is_playlist_created:
            p = Playlist()
            #print(name + str(c))
            p = p.df_to_playlist(clustered_songs.sample(min(25,clustered_songs.shape[0])), reduction + "_" +scale_method + "_" + name + "_" + str(c))
            
            cp = SpotifyHelper()
            p_id = cp.create_playlist(p)
        
    ax = plt.subplot(122, polar=False)
    ax.scatter(X.iloc[:, 1], X.iloc[:, 0], c=X[name])

    plt.title(name)
    plt.show()   
        
    
def main():
    scaler = "MinMax"   # RobustScaler Standard MaxAbs QuantileTransformer PowerTransformer MinMax
    reduction = "UMAP"   
    audio_features = ['danceability', 'energy', 'loudness',
                  'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
    
    spotify_helper.main()
    generate_clusters(scaler, features=audio_features, pca_rep_offset=0.95, normalize=False, is_playlist_created=True)

if __name__ == "__main__":
   main()


    