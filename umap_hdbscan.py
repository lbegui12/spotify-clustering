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

    data[features] = analyse_refactor.scale_numeric(data[features], method=scale_method, features=features, normalize=normalize)
    # all_X = analyse_refactor.scale_numeric(all_data[features], method=scale_method , features=features, normalize=normalize)
    
    pca_df, pca = analyse_refactor.perform_pca(data[features], len(features))
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
    X = reducer.fit_transform(data[features])
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
    
    N = len(features)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(13, 8))
    ax = plt.subplot(121, polar=True, )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], features)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.3,0.5,0.7], [".3",".5",".7"], color="grey", size=7)
    plt.ylim(-1,2)
    
    clusters = sorted(X.loc[:,name].unique())
    if -1 in clusters:
        #clusters.remove(-1)
        print("Non label songs found")
                      
    for c in clusters:
        # choose a random song from each cluster (random or by popularity)
        dd = pd.concat([data, X], axis=1)
        
        clustered_songs = dd[ dd[name] == c].sort_values(by="popularity", ascending=False)
        
        values=clustered_songs[features].mean().values.flatten().tolist()
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
    
    audio_features = ['danceability',       # Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
                  'energy',             # Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. 
                  'loudness',           # The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
                  #'speechiness',        # Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 
                  'acousticness',       # A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
                  'instrumentalness',   # Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. 
                  'liveness',           # Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
                  'valence',            # A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
                  'tempo',              # The overall estimated tempo of a track in beats per minute (BPM).
                  'duration_ms',        # The duration of the track in milliseconds.
                  #'key',                # The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
                  'mode',               # Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
                  'time_signature' ,    # An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
                  ]
    
    spotify_helper.main()
    generate_clusters(scaler, features=audio_features, pca_rep_offset=0.95, normalize=False, is_playlist_created=True)

if __name__ == "__main__":
   main()


    