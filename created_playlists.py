# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:35:41 2020

@author: Louis
"""


import pandas as pd
from analyse_refactor import open_csv
from playlist import Playlist
from track import audio_features
from sklearn.preprocessing import StandardScaler


import umap


data = analyse_refactor.open_csv("datasets\output\clusteredSongs_best_estimator.csv")

reducer = umap.UMAP(n_neighbors=8, min_dist=0.1)

songs = data[audio_features].values
scaled_songs = StandardScaler().fit_transform(songs)

embedding = reducer.fit_transform(scaled_songs)
print(scaled_songs.shape)
print(embedding.shape)

algos = ['MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering', 'AgglomerativeClustering', 'DBSCAN', 'OPTICS', 'Birch']

for algo in algos:
    fig = plt.figure()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data[algo])
    plt.title('UMAP projection ({})'.format(algo), fontsize=24)

plt.show()


