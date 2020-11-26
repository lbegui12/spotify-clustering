import time

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing

from sklearn.decomposition import PCA
import numpy as np
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score




from track import audio_features


def open_csv(path):
    df = pd.read_csv(path)
    if 'year' in list(df.columns):
        df['year'] = pd.to_datetime(df['year'])
    return df


def scale_numeric(df, method="MinMax", normalize = False, norm="l2"):
    # Only keep numerical values
    num_df = df[audio_features].select_dtypes(include='number')
    
    # Transform the data according to the choosen method
    if method == "Standard":
        X = StandardScaler().fit_transform(num_df)
    elif method == "MinMax":
        X = MinMaxScaler().fit_transform(num_df)
    elif method == "MaxAbs":
        X = MaxAbsScaler().fit_transform(num_df)
    elif method == "RobustScaler":
        X = RobustScaler().fit_transform(num_df)
    elif method == "QuantileTransformer":
        X = QuantileTransformer().fit_transform(num_df)
    elif method == "PowerTransformer":
        X = PowerTransformer(standardize=False).fit_transform(num_df)
    
    if normalize:
        X = preprocessing.normalize(X, norm=norm)   # normalise data if asked
        
    return pd.DataFrame(data = X, columns = audio_features)


def perform_pca(X, n=3, whiten=True, pca=None):
    pca_col = ["PCA_{}".format(i) for i in range(1,n+1)]
    if pca is None:
        pca = PCA(n_components=n, whiten=whiten) 
        principalComponents = pca.fit_transform(X)
    else:
        principalComponents = pca.transform(X)

    pca_df = pd.DataFrame(data = principalComponents, columns = pca_col)
    return (pca_df, pca)

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        score = silhouette_score(X, cluster_labels, metric='euclidean')     # bad -1 - 1 good
        chs = calinski_harabasz_score(X, cluster_labels)                    # higher is better  
        dhs = davies_bouldin_score(X, cluster_labels)                       # good 0 - 1 bad
        print("{} - {} - {}".format(score, chs, dhs))
        
        return score
  




# =============================================================================
# All the process in one place 
# =============================================================================
def process(df, scaler="RobustScaler", pca_rep_offset=0.8, plot=False):
    data = df.copy()
    # =============================================================================
    # First off : Rescale the data
    # =============================================================================
    X = scale_numeric(data, method=scaler, normalize = False, norm="l2")
    features = list(X.columns)
    
    # =============================================================================
    # PCA to reduce dimension before KMeans (n_componants so that it rep > 80%)
    # =============================================================================
    pca_df, pca = perform_pca(X, len(features))
    var_ratio = pca.explained_variance_ratio_
    
    # Select the number of componants based on an offset
    i=0
    cum_ratio = 0
    while(cum_ratio < pca_rep_offset):
        cum_ratio+=var_ratio[i]
        i+=1
        
    print("PCA has {} componants (represents {:.0f}% of variance)".format(i,100*cum_ratio))
    
    cols = ["PCA_{}".format(j) for j in range(1,i+1)]
    X = pca_df[cols].copy()     
    
    # =============================================================================
    # KMeans with a range of n_cluster (returning a metric ie Silhouette)
    # =============================================================================
    clustering_params = {
        ("KMeans", MiniBatchKMeans()) : {
            "n_clusters" : range(2,20)
        },
        
        ("AffinityPropagation", AffinityPropagation()) : {
              #"affinity": ["euclidean", "precomputed"], 
              "preference": range(-50,-240, -10),     # -240 -220 -50
              "damping" : np.linspace(0.5,1,10)       # [0.5-1]
        },
        
        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
        ("MeanShift", MeanShift()) : { 
            #"quantile" : np.linspace(0,1,11),   # [0-1] 
            #"n_sample" : range(2,500, 10)    # [1-all samples]
        },    
        
        ("SpectralClustering", SpectralClustering()) : {
            "n_clusters" : range(2,20)},
        
        ("AgglomerativeClustering", AgglomerativeClustering()) : {   # If linkage is “ward”, only “euclidean” is accepted
            "linkage" : ["ward", "complete", "average", "single"],
            "affinity" : ["euclidean", "cityblock", "cosine"],
            "n_clusters" : range(2,20)
        },
        
        ("DBSCAN", DBSCAN()) : {
            "eps" : np.linspace(0.1,50,20),        # 0.1-10
            "min_samples" : range(2,200,5)   # 2-100 ?# on noisy and large data sets it may be desirable to increase this parameter
        },
        
        ("OPTICS", OPTICS()) : {
            "min_samples" : np.linspace(0,1,20),          # or float [0-1]
            "metric" : ["cityblock", "cosine", "euclidean", "manhattan"],
            "xi" : np.linspace(0,1,50),                # [0-1]
            "min_cluster_size" : np.linspace(0,1,50)   # [0-1]
        },
        
        ("Birch", Birch()) : {
            "n_clusters" : range(2,20)
        }
    }
    
   
    for key, value in clustering_params.items():
        print("Algoooo : {}".format(key[0]))
        t0 = time.time()
        name, algorithm = key
        param_dict = value
        cv = [(slice(None), slice(None))]
        gs = GridSearchCV(estimator=algorithm, param_grid=param_dict, 
                  scoring=cv_silhouette_scorer, cv=cv, n_jobs=-1)
        gs.fit(X)
        
        # get reuslt as a df
        cv_result_df = pd.DataFrame(gs.cv_results_)
        # sort it by rank_test_score
        
        
        clustering_params[key]["best estimator"] = gs.best_estimator_
        clustering_params[key]["best params"] = gs.best_params_
        t1 = time.time()
        print(gs.best_estimator_)
        print("Took {} time".format(t1-t0))
    
    print(clustering_params)
    final_df = pd.concat([data[info_features], df[features], X], axis=1)    
    final_df.to_csv("clusteredSongs.csv")
    
    return (final_df, pca)


def main():
    data = open_csv("datasets\output\mySavedSongs.csv")
    
    info_features = ['id', 'name','artist','year']
    process(data, scaler="MinMax", pca_rep_offset=0.9, plot=True)
    #pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

