from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, MeanShift, SpectralClustering
from minisom import MiniSom
import pandas as pd
from sklearn.cluster import estimate_bandwidth

#######################################
############ HIERARQUICAL #############
#######################################

def hierarchical_clustering(data: pd.DataFrame) -> pd.DataFrame:

    model = AgglomerativeClustering()

    data['cluster'] = model.fit_predict(data)

    return data

#######################################
############### KMeans ################
#######################################

def kmeans_clustering(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    data['cluster'] = kmeans.fit_predict(data)

    return data

#######################################
################ SOM ##################
#######################################

def som_clustering(data: pd.DataFrame, x: int = 10, y: int = 10, input_len: int = None, sigma: float = 1.0, learning_rate: float = 0.5, iterations: int = 100) -> pd.DataFrame:

    if input_len is None:
        input_len = data.shape[1]

    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)

    som.random_weights_init(data.values)

    som.train_random(data.values, iterations)

    data['cluster'] = [som.winner(row) for row in data.values]

    return data

#######################################
############### DBScan ################
#######################################

def dbscan_clustering(data, eps: float, min_samples: int) -> pd.DataFrame:

    model = DBSCAN(eps=eps, min_samples=min_samples)

    data['cluster'] = model.fit_predict(data)

    return data

#######################################
############# Mean Shift ##############
#######################################

def mean_shift_clustering(data: pd.DataFrame, quantile: float, n_samples: int) -> pd.DataFrame:

    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)

    model = MeanShift(bandwidth=bandwidth)

    data['cluster'] = model.fit_predict(data)

    return data

#######################################
######## Spectral Clustering ##########
#######################################

def spectral_clustering(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:

    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    data['cluster'] = model.fit_predict(data)

    return data