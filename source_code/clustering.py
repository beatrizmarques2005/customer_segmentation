from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def kmeans_clustering(data, n_clusters=5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    data['cluster'] = kmeans.fit_predict(data)

    return data


def hier_clustering(data, linkage = 'ward'):
    hier = AgglomerativeClustering(distance_threshold= 0, linkage=linkage, n_clusters=None, compute_full_tree=True)
    data['hier'] = hier.fit_predict(data)
    return data 


