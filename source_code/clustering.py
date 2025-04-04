from sklearn.cluster import KMeans

def kmeans_clustering(data, n_clusters=5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    data['cluster'] = kmeans.fit_predict(data)

    return data