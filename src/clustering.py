from sklearn.cluster import KMeans

def cluster_with_kmeans(feature_matrix, n_clusters):
    '''
    Clusters the feature matrix (full or PCA-transformed) using the k-means algorithm.

    Parameters:
    - feature_matrix (np.ndarray): As returned by image_feature_extraction.get_feature_matrix() or dimensionality_reduction.pca_transform_feature_matrix().
    - n_clusters (int): Number of clusters for k-means.

    Returns:
    np.ndarray: 1D array with the cluster labels, assigning each part (matrix row) to a cluster.
    '''
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(feature_matrix)

    return kmeans.labels_

def group_parts_to_clusters(parts_feature, labels):
    '''
    Given part to feature dictionary and labels, returns a dictionary mapping each cluster to the list of parts (image filenames) assigned to it.

    Parameters:
    - parts_feature (Dict[str, np.ndarray]): Feature dictionary as returned by image_feature_extraction.map_parts_to_features().
    - labels (np.ndarray): Labels as returned by cluster_with_kmeans().

    Returns:
    Dict[int, List[str]]: Dictionary mapping each cluster to the list of parts (image filenames) assigned to it.
    '''
    filenames = list(parts_feature.keys())
    
    groups = {}
    
    # Iterate over each image filename and its predicted cluster label
    for file, cluster in zip(filenames, labels):
        if cluster not in groups.keys():
            groups[cluster] = []
        # add each filename to a list that is assigned to its cluster
        groups[cluster].append(file)

    return groups