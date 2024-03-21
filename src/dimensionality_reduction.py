from sklearn.decomposition import PCA
import numpy as np

def pca_transform_feature_matrix(feature_matrix, n_components):
    '''
    Applies PCA to reduce the number of features (columns) in the feature matrix.

    Parameters:
    - feature_matrix (np.ndarray): 2D numpy array, as returned by image_feature_extraction.get_feature_matrix().
    - n_components (int): Number of columns in the resulting transformed feature matrix.

    Returns:
    np.ndarray: Transformed feature matrix.
    '''
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(feature_matrix)
    transformed_feature_matrix = pca.transform(feature_matrix)

    print(f"Components before PCA: {feature_matrix.shape[1]}")
    print(f"Components after PCA: {pca.n_components}")
    print(f"Transformed feature matrix shape: {transformed_feature_matrix.shape}")

    return transformed_feature_matrix

def find_pca_n_components_with_variance_above_threshold(feature_matrix, pca_var):
    '''
    Applies PCA to find the number of components retaining cumulative explained variance above the given threshold.

    Parameters:
    - feature_matrix (np.ndarray): 2D numpy array, as returned by image_feature_extraction.get_feature_matrix().
    - pca_var (float): Threshold of cumulative explained variance, as proportion from 0 to 1.

    Returns:
    Tuple[np.ndarray, int]: Cumulative explained variance for all number of components and 
        number of components retaining cumulative explained variance >= pca_var.
    '''
    # Apply PCA
    pca = PCA(random_state=42)
    pca.fit(feature_matrix)
    
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components for pca_var cumulative explained variance
    n_components = np.argmax(cumulative_explained_variance >= pca_var) + 1
    print(f"Number of components for {pca_var * 100}% cumulative explained variance: {n_components}")
    return cumulative_explained_variance, n_components