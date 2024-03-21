#!/usr/bin/env python
# coding: utf-8

# Based on the tutorial: [https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34]

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
    
# import helper functions
import image_feature_extraction as im
import dimensionality_reduction as dim
import clustering as cl
import evaluation as ev
import visualization as vis

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run(image_folder,
        model, 
        preprocess_input,
        model_details,
        target_size=None,
        grayscale=False,
        n_components=None, # for PCA
        pca_var=None, # cumulative explained variance for finding the number of components for PCA
        n_clusters=500,
        reports_subfolder=None,
        threshold=0.7, # for evaluation
        evaluate_n_clusters=False, # for evaluation with different number of clusters
        reports_folder='../../reports/figures/image_clustering',
        processed_data_folder='../../data/processed',
        csv_file='SyrusMasterDataAnonymisedProc.csv'
        ):
    
    # Create feature vectors for all images
    feature_vectors_file = f'feature_vectors_{model_details}.pkl'
    feature_vectors_pickle_path = os.path.join(processed_data_folder, feature_vectors_file)
    if not os.path.exists(feature_vectors_pickle_path):
        logger.info('Preprocessing and passing each image through the predict method to get the feature vectors.')
        parts_feature = im.map_parts_to_features(image_folder, model, preprocess_input, dump_path=feature_vectors_pickle_path, target_size=target_size, grayscale=grayscale)
        # confirm the size of parts_feature dictionary to the actual number of images present in syrus image folder
        logger.debug(f"Image count: {len(parts_feature)}")

    # Load and reshape feature map
    logger.debug(f"Loading parts_feature from {feature_vectors_pickle_path}")
    parts_feature = im.load_from_pickle_file(feature_vectors_pickle_path)

    # The shape is (1, x) for every feature vector, where x is the number of nodes of the last dense layer in our model. We need to convert it to (x, ) for PCA and clustering.
    # Rows of `feature_matrix` correspond to the images and columns to the features.
    logger.debug(f"Reshaping parts_feature to feature_matrix")
    feature_matrix = im.get_feature_matrix(parts_feature)
    logger.info(f"Feature matrix shape: {np.shape(feature_matrix)}")

    # Dimensionality Reduction (PCA)
    if type(pca_var) is float and pca_var > 0:
        logger.info(f"Finding the number of principal components explaining {pca_var*100}% of variance")
        cumulative_explained_variance, n_components = dim.find_pca_n_components_with_variance_above_threshold(feature_matrix, pca_var) 
        # Plotting
        vis.plot_cumulative_explained_variance_vs_n_components(cumulative_explained_variance, n_components, pca_var, model_details)

    if n_components is not None and n_components < np.shape(feature_matrix)[1]:
        logger.debug(f"Doing a PCA, reducing the feature matrix to {n_components} components")
        transformed_feature_matrix = dim.pca_transform_feature_matrix(feature_matrix, n_components)
    else:
        logger.debug("Not doing a PCA, using the full feature matrix")

    # KMeans clustering
    # Clustering the parts using the feature matrix
    if n_components is not None and n_components < np.shape(feature_matrix)[1]:
        feature_matrix_to_cluster = transformed_feature_matrix
        logger.info(f"Clustering using the reduced feature matrix having {n_components} features into {n_clusters} clusters")
    else:
        feature_matrix_to_cluster = feature_matrix
        logger.info(f"Clustering using the full feature matrix having {np.shape(feature_matrix)[1]} features")
        
    predicted_labels = cl.cluster_with_kmeans(
        feature_matrix=feature_matrix_to_cluster,
        n_clusters=n_clusters
    )

    # Cluster visualisation
    logger.debug("Groupping the parts into their predicted clusters")
    groups = cl.group_parts_to_clusters(parts_feature, predicted_labels)

    logger.debug("Calculating and plotting a histogram of the cluster sizes")
    cluster_sizes = [len(image_list) for image_list in sorted(groups.values())]
    vis.plot_histogram_of_cluster_sizes(cluster_sizes, model_details, n_clusters)

    if reports_subfolder is not None:
        logger.debug("Saving the clustered images and groups")
        if not os.path.exists(os.path.join(reports_folder, reports_subfolder)):
            os.makedirs(os.path.join(reports_folder, reports_subfolder))

        groups_path = os.path.join(reports_folder, reports_subfolder, f"{reports_subfolder}_cluster_to_files.pkl")
        im.dump_to_pickle_file(groups, groups_path)
            
        for i in tqdm(range(len(groups)), desc=f"Generating images for cluster"):
            output_png = os.path.join(reports_folder, reports_subfolder, f"{reports_subfolder}_cluster{i}.png")
            vis.display_images_in_cluster(groups, image_folder, i, output_png=output_png)

    # Evaluation
    csv_path = os.path.join(processed_data_folder, csv_file)
    if evaluate_n_clusters:
        logger.info("Evaluation using `target` column as true labels for different number of clusters:")
        out_dir = os.path.join(processed_data_folder, model_details)  
        # Run clustering with kmeans with different cluster numbers and evaluate the results for `target` with all our metrics:
        results_df = ev.evaluate_clusters_numbers(csv_path, parts_feature, feature_matrix_to_cluster, out_dir, threshold=threshold)
        # Plot the results:
        vis.plot_score_vs_clusters_numbers(results_df, threshold, model_details)
    else:
        logger.info("Evaluation using `target` (PVFamily_PVSubfamily) column as true labels:")
        # Run clustering with kmeans with the given number of clusters and evaluate the results for `target` with all our metrics:
        target_labels = ev.get_target_labels(csv_path, target_col='target')
        frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, ari, _, _ =  ev.evaluate_clustering_4_metrics(target_labels, predicted_labels, threshold=threshold)
        results_df = pd.DataFrame({
        "n_clusters": [n_clusters],
        "predicted_labels": [predicted_labels],
        "frequency_correct_clusters": [frequency_correct_clusters],
        "frequency_clusters_above_threshold": [frequency_clusters_above_threshold],
        "frequency_main_target_per_cluster": [frequency_main_target_per_cluster],
        "ari": [ari]
    })

    logger.info("Evaluation using `PVFamily` column as true labels:")
    # Run clustering with kmeans with the given number of clusters and evaluate the results for `PVFamily` with all our metrics:
    target_labels = ev.get_target_labels(csv_path, target_col='PVFamily')
    ev.evaluate_clustering_4_metrics(target_labels, predicted_labels, threshold=threshold)  

    return results_df

def run_batches(
        train_generator,
        model,
        model_details,
        n_components=None, # for PCA
        pca_var=None, # cumulative explained variance for finding the number of components for PCA
        n_clusters=500,
        threshold=0.7, # for evaluation
        evaluate_n_clusters=False, # for evaluation with different number of clusters
        processed_data_folder='../../data/processed',
        csv_file='SyrusMasterDataAnonymisedProc.csv'
        ):
    
    # Create feature vectors for all images
    feature_vectors_file = f'feature_vectors_{model_details}.pkl'
    feature_vectors_pickle_path = os.path.join(processed_data_folder, feature_vectors_file)
    if not os.path.exists(feature_vectors_pickle_path):
        logger.info('Preprocessing and passing each image through the predict method to get the feature vectors.')
        feature_matrix = model.predict(train_generator, use_multiprocessing=True)
        im.dump_to_pickle_file(feature_matrix, feature_vectors_pickle_path)

    # Load feature map
    logger.debug(f"Loading feature_matrix from {feature_vectors_pickle_path}")
    feature_matrix = im.load_from_pickle_file(feature_vectors_pickle_path)

    # Dimensionality Reduction (PCA)
    if type(pca_var) is float and pca_var > 0:
        logger.info(f"Finding the number of principal components explaining {pca_var*100}% of variance")
        cumulative_explained_variance, n_components = dim.find_pca_n_components_with_variance_above_threshold(feature_matrix, pca_var) 
        # Plotting
        vis.plot_cumulative_explained_variance_vs_n_components(cumulative_explained_variance, n_components, pca_var, model_details)

    if n_components is not None and n_components < np.shape(feature_matrix)[1]:
        logger.debug(f"Doing a PCA, reducing the feature matrix to {n_components} components")
        transformed_feature_matrix = dim.pca_transform_feature_matrix(feature_matrix, n_components)
    else:
        logger.debug("Not doing a PCA, using the full feature matrix")

    # KMeans clustering
    # Clustering the parts using the feature matrix
    if n_components is not None and n_components < np.shape(feature_matrix)[1]:
        feature_matrix_to_cluster = transformed_feature_matrix
        logger.info(f"Clustering using the reduced feature matrix having {n_components} features into {n_clusters} clusters")
    else:
        feature_matrix_to_cluster = feature_matrix
        logger.info(f"Clustering using the full feature matrix having {np.shape(feature_matrix)[1]} features")
        
    predicted_labels = cl.cluster_with_kmeans(
        feature_matrix=feature_matrix_to_cluster,
        n_clusters=n_clusters
    )

    # Evaluation
    csv_path = os.path.join(processed_data_folder, csv_file)
    if evaluate_n_clusters:
        logger.info("Evaluation using `target` column as true labels for different number of clusters:")
        out_dir = os.path.join(processed_data_folder, model_details)  
        # Run clustering with kmeans with different cluster numbers and evaluate the results for `target` with all our metrics:
        results_df = ev.evaluate_clusters_numbers(csv_path,
                                                  parts_feature=None,
                                                  feature_matrix_to_cluster=feature_matrix_to_cluster,
                                                  out_dir=out_dir,
                                                  threshold=threshold)
        # Plot the results:
        vis.plot_score_vs_clusters_numbers(results_df, threshold, model_details)
    else:
        logger.info("Evaluation using `target` (PVFamily_PVSubfamily) column as true labels:")
        # Run clustering with kmeans with the given number of clusters and evaluate the results for `target` with all our metrics:
        frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, _, _ = ev.evaluate_clustering(csv_path, parts_feature, predicted_labels, target_col='target', threshold=threshold)
        results_df = pd.DataFrame({
        "n_clusters": [n_clusters],
        "predicted_labels": [predicted_labels],
        "frequency_correct_clusters": [frequency_correct_clusters],
        "frequency_clusters_above_threshold": [frequency_clusters_above_threshold],
        "frequency_main_target_per_cluster": [frequency_main_target_per_cluster]
    })

    logger.info("Evaluation using `PVFamily` column as true labels:")
    # Run clustering with kmeans with the given number of clusters and evaluate the results for `PVFamily` with all our metrics:
    ev.evaluate_clustering(csv_path, parts_feature=None, predicted_labels=predicted_labels, target_col='PVFamily', threshold=threshold)

    return results_df

if __name__ == '__main__':
    run()