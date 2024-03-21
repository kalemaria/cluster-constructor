import os
import numpy as np
import pandas as pd

import image_feature_extraction as im
import clustering as cl
from sklearn.metrics import adjusted_rand_score


def get_part_ids(csv_path, has_image_col='hasImage', id_col='ItemNumber', verbose=False):
    '''
    Returns part IDs in a CSV file for all parts having an image.
    
    Parameters:
    - csv_path (str): Path for the CSV file.
    - has_image_col (str): Boolean column name with 1 if the part in the current row has an image, default: 'hasImage'.
        None if it does not exist - all rows have an image.
    - id_col (str): Part names column name, default: 'ItemNumber'.
    - verbose (bool): If True, prints parts with images count, default: False.

    Returns:
    np.ndarray: 1D array with labels.
    '''
    # Load the tabular data for all parts having images:
    df = pd.read_csv(csv_path, index_col=0)
    if has_image_col is not None:
        df = df[df[has_image_col] == 1]
    if verbose:
        print(f"Parts with images count: {len(df)}")
    # Get the id column values:
    ids = df[id_col].values
    return ids


def get_target_labels(csv_path, has_image_col='hasImage', target_col='target', verbose=False):
    '''
    Returns target labels in a CSV file for all parts having an image.
    
    Parameters:
    - csv_path (str): Path for the CSV file.
    - has_image_col (str): Boolean column name with 1 if the part in the current row has an image, default: 'hasImage'.
        None if it does not exist - all rows have an image
    - target_col (str): Target labels column name, default: 'target'.
    - verbose (bool): If True, prints parts with images count, default: False.

    Returns:
    np.ndarray: 1D array with labels.
    '''
    # Load the tabular data for all parts having images:
    df = pd.read_csv(csv_path, index_col=0)
    if has_image_col is not None:
        df = df[df[has_image_col] == 1]
    if verbose:
        print(f"Parts with images count: {len(df)}")
    # Get the labels:
    labels = df[target_col].values
    return labels

def get_unique_target_labels(csv_path, has_image_col='hasImage', target_col='target', verbose=False):
    '''
    Returns unique target labels in a CSV file for all parts having an image.
    
    Parameters:
    - csv_path (str): Path for the CSV file.
    - has_image_col (str): Boolean column name with 1 if the part in the current row has an image, default: 'hasImage'.
        None if it does not exist - all rows have an image
    - target_col (str): Target labels column name, default: 'target'.
    - verbose (bool): If True, prints parts with images count and unique labels count, default: False.

    Returns:
    np.ndarray: 1D array with unique labels.
    '''
    # Load the tabular data for all parts having images:
    df = pd.read_csv(csv_path, index_col=0)
    if has_image_col is not None:
        df = df[df[has_image_col] == 1]
    if verbose:
        print(f"Parts with images count: {len(df)}")
    # Get the unique labels:
    unique_labels = df[target_col].unique()
    if verbose:
        print(f"Unique labels count: {len(unique_labels)}")
    return unique_labels

def assert_image_order(parts_feature, ids):
    '''
    Asserts that filenames without extension in `parts_feature.keys()` match the IDs from a CSV file.

    Parameters:
    parts_feature (dict): Dictionary with filenames (without extension) as keys.
    ids (iterable): Iterable containing part IDs.

    Returns:
    bool: True if all filenames match corresponding IDs, False otherwise.
    '''
    # Get filenames that are ordered like predicted_labels:
    filenames = list(parts_feature.keys())
    # Remove the extension:
    part_names = [os.path.splitext(file)[0] for file in filenames]
    # Flag to return
    all_correct = True
    # Compare with IDs one by one
    for (id, part_name) in zip(list(ids), part_names):
        # If an id and a part name do not match, set the flag to False and print them out
        if id != part_name:
            all_correct = False
            print(f"{id} != {part_name}")
    return all_correct

def make_pred_df_simple(target_labels, predicted_labels):
    '''
    Make a DataFrame needed for all evaluation methods.
    
    Parameters:
    target_labels (iterable): Iterable containing target labels (default: Family and sub-family).
    predicted_labels (iterable): Iterable containing predicted cluster labels.

    Returns:
    pandas.DataFrame: DataFrame with columns:
        - "predicted_cluster": Predicted cluster labels ("ML cluster").
        - "target": Target labels (default: Family and sub-family).
        - "predicted_target": Specific target label or 'Multiple'.
        - "correct": Binary indicator (1 if predicted target matches the actual target, 0 otherwise).
        - "main_target_per_predicted_cluster": Most frequent target label of the predicted cluster for that part ("Main sub-family per ML cluster").
        - "result": Binary indicator (1 if most frequent predicted target matches the actual target, 0 otherwise).
    '''
    # Assert that the lengths match
    if len(target_labels) != len(predicted_labels):
        raise ValueError("Lengths of 'target_labels' and 'predicted_labels' must match.")
    
    pred_df = pd.DataFrame({
        "predicted_cluster": predicted_labels, # "ML cluster"
        "target": target_labels, # (default: Family and sub-family)
        
    })
    # Sort by predicted_cluster
    pred_df.sort_values(by=['predicted_cluster'], inplace=True)
    # Does a predicted cluster contain only one specific or multiple targets?
    cluster_map = _map_cluster_to_specific_or_multiple_target(pred_df)
    # specific target label or 'Multiple'
    pred_df["predicted_target"] = pred_df.predicted_cluster.map(cluster_map)
    # whether the predicted cluster for that part contains only 1 target
    pred_df["correct"] = pred_df.apply(lambda x: 1 if x["target"] == x["predicted_target"] else 0, axis = 1) 
    # Asign the most frequent target label to each cluster
    cluster_map = _map_cluster_to_main_target(pred_df)
    # most frequent target label of the predicted cluster for that part
    pred_df["main_target_per_predicted_cluster"] = pred_df.predicted_cluster.map(cluster_map) # "Main sub-family per ML cluster"
    # Whether the most frequent target label of the predicted cluster for that part matches the target label
    pred_df["result"] = pred_df.apply(lambda x: 1 if x["target"] == x["main_target_per_predicted_cluster"] else 0, axis = 1)

    return pred_df

def make_pred_df(ids, target_labels, predicted_labels):
    '''
    Make a DataFrame needed for all evaluation methods.
    
    Parameters:
    ids (iterable): Iterable containing part IDs.
    target_labels (iterable): Iterable containing target labels (default: Family and sub-family).
    predicted_labels (iterable): Iterable containing predicted cluster labels.

    Returns:
    pandas.DataFrame: DataFrame with columns:
        - "id": Part IDs.
        - "predicted_cluster": Predicted cluster labels ("ML cluster").
        - "target": Target labels (default: Family and sub-family).
        - "predicted_target": Specific target label or 'Multiple'.
        - "correct": Binary indicator (1 if predicted target matches the actual target, 0 otherwise).
        - "main_target_per_predicted_cluster": Most frequent target label of the predicted cluster for that part ("Main sub-family per ML cluster").
        - "result": Binary indicator (1 if most frequent predicted target matches the actual target, 0 otherwise).
    '''
    # Assert that the lengths match
    if len(ids) != len(target_labels) or len(ids) != len(predicted_labels) or len(target_labels) != len(predicted_labels):
        raise ValueError("Lengths of 'ids', 'target_labels', and 'predicted_labels' must match.")
    
    pred_df = pd.DataFrame({
        "id": ids,
        "predicted_cluster": predicted_labels, # "ML cluster"
        "target": target_labels, # (default: Family and sub-family)
        
    })
    # Sort by predicted_cluster
    pred_df.sort_values(by=['predicted_cluster'], inplace=True)
    # Does a predicted cluster contain only one specific or multiple targets?
    cluster_map = _map_cluster_to_specific_or_multiple_target(pred_df)
    # specific target label or 'Multiple'
    pred_df["predicted_target"] = pred_df.predicted_cluster.map(cluster_map)
    # whether the predicted cluster for that part contains only 1 target
    pred_df["correct"] = pred_df.apply(lambda x: 1 if x["target"] == x["predicted_target"] else 0, axis = 1) 
    # Asign the most frequent target label to each cluster
    cluster_map = _map_cluster_to_main_target(pred_df)
    # most frequent target label of the predicted cluster for that part
    pred_df["main_target_per_predicted_cluster"] = pred_df.predicted_cluster.map(cluster_map) # "Main sub-family per ML cluster"
    # Whether the most frequent target label of the predicted cluster for that part matches the target label
    pred_df["result"] = pred_df.apply(lambda x: 1 if x["target"] == x["main_target_per_predicted_cluster"] else 0, axis = 1)

    return pred_df  

def _map_cluster_to_specific_or_multiple_target(pred_df):
    cluster_nums = pred_df.predicted_cluster.unique()
    cluster_map = {}
    for cluster in np.sort(cluster_nums):
        labels_in_cluster = pred_df[pred_df.predicted_cluster == cluster].target
        if labels_in_cluster.nunique() > 1:
            cluster_map[cluster] = "Multiple"
        else:
            cluster_map[cluster] = labels_in_cluster.unique()[0]
    return cluster_map

def _map_cluster_to_main_target(pred_df):
    cluster_nums = pred_df.predicted_cluster.unique()
    cluster_map = {}
    for cluster in np.sort(cluster_nums):
        labels_in_cluster = pred_df[pred_df.predicted_cluster == cluster].target
        cluster_map[cluster] = labels_in_cluster.value_counts().index[0]
    return cluster_map

def get_frequency_completely_correct_clusters(pred_df):
    '''
    Pedro's evaluation method. Returns the proportion of parts in completely correct clusters,
    where all predicted labels match with the targets.
    
    Parameters:
    pred_df (pandas.DataFrame): DataFrame containing the evaluation results.

    Returns:
    float: Proportion of parts in completely correct clusters.
    '''
    correct_value_counts = pred_df.correct.value_counts()
    if len(correct_value_counts) == 2:
        frequency_correct_clusters = pred_df.correct.value_counts()[1] / pred_df.correct.value_counts().sum()
    else:
        frequency_correct_clusters = 0
    print(f"Percent of completely correct clusters: {round(frequency_correct_clusters * 100, 2)}%")
    return frequency_correct_clusters

def get_frequency_clusters_with_mostly_one_target(threshold, pred_df):
    '''
    Ria's evaluation method. Returns the proportion of clusters with the most frequent target frequency >= threshold.

    Parameters:
    threshold (float): Threshold for the most frequent target frequency.
    pred_df (pandas.DataFrame): DataFrame containing the evaluation results.

    Returns:
    tuple: Tuple containing:
        - pandas.DataFrame: DataFrame with columns ['predicted_cluster', 'most_frequent_target', 'frequency_of_most_frequent_target'].
        - float: Proportion of clusters with most frequent target frequency >= threshold.
    
    '''
    # Create a frequency table, showing the relative frequency of parts in each predicted cluster per target group
    # Count table without margins
    count_table = pd.crosstab(pred_df['predicted_cluster'], pred_df['target'])
    # Normalize the counts per row and round to 2 decimal points
    frequency_table = count_table.apply(lambda x: (x / x.sum()).round(2), axis=1)

    # Find the most frequent target and its frequency for each cluster (row)
    frequency_df = _get_most_frequent_target_frequency(frequency_table)

    # Count clusters with frequency of most frequent target >= threshold
    count_clusters_above_threshold = (frequency_df['frequency_of_most_frequent_target'] >= threshold).sum()
    frequency_clusters_above_threshold = count_clusters_above_threshold / len(frequency_df)
    
    print(f"Percent of clusters with most frequent target frequency >= {threshold * 100}%: {(frequency_clusters_above_threshold * 100).round(2)}%")
    return frequency_df, frequency_clusters_above_threshold

def _get_most_frequent_target_frequency(frequency_table):
    most_frequent_target = frequency_table.idxmax(axis=1)
    frequency_of_most_frequent_target = frequency_table.max(axis=1)

    frequency_df = pd.DataFrame({
        'most_frequent_target': most_frequent_target.values,
        'frequency_of_most_frequent_target': frequency_of_most_frequent_target.values
    })
    return frequency_df

def get_frequency_main_target_per_cluster(pred_df):
    '''
    Tristan's evaluation method. Returns the proportion of parts with the most frequent target label of the predicted cluster matching the target label.

    Parameters:
    pred_df (pandas.DataFrame): DataFrame containing the evaluation results.

    Returns:
    float: Proportion of parts with the main target label per cluster matching the target label.
    '''
    frequency_main_target_per_cluster = pred_df["result"].sum() / pred_df["result"].count()
    print(f"Percent of main target per cluster: {round(frequency_main_target_per_cluster * 100, 2)}%")
    return frequency_main_target_per_cluster

def evaluate_clustering_4_metrics(target_labels, predicted_labels, threshold=0.7):
    '''
    Evaluate the performance of a clustering model based on predicted labels.

    Parameters:
    target_labels (iterable): Iterable containing true cluster labels.
    predicted_labels (iterable): Iterable containing predicted cluster labels.
    threshold (float): Threshold for the most frequent target frequency in clusters, default: 0.7."

    Returns:
    tuple: Tuple containing:
        - float: Proportion of parts in completely correct clusters (between 0 and 1) .
        - float: Proportion of clusters with most frequent target frequency >= threshold (between 0 and 1) .
        - float: Proportion of parts with the most frequent target label per cluster matching the target label (between 0 and 1) .
        - float: Adjusted Rand Index, similarity score between -0.5 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
        - pandas.DataFrame: DataFrame with evaluation results for each part, as returned by make_pred_df().
        - pandas.DataFrame: DataFrame with columns ['predicted_cluster', 'most_frequent_target', 'frequency_of_most_frequent_target'].
    '''
    print('Unique labels:', len(np.unique(target_labels)))
    pred_df = make_pred_df_simple(target_labels, predicted_labels)
    frequency_correct_clusters = get_frequency_completely_correct_clusters(pred_df)
    frequency_df, frequency_clusters_above_threshold = get_frequency_clusters_with_mostly_one_target(threshold, pred_df)
    frequency_main_target_per_cluster = get_frequency_main_target_per_cluster(pred_df)
    ari = adjusted_rand_score(labels_true=target_labels, labels_pred=predicted_labels)
    print("ARI:", round(ari, 2))
    return frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, ari, pred_df, frequency_df

def evaluate_clustering_simple(target_labels, predicted_labels, threshold=0.7):
    '''
    Evaluate the performance of a clustering model based on predicted labels.

    Parameters:
    target_labels (iterable): Iterable containing true cluster labels.
    predicted_labels (iterable): Iterable containing predicted cluster labels.
    threshold (float): Threshold for the most frequent target frequency in clusters, default: 70."

    Returns:
    tuple: Tuple containing:
        - float: Proportion of parts in completely correct clusters.
        - float: Proportion of clusters with most frequent target frequency >= threshold.
        - float: Proportion of parts with the most frequent target label per cluster matching the target label.
        - pandas.DataFrame: DataFrame with evaluation results for each part, as returned by make_pred_df().
        - pandas.DataFrame: DataFrame with columns ['predicted_cluster', 'most_frequent_target', 'frequency_of_most_frequent_target'].
    '''
    print('Unique labels:', len(np.unique(target_labels)))
    pred_df = make_pred_df_simple(target_labels, predicted_labels)
    frequency_correct_clusters = get_frequency_completely_correct_clusters(pred_df)
    frequency_df, frequency_clusters_above_threshold = get_frequency_clusters_with_mostly_one_target(threshold, pred_df)
    frequency_main_target_per_cluster = get_frequency_main_target_per_cluster(pred_df)
    return frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, pred_df, frequency_df

def evaluate_clustering(csv_path, parts_feature, predicted_labels, target_col='target', threshold=0.7, has_image_col='hasImage'):
    '''
    Evaluate the performance of a clustering model based on predicted labels.

    Parameters:
    csv_path (str): Path to the CSV file containing part information.
    parts_feature (Optional[dict]): Dictionary with filenames (without extension) as keys, representing the feature vectors of parts.
        If given, it will be checked that parts order is the same as in CSV, if None, the check wil be omited.
    predicted_labels (iterable): Iterable containing predicted cluster labels.
    target_col (str): Column name for the target labels in the CSV file, default: 'target'.
    threshold (float): Threshold for the most frequent target frequency in clusters, default: 70."
    - has_image_col (str): Boolean column name with 1 if the part in the current row has an image, default: 'hasImage'.
        None if it does not exist - all rows have an image.

    Returns:
    tuple: Tuple containing:
        - float: Proportion of parts in completely correct clusters.
        - float: Proportion of clusters with most frequent target frequency >= threshold.
        - float: Proportion of parts with the most frequent target label per cluster matching the target label.
        - pandas.DataFrame: DataFrame with evaluation results for each part, as returned by make_pred_df().
        - pandas.DataFrame: DataFrame with columns ['predicted_cluster', 'most_frequent_target', 'frequency_of_most_frequent_target'].
    '''
    ids = get_part_ids(csv_path, has_image_col=has_image_col)
    if parts_feature:
        is_image_order_correct = assert_image_order(parts_feature, ids) # TODO try to move this to where parts_feature is generated or used to generate the predicted labels
        if not is_image_order_correct:
            print("Number of parts in the the feature vector does not match with the CSV")
            return None

    target_labels = get_target_labels(csv_path, target_col=target_col, has_image_col=has_image_col)
    print('Unique labels:', len(np.unique(target_labels)))
    pred_df = make_pred_df(ids, target_labels, predicted_labels)
    frequency_correct_clusters = get_frequency_completely_correct_clusters(pred_df)
    frequency_df, frequency_clusters_above_threshold = get_frequency_clusters_with_mostly_one_target(threshold, pred_df)
    frequency_main_target_per_cluster = get_frequency_main_target_per_cluster(pred_df)
    return frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, pred_df, frequency_df

def evaluate_clusters_numbers(csv_path, parts_feature, feature_matrix, out_dir, target_col='target', threshold=70, min_clusters=100, max_clusters=500, step=100):
    '''
    Runs clustering with k-means with different cluster numbers and evaluates the results with all evaluation metrics.

    Parameters:
    csv_path (str): Path to the CSV file containing part information.
    parts_feature (dict): Dictionary with filenames (with extension) as keys and the feature vectors of parts as values, as returned by image_feature_extraction.map_parts_to_features().
    feature_matrix (numpy.ndarray): Feature matrix used for clustering.
    out_dir (str): Output directory to save the results and pickled files.
    target_col (str): Column name for the target labels in the CSV file, default: 'target'.
    threshold (float): Threshold for the most frequent target frequency in clusters, default: 70.
    min_clusters (int): Minimum number of clusters to evaluate, default: 100.
    max_clusters (int): Maximum number of clusters to evaluate, default: 500.
    step (int): Step size for the number of clusters, default: 100.

    Returns:
    pandas.DataFrame: DataFrame with evaluation results for each number of clusters.
    '''
    # Results to return:
    n_clusters_list = list(range(min_clusters, max_clusters+1, step))
    frequency_correct_clusters_list = []
    frequency_clusters_above_threshold_list = []
    frequency_main_target_per_cluster_list =[]
    # Results to save:
    predicted_labels_list = []

    for n in n_clusters_list:
        print(f"{n} clusters")
        # Clustering
        predicted_labels = cl.cluster_with_kmeans(
            feature_matrix=feature_matrix,
            n_clusters=n
        )
        predicted_labels_list.append(predicted_labels)
        
        # Evaluation using 'target' column 
        frequency_correct_clusters, frequency_clusters_above_threshold, frequency_main_target_per_cluster, _, _ = evaluate_clustering(csv_path, parts_feature, predicted_labels, target_col=target_col, threshold=threshold)
        frequency_correct_clusters_list.append(frequency_correct_clusters)
        frequency_clusters_above_threshold_list.append(frequency_clusters_above_threshold)
        frequency_main_target_per_cluster_list.append(frequency_main_target_per_cluster)

    # Make dictionary in format: n_clusters_list: predicted_labels_list
    n_cluster_to_labels = {n: labels for (n, labels) in zip(n_clusters_list, predicted_labels_list)}
    # Make the output directory, if it does not exist, and save the results to .pkl files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    im.dump_to_pickle_file(n_clusters_list, os.path.join(out_dir, "n_clusters_list.pkl"))
    im.dump_to_pickle_file(frequency_correct_clusters_list, os.path.join(out_dir, "frequency_correct_clusters_list.pkl"))
    im.dump_to_pickle_file(frequency_clusters_above_threshold_list, os.path.join(out_dir, "frequency_clusters_above_threshold_list.pkl"))
    im.dump_to_pickle_file(frequency_main_target_per_cluster_list, os.path.join(out_dir, "frequency_main_target_per_cluster_list.pkl")) # ROOD
    im.dump_to_pickle_file(n_cluster_to_labels, os.path.join(out_dir, "n_cluster_to_labels.pkl"))
    # Make and return a DataFrame with the results, also save it to a .csv file
    results_df = pd.DataFrame({
        "n_clusters": n_clusters_list,
        "predicted_labels": predicted_labels_list,
        "frequency_correct_clusters": frequency_correct_clusters_list,
        "frequency_clusters_above_threshold": frequency_clusters_above_threshold_list,
        "frequency_main_target_per_cluster": frequency_main_target_per_cluster_list
    })
    results_df.to_csv(os.path.join(out_dir, "evaluation.csv"))
    return results_df