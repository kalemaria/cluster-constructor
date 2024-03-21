from keras.preprocessing.image import load_img
import os
import matplotlib.pyplot as plt
import numpy as np

def display_images_in_cluster(groups, image_folder, i=None, output_png=None):
    '''
    Displays a maximum of 100 (10x10) images in any given cluster.

    Parameters:
    groups (dict): Dictionary as returned by group_parts_to_clusters.
    image_folder (str): Image folder.
    i (int or None): Cluster number, if None, prompts the user to enter a cluster number, default: None.
    output_png (str or None): If not None, saves the plot as a PNG; otherwise, shows the plot, default: None.
    '''
    if i is None:
        n_clusters = len(groups)
        while True:
            try:
                i = int(input(f"Enter any cluster number between 0 to {n_clusters - 1}: "))
                if 0 <= i < n_clusters:
                    break  # Exit the loop if the input is valid
                else:
                    print(f"Invalid input. Please enter a number between 0 and {n_clusters - 1}.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
    fig, axs = plt.subplots(10, 10, figsize=(28, 28))
    count = 0
    for file in groups[i]:
        file_path = os.path.join(image_folder, file)
        img = load_img(file_path)
        img = np.array(img)
        axs[count//10, count%10].imshow(img)
        axs[count//10, count%10].axis('off')
        count+=1
        if(count == 100):
            break

    if output_png is not None:
        fig.savefig(output_png)
    else:
        fig.show()

def plot_cumulative_explained_variance_vs_n_components(cumulative_explained_variance, n_components, pca_var, model_details):
    '''
    Plot cumulative explained variance against the number of principal components, highlighting a specific point.

    Parameters:
    cumulative_explained_variance (numpy.ndarray): Cumulative explained variance values.
    n_components (int): Number of principal components to highlight.
    pca_var (float): Percentage of explained variance at the highlighted point.
    model_details (str): Details of the model for the plot title.
    '''
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.axvline(x=n_components, color='red', linestyle='--', label=f'{pca_var * 100}% Explained Variance (n={n_components})')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Explained Variance by Different Principal Components of {model_details.replace("_", " ")} Features')
    plt.grid(True)
    plt.show()

def plot_histogram_of_cluster_sizes(cluster_sizes, model_details, n_clusters):
    '''
    Plot histogram of cluster sizes.

    Parameters:
    cluster_sizes (list): List of cluster sizes.
    model_details (str): Details of the model for the plot title.
    n_clusters (int): Number of clusters for the plot title.
    '''
    plt.hist(cluster_sizes, bins=50)
    plt.title(f'Distribution of Cluster Sizes using {model_details.replace("_", " ")} and {n_clusters} Clusters')
    plt.xlabel('Number of images per cluster')
    plt.ylabel('Count')
    plt.show()

def plot_score_vs_clusters_numbers(df, threshold, model_details):
    '''
    Plot scores against the number of clusters for k-means clustering evaluation.

    Parameters:
    df (pandas.DataFrame): DataFrame containing evaluation results for each number of clusters.
    threshold (float): Threshold for the most frequent target frequency in clusters.
    model_details (str): Details of the model for the plot title.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(df["n_clusters"], df["frequency_correct_clusters"],
             label='Completely correct clusters', marker='o')
    plt.plot(df["n_clusters"], df["frequency_clusters_above_threshold"],
             label=f'Clusters with most frequent target frequency >= {threshold * 100}%', marker='o')
    plt.plot(df["n_clusters"], df["frequency_main_target_per_cluster"],
             label='Main target per cluster', marker='o')
    plt.xlabel('Number of k-means clusters')
    plt.ylabel('Score %')
    plt.title(f'Evaluation of K-means Clustering using {model_details.replace("_", " ")}')
    plt.grid(True)
    plt.legend()
    plt.show()