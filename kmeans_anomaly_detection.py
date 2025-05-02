# %% [markdown]
# # Project: Anomaly Detection in Banking Transactions Using K-means
# 

# %% [markdown]
# ## Objective
# 
# Use K-means to automatically identify unusual behaviors in a set of synthetic transactions. The clusters model different behavior patterns, and anomalies can be detected by observing points that are too far from the cluster centers.

# %%
# Importing necessary modules for data processing clustering and visualization

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# %%
# This function is responsible for loading a dataset from a CSV file

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from the given CSV file.

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the transaction data
    """

    data_as_dataframe = pd.read_csv(file_path)
    return data_as_dataframe


# %%
# This function is used to select the useful variables for the program. 
# I chose to remove global_id, sender_id, and date because they are only useful for identifying transactions, 
# but this information will not help us. 
# Unless we consider specific days (e.g., Fridays) associated with fraud, which we won't analyze here 
# since we are not treating this as a time series problem.

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant numerical features for clustering.

    :param df: Full transaction DataFrame
    :return: DataFrame with selected features only
    """

    select_features_df = df.drop(columns=['global_id', 'sender_id', 'receiver_id', 'date']).copy()
    return select_features_df


# %%
# This function is used during the preprocessing step to standardize datasets. 
# Its goal is to mitigate bias that could arise from large variations in feature magnitudes within the dataset.

def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize the selected features using StandardScaler.

    :param df: DataFrame of selected features
    :return: Normalized NumPy array of features
    """
    
    scaler = StandardScaler()
    normalize_data = scaler.fit_transform(df)
    return normalize_data 


# %%
# The function below is used to apply K-Means clustering on the data with 6 clusters
#help(KMeans)

def apply_kmeans(data: np.ndarray, n_clusters: int = 6) -> tuple:
    """
    Apply KMeans clustering to the normalized transaction data.

    :param data: Normalized feature array
    :param n_clusters: Number of clusters
    :return: Tuple (trained model, labels)
    """
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit( data)
    return kmeans, kmeans.labels_

#type(apply_kmeans(normalize_select_features_df))

# %%
# This function is used to add metrics to the data, such as:
# - The distance from each point to its cluster centroid
# - The distance as a percentage, which helps us understand how "far" a point is compared to others this is useful especially when we are not domain experts
# I also implemented the Modified Z-Score, which takes into account the density of the cluster.
# 
# The Modified Z-Score is useful for detecting outliers. It tells us how far a point is from the median
# considering the spread using Median Absolute Deviation instead of standard deviation
# It works better than the standard Z-Score when data is not normally distributed or has outliers it's the case there
# I was inspired by this article : https://www.kaggle.com/code/praxitelisk/anomaly-detection-techniques-summary

def add_distance_centroid_zscore(data_kmeans: KMeans, data: np.ndarray) -> pd.DataFrame:
    """
    Adds to each point:
    - Euclidean distance to the centroid,
    - distance to the centroid as a percentage of the cluster's maximum length,
    - a z-score based on the distance to the centroid computed within each cluster.

    :param data_kmeans: Trained KMeans model
    :param data: Input data (n_samples, n_features) used for clustering
    :return: DataFrame enriched with cluster-specific metrics
    """
    
    cluster_labels = data_kmeans.labels_
    centroids = data_kmeans.cluster_centers_
    n_clusters = data_kmeans.n_clusters

    # Compute Euclidean distances from each point to its cluster's centroid
    distances_to_centroid = np.linalg.norm(data - centroids[cluster_labels], axis=1)
    percent_distances = np.zeros(len(data))
    zscores = np.zeros(len(data))

    for cluster_id in range(n_clusters):
        # Mask for points in the current cluster
        mask = cluster_labels == cluster_id
        cluster_distances = distances_to_centroid[mask]
        max_dist = cluster_distances.max()

        # Calculate distance in percentage
        if max_dist > 0:
            percent_distances[mask] = (cluster_distances / max_dist) * 100
        else:
            percent_distances[mask] = 0

        # Calculate modified z-scores: z = 0.6745 * (distance - median) / MAD (if MAD > 0)
        median_d = np.median(cluster_distances)
        mad = np.median(np.abs(cluster_distances - median_d))
        if mad > 0:
            zscores[mask] = 0.6745 * (cluster_distances - median_d) / mad
        else:
            zscores[mask] = 0

    # Build the enriched DataFrame
    df = pd.DataFrame(data, columns=[f"data_col_{i}" for i in range(data.shape[1])])
    df['cluster_label'] = cluster_labels
    df['distance_to_centroid'] = distances_to_centroid
    df['distance_percent'] = percent_distances
    df['zscore'] = zscores

    return df


# %%
# Based on the previous function, I can now specify a threshold in the form of a maximum distance, 
# as well as percentage and z-score limits, to identify an element as an anomaly.

def detect_anomalies(df, methods=['distance_percent', 'zscore'], thresholds=None):
    """
    Detect anomalies using simple thresholding on:
    - distance_percent: relative distance to cluster center
    - zscore: standard score within cluster
    - distance_to_centroid: raw Euclidean distance

    Parameters:
        df (DataFrame): Must contain the relevant columns.
        methods (list): List of methods to use.
        thresholds (dict): Dictionary of thresholds for each method.

    Returns:
        np.ndarray: Boolean mask of anomalies.
    """
    if thresholds is None:
        thresholds = {
            'distance_percent': 95,
            'zscore': 4,  # Points with a zscore above 4 considered anomalous
        }

    anomalies = np.zeros(len(df), dtype=bool)

    if 'distance_percent' in methods:
        anomalies |= df['distance_percent'] > thresholds['distance_percent']

    if 'zscore' in methods:
        anomalies |= df['zscore'] > thresholds['zscore']

    return anomalies


# %%
# Written with the assistance of Copilot but implementation has been reviewed and understood.

def plot_clusters(
    normalized_data: np.ndarray,
    labels: np.ndarray,
    original_df: pd.DataFrame,
    anomalies_mask: np.ndarray = None,
    sample_size: int = 5000
):
    """
    Plot 2D PCA clusters, highlight anomalies by surrounding their markers, and show original (non-normalized) values in hover.

    Parameters:
      - normalized_data: standardized feature array (n_samples, n_features)
      - labels: cluster labels (n_samples,)
      - original_df: DataFrame (n_samples, m) with original columns to display in hover
      - anomalies_mask: boolean array (n_samples,) marking anomalies
      - sample_size: maximum points to plot
    """

    # PCA → 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(normalized_data)

    # Build DataFrame for plotting
    df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df["cluster"] = labels.astype(str)
    df["anomaly"] = anomalies_mask if anomalies_mask is not None else False
    # Attach original (non-normalized) columns
    df = pd.concat([df, original_df.reset_index(drop=True)], axis=1)

    # Sampling if too many points
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Base scatter: display all points colored by cluster, with hover showing the original columns, cluster, anomaly, PC1, and PC2.
    hover_cols = list(original_df.columns) + ["cluster", "anomaly"]
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        symbol="anomaly",
        symbol_map={False: "circle", True: "circle"},
        hover_data=hover_cols,
        opacity=0.6,
        title=f"PCA Clusters (n={len(df)})",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=6))

    # Overlay anomalies: add an extra trace for points identified as anomalies
    # These anomalies will use the same color as their cluster but with a border (here in black) to distinguish them
    if anomalies_mask is not None:
        anom_df = df[df["anomaly"] == True]
        if not anom_df.empty:
            # Create a cluster-to-color mapping using Plotly Express’s color sequence
            clusters = sorted(df["cluster"].unique())
            colors = px.colors.qualitative.Set1
            color_map = {clust: colors[i % len(colors)] for i, clust in enumerate(clusters)}

            # For each cluster in the anomalies, add a trace with a border
            for clust in anom_df["cluster"].unique():
                sub = anom_df[anom_df["cluster"] == clust]
                # Hover logic
                cols = list(original_df.columns)
                hover_lines = [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(cols)]
                hover_lines += ["cluster: %{text}", "anomaly: True", "PC1: %{x}", "PC2: %{y}"]
                hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

                fig.add_trace(go.Scatter(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=6, # slightly larger size to highlight the anomaly
                        color=color_map[clust],
                        line=dict(width=3, color="black")# black border to outline
                    ),
                    name=f"Anomalies Cluster {clust}",
                    text=[clust] * len(sub),
                    customdata=sub[cols].to_numpy(),
                    hovertemplate=hovertemplate
                ))

    # Centroids: calculate and display the centroids projected into PCA space
    centroids = np.array([normalized_data[labels == k].mean(axis=0) for k in np.unique(labels)])
    centroids_2d = pca.transform(centroids)
    fig.add_trace(go.Scatter(
        x=centroids_2d[:, 0],
        y=centroids_2d[:, 1],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="black", line=dict(color="white", width=1)),
        text=[f"C{k}" for k in np.unique(labels)],
        textposition="top center",
        name="Centroids"
    ))

    fig.show()


# %% [markdown]
# ### Step 1: Loading the Data  
# Objective: Load the transactions.csv file containing 20,000 transactions.

# %%
def main():
    file_path = "transactions.csv"
    data_as_dataframe = load_data(file_path)
    data_as_dataframe.head(4)

    # Data Exploration
    print(data_as_dataframe.shape)
    print(data_as_dataframe.columns)
    data_as_dataframe.describe()

    print(data_as_dataframe.isnull().sum())

    # Step 2: Selecting Relevant Features  
    select_features_df = select_features(data_as_dataframe)
    select_features_df.head()

    # Step 3: Data Preparation and Normalization  
    normalize_select_features_df = normalize_features(select_features_df)

    # Step 4: Clustering with K-means  
    data_kmean, data_kmean_labels = apply_kmeans(data=normalize_select_features_df, n_clusters=6)

    print("data_kmean_labels  :  ", data_kmean_labels, "\n")
    print("len(data_kmean_labels) : ", len(data_kmean_labels), "\n")
    print("data_kmean.cluster_centers_ : ", data_kmean.cluster_centers_, "\n")
    print("type(data_kmean) : ", type(data_kmean))

    # Step 5: Anomaly Detection  
    analysis_kmeans = add_distance_centroid_zscore(data_kmean, normalize_select_features_df)
    print("Highest zscore:", analysis_kmeans['zscore'].max())
    print("Lowest zscore:", analysis_kmeans['zscore'].min())

    mask = detect_anomalies(analysis_kmeans, methods=['zscore'], thresholds={'zscore':4})
    analysis_kmeans['anomaly'] = mask

    df = data_as_dataframe.copy()
    df = df.join(
        analysis_kmeans[['cluster_label','distance_to_centroid','distance_percent','zscore','anomaly']],
        how='left'
    )

    print("Total anomalies :", df['anomaly'].sum())
    print("Anomalies per cluster:")
    print(df[df['anomaly']].groupby('cluster_label').size(), "\n")

    centroids = data_kmean.cluster_centers_
    for k in range(len(centroids)):
        sub = df[df['cluster_label']==k]
        if sub.empty:
            print(f"Cluster {k} empty\n")
            continue

        closest4 = sub.nsmallest(4, 'distance_to_centroid')
        farthest4 = sub.nlargest(4, 'distance_to_centroid')
        print(f"Cluster {k}:")
        print("  4 closest to centroid:")
        print(closest4)
        print("  4 farthest from centroid:")
        print(farthest4)
        print("\n")

    anomalies = analysis_kmeans['anomaly'].values
    plot_clusters(
      normalized_data=normalize_select_features_df,
      labels=data_kmean.labels_,
      original_df=select_features_df,
      anomalies_mask=anomalies,
      sample_size=20000
    )


if __name__ == "__main__":
    main()
