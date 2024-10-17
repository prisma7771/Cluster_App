import joblib
import streamlit as st
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from utils.utils import load_data, pre_process
from utils.pipa import pipeline_ori
from utils.computation import compute_pca2, compute_pca3, compute_tsne2, compute_tsne3


def plot_2d_pca(data, pred_cluster):
    pcadf_vis_array = compute_pca2(data)

    # create a new DataFrame with the transformed data
    pcadf_vis = pd.DataFrame(
        data=pcadf_vis_array, columns=["component_1", "component_2"]
    )

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))

    scat = sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=pcadf_vis,
        hue=pred_cluster,
        palette="Set2",
    )

    scat.set_title("Caffe_Order")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    st.pyplot(plt)


# Function to plot the t-SNE results
def plot_2d_tsne(data, pred_cluster):
    # Get the cached t-SNE result
    X_tsne = compute_tsne2(data)

    # Plotting
    plt.figure(figsize=(8, 8))
    scat_tsne = sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], s=50, hue=pred_cluster, palette="Set2"
    )
    scat_tsne.set_title("Caffe_Order - t-SNE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Use Streamlit to display the plot
    st.pyplot(plt)


# Function to plot the PCA results
def plot_3d_pca(data, pred_clusters, title="PCA 3D Visualization"):
    # Compute 3 component PCA
    pcadf_vis_array = compute_pca3(data)

    # Define a colormap similar to Matplotlib's Set2
    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    # Create a new DataFrame with the transformed data
    pcadf_vis = pd.DataFrame(
        data=pcadf_vis_array, columns=["component_1", "component_2", "component_3"]
    )

    # Initialize a 3D scatter plot figure
    fig = go.Figure()

    # Adding scatter plot traces for each cluster
    for cluster in set(pred_clusters):  # Iterate through unique cluster labels
        cluster_data = pcadf_vis[[pred == cluster for pred in pred_clusters]]
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data["component_1"],
                y=cluster_data["component_2"],
                z=cluster_data["component_3"],
                mode="markers",
                marker=dict(size=5, color=color_map[cluster]),
                name=f"Cluster {cluster}",
            )
        )

    # Update layout for 3D plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Component 1"),
            yaxis=dict(title="Component 2"),
            zaxis=dict(title="Component 3"),
        ),
        legend=dict(title="Clusters", x=0.9, y=0.9),
    )

    st.plotly_chart(fig)


# Function to plot the 3D t-SNE results
def plot_3d_tsne(data, pred_cluster, title="t-SNE 3D Visualization"):
    # Get the cached 3D t-SNE result
    X_tsne = compute_tsne3(data)

    # Define a colormap similar to Matplotlib's Set2
    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(
        data=X_tsne, columns=["component_1", "component_2", "component_3"]
    )

    # Initialize a 3D scatter plot figure
    fig = go.Figure()

    # Adding scatter plot traces for each cluster
    for cluster in set(pred_cluster):  # Iterate through unique cluster labels
        cluster_data = tsne_df[
            [pred == cluster for pred in pred_cluster]
        ]  # Filter rows by cluster label
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data["component_1"],
                y=cluster_data["component_2"],
                z=cluster_data["component_3"],
                mode="markers",
                marker=dict(size=5, color=color_map[cluster]),
                name=f"Cluster {cluster}",
            )
        )

    # Update layout for 3D plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Component 1"),
            yaxis=dict(title="Component 2"),
            zaxis=dict(title="Component 3"),
        ),
        legend=dict(title="Clusters", x=0.9, y=0.9),
    )

    # Use Streamlit to display the plot
    st.plotly_chart(fig)


# load_data
df = load_data()
df = pre_process(df)

model_kmeans = pipeline_ori(df)
# Load the saved PCA-enhanced K-Means model from joblib
model_kmeans_enhanced = joblib.load("data/kmeans_model_pca_firefly.joblib")


preprocessed_data1 = model_kmeans["preprocessor"].transform(df)
predicted_labels1 = model_kmeans["clusterer"]["kmeans"].labels_

preprocessed_data2 = model_kmeans_enhanced["preprocessor"].transform(df)
predicted_labels2 = model_kmeans_enhanced["clusterer"]["kmeans"].labels_

score1 = silhouette_score(preprocessed_data1, predicted_labels1)
score2 = silhouette_score(preprocessed_data2, predicted_labels2)

pcadf1 = pd.DataFrame(
    model_kmeans["preprocessor"].transform(df),
)
pcadf2 = pd.DataFrame(
    model_kmeans_enhanced["preprocessor"].transform(df),
)

col1, col2 = st.columns(2)

with col1:
    st.write("Original  K-Means")
    st.write(score1)

    plot_2d_pca(pcadf1, predicted_labels1)

    plot_2d_tsne(pcadf1, predicted_labels1)

    plot_3d_pca(pcadf1, predicted_labels1)

    plot_3d_tsne(pcadf1, predicted_labels1)


with col2:
    st.write("Enhanched K-means")
    st.write(score2)

    plot_2d_pca(pcadf2, predicted_labels2)

    plot_2d_tsne(pcadf2, predicted_labels2)

    plot_3d_pca(pcadf2, predicted_labels2)

    plot_3d_tsne(pcadf2, predicted_labels2)

st.subheader("Cluster Overview")
st.image("./plot/radar.png", caption="Radar Plot of Cluster", use_column_width=True)

# Reset indices to ensure they align
data = df.reset_index(drop=True)
# Create a new DataFrame 'data_cluster' with the predicted cluster labels
data_cluster = data.copy()  # Make a copy of the original DataFrame
data_cluster["predicted_cluster"] = predicted_labels2

# Display the first few rows of the new DataFrame
st.write(data_cluster.head())

# Count the number of occurrences in each cluster
cluster_counts = data_cluster['predicted_cluster'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
plt.title('Count of Data Points in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)

st.pyplot(plt)
st.write(cluster_counts)
