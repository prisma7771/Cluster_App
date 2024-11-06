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


def plot_2d_tsne(data, pred_cluster):
    X_tsne = compute_tsne2(data)

    plt.figure(figsize=(8, 8))
    scat_tsne = sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], s=50, hue=pred_cluster, palette="Set2"
    )
    scat_tsne.set_title("Caffe_Order - t-SNE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    st.pyplot(plt)


def plot_3d_pca(data, pred_clusters, title="PCA 3D Visualization"):
    pcadf_vis_array = compute_pca3(data)

    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    pcadf_vis = pd.DataFrame(
        data=pcadf_vis_array, columns=["component_1", "component_2", "component_3"]
    )

    fig = go.Figure()

    for cluster in set(pred_clusters):
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


def plot_3d_tsne(data, pred_cluster, title="t-SNE 3D Visualization"):
    X_tsne = compute_tsne3(data)

    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    tsne_df = pd.DataFrame(
        data=X_tsne, columns=["component_1", "component_2", "component_3"]
    )

    fig = go.Figure()

    for cluster in set(pred_cluster):
        cluster_data = tsne_df[[pred == cluster for pred in pred_cluster]]
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


df = load_data()
df = pre_process(df)

model_kmeans = pipeline_ori(df)

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


data = df.reset_index(drop=True)

data_cluster = data.copy()
data_cluster["predicted_cluster"] = predicted_labels2


st.write(data_cluster.head())


cluster_counts = data_cluster["predicted_cluster"].value_counts().sort_index()


plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
plt.title("Count of Data Points in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.xticks(rotation=0)

st.pyplot(plt)
st.write(cluster_counts)
