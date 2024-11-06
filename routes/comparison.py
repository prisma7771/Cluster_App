import joblib
import streamlit as st
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from utils.utils import load_data, pre_process
from utils.pipa import pipeline_ori
from utils.computation import compute_pca, compute_tsne
import os
from PIL import Image


# def plot_3d_pca(data, pred_clusters, title="PCA 3D Visualization"):
#     pcadf_vis_array = compute_pca3(data)

#     set2_colors = cm.Set2.colors
#     color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

#     pcadf_vis = pd.DataFrame(
#         data=pcadf_vis_array, columns=["component_1", "component_2", "component_3"]
#     )

#     fig = go.Figure()

#     for cluster in set(pred_clusters):
#         cluster_data = pcadf_vis[[pred == cluster for pred in pred_clusters]]
#         fig.add_trace(
#             go.Scatter3d(
#                 x=cluster_data["component_1"],
#                 y=cluster_data["component_2"],
#                 z=cluster_data["component_3"],
#                 mode="markers",
#                 marker=dict(size=5, color=color_map[cluster]),
#                 name=f"Cluster {cluster}",
#             )
#         )

#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis=dict(title="Component 1"),
#             yaxis=dict(title="Component 2"),
#             zaxis=dict(title="Component 3"),
#         ),
#         legend=dict(title="Clusters", x=0.9, y=0.9),
#     )

#     st.plotly_chart(fig)
# def plot_3d_tsne(data, pred_cluster, title="t-SNE 3D Visualization"):
#     X_tsne = compute_tsne3(data)

#     set2_colors = cm.Set2.colors
#     color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

#     tsne_df = pd.DataFrame(
#         data=X_tsne, columns=["component_1", "component_2", "component_3"]
#     )

#     fig = go.Figure()

#     for cluster in set(pred_cluster):
#         cluster_data = tsne_df[[pred == cluster for pred in pred_cluster]]
#         fig.add_trace(
#             go.Scatter3d(
#                 x=cluster_data["component_1"],
#                 y=cluster_data["component_2"],
#                 z=cluster_data["component_3"],
#                 mode="markers",
#                 marker=dict(size=5, color=color_map[cluster]),
#                 name=f"Cluster {cluster}",
#             )
#         )

#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis=dict(title="Component 1"),
#             yaxis=dict(title="Component 2"),
#             zaxis=dict(title="Component 3"),
#         ),
#         legend=dict(title="Clusters", x=0.9, y=0.9),
#     )

#     st.plotly_chart(fig)


def plot_2d_projection(data, pred_cluster, method="pca", save_path="plot.png"):
    # Check if a saved plot exists
    if os.path.exists(save_path):
        img = Image.open(save_path)
        return img  # Return the saved image if it exists

    # Compute 2D projection
    if method == "pca":
        data_vis_array = compute_pca(data, 2)
        title = "Caffe_Order - PCA"
    elif method == "tsne":
        data_vis_array = compute_tsne(data, 2)
        title = "Caffe_Order - t-SNE"
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # Convert to DataFrame with consistent column names
    data_vis = pd.DataFrame(data=data_vis_array, columns=["component_1", "component_2"])

    # Set up the plot
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=data_vis,
        hue=pred_cluster,
        palette="Set2",
        ax=ax,
    ).set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Save plot to the specified file path
    fig.savefig(save_path)

    # Return the saved image
    img = Image.open(save_path)
    return img


def plot_3d_projection(data, pred_cluster, method="pca", save_path="plot3d.png"):
    # Check if a saved plot exists
    if os.path.exists(save_path):
        img = Image.open(save_path)
        return img  # Return the saved image if it exists

    # Compute 3D projection
    if method == "pca":
        data_vis_array = compute_pca(data, 3)  # Assumes compute_pca3 gives 3 components
        title = "Caffe_Order - PCA (3D)"
    elif method == "tsne":
        data_vis_array = compute_tsne(
            data, 3
        )  # Assumes compute_tsne3 gives 3 components
        title = "Caffe_Order - t-SNE (3D)"
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # Convert to DataFrame with consistent column names for 3D
    data_vis = pd.DataFrame(
        data=data_vis_array, columns=["component_1", "component_2", "component_3"]
    )

    # Set up the 3D plot
    plt.style.use("fivethirtyeight")
    plt.tight_layout()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 3D Scatter plot
    scatter = ax.scatter(
        data_vis["component_1"],
        data_vis["component_2"],
        data_vis["component_3"],
        c=pred_cluster,
        cmap="Set2",
        s=50,
    )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.1, label="Cluster")

    # Save plot to the specified file path
    fig.savefig(save_path)

    # Return the saved image
    img = Image.open(save_path)
    return img


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

sse1 = model_kmeans["clusterer"]["kmeans"].inertia_  # SSE for model_kmeans
sse2 = model_kmeans_enhanced["clusterer"]["kmeans"].inertia_

df1 = pd.DataFrame(
    model_kmeans["preprocessor"].transform(df),
)
df2 = pd.DataFrame(
    model_kmeans_enhanced["preprocessor"].transform(df),
)

col1, col2 = st.columns(2)

with col1:
    st.write("Original  K-Means")
    st.write(f"SSE = {sse1:.3f}")
    st.write(f"Silhouette Score = {score1:.3f}")

    st.image(
        plot_2d_projection(
            df1,
            predicted_labels1,
            method="pca",
            save_path="plot/df1_pca_plot.png",
        ),
        use_column_width=True,
    )

    st.image(
        plot_2d_projection(
            df1,
            predicted_labels1,
            method="tsne",
            save_path="plot/df1_tsne_plot.png",
        ),
        use_column_width=True,
    )

    st.image(
        plot_3d_projection(
            df1,
            predicted_labels1,
            method="pca",
            save_path="plot/df1_pca3d_plot.png",
        ),
        use_column_width=True,
        clamp=True,
        output_format="PNG",
    )

    st.image(
        plot_3d_projection(
            df1,
            predicted_labels1,
            method="tsne",
            save_path="plot/df1_tsne3d_plot.png",
        ),
        use_column_width=True,
        clamp=True,
        output_format="PNG",
    )


with col2:
    st.write("Enhanched K-means")
    st.write(f"SSE = {sse2:.3f}")
    st.write(f"Silhouette Score = {score2:.3f}")

    st.image(
        plot_2d_projection(
            df2,
            predicted_labels2,
            method="pca",
            save_path="plot/df2_pca_plot.png",
        ),
        use_column_width=True,
    )

    st.image(
        plot_2d_projection(
            df2,
            predicted_labels2,
            method="tsne",
            save_path="plot/df2_tsne_plot.png",
        ),
        use_column_width=True,
    )

    st.image(
        plot_3d_projection(
            df2,
            predicted_labels2,
            method="pca",
            save_path="plot/df2_pca3d_plot.png",
        ),
        use_column_width=True,
        clamp=True,
        output_format="PNG",
    )

    st.image(
        plot_3d_projection(
            df2,
            predicted_labels2,
            method="tsne",
            save_path="plot/df2_tsne3d_plot.png",
        ),
        use_column_width=True,
        clamp=True,
        output_format="PNG",
    )


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
