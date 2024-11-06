import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.utils import load_data, pre_process

df = load_data()


st.subheader("First 5 Rows")
st.write(df.head())


st.subheader("Basic Statistics")
st.write(df.describe())


st.subheader("Data Types and Null Values")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


processed_df = pre_process(df)


st.subheader("Processed Data Preview 1")
st.write(processed_df.head())


scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(processed_df), columns=[processed_df.columns]
)

data_scaled


pca = PCA()
pca.fit(data_scaled)


explained_variance = pca.explained_variance_ratio_


cumulative_variance = explained_variance.cumsum()


plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o",
    label="Individual Explained Variance",
)
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
    label="Cumulative Explained Variance",
)
plt.axhline(y=0.9, color="r", linestyle="--", label="90% Explained Variance")
plt.title("Explained Variance by Principal Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.grid()
st.pyplot(plt)


st.write("Cumulative Variance")
st.text(cumulative_variance)


n_components = np.where(cumulative_variance > 0.8)
n_components = n_components[0][0] + 1

pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data_scaled)


data_reduced_df = pd.DataFrame(
    data_reduced, columns=[f"PC{i+1}" for i in range(n_components)]
)

distortions = []

K = range(1, 20)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
    kmeans.fit(data_reduced_df)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), distortions, marker="o")
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
st.pyplot(plt)
