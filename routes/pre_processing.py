import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data, pre_process

df = load_data()

st.subheader("First 5 Rows")
st.write(df.head())

processed_df = pre_process(df)

st.subheader("Processed Transformed Data")
st.write(processed_df.head())

scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(processed_df), columns=[processed_df.columns]
)

st.subheader("Processed Normalized Data")
st.write(data_scaled.head())

pca = PCA()
pca.fit(data_scaled)

cov_matrix = np.cov(data_scaled, rowvar=False)
np.fill_diagonal(cov_matrix, 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues_real = np.round(eigenvalues.real,3)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

fig, ax = plt.subplots()
hm = sns.heatmap(data=cov_matrix, annot=True,
                xticklabels=df.columns,
                yticklabels=df.columns
                )
st.pyplot(fig)

col_1, col_2 = st.columns([1,3]) 

pc_array = np.array([f'PC{i+1}' for i in range(0,9)])
pc_np = np.column_stack([pc_array, eigenvalues_real])
eigenvalues_df = pd.DataFrame(pc_np, columns=['PC', 'eigenvalue'])
eigenvalues_df.set_index('PC', inplace=True)
eigenvalues_df['eigenvalue'] = eigenvalues_df['eigenvalue'].str.strip("()").str.replace("+0j","")
eigenvalues_df["eigenvalue"] = pd.to_numeric(eigenvalues_df["eigenvalue"])
with col_1:
    st.write("**Eigenvalues**")
    st.write(eigenvalues_df)

with col_2:
    st.write("**Eigenvectors**")
    st.write(eigenvectors)

fig, ax = plt.subplots(figsize=(10,6))
bar_container = ax.bar(
    range(1, len(explained_variance)+1),
    explained_variance,
)
plt.xlabel("PC")
plt.ylabel("Explained Variance Ratio")
ax.bar_label(bar_container, fmt='{:,.3f}')
st.pyplot(plt)

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


n_components = np.where(cumulative_variance > 0.9)
n_components = n_components[0][0] + 1

pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data_scaled)


pca_df = pd.DataFrame(
    data_reduced, columns=[f"PC{i+1}" for i in range(n_components)]
)

st.subheader("Processed PCA Data")
st.write(pca_df.head())

distortions = []

K = range(1, 20)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
    kmeans.fit(pca_df)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), distortions, marker="o")
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
st.pyplot(plt)

