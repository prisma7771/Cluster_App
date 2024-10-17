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

# Display the first few rows of the dataset
st.subheader("First 5 Rows")
st.write(df.head())

# Show basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Show data types and null values
st.subheader("Data Types and Null Values")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Processed Data
processed_df = pre_process(df)

# Show Prcossed data-
st.subheader("Processed Data Preview 1")
st.write(processed_df.head())  # Display processed DataFrame's first 5 rows

# Standardize the combined data
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(processed_df),
    columns=[processed_df.columns]    
)

data_scaled

# Perform PCA on the standardized data
pca = PCA()
pca.fit(data_scaled)

# Calculate the explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_variance = explained_variance.cumsum()

# Plot the explained variance and cumulative variance
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

# Display the cumulative variance to find the optimal number of components
st.write("Cumulative Variance")
st.text(cumulative_variance)

# Find the first component with explained variance ratio above 0.8
n_components = np.where(
    cumulative_variance > 0.8
)  # Adding 1 to match the component number
n_components = n_components[0][0] + 1

pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data_scaled)

# Convert to DataFrame for easier analysis
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
