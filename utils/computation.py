from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st


@st.cache_data
def compute_pca(data, n):
    pca = PCA(n_components=n)
    X_pca2 = pca.fit_transform(data)
    return X_pca2


@st.cache_data
def compute_3d(data, n):
    tsne = TSNE(n_components=n, random_state=42)
    X_tsne2 = tsne.fit_transform(data, n)
    return X_tsne2

