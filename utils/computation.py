# Cache only the 2D PCA computation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st


@st.cache_data
def compute_pca2(data):
    pca = PCA(n_components=2)
    X_pca2 = pca.fit_transform(data)
    return X_pca2


# Cache only the 3D PCA computation
@st.cache_data
def compute_pca3(data):
    pca = PCA(n_components=3)
    X_pca3 = pca.fit_transform(data)
    return X_pca3


# Cache only the t-SNE computation
@st.cache_data
def compute_tsne2(data):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne2 = tsne.fit_transform(data)
    return X_tsne2


# Cache only the t-SNE computation
@st.cache_data
def compute_tsne3(data):
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne3 = tsne.fit_transform(data)
    return X_tsne3
